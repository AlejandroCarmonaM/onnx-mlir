/**********************************************
 * IMPORT LIBRARIES
 **********************************************/
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

// ONNX-MLIR Runtime API
#include "OnnxMlirRuntime.h"

/**********************************************
 * HELPER FUNCTION DEFINITIONS
 **********************************************/
inline float relu(float x) {
    return std::max(0.0f, x);
}

inline int64_t offset2d(const int64_t* strides, int64_t i, int64_t j) {
    return i * strides[0] + j * strides[1];
}

inline int64_t offset1d(const int64_t* strides, int64_t i) {
    return i * strides[0];
}

/**********************************************
 * MAIN RUNTIME FUNCTION DEFINITION
 **********************************************/
extern "C" void FusedGemm(
    OMTensor* Y_omTensor,
    OMTensor* A_omTensor,
    OMTensor* B_omTensor,
    OMTensor* Bias_omTensor, // Corresponds to Gemm's 'C' input
    const char *activation,
    float alpha,
    float beta,
    const char *domain_name,
    //const char *funcName,
    //int64_t numOfOutput,
    const char *onnx_node_name,
    int64_t transA,
    int64_t transB
) {
    // Validate input tensors
    if (!A_omTensor || !B_omTensor || !Y_omTensor) {
        fprintf(stdout, "Error: One or more input tensors are null.\n");
        return;
    }
    if (omTensorGetDataType(A_omTensor) != ONNX_TYPE_FLOAT ||
        omTensorGetDataType(B_omTensor) != ONNX_TYPE_FLOAT ||
        omTensorGetDataType(Y_omTensor) != ONNX_TYPE_FLOAT) {
        fprintf(stdout, "Error: Input tensors must be of type float.\n");
        return;
    }

    // Get shapes and strides
    const int64_t* A_shape = omTensorGetShape(A_omTensor);
    const int64_t* B_shape = omTensorGetShape(B_omTensor);
    const int64_t* Y_shape = omTensorGetShape(Y_omTensor);
    const int64_t* A_strides = omTensorGetStrides(A_omTensor);
    const int64_t* B_strides = omTensorGetStrides(B_omTensor);
    const int64_t* Y_strides = omTensorGetStrides(Y_omTensor);

    int64_t M = Y_shape[0];
    int64_t N = Y_shape[1];
    int64_t K = transA ? A_shape[0] : A_shape[1];

    float* A = static_cast<float*>(omTensorGetDataPtr(A_omTensor));
    float* B = static_cast<float*>(omTensorGetDataPtr(B_omTensor));
    float* Y = static_cast<float*>(omTensorGetDataPtr(Y_omTensor));
    float* Bias = Bias_omTensor ? static_cast<float*>(omTensorGetDataPtr(Bias_omTensor)) : nullptr;
    const int64_t* Bias_shape = Bias_omTensor ? omTensorGetShape(Bias_omTensor) : nullptr;
    const int64_t* Bias_strides = Bias_omTensor ? omTensorGetStrides(Bias_omTensor) : nullptr;
    int Bias_rank = Bias_omTensor ? omTensorGetRank(Bias_omTensor) : 0;

    // Detect row-major or column-major for each tensor
    auto is_row_major = [](const int64_t* shape, const int64_t* strides, int rank) -> bool {
        if (rank < 2) return true; // treat 1D as row-major
        // Row-major: strides[0] == shape[1], strides[1] == 1
        return (strides[0] == shape[1] && strides[1] == 1);
    };
    int A_rank = omTensorGetRank(A_omTensor);
    int B_rank = omTensorGetRank(B_omTensor);
    int Y_rank = omTensorGetRank(Y_omTensor);

    bool A_row_major = is_row_major(A_shape, A_strides, A_rank);
    bool B_row_major = is_row_major(B_shape, B_strides, B_rank);
    bool Y_row_major = is_row_major(Y_shape, Y_strides, Y_rank);

    // GEMM + Bias
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                // Handle A
                int64_t a_i = transA ? k : i;
                int64_t a_j = transA ? i : k;
                int64_t a_idx = A_row_major ? offset2d(A_strides, a_i, a_j) : offset2d(A_strides, a_j, a_i);

                // Handle B
                int64_t b_i = transB ? j : k;
                int64_t b_j = transB ? k : j;
                int64_t b_idx = B_row_major ? offset2d(B_strides, b_i, b_j) : offset2d(B_strides, b_j, b_i);

                acc += A[a_idx] * B[b_idx];
            }
            acc *= alpha;

            // Bias broadcasting
            float bias_val = 0.0f;
            if (Bias) {
                if (Bias_rank == 2 && Bias_shape[0] == M && Bias_shape[1] == N) {
                    int64_t bias_idx = Bias_strides ?
                        (is_row_major(Bias_shape, Bias_strides, Bias_rank) ?
                            offset2d(Bias_strides, i, j) :
                            offset2d(Bias_strides, j, i)) : i * N + j;
                    bias_val = Bias[bias_idx];
                } else if (Bias_rank == 1 && Bias_shape[0] == N) {
                    bias_val = Bias[offset1d(Bias_strides, j)];
                } else if (Bias_rank == 1 && Bias_shape[0] == M) {
                    bias_val = Bias[offset1d(Bias_strides, i)];
                } else if (Bias_rank == 0) {
                    bias_val = Bias[0];
                }
            }
            acc += beta * bias_val;

            // Activation (ReLU)
            if (activation && std::strcmp(activation, "Relu") == 0) {
                acc = relu(acc);
            }
            int64_t y_idx = Y_row_major ? offset2d(Y_strides, i, j) : offset2d(Y_strides, j, i);
            Y[y_idx] = acc;
        }
    }
}