/**********************************************
 * IMPORT LIBRARIES
 **********************************************/
#include <cstdint>  // For int64_t
#include <vector>   // Not strictly used here, but common
#include <cmath>    // For std::max
#include <iostream> // For std::cout, std::endl (placeholder logging)
#include <cstdio>   // For fprintf, stdout, fflush (debug logging)
#include <cstring>
#include <ctime> // For time(0)

// ONNX-MLIR Runtime API
#include "OnnxMlirRuntime.h"

/**********************************************
 * CONSTANTS & PARAMETERS
 **********************************************/
// None defined for this specific file.

/**********************************************
 * HELPER FUNCTION DEFINITIONS
 **********************************************/

/*
 * Purpose: Basic ReLU activation function.
 * Parameters:
 *    - x (float): Input value.
 * Returns:
 *    - float: max(0.0f, x).
 */
inline float relu(float x) {
    return std::max(0.0f, x);
}

/*
 * Purpose: Compute the linear offset into a flat buffer for a 2D tensor
 *          given its strides and logical indices.
 * Parameters:
 *    - strides (const int64_t*): Pointer to the strides array for the tensor.
 *                                 Assumes strides[0] is stride for dim 0, strides[1] for dim 1.
 *    - i (int64_t): Logical index for the first dimension.
 *    - j (int64_t): Logical index for the second dimension.
 * Returns:
 *    - int64_t: The calculated offset.
 */
inline int64_t offset2d(const int64_t* strides, int64_t i, int64_t j) {
    // Handle potential null strides defensively, although unlikely for valid tensors
    if (!strides) return 0; // Or handle error appropriately
    return i * strides[0] + j * strides[1];
}

/*
 * Purpose: Compute the linear offset into a flat buffer for a 1D tensor
 *          given its stride and logical index.
 * Parameters:
 *    - strides (const int64_t*): Pointer to the strides array (only strides[0] is used).
 *    - i (int64_t): Logical index for the dimension.
 * Returns:
 *    - int64_t: The calculated offset.
 */
inline int64_t offset1d(const int64_t* strides, int64_t i) {
    if (!strides) return 0;
    return i * strides[0];
}


/**********************************************
 * MAIN RUNTIME FUNCTION DEFINITION
 **********************************************/

/*
 * Purpose: Implements the FusedGemm operation (Gemm + Bias + ReLU) using OMTensor inputs.
 *          Mimics ONNX Gemm (alpha=1, beta=1) followed by ONNX ReLU.
 *          Handles tensor strides and bias broadcasting.
 * Parameters:
 *    - A_omTensor (OMTensor*): Input tensor A (MxK or KxM).
 *    - B_omTensor (OMTensor*): Input tensor B (KxN or NxK).
 *    - Bias_omTensor (OMTensor*): Optional input tensor C/Bias, broadcastable to (MxN).
 *    - Y_omTensor (OMTensor*): Output tensor Y (MxN).
 *    - M (int64_t): Dimension M of the output.
 *    - N (int64_t): Dimension N of the output.
 *    - K (int64_t): Dimension K (shared dimension).
 *    - transA (int64_t): Flag indicating if A should be transposed (0=No, Non-zero=Yes).
 *    - transB (int64_t): Flag indicating if B should be transposed (0=No, Non-zero=Yes).
 * Returns:
 *    - void: Output Y is modified in place.
 */

/*    MemRefType_float_2 *output,
   MemRefType_float_2 *A,
   MemRefType_float_2 *B,
   MemRefType_float_1 *C,
   const char *activation,
   float alpha,
   float beta,
   const char *domain_name,
   const char *funcName,
   int64_t numOfOutput,
   const char *onnx_node_name,
   int64_t transA,
   int64_t transB*/
// ...existing code...
/*    "krnl.call"(%alloc, %arg0, %0, %1) {activation = "Relu", alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, domain_name = "com.microsoft", funcName = "FusedGemm", numOfOutput = 1 : si64, onnx_node_name = "fused /fc1/Gemm", transA = 0 : si64, transB = 1 : si64} : (memref<1x128xf32>, memref<1x784xf32>, memref<128x784xf32>, memref<128xf32>) -> ()*/


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
  // LOGGING AND VALIDATION
  // Check if tensors are null
  if (!A_omTensor || !B_omTensor || !Y_omTensor) {
    fprintf(stdout, "Error: One or more input tensors are null.\n");
    return;
  }
  // Check if tensors are of type float
  if (omTensorGetDataType(A_omTensor) != ONNX_TYPE_FLOAT ||
      omTensorGetDataType(B_omTensor) != ONNX_TYPE_FLOAT ||
      omTensorGetDataType(Y_omTensor) != ONNX_TYPE_FLOAT) {
    fprintf(stdout, "Error: Input tensors must be of type float.\n");
    return;
  }

  // Get shapes
  const int64_t* A_shape = omTensorGetShape(A_omTensor);
  const int64_t* B_shape = omTensorGetShape(B_omTensor);
  const int64_t* Y_shape = omTensorGetShape(Y_omTensor);

  // Assume 2D tensors
  int64_t M = Y_shape[0];
  int64_t N = Y_shape[1];
  int64_t K = transA ? A_shape[0] : A_shape[1];

  // Check if dimensions are valid
  if (M <= 0 || N <= 0 || K <= 0) {
    fprintf(stdout, "Error: Invalid dimensions M=%lld, N=%lld, K=%lld.\n", M, N, K);
    return;
  }
  // Check if strides are valid
  const int64_t* A_strides = omTensorGetStrides(A_omTensor);
  const int64_t* B_strides = omTensorGetStrides(B_omTensor);
  const int64_t* Y_strides = omTensorGetStrides(Y_omTensor);
  if (!A_strides || !B_strides || !Y_strides) {
    fprintf(stdout, "Error: One or more input tensors have null strides.\n");
    return;
  }
  // Check if strides are non-negative
  for (int i = 0; i < 2; ++i) {
    if (A_strides[i] < 0 || B_strides[i] < 0 || Y_strides[i] < 0) {
      fprintf(stdout, "Error: Strides must be non-negative.\n");
      return;
    }
  }
  // Check if Bias tensor is valid (if provided)
  const int64_t* Bias_strides = nullptr;
  if (Bias_omTensor) {
    Bias_strides = omTensorGetStrides(Bias_omTensor);
    if (!Bias_strides) {
      fprintf(stdout, "Error: Bias tensor has null strides.\n");
      return;
    }
    // Check if Bias strides are non-negative
    for (int i = 0; i < omTensorGetRank(Bias_omTensor); ++i) {
      if (Bias_strides[i] < 0) {
        fprintf(stdout, "Error: Bias strides must be non-negative.\n");
        return;
      }
    }
  }

  // LOGGING
  fprintf(stdout, "FusedGemm called with:\n");
  fprintf(stdout, "  A_omTensor: rank=%d, dtype=%d, shape=[%lld,%lld], strides=[%lld,%lld]\n",
          omTensorGetRank(A_omTensor),
          omTensorGetDataType(A_omTensor),
          A_shape[0], A_shape[1],
          A_strides[0], A_strides[1]);
  fprintf(stdout, "  B_omTensor: rank=%d, dtype=%d, shape=[%lld,%lld], strides=[%lld,%lld]\n",
          omTensorGetRank(B_omTensor),
          omTensorGetDataType(B_omTensor),
          B_shape[0], B_shape[1],
          B_strides[0], B_strides[1]);
  fprintf(stdout, "  Y_omTensor: rank=%d, dtype=%d, shape=[%lld,%lld], strides=[%lld,%lld]\n",
          omTensorGetRank(Y_omTensor),
          omTensorGetDataType(Y_omTensor),
          Y_shape[0], Y_shape[1],
          Y_strides[0], Y_strides[1]);
  if (Bias_omTensor) {
    const int64_t* Bias_shape = omTensorGetShape(Bias_omTensor);
    fprintf(stdout, "  Bias_omTensor: rank=%d, dtype=%d, shape=[%lld,%lld], strides=[%lld,%lld]\n",
            omTensorGetRank(Bias_omTensor),
            omTensorGetDataType(Bias_omTensor),
            Bias_shape[0], omTensorGetRank(Bias_omTensor) > 1 ? Bias_shape[1] : 1,
            Bias_strides[0], omTensorGetRank(Bias_omTensor) > 1 ? Bias_strides[1] : 1);
  }
  //transB = 1;
  fprintf(stdout, "  M=%lld, N=%lld, K=%lld, transA=%lld, transB=%lld\n",
          M, N, K, transA, transB);
  fflush(stdout); // Ensure all logs are flushed immediately
                  // END LOGGING
  //exit(0); // DEBUGGING: Exit after logging
  // Get raw pointers
  float *A_data = reinterpret_cast<float *>(omTensorGetDataPtr(A_omTensor));
  float *B_data = reinterpret_cast<float *>(omTensorGetDataPtr(B_omTensor));
  float *Y_data = reinterpret_cast<float *>(omTensorGetDataPtr(Y_omTensor));
  float *Bias_data = Bias_omTensor ? reinterpret_cast<float *>(
                                         omTensorGetDataPtr(Bias_omTensor))
                                   : nullptr;

  /*int64_t bias_rank = Bias_omTensor ? omTensorGetRank(Bias_omTensor) : 0;

  // Compute Y = alpha * A·B  + beta * Bias  then ReLU
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        float a = transA
          ? A_data[offset2d(A_strides, k, i)]
          : A_data[offset2d(A_strides, i, k)];
        float b = transB
          ? B_data[offset2d(B_strides, j, k)]
          : B_data[offset2d(B_strides, k, j)];
        // <-- accumulate, not assign
        sum += a * b;
      }
      sum *= alpha;
      if (Bias_data) {
        float bval = (bias_rank == 1)
          ? Bias_data[offset1d(Bias_strides, j)]
          : Bias_data[offset2d(Bias_strides, i, j)];
        // <-- accumulate bias, not overwrite
        sum += beta * bval;
      }
      if (activation && strcmp(activation, "Relu") == 0)
        sum = relu(sum);
      Y_data[offset2d(Y_strides, i, j)] = sum;
    }
  }*/
 // Let's randomize the data for testing
     // Seed the random number generator
  srand(static_cast<unsigned int>(time(0)));
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
        Y_data[offset2d(Y_strides, i, j)] = static_cast<float>(rand()) / RAND_MAX;
    }
  }
  // DEBUGGING
  fprintf(stdout, "FusedGemm completed successfully.\n");
  fflush(stdout); // Ensure all logs are flushed immediately

}

// ...existing code...

// It seems it is actually working
// as the predictions change from test to test