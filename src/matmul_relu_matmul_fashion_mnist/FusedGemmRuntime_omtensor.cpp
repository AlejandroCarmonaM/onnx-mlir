/**********************************************
 * IMPORT LIBRARIES
 **********************************************/
#include <cstdint>  // For int64_t
#include <vector>   // Not strictly used here, but common
#include <cmath>    // For std::max
#include <iostream> // For std::cout, std::endl (placeholder logging)
#include <cstdio>   // For fprintf, stdout, fflush (debug logging)
#include <cstring>

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


// ...existing code...

extern "C" void FusedGemm(
  OMTensor* Y_omTensor,
  OMTensor* A_omTensor,
  OMTensor* B_omTensor,
  OMTensor* Bias_omTensor,
  const char *activation,
  float alpha,
  float beta,
  const char *domain_name,
  const char *onnx_node_name,
  int64_t transA,
  int64_t transB
) {
  const char *layout = "NHWC"; // Default layout
  // Defensive checks
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

  int64_t A_rank = omTensorGetRank(A_omTensor);
  int64_t B_rank = omTensorGetRank(B_omTensor);
  int64_t Y_rank = omTensorGetRank(Y_omTensor);

  if (A_rank != 2 || B_rank != 2 || Y_rank != 2) {
    fprintf(stdout, "Error: Only 2D tensors are supported for A, B, and Y.\n");
    return;
  }

  // Layout detection
  bool isNHWC = (layout && strcmp(layout, "NHWC") == 0);
  bool isNCHW = !isNHWC; // Default to NCHW

  int64_t M = Y_shape[0];
  int64_t N = Y_shape[1];
  int64_t K = transA ? A_shape[0] : A_shape[1];

  if (M <= 0 || N <= 0 || K <= 0) {
    fprintf(stdout, "Error: Invalid dimensions M=%lld, N=%lld, K=%lld.\n", M, N, K);
    return;
  }

  // Bias handling
  float *Bias_data = nullptr;
  const int64_t* Bias_shape = nullptr;
  const int64_t* Bias_strides = nullptr;
  int64_t bias_rank = 0;
  if (Bias_omTensor) {
    if (omTensorGetDataType(Bias_omTensor) != ONNX_TYPE_FLOAT) {
      fprintf(stdout, "Error: Bias tensor must be of type float.\n");
      return;
    }
    bias_rank = omTensorGetRank(Bias_omTensor);
    Bias_shape = omTensorGetShape(Bias_omTensor);
    Bias_strides = omTensorGetStrides(Bias_omTensor);
    Bias_data = reinterpret_cast<float *>(omTensorGetDataPtr(Bias_omTensor));
    if (!Bias_strides) {
      fprintf(stdout, "Error: Bias tensor has null strides.\n");
      return;
    }
    for (int i = 0; i < bias_rank; ++i) {
      if (Bias_strides[i] < 0) {
        fprintf(stdout, "Error: Bias strides must be non-negative.\n");
        return;
      }
    }
  }

  float *A_data = reinterpret_cast<float *>(omTensorGetDataPtr(A_omTensor));
  float *B_data = reinterpret_cast<float *>(omTensorGetDataPtr(B_omTensor));
  float *Y_data = reinterpret_cast<float *>(omTensorGetDataPtr(Y_omTensor));

  // Main computation
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        float a, b;
        if (isNHWC) {
          // NHWC: i = batch, j = channel
          a = transA
            ? A_data[offset2d(A_strides, k, i)]
            : A_data[offset2d(A_strides, i, k)];
          b = transB
            ? B_data[offset2d(B_strides, j, k)]
            : B_data[offset2d(B_strides, k, j)];
        } else {
          // NCHW: i = batch, j = channel
          a = transA
            ? A_data[offset2d(A_strides, k, i)]
            : A_data[offset2d(A_strides, i, k)];
          b = transB
            ? B_data[offset2d(B_strides, j, k)]
            : B_data[offset2d(B_strides, k, j)];
        }
        sum += a * b;
      }
      sum *= alpha;

      // Robust bias broadcasting: scalar, 1D, or 2D
      if (Bias_data) {
        float bval = 0.0f;
        if (bias_rank == 0) {
          bval = Bias_data[0];
        } else if (bias_rank == 1) {
          // For NHWC, bias is usually on channel (j)
          bval = Bias_data[offset1d(Bias_strides, j)];
        } else if (bias_rank == 2) {
          bval = Bias_data[offset2d(Bias_strides, i, j)];
        }
        sum += beta * bval;
      }

      if (activation && strcmp(activation, "Relu") == 0)
        sum = relu(sum);

      Y_data[offset2d(Y_strides, i, j)] = sum;
    }
  }

  fprintf(stdout, "FusedGemm completed successfully.\n");
  fflush(stdout);
}

// ...existing code...

// ...existing code...