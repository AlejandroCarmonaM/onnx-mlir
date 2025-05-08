/**********************************************
 * IMPORT LIBRARIES
 **********************************************/
#include <cstdint>  // For int64_t
#include <vector>   // Not strictly used here, but common
#include <cmath>    // For std::max
#include <iostream> // For std::cout, std::endl (placeholder logging)
#include <cstdio>   // For fprintf, stdout, fflush (debug logging)

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
extern "C" void ort_cpu_ep_fused_gemm(
    OMTensor* A_omTensor,
    OMTensor* B_omTensor,
    OMTensor* Bias_omTensor, // Corresponds to Gemm's 'C' input
    OMTensor* Y_omTensor,
    int64_t M,
    int64_t N,
    int64_t K,
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
  if (Bias_omTensor) {
    const int64_t* Bias_strides = omTensorGetStrides(Bias_omTensor);
    if (!Bias_strides) {
      fprintf(stdout, "Error: Bias tensor has null strides.\n");
      return;
    }
    // Check if Bias strides are non-negative
    for (int i = 0; i < 2; ++i) {
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
          omTensorGetShape(A_omTensor)[0],
          omTensorGetShape(A_omTensor)[1],
          A_strides[0], A_strides[1]);
  fprintf(stdout, "  B_omTensor: rank=%d, dtype=%d, shape=[%lld,%lld], strides=[%lld,%lld]\n",
          omTensorGetRank(B_omTensor),
          omTensorGetDataType(B_omTensor),
          omTensorGetShape(B_omTensor)[0],
          omTensorGetShape(B_omTensor)[1],
          B_strides[0], B_strides[1]);
  fprintf(stdout, "  Y_omTensor: rank=%d, dtype=%d, shape=[%lld,%lld], strides=[%lld,%lld]\n",
          omTensorGetRank(Y_omTensor),
          omTensorGetDataType(Y_omTensor),
          omTensorGetShape(Y_omTensor)[0],
          omTensorGetShape(Y_omTensor)[1],
          Y_strides[0], Y_strides[1]);
  fprintf(stdout, "  M=%lld, N=%lld, K=%lld, transA=%lld, transB=%lld\n",
          M, N, K, transA, transB);
  fflush(stdout); // Ensure all logs are flushed immediately
  // END LOGGING
  exit(EXIT_SUCCESS); // Placeholder for actual implementation
}