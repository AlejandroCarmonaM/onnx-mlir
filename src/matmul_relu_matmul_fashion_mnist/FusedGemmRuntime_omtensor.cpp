/**********************************************
 * IMPORT LIBRARIES
 **********************************************/
#include <cstdint>  // For int64_t
#include <vector>   // Not strictly used here, but common
#include <cmath>    // For std::max
#include <iostream> // For std::cout, std::endl (placeholder logging)
#include <cstdio>   // For fprintf, stderr, fflush (debug logging)

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

    /******************************************
     * INITIAL LOGGING & VALIDATION
     ******************************************/
    // Use fprintf for immediate output, helpful before potential crashes
    fprintf(stderr, ">>> C++ ort_cpu_ep_fused_gemm (OMTensor version) called:\n");
    fprintf(stderr, "    M=%lld, N=%lld, K=%lld, transA=%lld, transB=%lld\n",
            (long long)M, (long long)N, (long long)K, (long long)transA, (long long)transB);
    fprintf(stderr, "    A OMTensor*: %p, B OMTensor*: %p, Bias OMTensor*: %p, Y OMTensor*: %p\n",
            (void*)A_omTensor, (void*)B_omTensor, (void*)Bias_omTensor, (void*)Y_omTensor);
    fflush(stderr);

    // Check for NULL OMTensor pointers for required inputs/outputs
    if (!A_omTensor || !B_omTensor || !Y_omTensor) {
         fprintf(stderr, "    ERROR: Received NULL OMTensor pointer for A, B, or Y!\n");
         fflush(stderr);
         return; // Cannot proceed
    }

    /******************************************
     * EXTRACT DATA POINTERS & METADATA
     ******************************************/
    // Extract raw data pointers (assuming float32 based on typical usage)
    // TODO: Add type checking if supporting other data types is needed.
    const float* A_data = static_cast<const float*>(omTensorGetDataPtr(A_omTensor));
    const float* B_data = static_cast<const float*>(omTensorGetDataPtr(B_omTensor));
    float* Y_data       = static_cast<float*>(omTensorGetDataPtr(Y_omTensor));

    // Get strides for A, B, Y (crucial for correct indexing)
    const int64_t* strideA = omTensorGetStrides(A_omTensor);
    const int64_t* strideB = omTensorGetStrides(B_omTensor);
    const int64_t* strideY = omTensorGetStrides(Y_omTensor);

    // Check for NULL pointers *after* extraction
    if (!A_data || !B_data || !Y_data || !strideA || !strideB || !strideY) {
         fprintf(stderr, "    ERROR: Extracted data pointer or strides for A, B, or Y is NULL!\n");
         fprintf(stderr, "    A_data=%p, B_data=%p, Y_data=%p, strideA=%p, strideB=%p, strideY=%p\n",
                 (void*)A_data, (void*)B_data, (void*)Y_data, (void*)strideA, (void*)strideB, (void*)strideY);
         fflush(stderr);
         return; // Cannot proceed
    }

    // Extract Bias data (if present)
    const float* Bias_data = nullptr;
    const int64_t* strideBias = nullptr;
    const int64_t* dimsBias = nullptr;
    int biasRank = 0;
    if (Bias_omTensor) {
        Bias_data = static_cast<const float*>(omTensorGetDataPtr(Bias_omTensor));
        strideBias = omTensorGetStrides(Bias_omTensor);
        /*(doesn't work)*/ //dimsBias = omTensorGetDimensions(Bias_omTensor); // Needed for broadcasting rules
        dimsBias = omTensorGetShape(Bias_omTensor); // Assuming this function gives the shape/dims
        biasRank = omTensorGetRank(Bias_omTensor);

        if (!Bias_data || !strideBias || !dimsBias) {
             fprintf(stderr, "    WARNING: Bias OMTensor exists but extracted data pointer, strides, or dims is NULL! Treating as no bias.\n");
             fprintf(stderr, "    Bias_data=%p, strideBias=%p, dimsBias=%p\n", (void*)Bias_data, (void*)strideBias, (void*)dimsBias);
             fflush(stderr);
             Bias_data = nullptr; // Treat as if no bias was provided
        } else {
             fprintf(stderr, "    Bias Info: Rank=%d, Data=%p, Strides=%p, Dims=%p\n", biasRank, (void*)Bias_data, (void*)strideBias, (void*)dimsBias);
             // Optional: Print actual dims/strides
             // for(int i=0; i<biasRank; ++i) fprintf(stderr, " Bias dim[%d]=%lld, stride[%d]=%lld\n", i, (long long)dimsBias[i], i, (long long)strideBias[i]);
        }
    } else {
        fprintf(stderr, "    Bias OMTensor is NULL.\n");
    }
    fflush(stderr);


    /******************************************
     * CORE GEMM + BIAS + RELU LOGIC
     ******************************************/
    // Note: This implementation assumes alpha=1.0 and beta=1.0 as per Gemm defaults,
    // because these attributes are not passed to this custom function.

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {

            // --- GEMM Calculation (A' * B') ---
            float gemm_sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                // Determine logical indices based on transpose flags
                int64_t a_idx_dim0 = transA ? k : m;
                int64_t a_idx_dim1 = transA ? m : k;
                int64_t b_idx_dim0 = transB ? n : k;
                int64_t b_idx_dim1 = transB ? k : n;

                // Calculate physical offsets using strides
                int64_t offset_a = offset2d(strideA, a_idx_dim0, a_idx_dim1);
                int64_t offset_b = offset2d(strideB, b_idx_dim0, b_idx_dim1);

                // Accumulate product
                gemm_sum += A_data[offset_a] * B_data[offset_b];
            } // End K loop

            // --- Bias Addition (gemm_sum + C/Bias) ---
            // Implements unidirectional broadcasting for C/Bias to shape (M, N)
            bool biasIsValid = (Bias_data && strideBias && dimsBias);
            float bias_val = 0.0f;
            if (biasIsValid) {
              switch (biasRank) {
                case 0:
                  bias_val = Bias_data[0];
                  break;
                case 1:
                  // length == N? broadcast across rows
                  if (dimsBias[0] == N)
                    bias_val = Bias_data[offset1d(strideBias, n)];
                  // length == M? broadcast across cols
                  else if (dimsBias[0] == M)
                    bias_val = Bias_data[offset1d(strideBias, m)];
                  // length == 1? scalar
                  else if (dimsBias[0] == 1)
                    bias_val = Bias_data[0];
                  else
                    fprintf(stderr, "[FusedGemm] Bad 1D bias length %lld\n", (long long)dimsBias[0]);
                  break;
                case 2:
                  if (dimsBias[0] == M && dimsBias[1] == N)
                    bias_val = Bias_data[offset2d(strideBias, m, n)];
                  else if (dimsBias[0] == 1 && dimsBias[1] == N)
                    bias_val = Bias_data[offset2d(strideBias, 0, n)];
                  else if (dimsBias[0] == M && dimsBias[1] == 1)
                    bias_val = Bias_data[offset2d(strideBias, m, 0)];
                  else if (dimsBias[0] == 1 && dimsBias[1] == 1)
                    bias_val = Bias_data[0];
                  else
                    fprintf(stderr, "[FusedGemm] Bad 2D bias shape %lldx%lld\n",
                            (long long)dimsBias[0], (long long)dimsBias[1]);
                  break;
                default:
                  fprintf(stderr, "[FusedGemm] Unsupported bias rank %d\n", biasRank);
              }
            }
            // Assuming beta = 1.0 for Bias/C
            float biased_sum = gemm_sum + bias_val;

            // --- ReLU Activation ---
            float final_result = relu(biased_sum);

            // --- Store Result in Output Tensor Y ---
            // Calculate output offset using strides
            int64_t offset_y = offset2d(strideY, m, n);
            Y_data[offset_y] = final_result;

        } // End N loop
    } // End M loop

    /******************************************
     * FINAL LOGGING
     ******************************************/
    // std::cout << ">>> Finished C++ Placeholder: ort_cpu_ep_fused_gemm <<<" << std::endl; // Less critical logging
    fprintf(stderr, ">>> Finished C++ ort_cpu_ep_fused_gemm (OMTensor version) <<<\n");
    fflush(stderr);
}