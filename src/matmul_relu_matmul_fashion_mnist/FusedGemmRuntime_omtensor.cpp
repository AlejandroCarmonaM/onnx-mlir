#include <cstdint>
#include <vector>
#include <cmath>
#include <iostream>
#include <cstdio>

// Include the ONNX-MLIR Runtime header defining OMTensor and API functions
// Adjust the path based on your build/install location if necessary.
// Common locations might be include/onnx-mlir/Runtime/ or similar.
#include "OnnxMlirRuntime.h"

// Basic ReLU activation
float relu(float x) {
    return std::max(0.0f, x);
}

// Modified implementation accepting OMTensor*
extern "C" void ort_cpu_ep_fused_gemm(
    OMTensor* A_omTensor,  // OMTensor for Matrix A
    OMTensor* B_omTensor,  // OMTensor for Matrix B
    OMTensor* Bias_omTensor,// OMTensor for Bias (can be NULL)
    OMTensor* Y_omTensor,  // OMTensor for Output Y
    int64_t M,             // Dimension M (passed directly)
    int64_t N,             // Dimension N (passed directly)
    int64_t K,             // Dimension K (passed directly)
    int64_t transA,        // Transpose A flag (passed directly)
    int64_t transB         // Transpose B flag (passed directly)
) {

    // Use fprintf to stderr to ensure it prints immediately before a crash
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
         // Consider returning or aborting if essential tensors are missing
         return; // Or handle error appropriately
    }

    // Extract raw data pointers from OMTensors
    const float* A = static_cast<const float*>(omTensorGetDataPtr(A_omTensor));
    const float* B = static_cast<const float*>(omTensorGetDataPtr(B_omTensor));
    float* Y       = static_cast<float*>(omTensorGetDataPtr(Y_omTensor));
    const float* Bias = nullptr; // Initialize Bias pointer to null

    // Bias is optional, only extract if the OMTensor is not NULL
    if (Bias_omTensor) {
        Bias = static_cast<const float*>(omTensorGetDataPtr(Bias_omTensor));
    }

    fprintf(stderr, "    Extracted A ptr: %p, B ptr: %p, Bias ptr: %p, Y ptr: %p\n",
            (void*)A, (void*)B, (void*)Bias, (void*)Y);
    fflush(stderr);

    // Check for NULL pointers *after* extraction (omTensorGetDataPtr might return null)
    if (!A || !B || !Y) {
         fprintf(stderr, "    ERROR: Extracted data pointer for A, B, or Y is NULL!\n");
         fflush(stderr);
         return; // Or handle error appropriately
    }

    // --- Core Logic (remains the same, using extracted pointers A, B, Bias, Y) ---
    std::cout << ">>> Running C++ Placeholder: ort_cpu_ep_fused_gemm <<<" << std::endl;
    std::cout << "    M=" << M << ", N=" << N << ", K=" << K
              << ", transA=" << transA << ", transB=" << transB << std::endl;

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                int64_t a_idx = transA ? (k * M + m) : (m * K + k);
                int64_t b_idx = transB ? (n * K + k) : (k * N + n);

                // Basic bounds check (more important now with potentially complex layouts)
                // TODO: Consider using omTensorGetStride(A_omTensor, dim) if layout isn't guaranteed row-major
                // For now, assume dense row-major based on M, N, K for simplicity
                int64_t max_a_idx = transA ? (K * M) : (M * K);
                int64_t max_b_idx = transB ? (N * K) : (K * N);
                if (a_idx < 0 || a_idx >= max_a_idx || b_idx < 0 || b_idx >= max_b_idx) {
                     fprintf(stderr, "    ERROR: Calculated index out of bounds! a_idx=%lld (max %lld), b_idx=%lld (max %lld)\n",
                             (long long)a_idx, (long long)max_a_idx, (long long)b_idx, (long long)max_b_idx);
                     // Handle error: skip, return, abort?
                     continue;
                }

                sum += A[a_idx] * B[b_idx];

                // Debug print for first element
                if (m == 0 && n == 0 && k == 0) {
                     fprintf(stderr, "    Loop(0,0,0): a_idx=%lld, b_idx=%lld\n", (long long)a_idx, (long long)b_idx);
                     fprintf(stderr, "    Loop(0,0,0): A[a_idx]=%f, B[b_idx]=%f\n", A[a_idx], B[b_idx]);
                     fflush(stderr);
                }
            }

            // Add Bias (check Bias pointer is valid)
            float biased_sum = sum + (Bias ? Bias[n] : 0.0f);

            // Apply ReLU activation
            // TODO: Check bounds for Y write: m * N + n < M * N
            if ((m * N + n) >= (M * N)) {
                 fprintf(stderr, "    ERROR: Output index out of bounds! Y_idx=%lld (max %lld)\n",
                         (long long)(m * N + n), (long long)(M * N));
                 continue; // Skip write
            }
            Y[m * N + n] = relu(biased_sum);
        }
    }
    // --- End Core Logic ---

    std::cout << ">>> Finished C++ Placeholder: ort_cpu_ep_fused_gemm <<<" << std::endl;
    fprintf(stderr, ">>> Finished C++ ort_cpu_ep_fused_gemm (OMTensor version) <<<\n");
    fflush(stderr);
}