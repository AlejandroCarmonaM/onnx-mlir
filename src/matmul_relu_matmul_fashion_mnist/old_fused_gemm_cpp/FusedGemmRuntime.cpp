#include <cstdint> // For int64_t
#include <vector>
#include <cmath>   // For std::max
#include <iostream> // For debug prints
#include <cstdio> // Include for fprintf

// build command: cmake --build . --target OMTensorUtils

// Basic ReLU activation
float relu(float x) {
    return std::max(0.0f, x);
}

// Placeholder implementation for FusedGemm: Y = ReLU(alpha * (A @ B) + beta * Bias)
// We simplify to: Y = ReLU((A @ B) + Bias) by assuming alpha=1.0, beta=1.0 for this placeholder.
extern "C" void ort_cpu_ep_fused_gemm(
    const float* A,     // Pointer to Matrix A data
    const float* B,     // Pointer to Matrix B data
    const float* Bias,  // Pointer to Bias data
    float* Y,           // Pointer to Output data Y
    int64_t M,          // Dimension M of Output (Rows of A or B')
    int64_t N,          // Dimension N of Output (Cols of B or A')
    int64_t K,          // Dimension K (Cols of A or Rows of A', Rows of B or Cols of B')
    int64_t transA,     // Transpose A flag (0 or 1)
    int64_t transB      // Transpose B flag (0 or 1)
    // Note: alpha, beta, and activation type are ignored in this simple placeholder
) {

    // Use fprintf to stderr to ensure it prints immediately before a crash
    fprintf(stderr, ">>> C++ ort_cpu_ep_fused_gemm called:\n");
    fprintf(stderr, "    M=%lld, N=%lld, K=%lld, transA=%lld, transB=%lld\n",
            (long long)M, (long long)N, (long long)K, (long long)transA, (long long)transB);
    fprintf(stderr, "    A ptr: %p, B ptr: %p, Bias ptr: %p, Y ptr: %p\n",
            (void*)A, (void*)B, (void*)Bias, (void*)Y);
    fflush(stderr); // Ensure output is flushed

    // Check for NULL pointers immediately
    if (!A || !B || !Y) { // Bias might be optional
         fprintf(stderr, "    ERROR: Received NULL pointer for A, B, or Y!\n");
         fflush(stderr);
         // Decide how to handle: maybe return early or abort?
         // For debugging, maybe just continue to see where it crashes.
    }

    std::cout << ">>> Running C++ Placeholder: ort_cpu_ep_fused_gemm <<<" << std::endl;
    std::cout << "    M=" << M << ", N=" << N << ", K=" << K
              << ", transA=" << transA << ", transB=" << transB << std::endl;

    // Simple Row-Major Matrix Multiplication Logic
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                // Calculate indices based on transpose flags
                // Assuming Row-Major layout for both A and B
                int64_t a_idx = transA ? (k * M + m) : (m * K + k); // Index for A[m, k] or A[k, m]
                int64_t b_idx = transB ? (n * K + k) : (k * N + n); // Index for B[k, n] or B[n, k]

                // Basic bounds check (optional but good practice)
                // These checks depend heavily on how strides/leading dimensions would be handled
                // For placeholder, assume dense packing based on M, N, K
                // if (a_idx >= (transA ? K*M : M*K) || b_idx >= (transB ? N*K : K*N)) {
                //     std::cerr << "Error: Index out of bounds!" << std::endl;
                //     continue; // Skip this element
                // }

                sum += A[a_idx] * B[b_idx];

                // Add print inside the loop (maybe only first iteration)
                if (m == 0 && n == 0 && k == 0) {
                     fprintf(stderr, "    Loop(0,0,0): a_idx=%lld, b_idx=%lld\n", (long long)a_idx, (long long)b_idx);
                     // Maybe print A[a_idx] and B[b_idx] if pointers are valid
                     if (A && B) {
                         // Add bounds checks before accessing!
                         // Example check (adjust based on actual expected sizes):
                         bool a_ok = transA ? (k < K && m < M) : (m < M && k < K);
                         bool b_ok = transB ? (n < N && k < K) : (k < K && n < N);
                         if (a_ok && b_ok) {
                            // Calculate actual max index based on dimensions
                            int64_t max_a_idx = transA ? (K * M) : (M * K);
                            int64_t max_b_idx = transB ? (N * K) : (K * N);
                            if (a_idx >= 0 && a_idx < max_a_idx && b_idx >= 0 && b_idx < max_b_idx) {
                               fprintf(stderr, "    Loop(0,0,0): A[a_idx]=%f, B[b_idx]=%f\n", A[a_idx], B[b_idx]);
                            } else {
                               fprintf(stderr, "    Loop(0,0,0): Calculated indices a_idx=%lld or b_idx=%lld seem out of bounds (max_a=%lld, max_b=%lld)!\n",
                                       (long long)a_idx, (long long)b_idx, (long long)max_a_idx, (long long)max_b_idx);
                            }
                         } else {
                             fprintf(stderr, "    Loop(0,0,0): m, n, or k seem out of bounds for dimensions M, N, K!\n");
                         }
                     }
                     fflush(stderr);
                }
            }

            // Add Bias (assuming Bias has size N)
            float biased_sum = sum + (Bias ? Bias[n] : 0.0f);

            // Apply ReLU activation
            Y[m * N + n] = relu(biased_sum);
        }
    }
     std::cout << ">>> Finished C++ Placeholder: ort_cpu_ep_fused_gemm <<<" << std::endl;
     fprintf(stderr, ">>> Finished C++ ort_cpu_ep_fused_gemm <<<\n");
     fflush(stderr);
}