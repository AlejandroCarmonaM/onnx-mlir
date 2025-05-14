#include <stdio.h>
#include <stdint.h> // For int64_t
#include <stdlib.h> // For malloc/free (optional, could use stack arrays)

// Declare the external function from FusedGemmRuntime.o
extern void ort_cpu_ep_fused_gemm(
    const float* A,
    const float* B,
    const float* Bias,
    float* Y,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t transA,
    int64_t transB
);

// Helper function to print a matrix
void print_matrix(const char* name, const float* matrix, int64_t rows, int64_t cols) {
    printf("%s (%lld x %lld):\n", name, (long long)rows, (long long)cols);
    for (int64_t i = 0; i < rows; ++i) {
        printf("  [");
        for (int64_t j = 0; j < cols; ++j) {
            printf("%8.3f", matrix[i * cols + j]);
            if (j < cols - 1) printf(", ");
        }
        printf("]\n");
    }
    printf("\n");
}

int main() {
    // --- Define Sample Data ---
    // Example: A (2x3) @ B (3x2) + Bias (2) -> Y (2x2)
    // No transpose for simplicity first (transA=0, transB=0)

    const int64_t M = 2;
    const int64_t N = 2;
    const int64_t K = 3;
    const int64_t transA = 0;
    const int64_t transB = 0;

    // Matrix A (M x K) = (2 x 3)
    float A[] = {
        1.0f, 2.0f, 3.0f,  // Row 0
        4.0f, 5.0f, 6.0f   // Row 1
    };

    // Matrix B (K x N) = (3 x 2)
    float B[] = {
        7.0f,  8.0f,   // Row 0
        9.0f, 10.0f,   // Row 1
       11.0f, 12.0f    // Row 2
    };

    // Bias (N) = (2)
    float Bias[] = { 0.1f, -0.2f };

    // Output Matrix Y (M x N) = (2 x 2) - Allocate space
    float Y[M * N]; // Use stack allocation for small example

    printf("--- Input Data ---\n");
    print_matrix("Matrix A", A, M, K);
    print_matrix("Matrix B", B, K, N);
    print_matrix("Bias", Bias, 1, N); // Print bias as a row vector

    // --- Call the Fused GEMM function ---
    printf("--- Calling ort_cpu_ep_fused_gemm ---\n");
    ort_cpu_ep_fused_gemm(A, B, Bias, Y, M, N, K, transA, transB);
    printf("--- Returned from ort_cpu_ep_fused_gemm ---\n\n");

    // --- Print the Result ---
    print_matrix("Result Y", Y, M, N);

    // --- Expected Result Calculation (Manual for verification) ---
    // Y[0,0] = relu((1*7 + 2*9 + 3*11) + 0.1) = relu(7 + 18 + 33 + 0.1) = relu(58.1) = 58.1
    // Y[0,1] = relu((1*8 + 2*10 + 3*12) - 0.2) = relu(8 + 20 + 36 - 0.2) = relu(63.8) = 63.8
    // Y[1,0] = relu((4*7 + 5*9 + 6*11) + 0.1) = relu(28 + 45 + 66 + 0.1) = relu(139.1) = 139.1
    // Y[1,1] = relu((4*8 + 5*10 + 6*12) - 0.2) = relu(32 + 50 + 72 - 0.2) = relu(153.8) = 153.8
    printf("--- Expected Result (Manual Calculation) ---\n");
    printf("  [  58.100,   63.800]\n");
    printf("  [ 139.100,  153.800]\n\n");


    // --- Test with Transpose B ---
    // A (2x3) @ B' (2x3) -> Y (2x2) ? This doesn't match dimensions.
    // Let's redefine B to be (N x K) = (2 x 3) so B' is (K x N) = (3 x 2)
    printf("--- Testing Transpose B ---\n");
    const int64_t transB_test = 1;
    float B_t[] = { // B is now (N x K) = (2 x 3)
        7.0f, 9.0f, 11.0f, // Represents column 0 of original B
        8.0f, 10.0f, 12.0f // Represents column 1 of original B
    };
    print_matrix("Matrix A", A, M, K);
    print_matrix("Matrix B (Layout for Transpose)", B_t, N, K); // Note dimensions N, K
    print_matrix("Bias", Bias, 1, N);

    printf("--- Calling ort_cpu_ep_fused_gemm (transB=1) ---\n");
    ort_cpu_ep_fused_gemm(A, B_t, Bias, Y, M, N, K, transA, transB_test);
    printf("--- Returned from ort_cpu_ep_fused_gemm ---\n\n");
    print_matrix("Result Y (transB=1)", Y, M, N);
    printf("--- Expected Result (Should be same as before) ---\n");
    printf("  [  58.100,   63.800]\n");
    printf("  [ 139.100,  153.800]\n\n");


    return 0;
}