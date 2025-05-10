// FusedGemm.cpp
// Standalone implementation of the FusedGemm function called by krnl.call.
// Prints all attributes, tensor shapes and contents, then exits immediately.

#include <iostream>
#include <cstdlib>
#include <cstdint>

// MLIR C API memref types.
#include "mlir/IR/BuiltinTypes.h"

extern "C" {

// Signature must exactly match the krnl.call for function_name="FusedGemm":
void FusedGemm(
    MemRefType_f32_2 *output,
    MemRefType_f32_2 *A,
    MemRefType_f32_2 *B,
    MemRefType_f32_1 *C,
    const char *activation,
    float alpha,
    float beta,
    const char *domain_name,
    const char *funcName,
    int64_t numOfOutput,
    const char *onnx_node_name,
    int64_t transA,
    int64_t transB)
{
  std::cout << "=== FusedGemm called ===\n";

  // Print attributes
  std::cout << "activation      : " << activation      << "\n"
            << "alpha           : " << alpha           << "\n"
            << "beta            : " << beta            << "\n"
            << "domain_name     : " << domain_name     << "\n"
            << "funcName        : " << funcName        << "\n"
            << "numOfOutput     : " << numOfOutput     << "\n"
            << "onnx_node_name  : " << onnx_node_name  << "\n"
            << "transA          : " << transA          << "\n"
            << "transB          : " << transB          << "\n\n";

  // Print tensor shapes
  std::cout << "Shape A     : [" << A->sizes[0]      << ", " << A->sizes[1]      << "]\n"
            << "Shape B     : [" << B->sizes[0]      << ", " << B->sizes[1]      << "]\n"
            << "Shape C     : [" << C->sizes[0]      << "]\n"
            << "Shape output: [" << output->sizes[0] << ", " << output->sizes[1] << "]\n\n";

  // Print tensor contents
  std::cout << "Data A:\n";
  for (int i = 0; i < A->sizes[0]; ++i) {
    for (int j = 0; j < A->sizes[1]; ++j)
      std::cout << A->data[i * A->sizes[1] + j] << " ";
    std::cout << "\n";
  }
  std::cout << "\nData B:\n";
  for (int i = 0; i < B->sizes[0]; ++i) {
    for (int j = 0; j < B->sizes[1]; ++j)
      std::cout << B->data[i * B->sizes[1] + j] << " ";
    std::cout << "\n";
  }
  std::cout << "\nData C:\n";
  for (int i = 0; i < C->sizes[0]; ++i)
    std::cout << C->data[i] << " ";
  std::cout << "\n\nData output:\n";
  for (int i = 0; i < output->sizes[0]; ++i) {
    for (int j = 0; j < output->sizes[1]; ++j)
      std::cout << output->data[i * output->sizes[1] + j] << " ";
    std::cout << "\n";
  }
  std::cout << "=========================\n";

  std::exit(0);
}

} // extern "C"