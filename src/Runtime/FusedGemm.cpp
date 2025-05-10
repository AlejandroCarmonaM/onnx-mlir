#include "onnx-mlir/FusedGemm.h"
#include <iostream>
#include <cstdlib>

extern "C" void FusedGemm(
    MemRefType_float_2 *output,
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
    int64_t transB) {
  std::cout << "FusedGemm called" << std::endl;
  // Print attributes
  std::cout << "activation: " << activation << std::endl;
  std::cout << "alpha: " << alpha << std::endl;
  std::cout << "beta: " << beta << std::endl;
  std::cout << "domain_name: " << domain_name << std::endl;
  std::cout << "funcName: " << funcName << std::endl;
  std::cout << "numOfOutput: " << numOfOutput << std::endl;
  std::cout << "onnx_node_name: " << onnx_node_name << std::endl;
  std::cout << "transA: " << transA << std::endl;
  std::cout << "transB: " << transB << std::endl;

  // Print shapes
  std::cout << "Shape A: [" << A->sizes[0] << ", " << A->sizes[1] << "]" << std::endl;
  std::cout << "Shape B: [" << B->sizes[0] << ", " << B->sizes[1] << "]" << std::endl;
  std::cout << "Shape C: [" << C->sizes[0] << "]" << std::endl;
  std::cout << "Shape output: [" << output->sizes[0] << ", " << output->sizes[1] << "]" << std::endl;

  // Print data (first few elements)
  std::cout << "Data A:\n";
  for (int i = 0; i < A->sizes[0]; ++i) {
    for (int j = 0; j < A->sizes[1]; ++j)
      std::cout << A->data[i * A->sizes[1] + j] << " ";
    std::cout << std::endl;
  }

  std::cout << "Data B:\n";
  for (int i = 0; i < B->sizes[0]; ++i) {
    for (int j = 0; j < B->sizes[1]; ++j)
      std::cout << B->data[i * B->sizes[1] + j] << " ";
    std::cout << std::endl;
  }

  std::cout << "Data C:\n";
  for (int i = 0; i < C->sizes[0]; ++i)
    std::cout << C->data[i] << " ";
  std::cout << std::endl;

  std::cout << "Data output:\n";
  for (int i = 0; i < output->sizes[0]; ++i) {
    for (int j = 0; j < output->sizes[1]; ++j)
      std::cout << output->data[i * output->sizes[1] + j] << " ";
    std::cout << std::endl;
  }

  std::exit(0);
}
