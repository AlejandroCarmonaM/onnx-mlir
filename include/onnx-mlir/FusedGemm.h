#ifndef ONNX_MLIR_FUSEDGEMM_H
#define ONNX_MLIR_FUSEDGEMM_H

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Signature matching krnl.call for function "FusedGemm".
void FusedGemm(
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
    int64_t transB
);

#ifdef __cplusplus
}
#endif

#endif // ONNX_MLIR_FUSEDGEMM_H
