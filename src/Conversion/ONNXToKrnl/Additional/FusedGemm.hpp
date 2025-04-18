/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- FusedGemm.hpp - Lowering FusedGemm Custom Op --------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of FusedGemm lowering.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace onnx_mlir {

// Populate the pattern list for lowering ONNX FusedGemm 
// Custom operation to Krnl
void populateLoweringONNXFusedGemmOpPattern(mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx);

} // namespace onnx_mlir
