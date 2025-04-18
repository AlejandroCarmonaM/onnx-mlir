/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- FusedGemm.cpp - Lowering FusedGemm Custom Op--------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of FusedGemm ONNX Custom operation to Krnl
//
//===----------------------------------------------------------------------===//

// build command: cmake --build . --target OMONNXToKrnl

/*Current err: "root@7dca6f6f888a:~/onnx-mlir/src/matmul_relu_matmul_fashion_mnist# onnx-mlir --EmitMLIR mnist_model_cpu_optimized.onnx 
[1/3] Fri Apr 18 15:18:44 2025 (0s) Importing ONNX Model to MLIR Module from "mnist_model_cpu_optimized.onnx"
[2/3] Fri Apr 18 15:18:44 2025 (0s) Compiling and Optimizing MLIR Module
onnx-mlir: /workdir/llvm-project/mlir/lib/Conversion/LLVMCommon/StructBuilder.cpp:22:
 mlir::StructBuilder::StructBuilder(mlir::Value): Assertion
  `LLVM::isCompatibleType(structType) && "expected llvm type"' failed.
Aborted (core dumped)"*/


#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Conversion/ONNXToKrnl/Additional/FusedGemm.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXFusedGemmOpLowering : public OpConversionPattern<ONNXCustomOp> {
  ONNXFusedGemmOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXCustomOp customOp,
      ONNXCustomOpAdaptor operandAdaptor,
      ConversionPatternRewriter &rewriter) const final {

    // Only handle FusedGemm
    StringAttr funcNameAttr = customOp.getFunctionNameAttr();
    if (!funcNameAttr || funcNameAttr.getValue() != "FusedGemm")
      return failure();

    Operation *op = customOp.getOperation();
    Location loc = op->getLoc();
    ValueRange operands = operandAdaptor.getOperands();

    // Helper builders.
    MultiDialectBuilder<AffineBuilder, IndexExprBuilderForKrnl, KrnlBuilder,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnlIE);

    // Get shape using the shape helper.
    ONNXCustomOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Prepare output allocation (FusedGemm has a single output).
    // Prepare output allocation (FusedGemm has a single output).
    Type outputType = op->getResultTypes()[0];
    MemRefType outputMemRefType =
        mlir::cast<MemRefType>(typeConverter->convertType(outputType));

    // Compute M and N from input shapes (as in your code)
    MemRefBuilder memrefBuilder(rewriter, loc);
    IntegerAttr transAAttr = customOp->getAttrOfType<IntegerAttr>("transA");
    IntegerAttr transBAttr = customOp->getAttrOfType<IntegerAttr>("transB");
    int64_t transA = transAAttr ? transAAttr.getValue().getSExtValue() : 0;
    int64_t transB = transBAttr ? transBAttr.getValue().getSExtValue() : 0;
    Value A = operands[0];
    Value B = operands[1];
    Value M, N, K;
    if (transA == 0) {
      M = memrefBuilder.dim(A, 0);
      K = memrefBuilder.dim(A, 1);
    } else {
      K = memrefBuilder.dim(A, 0);
      M = memrefBuilder.dim(A, 1);
    }
    if (transB == 0) {
      N = memrefBuilder.dim(B, 1);
    } else {
      N = memrefBuilder.dim(B, 0);
    }

    // Allocate output buffer Y with dynamic dims [M, N] if needed.
    Value outputAlloc;
    if (outputMemRefType.getNumDynamicDims() == 2) {
      outputAlloc = create.mem.alignedAlloc(outputMemRefType, {M, N});
    } else if (outputMemRefType.getNumDynamicDims() == 1) {
      if (outputMemRefType.isDynamicDim(0))
        outputAlloc = create.mem.alignedAlloc(outputMemRefType, {M});
      else
        outputAlloc = create.mem.alignedAlloc(outputMemRefType, {N});
    } else {
      outputAlloc = create.mem.alignedAlloc(outputMemRefType);
    }

    // Prepare operands for KrnlCallOp.
    Value BiasOperand = (operands.size() > 2) ? operands[2] : Value();
    Value Y = outputAlloc;

    // Prepare integer attributes as values
    Value transAVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(transA));
    Value transBVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(transB));

    
    // Compose the parameter list for KrnlCallOp
    SmallVector<Value, 9> callOperands{A, B, BiasOperand, Y, M, N, K, transAVal, transBVal};

    // List of attributes to copy (if any, or empty if not needed)
    std::vector<std::string> attributeNames; // Add attribute names if you want to copy any

    // Lower to KrnlCallOp using the correct builder signature
    rewriter.create<KrnlCallOp>(
        loc,
        "ort_cpu_ep_fused_gemm", // function name as string
        SmallVector<Value, 1>{Y}, // outputs
        op,                       // original op (for attribute copying)
        callOperands,             // inputs
        attributeNames            // attributes to copy
    );

    rewriter.replaceOp(op, Y);
    return success();
      }
};

void populateONNXToKrnlConversionAdditionalPass(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXFusedGemmOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir

void onnx_mlir::populateLoweringONNXFusedGemmOpPattern(
    mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter,
    mlir::MLIRContext *ctx) {
  populateONNXToKrnlConversionAdditionalPass(patterns, typeConverter, ctx);
}