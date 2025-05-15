/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Custom.cpp - Lowering Custom Op--------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNXCustomOp to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXCustomOpLowering : public OpConversionPattern<ONNXCustomOp> {
  ONNXCustomOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXCustomOp customOp,
      ONNXCustomOpAdaptor operandAdaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = customOp.getOperation();
    Location loc = op->getLoc();
    ValueRange operands = operandAdaptor.getOperands();

    // Helper builders.
    MultiDialectBuilder<AffineBuilder, IndexExprBuilderForKrnl, KrnlBuilder,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnlIE);

    // Get function_name attribute.
    std::string functionName = customOp.getFunctionName().str();
    SmallVector<Type, 4> outputMemRefTypes;
    SmallVector<Value, 4> outputAllocs;

    bool handled = false;
    if (functionName == "FusedGemm") {
      // Output of FusedGemm is Y = A X B + C (where X is matmul)
      // Output shape is (M, N)
      // M is derived from input A (operands[0]) and transA attribute.
      // N is derived from input B (operands[1]) and transB attribute.

      Type outputTensorMLIRType = op->getResultTypes()[0];
      auto rankedOutputTensorType = mlir::dyn_cast<mlir::RankedTensorType>(outputTensorMLIRType);

      if (!rankedOutputTensorType)
        return rewriter.notifyMatchFailure(op, "FusedGemm result must be a ranked tensor");

      if (rankedOutputTensorType.getRank() != 2)
        return rewriter.notifyMatchFailure(op, "FusedGemm result is expected to be a 2D tensor");

      // Get attributes
      IntegerAttr transAAttr = customOp->getAttrOfType<IntegerAttr>("transA");
      IntegerAttr transBAttr = customOp->getAttrOfType<IntegerAttr>("transB");

      // FusedGemm expects transA and transB attributes.
      if (!transAAttr || !transBAttr)
        return rewriter.notifyMatchFailure(op, "FusedGemm is missing transA or transB attribute");

      // Use getValue().getSExtValue() for potentially signed integers,
      // or check the type more carefully if it could be unsigned.
      // For boolean conversion from 0 or 1, checking non-zero is fine.
      bool transA = (transAAttr.getValue().getSExtValue() != 0);
      bool transB = (transBAttr.getValue().getSExtValue() != 0);

      Value inputA = operands[0]; // Corresponds to %arg0 in ONNXIR
      Value inputB = operands[1]; // Corresponds to %0 in ONNXIR

      SmallVector<IndexExpr, 4> outputDims;

      // Determine M (output dimension 0)
      int64_t dimM_size = rankedOutputTensorType.getDimSize(0);
      if (mlir::ShapedType::isDynamic(dimM_size)) {
        if (transA) { // M = A.shape[1]
          outputDims.emplace_back(create.krnlIE.getShapeAsDim(inputA, 1));
        } else { // M = A.shape[0]
          outputDims.emplace_back(create.krnlIE.getShapeAsDim(inputA, 0));
        }
      // If M is static, use the static size.
      } else {
        outputDims.emplace_back(LiteralIndexExpr(dimM_size));
      }

      // Determine N (output dimension 1)
      int64_t dimN_size = rankedOutputTensorType.getDimSize(1);
      if (mlir::ShapedType::isDynamic(dimN_size)) {
        if (transB) { // N = B.shape[0]
          outputDims.emplace_back(create.krnlIE.getShapeAsDim(inputB, 0));
        } else { // N = B.shape[1]
          outputDims.emplace_back(create.krnlIE.getShapeAsDim(inputB, 1));
        }
      // If N is static, use the static size.
      } else {
        outputDims.emplace_back(LiteralIndexExpr(dimN_size));
      }

      // Allocate output memref.
      auto memRefType = mlir::cast<mlir::MemRefType>(typeConverter->convertType(outputTensorMLIRType));
      outputMemRefTypes.emplace_back(memRefType);
      Value alloc = create.mem.alignedAlloc(memRefType, outputDims);
      outputAllocs.emplace_back(alloc);
      handled = true;
    }

    if (!handled) {
      // Default: try shape helper, as before
      ONNXCustomOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
      if (failed(shapeHelper.computeShape())) {
        // If shape inference fails, emit a runtime error or fallback
        return rewriter.notifyMatchFailure(op, "Shape inference failed for custom op");
      }
      for (size_t idx = 0; idx < op->getResultTypes().size(); idx++) {
        Type ty = op->getResultTypes()[idx];
        MemRefType outputMemRefType =
            mlir::cast<MemRefType>(typeConverter->convertType(ty));
        outputMemRefTypes.emplace_back(outputMemRefType);
        Value alloc = create.mem.alignedAlloc(
            outputMemRefType, shapeHelper.getOutputDims(idx));
        outputAllocs.emplace_back(alloc);
      }
    }

    // Lower to Krnl for special CustomOp
    // Create Krnl.Call
    std::vector<std::string> excludeStrings = {"function_name",
        "shape_infer_pattern", "inputs_for_infer", "output_element_type"};
    std::vector<std::string> attributeNames;
    for (NamedAttribute namedAttr : customOp->getAttrs()) {
      std::string attrName = namedAttr.getName().getValue().str();
      if (std::find(excludeStrings.begin(), excludeStrings.end(), attrName) ==
          excludeStrings.end())
        attributeNames.push_back(attrName);
    }
    rewriter.create<KrnlCallOp>(loc, customOp.getFunctionName().str(),
        outputAllocs, op, operands, attributeNames);

    if (op->getNumResults() > 0)
      rewriter.replaceOp(op, outputAllocs);
    else
      rewriter.eraseOp(op);
    return success();
  }
};

void populateLoweringONNXCustomOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXCustomOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
