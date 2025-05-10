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
      // Manually infer output shape for FusedGemm without shape helper.
      Type ty = op->getResultTypes()[0];
      auto outTensorTy = mlir::dyn_cast<mlir::RankedTensorType>(ty);
      if (!outTensorTy)
        return rewriter.notifyMatchFailure(op, "FusedGemm result must be a ranked tensor");
      int rank = outTensorTy.getRank();
      // Build IndexExpr dims.
      SmallVector<IndexExpr, 4> outputDims;
      Value a = operands[0];
      for (int i = 0; i < rank; ++i) {
        int64_t dim = outTensorTy.getDimSize(i);
        if (mlir::ShapedType::isDynamic(dim))
          outputDims.emplace_back(create.krnlIE.getShapeAsDim(a, i));
        else
          outputDims.emplace_back(LiteralIndexExpr(dim));
      }
      // Allocate output memref.
      auto memRefType = mlir::cast<mlir::MemRefType>(typeConverter->convertType(ty));
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
