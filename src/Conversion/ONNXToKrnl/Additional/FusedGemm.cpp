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

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXFusedGemmOpLowering : public OpConversionPattern<ONNXCustomOp> {
  ONNXFusedGemmOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/2) {}

  // We match only the custom op with function_name = "FusedGemm"
  LogicalResult match(ONNXCustomOp customOp) const override {
    StringAttr funcName = customOp.getFunctionName();
    if (!funcName || funcName.getValue() != "FusedGemm")
      return failure();

    StringAttr domainName = customOp.getDomainName();
    if (!domainName || domainName.getValue() != "com.microsoft")
      return failure();

    // Successfully matched a FusedGemm custom op
    return success();
  }

  LogicalResult matchAndRewrite(ONNXCustomOp customOp,
      ONNXCustomOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = customOp.getOperation();
    Location loc = op->getLoc();
    ValueRange operands = adaptor.getOperands();

    // Helper builders
    MultiDialectBuilder<AffineBuilder, IndexExprBuilderForKrnl, KrnlBuilder,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnlIE);

    // Get shape
    ONNXCustomOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Prepare outputs for krnl.call
    SmallVector<Type, 4> outputMemRefTypes;
    SmallVector<Value, 4> outputAllocs;
    
    for (size_t idx = 0; idx < op->getResultTypes().size(); idx++) {
      Type ty = op->getResultTypes()[idx];
      MemRefType outputMemRefType =
          mlir::cast<MemRefType>(typeConverter->convertType(ty));
      outputMemRefTypes.emplace_back(outputMemRefType);
      Value alloc = create.mem.alignedAlloc(
          outputMemRefType, shapeHelper.getOutputDims(idx));
      outputAllocs.emplace_back(alloc);
    }

    // Extract attributes needed for the FusedGemm operation
    StringAttr activationAttr = nullptr;
    FloatAttr alphaAttr = nullptr;
    FloatAttr betaAttr = nullptr;
    IntegerAttr transAAttr = nullptr;
    IntegerAttr transBAttr = nullptr;

    // Read attributes but don't use them in krnl.call
    if (auto attr = customOp->getAttrOfType<StringAttr>("activation"))
      activationAttr = attr;
    if (auto attr = customOp->getAttrOfType<FloatAttr>("alpha"))
      alphaAttr = attr;
    if (auto attr = customOp->getAttrOfType<FloatAttr>("beta"))
      betaAttr = attr;
    if (auto attr = customOp->getAttrOfType<IntegerAttr>("transA"))
      transAAttr = attr;
    if (auto attr = customOp->getAttrOfType<IntegerAttr>("transB"))
      transBAttr = attr;

    // Pass all attributes to the krnl.call function
    std::vector<std::string> attributeNames;
    for (NamedAttribute namedAttr : customOp->getAttrs()) {
      std::string attrName = namedAttr.getName().getValue().str();
      // Skip shape inference related attributes
      if (attrName == "function_name" || attrName == "shape_infer_pattern" ||
          attrName == "inputs_for_infer" || attrName == "output_element_type")
        continue;
      attributeNames.push_back(attrName);
    }

    // Add a verification print to show the lowering is working
    rewriter.create<KrnlPrintfOp>(loc, 
        "=== FusedGemm operation successfully lowered: function_name=%s, domain_name=%s ===\n", 
        customOp.getFunctionName(), customOp.getDomainName());

    // Create the krnl.call with the "FusedGemm" function name
    // This is a placeholder for the actual function call that will be implemented later
    rewriter.create<KrnlCallOp>(loc, "ort_cpu_ep_fused_gemm", outputAllocs, op, operands,
        attributeNames);

    rewriter.replaceOp(op, outputAllocs);
    return success();
  }
};

void populateLoweringONNXFusedGemmOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXFusedGemmOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir