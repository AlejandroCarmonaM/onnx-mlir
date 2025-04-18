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
#include "mlir/IR/Attributes.h" // Required for StringAttr
#include "mlir/IR/Builders.h"   // Required for StringAttr::get
#include "llvm/Support/Debug.h"
// ... include headers ...
#include "mlir/IR/Diagnostics.h" // Might be needed, often included transitively
// include cassert
#include <cassert> // For assert

#include "mlir/IR/TypeUtilities.h" // For getElementTypeOrSelf
#include "mlir/Transforms/DialectConversion.h" // Required for TypeConverter

using namespace mlir;

namespace onnx_mlir {

  struct ONNXFusedGemmOpLowering : public OpConversionPattern<ONNXCustomOp> {
    ONNXFusedGemmOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern(typeConverter, ctx, /*benefit=*/2) {}

    // We match only the custom op with function_name = "FusedGemm"
    LogicalResult match(ONNXCustomOp customOp) const override {
      // Get attributes directly
      StringAttr funcNameAttr = customOp->getAttrOfType<StringAttr>("function_name");
      if (!funcNameAttr || funcNameAttr.getValue() != "FusedGemm")
        return failure();

      StringAttr domainNameAttr = customOp->getAttrOfType<StringAttr>("domain_name");
      if (!domainNameAttr || domainNameAttr.getValue() != "com.microsoft")
        return failure();

      // Successfully matched a FusedGemm custom op
      return success();
    }

    LogicalResult matchAndRewrite(ONNXCustomOp customOp,
        ONNXCustomOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const final {

      Operation *op = customOp.getOperation();
      Location loc = op->getLoc();
      ValueRange operands = adaptor.getOperands(); // Operands after type conversion

      // Helper builders
      MultiDialectBuilder<AffineBuilder, IndexExprBuilderForKrnl, KrnlBuilder,
          MemRefBuilder>
          create(rewriter, loc);
      IndexExprScope scope(create.krnlIE);

      // ----- Shape inference is now done in the ONNX dialect -----
      // We rely on the result type being already inferred.

      // Get the original result type (should be RankedTensorType after inference)
      Type originalResultType = op->getResult(0).getType();
      auto originalRankedType = mlir::dyn_cast<RankedTensorType>(originalResultType);
      if (!originalRankedType) {
          op->emitError("FusedGemm lowering expects the result type to be a RankedTensorType (shape inference failed?)");
          return failure();
      }

      // Convert the result type using the TypeConverter
      Type convertedResultType = getTypeConverter()->convertType(originalResultType);
      if (!convertedResultType) {
           op->emitError("Failed to convert FusedGemm result type");
           return failure();
      }
      auto outputMemRefType = mlir::dyn_cast<MemRefType>(convertedResultType);
      if (!outputMemRefType) {
          op->emitError("FusedGemm lowering expects the converted result type to be a MemRefType");
          return failure();
      }

      // Get allocation dimensions from the inferred output type AND input tensors
      SmallVector<IndexExpr, 2> outputDims;
      ArrayRef<int64_t> outputShape = outputMemRefType.getShape(); // Shape from inferred type
      Value A = op->getOperand(0); // Original input A
      Value B = op->getOperand(1); // Original input B
      // Use getValue().getSExtValue() instead of getInt()
      bool transA = customOp->getAttrOfType<IntegerAttr>("transA").getValue().getSExtValue() != 0;
      bool transB = customOp->getAttrOfType<IntegerAttr>("transB").getValue().getSExtValue() != 0;


      // Determine IndexExpr for dimension M (outputShape[0])
      if (outputShape[0] == ShapedType::kDynamic) {
          // M is dynamic, get it from input A's shape using getShapeAsDim
          uint64_t dim_idx_A_for_M = transA ? 1 : 0;
          outputDims.push_back(create.krnlIE.getShapeAsDim(A, dim_idx_A_for_M));
      } else {
          // M is static, get it from input A's shape using getShapeAsLiteral
          // (We could use outputShape[0] directly via IndexExpr(outputShape[0])
          // if that constructor worked, but getShapeAsLiteral is safer as it
          // reads from the actual input tensor's metadata).
          uint64_t dim_idx_A_for_M = transA ? 1 : 0;
          outputDims.push_back(create.krnlIE.getShapeAsLiteral(A, dim_idx_A_for_M));
      }

      // Determine IndexExpr for dimension N (outputShape[1])
      // N is asserted to be static in the shape inference phase.
      if (outputShape[1] == ShapedType::kDynamic) {
           // This case should ideally not happen based on shape inference logic
           op->emitError("Dynamic N dimension encountered unexpectedly during allocation dimension calculation.");
           return failure();
      } else {
          // N is static, get it from input B's shape using getShapeAsLiteral
          uint64_t dim_idx_B_for_N = transB ? 0 : 1;
          outputDims.push_back(create.krnlIE.getShapeAsLiteral(B, dim_idx_B_for_N));
      }


      // Check if dynamic dimension resolution failed (getShapeAsDim might return undefined)
      if (outputDims[0].isUndefined() || outputDims[1].isUndefined()) {
          op->emitError("Failed to determine allocation dimensions for FusedGemm output (dynamic dim resolution failed).");
          return failure();
      }

      // Allocate memory for the result
      Value alloc = create.mem.alignedAlloc(outputMemRefType, outputDims);
      SmallVector<Value, 1> outputAllocs = {alloc}; // Initialize with the allocated value


      // Pass relevant attributes to the krnl.call function
      std::vector<std::string> attributeNames;
      for (NamedAttribute namedAttr : customOp->getAttrs()) {
        std::string attrName = namedAttr.getName().getValue().str();
        // Skip attributes not meant for the runtime call or handled internally
        if (attrName == "function_name" || attrName == "domain_name" ||
            attrName == "shape_infer_pattern" || attrName == "inputs_for_infer" ||
            attrName == "output_element_type") // Also skip type/shape attrs
          continue;
        attributeNames.push_back(attrName);
      }

      // Add verification print
      mlir::emitRemark(loc, "ONNXFusedGemmOpLowering matched and rewriting op (using inferred shape)");

      // Create the krnl.call with the target function name
      // Use adaptor.getOperands() here as KrnlCallOp expects Values after type conversion
      rewriter.create<KrnlCallOp>(loc, "ort_cpu_ep_fused_gemm", outputAllocs, op, adaptor.getOperands(),
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