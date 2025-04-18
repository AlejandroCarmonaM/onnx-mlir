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
      ValueRange operands = adaptor.getOperands(); // Operands after type conversion (if any)
  
      // Helper builders
      MultiDialectBuilder<AffineBuilder, IndexExprBuilderForKrnl, KrnlBuilder,
          MemRefBuilder>
          create(rewriter, loc);
      IndexExprScope scope(create.krnlIE);
  
      // Get shape helper (still useful for input dims)
      // Use original operands (op->getOperands()) for shape helper as adaptor.getOperands()
      // might have types converted already, potentially losing shape info if conversion failed.
      ONNXCustomOpShapeHelper shapeHelper(op, op->getOperands(), &create.krnlIE);
      shapeHelper.computeShapeAndAssertOnFailure(); // Computes input shapes
  
      // ***** START MANUAL OUTPUT SHAPE CALCULATION *****
      // Get input values (use adaptor for potentially converted types if needed by KrnlCallOp later,
      // but use original op operands for shape calculation)
      Value A = op->getOperand(0); // Original Input tensor A
      Value B = op->getOperand(1); // Original Input tensor B (weights)
      // Value C = op->getOperand(2); // Original Input tensor C (bias) - needed for bias shape if applicable
  
      // Assume shape inference has run and inputs are ranked (this might fail if inputs are also unranked)
      auto aType = mlir::dyn_cast<RankedTensorType>(A.getType());
      auto bType = mlir::dyn_cast<RankedTensorType>(B.getType());
  
      // Check if input types are ranked, otherwise we cannot proceed
      if (!aType || !bType) {
           op->emitError("FusedGemm lowering requires ranked input tensors A and B.");
           return failure();
      }
  
  
      // Get Gemm attributes
      bool transA = customOp->getAttrOfType<IntegerAttr>("transA").getInt() != 0;
      bool transB = customOp->getAttrOfType<IntegerAttr>("transB").getInt() != 0;
  
      // Get dimensions (handle transpose)
      int64_t M = transA ? aType.getShape()[1] : aType.getShape()[0];
      int64_t K_A = transA ? aType.getShape()[0] : aType.getShape()[1]; // K from A
      int64_t K_B = transB ? bType.getShape()[1] : bType.getShape()[0]; // K from B
      int64_t N = transB ? bType.getShape()[0] : bType.getShape()[1];
  
      // Basic validation
      // Allow K dimensions to be dynamic but assert they are equal if both known
      if (K_A != ShapedType::kDynamic && K_B != ShapedType::kDynamic) {
          assert(K_A == K_B && "K dimensions of A and B for Gemm must match");
      }
      assert(aType.getRank() == 2 && "Input A must be 2D for this Gemm logic");
      assert(bType.getRank() == 2 && "Input B must be 2D for this Gemm logic");
  
      // Determine output element type (use original result type's element type)
      // Use the type from the original operation before potential type conversion
      Type outputElementType = mlir::getElementTypeOrSelf(op->getResult(0).getType());
  
      // Define the output shape: [M, N]
      // Handle dynamic M dimension if input A's first dim is dynamic
      SmallVector<int64_t, 2> outputShapeVec;
      if (M == ShapedType::kDynamic) {
           outputShapeVec.push_back(ShapedType::kDynamic); // Preserve dynamic batch dim
      } else {
           outputShapeVec.push_back(M);
      }
      // N dimension should be static based on weights B
      assert(N != ShapedType::kDynamic && "N dimension (from B) must be static");
      outputShapeVec.push_back(N);
      ArrayRef<int64_t> outputShape(outputShapeVec);
  
      // Create the *ranked* MemRefType for the output
      // Use the standard memory space (0)
      MemRefType outputMemRefType = MemRefType::get(outputShape, outputElementType);
      // ***** END MANUAL OUTPUT SHAPE CALCULATION *****
  
  
      SmallVector<Value, 1> outputAllocs; // Expecting only one output for FusedGemm

      // Use the manually computed MemRefType and its shape for allocation
      // Get dims for allocation (handle dynamic dim if necessary)
      SmallVector<IndexExpr, 2> outputDims;
      if (outputShape[0] == ShapedType::kDynamic) {
          // Need to get the dynamic dimension value from input A
          // Use getShapeAsDim to handle dynamic case, creating a DimIE if possible.
          outputDims.push_back(create.krnlIE.getShapeAsDim(A, 0));
      } else {
          // Dimension is static, use getShapeAsLiteral.
          // We need the original dimension index from A (0 if !transA, 1 if transA)
          uint64_t dim_idx_A = transA ? 1 : 0;
          outputDims.push_back(create.krnlIE.getShapeAsLiteral(A, dim_idx_A));
      }
      // Dimension N (outputShape[1]) is asserted to be static earlier. Use getShapeAsLiteral.
      // We need the original dimension index from B (1 if !transB, 0 if transB)
      uint64_t dim_idx_B = transB ? 0 : 1;
      outputDims.push_back(create.krnlIE.getShapeAsLiteral(B, dim_idx_B));

      // Check if getShapeAsDim returned undefined IndexExpr (QuestionMark)
      // This might happen if the dynamic dimension value couldn't be obtained.
      // getShapeAsLiteral asserts internally if the shape is not literal.
      if (outputDims[0].isUndefined() || outputDims[1].isUndefined()) {
          op->emitError("Failed to determine allocation dimensions for FusedGemm output.");
          return failure();
      }

      Value alloc = create.mem.alignedAlloc(outputMemRefType, outputDims);
      outputAllocs.emplace_back(alloc);
  
  
      // Pass relevant attributes to the krnl.call function
      std::vector<std::string> attributeNames;
      for (NamedAttribute namedAttr : customOp->getAttrs()) {
        std::string attrName = namedAttr.getName().getValue().str();
        // Skip attributes not meant for the runtime call
        if (attrName == "function_name" || attrName == "domain_name" ||
            attrName == "shape_infer_pattern" || attrName == "inputs_for_infer" ||
            attrName == "output_element_type")
          continue;
        attributeNames.push_back(attrName);
      }
  
      // Add verification print
      mlir::emitRemark(loc, "ONNXFusedGemmOpLowering matched and rewriting op (manual shape)");
  
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