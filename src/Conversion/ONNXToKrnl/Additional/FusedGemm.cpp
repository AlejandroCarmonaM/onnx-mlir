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

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "mlir/IR/Attributes.h" // Required for StringAttr, IntegerAttr
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Diagnostics.h" // Might be needed, often included transitively
#include <cassert> // For assert

#include "mlir/IR/TypeUtilities.h" // For getElementTypeOrSelf
#include "mlir/Transforms/DialectConversion.h" // Required for TypeConverter, OpConversionPattern, ConversionPatternRewriter

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h" // Required for matchPattern

#include "src/Dialect/ONNX/ONNXOps.hpp" // Include ONNX dialect ops definition
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "onnx_to_krnl_fusedgemm"

using namespace mlir;

namespace onnx_mlir {

// Helper function to get or insert the LLVM declaration for the external function
static FlatSymbolRefAttr getOrInsertExternFunc(StringRef funcName,
    ModuleOp module, mlir::Type funcType, PatternRewriter &rewriter) {
  auto *context = module.getContext();
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return SymbolRefAttr::get(context, funcName);

  // Insert the function declaration
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcType);
  return SymbolRefAttr::get(context, funcName);
}

// Lowering pattern for onnx.Custom operations identified as "FusedGemm"
struct ONNXFusedGemmOpLowering : public OpConversionPattern<ONNXCustomOp> {
  // Assign a high benefit to prioritize this pattern over generic custom op lowering
  ONNXFusedGemmOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXCustomOp customOp,
      ONNXCustomOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {

    // Check if this custom op is the one we want to handle ("FusedGemm")
    StringAttr funcNameAttr = customOp.getFunctionNameAttr();
    if (!funcNameAttr || funcNameAttr.getValue() != "FusedGemm") {
      return failure(); // Not the FusedGemm custom op, let other patterns handle it
    }

    LLVM_DEBUG(llvm::dbgs() << "ONNXFusedGemmOpLowering matched and rewriting op "
                           << customOp->getName() << "\n");

    Operation *op = customOp.getOperation();
    Location loc = op->getLoc();

    // Get operands using the adaptor (already converted by the framework)
    Value A = adaptor.getOperands()[0];
    Value B = adaptor.getOperands()[1];
    // Bias is the third operand, if present
    Value BiasOperand = (adaptor.getOperands().size() > 2) ? adaptor.getOperands()[2] : nullptr;

    // Get attributes directly from the customOp
    IntegerAttr transAAttr = customOp->getAttrOfType<IntegerAttr>("transA");
    IntegerAttr transBAttr = customOp->getAttrOfType<IntegerAttr>("transB");
    // StringAttr activationAttr = customOp->getAttrOfType<StringAttr>("activation"); // Assuming ReLU for now

    if (!transAAttr || !transBAttr) {
        customOp->emitError("Missing 'transA' or 'transB' attribute for FusedGemm");
        return failure();
    }
    int64_t transA = transAAttr.getInt();
    int64_t transB = transBAttr.getInt();

    // Operands A and B must have been converted to MemRefType by the framework
    auto aMemRefType = A.getType().dyn_cast<MemRefType>();
    auto bMemRefType = B.getType().dyn_cast<MemRefType>();
    if (!aMemRefType || !bMemRefType) {
       customOp->emitError("Operands A and B must be MemRefType after conversion");
       return failure();
    }

    // Get M, N, K dimensions from the *memref* types using MemRefBuilder helper
    MemRefBuilder createMemRef(rewriter, loc);
    Value M, N, K;
    if (transA == 0) { // A is M x K
      M = createMemRef.dim(A, 0);
      K = createMemRef.dim(A, 1);
    } else { // A is K x M
      K = createMemRef.dim(A, 0);
      M = createMemRef.dim(A, 1);
    }

    if (transB == 0) { // B is K x N
      // TODO: Add verification that K matches dim 0 of B if both are static
      N = createMemRef.dim(B, 1);
    } else { // B is N x K
      N = createMemRef.dim(B, 0);
      // TODO: Add verification that K matches dim 1 of B if both are static
    }

    // Allocate the output buffer Y
    Value Y;
    // Get the original output type and convert it using the TypeConverter
    auto outputType = customOp.getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!outputType) {
        customOp->emitError("Output must be a ranked tensor type");
        return failure();
    }
    auto outputMemRefType = typeConverter->convertType(outputType).dyn_cast<MemRefType>();
     if (!outputMemRefType) {
        customOp->emitError("Failed to convert output tensor type to memref type");
        return failure();
    }

    // Allocate memory for the output tensor
    if (hasAllConstantDimensions(outputMemRefType)) {
      Y = createMemRef.alloc(outputMemRefType);
    } else {
      // Pass dynamic dimensions M, N needed for allocation
      Y = createMemRef.alloc(outputMemRefType, {M, N});
    }

    // Prepare arguments for the external C++ function call
    // Signature: void ort_cpu_ep_fused_gemm(float* A, float* B, float* Bias, float* Y,
    //                                       int64_t M, int64_t N, int64_t K, int64_t transA, int64_t transB)
    auto int64Type = rewriter.getI64Type();
    // Assuming float type based on runtime function signature
    auto floatPtrType = LLVM::LLVMPointerType::get(rewriter.getF32Type());

    // Convert transA and transB attributes to LLVM constants
    Value llvmTransA = rewriter.create<LLVM::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(transA));
    Value llvmTransB = rewriter.create<LLVM::ConstantOp>(loc, int64Type, rewriter.getI64IntegerAttr(transB));

    // Get LLVM pointers to the data buffers of the MemRefs
    Value ptrA = createMemRef.data(A);
    Value ptrB = createMemRef.data(B);
    Value ptrY = createMemRef.data(Y);

    // Handle optional Bias: get pointer if BiasOperand is a valid MemRef, otherwise pass null
    Value ptrBias;
    if (BiasOperand && !mlir::isa<NoneType>(BiasOperand.getType())) {
         auto biasMemRefType = BiasOperand.getType().dyn_cast<MemRefType>();
         if (!biasMemRefType) {
             customOp->emitError("Bias operand must be MemRefType or None after conversion");
             return failure();
         }
         ptrBias = createMemRef.data(BiasOperand);
    } else {
         ptrBias = rewriter.create<LLVM::NullOp>(loc, floatPtrType);
    }


    // Define the LLVM function type for the external C++ function
    auto llvmF32Type = rewriter.getF32Type(); // Assuming float
    auto llvmI64Type = rewriter.getI64Type();
    auto llvmVoidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto funcType = LLVM::LLVMFunctionType::get(llvmVoidType,
        {floatPtrType, floatPtrType, floatPtrType, floatPtrType, // A*, B*, Bias*, Y*
         llvmI64Type, llvmI64Type, llvmI64Type,                // M, N, K
         llvmI64Type, llvmI64Type},                            // transA, transB
        false); // isVarArg

    // Get or insert the external function declaration into the module
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto externFuncRef = getOrInsertExternFunc(
        "ort_cpu_ep_fused_gemm", parentModule, funcType, rewriter);

    // Create the LLVM function call op
    rewriter.create<LLVM::CallOp>(loc, llvmVoidType, externFuncRef,
        ValueRange{ptrA, ptrB, ptrBias, ptrY, M, N, K, llvmTransA, llvmTransB});

    // Replace the original onnx.Custom op with the allocated output buffer Y
    rewriter.replaceOp(op, Y);
    return success();
  }
};

// Function to register the lowering pattern
void populateONNXToKrnlConversionAdditionalPass(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXFusedGemmOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir