/**********************************************
 * IMPORT LIBRARIES
 **********************************************/

/*
Libraries and tools used in this script, along with version info when applicable.
*/
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Conversion/ONNXToKrnl/Additional/FusedGemm.hpp"

using namespace mlir;

namespace onnx_mlir {

/**********************************************
 * FUNCTION DEFINITIONS
 **********************************************/

/*
 * Purpose: Lower the ONNXCustomOp "FusedGemm" to a KrnlCallOp that calls
 *          the C++ runtime function ort_cpu_ep_fused_gemm, passing
 *          raw pointers for tensors and int64_t for scalars.
 */
struct ONNXFusedGemmOpLowering : public OpConversionPattern<ONNXCustomOp> {
  ONNXFusedGemmOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ONNXCustomOp customOp,
      ONNXCustomOpAdaptor operandAdaptor,
      ConversionPatternRewriter &rewriter) const final {

    /******************************************
     * CHECK OP TYPE
     ******************************************/
    StringAttr funcNameAttr = customOp.getFunctionNameAttr();
    if (!funcNameAttr || funcNameAttr.getValue() != "FusedGemm")
      return failure();

    Operation *op = customOp.getOperation();
    Location loc = op->getLoc();
    ValueRange operands = operandAdaptor.getOperands();

    /******************************************
     * HELPERS AND SHAPE
     ******************************************/
    MultiDialectBuilder<AffineBuilder, IndexExprBuilderForKrnl, KrnlBuilder, MemRefBuilder> create(rewriter, loc);
    IndexExprScope scope(create.krnlIE);

    ONNXCustomOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    /******************************************
     * OUTPUT ALLOCATION
     ******************************************/
    Type outputType = op->getResultTypes()[0];
    MemRefType outputMemRefType = mlir::cast<MemRefType>(typeConverter->convertType(outputType));

    /******************************************
     * EXTRACT INPUTS AND ATTRIBUTES
     ******************************************/
    MemRefBuilder memrefBuilder(rewriter, loc);
    IntegerAttr transAAttr = customOp->getAttrOfType<IntegerAttr>("transA");
    IntegerAttr transBAttr = customOp->getAttrOfType<IntegerAttr>("transB");
    int64_t transA = transAAttr ? transAAttr.getValue().getSExtValue() : 0;
    int64_t transB = transBAttr ? transBAttr.getValue().getSExtValue() : 0;
    Value A = operands[0];
    Value B = operands[1];

    /******************************************
     * COMPUTE DIMS (index type)
     ******************************************/
    Value Midx, Nidx, Kidx;
    if (transA == 0) {
      Midx = memrefBuilder.dim(A, 0);
      Kidx = memrefBuilder.dim(A, 1);
    } else {
      Kidx = memrefBuilder.dim(A, 0);
      Midx = memrefBuilder.dim(A, 1);
    }
    if (transB == 0) {
      Nidx = memrefBuilder.dim(B, 1);
    } else {
      Nidx = memrefBuilder.dim(B, 0);
    }

    /******************************************
     * ALLOCATE OUTPUT BUFFER
     ******************************************/
    Value outputAlloc;
    if (outputMemRefType.getNumDynamicDims() == 2) {
      outputAlloc = create.mem.alignedAlloc(outputMemRefType, {Midx, Nidx});
    } else if (outputMemRefType.getNumDynamicDims() == 1) {
      if (outputMemRefType.isDynamicDim(0))
        outputAlloc = create.mem.alignedAlloc(outputMemRefType, {Midx});
      else
        outputAlloc = create.mem.alignedAlloc(outputMemRefType, {Nidx});
    } else {
      outputAlloc = create.mem.alignedAlloc(outputMemRefType);
    }

    /******************************************
     * PREPARE CALL OPERANDS
     ******************************************/
    Value BiasOperand = (operands.size() > 2) ? operands[2] : Value();
    Value Y = outputAlloc;
    Value transAVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(transA));
    Value transBVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(transB));

        SmallVector<Value, 9> callOperands{A, B, BiasOperand, Y, Midx, Nidx, Kidx, transAVal, transBVal};

    /******************************************
     * LOWER TO KrnlCallOp
     ******************************************/
    std::vector<std::string> attributeNames; // Add attribute names if you want to copy any

    rewriter.create<KrnlCallOp>(
        loc,
        "ort_cpu_ep_fused_gemm", // function name as string
        SmallVector<Value, 0>{}, // outputs
        op,                       // original op (for attribute copying)
        callOperands,             // inputs
        attributeNames            // attributes to copy
    );

    rewriter.replaceOp(op, Y);
    return success();
  }
};

/**********************************************
 * PATTERN REGISTRATION
 **********************************************/

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