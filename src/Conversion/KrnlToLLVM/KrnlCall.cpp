/**********************************************
 * IMPORT LIBRARIES
 **********************************************/

/*
Libraries and tools used in this script, along with version info when applicable.
*/
#include "llvm/ADT/TypeSwitch.h"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

/**********************************************
 * FUNCTION DEFINITIONS
 **********************************************/

/*
 * Purpose: Helper to check if the current KrnlCallOp is calling the custom
 *          ort_cpu_ep_fused_gemm function.
 * Parameters:
 *    - op (Operation*): The KrnlCallOp being lowered.
 * Returns:
 *    - bool: True if this is a call to ort_cpu_ep_fused_gemm, false otherwise.
 */
bool isFusedGemmCall(Operation *op) {
    if (auto callOp = llvm::dyn_cast<KrnlCallOp>(op)) {
        auto funcNameAttr = callOp.getFuncNameAttr();
        if (funcNameAttr && funcNameAttr.getValue() == "ort_cpu_ep_fused_gemm")
            return true;
    }
    return false;
}

/**********************************************
 * FUSEDGEMM SPECIALIZED LOWERING
 **********************************************/

/*
 * Purpose: Lower KrnlCallOp for ort_cpu_ep_fused_gemm.
 *          Pass OMTensor* wrappers for all tensors, and scalars as int64_t.
 * Parameters:
 *    - op (Operation*): The KrnlCallOp being lowered.
 *    - operands (ArrayRef<Value>): The operands to the call.
 *    - rewriter (ConversionPatternRewriter&): The MLIR rewriter.
 *    - llvmTypeConverter (LLVMTypeConverter*): The LLVM type converter.
 * Returns:
 *    - LogicalResult: Success or failure.
 */
LogicalResult lowerFusedGemmKrnlCallOp(Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter, LLVMTypeConverter *llvmTypeConverter) {

    /******************************************
     * INITIALIZE DATA
     ******************************************/
    Location loc = op->getLoc();
    KrnlCallOp krnlCallOp = llvm::cast<KrnlCallOp>(op);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    const auto &apiRegistry =
        RuntimeAPIRegistry(module, rewriter, *llvmTypeConverter);

    llvm::SmallVector<Type, 4> parameterTypeList;
    llvm::SmallVector<Value, 4> parameterList;
    llvm::SmallVector<Value, 4> omTensors;

    llvm::outs() << "[FusedGemm] Lowering KrnlCallOp for ort_cpu_ep_fused_gemm (OMTensor wrapper mode)\n";

    auto origParams = krnlCallOp.getParameters();
    auto numParams = origParams.size();

    // Defensive: Expecting 9 parameters (A, B, Bias, Y, M, N, K, transA, transB)
    if (numParams != 9 || operands.size() != 9) {
        llvm::outs() << "[FusedGemm] ERROR: Expected 9 parameters, got " << numParams << " and " << operands.size() << "\n";
        return failure();
    }

    /******************************************
     * RETRIEVE AND WRAP TENSOR ARGUMENTS AS OMTensor*
     ******************************************/
    llvm::SmallVector<std::string, 4> tensorNames = {"A", "B", "Bias", "Y"};
    for (int i = 0; i < 4; ++i) {
        Value origVal = origParams[i];
        Value convVal = operands[i];
        Type ty = origVal.getType();
        std::string name = tensorNames[i];

        if (auto memRefTy = mlir::dyn_cast<MemRefType>(ty)) {
            // Wrap MemRef as OMTensor*
            auto int64Ty = rewriter.getI64Type();
            auto memRefRank = memRefTy.getRank();
            auto memRefRankVal = create.llvm.constant(int64Ty, static_cast<int64_t>(memRefRank));
            Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
                RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});
            Type llvmElemTy = llvmTypeConverter->convertType(memRefTy.getElementType());
            krnl::fillOMTensorWithMemRef(convVal, llvmElemTy, omTensor,
                false /*outOwning*/, rewriter, loc, apiRegistry, module);
            auto int8Ty = IntegerType::get(op->getContext(), 8);
            auto opaquePtrTy = getPointerType(op->getContext(), int8Ty);
            parameterTypeList.emplace_back(opaquePtrTy);
            parameterList.emplace_back(omTensor);
            omTensors.emplace_back(omTensor);
            llvm::outs() << "[FusedGemm] " << name << " wrapped as OMTensor*\n";
        } else if (mlir::isa<NoneType>(ty)) {
            auto int8Ty = IntegerType::get(op->getContext(), 8);
            auto opaquePtrTy = getPointerType(op->getContext(), int8Ty);
            Value nullPtr = create.llvm.null(opaquePtrTy);
            parameterTypeList.emplace_back(opaquePtrTy);
            parameterList.emplace_back(nullPtr);
            llvm::outs() << "[FusedGemm] " << name << " is NoneType, passing nullptr\n";
        } else {
            llvm::outs() << "[FusedGemm] Unexpected type for tensor arg '" << name << "'\n";
            return failure();
        }
    }

    /******************************************
     * RETRIEVE AND CONVERT SCALAR ARGUMENTS
     ******************************************/
    for (int i = 4; i < 9; ++i) {
        Value origVal = origParams[i];
        Value convVal = operands[i];
        Type ty = origVal.getType();

        if (mlir::isa<IndexType>(ty)) {
            auto int64Ty = rewriter.getI64Type();
            Value casted = rewriter.create<arith::IndexCastOp>(loc, int64Ty, convVal);
            parameterTypeList.emplace_back(int64Ty);
            parameterList.emplace_back(casted);
            llvm::outs() << "[FusedGemm] Scalar (IndexType) cast to int64.\n";
        } else if (mlir::isa<IntegerType>(ty)) {
            Type llvmTy = llvmTypeConverter->convertType(ty);
            parameterTypeList.emplace_back(llvmTy);
            parameterList.emplace_back(convVal);
            llvm::outs() << "[FusedGemm] Scalar (IntegerType) passed directly.\n";
        } else {
            llvm::outs() << "[FusedGemm] Unexpected type for scalar arg\n";
            return failure();
        }
    }

    /******************************************
     * PERFORM CALL
     ******************************************/
    ValueRange returns = op->getResults();
    if (returns.size() == 0) {
        FlatSymbolRefAttr callRef =
            create.llvm.getOrInsertSymbolRef(module, krnlCallOp.getFuncName(),
                LLVM::LLVMVoidType::get(module.getContext()), parameterTypeList);
        create.llvm.call({}, callRef, parameterList);
        rewriter.eraseOp(op);
        llvm::outs() << "[FusedGemm] Call to ort_cpu_ep_fused_gemm emitted (void)\n";
    } else {
        assert(returns.size() == 1 &&
               "Only one return value is allowed for krnl.call now");
        Type llvmReturnType =
            llvmTypeConverter->convertType(returns[0].getType());

        FlatSymbolRefAttr callRef = create.llvm.getOrInsertSymbolRef(
            module, krnlCallOp.getFuncName(), llvmReturnType, parameterTypeList);
        auto llvmCall =
            create.llvm.call({llvmReturnType}, callRef, parameterList);
        rewriter.replaceOp(op, llvmCall.getDefiningOp()->getResults()[0]);
        llvm::outs() << "[FusedGemm] Call to ort_cpu_ep_fused_gemm emitted (with return)\n";
    }

    /******************************************
     * CLEANUP: Destroy OMTensor wrappers
     ******************************************/
    for (Value omt : omTensors) {
        RuntimeAPI::callApi(
            rewriter, loc, apiRegistry, RuntimeAPI::API::DESTROY_OMTENSOR, {omt});
    }

    return success();
}

/**********************************************
 * DEFAULT KrnlCallOp LOWERING
 **********************************************/

class KrnlCallOpLowering : public ConversionPattern {
public:
    explicit KrnlCallOpLowering(
        LLVMTypeConverter &typeConverter, MLIRContext *context)
        : ConversionPattern(
            typeConverter, KrnlCallOp::getOperationName(), 1, context) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
        ConversionPatternRewriter &rewriter) const override {

        /******************************************
         * SPECIAL CASE: FusedGemm
         ******************************************/
        if (isFusedGemmCall(op)) {
            llvm::outs() << "[KrnlCall] Detected FusedGemm KrnlCallOp, using specialized lowering\n";
            return lowerFusedGemmKrnlCallOp(op, operands, rewriter, (LLVMTypeConverter*)getTypeConverter());
        }

        /******************************************
         * DEFAULT KrnlCallOp LOWERING (unchanged)
         ******************************************/
        KrnlCallOpAdaptor krnlCallAdaptor(operands);
        Location loc = op->getLoc();
        KrnlCallOp krnlCallOp = llvm::cast<KrnlCallOp>(op);
        MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
        const LLVMTypeConverter *llvmTypeConverter =
            static_cast<const LLVMTypeConverter *>(getTypeConverter());

        ModuleOp module = op->getParentOfType<ModuleOp>();
        llvm::SmallVector<Type, 4> parameterTypeList;
        llvm::SmallVector<Value, 4> parameterList;
        llvm::SmallVector<Value, 4> omTensors;

        bool isFusedGemm = isFusedGemmCall(op);

        auto itConverted = krnlCallAdaptor.getParameters().begin();
        auto itOriginal = krnlCallOp.getParameters().begin();
        for (; itConverted != krnlCallAdaptor.getParameters().end();
             itConverted++, itOriginal++) {
            handleOneParameter(rewriter, op, *itConverted, *itOriginal,
                parameterTypeList, parameterList, omTensors, isFusedGemm);
        }

        // Handle the Attributes
        for (auto namedAttr : op->getAttrs()) {
            if (namedAttr.getName().getValue() == "funcName")
                continue;
            if (namedAttr.getName().getValue() == "numOfOutput")
                continue;
            handleOneAttribute(
                rewriter, op, namedAttr.getValue(), parameterTypeList, parameterList);
        }

        ValueRange returns = op->getResults();
        if (returns.size() == 0) {
            FlatSymbolRefAttr callRef =
                create.llvm.getOrInsertSymbolRef(module, krnlCallOp.getFuncName(),
                    LLVM::LLVMVoidType::get(module.getContext()), parameterTypeList);
            create.llvm.call({}, callRef, parameterList);
            rewriter.eraseOp(op);
        } else {
            assert(returns.size() == 1 &&
                   "Only one return value is allowed for krnl.call now");
            Type llvmReturnType =
                llvmTypeConverter->convertType(returns[0].getType());

            FlatSymbolRefAttr callRef = create.llvm.getOrInsertSymbolRef(
                module, krnlCallOp.getFuncName(), llvmReturnType, parameterTypeList);
            auto llvmCall =
                create.llvm.call({llvmReturnType}, callRef, parameterList);
            rewriter.replaceOp(op, llvmCall.getDefiningOp()->getResults()[0]);
        }

        // Destroy OMTensor wrappers of parameters (not used for fused gemm).
        if (!isFusedGemm) {
            const auto &apiRegistry =
                RuntimeAPIRegistry(module, rewriter, *llvmTypeConverter);
            for (Value omt : omTensors) {
                RuntimeAPI::callApi(
                    rewriter, loc, apiRegistry, RuntimeAPI::API::DESTROY_OMTENSOR, {omt});
            }
        }

        return success();
    }

private:
    /*
     * Purpose: Handle one parameter for KrnlCallOp lowering.
     *          For ort_cpu_ep_fused_gemm, pass OMTensor wrappers for tensors,
     *          and pass scalars directly (cast index to i64).
     */
    void handleOneParameter(PatternRewriter &rewriter, Operation *op,
        Value parameter, Value original,
        llvm::SmallVector<Type, 4> &parameterTypeList,
        llvm::SmallVector<Value, 4> &parameterList,
        llvm::SmallVector<Value, 4> &omTensors,
        bool isFusedGemm) const {
        MLIRContext *context = op->getContext();
        Location loc = op->getLoc();
        ModuleOp module = op->getParentOfType<ModuleOp>();
        MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
        const auto *llvmTypeConverter =
            static_cast<const LLVMTypeConverter *>(getTypeConverter());
        const auto &apiRegistry =
            RuntimeAPIRegistry(module, rewriter, *llvmTypeConverter);

        Type ty = original.getType();

        // Special-case for ort_cpu_ep_fused_gemm
        if (isFusedGemm) {
            if (auto memRefTy = mlir::dyn_cast<MemRefType>(ty)) {
                // TENSOR: Wrap as OMTensor*
                auto int64Ty = IntegerType::get(context, 64);
                auto memRefRank = memRefTy.getRank();
                auto memRefRankVal = create.llvm.constant(int64Ty, static_cast<int64_t>(memRefRank));
                Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
                    RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});
                Type llvmElemTy = llvmTypeConverter->convertType(memRefTy.getElementType());
                krnl::fillOMTensorWithMemRef(parameter, llvmElemTy, omTensor,
                    false /*outOwning*/, rewriter, loc, apiRegistry, module);
                auto int8Ty = IntegerType::get(context, 8);
                auto opaquePtrTy = getPointerType(context, int8Ty);
                parameterTypeList.emplace_back(opaquePtrTy);
                parameterList.emplace_back(omTensor);
                omTensors.emplace_back(omTensor);
                return;
            } else if (ty.isa<IndexType>()) {
                // SCALAR: index type, cast to i64
                auto int64Ty = rewriter.getI64Type();
                Value casted = rewriter.create<arith::IndexCastOp>(loc, int64Ty, parameter);
                parameterTypeList.emplace_back(int64Ty);
                parameterList.emplace_back(casted);
                return;
            } else if (ty.isa<IntegerType>() || ty.isa<FloatType>()) {
                // SCALAR: Pass directly (int64_t, float, etc.)
                Type llvmTy = llvmTypeConverter->convertType(ty);
                parameterTypeList.emplace_back(llvmTy);
                parameterList.emplace_back(parameter);
                return;
            } else if (mlir::isa<NoneType>(ty)) {
                // Pass null pointer for NoneType
                auto int8Ty = IntegerType::get(context, 8);
                auto opaquePtrTy = getPointerType(context, int8Ty);
                parameterTypeList.emplace_back(opaquePtrTy);
                Value nullPtr = create.llvm.null(opaquePtrTy);
                parameterList.emplace_back(nullPtr);
                return;
            }
            // Add more cases if needed
        }

        // Default lowering for other calls (OMTensor wrapping, etc.)
        if (auto originalMemRef = mlir::dyn_cast<MemRefType>(ty)) {
            auto int64Ty = IntegerType::get(context, 64);
            auto memRefTy = mlir::dyn_cast<LLVM::LLVMStructType>(parameter.getType());
            auto memRefRank = krnl::getRankFromMemRefType(memRefTy);
            auto memRefRankVal =
                create.llvm.constant(int64Ty, static_cast<int64_t>(memRefRank));
            Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
                RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});

            Type llvmOrigElemTy =
                llvmTypeConverter->convertType(originalMemRef.getElementType());
            krnl::fillOMTensorWithMemRef(parameter, llvmOrigElemTy, omTensor,
                false /*outOwning*/, rewriter, loc, apiRegistry, module);
            auto int8Ty = IntegerType::get(context, 8);
            auto opaquePtrTy = getPointerType(context, int8Ty);
            parameterTypeList.emplace_back(opaquePtrTy);
            parameterList.emplace_back(omTensor);
            omTensors.emplace_back(omTensor);
        } else if (mlir::isa<NoneType>(ty)) {
            auto int8Ty = IntegerType::get(context, 8);
            auto opaquePtrTy = getPointerType(context, int8Ty);
            parameterTypeList.emplace_back(opaquePtrTy);
            Value nullPtr = create.llvm.null(opaquePtrTy);
            parameterList.emplace_back(nullPtr);
        } else {
            parameterTypeList.emplace_back(parameter.getType());
            parameterList.emplace_back(parameter);
        }
    }

    /*
     * Purpose: Handle one attribute for KrnlCallOp lowering.
     */
    void handleOneAttribute(PatternRewriter &rewriter, Operation *op,
        Attribute attribute, llvm::SmallVector<Type, 4> &parameterTypeList,
        llvm::SmallVector<Value, 4> &parameterList) const {
        auto *context = op->getContext();
        Location loc = op->getLoc();
        ModuleOp module = op->getParentOfType<ModuleOp>();
        MultiDialectBuilder<KrnlBuilder, LLVMBuilder> create(rewriter, loc);
        const LLVMTypeConverter *llvmTypeConverter =
            static_cast<const LLVMTypeConverter *>(getTypeConverter());
        const auto &apiRegistry =
            RuntimeAPIRegistry(module, rewriter, *llvmTypeConverter);

        TypeSwitch<Attribute>(attribute)
            .Case<StringAttr>([&](StringAttr strAttr) {
                StringRef attrValue = strAttr.getValue();
                LLVM::GlobalOp globalStr = krnl::getOrCreateGlobalString(
                    attrValue, loc, rewriter, module, llvmTypeConverter);
                Value strPtr = krnl::getPtrToGlobalString(globalStr, loc, rewriter);
                auto int8Ty = IntegerType::get(context, 8);
                auto opaquePtrTy = getPointerType(context, int8Ty);
                parameterTypeList.emplace_back(opaquePtrTy);
                parameterList.emplace_back(strPtr);
            })
            .Case<IntegerAttr>([&](IntegerAttr integerAttr) {
                auto int64Ty = IntegerType::get(context, 64);
                Value cst =
                    rewriter.create<LLVM::ConstantOp>(loc, int64Ty, integerAttr);
                parameterTypeList.emplace_back(int64Ty);
                parameterList.emplace_back(cst);
            })
            .Case<FloatAttr>([&](FloatAttr floatAttr) {
                auto f64Ty = rewriter.getF64Type();
                Value cst = rewriter.create<LLVM::ConstantOp>(loc, f64Ty,
                    rewriter.getFloatAttr(f64Ty, floatAttr.getValueAsDouble()));
                parameterTypeList.emplace_back(f64Ty);
                parameterList.emplace_back(cst);
            })
            .Case<DenseElementsAttr>([&](DenseElementsAttr denseAttr) {
                auto tensorTy = mlir::cast<TensorType>(denseAttr.getType());
                auto memRefTy =
                    MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
                Value constantGlobal =
                    create.krnl.constant(memRefTy, "constant_", denseAttr);
                Value convertedConstantGlobal =
                    rewriter
                        .create<UnrealizedConversionCastOp>(loc,
                            llvmTypeConverter->convertType(memRefTy), constantGlobal)
                        .getResult(0);

                auto int64Ty = IntegerType::get(context, 64);
                auto memRefRank = memRefTy.getRank();
                auto memRefRankVal =
                    create.llvm.constant(int64Ty, static_cast<int64_t>(memRefRank));
                Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
                    RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});

                Type llvmElemTy =
                    llvmTypeConverter->convertType(memRefTy.getElementType());
                krnl::fillOMTensorWithMemRef(convertedConstantGlobal, llvmElemTy,
                    omTensor, false /*outOwning*/, rewriter, loc, apiRegistry,
                    module);
                auto int8Ty = IntegerType::get(context, 8);
                auto opaquePtrTy = getPointerType(context, int8Ty);
                parameterTypeList.emplace_back(opaquePtrTy);
                parameterList.emplace_back(omTensor);
            })
            .Default([&](Attribute attr) {
                llvm_unreachable("This type of Attribute used by krnl.call is not "
                                 "yet implemented");
            });
    }
};

/**********************************************
 * PATTERN REGISTRATION
 **********************************************/

void populateLoweringKrnlCallOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
    patterns.insert<KrnlCallOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir