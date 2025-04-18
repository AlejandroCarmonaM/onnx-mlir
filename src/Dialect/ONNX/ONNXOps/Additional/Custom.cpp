/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ .cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Custom operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "mlir/IR/TypeUtilities.h" // For getElementTypeOrSelf
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp" // For shape helper base class
#include <cassert> // For assert

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXCustomOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {

  // Special handling for FusedGemm
  if (getFunctionName() == "FusedGemm") {
    // Check if inputs A and B are RankedTensorType first.
    if (!mlir::isa<RankedTensorType>(getInputs()[0].getType()) ||
        !mlir::isa<RankedTensorType>(getInputs()[1].getType())) {
      // Cannot infer shape if inputs are not ranked tensors yet.
      return success();
    }

    // Inputs are ranked, cast them. aType and bType are now in scope.
    auto aType = mlir::cast<RankedTensorType>(getInputs()[0].getType());
    auto bType = mlir::cast<RankedTensorType>(getInputs()[1].getType());

    // Check if inputs are 2D.
    if (aType.getRank() != 2 || bType.getRank() != 2) {
      // Cannot infer shape if inputs are not 2D.
      // Let the verifier handle this as a potential error later.
      return success();
    }

    // Get Gemm attributes
    // Use getAttrOfType for safety, provide default if missing? Assume present for now.
    bool transA = (*this)->getAttrOfType<IntegerAttr>("transA").getValue().getSExtValue() != 0;
    bool transB = (*this)->getAttrOfType<IntegerAttr>("transB").getValue().getSExtValue() != 0;

    // Get dimensions (handle transpose) - aType and bType are now accessible
    int64_t M = transA ? aType.getShape()[1] : aType.getShape()[0];
    int64_t K_A = transA ? aType.getShape()[0] : aType.getShape()[1]; // K from A
    int64_t K_B = transB ? bType.getShape()[1] : bType.getShape()[0]; // K from B
    int64_t N = transB ? bType.getShape()[0] : bType.getShape()[1];

    // Validation: K dimensions must match if both are static
    if (K_A != ShapedType::kDynamic && K_B != ShapedType::kDynamic && K_A != K_B) {
       // This should ideally be caught by a verifier, but good to check here too.
       // return emitOptionalError(getLoc(), "FusedGemm K dimensions mismatch");
       return success(); // Let verifier handle error
    }

    // Determine output element type
    // Use attribute if present, otherwise infer from input A (or B, should match)
    Type outputElementType = getOutputElementType().value_or(aType.getElementType());

    // Define the output shape: [M, N]
    SmallVector<int64_t, 2> outputShapeVec;
    outputShapeVec.push_back(M); // M can be dynamic
    // N must be static for standard Gemm lowering/allocation.
    // If N is dynamic, it might indicate an issue or require different handling.
    if (N == ShapedType::kDynamic) {
        // Let the verifier handle this potential issue later.
        return success();
    }
    outputShapeVec.push_back(N);

    // Manually set the result type to tensor<[M x N], outputElementType>
    RankedTensorType newResTy =
        RankedTensorType::get(outputShapeVec, outputElementType);

    // opResult(0) is the first result
    getResult(0).setType(newResTy);

    return success(); // Successfully inferred shape for FusedGemm
  }

  // Original logic for other custom ops using shape_infer_pattern
  if (!hasShapeAndRank(getOperation()))
    return success();

  if (!getShapeInferPattern().has_value()) {
    // When no shape inference pattern provided, Just return.
    return success();
  } else if (getResults().size() > 1) {
    // ToFix: implementation limitation of existing ShapeHelper
    return emitError(
        "Shape inference pattern for multiple outputs NOT supported");
  }

  // Determine the element type of output.
  // Use output_element_type attribute if specified.
  // Otherwise, use the first input in the list of inputs_for_infer.
  std::optional<ArrayAttr> inputIndexAttrs = getInputsForInfer();
  int64_t inputIdx = 0;
  if (inputIndexAttrs.has_value() && !inputIndexAttrs.value().empty()) {
    // Ensure the index is valid
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(inputIndexAttrs.value().getValue()[0])) {
        inputIdx = intAttr.getInt();
        if (inputIdx < 0 || inputIdx >= (int64_t)getInputs().size()) {
             return emitError("Invalid input index in inputs_for_infer");
        }
    } else {
        return emitError("Non-integer attribute in inputs_for_infer");
    }
  } else if (getInputs().empty()) {
      // Cannot infer element type if there are no inputs and no attribute
      return emitError("Cannot infer output element type: no inputs and no output_element_type attribute");
  }


  Type elementType = getOutputElementType().value_or(
      getElementType(getInputs()[inputIdx].getType()));

  // Use the base ShapeHelper for pattern-based inference
  ONNXCustomOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}