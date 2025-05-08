##############################################
# IMPORT LIBRARIES
##############################################
import onnx
from onnx import TensorProto, helper

##############################################
# CONSTANTS & PARAMETERS
##############################################
MODEL_FILENAME = "model_with_mycallerop.onnx"
CUSTOM_OP_DOMAIN = "my.custom.domain"
CUSTOM_OP_NAME = "MyCallerOp"
CUSTOM_OP_VERSION = 1
ONNX_OPSET_VERSION = 15 # Using a common opset version

##############################################
# FUNCTION DEFINITIONS
##############################################
def create_and_save_model():
    """
    Creates an ONNX model with a custom operator and saves it.
    The model takes two 2x2 float tensors (XA, XB) and uses MyCallerOp
    to produce a 2x2 float tensor (XY).
    """
    # Define inputs
    XA = helper.make_tensor_value_info('XA', TensorProto.FLOAT, [2, 2])
    XB = helper.make_tensor_value_info('XB', TensorProto.FLOAT, [2, 2])

    # Define output
    XY = helper.make_tensor_value_info('XY', TensorProto.FLOAT, [2, 2])

    # Define the node using the custom op
    node_def = helper.make_node(
        CUSTOM_OP_NAME,      # Op type
        ['XA', 'XB'],        # Inputs
        ['XY'],              # Outputs
        domain=CUSTOM_OP_DOMAIN # Custom domain for the op
    )

    # Create the graph
    graph_def = helper.make_graph(
        [node_def],
        'my-caller-graph',
        [XA, XB],
        [XY]
    )

    # Create the model
    model_def = helper.make_model(graph_def, producer_name='onnx-example-fusedgemm-caller')

    # Set ONNX opset version
    model_def.opset_import[0].version = ONNX_OPSET_VERSION

    # Add custom domain to opset imports
    custom_domain_opset = model_def.opset_import.add()
    custom_domain_opset.domain = CUSTOM_OP_DOMAIN
    custom_domain_opset.version = CUSTOM_OP_VERSION

    # Save the model
    onnx.save(model_def, MODEL_FILENAME)
    print(f"Saved ONNX model: {MODEL_FILENAME}")

##############################################
# MAIN PROGRAM / SCRIPT EXECUTION
##############################################
if __name__ == "__main__":
    create_and_save_model()