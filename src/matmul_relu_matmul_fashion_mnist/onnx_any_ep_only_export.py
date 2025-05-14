##############################################
# IMPORT LIBRARIES ###########################
##############################################

"""
Libraries and packages used in this script and Versions Info for tools, libraries, and packages used in this script.
"""
import numpy as np      # Version: 1.26.4 (latest as of cutoff)
import time
import pickle as pkl
import torch            # Version: 2.2.2 (latest as of cutoff)
import torchvision      # Version: 0.17.2 (latest as of cutoff)
import torch.nn as nn
import onnxruntime as ort # Version: 1.17.1 (latest as of cutoff)
import os

###############################################
# CONSTANTS & PARAMETERS ######################
###############################################

"""
Constants and parameters used in this script.
"""
# Paths
DATA_ROOT = "data"
MODEL_PARAMS_PATH = "fasionmnist_mlp_params.pkl"
ONNX_FP32_PATH = "mnist_model_cpu_initial.onnx"
OPTIMIZED_ONNX_FP32_PATH = "mnist_model_cpu_optimized.onnx" # Path for the optimized model

# Device and Execution Provider settings
DEVICE = "cpu" # For PyTorch model (can be "cuda" if GPU available and desired)
EXECUTION_PROVIDER = "CPUExecutionProvider"  # Change to e.g. "CUDAExecutionProvider" or "OpenVINOExecutionProvider" as needed

# Dataset configuration
BATCH_SIZE = 1
FEATURE_DIM = 784  # 28x28 flattened
IMG_SHAPE = (1, 28, 28)
NUM_CLASSES = 10

# Benchmark configuration
NUM_ITERATIONS = 500
WARMUP_ITERATIONS = 100  # Number of warmup iterations

# Class names for interpretation
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Starting ONNX Runtime with CPU EP benchmark and optimization script")

##############################################
# FUNCTION DEFINITIONS #######################
##############################################

class MLPModel(nn.Module):
    """
    Defines a simple Multi-Layer Perceptron model.

    Args:
        w0 (np.ndarray): Weight matrix for the first linear layer.
        b0 (np.ndarray): Bias vector for the first linear layer.
        w1 (np.ndarray): Weight matrix for the second linear layer.
        b1 (np.ndarray): Bias vector for the second linear layer.
        dtype (torch.dtype): Data type for the model parameters (default: torch.float32).
    """
    def __init__(self, w0, b0, w1, b1, dtype=torch.float32):
        super(MLPModel, self).__init__()
        # Ensure dimensions match expected PyTorch linear layer (out_features, in_features)
        self.fc1 = nn.Linear(w0.shape[1], w0.shape[0])
        self.fc2 = nn.Linear(w1.shape[1], w1.shape[0])
        self.relu = nn.ReLU()

        # Load weights and biases, ensuring correct data type
        self.fc1.weight = nn.Parameter(torch.tensor(w0, dtype=dtype))
        self.fc1.bias = nn.Parameter(torch.tensor(b0, dtype=dtype))
        self.fc2.weight = nn.Parameter(torch.tensor(w1, dtype=dtype))
        self.fc2.bias = nn.Parameter(torch.tensor(b1, dtype=dtype))

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor (logits).
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

##################################################################################
# MAIN PROGRAM ###################################################################
##################################################################################

if __name__ == "__main__":
    #########################################
    # DATASET LOADING #######################
    #########################################
    print("Loading FashionMNIST dataset...")
    test_data = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    # Get a sample input for tracing/export
    img_sample, _ = next(iter(test_loader))
    img_sample = img_sample.reshape(-1, FEATURE_DIM) # Flatten
    print(f"Sample input image shape: {img_sample.shape}, dtype: {img_sample.dtype}")

    #########################################
    # LOAD MODEL & WEIGHTS ##################
    #########################################
    print(f"Loading pre-trained weights from {MODEL_PARAMS_PATH}...")
    try:
        mlp_params = pkl.load(open(MODEL_PARAMS_PATH, "rb"))
    except FileNotFoundError:
        print(f"❌ Error: Model parameters file not found at {MODEL_PARAMS_PATH}")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading model parameters: {e}")
        exit(1)

    # Create FP32 model instance on CPU
    model_fp32 = MLPModel(
        mlp_params["w0"],
        mlp_params["b0"],
        mlp_params["w1"],
        mlp_params["b1"],
        dtype=torch.float32
    ).to(DEVICE) # Ensure model is on CPU
    model_fp32.eval() # Set to evaluation mode
    print("✅ Model created and weights loaded onto CPU.")

    #########################################
    # INITIAL ONNX EXPORT (FP32) ############
    #########################################
    print(f"Exporting initial FP32 model to {ONNX_FP32_PATH}...")
    try:
        # Prepare input tensor on the correct device (CPU)
        tracing_input_fp32 = img_sample.to(DEVICE, dtype=torch.float32)

        torch.onnx.export(
            model_fp32,                   # model being run
            tracing_input_fp32,           # model input (or a tuple for multiple inputs)
            ONNX_FP32_PATH,               # where to save the model
            export_params=True,           # store the trained parameter weights inside the model file
            opset_version=11,             # the ONNX version to export the model to
            do_constant_folding=True,     # whether to execute constant folding for optimization
            input_names=['input'],        # the model's input names
            output_names=['output']       # the model's output names
            # dynamic_axes removed to hardcode batch size based on tracing_input_fp32
        )
        print(f"✅ Successfully exported initial FP32 ONNX model to {ONNX_FP32_PATH}")
    except Exception as e:
        print(f"❌ Error exporting initial ONNX model: {e}")
        exit(1)

    ####################################################
    # ONNX RUNTIME SESSION & OPTIMIZATION ##############
    ####################################################
    print(f"Initializing ONNX Runtime with {EXECUTION_PROVIDER} and enabling optimization...")

    # Check available providers
    available_providers = ort.get_available_providers()
    print(f"Available ONNX Runtime providers: {available_providers}")
    if EXECUTION_PROVIDER not in available_providers:
        print(f"❌ Critical Error: {EXECUTION_PROVIDER} is not available in this ONNX Runtime build!")
        exit(1)

    try:
        # Configure session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = OPTIMIZED_ONNX_FP32_PATH

        # Create the inference session with the selected provider
        onnx_session = ort.InferenceSession(
            ONNX_FP32_PATH,
            sess_options=sess_options,
            providers=[EXECUTION_PROVIDER]
        )
        print(f"✅ Successfully created ONNX Runtime session with {EXECUTION_PROVIDER}.")
        print(f"✅ ONNX Runtime optimizations applied. Optimized model saved to: {OPTIMIZED_ONNX_FP32_PATH}")

        if not os.path.exists(OPTIMIZED_ONNX_FP32_PATH):
            print(f"⚠️ Warning: Optimized model file was expected but not found at {OPTIMIZED_ONNX_FP32_PATH}")

    except Exception as e:
        print(f"❌ Failed to create ONNX session or apply optimizations: {e}")
        exit(1)

    print(f"\nExport complete. Vanilla ONNX model: {ONNX_FP32_PATH}")
    print(f"Optimized ONNX model: {OPTIMIZED_ONNX_FP32_PATH}")
    print("Script finished.")