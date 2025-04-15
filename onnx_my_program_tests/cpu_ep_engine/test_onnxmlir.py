##############################################
# IMPORT LIBRARIES ###########################
##############################################

"""
Libraries and packages used in this script and Versions Info for tools, libraries, and packages used.
- torch: 2.0.1+cu117 (Installed version, latest as of cutoff might differ)
- torch.nn: part of torch
- torch.nn.functional: part of torch
- onnxruntime: 1.17.3 (Installed version, latest as of cutoff might differ)
- onnx-mlir: latest as of cutoff (tool used via command line)
- PyRuntime: Part of onnx-mlir (latest as of cutoff)
- numpy: 1.26.4 (Installed version, latest as of cutoff might differ)
- os: Standard Python library
- subprocess: Standard Python library
"""
import torch # Version: 2.0.1+cu117
import torch.nn as nn
import torch.nn.functional as F
import os
import onnxruntime as ort # Version: 1.17.3
import subprocess
import numpy as np # Version: 1.26.4
try:
    # PyRuntime is part of the onnx-mlir build
    from PyRuntime import OMExecutionSession # PyRuntime (latest as of cutoff)
    PYRUNTIME_AVAILABLE = True
except ImportError:
    PYRUNTIME_AVAILABLE = False
    print("⚠️ Warning: PyRuntime module not found. Cannot run inference with onnx-mlir compiled model.")
    print("Ensure onnx-mlir is built with Python bindings and PyRuntime is in the PYTHONPATH or linked correctly.")
    print("Example PYTHONPATH: export PYTHONPATH=<path_to_onnx_mlir_build_dir>/Debug/lib:$PYTHONPATH")


###############################################
# CONSTANTS & PARAMETERS ######################
###############################################

"""
Constants and parameters used in this script.
"""
# Input image dimensions for MNIST
INPUT_CHANNELS = 1
IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CLASSES = 10

# File paths
ONNX_MODEL_PATH = "convnet_mnist.onnx"
OPTIMIZED_ONNX_CPU_PATH = "convnet_mnist_optimized_cpu.onnx"
ONNXMLIR_OUTPUT_LIB_PATH = "convnet_mnist.so" # Output shared library from onnx-mlir

##############################################
# FUNCTION DEFINITIONS #######################
##############################################

"""
Define the PyTorch model that replicates the Keras convolutional architecture.

Architecture:
    - Conv2D (32, kernel_size=3x3, ReLU)
    - MaxPool2D (pool_size=2x2)
    - Conv2D (64, kernel_size=3x3, ReLU)
    - MaxPool2D (pool_size=2x2)
    - Flatten
    - Dense (128, ReLU)
    - Dropout (0.5)
    - Dense (10, Softmax)
Note: In PyTorch, it is common to use nn.CrossEntropyLoss which expects logits, so softmax is usually applied
within the loss function. However, to reproduce the Keras model exactly, softmax is applied in forward().
"""

class ConvNet(nn.Module):
    def __init__(self):
        """Initializes the ConvNet model layers."""
        super(ConvNet, self).__init__()
        # Convolution and pooling layers
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=3)  # output: (32, 26, 26)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)           # output: (32, 13, 13)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)               # output: (64, 11, 11)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)           # output: (64, 5, 5)

        # Fully-connected layers
        # Calculate input size for fc1 dynamically based on pooling output
        # After conv1 (28-3+1=26), pool1 (26/2=13)
        # After conv2 (13-3+1=11), pool2 (11/2=5, integer division)
        fc1_input_features = 64 * 5 * 5
        self.fc1 = nn.Linear(fc1_input_features, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output tensor with probabilities for each class (batch_size, NUM_CLASSES).
        """
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64*5*5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1) # Apply softmax for probability distribution
        return x

##################################################################################
# MAIN PROGRAM ###################################################################
##################################################################################

"""
Main program to:
1. Define and initialize a ConvNet model.
2. Export the model to ONNX format.
3. Optimize the ONNX model for CPU using ONNX Runtime (optional comparison).
4. Compile the original ONNX model using onnx-mlir command-line tool.
5. Run inference using the onnx-mlir compiled model via PyRuntime.
"""

if __name__ == "__main__":
    #########################################
    # INITIALIZE MODEL & DUMMY INPUT ########
    #########################################
    model = ConvNet()
    model.eval() # Set model to evaluation mode
    print("--- Initialized PyTorch Model ---")
    print(model)

    # Create a dummy input batch matching model's expected input
    # Shape: (batch_size, channels, height, width)
    dummy_input_torch = torch.randn(1, INPUT_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    inputs_dict = {"pixel_values": dummy_input_torch}
    print(f"\n--- Created Dummy Input ---")
    print(f"Shape: {dummy_input_torch.shape}, Dtype: {dummy_input_torch.dtype}")

    #########################################
    # EXPORT MODEL TO ONNX FORMAT ###########
    #########################################
    print(f"\n--- Exporting PyTorch model to ONNX: {ONNX_MODEL_PATH} ---")
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,                                  # Model to export
                dummy_input_torch,                      # Model input (single tensor)
                ONNX_MODEL_PATH,                        # Output path
                export_params=True,                     # Store trained weights
                opset_version=13,                       # ONNX version
                do_constant_folding=True,               # Optimize constants
                input_names=['pixel_values'],           # Input names (list)
                output_names=['logits'],                # Output names (list)
                dynamic_axes={                          # Dynamic dimensions
                    'pixel_values': {0: 'batch_size'},  # batch_size is dynamic
                    'logits': {0: 'batch_size'}         # batch_size is dynamic
                }
            )
        if os.path.exists(ONNX_MODEL_PATH):
            print(f"✅ Model successfully exported to: {ONNX_MODEL_PATH}")
        else:
            # This else might not be reachable if export fails, as it usually raises an exception
            print(f"❌ Failed to export model (file not found after export call): {ONNX_MODEL_PATH}")
            exit(1)
    except Exception as e:
        print(f"❌ Failed to export model to ONNX: {e}")
        exit(1)

    ##############################################################
    # OPTIMIZE ONNX MODEL FOR CPU (ONNX RUNTIME) - Optional Step #
    ##############################################################
    # This step uses ONNX Runtime's optimization capabilities.
    # It's separate from onnx-mlir compilation. Useful for comparison.
    print(f"\n--- Optimizing ONNX model for CPU using ONNX Runtime -> {OPTIMIZED_ONNX_CPU_PATH} ---")
    try:
        cpu_sess_options = ort.SessionOptions()
        # Enable basic optimizations
        cpu_sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        cpu_sess_options.optimized_model_filepath = OPTIMIZED_ONNX_CPU_PATH

        # Create session to trigger optimization and save the optimized model
        _ = ort.InferenceSession(ONNX_MODEL_PATH, cpu_sess_options, providers=['CPUExecutionProvider'])

        if os.path.exists(OPTIMIZED_ONNX_CPU_PATH):
            print(f"✅ Optimized CPU ONNX model saved via ONNX Runtime: {OPTIMIZED_ONNX_CPU_PATH}")
            # You could potentially use this optimized model path later if needed
        else:
            print(f"⚠️ Optimized CPU model file not created by ONNX Runtime at: {OPTIMIZED_ONNX_CPU_PATH}")
            # Fallback or proceed with the original ONNX model if optimization failed to save
    except Exception as e:
        print(f"❌ Failed to optimize ONNX model using ONNX Runtime: {e}")
        # Continue with the original ONNX model path for onnx-mlir compilation

    ##############################################################
    # COMPILE ONNX MODEL WITH ONNX-MLIR ##########################
    ##############################################################
    print(f"\n--- Compiling ONNX model ({ONNX_MODEL_PATH}) with onnx-mlir -> {ONNXMLIR_OUTPUT_LIB_PATH} ---")
    # Ensure the input ONNX model exists before attempting compilation
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ Cannot compile with onnx-mlir: Input ONNX file not found at {ONNX_MODEL_PATH}")
        exit(1)

    # Construct the command based on documentation: onnx-mlir -O3 model.onnx -o output_base_name
    # onnx-mlir automatically appends '.so' (or '.dylib' on macOS, '.dll' on Windows)
    output_base_name = os.path.splitext(ONNXMLIR_OUTPUT_LIB_PATH)[0]
    compile_command = [
        "onnx-mlir",
        "-O3", # Optimization level
        ONNX_MODEL_PATH,
        "-o", output_base_name
    ]
    print(f"Running command: {' '.join(compile_command)}")

    try:
        # Execute the compilation command
        result = subprocess.run(compile_command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("--- onnx-mlir Compilation Output ---")
        print("STDOUT:")
        print(result.stdout if result.stdout else "<No stdout>")
        print("\nSTDERR:")
        print(result.stderr if result.stderr else "<No stderr>") # Stderr might contain progress/info

        # Verify that the expected output shared library was created
        if os.path.exists(ONNXMLIR_OUTPUT_LIB_PATH):
            print(f"✅ ONNX model successfully compiled by onnx-mlir: {ONNXMLIR_OUTPUT_LIB_PATH}")
        else:
            print(f"❌ onnx-mlir command finished, but output file ({ONNXMLIR_OUTPUT_LIB_PATH}) was not found.")
            print("Check compilation output above for errors.")
            exit(1) # Exit if compilation succeeded according to return code, but file is missing

    except FileNotFoundError:
        print("❌ Error: 'onnx-mlir' command not found.")
        print("Please ensure onnx-mlir executable is installed and in your system's PATH.")
        print("Refer to onnx-mlir documentation for installation and environment setup.")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during onnx-mlir compilation (return code {e.returncode}):")
        print("STDOUT:")
        print(e.stdout if e.stdout else "<No stdout>")
        print("\nSTDERR:")
        print(e.stderr if e.stderr else "<No stderr>")
        exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred during onnx-mlir compilation: {e}")
        exit(1)

    ##############################################################
    # RUN INFERENCE WITH ONNX-MLIR COMPILED MODEL (PyRuntime) ####
    ##############################################################
    print(f"\n--- Running inference with onnx-mlir compiled model ({ONNXMLIR_OUTPUT_LIB_PATH}) ---")
    if not PYRUNTIME_AVAILABLE:
        print("❌ PyRuntime not available. Skipping inference.")
        exit(1) # Exit if PyRuntime is needed but not found

    if not os.path.exists(ONNXMLIR_OUTPUT_LIB_PATH):
         print(f"❌ Cannot run inference: Compiled model file not found at {ONNXMLIR_OUTPUT_LIB_PATH}")
         exit(1)

    try:
        # Load the compiled model using OMExecutionSession
        print(f"Loading compiled model from: {ONNXMLIR_OUTPUT_LIB_PATH}")
        session = OMExecutionSession(ONNXMLIR_OUTPUT_LIB_PATH)
        print("✅ OMExecutionSession loaded successfully.")

        # Print model signatures (optional, useful for debugging)
        try:
            print("Input signature:", session.input_signature())
            print("Output signature:", session.output_signature())
        except Exception as sig_e:
            print(f"⚠️ Could not retrieve model signatures: {sig_e}")


        # Prepare the input data: PyRuntime expects a list of NumPy arrays
        # Convert the dummy torch tensor to numpy
        input_data_np = [dummy_input_torch.numpy()]
        print(f"Input data shape for PyRuntime: {input_data_np[0].shape}, dtype: {input_data_np[0].dtype}")

        # Run inference
        print("Running inference...")
        outputs = session.run(input_data_np)
        print("✅ Inference completed using PyRuntime.")

        # Process and print results
        if outputs and isinstance(outputs, list) and len(outputs) > 0:
            output_np = outputs[0] # Assuming the first output is the primary result
            print(f"Output shape from PyRuntime: {output_np.shape}, dtype: {output_np.dtype}")
            # Print the first 10 elements of the output tensor (or fewer if less than 10)
            print("Output probabilities (first {} values):".format(min(10, output_np.size)), output_np.flatten()[:10])

            # Find the predicted class index (index with the highest probability)
            predicted_class_index = np.argmax(output_np, axis=1)[0] # Get index for the first item in batch
            predicted_probability = output_np[0, predicted_class_index]
            print(f"Predicted class index: {predicted_class_index} with probability: {predicted_probability:.4f}")
        else:
            print("❓ Unexpected or empty output format from PyRuntime:", outputs)

    except Exception as e:
        print(f"❌ Error during inference with PyRuntime: {e}")
        print("Check if the compiled model is compatible and the input data format is correct.")
        exit(1)


    #########################################
    # CLEANUP (Optional) ####################
    #########################################
    # Uncomment the following lines to remove generated files after execution
    # print("\n--- Cleanup ---")
    # files_to_remove = [ONNX_MODEL_PATH, OPTIMIZED_ONNX_CPU_PATH, ONNXMLIR_OUTPUT_LIB_PATH]
    # for file_path in files_to_remove:
    #     if os.path.exists(file_path):
    #         try:
    #             os.remove(file_path)
    #             print(f"Removed: {file_path}")
    #         except OSError as e:
    #             print(f"Error removing {file_path}: {e}")

    print("\n--- Script Finished Successfully ---")