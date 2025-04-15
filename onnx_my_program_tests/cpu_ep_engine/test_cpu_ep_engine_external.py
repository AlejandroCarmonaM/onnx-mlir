##############################################
# IMPORT LIBRARIES ###########################
##############################################

"""
Libraries and packages used in this script and Versions Info for tools, libraries, and packages used.
- torch: 2.0.1 (latest as of cutoff)
- torch.nn: part of torch
- torch.nn.functional: part of torch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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

##############################################
# FUNCTION DEFINITIONS #######################
##############################################

"""
Define the PyTorch model that replicates the Keras convoautional architecture.

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
        super(ConvNet, self).__init__()
        # Convolution and pooling layers
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=3)  # output: (32, 26, 26)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)           # output: (32, 13, 13)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)               # output: (64, 11, 11)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)           # output: (64, 5, 5)
        
        # Fully-connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).
            
        Returns:
            torch.Tensor: Output tensor with probabilities for each class.
        """
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

##############################################
# MAIN PROGRAM ###############################
##############################################

"""
Main program to export the ConvNet model to ONNX format and optimize it for CPU using ONNX Runtime.
"""

if __name__ == "__main__":
    #########################################
    # INITIALIZE MODEL & DUMMY INPUT ########
    #########################################
    model = ConvNet()
    print("Model architecture:\n", model)
    
    # Create a dummy input batch (batch size, channels, height, width) as per MNIST dimensions
    dummy_input = torch.randn(1, INPUT_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    inputs = {"pixel_values": dummy_input}
    
    #########################################
    # Step 3: Export the model to ONNX format
    #########################################
    print("\n--- Step 3: Exporting model to ONNX format ---")
    onnx_path = "convnet_mnist.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,                                  # Model to export
            tuple(inputs.values()),                 # Model inputs
            onnx_path,                              # Output path
            export_params=True,                     # Store trained weights
            opset_version=13,                       # ONNX version (latest as of cutoff)
            do_constant_folding=True,               # Optimize constants
            input_names=list(inputs.keys()),        # Input names
            output_names=["logits"],                # Output names
            dynamic_axes={                          # Dynamic dimensions
                'pixel_values': {0: 'batch_size'}
            }
        )
    print(f"Model exported to: {onnx_path}")
    
    ##############################################################
    # Optimize ONNX model for CPU using ONNX Runtime (v1.15.0) ####
    ##############################################################
    import os
    import onnxruntime as ort  # ONNX Runtime v1.15.0 (latest as of cutoff)
    
    optimized_onnx_cpu_path = "convnet_mnist_optimized_cpu.onnx"
    cpu_sess_options = ort.SessionOptions()
    cpu_sess_options.optimized_model_filepath = optimized_onnx_cpu_path
    print("\n--- Optimizing ONNX model for CPU ---")
    _ = ort.InferenceSession(onnx_path, cpu_sess_options, providers=['CPUExecutionProvider'])
    if os.path.exists(optimized_onnx_cpu_path):
        print(f"✅ Optimized CPU ONNX model saved to: {optimized_onnx_cpu_path}")
        onnx_cpu_model_path = optimized_onnx_cpu_path
    else:
        print("❌ Optimized CPU model not found, continuing with the original ONNX model for CPU.")
        onnx_cpu_model_path = onnx_path