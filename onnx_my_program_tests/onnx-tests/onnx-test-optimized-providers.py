# ResNet-50 ONNX and ONNX Runtime Demo with TensorRT
import os
import time
import numpy as np
import torch
import argparse
import subprocess
from PIL import Image
import onnxruntime as ort
from transformers import AutoImageProcessor, ResNetForImageClassification
from datasets import load_dataset

def export_optimized_model(model_path, ort_output_dir=".", use_cuda=False):
    """Convert ONNX model to ORT format using command-line tool,
    targeting a runtime optimization style and a target platform.
    If use_cuda is True, it uses the Runtime optimization and also
    passes a target platform flag to target CUDA GPUs (e.g. amd64).
    """
    model_path_without_ext = os.path.splitext(model_path)[0]
    ort_model_path = model_path_without_ext + ".ort"
    
    # Build the command with the output directory
    cmd = [
        "python3", "-m", "onnxruntime.tools.convert_onnx_models_to_ort",
        "--output_dir", ort_output_dir
    ]
    # Append optimization style and target platform flags
    if use_cuda:
        cmd.extend(["--optimization_style", "Runtime", "--target_platform", "amd64"])
    else:
        cmd.extend(["--optimization_style", "Fixed", "--target_platform", "amd64"])
    
    # Append double-dash to signal end of flag arguments
    cmd.append("--")
    # Append the model path as the positional argument.
    cmd.append(model_path)
    
    try:
        print("Running:", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print("Command output:")
        print(result.stdout)
        
        # Check if the file was created
        if os.path.exists(ort_model_path):
            print(f"✅ ORT model created successfully at: {ort_model_path}")
            return ort_model_path
        else:
            # Look for any .ort file in the specified directory
            for file in os.listdir(ort_output_dir):
                if file.endswith(".ort") and file.startswith(os.path.basename(model_path)):
                    actual_path = os.path.join(ort_output_dir, file)
                    print(f"✅ ORT model found at different path: {actual_path}")
                    return actual_path
            print("❌ ORT model not found after conversion")
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Command execution failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Failed to run conversion command: {str(e)}")
        return None

# Parse command line arguments
parser = argparse.ArgumentParser(description='ONNX Runtime benchmarking for ResNet-50')
parser.add_argument('--use-tensorrt', action='store_true', 
                    help='Enable TensorRT execution provider (may cause crashes if not properly installed)')
parser.add_argument('--export-ort', action='store_true',
                    help='Export optimized ORT format model in addition to ONNX')
args = parser.parse_args()

# Check available hardware and providers
has_cuda = torch.cuda.is_available()
print(f"CUDA available: {has_cuda}")
print(f"ONNX Runtime Providers: {ort.get_available_providers()}")
print(f"TensorRT usage requested: {args.use_tensorrt}")
print(f"ORT model export requested: {args.export_ort}")
print(f"ONNX Runtime version: {ort.__version__}")

# Function to measure inference time
def measure_inference_time(func, num_runs=10):
    # Warmup runs
    for _ in range(3):
        func()
    
    # Timed runs
    start_time = time.time()
    for _ in range(num_runs):
        func()
    end_time = time.time()
    
    # Return average time per run
    return (end_time - start_time) / num_runs

# ------------------- Main Program -------------------

# Step 1: Load the ResNet-50 model and prepare a test image
print("\n--- Step 1: Loading ResNet-50 model and test image ---")
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
print(f"Loaded test image of size: {image.size}")

# Load model and processor
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model.eval()  # Set to evaluation mode
print("Loaded ResNet-50 model from Hugging Face")

# Prepare input for the model
inputs = processor(image, return_tensors="pt")
print(f"Input shape: {inputs['pixel_values'].shape}")

# Step 2: Run inference with PyTorch model
print("\n--- Step 2: Running inference with PyTorch model (CPU) ---")
def run_pytorch_inference():
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits

pytorch_outputs = run_pytorch_inference()
pytorch_prediction = pytorch_outputs.argmax(-1).item()
print(f"(CPU) PyTorch prediction: {model.config.id2label[pytorch_prediction]}")
pytorch_time = measure_inference_time(run_pytorch_inference)
print(f"(CPU) PyTorch average inference time: {pytorch_time*1000:.2f} ms")

# Step 2.1: Run inference with PyTorch model on GPU (if available)
if has_cuda:
    print("\n--- Step 2.1: Running inference with PyTorch model (GPU) ---")
    model.to("cuda")
    inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
    def run_pytorch_gpu_inference():
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.logits
    pytorch_gpu_outputs = run_pytorch_gpu_inference()
    pytorch_gpu_prediction = pytorch_gpu_outputs.argmax(-1).item()
    print(f"(GPU) PyTorch prediction: {model.config.id2label[pytorch_gpu_prediction]}")
    pytorch_gpu_time = measure_inference_time(run_pytorch_gpu_inference)
    print(f"(GPU) PyTorch average inference time: {pytorch_gpu_time*1000:.2f} ms")

# Step 3: Export the model to ONNX format
print("\n--- Step 3: Exporting model to ONNX format ---")
onnx_path = "resnet50.onnx"
with torch.no_grad():
    torch.onnx.export(
        model,                                  # Model to export
        tuple(inputs.values()),                # Model inputs
        onnx_path,                             # Output path
        export_params=True,                    # Store trained weights
        opset_version=13,                      # ONNX version
        do_constant_folding=True,              # Optimize constants
        input_names=list(inputs.keys()),       # Input names
        output_names=["logits"],               # Output names
        dynamic_axes={                         # Dynamic dimensions
            'pixel_values': {0: 'batch_size'}
        }
    )
print(f"Model exported to: {onnx_path}")

# ---- New Code: Optimize ONNX model for CPU and CUDA providers ----

# Optimize for CPU execution
optimized_onnx_cpu_path = "resnet50_optimized_cpu.onnx"
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

# Optimize for CUDA execution (if CUDA provider available)
onnx_cuda_model_path = onnx_path  # default fallback
if 'CUDAExecutionProvider' in ort.get_available_providers():
    optimized_onnx_cuda_path = "resnet50_optimized_cuda.onnx"
    cuda_sess_options = ort.SessionOptions()
    cuda_sess_options.optimized_model_filepath = optimized_onnx_cuda_path
    print("\n--- Optimizing ONNX model for CUDA ---")
    _ = ort.InferenceSession(onnx_path, cuda_sess_options, providers=['CUDAExecutionProvider'])
    if os.path.exists(optimized_onnx_cuda_path):
        print(f"✅ Optimized CUDA ONNX model saved to: {optimized_onnx_cuda_path}")
        onnx_cuda_model_path = optimized_onnx_cuda_path
    else:
        print("❌ Optimized CUDA model not found, continuing with the original ONNX model for CUDA.")
else:
    print("CUDA provider not available, skipping CUDA model optimization.")

# Step 3.1: Export the model to ORT format if requested
ort_path = None
if args.export_ort:
    print("\n--- Step 3.1: Exporting model to ORT format ---")
    ort_path = export_optimized_model(onnx_path, use_cuda=True)
    if ort_path:
        print(f"Model exported to ORT format: {ort_path}")
    else:
        print("   Continuing without ORT optimized model.")

# Step 4: Create ONNX Runtime sessions for different providers
print("\n--- Step 4: Creating ONNX Runtime sessions ---")
# Convert PyTorch tensors to NumPy arrays (always on CPU)
ort_inputs = {name: tensor.cpu().numpy() for name, tensor in inputs.items()}

# Create CPU session using optimized CPU model
cpu_session = ort.InferenceSession(onnx_cpu_model_path, providers=['CPUExecutionProvider'])
print("Created CPU inference session (ONNX)")

# Create CPU session with ORT model (if available)
ort_cpu_session = None
if ort_path and os.path.exists(ort_path):
    ort_cpu_session = ort.InferenceSession(ort_path, providers=['CPUExecutionProvider'])
    print("Created CPU inference session (ORT optimized)")

# Create CUDA session using optimized CUDA model (if available)
cuda_session = None
if 'CUDAExecutionProvider' in ort.get_available_providers():
    cuda_session = ort.InferenceSession(onnx_cuda_model_path, providers=['CUDAExecutionProvider'])
    print("Created CUDA inference session (ONNX)")
else:
    print("CUDA provider not available, skipping CUDA session creation")

# Create CUDA session with ORT model (if available)
ort_cuda_session = None
if ort_path and os.path.exists(ort_path) and 'CUDAExecutionProvider' in ort.get_available_providers():
    ort_cuda_session = ort.InferenceSession(ort_path, providers=['CUDAExecutionProvider'])
    print("Created CUDA inference session (ORT optimized)")

# Create TensorRT session only if explicitly requested
tensorrt_session = None
tensorrt_usable = False
if args.use_tensorrt and 'TensorrtExecutionProvider' in ort.get_available_providers():
    print("\nAttempting to create TensorRT inference session (as requested)...")
    try:
        provider_options = [{
            'device_id': 0,
            'trt_max_workspace_size': 2147483648,  # 2GB workspace
            'trt_fp16_enable': True,               # Enable FP16 precision
            'trt_engine_cache_enable': True,       # Enable engine caching
            'trt_engine_cache_path': '.'
        }]
        tensorrt_session = ort.InferenceSession(
            onnx_path, 
            providers=[('TensorrtExecutionProvider', provider_options[0]), 'CUDAExecutionProvider']
        )
        print("✅ Created TensorRT inference session successfully")
    except Exception as e:
        print(f"❌ Failed to create TensorRT session: {e}")
        print("   This could be due to missing TensorRT libraries or incompatible ops.")
        tensorrt_session = None
elif args.use_tensorrt:
    print("TensorRT provider not available in this ONNX Runtime build.")
else:
    print("TensorRT usage not requested. Skipping TensorRT initialization.")

# Step 5: Run inference with ONNX Runtime on CPU
print("\n--- Step 5: Running inference with ONNX Runtime on CPU ---")
def run_onnx_cpu_inference():
    return cpu_session.run(None, ort_inputs)
onnx_cpu_outputs = run_onnx_cpu_inference()
onnx_cpu_prediction = np.argmax(onnx_cpu_outputs[0], axis=1)[0]
print(f"ONNX CPU prediction: {model.config.id2label[onnx_cpu_prediction]}")
onnx_cpu_time = measure_inference_time(run_onnx_cpu_inference)
print(f"ONNX CPU average inference time: {onnx_cpu_time*1000:.2f} ms")

# Step 5.1: Run inference with ORT optimized model on CPU (if available)
ort_cpu_time = None
ort_cpu_prediction = None
if ort_cpu_session is not None:
    print("\n--- Step 5.1: Running inference with ORT optimized model on CPU ---")
    def run_ort_cpu_inference():
        return ort_cpu_session.run(None, ort_inputs)
    ort_cpu_outputs = run_ort_cpu_inference()
    ort_cpu_prediction = np.argmax(ort_cpu_outputs[0], axis=1)[0]
    print(f"ORT CPU prediction: {model.config.id2label[ort_cpu_prediction]}")
    ort_cpu_time = measure_inference_time(run_ort_cpu_inference)
    print(f"ORT CPU average inference time: {ort_cpu_time*1000:.2f} ms")

# Step 6: Run inference with ONNX Runtime on CUDA (if available)
onnx_cuda_time = None
onnx_cuda_prediction = None
if cuda_session is not None:
    print("\n--- Step 6: Running inference with ONNX Runtime on CUDA ---")
    def run_onnx_cuda_inference():
        return cuda_session.run(None, ort_inputs)
    onnx_cuda_outputs = run_onnx_cuda_inference()
    onnx_cuda_prediction = np.argmax(onnx_cuda_outputs[0], axis=1)[0]
    print(f"ONNX CUDA prediction: {model.config.id2label[onnx_cuda_prediction]}")
    onnx_cuda_time = measure_inference_time(run_onnx_cuda_inference)
    print(f"ONNX CUDA average inference time: {onnx_cuda_time*1000:.2f} ms")

# Step 6.1: Run inference with ORT optimized model on CUDA (if available)
ort_cuda_time = None
ort_cuda_prediction = None
if ort_cuda_session is not None:
    print("\n--- Step 6.1: Running inference with ORT optimized model on CUDA ---")
    def run_ort_cuda_inference():
        return ort_cuda_session.run(None, ort_inputs)
    ort_cuda_outputs = run_ort_cuda_inference()
    ort_cuda_prediction = np.argmax(ort_cuda_outputs[0], axis=1)[0]
    print(f"ORT CUDA prediction: {model.config.id2label[ort_cuda_prediction]}")
    ort_cuda_time = measure_inference_time(run_ort_cuda_inference)
    print(f"ORT CUDA average inference time: {ort_cuda_time*1000:.2f} ms")

# Step 7: Run inference with ONNX Runtime on TensorRT (if available and requested)
onnx_tensorrt_time = None
onnx_tensorrt_prediction = None
if tensorrt_session is not None and args.use_tensorrt:
    print("\n--- Step 7: Running inference with ONNX Runtime on TensorRT ---")
    def run_onnx_tensorrt_inference():
        return tensorrt_session.run(None, ort_inputs)
    try:
        onnx_tensorrt_outputs = run_onnx_tensorrt_inference()
        onnx_tensorrt_prediction = np.argmax(onnx_tensorrt_outputs[0], axis=1)[0]
        print(f"ONNX TensorRT prediction: {model.config.id2label[onnx_tensorrt_prediction]}")
        onnx_tensorrt_time = measure_inference_time(run_onnx_tensorrt_inference)
        print(f"ONNX TensorRT average inference time: {onnx_tensorrt_time*1000:.2f} ms")
        tensorrt_usable = True
    except Exception as e:
        print(f"❌ TensorRT inference failed: {e}")
        print("   Skipping TensorRT in performance comparison.")
        tensorrt_usable = False

# Step 8: Compare results and performance
print("\n--- Step 8: Comparing results and performance ---")
if pytorch_prediction == onnx_cpu_prediction:
    print("✅ PyTorch (CPU) and ONNX CPU predictions match")
else:
    print("❌ PyTorch (CPU) and ONNX CPU predictions differ")
if ort_cpu_prediction is not None:
    if pytorch_prediction == ort_cpu_prediction:
        print("✅ PyTorch (CPU) and ORT CPU predictions match")
    else:
        print("❌ PyTorch (CPU) and ORT CPU predictions differ")
    if onnx_cpu_prediction == ort_cpu_prediction:
        print("✅ ONNX CPU and ORT CPU predictions match")
    else:
        print("❌ ONNX CPU and ORT CPU predictions differ")
if cuda_session is not None:
    if pytorch_gpu_prediction == onnx_cuda_prediction:
        print("✅ PyTorch (GPU) and ONNX CUDA predictions match")
    else:
        print("❌ PyTorch (GPU) and ONNX CUDA predictions differ")
    if onnx_cpu_prediction == onnx_cuda_prediction:
        print("✅ ONNX CPU and ONNX CUDA predictions match")
    else:
        print("❌ ONNX CPU and ONNX CUDA predictions differ")
if ort_cuda_prediction is not None:
    if pytorch_gpu_prediction == ort_cuda_prediction:
        print("✅ PyTorch (GPU) and ORT CUDA predictions match")
    else:
        print("❌ PyTorch (GPU) and ORT CUDA predictions differ")
    if onnx_cuda_prediction == ort_cuda_prediction:
        print("✅ ONNX CUDA and ORT CUDA predictions match")
    else:
        print("❌ ONNX CUDA and ORT CUDA predictions differ")
if tensorrt_usable and onnx_tensorrt_prediction is not None:
    if pytorch_gpu_prediction == onnx_tensorrt_prediction:
        print("✅ PyTorch (GPU) and ONNX TensorRT predictions match")
    else:
        print("❌ PyTorch (GPU) and ONNX TensorRT predictions differ")
    if onnx_cuda_prediction == onnx_tensorrt_prediction:
        print("✅ ONNX CUDA and ONNX TensorRT predictions match")
    else:
        print("❌ ONNX CUDA and ONNX TensorRT predictions differ")
print("\nPerformance comparison:")
print(f"PyTorch (CPU) vs ONNX CPU: {pytorch_time/onnx_cpu_time:.2f}x speedup")
if ort_cpu_time is not None:
    print(f"PyTorch (CPU) vs ORT CPU: {pytorch_time/ort_cpu_time:.2f}x speedup")
    print(f"ONNX CPU vs ORT CPU: {onnx_cpu_time/ort_cpu_time:.2f}x speedup")
if cuda_session is not None:
    print(f"PyTorch (GPU) vs ONNX CUDA: {pytorch_gpu_time/onnx_cuda_time:.2f}x speedup")
    print(f"ONNX CPU vs ONNX CUDA: {onnx_cpu_time/onnx_cuda_time:.2f}x speedup")
if ort_cuda_time is not None:
    print(f"PyTorch (GPU) vs ORT CUDA: {pytorch_gpu_time/ort_cuda_time:.2f}x speedup")
    print(f"ONNX CUDA vs ORT CUDA: {onnx_cuda_time/ort_cuda_time:.2f}x speedup")
    print(f"ORT CPU vs ORT CUDA: {ort_cpu_time/ort_cuda_time:.2f}x speedup")
if tensorrt_usable and onnx_tensorrt_time is not None:
    print(f"PyTorch (GPU) vs ONNX TensorRT: {pytorch_gpu_time/onnx_tensorrt_time:.2f}x speedup")
    print(f"ONNX CUDA vs ONNX TensorRT: {onnx_cuda_time/onnx_tensorrt_time:.2f}x speedup")
    print(f"ONNX CPU vs ONNX TensorRT: {onnx_cpu_time/onnx_tensorrt_time:.2f}x speedup")
    if ort_cuda_time is not None:
        print(f"ORT CUDA vs ONNX TensorRT: {ort_cuda_time/onnx_tensorrt_time:.2f}x speedup")

# Step 9: Save the results to a CSV file
print("\n--- Step 9: Saving results to a CSV file ---")
import csv
import datetime
import platform
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"resnet50_benchmark_{timestamp}.csv"
benchmark_data = []
benchmark_data.append({
    'Runtime': 'PyTorch',
    'Provider': 'CPU',
    'Inference Time (ms)': f"{pytorch_time*1000:.2f}",
    'Prediction': model.config.id2label[pytorch_prediction],
    'Prediction ID': pytorch_prediction,
    'Reference Match': 'N/A (Reference)'
})
benchmark_data.append({
    'Runtime': 'ONNX',
    'Provider': 'CPU', 
    'Inference Time (ms)': f"{onnx_cpu_time*1000:.2f}",
    'Prediction': model.config.id2label[onnx_cpu_prediction],
    'Prediction ID': onnx_cpu_prediction,
    'Reference Match': 'Yes' if pytorch_prediction == onnx_cpu_prediction else 'No',
    'Speedup vs PyTorch CPU': f"{pytorch_time/onnx_cpu_time:.2f}x"
})
if ort_cpu_prediction is not None:
    benchmark_data.append({
        'Runtime': 'ORT',
        'Provider': 'CPU',
        'Inference Time (ms)': f"{ort_cpu_time*1000:.2f}",
        'Prediction': model.config.id2label[ort_cpu_prediction],
        'Prediction ID': ort_cpu_prediction, 
        'Reference Match': 'Yes' if pytorch_prediction == ort_cpu_prediction else 'No',
        'Speedup vs PyTorch CPU': f"{pytorch_time/ort_cpu_time:.2f}x",
        'Speedup vs ONNX CPU': f"{onnx_cpu_time/ort_cpu_time:.2f}x"
    })
if has_cuda:
    benchmark_data.append({
        'Runtime': 'PyTorch',
        'Provider': 'CUDA',
        'Inference Time (ms)': f"{pytorch_gpu_time*1000:.2f}",
        'Prediction': model.config.id2label[pytorch_gpu_prediction],
        'Prediction ID': pytorch_gpu_prediction,
        'Reference Match': 'N/A (GPU Reference)',
        'Speedup vs PyTorch CPU': f"{pytorch_time/pytorch_gpu_time:.2f}x"
    })
if cuda_session is not None:
    benchmark_data.append({
        'Runtime': 'ONNX',
        'Provider': 'CUDA',
        'Inference Time (ms)': f"{onnx_cuda_time*1000:.2f}",
        'Prediction': model.config.id2label[onnx_cuda_prediction],
        'Prediction ID': onnx_cuda_prediction,
        'Reference Match': 'Yes' if pytorch_gpu_prediction == onnx_cuda_prediction else 'No',
        'Speedup vs PyTorch GPU': f"{pytorch_gpu_time/onnx_cuda_time:.2f}x",
        'Speedup vs ONNX CPU': f"{onnx_cpu_time/onnx_cuda_time:.2f}x"
    })
if ort_cuda_prediction is not None:
    benchmark_data.append({
        'Runtime': 'ORT',
        'Provider': 'CUDA',
        'Inference Time (ms)': f"{ort_cuda_time*1000:.2f}",
        'Prediction': model.config.id2label[ort_cuda_prediction], 
        'Prediction ID': ort_cuda_prediction,
        'Reference Match': 'Yes' if pytorch_gpu_prediction == ort_cuda_prediction else 'No',
        'Speedup vs PyTorch GPU': f"{pytorch_gpu_time/ort_cuda_time:.2f}x",
        'Speedup vs ONNX CUDA': f"{onnx_cuda_time/ort_cuda_time:.2f}x",
        'Speedup vs ORT CPU': f"{ort_cpu_time/ort_cuda_time:.2f}x" if ort_cpu_time is not None else 'N/A'
    })
if tensorrt_usable and onnx_tensorrt_prediction is not None:
    benchmark_data.append({
        'Runtime': 'ONNX',
        'Provider': 'TensorRT',
        'Inference Time (ms)': f"{onnx_tensorrt_time*1000:.2f}",
        'Prediction': model.config.id2label[onnx_tensorrt_prediction],
        'Prediction ID': onnx_tensorrt_prediction,
        'Reference Match': 'Yes' if pytorch_gpu_prediction == onnx_tensorrt_prediction else 'No',
        'Speedup vs PyTorch GPU': f"{pytorch_time/onnx_tensorrt_time:.2f}x",
        'Speedup vs ONNX CUDA': f"{onnx_cuda_time/onnx_tensorrt_time:.2f}x",
        'Speedup vs ONNX CPU': f"{onnx_cpu_time/onnx_tensorrt_time:.2f}x",
        'Speedup vs ORT CUDA': f"{ort_cuda_time/onnx_tensorrt_time:.2f}x" if ort_cuda_time is not None else 'N/A'
    })
system_info = {
    'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'Model': 'ResNet-50',
    'System': platform.system(),
    'Python Version': platform.python_version(),
    'PyTorch Version': torch.__version__,
    'ONNX Runtime Version': ort.__version__,
    'CUDA Available': str(has_cuda),
    'TensorRT Used': str(tensorrt_usable)
}
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['System Information'])
    for key, value in system_info.items():
        csv_writer.writerow([key, value])
    csv_writer.writerow([])
    if benchmark_data:
        all_fieldnames = set()
        for entry in benchmark_data:
            all_fieldnames.update(entry.keys())
        fieldnames = sorted(list(all_fieldnames))
        dict_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(benchmark_data)
print(f"Benchmark results saved to {csv_filename}")
print("\nDone!")