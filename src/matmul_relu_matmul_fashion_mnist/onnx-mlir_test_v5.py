##############################################
# IMPORT LIBRARIES ###########################
##############################################

"""
Libraries and packages used in this script and Versions Info for tools, libraries, and packages used in this script.
"""
import numpy as np          # Version: 1.26.4 (latest as of 2025-04-14)
import time
# import pickle as pkl # Not used directly if loading ONNX
import torch                # Version: 2.2.2 (latest as of 2025-04-14)
import torchvision          # Version: 0.17.2 (latest as of 2025-04-14)
# import torch.nn as nn # Not needed if only loading ONNX
import os
import subprocess           # Version: Python Standard Library
import sys                  # Version: Python Standard Library
# import tempfile # Not used if INFERENCE_TEMP_DIR_BASE is used
import shutil               # Version: Python Standard Library
import importlib.util       # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
import json                 # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
import argparse             # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
import signal               # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
import shlex                # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
import tempfile             # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
import onnx                 # Version: 1.16.0 (latest as of 2025-04-14, required by RunONNXModel.py)


###############################################
# CONSTANTS & PARAMETERS ######################
###############################################

"""
Constants and parameters used in this script.
"""
# Paths
MODEL_PATH = "mnist_model_cpu_optimized.onnx" # Path to the ONNX model to compile
# COMPILED_MODEL_BASE = "mnist_model_cpu_initial" # No longer needed, will use "model" inside a dir
COMPILED_MODEL_DIR = "compiled_model_onnx_mlir" # Directory to store the compiled model.so and model.constants.bin
DATA_ROOT = "data"
ONNX_MLIR_PATH = "onnx-mlir" # Assumes onnx-mlir executable is in PATH or provide full path
RUN_ONNX_MODEL_SCRIPT_PATH = "/workdir/onnx-mlir/utils/RunONNXModel.py" # Absolute path to the script
# Directory to store temporary files for each inference run, created in the script's CWD
INFERENCE_TEMP_DIR_BASE = "onnx_mlir_inference_runs"

# Dataset configuration
BATCH_SIZE = 1
FEATURE_DIM = 784  # 28x28 flattened
IMG_SHAPE = (1, 28, 28)
NUM_CLASSES = 10

# Benchmark configuration
NUM_ITERATIONS = 10 # Reduced for potentially slower subprocess calls
WARMUP_ITERATIONS = 1  # Number of warmup iterations

# Class names for interpretation
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Starting ONNX-MLIR benchmark script (using RunONNXModel.py via subprocess)")

##############################################
# FUNCTION DEFINITIONS #######################
##############################################

def find_executable(name, default_path):
    """
    Finds an executable by checking the default path and then the system PATH.

    Args:
        name (str): The name of the executable.
        default_path (str): The default path to check first.

    Returns:
        str: The full path to the executable if found. Exits if not found.
    """
    if os.path.exists(default_path) and os.access(default_path, os.X_OK):
        print(f"✅ Found '{name}' at specified path: {default_path}")
        return os.path.abspath(default_path)

    found_path = shutil.which(name)
    if found_path:
        print(f"✅ Found '{name}' in system PATH: {found_path}")
        return os.path.abspath(found_path)

    print(f"❌ Error: '{name}' not found at '{default_path}' or in system PATH.")
    sys.exit(1) # Exit if executable not found

def find_script(name, default_path):
    """
    Finds a script file.

    Args:
        name (str): The name of the script.
        default_path (str): The path to check for the script.

    Returns:
        str: The full path to the script if found. Exits if not found.
    """
    absolute_path = os.path.abspath(default_path)
    if os.path.exists(absolute_path) and os.path.isfile(absolute_path):
         print(f"✅ Found script '{name}' at specified path: {absolute_path}")
         return absolute_path
    print(f"❌ Error: Script '{name}' not found at '{default_path}' (abs: {absolute_path}). Please provide the correct path.")
    sys.exit(1) # Exit if script not found

# ...existing code...

def compile_onnx_model(onnx_model_path, output_dir, onnx_mlir_exec_path):
    """
    Compiles an ONNX model using onnx-mlir, then compiles and links with FusedGemmRuntime_new_age.cpp.
    """
    print(f"Compiling ONNX model '{onnx_model_path}' with onnx-mlir...")
    absolute_onnx_model_path = os.path.abspath(onnx_model_path)
    absolute_output_dir = os.path.abspath(output_dir)

    if not os.path.exists(absolute_onnx_model_path):
        print(f"❌ Error: Input ONNX model not found at '{absolute_onnx_model_path}'")
        return False, None

    # Ensure output directory exists and is clean
    if os.path.exists(absolute_output_dir):
        print(f"Removing existing compilation output directory: {absolute_output_dir}")
        shutil.rmtree(absolute_output_dir)
    os.makedirs(absolute_output_dir, exist_ok=True)
    print(f"Created compilation output directory: {absolute_output_dir}")

    # Define the base name for output files *inside* the output directory
    output_base_name = os.path.join(absolute_output_dir, "model")
    expected_lib_path = output_base_name + ".so"
    expected_obj_path = output_base_name + ".o"

    # Step 1: Emit object file from ONNX-MLIR
    compile_command = [
        onnx_mlir_exec_path,
        "--EmitObj",
        absolute_onnx_model_path,
        "-o", output_base_name
    ]
    print(f"Running command: {' '.join(compile_command)}")
    try:
        result = subprocess.run(compile_command, check=True, capture_output=True, text=True, timeout=300)
        print("✅ ONNX-MLIR object emission successful.")
        if not os.path.exists(expected_obj_path):
            print(f"❌ Error: Compiled object '{expected_obj_path}' not found after compilation.")
            print("Compiler Stdout:\n", result.stdout)
            print("Compiler Stderr:\n", result.stderr)
            return False, None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: ONNX-MLIR object emission failed with CalledProcessError: {e}")
        print("Compiler Stdout:\n", e.stdout)
        print("Compiler Stderr:\n", e.stderr)
        return False, None
    except Exception as e:
        print(f"❌ Error: ONNX-MLIR object emission failed: {e}")
        return False, None

    # Step 2: Compile FusedGemmRuntime_new_age.cpp to shared library
    fused_cpp = os.path.abspath("FusedGemmRuntime_new_age.cpp")
    # Output .so directly into the COMPILED_MODEL_DIR for simplicity in linking step 3
    fused_so = os.path.join(absolute_output_dir, "libFusedGemmRuntime_new_age.so")

    onnx_mlir_include = "/workdir/onnx-mlir/include"  # For OnnxMlirRuntime.h
    # Adjust these paths if your ONNX Runtime installation is different
    onnx_runtime_include_dir = "/workdir/onnxruntime-linux-x64-gpu-1.21.1/include"
    onnx_runtime_lib_dir = "/workdir/onnxruntime-linux-x64-gpu-1.21.1/lib"

    if not os.path.exists(fused_cpp):
        print(f"❌ Error: FusedGemmRuntime_new_age.cpp not found at {fused_cpp}")
        return False, None

    # Add MLIR include directory from MLIR_DIR env var
    # Point to MLIR top-level include directory so mlir-c headers are reachable
    mlir_include_dir = "/workdir/llvm-project/mlir/include"
    llvm_include_dir = "/workdir/llvm-project/llvm/include"  # Add this path
    if os.path.isdir(mlir_include_dir):
        print(f"Using MLIR include directory: {mlir_include_dir}")
    else:
        print(f"❌ Error: MLIR headers not found at {mlir_include_dir}")
        mlir_include_dir = None

    if os.path.isdir(llvm_include_dir):
        print(f"Using LLVM include directory: {llvm_include_dir}")
    else:
        print(f"❌ Error: LLVM headers not found at {llvm_include_dir}")
        llvm_include_dir = None

    clang_compile_cmd = [
        "clang++", "-fPIC", "-shared",
        f"-I{onnx_mlir_include}",
        f"-I{onnx_runtime_include_dir}",
        *( [f"-I{mlir_include_dir}"] if mlir_include_dir else [] ),
        *( [f"-I{llvm_include_dir}"] if llvm_include_dir else [] ),  # Add LLVM include path
        "-o", fused_so,
        fused_cpp,
        f"-L{onnx_runtime_lib_dir}", "-lonnxruntime",
        f"-Wl,-rpath,{onnx_runtime_lib_dir}"
    ]
    print(f"Compiling FusedGemmRuntime_new_age.cpp: {' '.join(clang_compile_cmd)}")
    try:
        result = subprocess.run(clang_compile_cmd, check=True, capture_output=True, text=True, timeout=120)
        print("✅ FusedGemmRuntime_new_age.cpp compilation successful.")
        if not os.path.exists(fused_so):
            print(f"❌ Error: Compiled shared library '{fused_so}' not found after successful compilation command.")
            print("Compiler Stdout:\n", result.stdout)  # Should not happen if check=True
            print("Compiler Stderr:\n", result.stderr)  # Should not happen if check=True
            return False, None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: FusedGemmRuntime_new_age.cpp compilation failed with CalledProcessError.")
        print(f"Command: {' '.join(e.cmd)}")
        print("Return code:", e.returncode)
        print("Compiler Stdout:\n", e.stdout)
        print("Compiler Stderr:\n", e.stderr)
        return False, None
    except Exception as e:
        print(f"❌ Error: FusedGemmRuntime_new_age.cpp compilation failed with an unexpected Exception: {e}")
        return False, None

    # Step 3: Link ONNX-MLIR object and FusedGemmRuntime_new_age.so into final model.so
    clang_link_cmd = [
        "clang++",
        expected_obj_path,
        fused_so, # Link against the .so we just built
        "-o", expected_lib_path,
        "-shared", "-fPIC",
        "-L/workdir/onnx-mlir/build/Debug/lib", "-lcruntime",
        # Add rpath for libFusedGemmRuntime_new_age.so if it's in a different directory than model.so
        # Since we are placing fused_so in absolute_output_dir (same as expected_lib_path),
        # an rpath like '$ORIGIN' might be implicitly enough or not needed if system linker paths are set.
        # For robustness, if fused_so were elsewhere, an rpath would be good.
        # Here, they are in the same directory.
        # Also, ensure model.so can find libonnxruntime.so (via libFusedGemmRuntime_new_age.so's rpath)
    ]
    print(f"Linking model and FusedGemmRuntime_new_age.so: {' '.join(clang_link_cmd)}")
    try:
        result = subprocess.run(clang_link_cmd, check=True, capture_output=True, text=True, timeout=120)
        print("✅ Linking successful.")
        if not os.path.exists(expected_lib_path):
            print(f"❌ Error: Linked library '{expected_lib_path}' not found after linking.")
            print("Linker Stdout:\n", result.stdout)
            print("Linker Stderr:\n", result.stderr)
            return False, None
        return True, absolute_output_dir
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Linking failed with CalledProcessError.")
        print(f"Command: {' '.join(e.cmd)}")
        print("Return code:", e.returncode)
        print("Linker Stdout:\n", e.stdout)
        print("Linker Stderr:\n", e.stderr)
        return False, None
    except Exception as e:
        print(f"❌ Error: Linking failed: {e}")
        return False, None

# ...existing code...

def run_inference_with_script(run_script_path, compiled_model_dir_path, input_data_list, output_dir):
    """
    Runs inference by calling the RunONNXModel.py script via subprocess.
    Assumes compiled model is 'model.so' inside compiled_model_dir_path.
    Saves input numpy arrays and a loader script to output_dir, then executes RunONNXModel.py.

    Args:
        run_script_path (str): Absolute path to the RunONNXModel.py script.
        compiled_model_dir_path (str): Absolute path to the directory containing the
                                       compiled 'model.so'.
        input_data_list (list): A list of NumPy arrays, one for each model input.
        output_dir (str): The directory (relative to CWD) to use for this specific
                          run's input/output files.

    Returns:
        tuple(list or None, float): A tuple containing:
            - list or None: A list of NumPy arrays representing the model outputs if successful, otherwise None.
            - float: The time taken for the subprocess call (includes overhead).
    """
    loaded_outputs = None

    # Ensure paths are absolute
    absolute_run_script_path = os.path.abspath(run_script_path)
    absolute_compiled_model_dir = os.path.abspath(compiled_model_dir_path)
    absolute_output_dir = os.path.abspath(output_dir) # Dir for this run's I/O

    # --- Basic Checks ---
    if not os.path.exists(absolute_run_script_path):
        print(f"❌ Error: Run script not found at: {absolute_run_script_path}")
        return None, 0
    if not os.path.isdir(absolute_compiled_model_dir):
        print(f"❌ Error: Compiled model directory not found or not a directory: {absolute_compiled_model_dir}")
        return None, 0
    expected_model_so_path = os.path.join(absolute_compiled_model_dir, "model.so")
    if not os.path.exists(expected_model_so_path):
        print(f"❌ Error: Expected compiled model 'model.so' not found in: {absolute_compiled_model_dir}")
        return None, 0
    # --- End Basic Checks ---

    # Create the output directory for this specific run
    os.makedirs(absolute_output_dir, exist_ok=True)

    # --- Prepare Input Files and Loader Script ---
    loader_script_path = os.path.join(absolute_output_dir, "_loader.py")
    loader_script_content = """
# Generated loader script for RunONNXModel.py --load-ref-from-numpy
import numpy as np
import os

inputs = []
i = 0
while True:
    input_npy_path = os.path.join(os.path.dirname(__file__), f"input_{i}.npy")
    if os.path.exists(input_npy_path):
        try:
            inputs.append(np.load(input_npy_path))
            i += 1
        except Exception as e:
            print(f"Error loading {input_npy_path}: {e}")
            # Decide how to handle load errors, e.g., raise or break
            break
    else:
        break

# Optional: Define outputs = [] if needed for verification later
# outputs = []
"""
    try:
        # Save the input numpy arrays
        for i, data in enumerate(input_data_list):
            input_path_abs = os.path.join(absolute_output_dir, f"input_{i}.npy")
            np.save(input_path_abs, data)
            # print(f"   Saved input {i} to {input_path_abs}") # Optional debug print

        # Save the loader script
        with open(loader_script_path, "w") as f:
            f.write(loader_script_content)
        # print(f"   Saved loader script to {loader_script_path}") # Optional debug print

    except Exception as e:
        print(f"❌ Error preparing input files/loader script in {absolute_output_dir}: {e}")
        return None, 0
    # --- End Input Preparation ---


    # Construct the command for RunONNXModel.py
    run_command = [
        sys.executable,
        absolute_run_script_path,
        "--load-model", absolute_compiled_model_dir, # Pass the DIRECTORY containing model.so
        "--save-ref", absolute_output_dir,           # Save outputs to this run's dir
        "--load-ref-from-numpy", loader_script_path  # Use the loader script
    ]

    print(f"Running inference command: {' '.join(run_command)}")
    # Set the working directory for the subprocess to the run's output directory.
    # This is where --save-ref will save files and where _loader.py will look for input_*.npy
    print(f"  Working directory for subprocess: {absolute_output_dir}")
    # Run the script with cwd set to the specified output directory
    env = os.environ.copy()
    if "ONNX_MLIR_HOME" not in env:
            print("⚠️ Warning: ONNX_MLIR_HOME environment variable not found. RunONNXModel.py might fail.")
            # Consider adding ONNX_MLIR_HOME if known and missing? For now, just warn.
    start_time = time.time()
    result = subprocess.run(run_command, check=True, capture_output=True, text=True, cwd=absolute_output_dir, timeout=60, env=env)
    end_time = time.time()
    # print("RunONNXModel Output:\n", result.stdout) # Often empty on success
    # print("RunONNXModel Stderr:\n", result.stderr) # Check stderr for potential info

    # --- Load output files from the specified output directory ---
    loaded_outputs = []
    i = 0
    while True:
        # Look for output files in the absolute_output_dir (where --save-ref saved them)
        # RunONNXModel.py saves outputs as output_0.pb, output_1.pb etc. with --save-ref
        output_path = os.path.join(absolute_output_dir, f"output_{i}.pb") # <-- Changed extension to .pb

        if os.path.exists(output_path):
            try:
                # Load the protobuf tensor
                output_ts = onnx.TensorProto()
                with open(output_path, "rb") as f:
                    output_ts.ParseFromString(f.read())
                # Convert to numpy array
                output_np = onnx.numpy_helper.to_array(output_ts)
                loaded_outputs.append(output_np)
                print(f"   ✅ Loaded output file: {output_path}")
                i += 1
            except Exception as load_err: # Use more specific exception if possible
                print(f"   ⚠️ Error loading output file {output_path}: {load_err}")
                loaded_outputs = None # Mark as failed if loading fails
                break
        else:
            # Stop if output_i.pb is not found.
            # If i is 0, it means no output files (output_0.pb) were found at all.
            break # Exit the while loop

    if not loaded_outputs:
            # This warning now means the script ran successfully but didn't produce
            # output_0.pb in the specified directory.
            print(f"⚠️ Warning: No valid output files (e.g., output_0.pb) found or loaded from directory: {absolute_output_dir}")
            # Print stdout/stderr from the script to help debug why it didn't save output
            print("RunONNXModel stdout:\n", result.stdout)
            print("RunONNXModel stderr:\n", result.stderr)
            # exit message and exit program
            print("❌ Exiting due to missing output files.")
            exit(1)

    
    elapsed_time = end_time - start_time
    return loaded_outputs, elapsed_time

##################################################################################
# MAIN PROGRAM ###################################################################
##################################################################################

"""
Main execution block for the ONNX-MLIR benchmark script.
"""

if __name__ == "__main__":

    #########################################
    # SETUP OUTPUT DIRECTORY ################
    #########################################
    # Create a base directory for all inference runs in the current working directory
    abs_inference_base_dir = os.path.abspath(INFERENCE_TEMP_DIR_BASE)
    if os.path.exists(abs_inference_base_dir):
        print(f"Removing existing base inference output directory: {abs_inference_base_dir}")
        shutil.rmtree(abs_inference_base_dir)
    os.makedirs(abs_inference_base_dir, exist_ok=True)
    print(f"Created base directory for inference outputs: {abs_inference_base_dir}")


    #########################################
    # CHECK PREREQUISITES ###################
    #########################################
    print("Checking prerequisites...")
    onnx_mlir_executable = find_executable("onnx-mlir", ONNX_MLIR_PATH)
    absolute_run_script = find_script("RunONNXModel.py", RUN_ONNX_MODEL_SCRIPT_PATH)

    # No need to check absolute_run_script again, find_script exits if not found
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Input ONNX model not found at '{MODEL_PATH}'. Please set the MODEL_PATH constant.")
        sys.exit(1)
    print("✅ Prerequisites check passed.")

    #########################################
    # COMPILE MODEL #########################
    #########################################
    # compile_onnx_model now returns the absolute path to the *directory* containing model.so
    compilation_success, absolute_compiled_model_dir = compile_onnx_model(
        MODEL_PATH,
        COMPILED_MODEL_DIR, # Pass the desired output directory name
        onnx_mlir_executable
    )
    if not compilation_success:
        print("❌ Exiting due to compilation failure.")
        sys.exit(1)
    print(f"✅ Model compiled successfully. Library 'model.so' is in: {absolute_compiled_model_dir}")


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
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False) # Use shuffle=False for consistency
    print("✅ Dataset loaded.")

    #########################################
    # BENCHMARK PREPARATION #################
    #########################################
    num_samples_to_load = min(len(test_data), NUM_ITERATIONS + WARMUP_ITERATIONS)
    print(f"Preloading {num_samples_to_load} test samples for benchmarking...")
    test_samples = []
    test_loader_iter = iter(test_loader)

    for i in range(num_samples_to_load):
        try:
            img, label = next(test_loader_iter)
        except StopIteration:
            print("Warning: Reached end of dataset during preloading.")
            break # Stop preloading if dataset ends early

        # Prepare input as expected by the model (flattened float32 numpy array)
        img_np = img.reshape(BATCH_SIZE, FEATURE_DIM).numpy().astype(np.float32)
        label_np = label.numpy() # Keep label as numpy scalar/array
        test_samples.append((img_np, label_np))

        if i == 0:
            print(f"Sample input numpy shape: {img_np.shape}, dtype: {img_np.dtype}")

    actual_warmup_iters = min(WARMUP_ITERATIONS, len(test_samples))
    actual_bench_iters = min(NUM_ITERATIONS, len(test_samples) - actual_warmup_iters)

    if len(test_samples) < WARMUP_ITERATIONS + NUM_ITERATIONS:
         print(f"⚠️ Warning: Loaded only {len(test_samples)} samples, less than requested {WARMUP_ITERATIONS + NUM_ITERATIONS}.")
         print(f"Adjusting: Warmup={actual_warmup_iters}, Benchmark={actual_bench_iters}")

    if actual_bench_iters <= 0 and actual_warmup_iters <=0:
        print("❌ Error: No samples loaded for benchmarking or warmup. Exiting.")
        sys.exit(1)

    print(f"✅ Preloaded {len(test_samples)} samples.")

    #########################################
    # BENCHMARKING ##########################
    #########################################
    print(f"\nRunning benchmark using '{absolute_run_script}'...")
    total_time = 0
    correct_predictions = 0
    inference_failed = False

    # --- Warmup Phase ---
    if actual_warmup_iters > 0:
        print(f"Starting {actual_warmup_iters} warmup iterations...")
        for i in range(actual_warmup_iters):
            img_np, _ = test_samples[i]
            # Create a unique subdirectory for this warmup run
            run_output_dir = os.path.join(abs_inference_base_dir, f"warmup_{i}")
            # Pass the absolute path to the directory containing model.so
            outputs, _ = run_inference_with_script( # Ignore time for warmup
                absolute_run_script, absolute_compiled_model_dir, [img_np], run_output_dir
            )
            # Check for failure even in warmup
            if outputs is None:
                 print(f"❌ Warmup inference failed for sample {i}. Stopping.")
                 inference_failed = True
                 break
        if not inference_failed:
            print(f"✅ Completed {actual_warmup_iters} warmup iterations.")
    else:
        print("Skipping warmup phase (0 iterations).")


    # --- Benchmarking Phase ---
    if not inference_failed and actual_bench_iters > 0:
        print(f"Starting {actual_bench_iters} benchmarking iterations...")
        start_index = actual_warmup_iters
        for i in range(actual_bench_iters):
            sample_index = start_index + i
            img_np, label_np = test_samples[sample_index]

            # Create a unique subdirectory for this benchmark run
            run_output_dir = os.path.join(abs_inference_base_dir, f"run_{i}")

            # Time the inference script call, passing the absolute path to the model directory
            outputs, elapsed_time = run_inference_with_script(
                absolute_run_script, absolute_compiled_model_dir, [img_np], run_output_dir
            )

            if outputs is None:
                print(f"❌ Inference failed for sample {sample_index}. Stopping benchmark.")
                inference_failed = True
                break # Stop benchmarking loop on failure

            total_time += elapsed_time

            # Process prediction (outside timing loop)
            # Check if outputs is a non-empty list containing at least one numpy array
            if outputs and isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], np.ndarray):
                # Assuming the first output contains the logits
                try:
                    pred = np.argmax(outputs[0], axis=1) # Get predicted class index
                    # Compare prediction with the ground truth label
                    if pred == label_np: # Assumes label_np is a scalar or 1-element array
                        correct_predictions += 1
                except IndexError:
                     # Handle cases where argmax might fail (e.g., unexpected output shape)
                     print(f"⚠️ Error processing output for sample {sample_index}. Output shape: {outputs[0].shape}")
            else:
                 # This case is hit if run_inference_with_script returned an empty list or None
                 # (though the None case should have been caught earlier)
                 print(f"⚠️ No valid output data loaded for sample {sample_index}, cannot check accuracy.")

        if not inference_failed:
            print(f"✅ Completed {actual_bench_iters} benchmarking iterations.")
    elif not inference_failed:
        print("Skipping benchmarking phase (0 iterations).")


    #########################################
    # RESULTS REPORTING #####################
    #########################################
    print("\n======= ONNX-MLIR BENCHMARK RESULTS (via RunONNXModel.py) =======")
    print(f"NOTE: Timing includes subprocess overhead for each inference call.")
    if inference_failed:
        print("Benchmark stopped early due to inference failure.")
    elif actual_bench_iters > 0:
        avg_time_ms = (total_time / actual_bench_iters) * 1000
        # Ensure accuracy calculation avoids division by zero if correct_predictions is somehow > 0 but actual_bench_iters is 0
        accuracy = (correct_predictions / actual_bench_iters) * 100 if actual_bench_iters > 0 else 0
        # Ensure throughput calculation avoids division by zero
        throughput = actual_bench_iters / total_time if total_time > 0 else 0

        print(f"Compiled Model Directory: {absolute_compiled_model_dir}")
        print(f"Inference Script: {absolute_run_script}")
        print(f"Total Samples Benchmarked: {actual_bench_iters}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Total Inference Script Time: {total_time:.3f} s")
        print(f"Avg. Inference Script Time: {avg_time_ms:.3f} ms/inference")
        print(f"Throughput: {throughput:.2f} inferences/second (including overhead of running external file)")
        print(f"Input/Output files stored under: {abs_inference_base_dir}")
    else:
        print("No benchmark iterations were successfully run.")

    #########################################
    # CLEANUP (Optional) ####################
    #########################################
    # Keep the INFERENCE_TEMP_DIR_BASE and COMPILED_MODEL_DIR for inspection by default.
    # Uncomment below to clean up.
    # print(f"\nCleaning up inference run directory: {abs_inference_base_dir}")
    # if os.path.exists(abs_inference_base_dir):
    #     shutil.rmtree(abs_inference_base_dir)

    # print(f"Cleaning up compiled model directory: {absolute_compiled_model_dir}")
    # if os.path.exists(absolute_compiled_model_dir):
    #     shutil.rmtree(absolute_compiled_model_dir)

    print("\nScript finished.")