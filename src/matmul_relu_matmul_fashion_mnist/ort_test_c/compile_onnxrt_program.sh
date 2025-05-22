#!/bin/bash
# filename: compile_onnxrt_program.sh
# This script compiles a C or C++ source file using the ONNX Runtime library.
# C source file example: fused_gemm_test.c


# Check if the user provided a file name
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <source_file>"
    exit 1
fi

# Extract the file name and extension
SOURCE_FILE="$1"
EXTENSION="${SOURCE_FILE##*.}"
OUTPUT_FILE="${SOURCE_FILE%.*}"

# Determine the compiler based on the file extension
if [ "$EXTENSION" == "c" ]; then
    COMPILER="gcc"
elif [ "$EXTENSION" == "cpp" ] || [ "$EXTENSION" == "cxx" ]; then
    COMPILER="g++"
else
    echo "Unsupported file extension: .$EXTENSION"
    echo "Supported extensions are .c (C) and .cpp/.cxx (C++)"
    exit 1
fi

# Compile the source file
$COMPILER -I/workdir/onnxruntime-linux-x64-gpu-1.21.1/include \
    -L/workdir/onnxruntime-linux-x64-gpu-1.21.1/lib \
    -o "$OUTPUT_FILE" "$SOURCE_FILE" \
    -lonnxruntime -std=c11 -Wall -Wextra -g \
    -Wl,-rpath,/workdir/onnxruntime-linux-x64-gpu-1.21.1/lib

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Output binary: $OUTPUT_FILE"
else
    echo "Compilation failed."
    exit 1
fi