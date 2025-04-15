[./BuildONNX.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Installing `third_party ONNX` for Backend Tests or Rebuilding ONNX Operations

Backend tests are triggered by `make check-onnx-backend` in the build directory and require a few preliminary steps to run successfully. Similarly, rebuilding the ONNX operations in ONNX-MLIR from their ONNX descriptions is triggered by `make OMONNXOpsIncTranslation`.

You will need to install python 3.x if its not default in your environment, and possibly set the cmake `PYTHON_EXECUTABLE` variable in your top cmake file.

You will also need `pybind11` which may need to be installed (mac: `brew install pybind11` or linux: `apt -y install python3-pybind11` for example) and you may need to indicate where to find the software (Mac, POWER, possibly other platforms: `export pybind11_DIR=<your path to pybind>`). Then install the `third_party/onnx` software (Mac: `pip install third_party/onnx`) typed in the top directory.

 ## Upgrading ONNX in ONNX-MLIR

Here are the steps taken to upgrade the ONNX version:

1.	Create your own branch

2.	"cd" into `third_party/onnx` and checkout the commit for the latest version of onnx (You can find the latest commit here: https://github.com/onnx/onnx/releases)

3.	"pip uninstall onnx" (remove older version)

4.	In `onnx-mlir/` directory, "pip install third_party/onnx" (install onnx from the commit and not online version)

5.	Update `utils/gen_onnx_mlir.py` file with the correct version number

6.	Build onnx in the `build/` directory using: set CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON

7.	Run in the `build/` directory : "make OMONNXOpsIncTranslation" 

8.	Run in `build/` directory : "make onnx-mlir-docs"

9.	Run in `build/` directory : "make check-onnx-backend-case"

10.	Update the [new backend tests](https://github.com/onnx/onnx-mlir/blob/main/test/backend/all_test_names.txt) based on the results from `step 9`

11.	Update the [Opset documentation for cpu](https://github.com/onnx/onnx-mlir/blob/main/test/backend/inference_backend.py) and then issue the following command in the `build/` directory: "make onnx_mlir_supported_ops_cpu"

12.	Update the [Opset documentation for NNPA](https://github.com/onnx/onnx-mlir/blob/main/test/backend/inference_backend.py) and then issue the following command in the `build/` directory: "make onnx_mlir_supported_ops_NNPA"

13.	Ensure the lit tests and backend tests pass successfully and then you are done!


**Note: Please use `git add <filename>` for files that might have been changed before doing a PR.** 

## Known issues

On Macs/POWER and possibly other platforms, there is currently an issue that arises when installing ONNX. If you get an error during the build, try a fix where you edit the top CMakefile as reported in this PR: `https://github.com/onnx/onnx/pull/2482/files`.

While running `make check-onnx-backend` on a Mac you might encounter the following error:

```shell
Fatal Python error: Aborted

Current thread 0x0000000107919e00 (most recent call first):
  File "/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 2632 in getproxies_macosx_sysconf
  File "/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 2650 in getproxies
  File "/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 795 in __init__
  ...
 ```

 A known workaround is to export the `no_proxy` environment variable in your shell as follow, and rerun the tests.

 ```shell
 % export no_proxy="*"
 ```
 


[./ConstPropagationPass.md]:

# Constant Propagation for ONNX operations

This document describes `--constprop-onnx` pass which is used to do
constant propagation for operations in the ONNX dialect.

[source
code](https://github.com/onnx/onnx-mlir/blob/main/src/Transform/ONNX/ConstProp.td).

## Example
Given the following code:
```mlir
func @foo() -> tensor<1xf32> {
  %0 = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %1 = "onnx.Constant"() {value = dense<[2.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1xf32> , tensor<1xf32>) -> tensor<1xf32>
  %3 = "onnx.Constant"() {value = dense<[3.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %4 = "onnx.Add"(%2, %3) : (tensor<1xf32> , tensor<1xf32>) -> tensor<1xf32>
  "std.return"(%4) : (tensor<1xf32>) -> ()
}
```

If we call `onnx-mlir-op --constprop-onnx`, we will get:
```mlir
func @foo() -> tensor<1xf32> {
  %0 = "onnx.Constant"() {value = dense<[6.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  "std.return"(%0) : (tensor<1xf32>) -> ()
}
```

## Remark

ONNXConstantOp uses MLIR DenseElementsAttr to store constant values. It is
important to note that, once a DenseElementsAttr is created, it is alive and
consumes memory until the end of compilation. In [Example](#example), all the
three DenseElementsAttrs in the three ONNXConstantOps exist until the end of
compilation. Especially, two intermediate DenseElementsAttrs in the two
ONNXConstantOps produced by folding the two ONNXAddOps also exist. For a
real world model, the number of intermediate DenseElementsAttrs will increase
quickly, which leads to a large memory footprint during compilation.

To avoid creating too many DenseElementsAttrs for intermediate ONNXConstantOps
during `--constprop-onnx`, we designed a mechanism that dynamically allocates and
deallocates buffers for intermediate ONNXConstantOps and only creates
DenseElementsAttr after constant propagation and other ONNX dialect passes,
just before lowering to Krnl (or any other target dialect).

This is accomplished with a custom attribute DisposableElementsAttr which
acts as a substitute for DenseElementsAttr for the common case of
non-complex scalar element types: bool and integer and floating point types.
DisposableElementsAttr implements the same ElementsAttr interface as
DenseElementsAttr and in most cases they are functionally identical and
the surrounding code doesn't need to distinguish. It just needs to use the
OnnxElementsAttrBuilder class and ElementsAttrHelper functions to
construct and access ElementsAttr instances to reap the the memory footprint
and performance benefits.

The deallocation of DisposableElementsAttr buffers happens between compiler
passes in DisposableGarbageCollector, which is run by the PassManager
between "module" passes (which are guaranteed to "stop the world" with no
other passes executing in parallel) as an "instrumentation".

DisposableElementsAttr offers other memory and speed benefits which are
outlined in the comments in the class source file and are
explained in the presentation from November 2022, linked from the
[meeting wiki page](https://github.com/onnx/onnx-mlir/wiki/Informal-meeting-agenda-and-notes#nov-29th).

## Write rules for constant propagation

We use MLIR declarative rewriting rules (DRR) to write patterns for constant
propagation. The DRR definition used for defining patterns is shown below:
```
class Pattern<
   dag sourcePattern,
   list<dag> resultPatterns,
   list<dag> additionalConstraints = [],
   list<dag> supplementalPatterns = [],
   dag benefitsAdded = (addBenefit 0)
>;
```

More information about DRR can be found [here](https://mlir.llvm.org/docs/DeclarativeRewrites/).

Now, we go through a simple example that adds constant propagation for ONNXAddOp.

### Step 1: Write DRR patterns <a id="step1"></a>

We first add a pattern to
[ConstProp.td](https://github.com/onnx/onnx-mlir/blob/main/src/Transform/ONNX/ConstProp.td).

```mlir
// Constant Propagation for Add
def AddConstProp : Pat<
    // source patten: From add(lhs, rhs).
    (ONNXAddOp:$addOp (ONNXConstantOp:$lhs $_, $_, $_, $_, $_, $_, $_, $_),
                      (ONNXConstantOp:$rhs $_, $_, $_, $_, $_, $_, $_, $_)),
    // result pattern: To c = lhs + rhs
    (CreateAddOfTwoConst $addOp, $lhs, $rhs),
    // Additional constraints: if both lhs and rhs are dense constants.
    [(IsFromDenseONNXConstantOp:$lhs), (IsFromDenseONNXConstantOp:$rhs)]>;
```

The above pattern will replace an ONNXAddOp whose inputs are constants
by a new constant by adding the inputs at compile time. To check if an input is
a constant, using ONNXConstantOp is not enough since the constant tensor can be
sparse and we now support dense constant tensors only. We need additionallly
check a dense constant tensor by using `IsFromDenseONNXConstantOp`.

In the result pattern, to produce a ONNXConstantOp, we will add `lhs`
and `rhs` at compile time, and emit an ONNXConstantOp. To minimize the
memory footprint, this ONNXConstantOp has a DisposableElementsAttr instead of a conventional DenseElementsAttr.

Function `CreateAddOfTwoConst` will do the addition at compile time and return
an ONNXConstantOp.

```
def CreateAddOfTwoConst :
   NativeCodeCall<"ConstPropElementwiseBinary<mlir::ONNXAddOp>($_builder, $0, $1, $2)">;
```

### Step 2: Prepare array buffers for inputs and result <a id="step2"></a>

Function `CreateAddOfTwoConst` in the pattern calls
`ConstPropElementwiseBinary` in [ConstProp.cpp](https://github.com/onnx/onnx-mlir/blob/main/src/Transform/ONNX/ConstProp.cpp) whose content is as follows.

```c++
template <typename ElementwiseBinaryOp>
Value ConstPropElementwiseBinary(PatternRewriter &rewriter,
    Value replacingValue, Value lhsValue, Value rhsValue) {
  ConstPropCounters::count("ElementwiseBinary", {lhsValue, rhsValue});
  Type replacingType = mlir::cast<ShapedType>(replacingValue.getType());

  // Get lhs and rhs ElementsAttr from the values' defining constant ops.
  ElementsAttr lhs = getConstValueElements(lhsValue);
  ElementsAttr rhs = getConstValueElements(rhsValue);

  Type operandsElemType = lhs.getElementType();
  assert(operandsElemType == rhs.getElementType() &&
         "all element-wise binary ops have matching operands element types");
  OnnxElementsAttrBuilder elementsBuilder(rewriter.getContext());
  ElementsAttr resultElements = elementsBuilder.combine(lhs, rhs, replacingType,
      combinerOfElementwiseBinaryOp<ElementwiseBinaryOp>(operandsElemType));

  // Construct and return a new ONNXConstantOp with the resultElements attribute.
  return createReplacingConstantOp(rewriter, replacingValue, resultElements)
      .getResult();
}
```
where `OnnxElementsAttrBuilder.combine(...)` broadcasts the lhs and rhs elements,
as needed, and constructs a new (Disposable) ElementsAttr whose elements are the
result of element-wise application of the binary function
`combinerOfElementwiseBinaryOp<ElementwiseBinaryOp>(operandsElemType)`
which maps the ElementwiseBinaryOp ONNX op to a c++ operator.

### TODO: Describe how to add OnnxElementsAttrBuilder builder methods for new ops

For more information about constant propagation, please see [ConstProp.td](https://github.com/onnx/onnx-mlir/blob/main/src/Transform/ONNX/ConstProp.td)
and
[ConstProp.cpp](https://github.com/onnx/onnx-mlir/blob/main/src/Transform/ONNX/ConstProp.cpp).


[./SupportedONNXOps-NNPA.md]:

<!--- Automatically generated, do not edit. -->
<!--- To update, run `make onnx_mlir_supported_ops_NNPA' -->

# Supported ONNX Operation for Target *NNPA*.

Onnx-mlir currently supports ONNX operations targeting up to opset 22. Limitations are listed when applicable. This documentation highlights the minimum and maximum opset versions that are fully supported by onnx-mlir and not the version changes.

* Operations are defined by the [ONNX Standard](https://github.com/onnx/onnx/blob/main/docs/Operators.md).
* **Supported Opsets** indicates the lowest and highest opset a model may have for onnx-mlir to support compiling a model with the operator.
   * A * indicates onnx-mlir is compatible with the latest version of that operator available as of opset 22.
   * A ^ indicates onnx-mlir is compatible with the latest level of the NNPA Architecture which is z16.


NNPA has hardware limitations in dimension index size and tensor size, which are described in [NNPALimit.hpp](../src/Accelerators/NNPA/Support/NNPALimit.hpp). They are large enough for normal use cases, but if your model exceeds the limitations, CPU is used instead of NNPA. NNPA currently only support DLFLOAT16 as its data type. Common data formats like FP32, FP16, BFLOAT need to undergo data conversions to the NNPA internal format DLFLOAT16. Hence ONNX ops which updated their tensors to BFLOAT16 will not be natively supported on NNPA.  Onnx-mlir with NNPA utilizes hardware when possible. To accomplish this, the compiler converts ONNX ops to [ZHigh](Dialects/zhigh.md) ops, [ZLow](Dialects/zlow.md) ops, and are processed by the [IBM Z Deep Neural Network Library (zDNN)](https://github.com/IBM/zDNN).


| Op |Supported Opsets (inclusive) |Minimum NNPA Level(Inclusive) |Limitations |Notes |
| --- |--- |--- |--- |--- |
| **Add** |6 - * |z16 |- Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **AveragePool** |6 - * |z16 |- `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- `ceil_mode` must be default value(0) <br>- Input and output tensors must be 4D tensors (N x C x H x W).<br>- `kernel_shape` must be static.<br>- `count_include_pad` must be default value(0).<br>- `ceil_mode` must be default value(0). | |
| **BatchNormalization** |6 - * |z16 |Input and output tensor must be 4D(N x C x H x W). | |
| **Conv** |6 - * |z16 |- `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- Dimension in Height and weight must be static.<br>- `group` must be default value(1).<br>- `dilations` must be default value(1).<br>- Input and output tensors must have 4D (N x C x H x W).<br>- `kernel_shape` must be static. | |
| **ConvTranspose** |6 - * |z16 |- 1D and 3D not supported because Conv1D and Conv3D not supported in zDNN. non-default `dilations` not supported because dilated convolution not supported in zDNN. | |
| **Div** |6 - * |z16 |- Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **Exp** |6 - * |z16 |Input tensor must have 4 dimensions. | |
| **GRU** |7 - * |z16 |- `direction` and `hidden_size` in `W` must have static dimensions.<br>- `R` must have static dimensions.<br>- If `B` and `initial_h` are given, they must have static dimensions.<br>- `sequence_lens` is not supported for bidirectional GRU.<br>- `activations` must be `["Sigmoid", "Tanh", "Tanh"]`.<br>- `clip` is not supported.<br>- `linear_before_reset` must be 1.<br>- `layout` is not supported. | |
| **Gemm** |6 - * |z16 |- `alpha` and `beta` must be default value(1).<br>- Rank of `C` must be 1 or 2. If the rank is 1, the dimension of `C` must be the same with the seconde dimension of `B`.<br>. | |
| **GlobalAveragePool** |6 - * |z16 |- Input shape must be 4D tensor(NCHW).<br>- Dimensions in `H` and `W` must be static. | |
| **LSTM** |7 - * |z16 |- `direction` and `hidden_size` in `W` must have static dimensions.<br>- `R` must have static dimensions.<br>- `B` and `initial_h` have static dimensions if given. `B`'s direction dim must be 1 or 2.<br>- `P`(peepholes), `activation_alpha`, and `activation_beta` are not supported.<br>- `activations` must be `["Sigmoid", "Tanh", "Tanh"]`.<br>- `clip` is not supported.<br>- `input_forget` must be default value(0).<br>- `layout` is not supported. | |
| **Log** |6 - * |z16 |Input tensor must have 4 dimensions. | |
| **LogSoftmax** |6 - * |z16 | | |
| **MatMul** |6 - * |z16 |Ranks of input tensors must be (Rank of A, Rank of B) = (M, N), where M >= 2 and N >= 2. | |
| **Max** |6 - * |z16 |- Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **MaxPool** |6 - * |z16 |- `auto_pad` must be `NOTSET`, `VALID`, and `SAME_UPPER`. If `NOTSET` is used, `pads` must be set so that the padding valid type or same upper.<br>- `ceil_mode` must be default value(0) <br>- Input and output tensors must be 4D tensors(N x C x H x W).<br>- `kernel_shape` must be static.<br>- `ceil_mode` must be default value(0).<br>- `dilations` must be default value(1). | |
| **Min** |6 - * |z16 |- Shape of input tensors must be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **Mul** |6 - * |z16 |- Shape of input tensors should be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **Pow** |7 - * |z16 |- Exponent should be a scalar integer and less or equal to 64. | |
| **ReduceMean** |6 - * |z16 |- `keepdims` must be 1.<br>- Input tensor must be 4D tensors and `axis` must be [2, 3]. | |
| **Relu** |6 - * |z16 |Input tensor must be less than or equal to 4 dimensions. | |
| **Sigmoid** |6 - * |z16 |Input tensor must be less than or equal to 4 dimensions. | |
| **Softmax** |6 - * |z16 |- `axis` must be the last dimension, i.e. `rank - 1` or -1. | |
| **Softplus** |6 - * |z16 |The operations immediately before and after the Softplus operation must be executed on the NNPA. Otherwise, Softplus is executed on the CPU. This limitation is set to avoid performance degradation. | |
| **Sub** |6 - * |z16 |- Shape of input tensors should be the same since broadcasting is not supported.<br>- Input tensors must have static dimensions. | |
| **Sum** |6 - * |z16 |- All inputs must have the same static shape (Broadcasting not supported.)<br>- Single input not supported. | |
| **Tanh** |6 - * |z16 |Input tensor must be less than or equal to 4 dimensions. | |


[./Docker.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Building and Developping ONNX-MLIR using Docker

There are three ways to use ONNX-MLIR with Docker.
1. [Using a prebuild image](#prebuilt-containers), recommended for using ONNX-MLIR but not developing it.
2. [Using a script](#easy-script-to-compile-a-model), recommended for testing our infrastructure quickly without explicitly installing a Docker image.
3. [Using a custom build image](#building-and-developping-onnx-mlir-using-docker), recommended for developing ONNX-MLIR.

## Prebuilt Images

An easy way to get started with ONNX-MLIR is to use a prebuilt Docker image.
These images are created as a result of a successful merge build on the trunk.
This means that the latest image represents the tip of the trunk.
Currently there are both Release and Debug mode images for `amd64`, `ppc64le` and `s390x` saved in Docker Hub as, respectively, [onnxmlir/onnx-mlir](https://github.com/users/onnxmlir/packages/container/onnx-mlir) and [onnxmlir/onnx-mlir-dev](https://github.com/users/onnxmlir/packages/container/onnx-mlir-dev).
To use one of these images either pull it directly from Docker Hub, launch a container and run an interactive bash shell in it, or use it as the base image in a Dockerfile.

Here are the differences between the two Docker images.
* The `onnx-mlir` image just contains the built compiler and you can use it immediately to compile your model without any installation. It does not include any support to run compiled model.
* The `onnx-mlir-dev` image contains the built compiler plus all of the tools and support needed for development, including support to run our tests locally. The image also support tools to run compiled models, such as support for our python interface.

## Easy Script to Compile a Model

A python convenience script is provided to allow you to run ONNX-MLIR inside a Docker container as if running the ONNX-MLIR compiler directly on the host.
The resulting output is an Linux ELF library implementing the ONNX model.
The `onnx-mlir.py` script is located in the [docker](../docker) directory. For example, compiling a MNIST model can be done as follows.
```
# docker/onnx-mlir.py -O3 --EmitLib mnist/model.onnx
505a5a6fb7d0: Pulling fs layer
505a5a6fb7d0: Verifying Checksum
505a5a6fb7d0: Download complete
505a5a6fb7d0: Pull complete
Shared library model.so has been compiled.
```

The script will pull the onnx-mlir image if it's not available locally, mount the directory containing the `model.onnx` into the container, and compile and generate the `model.so` in the same directory.

This script takes the same option as the normal  `onnx-mlir` command used to compile a ONNX model. Typical options are `-O0` (default) or `-O3` to define an optimization level and `--EmitLib` (default) or `--EmitJNI` to generate a dynamic library or a jar file.
A complete list of options is provided by using the traditional `--help` option.

This script generates codes that can be executed on a Linux system or within a Docker container.

## Building ONNX-MLIR in a docker environment

The onnx-mlir-dev image contains the full build tree including the prerequisites and a clone of the source code.
The source can be modified and `onnx-mlir` can be rebuilt from within the container, so it is possible to use it as a development environment.
New pull requests can be generated, and the repository can be updated to the latest using git commands.
It is also possible to attach vscode to the running container.
An example Dockerfile useful for development and vscode configuration files can be seen in the [docs/docker-example](docker-example) folder.
If the workspace directory and the vscode files are not present in the directory where the Docker build is run, then the lines referencing them should be commented out or deleted.

The Dockerfile is shown here, and should be modified according to one's need. The file below includes debugging tools as well as pytorch, which can be used to train the mnist model in our end-to-end example provided in the [docs/mnist_example](mnist_example) directory.

[same-as-file]: <> (docs/docker-example/Dockerfile)
```
FROM ghcr.io/onnxmlir/onnx-mlir-dev
WORKDIR /workdir
ENV HOME=/workdir

# 1) Install packages.
ENV PATH=$PATH:/workdir/bin
RUN apt-get update
RUN apt-get install -y python3-numpy
RUN apt-get install -y python3-pip
RUN python -m pip install --upgrade pip
RUN apt-get install -y gdb
RUN apt-get install -y lldb

# 2) Instal optional packages, comment/uncomment/add as you see fit.
RUN apt-get install -y vim
RUN apt-get install -y emacs
RUN apt-get install -y valgrind
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y clang-format
RUN python -m pip install wheel
RUN python -m pip install numpy
RUN python -m pip install torch==2.0.0+cpu torchvision==0.15.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone https://github.com/onnx/tutorials.git
# Install clang
RUN apt-get install -y lsb-release wget software-properties-common
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
# For development
RUN apt-get install -y ssh-client

# 3) When using vscode, copy your .vscode in the Dockerfile dir and
#    uncomment the two lines below.
# WORKDIR /workdir/.vscode
# ADD .vscode /workdir/.vscode

# 4) When using a personal workspace folder, set your workspace sub-directory
#    in the Dockerfile dir and uncomment the two lines below.
# WORKDIR /workdir/workspace
# ADD workspace /workdir/workspace

# 5) Fix git by reattaching head and making git see other branches than main.
WORKDIR /workdir/onnx-mlir
# Add optional personal fork and disable pushing to upstream (best practice).
# RUN git remote add origin https://github.com/<<user>>/onnx-mlir.git
# RUN git remote set-url --push upstream no_push

# 6) Set the PATH environment vars for make/debug mode. Replace Debug
#    with Release in the PATH below when using Release mode.
WORKDIR /workdir
ENV NPROC=4
ENV PATH=$PATH:/workdir/onnx-mlir/build/Debug/bin/:/workdir/onnx-mlir/build/Debug/lib:/workdir/llvm-project/build/bin
```

The first step is to copy the [docs/docker-example](docker-example) directory to another directory outside of the repo, say `~/DockerOnnxMlir`. Or simply download the `Dockerfile` and the `.vscode` file if you intend to use VSCode.

Then, the `Dockerfile` in the copied directory should then be modified to suit one's need. In particular, we recommend developers to use their own fork for development. Uncomment the lines associated with git (Step 5 in the file) and substitute the appropriate GitHub Id in the commented out directives. 
The lines associated with VSCode (Step 3 in the file) should be also uncommented when using VSCode. 
Finally, we recommend creating a subdirectory named `workspace` that contains test examples you would like to have in your Docker Image and Container. 
If so, uncomment the lines associated with copying a personal workspace folder (Step 4 in the file), and that subdirectory's content will be copied over to the Docker Image.

The next step is to create a Docker image. This step can be performed using the `docker build --tag imageName .` shell command. Once this command is successful, we must start a container. This can be done by a command line (e.g. `docker run -it imageName`) or by opening the Docker Dashboard, locating the Image Tab, and clicking the `run` button associated with the image just created (e.g. `imageName` above).

These steps are summarized here.
``` shell
# Starting in the onnx-mlir directory, copy the Docker example directory.
cp -prf docs/docker-example ~/DockerOnnxMlir
cd ~/DockerOnnxMlir
# Edit the Dockerfile.
vi Dockerfile
# Build the Docker image.
docker build --tag ghcr.io/onnxmlir/onnx-mlir-dev .
# Start a container using the Docker dashboard or a docker run command.
docker run -it ghcr.io/onnxmlir/onnx-mlir-dev
```

**NOTE:** If you are using a MacBook with the Apple M1 chip, please follow the steps below for configuration:
``` shell
# Starting in the onnx-mlir directory, copy the Docker example directory.
cp -prf docs/docker-example ~/DockerOnnxMlir
cd ~/DockerOnnxMlir
# Edit the Dockerfile.
vi Dockerfile
# Pull the Docker image with the specified platform
docker pull --platform linux/amd64 ghcr.io/onnxmlir/onnx-mlir-dev
# Build the Docker image.
docker build --platform linux/amd64 --tag ghcr.io/onnxmlir/onnx-mlir-dev .
# Start a container using the Docker dashboard or a docker run command.
docker run --platform linux/amd64 -it ghcr.io/onnxmlir/onnx-mlir-dev
```

Tip: Instead of adding the platform flag for every docker pull, build, and run command. You can set the environment variable `DOCKER_DEFAULT_PLATFORM` and use the first set of steps:
```
export DOCKER_DEFAULT_PLATFORM=linux/amd64
```

### Developing with Docker in VSCode

The next step is to open VSCode, load the Docker Extension if not already present, and then open the Docker tab on the left pane. Locate the container that was just started in the previous step, right click on it, and select the `Attach Visual Studio Code` option.
This will open a new VSCode window. Open a local folder on the `workdir` directory, this will give you access to all of the ONNX/MLIR/LLVM code as well as the `workspace` subdirectory.

You may then open a shell, go to the `onnx-mlir` subdirectory, and check that all of the git is properly setup.

If you opted to add your own fork, it will be listed under `origin` with `upstream` being the official ONNX-MLIR repo. For example:
``` shell
git remote -v
#origin   https://github.com/AlexandreEichenberger/onnx-mlir.git (fetch)
#origin   https://github.com/AlexandreEichenberger/onnx-mlir.git (push)
#upstream https://github.com/onnx/onnx-mlir.git (fetch)
#upstream no_push (push)
```

Now, you may fetch your own branches using `git fetch origin`, and switch to one of your branch (say `my-opt`) using the `git checkout --track origin/my-opt` command. The `--track` option is recommended as `upstream` was cloned and `origin` was added as remote. Once you want to push your changes, you should use `git push -u origin my-opt`, using the `-u` option to link the local branch with the `origin` remote repo.

The `main` branch will default to the upstream repo. If you prefer it to be associated with your own fork's `main` branch, you may update your main branch to the latest and associate the local main branch with `origin` using the commands listed below.
``` shell
git checkout main
git branch --unset-upstream
git push --set-upstream origin main
```

A Docker container can be used to investigate a bug, or to develop a new feature. Some like to create a new images for each new version of ONNX-MLIR; others prefer to create one image and use git to update the main branch and use git to switch between multiple branches. Both are valid approaches.

## Using a devcontainer
Another way of building onnx-mlir for development in VSCode is using a devcontainer. This way you only mount your source folder, meaning that changes you do are saved on your local machine. For this setup to work you need a `Dockerfile` and a `devcontainer.json` file. Both are provided in `docs/devcontainer-example`. 

The [`Dockerfile`](devcontainer-example/Dockerfile.llvm-project) is a simple Dockerfile based on the precompiled LLVM/MLIR image that is shared. It installs additional software that is useful for developing and also sets `LLVM_PROJECT_ROOT` to easily refer to the LLVM path.


The [`devcontainer.json`](devcontainer-example/devcontainer.json) preinstalls extensions and defines settings for the VS Code server running inside the container. This way you don't have to setup VS Code everytime you enter the container. In `postAttachCommand` ONNX is installed.

To use this setup you first clone onnx-mlir and all submodules (for example with` git clone --recursive https://github.com/onnx/onnx-mlir.git`). You then create a new folder named `.devcontainer` in the source root. After that you copy the two files in `docs/devcontainer-example` into that folder. Now simply press `CTRL+SHIFT+P` and execute `Dev Containers: Reopen in Container`. VSCode will now create the docker image and mount the source folder.

You can now configure onnx-mlir as described in [BuildOnLinuxOSX](BuildOnLinuxOSX.md). `MLIR_DIR` is already set for you, so you can skip that step.

**Note:** To run this on M1/2 Macs something like Rosetta is needed. This is related to https://github.com/docker/roadmap/issues/384


[./Prerequisite.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Getting the prerequisite software

<!-- Keep list below in sync with README.md. -->
```
python >= 3.8
gcc >= 6.4
protobuf >= 4.21.12
cmake >= 3.13.4
make >= 4.2.1 or ninja >= 1.10.2
java >= 1.11 (optional)
```

Onnx-mlir is tested to work with python 3.8 and 3.9 but not yet fully tested with 3.10+. It may work with older 3.x python versions but not recommended since those either have already reached or are close to reach their EOL. To check the python version, run `python --version`.

GCC can be installed with distro specific package manager on Linux such as yum on RHEL/Fedora, apt on Debian/Ubuntu, or brew on MacOS. If you prefer to compile gcc yourself, instructions can be found [here](https://gcc.gnu.org/install/). To check the gcc version, run `gcc --version`.

Protobuf can be installed with brew on MacOS. Prebuilt protobuf packages on most Linux distros do not meet the required level. You can download its binary releases or compile it yourself. The instructions can be found [here](https://github.com/protocolbuffers/protobuf). To check the protobuf version, run `protoc --version`.

Cmake can be installed with distro specific package manager on Linux such as yum on RHEL/Fedora, apt on Debian/Ubuntu, or brew on MacOS. If you prefer to compile cmake yoursellf, instructions can be found [here](https://cmake.org/install/). However, to use Cmake, you need to follow the "How to Install For Command Line Use" tutorial, which can be found in Cmake under Tools>How to Install For Command Line Use. To check the cmake version, you can either look in the desktop version under CMake>About, or run `cmake --version`.

GNU make can be installed with distro specific package manager on Linux such as yum on RHEL/Fedora, apt on Debian/Ubuntu, or brew on MacOS. If you prefer to compile make yourself, instructions can be found [here](http://git.savannah.gnu.org/cgit/make.git/tree/README.git). To check the make version, run `make --version`.

Ninja can be installed with apt on Debian/Ubuntu Linux, or brew on MacOS. On RHEL/Fedora Linux, or if you want to compile ninja yourself, the instructions can be found [here](https://ninja-build.org/). To check the ninja version, run `ninja --version`.

Java SDK can be installed with distro specific package manager on Linux such as yum on RHEL/Fedora, apt on Debian/Ubuntu, or brew on MacOS. Java SDK is only required if you plan to use the onnx-mlir `--EmitJNI` option to compile a model into a jar file for use in a Java environment. Note that the jar file contains native model runtime library called through JNI so it is not portable across different architectures. To check the java version, run `java --version`.

All the `PyPi` package dependencies and their appropriate versions are captured in [requirements.txt](requirements.txt).


[./DocumentList.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Index of documents
This document serves as an index for onnx-mlir documents.

# Supported ONNX Ops
* CPU support is covered [here](SupportedONNXOps-cpu.md).
* NNPA support is covered [here](SupportedONNXOps-NNPA.md).

# Working environment
* Installation is covered by [README.md](../README.md).
* [Workflow.md](Workflow.md) describes how to contribute in github environment.
* [This guideline](Documentation.md) is used to keep documentation and code consistent.
* [UpdatingLLVMCommit.md](UpdatingLLVMCommit.md) describes how to update the commit of LLVM that ONNX-MLIR depends on.

# Development
* Onnx operation are represented with  [ONNX dialect](Dialects/onnx.md) in onnx-mlir.
* This [document](ImportONNXDefs.md#add_operation)
tell you how to generate an ONNX operation into ONNX dialect.
* After an ONNX model is imported into onnx-mlir, several graph-level transformations will be applied.
These transformations include operation decomposition, [constant propagation](ConstPropagationPass.md),
shape inference, and canonicalization. 
* Then the ONNX dialect is [lowered to Krnl dialect](LoweringCode.md). 
To help debugging and performance tuning, onnx-mlir supports [instrumentation](Instrumentation.md)
at the ONNX operand level.
* All the passes may be controlled with [options](Options.md).
* How to handle errors can be found [here](ErrorHandling.md).
* How to support a new accelerator can be found [here](AddCustomAccelerators.md).
* How to analyze unknown dimensions and query their equality at compile time can be found [here](DynamicDimensionAnalysis.md).
* A Jenkins monitor job was setup to help with updating LLVM commit. It locates the next commit we can update to without breaking ONNX-MLIR, as well as the commit that will break ONNX-MLIR. You can see the commit(s) here: [s390x](https://www.onnxmlir.xyz/jenkins/job/LLVM-Watch-Docker-Build/LLVM_20Watch_20Report/), [ppc64le](https://www.onnxmlir.xyz/jenkinp/job/LLVM-Watch-Docker-Build/LLVM_20Watch_20Report/), [amd64](https://www.onnxmlir.xyz/jenkinx/job/LLVM-Watch-Docker-Build/LLVM_20Watch_20Report/).

[#](#) Execution
The compiled ONNX model can be executed with either a
[C/C++ driver](mnist_example/README.md#write-a-c-driver-code)
[python driver](mnist_example/README.md#write-a-python-driver-code). or a
[java driver](mnist_example/README.md#write-a-java-driver-code).
The routine testing for onnx-mlir build is describe in this [document](Testing.md).


[./Options.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Define and Use Command-line Options for ONNX-MLIR

Command-line options can be used to alter the default behavior of onnx-mlir, or onnx-mlir-opt, and help user experimenting, debugging or performance tuning. We implemented command-line in ONNX-MLIR based on the command-line utility provided by LLVM. We did not define `Option` or `ListOption` with MLIR pass classes(see discussion). 
 
## Organize Options
Refer [llvm document](https://llvm.org/docs/CommandLine.html) for basic idea of how to define an option. In ONNX-MLIR, options are put into groups (`llvm::cl::OptionCategory`). All command-line options for onnx-mlir are in the `OnnxMlirOptions` group.

## Code structure
Command-line options should be placed in `src/Compiler/CompilerOptions.cpp` and declared in `src/Compiler/CompilerOptions.hpp`.

## Define an option
- Add a declaration of the option in `src/Compiler/CompilerOptions.hpp`
- In `src/Compiler/CompilerOptions.cpp`, define the option
- Do **not** include `src/Compiler/CompilerOptions.hpp` in new source files; it should only be used in the onnx-mlir and onnn-mlir-opt command-line tools.
- Do create 'Pass Options' to pass information to specific passses and transformations

## Define an option local to a transformation
Use MLIR's Pass Options to configure passes.


[./ImportONNXDefs.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->
# Import ONNX Definitions and Support Operations

# Table of Contents
  1. [Overview](#overview)
  2. [Add an Operation](#add_operation)
  3. [Customize an Operation](#customize)
  4. [Build](#build)
  5. [Details about version](#version)
# Overview <a name="overview"></a>
ONNX-MLIR defines an ONNX dialect to represent operations specified by ONNX.The ONNX dialect is created with MLIR table
gen tool. The definition of each operation is transferred from ONNX automatically with a
 python script,
[utils/gen_onnx_mlir.py](../utils/gen_onnx_mlir.py).
This script retrieves operation definition from
 ONNX package to generate ONNXOps.td.inc for dialect table gen and OpBuilderTable.inc for
ONNX model importer in ONNX-MLIR.
The following sections will describe how to use gen_onnx_mlir.py to add an operation into ONNX
 dialect in ONNX-MLIR and how to refine the definition of the operation.

# Add an Operation <a name="add_operation"></a>
To generate an operation for ONNX dialect, add this operation into the dictionary,
'version_dict', in gen_onnx_mlir.py.
The key of this directory is the operation name and the value is the list of
opset for this operation. Usually only the top version opset of this operation (in onnx-mlir/third_party/onnx) is supported. Details about versioning can be found in [version section](#version).
With this entry, the script will generate the operation definition for ONNX dialect.

# Customization <a name="customize"></a>

## Add Interface and Trait
* By default, all operation has shape inference interface and `Pure` trait.
* If an operation has `ResultTypeInferenceOpInterface`, add it to dictionary `OpsWithResultTypeInference`.
This interface infers the type of result tensor, not shape.
* If an operation has subgraph, it will has interface `HasOnnxSubgraphOpInterface`.
This attribute is inferred from the ONNX operation definition.
* You can define helper function for an operation with dictionary `OpsWithHelpers`.

By default, all operation has shape inference interface and `Pure` trait.
If an operation has `ResultTypeInferenceOpInterface`, use dictionary `OpsWithResultTypeInference`.
This interface infers the type of result tensor, not shape.
If an operation has subgraph, it will has interface `HasOnnxSubgraphOpInterface`.

## Add canonicalization interface
If a transformation should be applied locally to an operation across passes, canonicalization
interface can be used for this transformation. To enable the canonicalization for an operation,
add the name of this operation into this list of  `OpsWithCanonicalizer` and then the operation
will have `hasCanonicalizer = 1;` in its definition.

## Customize builder
The default builders for an operation require the type of results as a parameter. However, the type
of results can be inferred. A customize builder may be a useful to simplify the code. Based on the
type of inference, there are two kinds builder, unranked type and broadcast type. To enable the
special builder for an operation, you can add its name into `custom_builder_unranked_ops_list`
 and `custom_builder_broadcast_ops_list` respectively.

Please note that the need of special builder in rewriting rules can be avoided
with the use of `returnType`. Refer to [MLIR doc](https://mlir.llvm.org/docs/DeclarativeRewrites/) or
the [example in ONNX-MLIR](../src/Transform/ONNX/Decompose.td).
 It may be a better solution to just move such
type inference code into ONNXOpHelper.cpp and get rid of customize builder.

Please note that the need of special builder in rewriting rules can be avoided with the use of `returnType`. It may be a better solution to just move such type inference code into ONNXOpHelper.cpp
and get rid of customize builder.


## Customize verifier
The operation description for an operation lists out the allowed types of each input/output and
attribute. The table gen will generate a default verifier to check IR for the allowed types.
If an operation has extra constraints, a customized verifier should be defined to enhance error detection.
For example, two inputs of an operation may require the same element type or same rank.
Such information can be found in the ONNX operation definition, but can not be expressed with the dialect definition.
The best way to test for these constraints are in a verifier. To add the interface of customized verifier to an operation, locate the array below in `gen_onnx_mlir.py` and add your operation in it.
```
OpsWithVerifier = ['AveragePool', 'Conv', 'InstanceNormalization', 'Mod']
```
Then you will find the following line in operation definition in ONNXOps.td.inc:
```
let verifier = [{ return ::verify(*this); }];
```

You will need to add the implementation code in the `src/Dialect/ONNX/ONNXOps.cpp` when the new op was declared as using a customized verifier.  Best is to look at other operations to get the general pattern, by searching for [static LogicalResult verify(ONNXInstanceNormalizationOp op)](../src/Dialect/ONNX/ONNXOps.cpp), for example. Note that a verifier will execute each time that one such op is created. So you will need to ensure that it can work with tensors and MemRefs, and possibly unranked tensors. So guard each of your tests to the proper circumstances. For examples, once a tensor is ranked, you may then verify that the rank is within the approved range (if there is such a constraint); before it is ranked, do not perform this test yet.

Tips:
* Use `operandAdaptor` object to get the inputs (must use  `operandAdaptor` to get the current values of the inputs) and the `op` object to get the attributes (can use `op` because attributes are typically immutable).
* Use `hasShapeAndRank(X)` to test if `X` input is currently shaped and ranked. If not, return success as we will get a chance later to test the operation with this info. Note that some inputs may be scalar too, in which case they may or may not be encoded as a shape type.
* You can then use MLIR call `mlir::cast<ShapedType>(X.getType())` to get a shape types, for which you can get the rank and the dimensions. At this time, we only check dimension validity for values known at runtime. Unknown dimensions are encoded as a negative number. Please only use the cast when you are sure that it will not assert, i.e. the type is indeed a `ShapedType`.
* When you find an error, report it with a friendly error message using `op->emitError(msg)`.

## Customize importer
`special_op_handler`: creates special import function in frontend_dialect_transformer.cpp. Currently, a special handler is used for operations with operational arguments

## Arbitrary extra definition
If the definition of an operation needs extra code other than described above, you can put
the code in the dictionary `custom_definition_misc`. The key is the operation name and the value is the code.

## Customize importer
`special_op_handler`: creates special import function in frontend_dialect_transformer.cpp. Currently, a special handler is used for operations with operational arguments

## Arbitrary extra definition
If the definition of an operation needs extra code other than described above, you can put
the code in the dictionary `custom_definition_misc`. The key is the operation name and the value is the code.

# Build <a name="build"></a>
In order to run gen_onnx_mlir.py, ONNX has to be installed. Refer to Readme.
In your build directory, execute the following command.
 ```
 make OMONNXOpsIncTranslation
 ```
This command will generate those two files (src/Dialect/ONNX/ONNXOps.td.inc and
OpBuilderTable.inc), and copy them to the right place in src directory.
If you modified gen_onnx_mlir.py, you need to check in two generated files too. They are treated
source file in ONNX-MLIR build so that user of ONNX-MLIR does not need to install the particular
version of ONNX. Do not modify these files directly.
You can also run the script directly with the files generated in utils directory. `python ../utils/gen_onnx_mlir.py`.

## Update the documentation

When adding a new op version or making changes to the ONNX version, we would like to also reflect these changes in the ONNX documentation of our supported operations. While the latest [ONNX specs](https://github.com/onnx/onnx/blob/main/docs/Operators.md) are always available, the specs that we support are often a bit back, plus we support older versions under the versioned name as mentioned in the previous section.

There is a convenient command to update both the ONNX and Krnl dialect, as shown below.
```
make onnx-mlir-docs
```
The above command is run in the usual `build` directory and it will install the new dialect md files directly into the `docs/Dialects` directory.

The same command should be used when adding operations/making changes to the Krnl dialect.

# Operation Version <a ref="version"></a>
ONNX-MLIR project started when ONNX was at version 1.7.0 and does not intended to be backward compatible. We relies on onnx/converter to convert the model to the version which ONNX-MLIR supports. As ONNX version is evolving, ONNX-MLIR tries to follow but may be behind the latest version.

## Version of Operations
As stated previous, we try to support the latest version of ONNX operations. The version of each operation currently supported is recorded in [utils/gen_onnx_mlir.py](../utils/gen_onnx_mlir.py). This mechanism provides some stability in version. To check the changes in version, run gen_onnx_mlir.py with flag "--check-version" and the changes will be reported. To move to a newer version, manually update the version dictionary in the script.

## Support Multiple versions
To support multiple versions of an op, the selected version should be added in the version dictionary in [utils/gen_onnx_mlir.py](../utils/gen_onnx_mlir.py). For example, there are two versions (opset), 11 and 13, forReduceSum that are supported. The corresponding entry in version_dic is `'ReduceSum': [13, 11]`.

In ONNX dialect, the op for the top version has no version in the op name, while other version with name followed by 'V' and version number. For example, ReduceSum of opset 13 will be `ONNXReduceSumOp`, while ReduceSum of opset 11 is 'ONNXReduceSumV11Op`. Since most of ONNX op are compatible when upgraded to higher version, we can keep the name of the operation in the dialect and just update version_dict in gen_onnx_mlir.py without touching the code in ONNX-MLIR.

When a model is imported, the highest version which is not higher than the next available version is used. For the example of ReduceSum, if the opset is 12, ONNXReduceSumV11Op is chosen.

## Migrating
To migrate a new version ONNX, first the third_part/onnx should be upgraded and your installation
of ONNX.
Then you can run gen_onnx_mlir.py with flag `--check_operation_version`. The top version for all
operation will be outputted as a new `version_dict`.
If the interface of an operation remains the same (from the change document of ONNX), you can
just use the new version.
If the interface does change, you can insert the new version as the first in the version list.
For the existing code, all the corresponding code has to be changed. For example, when ReduceSum
is moved from version 11 to 13, ONNXReduceSumOp is replaced with ONNXReduceSumOpV11 first.
Then the code for version 13 will use ONNXReduceSumOp.
The reason for such design is that most of ONNX changes do not change the interface. We do not
want to put burden on developer to remember which version of operation is used unless absolutely
necessary.
It is not always needed to keep the code for an older version, which may be rewritten into the new
operation. Thus, we just need to have the dialect definition, but not the code for inference or
lowering.


[./UpdatingLLVMCommit.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Updating the LLVM commit and StableHLO submodule

ONNX-MLIR depends on `LLVM project` (among various other projects such as `StableHLO`). The `LLVM project` dependency is captured in [../utils/clone-mlir.sh](clone-mlir.sh). `StableHLO` is a submodule found in the `third_party` directory.

We plan to update `LLVM project` and `StableHLO` biweekly in order to keep up-to-date with the advancements made in `mlir`, but also to decrease the complexity of each update.

## Which LLVM commit should I pick?

Since downstream projects may want to build ONNX-MLIR (and thus LLVM and StableHLO) in various configurations (Release versus Debug builds; on Linux, Windows, or macOS; possibly with Clang, LLD, and LLDB enabled), it is crucial to pick LLVM commits that pass tests for all combinations of these configurations.

Rather than picking independent LLVM commits from other `mlir`-related projects, we leverage the commits identified by `StableHLO` which is based on `XLA`. Biweekly, StableHLO will bump the revision of LLVM to match what `openxla/xla` is using.

We've started an update rotation that is described [here](https://github.com/onnx/onnx-mlir/wiki/LLVM-Update-Schedule).

## What is the update process?

1. **Lookup commit hashes**: You can find the LLVM commit [here](https://github.com/openxla/stablehlo/blob/main/build_tools/llvm_version.txt) and the StableHLO commit associated with the LLVM commit [here](https://github.com/openxla/stablehlo/tree/main/build_tools). Please check the PR for LLVM to obtain the appropriate commit for StableHLO, we want to upgrade both dependencies in order to avoid any errors.
2. **Update the `llvm-project` commit**: Update the LLVM commit referenced in the source tree to the commit hash for the LLVM project from Step 1. The current locations that need to be updated are [utils/clone-mlir.sh](../utils/clone-mlir.sh), [docs/BuildOnLinuxOSX.md](BuildOnLinuxOSX.md) and  [docs/BuildOnWindows.md](BuildOnWindows.md).
3. **Update the `stablehlo` submodule**: In the `third-party/stablehlo` directory, run `git fetch` followed by `git checkout <stablehlo-commit-hash>` (where `<stablehlo-commit-hash>` is the commit hash for the  project from Step 1).
4. **Rebuild and test ONNX-MLIR**: This might involve fixing various API breakages introduced upstream (they are likely unrelated to what you are working on).  If these fixes are too complex, please file a work-in-progress PR explaining the issues you are running into asking for help so that someone from the community can help.

Here is an example of a PR updating the LLVM commit and StableHLO submodule:

- https://github.com/onnx/onnx-mlir/pull/2662


[./BuildPyRuntimeLit.md]:

# How to build and use PyRuntime lit

## Purpsoe

PyRuntime lit is a different way to build the original PyRuntime (src/Runtime/python).
All necessary dependence, such as llvm_project and onnx-mlir compiler is removed. The purpose is to easily build the python driver for the model execution on 
different systems. Currently, only the OMTenserUtils (src/Runtime), Python driver (src/Runtime/python), third_party/onnx and third_party/pybind11 are built.

The build of PyRuntime lit is controlled by a CMake option: ONNX_MLIR_ENABLE_PYRUNTIME_LIT. Without this option to cmake, the whole system remains the same.

## Functionalities
1. Build the python driver without llvm_project and onnx-mlir compiler built.
2. The python driver can be used with utils/RunONNXModel.py, or onnxmlir python package.
3. With PyRuntime lit, the compiler has not been built locally and docker image of onnx-mlir has to be usd to compile the model. The onnxmlir package contains
the python code to use python docker package to perform the compilation. Alternatively, the old script, onnx-mlir/docker/onnx-mlir.py, can do the fulfill the same task with subprocess and docker CLI.

## How to use
You can find the script for build and run at "onnx-mlir/utils/build-pyruntime-lit.sh.
```
#!/bin/bash

# Assume you are in an empty directory for build in cloned onnx-mlir.
# Usually it is "your_path/onnx-mlir/build"
# then you can run this script as "../util/build-pyruntime-lit.sh"

cmake .. -DONNX_MLIR_ENABLE_PYRUNTIME_LIT=ON
make
make OMCreatePyRuntimePackage

# Install the package
pip3 install -e src/Runtime/python/onnxmlir
# -e is necessary for current package. Need to add resource description
# to install the pre-compiled binary

# Run test case
cd src/Runtime/python/onnxmlir/tests
python3 test_1.py
# Current limitation on where the model is
```


[./BuildOnLinuxOSX.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Installation of ONNX-MLIR on Linux / OSX

We provide here directions to install ONNX-MLIR on Linux and OSX.
On Mac, there are a couple of commands that are different.
These differences will be listed in the explanation below, when relevant. Installing ONNX-MLIR on Apple silicon is natively supported and it is recommended to use brew to manage prerequisites.


## MLIR

Firstly, install MLIR (as a part of LLVM-Project):

[same-as-file]: <> (utils/clone-mlir.sh)
``` bash
git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout b270525f730be6e7196667925f5a9bfa153262e9 && cd ..
```

[same-as-file]: <> (utils/build-mlir.sh)
``` bash
mkdir llvm-project/build
cd llvm-project/build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DENABLE_LIBOMPTARGET=OFF \
   -DLLVM_ENABLE_LIBEDIT=OFF

cmake --build . -- ${MAKEFLAGS}
cmake --build . --target check-mlir
```

To enable parallelization for onnx-mlir, llvm-project should be configured as
```
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_ENABLE_LIBEDIT=OFF
```

## ONNX-MLIR (this project)

### Build

The `MLIR_DIR` cmake variable must be set before building onnx-mlir. It should point to the mlir cmake module inside an llvm-project build or install directory (e.g., llvm-project/build/lib/cmake/mlir).

This project uses lit ([LLVM's Integrated Tester](https://llvm.org/docs/CommandGuide/lit.html)) for unit tests. When running cmake, we can also specify the path to the lit tool from LLVM using the `LLVM_EXTERNAL_LIT` variable but it is not required as long as MLIR_DIR points to a build directory of llvm-project. If `MLIR_DIR` points to an install directory of llvm-project, `LLVM_EXTERNAL_LIT` is required.

To build ONNX-MLIR, use the following commands (maybe with additional `-DCMAKE_CXX_FLAGS` argument described [below](#enable-cpu-optimizations)):

[same-as-file]: <> ({"ref": "utils/install-onnx-mlir.sh", "skip-doc": 2})
```bash
git clone --recursive https://github.com/onnx/onnx-mlir.git

# MLIR_DIR must be set with cmake option now
MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir
mkdir onnx-mlir/build && cd onnx-mlir/build
if [[ -z "$pythonLocation" ]]; then
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_DIR=${MLIR_DIR} \
        ..
else
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DPython3_ROOT_DIR=$pythonLocation \
        -DMLIR_DIR=${MLIR_DIR} \
        ..
fi
cmake --build .

# Run lit tests:
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
```

Since OSX Big Sur, add the `-DCMAKE_CXX_COMPILER=/usr/bin/c++` option to the above `cmake ..` command due to changes in default compilers.

The environment variable `$pythonLocation` may be used to specify the base directory of the Python compiler.

After the above commands succeed, an `onnx-mlir` executable should appear in the `Debug/bin` or `Release/bin` directory.

### Enable CPU Optimizations

To make the compiler run faster (without any affect on the generated code)
you can pass `-DCMAKE_CXX_FLAGS=-march=native` to the `cmake -G Ninja ..` build configuration step above to generate code that exploits all the features of your local CPU, at the expense of portability. Or you can enable a specific CPU feature, e.g. `-DCMAKE_CXX_FLAGS=-mf16c` to enable the F16C feature to enable native conversion between float16 and (32 bit) float. It is used in `src/Support/SmallFP.hpp`.

### Known MacOS Issues

#### jsoniter issue

There is a known issue when building onnx-mlir. If you see an error of this sorts:

``` shell
Cloning into '/home/user/onnx-mlir/build/src/Runtime/jni/jsoniter'...

[...]

make[2]: *** [src/Runtime/jni/CMakeFiles/jsoniter.dir/build.make:74: src/Runtime/jni/jsoniter/target/jsoniter-0.9.23.jar] Error 127
make[1]: *** [CMakeFiles/Makefile2:3349: src/Runtime/jni/CMakeFiles/jsoniter.dir/all] Error 2
make: *** [Makefile:146: all] Error 2.
```

The suggested workaround until jsoniter is fixed is as follows: install maven (e.g. `brew install maven`) and run `alias nproc="sysctl -n hw.logicalcpu"` in your shell.

#### Protobuf issue (Mac M1, specific to protobuf 4.21.12 which is currently required)

On Mac M1, you may have some issues building protobuf. In particular, you may fail to install onnx (via `pip install -e third_party/onnx`) or you may fail to compile `onnx-mlir` (no arm64 symbol for `InternalMetadata::~InternalMetadata`).

The first failure is likely an issue with having multiple versions of protobuf.
Installing a version with `brew` was not helpful (version 4.21.12 because of a known bug that can be corrected with a patch below).
Uninstall the brew version, and make sure you install the right one with pip: `pip install protobuf== 4.21.12`.

The second failure can be remediated by downloading protobuf source code, applying a patch, and installing it on the local machine.
See [Dockerfile.llvm-project](../docker/Dockerfile.llvm-project) on line 66 for cloning instructions. After cloning the right version, you should apply a patch [patch](https://github.com/protocolbuffers/protobuf/commit/0574167d92a232cb8f5a9107aabda0aefbc39e8b) by downloading from the link above and applying it.
Then you should follow the steps in the [Dockerfile.llvm-project](../docker/Dockerfile.llvm-project) file (skipped the `ldconfig` step without consequences).
You may have to brew a couple of the tools, see the `yum install` in the `Dockerfile.llvm-project` file above.
You should then be able to successfully install protobuf and compile `onnx-mlir`.
As the dependences between `third_party` and `onnx-mlir` might cause issues, it is always safe to delete the `third_party` directory, reinstall using `git submodule update --init --recursive`, reinstall `onnx`, delete `onnx-mlir/build` and rebuild `onnx-mlir` from scratch.


### Trouble shooting build issues

Check this [page](TestingHighLevel.md) for helpful hints.


[./LoweringCode.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Lowering Code

## Generating Standard or MemRef code

### Traditional approach

The traditional way to generate code in MLIR is to use the `create` methods, which internally employ the `builder` methods associated with each MLIR operation. For example, creating an addition of two values is done as shown below.
``` C++
// Declaration for the input values, to be filled accordingly
Value firstIntVal, secondIntVal;
Value firstFloatVal, secondFloatVal;
OpBuilder rewriter; // Typically inherited from a caller context.
Location loc; // Typically derived from an operation.
Value intRes = rewriter.create<AddIOp>(loc, firstIntVal, secondIntVal);
Value floatRes = rewriter.create<AddFOp>(loc, firstFloatVal, secondFloatVal);
``` 
***Code: Traditional way to add numbers.***

In the above code, we need to distinguish between int and float type operations. We also need to repetitively pass the location.

### Math builder

A newer approach suggested by the MLIR community is to create a math builder, described below. The same code can be generated using the following.
``` C++
// Using hte same declaration as above for values, rewriter, and location.
MathBuilder createMath(rewriter, loc);
Value intRes = createMath.add(firstIntVal, secondIntVal);
Value floatRes = createMath.add(firstFloatVal, secondFloatVal);
```
***Code: New approach to add numbers.***

MLIR recommends this approach as it reads better, namely "we are creating a math add of two values", and the rewriter and location fields are now "hidden" inside the lightweight `createMath` object. In addition, the method deals with the different MLIR operations for adding integer and float internally.

In general, this and all other builders can be created as follows.
``` C++
// Constructors in class declaration.
struct MathBuilder : DialectBuilder {
  MathBuilder(OpBuilder &b, Location loc);
  MathBuilder(const DialectBuilder &db);
};

// Usage.
MathBuilder createMath(rewriter, loc); // Use original info.
MathBuilder createMath(createKrnl);    // Use info stored in another builder.
```

The Math builder contains the operations listed below. Most are self explanatory. They handle both integer and float operations, and will generate an assert when a specific operation is not supported for a specific type.  Up to date info should be looked from the [MLIRDialectBuilder.hpp](../src/Dialect/Mlir/DialectBuilder.hpp) file.

```C++
struct MathBuilder : DialectBuilder {
  MathBuilder(OpBuilder &b, Location loc);
  MathBuilder(const DialectBuilder &db);

  Value andi(Value lhs, Value rhs);
  Value add(Value lhs, Value rhs);
  Value sub(Value lhs, Value rhs);
  Value mul(Value lhs, Value rhs);
  Value div(Value lhs, Value rhs);
  Value exp(Value val);
  Value select(Value cmp, Value lhs, Value rhs);
  Value sgt(Value lhs, Value rhs);
  Value slt(Value lhs, Value rhs);
  Value eq(Value lhs, Value rhs);
};
```
***Code: Math builder class.***

Note using the builders does not preclude making calls to the old interface. For any builders, we can extract, respectively, the rewriter and the location needed for the old interfaces using the `DialectBuilder` inherited methods `getRewriter()` and `getLoc()`.

### MemRef builder

An equivalent builder exists for some MemRef operation. At a high level, the following operations are supported.

``` C++
struct MemRefBuilder : DialectBuilder {
  MemRefBuilder(OpBuilder &b, Location loc);
  MemRefBuilder(const DialectBuilder &db);

  memref::AllocOp alloc(MemRefType type, ValueRange dynSymbols);
  memref::AllocaOp alloca(MemRefType type);
  memref::DeallocOp dealloc(Value val);
  Value dim(Value val, int64_t index);
};
```
***Code: MemRef builder class.***

It defines 4 distinct methods: how to allocate memory (`alloc`) and free (`dealloc`) memory from the heap, how to allocate memory on the stack (`alloca`), and how to extract the dimension of a multi-dimensional memory reference for a given dimension. The `alloca` method above allows for the multi-dimensional memory to have dynamic dimensions; these dynamic dimensions are specified by the parameter `dynSymbols`.  There are variant of these methods for static dimensions only and for providing alignment constraints. See the [MLIRDialectBuilder.hpp](../src/Dialect/Mlir/DialectBuilder.hpp) file for the full set of supported operations.

## Generating Krnl Operations

The krnl dialect is our main dialect to lower ONNX operations into loops. This dialect is one step above the MLIR affine dialect in that in enables us to express higher level loop constructs and loop optimizations.

## Builder based interface to generate Krnl loops

The new approach uses a Krnl builder class to construct Krnl dialect operation. The basic methods to build loops are the one listed below. Up to date info is found in the [KrnlHelper.hpp](../src/Dialect/Krnl/KrnlHelper.hpp) file.

``` C++
struct KrnlBuilder : public DialectBuilder {
  KrnlBuilder(OpBuilder &b, Location loc);
  KrnlBuilder(DialectBuilder &db);

  ValueRange defineLoops(int64_t originalLoopNum);

  void iterate(ValueRange originalLoops, ValueRange optimizedLoops,
      ValueRange lbs, ValueRange ubs,
      function_ref<void(const KrnlBuilder &createKrnl, ValueRange indices)>
          bodyBuilderFn);
};
```
***Code: Krnl builder class to minimally create a loop.***

The first method, `defineLoops` creates a set of loop descriptors that characterizes a loop iteration space. Initially, a set of loop descriptors characterizes the original loop iteration space, shortly, one such modified set can also be used to characterize an optimized iteration spaces, for example to represent a loop tiled iteration space after applying loop blocking and loop permutation.

The second method above, `iterate` is used to create a set of loops and its corresponding loop body. Until we optimize loops, both the `originalLoops` and the `optimizedLoops` are set to the output of a `defineLoops` method call. These sets describe the iteration space and its dimensionality. The next two parameters are used to describe the lower and the upper bounds of the loop. The last parameter defines a lambda function that implements the body of the loop. This lambda function is invoked with two parameters: an object to create further Krnl operations within the loop body and a list of the current loop index values.

The usage of this builder will become clearer with our example, setting an array to value zero. This is the same example as in the prior section.
``` C++
// Defined values 0 and a 2 dimensional array with dim ub0 and ub1
Value zero, array, ub0, ub1;

// Define the krnl builder.
KrnlBuilder createKrnl(rewriter, loc);

// Define a 2-dimensional iteration space.
ValueRange loopDef = createKrnl.defineLoops(2);

// Create the loop.
createKrnl.iterate(loopDef, loopDef, {zero, zero}, {ub0, ub1},
  [&](const KrnlBuilder  &createKrnl, ValueRange loopInd){
    // Loop body.
    createKrnl.store(zero, array, loopInd);
  });
```
***Code: Zeroing an array using the new builder interface***

Using this new scheme, we first define the 2D loop iteration space and then create the loop iteration structure using the `iterate` method. Since the loop is unoptimized, the same `loopDef` value range is passed as the first 2 parameters. The bounds are passed as 2 sets of ordered values.

Note that the lambda function creates an `createKrnl` builder that is similar to that of the external environment (outside the loop), but customized for inside the loop. So we can continue to use this overloaded builder to continue constructing krnl operations. In our case, we simply use the `loopInd` (2nd parameter of the lambda function), which are the current loop induction values, to define the element of the array that is set to zero.

Some of the other operations that are often used are listed below.
``` C++
struct KrnlBuilder : public DialectBuilder {
  // in addition to above...

  // Memory operations.
  Value load(Value memref, ValueRange indices = {});
  void store(Value val, Value memref, ValueRange indices = {});

  // Loop optimizations.
  ValueRange block(Value loop, int64_t blockSize);
  void permute(ValueRange loops, ArrayRef<int64_t> map);

  // Simple setter for entire arrays.
  void memcpy(Value dest, Value src, Value size);
  void memset(Value dest, Value val);
};
```
***Code:Additional Krnl ops supported by the Krnl builder interface.***

Above, both the load and store operations are used to create Krnl memory load and store operations. They should be used instead of the MLIR Affine or Standard dialect operations.

The `block` method takes one loop definition (one value extracted from the output of a `defineLoop` operation) and will split that loop definition into 2, where the first one iterates over blocks of the given side, and the second one iterates inside of a given block. The two loop definitions are returned by the `block` method as a value range containing the two split loops described above.

The `permute` method takes a list of loop definitions and ensures that the loops will iterate according to the permuted order.

The `memcopy` method results in the array given by `dest` to be overwritten by `size` values from the array given by `src`. The `memset` method sets the entire array given by `dest` to the value passed in `val`, typically zero.

## Builder based interface to generate optimized Krnl loops

Let us now look how we can optimize loops using the Krnl builder. Consider our same example, setting an array to zero, and say we whish to tile the loop along both dimensions. Let us first tile a 1-dimensional loop iteration space.

``` C++
// Defined values 0 and a 1 dimensional array with dim ub0.
Value zero, array, ub0;
// Define a 2-dimensional iteration space.
ValueRange loopDef = createKrnl.defineLoops(1);
// Block the loop by a factor 4. First returned value in ValueRange 
// loops over blocks, the second return value loops inside a block.
ValueRange loopBlockDef = createKrnl.block(loopDef, 4);
// Permute the blocked loops
createKrnl.permute({loopBlockDef[0], loopBlockDef[1], {0,1});
// Create the loop iterating over the blocks.
createKrnl.iterate(loopDef, {loopBlockDef[0], loopBlockDef[0]}, {zero}, {ub0},
  [&](const KrnlBuilder  &createKrnl, ValueRange blockLoopInd){
    // Loop body.
    createKrnl.store(zero, array, loopInd);
  });
```
***Code: Blocked loop zeroing 1D array.***

In the code above, we block the original 1D loop iteration space defined by `defineLoop(1)` into two loops, one looping over the blocks of size 4, and the other looping inside a block. We then need to instruct the order of the optimized loop iteration space using a `permute` method. We can then perform an `iterate` method call, where the first parameter describes the original loop iteration space along with the lower and upper bound sets. In that same call, the second parameter indicates the actual loop iterations that we want to perform in the optimized iteration space, namely the loops over the blocks (`loopBlockDef[0]`) and loops inside a block (`loopBlockDef[1]`).

We now consider tiling our original 2-dimensional example below.
``` C++
  // Defined values 0 and a 2 dimensional array with dim ub0 and ub1
  Value zero, array, ub0, ub1;

  // Define a 2-dimensional iteration space.
  ValueRange loopDef = createKrnl.defineLoops(2);
  Value outerLoopDef(loopDef[0]), innerLoopDef(loopDef[1]);
  // Block each of the 2 dimensions: outer by 4, inner by 8.
  ValueRange outerLoopBlockDef = createKrnl.block(outerLoopDef, 4);
  ValueRange innerLoopBlockDef = createKrnl.block(innerLoopDef, 8);
  // Permute the loops (first loop over blocks, the loop inside blocks).
  createKrnl.permute({outerLoopBlockDef[0], outerLoopBlockDef[1],
    innerLoopBlockDef[0], innerLoopBlockDef[1]}, {0,2,1,4});
  // Create the loop iterating over the blocks.
  createKrnl.iterate(loopDef, {outerLoopBlockDef[0], innerLoopBlockDef[0]},
    {zero, zero}, {ub0, ub1},
    [&](const KrnlBuilder  &createKrnl, ValueRange blockLoopInd){
      // Create the loop iterating inside the blocks.
      createKrnl.iterate({}, {outerLoopBlockDef[1], innerLoopBlockDef[1]},
        {}, {}, [&](const KrnlBuilder  &createKrnl, ValueRange loopInd) {
          // Loop body.
          createKrnl.store(zero, array, loopInd);
        });
    });
```
***Code:Tiled loops zeroing 2D array.***

In the code above, we first renamed the 2-dimensional loop iteration space defined by the `defineLoops` method as outer and inner loop defs, corresponding respectively to the first and second value in the value range named `loopDef`. Then we block each of the outer and inner loops, resulting in 4 loops, 2 going over the blocks of the outer/inner loop and two going inside the blocks. The `permute` method defines the desired order, namely the blocked loop first, and the loops for the elements inside the block second.

All of the 4 loops could be now instantiated by a single `iterate` methods. We have chosen here to create the 2 sets of loop separately, as this pattern may be more prevalent in realistic code where we insert some additional code between the blocked loops and the loops iterating over the elements inside the blocks.

The first `iterate` calls provide the original unoptimized loop defs, as these loops are key to provide the lower and upper bounds of the original loop iteration space.  In other word, in this first `iterate` call, we inform the program that the original loops (defined by the value range returned by `defineLoops`) have lower bounds `{zero, zero}` and upper bounds `{ub0, ub1}`. This first `iterate` calls also indicates that we are interested issue loops for the 2 blocked loops, defined as `{outerLoopBlockDef[0], innerLoopBlockDef[0]}`.

The second `iterate` call does not need to redefine the original loop defs, as we have already provided the lower and upper bounds. So all these fields are left blank. In the second parameter to this call, we provide the next set of loops for which we want code to be generated, namely `{outerLoopBlockDef[1], innerLoopBlockDef[1]}`. Recall that the second parameter in the value range returned by a `permute` method corresponds to the iterating over the elements inside a given block.

Note also that we use the loop indices `loopInd` directly in the memory operation, as the loop indices are always the actual iteration number corresponding to the original loop. For example, consider an iteration space of 0..12. If we block it by a factor 4, the indices of the blocked loop will be 0, 4 and 8. And the indices of the loops iterating inside a given block will be 0,1,2, and 3 for the first block, 4,5,6,and 7 for the second block, and 8, 9, 10, and 11 for the third block. Say if the original loop trip count was only up to 11 instead of 12, the third block would iterate over the indices 8, 9, and 10 only.

## Generating affine loops

There is one more builder to assist the lowering of the Krnl dialect into the affine dialect. This builder is named `AffineBuilder` and is found in [KrnlToAffine.cpp](../src/Conversion/KrnlToAffine/KrnlToAffine.cpp)  file. It provides helper methods to generate multiple nested `affine.for` loops as well as `affine.if then else` constructs.

## Generating SCF operations

There is an additional builder for generating MLIR's SCF dialect.

## Combining multiple builders

Instead of creating multiple builders, e.g.

```C++
  KrnlBuilder createKrnl(rewriter, loc);
  MathBuilder createMath(createKrnl);
  MemRefBuilder createMemRef(createKrnl);
```
and then using them like this

```C++
  createKrnl.defineLoop(1);
  createMath.add(i1, i2);
  createMemRef.alloca(type);
```

we can create a single builder composed of multiple types and then as follows.

```C++
  MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder>
    create(rewriter, loc);

  create.krnl.defineLoop(1);
  create.math.add(i1, i2);
  create.mem.alloca(type);
```

Types that can be used here are listed here.
  *  `KrnlBuilder`, accessed with `krnl` field.
  *  `MathBuilder`, accessed with `math` field.
  *  `MemRefBuilder`, accessed with `mem` field.
  *  `ONNXBuilder`, accessed with `onnx` field.
  *  `SCFBuilder`, accessed with the `scf` field.


[./Documentation.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# About Documentation

## How to add a new documentation page

Firstly, `/docs` is the root directory of the documentation website, meaning that any
documentation page you wish to display to a user must be located within `/docs`.

Secondly, add the documentation page into the navigation configuration file located at
`/docs/_data/navigation.yaml`. Edit the table of content to include the path to
the newly created documentation page with a descriptive title.

Then, capture the changes done in a patch and submit a pull request; once the patch is
merged into `onnx-mlir` codebase, a link pointing to the file path specified with the
descriptive title you provided will appear on the navigation panel.

[./AddCustomAccelerators.md]:

# A guideline on adding a new custom accelerator

In general, onnx-mlir handles custom accelerators as pluggins which can be turned on/off when building onnx-mlir and compiling a model. The handling is mainly via `cmake` and we will outline its procedure in this document.

Besides this document, [NNPA accelerator](../src/Accelerators/NNPA) can be used as an example that has been deployed in onnx-mlir.

## 1. Code folder

In onnx-mlir, all code for an accelerator should be put inside a separate folder under `src/Accelerators`. Thus, the first step to support an accelerator is to create a folder for it inside `src/Accelerators`.

The folder name will be used as the accelerator name in onnx-mlir. In particular, it is used to
1. instruct `cmake` to build the code inside the accelerator folder,
2. compile a model for the accelerator when using `onnx-mlir` command, and
3. enable passes related to the accelerator when using `onnx-mlir-opt` command.

The folder content is flexible depending on each accelerator. However, we recomment to follow the same structure as the root folder of `onnx-mlir` as much as possbile. This helps maintain the consitency across the whole project.

### 1.1 Build accelerators in onnx-mlir

To build accelerators in onnx-mlir, use the cmake variable `ONNX_MLIR_ACCELERATORS` when building onnx-mlir. `ONNX_MLIR_ACCELERATORS` accepts a semicolon-separated list of accelerator names. For example,
```bash
$ cd build
$ cmake .. -DONNX_MLIR_ACCELERATORS='accel1;accel2'
```
Note that the list should be quoted.

### 1.2 Compile a model to run with selected accelerators.

The compiler command `onnx-mlir` has an option, i.e. `--maccel`, to compile a model for selected accelerators. For each accelerator add a `--maccel=accel_name` entry. For example,

```bash
$ onnx-mlir --maccel=accel1 --maccel=accel2 model.onnx
```

Only built accelerators can be used with `--maccel`.

### 1.3 Run passes related to selected accelerators.

Passes defined by an accelerator can be run or tested via `onnx-mlir-opt` command by using option `--maccel` which is similar to `--maccel` in `onnx-mlir` (See Sec. [1.2](#1.2-compile-a-model-to-run-with-selected-accelerators)). For example, to call a pass `--optimize-data-layout` defined by accelerator `accel1`:

```bash
$ onnx-mlir-opt --maccel=accel1 --optimize-data-layout model.mlir
```

Only built accelerators can be used with `--maccel`.

## 2. Code integration

### 2.1 Macro

Each accelerator is required to define a few macros. These needs to be included in [onnx_mlir::accel::Accelerator](../src/Accelerators/Accelerator.hpp). These macros are:

1. `INSTRUMENTSTAGE_ENUM_<accel_name>`
2. `INSTRUMENTSTAGE_CL_ENUM_<accel_name>`
3. `PROFILEIR_CL_ENUM_<accel_name>`
4. `OPTREPORT_ENUM_<accel_name>`
5. `OPTREPORT_CL_ENUM_<accel_name>`

Replace `<accel_name>` with the name of the accelerator, for example if your accelerator is named `ACCEL1` use:

```C
#define INSTRUMENTSTAGE_ENUM_ACCEL1
#define INSTRUMENTSTAGE_CL_ENUM_ACCEL1
#define PROFILEIR_CL_ENUM_ACCEL1
#define OPTREPORT_ENUM_ACCEL1
#define OPTREPORT_CL_ENUM_ACCEL1
```

### 2.2 Dialects and passes

Writing code in MLIR typically involves desiging dialects and passes. So does supporting an accelerator. Thus, to integrate accelerator code into onnx-mlir is to register dialects and passes in onnx-mlir.

We provide a base class [onnx_mlir::accel::Accelerator](../src/Accelerators/Accelerator.hpp) from which users can define an inherited class and write hooks to register dialects and passes.

```C
//===--------------------------------------------------------------------===//
// Hooks for onnx-mlir driver
//===--------------------------------------------------------------------===//

/// Add the transformations necessary to support the accelerator.
virtual void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm,
    onnx_mlir::EmissionTargetType &emissionTarget) const = 0;

//===--------------------------------------------------------------------===//
// Hooks for onnx-mlir-opt driver
//===--------------------------------------------------------------------===//

/// Register the MLIR dialects required to support an accelerator.
virtual void registerDialects(mlir::DialectRegistry &registry) const = 0;

/// Register accelerator transformation passes to make available as
/// command line options.
virtual void registerPasses(int optLevel) const = 0;

//===--------------------------------------------------------------------===//
// Hooks for both onnx-mlir and onnx-mlir-opt drivers
//===--------------------------------------------------------------------===//

/// Configure passes for the accelerator.
virtual void configurePasses() const = 0;

//===--------------------------------------------------------------------===//
// Hooks for onnx-to-krnl pass
//===--------------------------------------------------------------------===//

/// Convert TensorType to MemRefType.
/// Acccelators may have special versions of TensorType. If not, override this
/// method and return nullptr.
virtual mlir::MemRefType convertTensorTypeToMemRefType(
    const mlir::TensorType tensorType) const = 0;

/// Define conversion target to be used with ONNXToKrnl.
virtual void conversionTargetONNXToKrnl(
    mlir::ConversionTarget &target) const = 0;

/// Define rewrite patterns to be used with ONNXToKrnl.
virtual void rewritePatternONNXToKrnl(mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) const = 0;

//===--------------------------------------------------------------------===//
// Hooks for krnl-to-llvm pass
//===--------------------------------------------------------------------===//

/// Define conversion target to be used with KrnlToLLVM.
virtual void conversionTargetKrnlToLLVM(
    mlir::ConversionTarget &target) const = 0;

/// Define rewrite patterns to be used with KrnlToLLVM.
virtual void rewritePatternKrnlToLLVM(mlir::RewritePatternSet &patterns,
    mlir::LLVMTypeConverter &typeConverter, mlir::MLIRContext *ctx) const = 0;
```

Though there are many passes in onnx-mlir, we provide hooks for two passes `onnx-to-krnl` and `krnl-to-llvm` only. The reason is that in principal they are the first and the last passes in onnx-mlir. Pass `onnx-to-krnl` is the place where we can decide which ONNX operators will be run on host (by lowering them to Krnl dialect) or on an accelerator (by lowering them to a dialect defined for the accelerator). Pass `krnl-to-llvm` is the place where we lower Krnl and accelerator operators to LLVM dialect, e.g. generate assembly code or simply call external APIs for the accelerator. There can have any dialects and passes for the accelerator between `onnx-to-krnl` and `krnl-to-llvm`.

For example, for NNPA acclerator, we define [ZHigh dialect](../src/Accelerators/NNPA/Dialect/ZHigh) to be used in `onnx-to-krnl` and [ZLow dialect](../src/Accelerators/Dialect/ZLow) to be used in `krnl-to-llvm`.

## 3. Testing

Tests for accelerators should be put inside the folder [test](../test). In particular,
- LIT tests are placed inside a newly-created folder under [mlir/accelerators](../test/mlir/accelerators)
- Other tests are place inside a newly-created folder under [accelerators](../test/accelerators)


[./SupportedONNXOps-cpu.md]:

<!--- Automatically generated, do not edit. -->
<!--- To update, run `make onnx_mlir_supported_ops_cpu' -->

# Supported ONNX Operation for Target *cpu*.

Onnx-mlir currently supports ONNX operations targeting up to opset 22. Limitations are listed when applicable. This documentation highlights the minimum and maximum opset versions that are fully supported by onnx-mlir and not the version changes.

* Operations are defined by the [ONNX Standard](https://github.com/onnx/onnx/blob/main/docs/Operators.md).
* **Supported Opsets** indicates the lowest and highest opset a model may have for onnx-mlir to support compiling a model with the operator.
   * A * indicates onnx-mlir is compatible with the latest version of that operator available as of opset 22.


| Op |Supported Opsets (inclusive) |Limitations |Notes |
| --- |--- |--- |--- |
| **Abs** |6 - * | | |
| **Acos** |7 - * | | |
| **Acosh** |9 - * | | |
| **Adagrad** |none | | | |
| **Adam** |none | | | |
| **Add** |6 - * |No support for short integers. | |
| **And** |7 - * | | |
| **ArgMax** |6 - * | | |
| **ArgMin** |13 - * | | |
| **ArrayFeatureExtractor** |none | | | |
| **Asin** |7 - * | | |
| **Asinh** |9 - * | | |
| **Atan** |7 - * | | |
| **Atanh** |9 - * | | |
| **AveragePool** |6 - * | | |
| **BatchNormalization** |6 - * |Training not supported. | |
| **Bernoulli** |none | | | |
| **Binarizer** |none | | | |
| **BitShift** |none | | | |
| **BitwiseAnd** |18 - * | | |
| **BitwiseNot** |none | | | |
| **BitwiseOr** |18 - * | | |
| **BitwiseXor** |18 - * | | |
| **BlackmanWindow** |none | | | |
| **Cast** |6 - * |Cast only between float and double types. Only ppc64le and MacOS platforms support float16. Does not support int4 and uint4. | |
| **CastLike** |19 - * |CastLike only between float and double types. Only ppc64le and MacOS platforms support float16. Does not support int4 and uint4. | |
| **CastMap** |none | | | |
| **CategoryMapper** |none | | | |
| **Ceil** |6 - * | | |
| **Celu** |none | | | |
| **CenterCropPad** |none | | | |
| **Clip** |6 - * |No support for short integers. | |
| **Col2Im** |none | | | |
| **Compress** |9 - * | | |
| **Concat** |6 - * | | |
| **ConcatFromSequence** |none | | | |
| **Constant** |6 - * |Does not support int4 and uint4. | |
| **ConstantOfShape** |9 - * |Does not support int4 and uint4. | |
| **Conv** |6 - * | | |
| **ConvInteger** |none | | | |
| **ConvTranspose** |6 - * |Spatial dimensions (H and W in input `X`, and kH and kW in input `W`) must be static dimension. | |
| **Cos** |7 - * | | |
| **Cosh** |9 - * | | |
| **CumSum** |11 - * | | |
| **DFT** |17 - * | | |
| **DeformConv** |none | | | |
| **DepthToSpace** |13 - * | | |
| **DequantizeLinear** |10 - * |Only support for per-tensor or layer dequantization. No support for per-axis dequantization. Does not support int4 and uint4. | |
| **Det** |none | | | |
| **DictVectorizer** |none | | | |
| **Div** |6 - * |No support for short integers. | |
| **Dropout** |6 - * |Does not support masked and training. | |
| **DynamicQuantizeLinear** |11 - * | | |
| **Einsum** |12 - * |Limited to the types supported by ReduceSum and MatMul (which we decompose to in most cases) which exclude integers with width < 32. `inputs` must have static dimensions. | |
| **Elu** |6 - * | | |
| **Equal** |7 - * | | |
| **Erf** |9 - * | | |
| **Exp** |6 - * | | |
| **Expand** |8 - * |Input `shape` must have static shape. | |
| **EyeLike** |none | | | |
| **FeatureVectorizer** |none | | | |
| **Flatten** |6 - * |Does not support int4 and uint4. | |
| **Floor** |6 - * | | |
| **GRU** |7 - * |W, B and R must be constants. | |
| **Gather** |6 - * | | |
| **GatherElements** |11 - * | | |
| **GatherND** |11 - * | | |
| **Gelu** |20 - * | | |
| **Gemm** |6 - * | | |
| **GlobalAveragePool** |6 - * | | |
| **GlobalLpPool** |none | | | |
| **GlobalMaxPool** |6 - * | | |
| **Gradient** |none | | | |
| **Greater** |7 - * | | |
| **GreaterOrEqual** |12 - * | | |
| **GridSample** |none | | | |
| **GroupNormalization** |18 - * | | |
| **HammingWindow** |none | | | |
| **HannWindow** |none | | | |
| **HardSigmoid** |6 - * | | |
| **HardSwish** |none | | | |
| **Hardmax** |6 - * | | |
| **Identity** |16 - * |Sequence identity not supported. Does not support int4 and uint4. | |
| **If** |16 - * |Sequence and Optional outputs are not supported. Does not support int4 and uint4. | |
| **Imputer** |none | | | |
| **InstanceNormalization** |6 - * | | |
| **IsInf** |20 - * |Currently no support for float16 infinity value. Only for float32 and float64. | |
| **IsNaN** |20 - * | | |
| **LRN** |6 - * | | |
| **LSTM** |7 - * |W, B and R must be constants. | |
| **LabelEncoder** |none | | | |
| **LayerNormalization** |17 - * | | |
| **LeakyRelu** |6 - * | | |
| **Less** |7 - * | | |
| **LessOrEqual** |12 - * | | |
| **LinearClassifier** |none | | | |
| **LinearRegressor** |none | | | |
| **Log** |6 - * | | |
| **LogSoftmax** |13 - * |Axis 0, 1, and default currently disabled due to changes in ONNX 1.8.1/Opset 13. |Temporally removed due to changes in onnx 1.8.1. |
| **Loop** |6 - * |Input must have static shape. Does not support int4 and uint4. | |
| **LpNormalization** |none | | | |
| **LpPool** |none | | | |
| **MatMul** |6 - * | | |
| **MatMulInteger** |10 - * | | |
| **Max** |6 - * |No support for unsigned int. Only ppc64le and MacOS platforms support float16. | |
| **MaxPool** |6 - * |Does not support argmax and short ints. Support single output only. | |
| **MaxRoiPool** |none | | | |
| **MaxUnpool** |none | | | |
| **Mean** |6 - * | | |
| **MeanVarianceNormalization** |none | | | |
| **MelWeightMatrix** |none | | | |
| **Min** |6 - * |Does not support unsigned numbers. Only ppc64le and MacOS platforms support float16. | |
| **Mish** |none | | | |
| **Mod** |10 - * |Support float and double only. Only ppc64le and MacOS platforms support float16. | |
| **Momentum** |none | | | |
| **Mul** |6 - * |Does not support short integers. | |
| **Multinomial** |none | | | |
| **Neg** |6 - * | | |
| **NegativeLogLikelihoodLoss** |none | | | |
| **NonMaxSuppression** |10 - * | | |
| **NonZero** |9 - * | | |
| **Normalizer** |none | | | |
| **Not** |6 - * | | |
| **OneHot** |9 - * | | |
| **OneHotEncoder** |none | | | |
| **Optional** |none | | | |
| **OptionalGetElement** |none | | | |
| **OptionalHasElement** |none | | | |
| **Or** |7 - * | | |
| **PRelu** |6 - * | | |
| **Pad** |6 - * |axes input not supported. Does not support int4 and uint4. | |
| **Pow** |7 - * |No support for power with integer types. | |
| **QLinearConv** |none | | | |
| **QLinearMatMul** |none | | | |
| **QuantizeLinear** |10 - * |Does not support per-axis and i8 quantization. Does not support int4 and uint4. | |
| **RNN** |7 - * |W, B and R must be constants. | |
| **RandomNormal** |none | | | |
| **RandomNormalLike** |none | | | |
| **RandomUniform** |none | | | |
| **RandomUniformLike** |none | | | |
| **Range** |11 - * | | |
| **Reciprocal** |6 - * | | |
| **ReduceL1** |13 - * |do_not_keep_dim not supported. | |
| **ReduceL2** |13 - * |do_not_keep_dim not supported. | |
| **ReduceLogSum** |13 - * |do_not_keep_dim not supported. | |
| **ReduceLogSumExp** |13 - * |do_not_keep_dim not supported. | |
| **ReduceMax** |6 - * |do_not_keep_dims not supported. | |
| **ReduceMean** |6 - * |do_not_keep_dims not supported. | |
| **ReduceMin** |6 - * |do_not_keep_dims not supported. | |
| **ReduceProd** |13 - * |do_not_keep_dim not supported. | |
| **ReduceSum** |6 - * |Default axis and do_not_keep_dim not supported. |Default axis and do_not_keep_dim temporarily removed due to changes in onnx 1.8.1. |
| **ReduceSumSquare** |13 - * |Default axis and do_not_keep_dim not supported. | |
| **Relu** |6 - * | | |
| **Reshape** |6 - * |allowzero not supported. Input `shape` must have static dimension. Does not support int4 and uint4. | |
| **Resize** |10 - * |Missing support for linear, cubic, crop, pytorch_half_pixel, and floor. Attributes antialias, axes and keep_aspect_ratio_policy are not supported. `scales` and `sizes` must have static dimension. | |
| **ReverseSequence** |10 - * | | |
| **RoiAlign** |none | | | |
| **Round** |11 - * | | |
| **STFT** |none | | | |
| **SVMClassifier** |none | | | |
| **SVMRegressor** |none | | | |
| **Scaler** |none | | | |
| **Scan** |8 - * |Does not support dynamic shapes. Does not support int4 and uint4. |Precision issue with newer opset, maybe just unsupported. Dynamic shape?. |
| **Scatter** |none | | | |
| **ScatterElements** |11 - * |Does not support duplicate indices. | |
| **ScatterND** |11 - * |Does not support scatternd add/multiply. | |
| **Selu** |6 - * | | |
| **SequenceAt** |none | | | |
| **SequenceConstruct** |none | | | |
| **SequenceEmpty** |none | | | |
| **SequenceErase** |none | | | |
| **SequenceInsert** |11 - * |Does not support unranked sequence element. | |
| **SequenceLength** |none | | | |
| **SequenceMap** |none | | | |
| **Shape** |15 - * |Does not support start and end attributes. Does not support int4 and uint4. | |
| **Shrink** |none | | | |
| **Sigmoid** |6 - * | | |
| **Sign** |9 - * | | |
| **Sin** |7 - * | | |
| **Sinh** |9 - * | | |
| **Size** |13 - * |Does not support int4 and uint4. | |
| **Slice** |13 - * |Axis must be a constant argument. |Add tests to slices, currently have none. |
| **Softmax** |6 - * | | |
| **SoftmaxCrossEntropyLoss** |none | | | |
| **Softplus** |6 - * | | |
| **Softsign** |6 - * | | |
| **SpaceToDepth** |13 - * | |Example works, the other is imprecise. To investigate. |
| **Split** |6 - * |Does not support static and dynamic shape, zero size splits. |Temporally removed due to changes in onnx 1.8.1. |
| **SplitToSequence** |none | | | |
| **Sqrt** |6 - * | | |
| **Squeeze** |6 - * |Does not support static and dynamic shape. Does not support int4 and uint4. |Temporally removed due to changes in onnx 1.8.1. |
| **StringNormalizer** |none | | | |
| **Sub** |6 - * |Does not support short integers. | |
| **Sum** |6 - * | | |
| **Tan** |7 - * | | |
| **Tanh** |6 - * | | |
| **TfIdfVectorizer** |none | | | |
| **ThresholdedRelu** |none | | | |
| **Tile** |6 - * | | |
| **TopK** |10 - * |`K`, the number of top elements to retrieve, must have static shape. | |
| **Transpose** |6 - * |Does not support int4 and uint4. | |
| **TreeEnsembleClassifier** |none | | | |
| **TreeEnsembleRegressor** |none | | | |
| **Trilu** |14 - * | | |
| **Unique** |11 - * | | |
| **Unsqueeze** |6 - * |Does not support static and dynamic shape. Does not support int4 and uint4. |Temporally removed due to changes in onnx 1.8.1. |
| **Upsample** |7 - * |Input `X` and `Y` must have static shape. | |
| **Where** |9 - * | | |
| **Xor** |7 - * | | |
| **ZipMap** |none | | | |


[./TestingHighLevel.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNX-MLIR: Build trouble-shooting and testing ONNX_MLIR

## Trouble shooting the building of ONNX-MLIR

If you have issues during the first `onnx-mlir` build, you may need to check the cmake variables used by our build. See the last section of this page for help.

If you have used the source directory successfully for a while, you may experience difficulties to rebuild `onnx-mlir` after merging the latest changes from the `main` branch.

Below is a couple of steps you may perform. If any of them apply, it is recommended to remove the `onnx-mlir/build` subdirectory and rebuild from scratch using the `cmake` commands.

### 1) Checking the right commit of the llvm-project

If the latest `onnx-mlir` `main` branch has moved to a newer commit level of the `llvm-project`, the build process will typically experience multiple compiler failures related to LLVM and MLIR code.

Level required is found in the first code box of the [Building ONNX-MLIR](BuildOnLinuxOSX.md#mlir) page next to the `git checkout` command.

Level used in the code is found by executing a `git log` in the `llvm-project` subdirectory.

If they don't match, please update the llvm project to the required level.

### 2) Checking the right third_party support

Typically, when we update the ONNX op level, it results in new software in the `third_party/onnx` subdirectory. Failing to update that code results typically in compiler failures related to ONNX dialect code.

It is easier to simply remove the `third_party` directory and then reinstalling the code using `git submodule update --init --recursive`.

### 3) Dialect update

Sometimes a dialect update requires the entire build directory to be rebuilt. Typical errors that you may see are missing declarations, for example to `verifier` methods. The recommendation is to simply remove the `onnx-mlir/build` subdirectory and rebuild from scratch using the `cmake` commands.

### 4) Protobuf related issues

If you run into protobuf related errors during the build, check the following potential causes:

* protobuf version is too low or too new (relative to the prereq)
* libprotobuf version and python binding version mismatch
* llvm-project, onnx, and/or onnx-mlir are built against different versions of protobuf, because after updating protobuf you only rebuild one of them
* llvm-project, onnx, and/or onnx-mlir may detect different versions of python3 (so watch their cmake output) if you have multiple python versions installed
* cmake caches stuff and you should never use "make clean" when rebuilding. Instead remove everything under the build tree and start from scratch.

These and many other trickeries for setting up the build env are the reason why we recommend using the [onnxmlir/onnx-mlir-dev](https://github.com/users/onnxmlir/packages/container/onnx-mlir-dev) docker image for development.

## High level testing of ONNX-MLIR

To run the lit ONNX-MLIR tests, use the following command:

[same-as-file]: <> ({"ref": "utils/check-onnx-mlir.cmd", "skip-ref": 1})
```shell
call cmake --build . --config Release --target check-onnx-lit
```

Or simply invoke the `check-onnx-lit` target for `ninja` or `make` in the build directory.

To run the numerical ONNX-MLIR tests, use the following command:

[same-as-file]: <> ({"ref": "utils/check-onnx-numerical.cmd", "skip-ref": 1})
```shell
call cmake --build . --config Release --target check-onnx-numerical
```

Or simply invoke the `check-onnx-numerical` target for `ninja` or `make` in the build directory.

To run the doc ONNX-MLIR tests, use the following command after installing third_party ONNX shown below. Details to first install the third_party ONNX project are detailed [here](BuildONNX.md). Note that it is key to install the ONNX project's version listed in our third_party subdirectory, as ONNX-MLIR may be behind the latest version from the ONNX standard.

[same-as-file]: <> ({"ref": "utils/check-docs.cmd", "skip-ref": 1})
```shell
call cmake --build . --config Release --target check-docs
```

Or simply invoke the `check-docs` target for `ninja` or `make` in the build directory.

# Summary of LLVM and ONNX-MLIR Cmake Variables

The following CMake variables from LLVM and ONNX-MLIR can be used when compiling ONNX-MLIR.

**MLIR_DIR**:PATH
  Path to to the mlir cmake module inside an llvm-project build or install directory (e.g., c:/repos/llvm-project/build/lib/cmake/mlir).
  This is required if **MLIR_DIR** is not already set from a previous cmake invocation.

**LLVM_EXTERNAL_LIT**:PATH
  Path to the lit tool. Defaults to an empty string and LLVM will find the tool based on **MLIR_DIR** if possible.
  This is required when **MLIR_DIR** points to an install directory.



[./Instrumentation.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Instrumentation

Instrumentation is prototyped in onnx-mlir and can be used to debug runtime issue.

## Compile for instrumentation

By default, instrumentation is turned off. You need to use following command line options to turn it on. The pass for instrumentation will be inserted in some stages by using `--instrument-stage` option. For example, when you specify `Onnx`, the instrumentation will be inserted after onnx-to-onnx conversion to get onnx-level profiling. The `--instrument-ops` option is an option to specify operations to be instrumented. You can use `onnx.Conv` for onnx Conv operations for example. Also, you can use asterisk such as `onnx.*` for all onnx operations, and specify two expressions with `,` such as `onnx.Conv,onnx.Add` for both Conv and Add operations. The `--InstrumentBeforeOp` and `--InstrumentAfterOp` are options to insert instrumentation before and/or after the specified operations. When you use `--instrument-ops=onnx.* --InstrumentBeforeOp --InstrumentAfterOp`, the instrumantation will be inserted before and after all onnx operations.
For NNPA, additional stages for `ZHigh` and `ZLow` are provided. You can get profile for onnx and zhigh ops using `--instrument-stage=ZHigh` and `--instrument-ops=onnx.*,zhigh.*`, and for zlow ops using `--instrument-stage=ZLow` and `--instrument-ops=zlow.*`.

```
  --instrument-stage=<value>                        - Specify stage to be instrumented:
    =Onnx                                             -   Profile for onnx ops. For NNPA, profile onnx ops before lowering to zhigh.
    =ZHigh                                            -   NNPA profiling for onnx and zhigh ops.
    =ZLow                                             -   NNPA profiling for zlow ops.

  --instrument-ops=<string>                         - Specify operations operations to be instrumented:
                                                      "NONE" or "" for no instrument,
                                                      "ops1,ops2, ..." for the multiple ops.
                                                      e.g. "onnx.Conv,onnx.Add" for Conv and Add ops.
                                                      Asterisk is also available.
                                                      e.g. "onnx.*" for all onnx operations.

  Specify what instrumentation actions at runtime:
      --InstrumentBeforeOp                          - insert instrument before op,
      --InstrumentAfterOp                           - insert instrument after op,
      --InstrumentReportTime                        - instrument runtime reports time usage,
      --InstrumentReportMemory                      - instrument runtime reports memory usage.
```

Currently, the call of initialization, OMInstrumentInit, need to be added before you load the dynamic library. It is being considered to add it to the beginning of main_graph by compiler. 

## Run with instrumentation
Run the model in the same way as usual.
The instrumentation library will print out the time and memory usage along at each instrumentation point.
For example, a model, `mymodel.onnx`, is compiled with `onnx-mlir  --instrument-stage=Onnx --instrument-ops=onnx.* --InstrumentAfterOp --InstrumentReportMemory --InstrumentReportTime mymodel.onnx`.
Its runtime output is listed below:

```
==PERF-REPORT==, onnx.Cast, bert/encoder/Reshape__27, before, 0.000001, 1692654182.738546
==PERF-REPORT==, onnx.Cast, bert/encoder/Reshape__27, after, 0.000001, 1692654182.738547
==PERF-REPORT==, onnx.Concat, bert/encoder/Reshape__27, before, 0.000000, 1692654182.738547
==PERF-REPORT==, onnx.Concat, bert/encoder/Reshape__27, after, 0.000001, 1692654182.738548
==PERF-REPORT==, onnx.Reshape, bert/encoder/Reshape, before, 0.000001, 1692654182.738549
==PERF-REPORT==, onnx.Reshape, bert/encoder/Reshape, after, 0.000001, 1692654182.738550
```

The output for the time measurement is explained here.
* The first column is a string to identify the performance being gathered, `PERF-REPORT` here.
* The second column is the name of op.
* The third column is the node name of op. This is displayed when the op has `onnx_node_name` attribute.
* The fourth column indicates if the time being reported is `before` or `after` the onnx operation being analyzed here.
* The fifth column indicates the elapsed time since the previous instrumentation point.
* The sixth column indicates the accumulated: time, in second, from instrumentationInit.

The output for the memory measurement is explained here.
* First column is a string to identify the performance being gathered, `MEM-REPORT` here.
* The second and third column are defined as above.
* The fourth column indicates VMem, the virtual memory size (in kb) used by this process.

Other example for NNPA
- Performance profiling for onnx ops before lowering to zhigh ops:
  `onnx-mlir --march=z16 --maccel=NNPA --instrument-stage=Onnx --instrument-ops=onnx.* --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime mymodel.onnx`
- Performance profiling for onnx and zhigh ops:
  `onnx-mlir --march=z16 --maccel=NNPA --instrument-stage=ZHigh --instrument-ops=onnx.*,zhigh.* --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime mymodel.onnx`
- Performance profiling for zlow ops:
  `onnx-mlir --march=z16 --maccel=NNPA --instrument-stage=ZLow --instrument-ops=zlow.* --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime mymodel.onnx`

## Control instrument at runtime
By providing certain env variable at runtime, you can disable reports from  instrument library.
* If the environment variable `ONNX_MLIR_NO_INSTRUMENT` is set, no report at all
* If the environment variable `ONNX_MLIR_NO_INSTRUMENT_TIME` is set, the report of time usage is disabled
* If the environment variable `ONNX_MLIR_NO_INSTRUMENT_MEMORY` is set, the report of memory usage is disabled
* If the environment variable `ONNX_MLIR_INSTRUMENT_FILE` is set, then this variable provide the file name in which to save the instrumentation.

Please note that the only way to enable instrumentation is to request it at compile time. If none of the detailed report (such as time and memory so far) is turned on at runtime, progress of instrument point will still be print out. This feature is thought to be useful as progress indicator. To fully disable any outputs requested at compile time, you must set `ONNX_MLIR_NO_INSTRUMENT`.

## Used in gdb
The function for instrument point is called `OMInstrumentPoint`. Breakpoint can be set inside this function to kind of step through onnx ops.


[./Testing.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Testing

In onnx-mlir, there are three types of tests to ensure correctness of implementation:
1. [ONNX Backend Tests](#onnx-backend-tests)
2. [LLVM FileCheck Tests](#llvm-filecheck-tests)
3. [Numerical Tests](#numerical-tests)
4. [Use gdb](#use-gdb)
4. [ONNX Model Zoo](#onnx-model-zoo)

## ONNX Backend Tests

Backend tests are end-to-end tests for onnx-mlir based on onnx node and model tests. They are available for testing both the C/C++ .so library and the JNI .jar archive. For each C/C++ test target, adding the `-jni` suffix gives the corresponding JNI test target.
To invoke the test, use the following command:

```
cmake --build . --config Release --target check-onnx-backend[-jni]
``` 
Packages, such as third_party/onnx, needs to be installed to run the backend test. You can install your own onnx package with command `pip install your-onnx-mlir/third_party/onnx`.
JNI test requires the jsoniter jar which is downloaded from its maven repository by default if no installed version is found on the system. If the user turns on the cmake option `ONNX_MLIR_BUILD_JSONITER` when building ONNX-MLIR, the jsoniter jar will be built locally from the source cloned from its github repository. Note that building jsoniter jar locally requires the maven build tool to be installed.

All the test cases provided by onnx package are listed in file `test/backend/all_test_names.txt`. check-onnx-backend will selectively run some of them. 
The node and model tests in onnx that will be run by check-onnx-backend is defined by variable test_to_enable in `test/backend/test.py`. User can test one test case by environment variable `TEST_CASE_BY_USER`. For example,
```
TEST_CASE_BY_USER=selected_test_name cmake --build . --config Release --target check-onnx-backend[-jni]
```
With `TEST_CASE_BY_USER` specified, the intermediate result, the .onnx file and .so file, are kept in `build/test/backend` for debugging. If you need to check whether a particular instruction is included in the generated shared library, set the environment variable `TEST_INSTRUCTION_CHECK` to true and add the instruction name after the test name, like `TEST_CASE_BY_USER=selected_test_name,instruction_name`.
Please note to add suffix `_cpu` to the onnx test name.

### Test cases supported by ONNX

File, test/backend/all_test_names.txt, contains all the test cases provided
by ONNX package. You can enable a test case by adding it into test/backend/inference_backend.py.
The all_test_names.txt is automatically generated with command "make check-onnx-backend-case". The update is only needed when ONNX package is upgraded.

### Adding ONNX-supported test cases to the current set of backend tests

When the ONNX-to-Krnl conversion of an operator is added, the corresponding backend tests for this operator should be added to test.py. The available test cases can be found in `third_party/onnx/onnx/backend/test/case/node`. You can identify new tests by looking for the new operator in `test/backend/all_test_names.txt`. Once you have located new tests, you may add the new tests in the `test/backend/inference_backend.py.` Please note to add suffix `_cpu` to the onnx test name. Associated with the test, you can define how to run the tests for the new operator. For example:
```
        "test_and2d_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
```
indicates that the test `test_and2d_cpu` can run (1) with static shape, (2) with all of its inputs forced to be dynamic shapes, or (3) with all of its input forced to be defined constants. This is the recommended setting for most operators. However, some do not tolerate dynamic shapes for certain arguments; for these, one can explicitly decide which argument to the function can be of dynamic shape. This is specified with the `{-1:{-1}}` expression. The `test/backend/inference_backend.py.` file contains explicit instructions on how to specify which argument and/or argument dimensions can be set to dynamic.

### Tests with unknown dimensions

Testing with dynamic tensor sizes is most easily performed by using the following command, also used by our checkers. 
```
cmake --build . --config Release --target check-onnx-backend-dynamic[-jni]
``` 

The onnx node tests usually have known dimension size for input tensors. So, to test tensor with unknown dimension, the model importer (Build/FrontendONNXTransformer.cpp) provides a functionality to generate such cases. When the environment variable, `IMPORTER_FORCE_DYNAMIC`, is set, the frontend import will turn the all the dimensions (by default) of all the input tensors of the model into -1. For example,
```
IMPORTER_FORCE_DYNAMIC='-1:-1' all dimensions of all the inputs will be changed
IMPORTER_FORCE_DYNAMIC='0:-1' all dimensions of the first input will be changed
IMPORTER_FORCE_DYNAMIC='0:-1|1:0,1' all dimensions of the first input and the 1st and 2nd dimensions of the second input will be changed
```

The Backus-Naur Form (BNF) for `IMPORTER_FORCE_DYNAMIC` is as follows.
```
<ImportForceDynamicExpr> :== `'` <expr> `'`
                  <expr> ::= <inputString> | <inputString> `|` <expr>
            <inputString ::= <inputIndex> `:` <dimString>
             <dimString> ::= <dimIndex> | <dimIndex> `,` <dimString>
            <inputIndex> ::= <index>
              <dimIndex> ::= <index>
                 <index> ::= -1 | <number>
                <number> ::= <digit> | <digit><number>
                 <digit> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
```
Value `-1` semantically represents all inputs or all dimensions, and it has the highest priority. E.g. `'0: -1, 0'` means all dimensions of the first input will be changed. Input and dimension indices start from 0.

For example, the default model for test_add_cpu is:
```
func @main_graph(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
```
with `IMPORTER_FORCE_DYNAMIC='-1:-1'`, the result is:
```
func @main_graph(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
```
with `IMPORTER_FORCE_DYNAMIC='0:-1'`, the result is:
```
func @main_graph(%arg0: tensor<?x?x?xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
```
with `IMPORTER_FORCE_DYNAMIC='0:0,2|1:1'`, the result is:
```
func @main_graph(%arg0: tensor<?x4x?xf32>, %arg1: tensor<3x?x5xf32>) -> tensor<3x4x5xf32>
```
This is a way to use existing node test for dynamic tensors. Since not all test case can pass with dynamic tensor, there is a list in test/backend/test.py, test_not_for_dynamic, to specify which test can not pass with `IMPORTER_FORCE_DYNAMIC` is defined.

### Tests with constant inputs

Because the onnx node tests accepts input tensors at runtime, the inputs are not
constants when compiling the onnx model. However, in practice, inputs can be
constants and we want to test such a situation.

Testing with constant inputs is most easily performed by using the following
command, also used by our checkers.
```
cmake --build . --config Release --target check-onnx-backend-constant[-jni]
```

To test a single onnx node, e.g. `test_add_cpu`, use two environment variables
`TEST_CONSTANT` and `IMPORTER_FORCE_CONSTANT`, e.g.:
```
TEST_CONSTANT=true IMPORTER_FORCE_CONSTANT="0" TEST_CASE_BY_USER=test_add_cpu make check-onnx-backend[-jni]
```
which turns the first input (index 0) to a constant, and thus the model now has
only one input instead of two.

The environment variable `IMPORTER_FORCE_CONSTANT` is a list of indices
separated by `,` (starting from 0, or -1 for all input indices), e.g. `0, 2, 3`
or `-1`.

### Input Signature tests

Testing input signature of an onnx models with a variety of data type by using the following command, also used by our checkers.

```
cmake --build . --config Release --target check-onnx-backend-signature
```

### Enable SIMD instructions

On supported platforms (currently s390x z14 and up, x86, and arm), backend tests can generate SIMD instructions for the compiled models. To enable SIMD, set the TEST_MARCH environment variable, e.g.,
```
TEST_MARCH=z16 cmake --build . --config Release --target check-onnx-backend[-jni]
```

### Execution of backend tests

A tool defined in `utils/RunONNXLib.cpp` can be used to easily execute files from their `.so`
models, such as the ones generated using the
`TEST_CASE_BY_USER=selected_test_name make check-onnx-backend` command.
Models can also be preserved when built in other manners by setting the
`overridePreserveFiles` value in the `onnx-mlir/src/Compiler/CompilerUtils.cpp` file to
`KeepFilesOfType::All`, for example.

When the onnx model is older than the current version supported by onnx-mlir, 
onnx version converter can be invoked with environment variable `INVOKECONVERTER` set 
to true. For example, converter will be called for all test cases for 
`INVOKECONVERTER=true make check-onnx-backend`. 
In test.py, there is a list called `test_need_converter` for you to invoke converter on individual cases.

The tool directly scans the signature provided by the model, initializes the needed inputs with random
values, and then makes a function call into the model. The program can then be used in conjunction
with other tools, such as `gdb`, `lldb`, or `valgrind`.
To list the utility options, simply use the `-h` or `--help` flags at runtime.

We first need to compile the tool, which can be done in one of two modes.
In the first mode, the tool is compiled with a statically linked model.
This mode requires the `-D LOAD_MODEL_STATICALLY=0` option during compilation in addition to including the `.so` file.
Best is to use the `build-run-onnx-lib.sh` script in the `onnx-mlir/utils` directory to compile the tool with its model, which is passed as a parameter to the script.
To avoid library path issues on Mac, run the compiled tool in the directory where the model was built.

``` sh
# Compile tool with model.
cd onnx-mlir/build
sh ../utils/build-run-onnx-lib.sh test/backend/test_add/test_add.so
# Run the tool to run the model (substitute `Release` for `Debug` for the release version).
Debug/bin/run-onnx-lib
# or, on Mac, run the tool in the directory where the model was built
(cd test/backend; ../../Debug/bin/run-onnx-lib)
# if test_add.so was built in `test/backend`:
cd test/backend; ../../Debug/bin/onnx-mlir --EmitLib test_add/test_add.onnx
```
(You can see the path of the library with `otool -L test_add.so` on Mac.)

In the second mode, the tool is compiled without models, which will be passed at runtime.
To enable this option, simply compile the tool with the `-D LOAD_MODEL_STATICALLY=1` option.
You may use the same script as above but without arguments. The tool can then be be run from
any directories as long as you pass the `.so` model file at runtime to the tool.

``` sh
# Compile tool without a model.
cd onnx-mlir/build
sh ../utils/build-run-onnx-lib.sh
# Run the tool with an argument pointing to the model.
Debug/bin/run-onnx-lib test/backend/test_add/test_add.so
```

## LLVM FileCheck Tests

We can test the functionality of one pass by giving intermediate representation
as input and checking the output IR with LLVM FileCheck utility.
For example, we have a test case, test.mlir,  for shape inference.
```
func @test_default_transpose(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
```

You can run the shape inference pass  on this test case, and get the following 
output:
```
module  {
  func @test_default_transpose(%arg0: tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [3, 2, 1, 0]} : (tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32>
    return %0 : tensor<32x1x5x5xf32>
  }
}
```
Manually check whether the output is correct.
If the output is correct, cover the output to what can be automatically checked
in future. Use command:
```
Debug/bin/onnx-mlir-opt --shape-inference test.mlir | python ../utils/mlir2FileCheck.py 
```
You will get the following:
```
// mlir2FileCheck.py
// CHECK-LABEL:  func @test_default_transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [3, 2, 1, 0]} : (tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32>
// CHECK:           return [[VAR_0_]] : tensor<32x1x5x5xf32>
// CHECK:         }
```
Combine the source and the check code and add to the adequate test cases. 
All the test cases for onnx dialect are collected under test/mlir/onnx directory.
These test cases can be invoked with `make check-onnx-lit`. 
This target is an essential requirement for a build.

## Numerical Tests

Numerical tests are used to test for numerical correctness in addition to the tests provided by the ONNX package.
The goal is to provide extensive numerical value based unit tests; this is very important for ensuring that
optimization transformations are valid and correct: more corner cases will arise as we specialize for specific 
architecture parameters (like vector width). Numerical tests generates extensive amount of numerical value-based 
unit tests based on simple, naive (and extremely slow) implementation of operations being tested, used to verify 
the correctness of our operation lowering and optimization.

Numerical tests should be structured such that the following two components are independent and separate:
- Generation of test case parameters (for instance, the dimensions of convolutions N, C, H, W, kH, kW ...).
- Checking that the values produced by onnx-mlir is consistent with those produced by naive implementation.

The motivation is that there are two ways we want to generate test case parameters:
- Exhaustive generation of test case parameters. Where we want to exhaustively test the correctness of a small range
of parameters (for instance, if we would like to test and verify that 3x3 convolution is correctly implemented for
all valid padding configurations.)
- When the possible parameter space is extremely large, we can rely on RapidCheck to randomly generate test cases
that becomes increasingly large as smaller test cases succeed. And it also automatically shrinks the test cases
in the event that an error occurs. For example, the following RapidCheck test case automatically generates test
case parameters (N from between 1 and 10, C from within 1 and 20 etc...). By default rc::check will draw 100 sets of
test case parameters and invoke the value checking function `isOMConvTheSameAsNaiveImplFor`.

```cpp
  // RapidCheck test case generation.
  bool success = rc::check("convolution implementation correctness", []() {
    const auto N = *rc::gen::inRange(1, 10);
    const auto C = *rc::gen::inRange(1, 20);
    const auto H = *rc::gen::inRange(5, 20);
    const auto W = *rc::gen::inRange(5, 20);

    const auto kH = *rc::gen::inRange(1, 15);
    const auto kW = *rc::gen::inRange(1, 15);

    // We don't want an entire window of padding.
    const auto pHBegin = *rc::gen::inRange(0, kH - 1);
    const auto pHEnd = *rc::gen::inRange(0, kH - 1);
    const auto pWBegin = *rc::gen::inRange(0, kW - 1);
    const auto pWEnd = *rc::gen::inRange(0, kW - 1);

    // Make sure we have at least 1 output per dimension.
    RC_PRE((H >= kH) && (W > kW));

    RC_ASSERT(isOMConvTheSameAsNaiveImplFor(
        N, C, H, W, kH, kW, pHBegin, pHEnd, pWBegin, pWEnd));
  });
  assert(success && "error while performing RapidCheck tests");
```
  
Sometimes it is convenient to be able to see the mlir files associated with a
numerical tests. To do so, the easiest is to set the `overridePreserveFiles`
variable in `src/Compiler/CompilerUtils.cpp` to the types of files that you want to
preserve (e.g. `KeepFilesOfType::All`). Then, no matter how you compile
your model, input and output mlir files will be preserved, as well as
unoptimized and optimized bytecode files as well as a few additional binaries.

In case of failures, both RapidCheck (infrastructure used for numerical testing) and the onnx models allow a user to re-run a test with the same values. When running a test, you may get the following output.
```
Model will use the random number generator seed provided by "TEST_SEED=1440995966"
RapidCheck Matrix-Vector test case generation.
Using configuration: seed=4778673019411245358
```

By recording the seed values in the following two environment variables:
```
export RC_PARAMS="seed=4778673019411245358"
export TEST_SEED=1440995966
```
you can force, respectively, the random seeds used in RapidCheck and the random seeds used to populate the ONNX input vectors to be the same. Set only the first one (`RC_PARAMS`) and you will see the same test configurations being run but with different input values. Set both and you will see the same configuration and the same input being used for a completely identical run.

If you need to change ATOL and RTOL for accuracy checks, set the environment variables `TEST_ATOL` and `TEST_RTOL` to the new ones.

### Enable SIMD instructions

On supported platforms (currently s390x z14 and up, x86, and arm), numerical tests can generate SIMD instructions for the compiled models. To enable SIMD, set the `TEST_ARGS` environment variable, e.g.,
```
TEST_ARGS="-march=z16" CTEST_PARALLEL_LEVEL=$(nproc) cmake --build . --config Release --target check-onnx-numerical
```

### Testing of specific accelerators

Currently we provide testing for accelerator NNPA. It is described [here](AccelNNPAHowToUseAndTest.md).

## Use gdb
### Get source code for ONNX model
When you compile an ONNX model, add option `--preserveMLIR`. A source code for the  model in MLIR format, named your_model_name.input.mlir,  will be created. The line information for operation will be attached and propagated all the way to binary.
When you run the compiled library in gdb, you can stop in the model and step through with respect to the ONNX operations. Here is an example for model test_add.onnx:

```
$Debug/bin/onnx-mlir --preserveMLIR test_add.onnx
$. ../utils/build-run-onnx-lib.sh
$gdb Debug/bin/run-onnx-lib
(gdb) b run_main_graph
(gdb) run ./test_add.so
(gdb) list
1	builtin.module  {
2	  builtin.func @main_graph(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
3	    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
4	    return %0 : tensor<3x4x5xf32>
5	  }
(gdb) b 3
Breakpoint 2 at 0x3fffdf01778: file /home/chentong/onnx-mlir/build/test_add.input.mlir, line 3.
(gdb) c
Continuing.

Breakpoint 2, main_graph () at /home/chentong/onnx-mlir/build/test_add.input.mlir:3
3	    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
(gdb) n
[Detaching after vfork from child process 2333437]
#  0) before op=     Add VMem:  6804
[Detaching after vfork from child process 2333470]
#  1) after  op=     Add VMem:  6804
4	    return %0 : tensor<3x4x5xf32>
(gdb)
```
Note that the output of instrumentation showed that the gdb step at the onnx op level correctly. You need extra flags for onnx-mlir to run on instrumentation, which is not necessary for gdb. The source file is test_add.input.mlir.
One of furtuer works is to support symbols at onnx level in gdb. It would be really useful if tensors can be printed out in gdb.

## Use LLVM debug support

The standard way to add tracing code in the LLVM and MLIR projects is to use the LLVM_DEBUG macro. Official documentation from LLVM is [here](https://llvm.org/docs/ProgrammersManual.html#the-llvm-debug-macro-and-debug-option).

To insert a single "printout" under debug control, the following template can be used.
```C++
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "my_opt_name_here"
...
LLVM_DEBUG(llvm::dbgs() << "debug msg here" <<  obj_to_print << "\n");
```
To trigger the debug trace one would simply invoke the compiler with --debug-only=my_opt_name_here.

Another macro called `DEBUG_WITH_TYPE` can be used situations where a source file has maybe just one tracing message. In that case you can forgo defining `DEBUG_TYPE` and use the following instead.

```C++
DEBUG_WITH_TYPE("my_debug_msg", llvm::dbgs() << "my trace msg here\n");
```
To protect larger portion of code, this template can be used.
```C++
LLVM_DEBUG({
  for(i...) {
    llvm::dbgs() << "emit trace for a: " << a << "\n";
    compute b;  // should be side effects free
    llvm::dbgs() << "emit trace for 'b':" << b << "\n";
    ...
});
```

Some examples that uses this support in the project are in these files.

* src/Conversion/KrnlToAffine/KrnlToAffine.cpp
* src/Conversion/ONNXToKrnl/Math/Gemm/Gemm.cpp

Again, these debug statements can then be activated by adding the `--debug-only=my_opt_name_here` option to `onnx-mlir` or `onnx-mlir-opt`.

## ONNX Model Zoo

We provide a Python script [RunONNXModelZoo.py](../utils/RunONNXModelZoo.py) to check inference accuracy with models in the [ONNX model zoo](https://github.com/onnx/models).  [RunONNXModelZoo.py](../utils/RunONNXModelZoo.py) requires [RunONNXModel.py](../utils/RunONNXModel.py) to be in the same folder. For example, to check inference accuracy with mnist-8:

```bash
$ mkdir test && cd test
$ ln -s /onnx-mlir/utils/RunONNXModel.py
$ ln -s /onnx-mlir/utils/RunONNXModelZoo.py
$ ONNX_MLIR_HOME=/onnx-mlir/build/Release/ python RunONNXModelZoo.py -m mnist-8 -c="-O3"
```
Run the script with `-h` to see all the options. In addition to the `-m` flag to specify a model and `-c` flag to specify the compile options, useful options are the `-k` flag to leave the onnx model in the current directory as a `.tgz` file, and the `-l debug` flag to print lots of debugging info.

To find out which models are available, run the script with `-p` to print the list of available models; or `-m` followed by an incomplete name, and the script will suggest the exact names. 

Without specifying a model using `-m`, the script will check all models in the ONNX model zoo.

### ONNX Model Zoo Performance analysis

If you want to gather performance info about a model zoo (or any models, for that matter), simplest is to request the desired statistic at compile time (using `-profile-ir` flag), divert the output statistic to a file, and then analyze it using `make-report.py`. For example:
```
> ONNX_MLIR_INSTRUMENT_FILE=run.log RunONNXModelZoo.py -c "-O3 --march=arm64 --profile-ir=Onnx" -m bertsquad-10
...
> make-report.py -r run.log
...
Statistics start (all ops).
  onnx.Add, 112, 0.0130570
  onnx.Cast, 105, 0.0001860
  onnx.Concat, 55, 0.0001290
  onnx.Constant, 473, 0.0008220
```

The runtime profiling info can be combined with specific compile-time statistics as well. Let's say that we are interested in SIMD statistics. We inform the compiler of the compile-time statistic to emit using `-opt-report` option, and inform `RunONNXModelZoo.py` that we want to preserve the compiler output using the `--log-to-file` option. For example
```
> ONNX_MLIR_INSTRUMENT_FILE=run.log RunONNXModelZoo.py -c "-O3 --march=arm64 -opt-report=Simd --profile-ir=Onnx" -m bertsquad-10 --log-to-file compile.log
...
> make-report.py -c compile.log -r run.log
...
Statistics start (all ops).
  onnx.Add-simd, 112, 0.0130570
  onnx.Cast, 23, 0.0000650
  onnx.Gemm, 1, 0.0003570
  onnx.Gemm-simd, 72, 0.8109330
```
In the listing above, the operations that were vectorized are summarized separately with a  `-simd` postfix appended to their respective operation names.

The same options and environment variables works equally well for `RunONNXModel.py` and `RunONNXModelZoo.py`.


[./Workflow.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->
# Contribution Guide

## Step 1: Fork ONNX-MLIR on the GitHub web interface

We strongly encourage contributors to work in their own forks of the ONNX-MLIR project.

Creating a new fork is easy:

1. Visit https://github.com/onnx/onnx-mlir
2. Click `Fork` button (top right) to establish a fork.
3. Navigate to your newly created fork, click on the the green `Code` button to get the link to *your* newly-created ONNX-MLIR fork:
```sh
git@github.com:<user>/onnx-mlir.git
```
or
```sh
https://github.com/<user>/onnx-mlir.git
```

where `<user>` is your GitHub username.

## Step 2: Setup MLIR

Depending on whether you are using docker or not, either follow Step 2a or Step 2b below.

### Step 2a: Setup using Docker

Use the template provided in [here](Docker.md#building-onnx-mlir-in-a-docker-environment) to establish a docker image that uses your ONNX-MLIR fork by modifying it as follows:

1. Since the base image used by the template already contains a clone of the ONNX-MLIR main repository, in step 5, add your fork as a remote repository by uncommenting:
```sh
RUN git remote add origin https://github.com/<user>/onnx-mlir.git
```

Replace `<user>` with your GitHub user name.

As a best practice, uncomment the line which disables the pushing to upstream to avoid accidental pushes:
```sh
RUN git remote set-url --push upstream no_push
```

At the end of the commands in Step 5:
- `upstream` will refer to the original ONNX-MLIR repository.
- `origin` will refer to your own fork of ONNX-MLIR.


2. Uncomment either step 3 or 4 depending on whether you plan to use VSCode in conjunction with the ONNX-MLIR image.

3. By default, ONNX-MLIR is built in `Debug` mode. Make the appropriate changes in step 6 if you wish to build ONNX-MLIR in `Release` mode.

At any point you can access your Docker image interactively:
```sh
docker run -it myImageName /bin/bash
```

Once inside the image you can navigate to the ONNX-MLIR GitHub repository:
```sh
cd /workdir/onnx-mlir
```

Once inside the repository you can interact with Git via the usual Git commands.


### Step 2b: Setup without Docker

Define a local working directory:

```sh
working_dir={your working directory}
```

Then follow the directions in this section of the top level [README](../README.md) and OS specific instructions [Linux](BuildOnLinuxOSX.md#MLIR) or [Windows](BuildOnWindows.md#MLIR) for installing the currently supported MLIR version in your working directory.

If you already have an MLIR copy in your working directory, you should ensure that you have the latest copy. To do so, compare the most recent commit ID from a `git log` command with the specific branch version extracted by the `git checkout` command listed [here](BuildOnLinuxOSX.md#MLIR). If your MLIR in not up to date, you must  bring it up to the correct commit level by either reinstalling it or updating it with `git fetch`, `git merge`, and `git checkout` commands.

Create your clone (replace `<user>` with your GitHub username):

```sh
mkdir -p $working_dir
cd $working_dir
git clone --recursive https://github.com/<user>/onnx-mlir.git
# or: git clone --recursive git@github.com:<user>/onnx-mlir.git

cd $working_dir/onnx-mlir
git remote add upstream https://github.com/onnx/onnx-mlir.git
# or: git remote add upstream git@github.com:onnx/onnx-mlir.git

# Never push to upstream main since you do not have write access.
git remote set-url --push upstream no_push

# Confirm that your remotes make sense:
# It should look like:
# origin    https://github.com/$user/onnx-mlir.git (fetch)
# origin    https://github.com/$user/onnx-mlir.git (push)
# upstream  https://github.com/onnx-mlir/onnx-mlir.git (fetch)
# upstream  no_push (push)
git remote -v
```

## Step 3: Understanding the repository structure

Regardless of whether you are using a Docker image or not, the steps below are again common to both environments.

At the end of the repository setup commands above:
- `upstream` will refer to the original ONNX-MLIR repository.
- `origin` will refer to your own fork of ONNX-MLIR.

Never commit anything to your fork's `main` branch, the only way you should update `main` is from `upstream`. The procedure to update your fork's `main` branch is listed in Step 4.

## Step 4: Keeping your repository up to date

To keep your ONNX-MLIR fork's `main` up to date perform the following steps:

1. Fetch the latest versions of your fork (`origin`) and the `upstream` repositories:
```sh
git fetch --all
```

2. Update the `main` branch on your fork:
```sh
git checkout main
git merge origin/main
git merge upstream/main
git push origin main
```

Provided you have never committed anything to your fork's `main` branch directly, all the updates to your fork's `main` should be fast forwards.

3. The `main` branch of your fork should now be identical to the `main` branch of `upstream`. To check you can do:
```sh
git diff upstream/main
```
and the command will return immediately signaling that no differences exist between `upstream/main` and `origin/main`

## Step 5: Create a branch for your changes

To create a branch off your fork's `main` branch ensure your current branch is `main` by doing:

```sh
git checkout main
```

Then create your new branch:

```sh
git checkout -b my-branch
```

At this point you are ready to develop the code.

## Step 6: Develop

### Edit your code

You can now edit the code on the `my-branch` branch.

### Run cmake & make

Follow the directions to build ONNX-MLIR for the OS that you are using [Linux](BuildOnLinuxOSX.md#Build) or [Windows](BuildOnWindows.md#Build).

We expect code to compile without generating any compiler warnings.

### Run Test

In general, the new features must be tested in one or more of our test suite.
At a high level, our testing strategy includes `literal` tests (`check-onnx-lit` below), end-to-end tests derived from the ONNX Standard (`check-onnx-backend` and derivatives below, and semi-exhaustive numerical tests (`test` below).

```sh
# Run unit test to make sure all test passed.
make check-onnx-lit
make check-onnx-backend
make check-onnx-backend-dynamic
make check-onnx-backend-constant
make check-onnx-numerical
```
Specific testing help is provided in these pages to [run](TestingHighLevel.md) and[generate new tests](Testing.md).

## Step 7: Commit & Push

ONNX-MLIR requires committers to sign their code using the [Developer Certificate of Origin (DCO)](https://developercertificate.org).
There is a one time setup to register your name and email.
The commands are listed below, where you substitute your name and email address in the "John Doe" fields.

```sh
git config --global user.name "John Doe"
git config --global user.email johndoe@example.com
```

You may also be asked to sign a Developer Certificate of Origin (DCO)
at some times during the PR review.
If you do, you will have to accept in order to contribute code.

Once these initial tasks are done, you are ready to sign your code by using the `-s` flag during your commits.

```sh
git commit -s
```

Push your changes:
```sh
git push origin my-branch
```

Note that even if branches are pushing to one's own fork, the PR will be created on the shared https://github.com/onnx/onnx-mlir/pulls site for everyone to review.

## Step 8: Update your branch

Assuming your `main` is up to date (Step 4), to update any branches you are currently working on to use the latest ONNX-MLIR, you need to do the following:

```sh
git checkout my-branch
git merge origin/main
```

If no conflicts are signaled and the merge is complete do:

```sh
git push origin my-branch
```

However, if conflicts appear, the merge will be interrupted until the conflicts are resolved. A list of files will be marked as containing conflicts. To identify those files do:

```sh
git status -uno
```

The files in red are the files containing conflicts. Go to all the files which contain a conflict and resolve the conflicts.
When the conflicts are resolved do a `git add` on each conflicted file:

```sh
git add path/to/file1
git add path/to/file2
...
```

When all conflicted files have been added do:
```sh
git commit -s
```

Followed by a git push:
```sh
git push origin my-branch
```

Your branch is now up to date with the latest ONNX-MLIR.


## Step 9: Create a pull request

1. Visit your fork at `https://github.com/<user>/onnx-mlir` (replace `<user>` obviously).
2. Click the `Compare & pull request` button next to your `my-branch` branch.

## Step 10: Get a code review

Once your pull request has been opened and is not in draft mode anymore, one of us will review the code.
The reviewer(s) will do a thorough code review, looking for correctness, bugs, opportunities for improvement, testing, documentation and comments, and style.

Commit changes made in response to review comments to the same branch on your fork. Continue to do a sequence of `git commit -s` and `git push` commands (Step 7) to update GitHub of your changes.

If you wish to update your branch to contain the latest ONNX-MLIR changes perform Step 8.

This step can also be performed on the GitHub website by visiting your PR page and clicking the `Update` button. This step will merge the latest `upstream/main` branch into your branch without updating the `main` branch of your fork.

## Step 11: Pull request approval

When the PR has been approved by one or more reviewers and all the CIs have passed, the PR can now be merged into the main branch of ONNX-MLIR.

Your PR will be squashed into a single commit before being merged into the ONNX-MLIR main branch.

This step will be performed by an ONNX-MLIR admin.

By default, the log of your commit will be made to consist of:
- description consisting of the title of your PR
- the reviewer sign-off
- any co-authors

For contributors who wish to provide a custom description you will have to do the squashing of the commits in your PR yourself by performing an interactive rebase on the latest ONNX-MLIR.

For lengthy, detailed descriptions please use the main comment box in your PR.

### Collaborators with Write access guidelines

By default, the log will include the messages of every `commit` performed during the development, which is necessary for smooth reviewing but is unnecessarily long. During the merge phase this message will be replaced with the title of the patch unless the author of the patch has already squashed all his commits via an interactive rebase and provided his own custom (but brief) description of the patch.

Using the GitHub interface:
 1. In the web page associated with the PR, click the `Squash and Merge` button;
 2. In the text box above the green `Confirm squash and merge` button, edit the log.
 3. Ideally, it should have only one short paragraph describing the work, plus the relevant `Sign-off-by` and `Co-authored-by` information. If the user has provided this already do step 4. If not, clear the intermediate commit messages and use the patch title as the description, add sign-off and co-author information. 
 4. Only once the log is properly edited, click on the `Confirm squash and merge` button.

## Code style

Very small PRs are easy to review. Very large PRs are very difficult to review.

Follow the [coding style](https://llvm.org/docs/CodingStandards.html) used by LLVM for your code. We use the `clang-format` command to get the proper style, which is also tested by our CIs. It is acceptable to run the command on all of the files that were modified by your PR. We recommend using VS code where the clang formatter will be run automatically using the clang format configuration file already present in the repository.

For python code, we use the [black](https://pypi.org/project/black) code formatter. You should run the `black` command on all the python code modified by your PR, which must pass the black code formatter CI check before it can be merged.


[./DynamicDimensionAnalysis.md]:

It is often the case where we want to know two dynamic dimensions are equal or not at compile time. This helps with decision on how to lowering an ONNX operator. For example, given an ONNXAddOp as follows:

```mlir
%0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x3x5xf32>, tensor<?x3x5xf32>) -> tensor<?x3x5xf32>
```
If we know at compile time that the first dimensions of `%arg0` and `%arg1` are the same (e.g., coming from the same tensor), there is no need to generate runtime code to handle broadcasting rules.

This also helps generate code for accelerators. If an accelerator does not support broadcasting, we can check at compile to decide whether the ONNXAddOp will be offloaded to the accelerator or not.

We provide a helper class [DimAnalysis](../src/Transform/ONNX/ONNXDimAnalysis.hpp) to analyze dynamic dimensions and to check whether two dynamic dimensions are the same or not. Below is an example of using DimAnalysis:

```C
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

// Run the dynamic dimension analysis to help check equality of dynamic
// dimensions at compile time.
ModuleOp moduleOp = getOperation();
onnx_mlir::DimAnalysis dimAnalysis(moduleOp);
dimAnalysis.analyze();
```

DimAnalysis is constructed for a ModuleOp so that all operations in the ModuleOp will be analyzed.
Then, actual analysis is done via calling `analyze()` function.
After that, we can query if two dynamic dimensions are the same or not via calling
```C
bool sameDim = dimAnalysis.sameDynDim(tensor1, dimAxis1, tensor2, dimAxis2);
```
where the first dynamic dimension is identified by its tensor `tensor1` and its axis `dimAxis1`, and the second dynamic dimension by `tensor2` and `dimAxis2`.

DimAnalysis has been using for NNPA, please see [ONNXToZHigh](../src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.cpp) for more information.


[./BuildOnWindows.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Installation of ONNX-MLIR on Windows

Building onnx-mlir on Windows requires building some additional prerequisites that are not available by default.

Note that the instructions in this file assume you are using [Visual Studio  2019 Community Edition](https://visualstudio.microsoft.com/downloads/) with ninja.
It is recommended that you have the **Desktop development with C++** and **Linux development with C++** workloads installed.
This ensures you have all toolchains and libraries needed to compile this project and its dependencies on Windows.

Run all the commands from a shell started from **"Developer Command Prompt for VS 2019"**.

## Protobuf
Build protobuf as a static library.

[same-as-file]: <> (utils/install-protobuf.cmd)
```shell
REM Check out protobuf v21.12
set protobuf_version=21.12
git clone -b v%protobuf_version% --recursive https://github.com/protocolbuffers/protobuf.git

set root_dir=%cd%
md protobuf_build
cd protobuf_build
call cmake %root_dir%\protobuf\cmake -G "Ninja" ^
   -DCMAKE_INSTALL_PREFIX="%root_dir%\protobuf_install" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -Dprotobuf_BUILD_EXAMPLES=OFF ^
   -Dprotobuf_BUILD_SHARED_LIBS=OFF ^
   -Dprotobuf_BUILD_TESTS=OFF ^
   -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ^
   -Dprotobuf_WITH_ZLIB=OFF

call cmake --build . --config Release
call cmake --build . --config Release --target install
```

Before running CMake for onnx-mlir, ensure that the bin directory to this protobuf is before any others in your PATH:
```shell
set PATH=%root_dir%\protobuf_install\bin;%PATH%
```

If you wish to be able to run all the ONNX-MLIR tests, you will also need to install the matching version of protobuf through pip. Note that this is included in the requirements.txt file at the root of onnx-mlir, so if you plan on using it, you won't need to explicitly install protobuf.
```shell
python3 -m pip install protobuf==4.21.12
```

#### MLIR
Install MLIR (as a part of LLVM-Project):

[same-as-file]: <> (utils/clone-mlir.sh)
```shell
git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout b270525f730be6e7196667925f5a9bfa153262e9 && cd ..
```

[same-as-file]: <> (utils/build-mlir.cmd)
```shell
set root_dir=%cd%
md llvm-project\build
cd llvm-project\build
call cmake %root_dir%\llvm-project\llvm -G "Ninja" ^
   -DCMAKE_INSTALL_PREFIX="%root_dir%\llvm-project\build\install" ^
   -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" ^
   -DLLVM_TARGETS_TO_BUILD="host" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -DLLVM_ENABLE_ASSERTIONS=ON ^
   -DLLVM_ENABLE_RTTI=ON ^
   -DLLVM_ENABLE_ZLIB=OFF ^
   -DLLVM_INSTALL_UTILS=ON ^
   -DENABLE_LIBOMPTARGET=OFF ^
   -DLLVM_ENABLE_LIBEDIT=OFF

call cmake --build . --config Release
call cmake --build . --config Release --target install
call cmake --build . --config Release --target check-mlir
```

## ONNX-MLIR (this project)

### Build
The following environment variables can be set before building onnx-mlir (or alternatively, they need to be passed as CMake variables):
- MLIR_DIR should point to the mlir cmake module inside an llvm-project build or install directory (e.g., c:/repos/llvm-project/build/lib/cmake/mlir).

This project uses lit ([LLVM's Integrated Tester](https://llvm.org/docs/CommandGuide/lit.html)) for unit tests. When running CMake, we can specify the path to the lit tool from LLVM using the LLVM_EXTERNAL_LIT define, as in the example below. If MLIR_DIR points to an install directory of llvm-project, LLVM_EXTERNAL_LIT is required and %lit_path% should point to a valid lit. It is not required if MLIR_DIR points to a build directory of llvm-project, which will contain lit.

To build ONNX-MLIR, use the following commands:

[same-as-file]: <> ({"ref": "utils/build-onnx-mlir.cmd", "skip-doc": 2})
```shell
git clone --recursive https://github.com/onnx/onnx-mlir.git

set root_dir=%cd%

md onnx-mlir\build
cd onnx-mlir\build
call cmake %root_dir%\onnx-mlir -G "Ninja" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -DCMAKE_PREFIX_PATH=%root_dir%\protobuf_install ^
   -DLLVM_EXTERNAL_LIT=%lit_path% ^
   -DLLVM_LIT_ARGS=-v ^
   -DMLIR_DIR=%root_dir%\llvm-project\build\lib\cmake\mlir ^
   -DONNX_MLIR_ENABLE_STABLEHLO=OFF ^
   -DONNX_MLIR_ENABLE_WERROR=ON

call cmake --build . --config Release
```
After the above commands succeed, an `onnx-mlir` executable should appear in the `Debug/bin` or `Release/bin` directory.

### Trouble shooting build issues

Check this [page](TestingHighLevel.md) for helpful hints.


[./SequenceType.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Handle ONNX Sequence Type

## ONNX Sequence Type
ONNX sequence type is a type for aggregation of values. It can be sequence of 
Tensor type, or sequence of Map type in ONNX. Currently onnx-mlir supports only sequence of tensor.
In ONNX dialect defined in onnx-mlir, the sequence type is defined as `SeqType`,
and shown as `!onnx.Seq<T>` in .mlir files. There are two access function defined for sequence type:
- Type elementType(). The type of the elements in the sequence. When the elements are 
  tensors with different shape, the type of elements has to be a super type of
  each elements. Shape inference will take care the type merging and refining. 
- int64_t length(). The number of elements in the sequence. -1 for statically unknown.

There are 4 basic sequence-related operations in ONNX:
- SequenceEmpty: create an empty sequence with certain element type
- SequenceInsert: add an element into the input sequence at specified position and return the result as a new sequence
- SequenceConstruct: construct a new sequence from the input elements.
- SequenceErase: remove an element at a specified position from the input sequence and return the result as a new sequence

## Lower ONNX Sequence Type to memref
Sequence type is an indexed container type to/from which an element can be stored
or loaded at a specified position, similar to 'std::vector<T>', or 'MemRefType' in MLIR..
Due to the SSA semantics of ONNX operations, a sequence is created once and is not further modified.
The container for sequence type should have a fixed size.
In onnx-mlir, tensor is lowered to memref. We choose to lower ONNX Sequence tye of tensor to
'memref<?xmemref<*xT>>'.
The outer memref in memref<?xmemref<*xT>> is a 1D memref for the 
sequence. The dim size of this memref is the length of the sequence.
The inner memref type is for the element type. It should the super type of all possible
element types, as discussed in the previous session.

The advantange is that we can make use of the memref dialect without introducing external
data structure. [reference other work]. The same optimization over MemRefType can be used
on tensor operations and sequence operations.
The store/load operation of a memref (for element) to/from a memref of memref (for the sequence) is directly supported by MLIR. The index for the sequence position will be the index
for the memref for sequence.
The following code is llvm code for storing a memref into a memref of memref (such as `memref.store %1, %2[%3] : memref<?xmemref<?xf32>>`)
```
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.getelementptr %13[%12] : (!llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    // The struct is the descriptor for the element memref
    // The first two fields are the pointer and aligned pointer for the data.
    // The rest of field is for the shape information
    llvm.store %11, %14 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
```
In the store, the descriptor for the element memref containing the dynamic shape
informantion and data pointer of the memref, is stored into the memref of memref.
Please note the content of the element memref is not stored.  
Correspondingly, the load will construct a memref from the memref of memref.
When an element loaded from sequence, the descriptor with the dynamic info is 
loaded from the memref of memref, while its static type is the element type
of the sequence.

The basic operations seem to work.  The sequence related operations in ONNX can be easily lowered using memref allocation, load or store.  However, there is an issue with buffer deallocation.

## Issues with deallocation
onnx-mlir relies on MLIR [Bufferization::Deallocation pass](https://mlir.llvm.org/docs/BufferDeallocationInternals/) to insert deallocation for memrefs. 
When a memref for the element is stored into a sequence, its data pointer along with shape
information is store and the stored memref is invisible in the operation graph.
This operation breaks the assumption of value based SSA assume for MLIR.
As a result, the deallocation pass
will add a deallocation for the element memref after its last visible use. 
Consequently, when this element is loaded from the sequence, the memref will 
have a dangling pointer to its data.
The source of this issue is that the data pointer for element is saved in the 
sequence. This operation breaks the basic assume of operations on "values".
Another issue is with the deallocation of memref<memref> for sequence. 
If the memref for the element were not freed by deallocation pass, there would be issue
on deallocation of the memref for sequence: deep deallocation for the elements in the
sequence is needed.

## Solution
We could extend the deallocation pass to handle the load/store of memref<memref<T>>.
When a source memref is stored into a destination memref, the source memref 
could be marked as `escaped` and then no deallocation would be added by the 
deallocation pass. Such change will involve how to add clone op with the present of 
control flow. Our current solution is based the existing deallocation pass.

### Store an element into a sequence
To avoid the deallocation of the element, we can save a copy of the element into the
sequence. To generate this copy, we need first to allocate a memref and then use memref.copy
to copy the value. However, deallocation pass may again add deallocation to
the newly allocated memref. To avoid this issue, we should wrap all the operations of
memref the allocation, copy and store into one krnl Op, krnlSeqStoreOp, which will be
lowered AFTER the deallocation pass.

Since the type of sequence element is a super type for all possible elements, memref.cast may be
needed before the store. KrnlSeqStoreOp will be lowered to the code segment below.
```
// The input op
// "krnl.seqstore"(%seq, %element, %pos) : (memref<?xmemref<?x2xf32>>, memref<3x2xf32>, index) -> ()
// Notice that the element type of seq is memref<?x2xf32>
// and the insert element type is memref<3x2xf32>
// The output result:
      %33 = memref.alloc(%32) {alignment = 16 : i64} : memref<3x2xf32>
      memref.copy %element, %33 : memref<3x2xf32> to memref<3x2xf32>
      %34 = memref.cast %33 : memref<3x2xf32> to memref<?x2xf32>
      memref.store %34, %seq[%pos] : memref<?xmemref<?x2xf32>>
```

### Allocate a sequence
Though the basic operation of sequence allocation is just memref.alloc, we introduced
KrnlSeqAllocOp so that we can define a customized deallocation function for sequence.
We use interface in MLIR Bufferization to specify that KrnlSeqAllocOp has allocation 
traits and a customized free function, which will perform a deep deallocation for the
elements as well as the sequence itself. Currently, the KrnlSeqDeallocOp is used for the
deallocation and it will be lowered to scf and memref after deallocation pass.

### Load an element from a sequence
A memref.load could be used to load an element from a sequence. But the loaded memref may
have a life span longer than the sequence itself, and will have dangling data pointer after
the sequence has been freed.

To overcome this issue, KrnlSeqExtractOp is introduced. This Op will use memref.load to
load the element, then use allocate a new memref and copy the data, and finally return
the copied memref.  This Op is marked with allocation interface and the deallocation pass will insert deallocation for the returned memref automatically. 

### Construct a new sequence from an old sequence
The sequence ops, SequenceInsert and SequenceErase, construct a new sequence
with the elements from an old sequence. SequenceInsert constructs a new sequence
by inserting an element at specified position into the input sequence, while SequenceErase constructs 
a new sequence by deleting an elment at a specified position from the input sequence. Other than the modified element, the elements in the input sequence need 
need to be copied into the new sequence.
It is correct to use KrnlSeqExtractOp to load an
element from the input sequence and use KrnlSeqStoreOp to store it into the new sequence. But there will be
two copying operations for each element. Since it is known that the loaded
element is only used within the sequence constructing process and the input
sequence is guaranteed to be alive (not deallocated), a regular memref.load 
can be used, instead of KrnlSeqExtract, to save one copying. This is a simple optimization for sequence lowering.

## Example

Original .mlir code:

```
func.func @test_sequence_insert(%arg0: !onnx.Seq<tensor<?x4x5xf32>>, %arg1:tensor<3x4x5xf32>) -> tensor<3xi64>  {
  %0 = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<i64>
  %1 = "onnx.Add"(%arg1, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
  %2 = "onnx.NoValue"() {value} : () -> none
  %6 = "onnx.SequenceInsert"(%arg0, %1, %2) : (!onnx.Seq<tensor<?x4x5xf32>>, tensor<3x4x5xf32>, none) -> !onnx.Seq<tensor<?x4x5xf32>>
  %4 = "onnx.SequenceAt"(%6, %0) : (!onnx.Seq<tensor<?x4x5xf32>>, tensor<i64>) -> tensor<?x4x5xf32>
  %5 = "onnx.Shape"(%4) : (tensor<?x4x5xf32>) -> tensor<3xi64>
  return %5 : tensor<3xi64>
}
```

After --convert-onnx-to-krnl pass

```
  func.func @test_sequence_insert(%arg0: memref<?xmemref<?x4x5xf32>>, %arg1: memref<3x4x5xf32>) -> memref<3xi64> {

    // onnx.Add
    %1 = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
    %2:3 = krnl.define_loops 3
    krnl.iterate(%2#0, %2#1, %2#2) with (%2#0 -> %arg2 = 0 to 3, %2#1 -> %arg3 = 0 to 4, %2#2 -> %arg4 = 0 to 5){
      %22:3 = krnl.get_induction_var_value(%2#0, %2#1, %2#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      ...
      %23 = krnl.load %arg1[%22#0, %22#1, %22#2] : memref<3x4x5xf32>
      %24 = krnl.load %arg1[%22#0, %22#1, %22#2] : memref<3x4x5xf32>
      %25 = arith.addf %23, %24 : f32
      krnl.store %25, %1[%22#0, %22#1, %22#2] : memref<3x4x5xf32>
    }

    // Sequence Insert
    %6 = "krnl.seqalloc"(%5) : (index) -> memref<?xmemref<?x4x5xf32>>
    %c0_8 = arith.constant 0 : index
    %7 = krnl.define_loops 1
    // Copy elements before the insertion position
    krnl.iterate(%7) with (%7 -> %arg2 = 0 to %4){
      %22 = krnl.get_induction_var_value(%7) : (!krnl.loop) -> index
      %23 = krnl.load %arg0[%22] : memref<?xmemref<?x4x5xf32>>
      "krnl.seqstore"(%23, %6, %4) : (memref<?x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()
    }
    %c1_9 = arith.constant 1 : index
    %8 = affine.apply #map0()[%4]
    %9 = krnl.define_loops 1
    
    // Copy elements after the insertion position
    krnl.iterate(%9) with (%9 -> %arg2 = #map0()[%4] to %4){
      %22 = krnl.get_induction_var_value(%9) : (!krnl.loop) -> index
      %23 = krnl.load %arg0[%22] : memref<?xmemref<?x4x5xf32>>
      %c1_18 = arith.constant 1 : index
      %24 = arith.addi %22, %c1_18 : index
      "krnl.seqstore"(%23, %6, %24) : (memref<?x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()
    }
    // Insert the element
    "krnl.seqstore"(%1, %6, %4) : (memref<3x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()

    // SequenceAt
    ...
    %16 = "krnl.seqextract"(%6, %15) {copy = 1 : ui1} : (memref<?xmemref<?x4x5xf32>>, index) -> memref<?x4x5xf32>

    // onnx.Shape
    %17 = memref.alloc() {alignment = 16 : i64} : memref<3xi64>
    ...
    krnl.store %19, %17[%c0_16] : memref<3xi64>
    krnl.store %20, %17[%c1_17] : memref<3xi64>
    krnl.store %21, %17[%c2] : memref<3xi64>

    return %17 : memref<3xi64>
  }
```

After --buffer-deallocation pass
```
  func.func @test_sequence_insert(%arg0: memref<?xmemref<?x4x5xf32>>, %arg1: memref<3x4x5xf32>) -> memref<3xi64> {

    // onnx.Add
    %1 = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
    %2:3 = krnl.define_loops 3
    krnl.iterate(%2#0, %2#1, %2#2) with (%2#0 -> %arg2 = 0 to 3, %2#1 -> %arg3 = 0 to 4, %2#2 -> %arg4 = 0 to 5){
      %22:3 = krnl.get_induction_var_value(%2#0, %2#1, %2#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
      ...
      %23 = krnl.load %arg1[%22#0, %22#1, %22#2] : memref<3x4x5xf32>
      %24 = krnl.load %arg1[%22#0, %22#1, %22#2] : memref<3x4x5xf32>
      %25 = arith.addf %23, %24 : f32
      krnl.store %25, %1[%22#0, %22#1, %22#2] : memref<3x4x5xf32>
    }

    // Sequence Insert
    %6 = "krnl.seqalloc"(%5) : (index) -> memref<?xmemref<?x4x5xf32>>
    %c0_8 = arith.constant 0 : index
    %7 = krnl.define_loops 1
    krnl.iterate(%7) with (%7 -> %arg2 = 0 to %4){
      %22 = krnl.get_induction_var_value(%7) : (!krnl.loop) -> index
      %23 = krnl.load %arg0[%22] : memref<?xmemref<?x4x5xf32>>
      "krnl.seqstore"(%23, %6, %4) : (memref<?x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()
    }
    %c1_9 = arith.constant 1 : index
    %8 = affine.apply #map0()[%4]
    %9 = krnl.define_loops 1
    krnl.iterate(%9) with (%9 -> %arg2 = #map0()[%4] to %4){
      %22 = krnl.get_induction_var_value(%9) : (!krnl.loop) -> index
      %23 = krnl.load %arg0[%22] : memref<?xmemref<?x4x5xf32>>
      %c1_18 = arith.constant 1 : index
      %24 = arith.addi %22, %c1_18 : index
      "krnl.seqstore"(%23, %6, %24) : (memref<?x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()
    }
    "krnl.seqstore"(%1, %6, %4) : (memref<3x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()

    // Dealloc the memref generated by onnx.Add
    // It is only used by SequenceInsert
    memref.dealloc %1 : memref<3x4x5xf32>

    // SequenceAt
    ...
    %16 = "krnl.seqextract"(%6, %15) {copy = 1 : ui1} : (memref<?xmemref<?x4x5xf32>>, index) -> memref<?x4x5xf32>
    // Sequence becomes death after the last use by seqextract
    "krnl.seqdealloc"(%6) : (memref<?xmemref<?x4x5xf32>>) -> ()

    // onnx.Shape
      ...
    // After onnx.Shape, the element extracted from sequence becomes dead
    memref.dealloc %16 : memref<?x4x5xf32>
}
```

After --convert-seq-to-memref pass:
```
    // KrnlSeqStore
    %10 = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
    memref.copy %1, %10 : memref<3x4x5xf32> to memref<3x4x5xf32>
    %11 = memref.cast %10 : memref<3x4x5xf32> to memref<?x4x5xf32>
    memref.store %11, %6[%4] : memref<?xmemref<?x4x5xf32>>

    // KrnlSeqExtract
    %18 = memref.load %6[%17] : memref<?xmemref<?x4x5xf32>>
    %c0_12 = arith.constant 0 : index
    %19 = memref.dim %18, %c0_12 : memref<?x4x5xf32>
    %20 = memref.alloc(%19) {alignment = 16 : i64} : memref<?x4x5xf32>
    memref.copy %18, %20 : memref<?x4x5xf32> to memref<?x4x5xf32>
 ...
    // KrnlSeqDealloc
    // Loop to dealloc all elements
    scf.for %arg2 = %c0_14 to %5 step %c1_15 {
      %26 = memref.load %6[%arg2] : memref<?xmemref<?x4x5xf32>>
      memref.dealloc %26 : memref<?x4x5xf32>
    }
    // Dealloc the sequence itself
    memref.dealloc %6 : memref<?xmemref<?x4x5xf32>>
}
```

## Discussion
### No dangling pointer
Data pointer for memref is stored for sequence and may be read out.
Memref is copied both when its pointer is saved into the sequence and when its pointer is
read out from the sequence. Such operations maintains the kind of "std::unique_ptr" 
property for the  pointers of the elements saved in the sequence. And
these pointers will be freed only by the deallocation of the sequence.

### No memory leak
- The memref to be added to sequence (input of SequenceInsert)  will be freed as a regular tensor.
- The memref copied and saved in a sequence will be freed when the sequence is freed.
- The memref copied and returned from a sequence (SequenceAt) are freed by normal memref.dealloc because the SequenceExtract is marked as allocation for bufferization.

### Optimization
Since the tensor/memref for element is read only, there is no real need to copy it.
The copy is added due to two reasons:
-1. No interface to communicate with existing deallocation pass.
-2. For pointers possibly in multiple sequence.

If compiler analysis can guarantee the element is only in one live sequence (which is usually
the case in program), we can lower the ONNX sequence Ops in a different way:
- Use normal memref.alloc, instead of the KrnlSeqAllocOp, for the new sequences
- Use memref.store, instead of KrnlSeqStore, to store the elements from the old sequence
to the new sequence.
With such optimization, an element in the final sequence are copied at most twice for 
in most of applications. 

Another direction of optimization is to use std::shared_ptr for memref to manage the 
pointers dynamically. The interface provided by one-shot bufferization may also help.

## ToFix
- Shape inference with control flow. The result for test case with LoopOp(test_loop13_seq.onnx) is not correct.
- Refine the output of SequenceEmpty. The ONNX op to create an empty sequence,
  SequenceEmpty, is specified to generate a sequence of unranked tensor, which is not 
  supported in onnx-mlir.
- Handle program argument or return with SeqType.

## Runtime test case
ToDo: add it into test case

Source file for test case, seq_insert.mlir
```
module {
func.func @main_graph(%arg0: tensor<?x4x5xf32>, %arg1:tensor<3x4x5xf32>) -> tensor<3xi64>  {
  %0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<i64>
  %c1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<i64>
  %1 = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<?x4x5xf32>>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.SequenceInsert"(%1, %arg1, %0) : (!onnx.Seq<tensor<?x4x5xf32>>, tensor<3x4x5xf32>, tensor<i64>) -> !onnx.Seq<tensor<?x4x5xf32>>
  %6 = "onnx.SequenceInsert"(%3, %arg0, %0) : (!onnx.Seq<tensor<?x4x5xf32>>, tensor<?x4x5xf32>, tensor<i64>) -> !onnx.Seq<tensor<?x4x5xf32>>
  %4 = "onnx.SequenceAt"(%6, %0) : (!onnx.Seq<tensor<?x4x5xf32>>, tensor<i64>) -> tensor<?x4x5xf32>
  %5 = "onnx.Shape"(%4) : (tensor<?x4x5xf32>) -> tensor<3xi64>
  return %5 : tensor<3xi64>
}
"onnx.EntryPoint"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32} : ()->()
}
```

Script to run the test
```
import numpy as np
import onnx
from onnx import numpy_helper
from PyRuntime import OMExecutionSession

model = './seq_insert.so'
sess = OMExecutionSession(model)

x1_np = np.random.randn(6, 4, 5).astype(np.float32)
x2_np = np.random.randn(3, 4, 5).astype(np.float32)

inputs = [x1_np, x2_np]

print("before run")
y = sess.run(inputs)
print("after run")
print("output shape: ", y[0].shape)
print(y[0]);
```

Execution result:
```
before run
after run
output shape:  (3,)
[6 4 5]
```


[./Quantization-NNPA.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Overview 
 
NNPA in IBM Telum II supports 8-bit signed-integer quantized matrix multiplications. This document shows how to compile an ONNX model for 8-bit quantization on NNPA. When not following these steps, models will still be accelerated when targeting Telum systems using a mixture of 16-bit floating-point numbers for computations mapped to the Telum's Integrated AI accelerator and 32-bit floating-point numbers for computations mapped to the Telum CPUs.

There are two approaches to using quantization in the onnx-mlir compiler, depending on the input ONNX model to the compile:
- The input model is a quantized model that was quantized by other frameworks such as ONNX Runtime. In this case, the input ONNX model contains 8-bit operations, and the onnx-mlir compiler selects suitable 8-bit operations to run on NNPA. There is no special compile flags needed to enable quantization when compiling this quantized model. Hence, we do not discuss this case in this document.
  - In this approach, the compiler supports both static and dynamic quantized models.
- The input model is a non-quantized model, e.g. operations operate on float32 data types. In this case, the onnx-mlir compiler provides several quantization options in order to quantize the model during compilation, then run the compiled model on NNPA. The remaining of this document describes this approach.
  - In this approach, the compiler only supports dynamic quantization.

In both approaches, the following constraints are applied:
- Only per-tensor quantization is supported, meaning `scale` and `zero_point` are computed per-tensor and are scalar values.
- Target quantization data type is 8-bit signed-integer.
 
Quantization requires NNPA in IBM Telum II, meaning that the following compile flags must be specified to enable quantization: `-maccel=NNPA -march=arch15`.

# Dynamic quantization by the compiler

Again, it is important to note that the onnx-mlir compiler currently:
- supports per-tensor dynamic quantization, and
- quantizes data tensors from float32 to 8-bit signed integer. If a data tensor in the input model is already in 8-bit singed integer, the compiler will not quantize it again.

The compiler provides two compile flags for dynamically quantizing a model at compile time:
- `--nnpa-quant-dynamic` to enable dynamic quantization.
- `--nnpa-quant-op-types` to specify the types of ONNX operations to quantize manually, e.g. `MatMul,Conv`.

Users can specify whether or not to symmetrize data for activations and weights by using options `symActivation, asymActivation, symWeight, asymWeight` as values for `--nnpa-quant-dynamic`.
For examples, to asymmetrize data for activations and to symmetrize data for weights, one can use `--nnpa-quant-dynamic=asymActivation,symWeight`.

By specifying `--nnpa-quant-dynamic` only, the compiler will decide quantization options and operation types by itself.

## Computing `scale` and `zero_point` 
The compiler uses the following equations to compute `scale` and `zero_point` for 8-bit signed integer quantization.

Asymmetric quantization
```
scale = (maximum(0, max(x)) - minimum(0, min(x))) / (qmax - qmin)
zero_point = cast(round(saturate(qmin - min(x)/scale)))
```
where
- `x` is the input tensor to quantize,
- data range is adjusted to include 0,
- `qmax=127` and `qmin=-128` are the max and min values for quantization range.
- `saturate` is to saturate to `[-128, 127]`.

Symmetric quantization
```
scale = max(abs(x)) / 127
zero_point = 0
```

Given `scale` and `zero_point`, the input `x` is quantized to
```
quantized_x = x/scale + zero_point
```

# Performance notes

It is often the case that symmetric quantization leads to better inference performance but poorer accuracy than asymmetric quantization.
Users may want to experiment with different quantization schemes to find the best combination for their own model.

# Resources
- [A visual guide to quantization](https://www.maartengrootendorst.com/blog/quantization/)


[./ONNXAI.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# About

ONNX-MLIR is an open-source project for compiling ONNX models into native code
on x86, Power, s390x and other architectures. It is built on top of Multi-Level
Intermediate Representation (MLIR) compiler infrastructure.

# Slack channel

We have a slack channel established under the Linux Foundation AI and Data Workspace, named `#onnx-mlir-discussion`.
This channel can be used for asking quick questions related to this project.
A direct link is [here](https://lfaifoundation.slack.com/archives/C01J4NAL4A2).
Join this workspace using this [link](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA).


[./LocationInfo.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Maintain and Use Location Info in onnx-mlir

1. [Summary](#Summary)
2. [ONNX Model](#ONNX-model)
3. [MLIR File](#MLIR-file)

## Summary
Support of Location info propagation in transformation is one of the attractive features of MLIR. onnx-mlir can takes advantage of this feature in compiler transformation, and runtime debugging. This document describes how to maintain and use the location info in onnx-mlir. In summary:
- All onnx-mlir transformations are required to propagate the location info from the source to the target
- Create location info when an ONNX model is imported. If there is `onnx_node_name` string attribute for an operation, the string is transferred to its location. Otherwise, Unknown location is used.
- MLIR adds file location (in form of filename:line:column) to nodes when reading in a MLIR file, unless the MLIR file already contains location.
- Use the flag `--preserveLocations` to turn on location info in the output.
- With the previous two combined, we can track the source of error by dumping out the MLIR file(without `--preserveLocations`) at desired stage (for example, EmitONNXIR, or EmitMLIR), and then continuing transformation by loading the dumped file. The location info will be line number for that dumped file, providing more details than just from the onnx model. 

## ONNX model
When reading an ONNX model (.onnx file), onnx-mlir tries to attach location info to the generated IR. 
Some ONNX exporter annotates every operation with an StringAttr, "onnx_node_name", with an unique string for that operation. 
The importer of onnx-mlir converts the "onnx_node_name" attribute in the ONNX file tostring location info for the operation.
If the ONNX model does not have "onnx_node_name" attribute, Unknown location is attached.

### Example with onnx_node_name

This roberta  model is downloaded from onnx model zoo. Compile it with command
`onnx-mlir roberta-base-11.onnx --preserveLocations --EmitONNXBasic`.
The location info will be displayed in the output when the flag `--preserveLocations` is used.
In the output file(roberta-base-11.onnx.mlir), two nodes and their location info are shown below.

```
    %392 = "onnx.Sub"(%390, %391) {onnx_node_name = "Sub_109"} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32> loc(#loc393)
    %393 = onnx.Constant dense<2.000000e+00> : tensor<f32> loc(#loc394)

#loc393 = loc("Sub_109")
#loc394 = loc("Constant_110")
```
The 'onnx_node_name` attribute is only for operations, not for the constant. The location info for Constant is given by the importer.

### Example without onnx_node_name
The following model, test_add.onnx, came from onnx backend test. It is not from
onnx exporter and does not have `onnx_node_name` attribute.
The output of command `onnx-mlir test_add.onnx --preserveLocations --EmitONNXBasic`:

```
#loc = loc(unknown)

...
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32> loc(#loc)
...
```
There is no useful location info.

## MLIR file

MLIR automatically creates location info when the intermediate file (.mlir file) is read in as long as there is no location info in that .mlir file.  
For example, though there is no location info for the test_add.onnx, we can dump the importer result and load it again. Then we can find useful location info in the output.
Commands:
```
onnx-mlir test_add.onnx --EmitONNXBasic`
onnx-mlir test_add.onnx.mlir --preserveLocations --EmitONNXBasic`
```
...
Then location info can be found in the output of test_add.onnx.onnx.mlir
```
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32> loc(#loc4)
    onnx.Return %0 : tensor<3x4x5xf32> loc(#loc5)
...

#loc4 = loc("test_add.onnx.mlir":3:10)
#loc5 = loc("test_add.onnx.mlir":4:5)
```
The test_add.onnx.mlir content:

```
  1 module attributes {llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-apple-darwin22.3.0", "onnx-mlir.symbol-postfix" = "test_add"} {
  2   func.func @main_graph(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
  3     %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
  4     onnx.Return %0 : tensor<3x4x5xf32>
  5   }
  6   "onnx.EntryPoint"() {func = @main_graph} : () -> ()
  7 }
```

If you want to track the operations for krnl IR, dump the file after lowering to krnl.



[./DebuggingNumericalError.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Debugging Numerical Error

Use `utils/RunONNXModel.py` python script to debug numerical errors, when
onnx-mlir-compiled inference executable produces numerical results that are
inconsistent with those produced by the training framework. This python script
will run the model through onnx-mlir and a reference backend, and compare the
intermediate results produced by these two backends layer by layer.

## Prerequisite
- Set `ONNX_MLIR_HOME` environment variable to be the path to the HOME
  directory for onnx-mlir. The HOME directory for onnx-mlir refers to the
  parent folder containing the `bin`, `lib`, etc sub-folders in which ONNX-MLIR
  executables and libraries can be found.

## Reference backend
Outputs by onnx-mlir can be verified by using a reference ONNX backend or
reference inputs and outputs in protobuf.
- To verify using a reference backend, install onnxruntime by running `pip
  install onnxruntime`. To use a different testing backend, simply replace code
  importing onnxruntime to some other ONNX-compliant backend.
- To verify using reference outputs, use `--verify=ref --load-ref=data_folder`
  where `data_folder` is the path to a folder containing protobuf files for
  inputs and outputs. [This
  guideline](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#manipulating-tensorproto-and-numpy-array)
  is a how-to for creating protobuf files from numpy arrays.

## Usage

`utils/RunONNXModel.py` supports the following command-line options:

```
$ python ../utils/RunONNXModel.py  --help
usage: RunONNXModel.py [-h] [--log-to-file [LOG_TO_FILE]] [--model MODEL] [--compile-args COMPILE_ARGS] [--compile-only] [--compile-using-input-shape] [--print-input]
                       [--print-output] [--save-onnx PATH] [--verify {onnxruntime,ref}] [--verify-all-ops] [--verify-with-softmax] [--verify-every-value] [--rtol RTOL]
                       [--atol ATOL] [--save-so PATH | --load-so PATH] [--save-ref PATH] [--load-ref PATH | --shape-info SHAPE_INFO] [--lower-bound LOWER_BOUND]
                       [--upper-bound UPPER_BOUND]

optional arguments:
  -h, --help                  show this help message and exit
  --log-to-file [LOG_TO_FILE] Output compilation messages to file, default compilation.log
  --model MODEL               Path to an ONNX model (.onnx or .mlir)
  --compile-args COMPILE_ARGS Arguments passed directly to onnx-mlir command. See bin/onnx-mlir --help
  --compile-only              Only compile the input model
  --compile-using-input-shape Compile the model by using the shape info getting from the inputs in the reference folder set by --load-ref
  --print-input               Print out inputs
  --print-output              Print out inference outputs produced by onnx-mlir
  --save-onnx PATH            File path to save the onnx model. Only effective if --verify=onnxruntime
  --verify {onnxruntime,ref}  Verify the output by using onnxruntime or reference inputs/outputs. By default, no verification. When being enabled, --verify-with-softmax or --verify-every-value must be used to specify verification mode.
  --verify-all-ops            Verify all operation outputs when using onnxruntime
  --verify-with-softmax       Verify the result obtained by applying softmax to the output
  --verify-every-value        Verify every value of the output using atol and rtol
  --rtol RTOL                 Relative tolerance for verification
  --atol ATOL                 Absolute tolerance for verification
  --save-so PATH              File path to save the generated shared library of the model
  --load-so PATH              File path to load a generated shared library for inference, and the ONNX model will not be re-compiled
  --save-ref PATH             Path to a folder to save the inputs and outputs in protobuf
  --load-ref PATH             Path to a folder containing reference inputs and outputs stored in protobuf. If --verify=ref, inputs and outputs are reference data for verification
  --shape-info SHAPE_INFO     Shape for each dynamic input of the model, e.g. 0:1x10x20,1:7x5x3. Used to generate random inputs for the model if --load-ref is not set
  --lower-bound LOWER_BOUND   Lower bound values for each data type. Used inputs. E.g. --lower-bound=int64:-10,float32:-0.2,uint8:1. Supported types are bool, uint8, int8, uint16, int16, uint32, int32, uint64, int64,float16, float32, float64
  --upper-bound UPPER_BOUND   Upper bound values for each data type. Used to generate random inputs. E.g. --upper-bound=int64:10,float32:0.2,uint8:9. Supported types are bool, uint8, int8, uint16, int16, uint32, int32, uint64, int64, float16, float32, float64
```

## Helper script to compare a model under two distinct compile option.

Based on the above `utils/runONNXModel.py`, the `utils/checkONNXModel.py` allows a user to run a given model twice, under two distinct compile options, and compare its results.
This let a user simply test a new option, comparing the safe version of the compiler (e.g. `-O0` or `-O3`) with a more advanced version (e.g. `-O3` or `-O3 --march=x86-64`). Simply specify the compile options using the `--ref-compile-args` and `--test-compile-args` flags, a model using the `--model` flag, and possibly a `--shape-info` in presence of dynamic shape inputs.
Full options are listed under the `--help` flag.

## Debugging the Code Generated for an Operator.

If you know, or suspect, that a particular ONNX MLIR operator produces an incorrect result, and want to narrow down the problem, we provide a couple of useful Krnl operators that allow printing (at runtime) the value of a tensor, or a value that has a primitive data type. 

To print out the value of a tensor at a particular program point, inject the following code (where `X` is the tensor to be printed):

```code
create.krnl.printTensor("Tensor X: ", X);
```

Note: currently the content of the tensor is printed only when the tensor rank is less than four.

To print a message followed by one value, inject the following code (where `val` is the value to be printed and `valType` is its type):

```code
create.krnl.printf("inputElem: ", val, valType);
```

## Finding memory errors

If you know, or suspect, that an onnx-mlir-compiled inference executable
suffers from memory allocation related issues, the
[valgrind framework](https://valgrind.org/) or
[mtrace memory tool](https://github.com/sstefani/mtrace) can be used to facilitate debugging.
These tools trace memory
allocation/free-related APIs, and can detect memory issues, such as memory leaks.

However if the problems relating to memory access, especially buffer overrun problems, are notoriously difficult to debug because run-time errors occur outside of the point containing the problem. 
The ["Electric Fence library"](https://github.com/CheggEng/electric-fence) can be
used for debugging these problems. It helps you detect two common programming problems: software that overruns the boundaries of a malloc() memory allocation, and
software that touches a memory allocation
that has been released by free(). Unlike other memory debuggers, Electric
Fence will detect read accesses as well as writes, and it will pinpoint the
exact instruction that causes an error.

Since the Electric Fence library is not officially supported by RedHat, you
need to download, build and install the source code by yourself on yours.
After installing it, link this library by using the "-lefence" option when
generating inference executables. Then simply execute it, which will
cause a runtime error and stop at the place causing memory access problems. You can
identify the place with a debugger or debugging print functions
described in the previous section.


[./UsingPyRuntime.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Using Python interfaces

Onnx-mlir has runtime utilities to compile and run ONNX models in Python.
These utilities are implemented by the `OnnxMlirCompiler` compiler interface
(include/OnnxMlirCompiler.h) and the `ExecutionSession` class
(src/Runtime/ExecutionSession.hpp).
Both utilities have an associated Python binding generated by [pybind library](https://github.com/pybind/pybind11).

## Configuring the Python interfaces

Using pybind, a C/C++ binary can be directly imported by the Python interpreter.
For onnx-mlir, there are five such libraries, one to compile onnx-mlir models, 
two to run the models and the other two are to compile and run the models.

1. The shapred library to compile onnx-mlir models is generated by `PyOMCompileSession` (src/Compiler/PyOMCompileSession.hpp) and build as a shared library to `build/Debug/lib/PyCompile.cpython-<target>.so`.
2. The shared library to run onnx-mlir models is generated by `PyExecutionSession` (src/Runtime/PyExecutionSession.hpp) and built as a shared library to `build/Debug/lib/PyRuntimeC.cpython-<target>.so`.
3. The Python library to run onnx-mlir models (src/Runtime/python/PyRuntime.py).
4. The shared library to compile and run onnx-mlir models is generated by `PyOMCompileExecutionSessionC` (src/Runtime/PyOMCompileExecutionSession.hpp) and built as a shared library to `build/Debug/lib/PyCompileAndRuntimeC.cpython-<target>.so`.
5. The Python library to compile run onnx-mlir models (src/Runtime/python/PyCompileAndRuntime.py). This library takes an .onnx file and the options as inputs, it will load it and then compile and run it.


The module can be imported normally by the Python interpreter as long as it is in your
PYTHONPATH. Another alternative is to create a symbolic link to it in your working directory.

```shell
cd <working directory>
ln -s <path to the shared library to copmpile onnx-mlir models>(e.g. `build/Debug/lib/PyCompile.cpython-<target>.so`) .
ln -s <path to the shared library to run onnx-mlir models>(e.g. `build/Debug/lib/PyRuntimeC.cpython-<target>.so`) .
ln -s <path to the Python library to run onnx-mlir models>(e.g. src/Runtime/python/PyRuntime.py) .
ln -s <path to the shared library to compile and run onnx-mlir models>(e.g. `build/Debug/lib/PyCompileAndRuntimeC.cpython-<target>.so`) .
ln -s <path to the Python library to compile and run onnx-mlir models>(e.g. src/Runtime/python/PyCompileAndRuntime.py) .
python3
```

# Python interface to run models: PyRuntime

## Running the PyRuntime interface

An ONNX model is a computation graph and it is often the case that the graph
has a single entry point to trigger the computation. Below is an example of doing
inference for a model that has a single entry point.

```python
import numpy as np
from PyRuntime import OMExecutionSession

model = 'model.so' # LeNet from ONNX Zoo compiled with onnx-mlir

# Create a session for this model.
session = OMExecutionSession(shared_lib_path=model)
# Input and output signatures of the default entry point.
print("input signature in json", session.input_signature())
print("output signature in json",session.output_signature())
# Do inference using the default entry point.
a = np.full((1, 1, 28, 28), 1, np.dtype(np.float32))
outputs = session.run(input=[a])

for output in outputs:
    print(output.shape)
```

In case a computation graph has multiple entry points, users have to set a specific
entry point to do inference. Below is an example of doing inference with multiple
entry points.
```python
import numpy as np
from PyRuntime import OMExecutionSession

model = 'multi-entry-points-model.so'

# Create a session for this model.
session = OMExecutionSession(shared_lib_path=model, use_default_entry_point=False) # False to manually set an entry point.

# Query entry points in the model.
entry_points = session.entry_points()

for entry_point in entry_points:
  # Set the entry point to do inference.
  session.set_entry_point(name=entry_point)
  # Input and output signatures of the current entry point.
  print("input signature in json", session.input_signature())
  print("output signature in json",session.output_signature())
  # Do inference using the current entry point.
  a = np.arange(10).astype('float32')
  b = np.arange(10).astype('float32')
  outputs = session.run(input=[a, b])
  for output in outputs:
    print(output.shape)
```

### Using model tags

If a model was compiled by using `--tag`, the value of `--tag` must be passed to OMExecutionSession.
Using tags is useful when there are multiple sessions for multiple models in the same python script.
Below is an example of doing multiple inferences using tags.
```python
import numpy as np
from PyRuntime import OMExecutionSession

encoder_model = 'encoder/model.so' # Assumed that the model was compiled using `--tag=encoder`
decoder_model = 'decoder/model.so' # Assumed that the model was compiled using `--tag=decoder`

# Create a session for the encoder model.
encoder_sess = OMExecutionSession(shared_lib_path=encoder_model, tag="encoder")
# Create a session for the decoder model.
decoder_sess = OMExecutionSession(shared_lib_path=decoder_model, tag="decoder")
```

In case two models were NOT compiled by using `--tag`, they must be compiled
with different .so filenames if they are to be used in the same process. Indeed,
when no tags are given, we use the file name as its default tag.
Below is an example of doing multiple inferences without using tags.
```python
import numpy as np
from PyRuntime import OMExecutionSession

encoder_model = 'my_encoder.so'
decoder_model = 'my_decoder.so'

# Create a session for the encoder model.
encoder_sess = OMExecutionSession(shared_lib_path=encoder_model) # tag will be `my_encoder` by default.
# Create a session for the decoder model.
decoder_sess = OMExecutionSession(shared_lib_path=decoder_model) # tag will be `my_decoder` by default.
```

To use functions without tags, e.g. `run_main_graph`, set `tag = "NONE"`.

## PyRuntime model API
The complete interface to `OMExecutionSession` can be seen in the sources mentioned previously.
However, using the constructor and run method is enough to perform inferences.

```python
def __init__(self, shared_lib_path: str, tag: str, use_default_entry_point: bool):
    """
    Args:
        shared_lib_path: relative or absolute path to your .so model.
        tag: a string that was passed to `--tag` when compiling the .so model. By default, it is the output file name without its extension, namely, `filename` in `filename.so`
        use_default_entry_point: use the default entry point that is `run_main_graph_{tag}` or not. Set to True by default.
    """

def run(self, input: List[ndarray]) -> List[ndarray]:
    """
    Args:
        input: A list of NumPy arrays, the inputs of your model.

    Returns:
        A list of NumPy arrays, the outputs of your model.
    """

def input_signature(self) -> str:
    """
    Returns:
        A string containing a JSON representation of the model's input signature.
    """

def output_signature(self) -> str:
    """
    Returns:
        A string containing a JSON representation of the model's output signature.
    """

def entry_points(self) -> List[str]:
    """
    Returns:
        A list of entry point names.
    """

def set_entry_point(self, name: str):
    """
    Args:
        name: an entry point name.
    """
```

# Python interface to compile models: PyCompile

## Running the PyCompile interface

An ONNX model can be compiled directly from the command line. The resulting library can then be executed using Python as shown in the previous sections. At times, it might be convenient to also compile a model directly in Python. This section explores the Python methods to do so.

The OMCompileSession object will take a file name while constructing. For the compilation, `compile()` will take a `flags` string as an input which will override any default options set from the env var.

```python
import numpy as np
from PyCompile import OMCompileSession

# Load onnx model and create OMCompileSession object.
file = './mnist.onnx'
compiler = OMCompileSession(file)
# Generate the library file. Success when rc == 0 while set the opt as "-O3"
rc = compiler.compile("-O3")
# Get the output file name
model = compiler.get_compiled_file_name()
if rc:
    print("Failed to compile with error code", rc)
    exit(1)
print("Compiled onnx file", file, "to", model, "with rc", rc)
```

The `PyCompile` module exports the `OMCompileSession` class to drive the
compilation of a ONNX model into an executable model.
Typically, a compiler object is created for a given model by giving it the file name of the ONNX model.
Then, all the compiler options can be set as a whole `std::string` to generate the desired executable.
Finally, the compilation itself is performed by calling the `compile()` command where the user passes the options string as the input of this function.

The `compile()` commands returns a return code reflecting the status of the compilation.
A zero value indicates success, and nonzero values reflect the error code.
Because different Operating Systems may have different suffixes for libraries,
the output file name can be retrieved using the `get_compiled_file_name()` method.

## PyCompile model API

The complete interface to OnnxMlirCompiler can be seen in the sources mentioned previously.
However, using the constructor and the methods below are enough to compile models.

```python
def __init__(self, file_name: str):
    """
    Constructor for an ONNX model contained in a file.
    Args:
        file_name: relative or absolute path to your ONNX model.
    """
def __init__(self, input_buffer: void *, buffer_size: int):
    """
    Constructor for an ONNX model contained in an input buffer.
    Args:
        input_buffer: buffer containing the protobuf representation of the model.
        buffer_size: byte size of the input buffer.
    """
def compile(self, flags: str):
    """
    Method to compile a model from a file.
    Args:
        flags: all the options users would like to set.
    Returns:
        Zero on success, error code on failure.
    """
def compile_from_array(self, output_base_name: str, target: OnnxMlirTarget):
    """
    Method to compile a model from an array.
    Args:
        output_base_name: base name (relative or absolute, without suffix)
        where the compiled model should be written into.
        target: target for the compiler's output. Typical values are
        OnnxMlirTarget.emit_lib or emit_jni.
    Returns:
        Zero on success, error code on failure.
    """
def get_compiled_file_name(self):
    """
    Method to provide the full (absolute or relative) output compiled file name, including
    its suffix.
    Returns:
        String containing the fle name after successful compilation; empty string on failure.
    """
def get_error_message(self):
    """
    Method to provide the compilation error message.
    Returns:
        String containing the error message; empty string on success.
    """
```

# Python interface to compile and run models: PyCompileAndRuntime

## Running the PyCompileAndRuntime interface

```python
import numpy as np
from PyCompileAndRuntime import OMCompileExecutionSession

# Load onnx model and create OMCompileExecutionSession object.
inputFileName = './mnist.onnx'
# Set the full name of compiled model
sharedLibPath = './mnist.so'
# Set the compile option as "-O3"
session = OMCompileExecutionSession(inputFileName,sharedLibPath,"-O3")

# Print the models input/output signature, for display.
# Signature functions for info only, commented out if they cause problems.
session.print_input_signature()
session.print_output_signature()

# Do inference using the default entry point.
a = np.full((1, 1, 28, 28), 1, np.dtype(np.float32))
outputs = session.run(input=[a])

for output in outputs:
    print(output.shape)
```

## PyCompileAndRuntime model API

The PyCompileAndRuntime is a new class, which combines compile and execution. Its constructor takes the `.onnx` input file and compile the model with the options given by the user and then run the model with an input.

```python
def __init__(self, input_model_path: str, compiled_file_path: str, flags: str, use_default_entry_point: bool):
    """
    Constructor for an ONNX model contained in a file.
    Args:
        input_model_path: relative or absolute path to your ONNX model.
        compiled_file_path: relative or absolute path to your compiled file.
        flags: all the options users would like to set.
        use_default_entry_point: use the default entry point that is `run_main_graph` or not. Set to True by default.
    """
def get_compiled_result(self):
    """
    Method to provide the results of the compilation.
    Returns:
        Int containing the results. 0 represents successful compilation; others on failure.
    """
def get_compiled_file_name(self):
    """
    Method to provide the full (absolute or relative) output file name, including
    its suffix.
    Returns:
        String containing the fle name after successful compilation; empty string on failure.
    """
def get_error_message(self):
    """
    Method to provide the compilation error message.
    Returns:
        String containing the error message; empty string on success.
    """
def entry_points(self) -> List[str]:
    """
    Returns:
        A list of entry point names.
    """
def set_entry_point(self, name: str):
    """
    Args:
        name: an entry point name.
    """
def run(self, input: List[ndarray]) -> List[ndarray]:
    """
    Args:
        input: A list of NumPy arrays, the inputs of your model.

    Returns:
        A list of NumPy arrays, the outputs of your model.
    """
def input_signature(self) -> str:
    """
    Returns:
        A string containing a JSON representation of the model's input signature.
    """

def output_signature(self) -> str:
    """
    Returns:
        A string containing a JSON representation of the model's output signature.
    """
```


[./DevicePlacement-NNPA.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Device placement

Device placement is how the compiler place one operation on CPU or NNPA.

## Query device placement configuration

There are two ways to know which device an operation is placed on:
- Using `onnx-mlir --EmitONNXIR --maccel=NNPA model.onnx`, or
- Using `onnx-mlir --nnpa-save-device-placement-file=cfg.json model.onnx`.
 
1. Using `--EmitONNXIR --maccel=NNPA`

When using `--EmitONNXIR --maccel=NNPA` options, each operation in the generated IR is annotated with an attribute `device` to show which device the operation is placed on. There are three posible values for `device`:
- "": the operation may be on CPU or NNPA depending on optimizations in the compiler. 
- "nnpa": the operation is on NNPA.
- "cpu": the operation is on CPU.

Below is an example of the output of `--EmitONNXIR --maccel=NNPA`:
```mlir
%0 = "onnx.Relu"(%arg0) {onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
%1 = "onnx.Relu"(%0) {device="cpu", onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
%2 = "onnx.Relu"(%1) {onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
%3 = "onnx.Sigmoid"(%2) {device="nnpa", onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
```

2. Using `--nnpa-save-device-placement-file=cfg.json`

The option is to save the device placement configuration into a JSON file. This option is convenient when users don't want to interrupt the compilation.

The JSON file will contains a list of operation records. Each record includes three key-value pairs where keys are: 
- "device": similar to `device` attribute in the operation.
- "node_type": ONNX node type, e.g. `onnx.Conv`, `onnx.MatMul`.
- "onnx_node_name": a string to denote ONNX node names.

Below is one example of a JSON file:
```json
{
  "device_placement": [
    {
      "device":"nnpa",
      "node_type":"onnx.Relu",
      "onnx_node_name":"Relu_0"
    },
    {
      "device":"cpu",
      "node_type":"onnx.Relu",
      "onnx_node_name":"Relu_1"},
    {
      "device":"nnpa",
      "node_type":"onnx.Relu",
      "onnx_node_name":"Relu_2"
    },
    {
      "device":"nnpa",
      "node_type":"onnx.Sigmoid",
      "onnx_node_name":"Sigmoid_0"
    }
  ]
}
```

## Set device placement manually.

We allow users to force one operation to run on a specific device. However, at this moment, only placing on CPU is guaranted to be successful done. It means that even when `device=NNPA` is specified, it is not guaranted that the operation will run on NNPA. 

There are two ways to change device of an operation:
- by editing the output of `--EmitONNXIR --maccel=NNPA` directly and compile again.
- by passing a JSON file for device placement to the compiler by using `--nnpa-load-device-placement-file=json`.

For the former option, it is straighforward, just changing the value of the `device` attribute of an operation, for example, changing `device=nnpa` to `device=cpu`.

For the later option, users can obtain a template file from `--nnpa-save-device-placement-file`, and use it as the starting point of modification.
We use C++ std::regex_match function to match operations based on `node_type` and `onnx_node_name`. Both `node_type` and `onnx_node_name` must be satisfied.
The JSON file will contain a list of records for each operation matching. The order of the records does matter. If one operation matches a record and is set device, it will not be set device again even when it matches the later records in the list. If one operation does not match a record but matches a later record, the operation is still set device by the later record. In other words, the device of an operation is set by the first matched record.

Below are some examples for the later option. Given an input program:
```mlir
func.func @test_load_config_file_all_on_cpu(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "onnx.Relu"(%0) {onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "onnx.Relu"(%1) {onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "onnx.Sigmoid"(%2) {onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  onnx.Return %3 : tensor<?x?x?xf32>
```

1. Schedule all operations to run on CPU
```json
{
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "onnx.*",
      "onnx_node_name": ".*"
    }
  ]
}
```

2. Schedule all Relu operations to run on CPU:
```json
{
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "onnx.Relu",
      "onnx_node_name": ".*"
    }
  ]
}
```
3.  Schedule operations using onnx_node_name: here we use regex to chose only Relu_1 and Relu_2 operations, exact match is used for onnx.Sigmoid.
```json
{
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "onnx.Relu",
      "onnx_node_name": "Relu_(1|2)"
    },
    {
      "device": "nnpa",
      "node_type": "onnx.Sigmoid",
      "onnx_node_name": "Sigmoid_0"
    }
  ]
}
```

4. `onnx.Relu` does not match because there is no operation with `node_type = Relu`, so only `onnx.Sigmoid` is set device.
```json
{
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "Relu",
      "onnx_node_name": "Relu_(1|2)"
    },
    {
      "device": "cpu",
      "node_type": "onnx.Sigmoid",
      "onnx_node_name": "Sigmoid_0"
    }
  ]
}
```

5. We have two overlapping records both matching on `onnx.Relu`. In this case, only the first matched record will set device. Thus, `Relu_0` and `Relu_1` have device "cpu" by matching the first record, `Relu_2` operation has device "cpu" by matching the third record.
```json
{
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "onnx.Relu",
      "onnx_node_name": "Relu_(0|1)"
    },
    {
      "device": "nnpa",
      "node_type": "onnx.Sigmoid",
      "onnx_node_name": "Sigmoid_0"
    },
    {
      "device": "cpu",
      "node_type": "onnx.Relu",
      "onnx_node_name": "Relu_(1|2)"
    }
  ]
}
```


[./PythonPackage.md]:

The Python package, onnxmlir, provides an installable package to use onnx-mlir
compiler in a similar way to onnxruntime. Also the package supports the way to 
run model by `utils/RunONNXModel.py`.

The source of the package is located at `onnx-mlir/utils/onnxmlir`. The main python code, `onnxmlir/src/onnxmlir/RunONNXModel.py` should be the same as `onnx-mlir/utils/RunONNXModel.py`. You can use target `OMCreateONNXMLIRSource` to create the installable directory in your build directory.
The package can be installed from your local directory with `pip3 install your_path/onnx-mlir/build/utils/onnxmlir`

Follow instructions in https://packaging.python.org/en/latest/tutorials/packaging-projects/
commands to use under the top directory onnxmlir
```
python3 -m pip install --upgrade build
python3 -m build
#After get the api-token
python3 -m pip install --upgrade twine
python3 -m twine upload --repository testpypi dist/*
```
Different from document, the prompt asked only for the api-token

Examples can be found at onnx-mlir/util/onnxmlir/tests.


[./AccelNNPAHowToUseAndTest.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Build and test for Accelerator NNPA

Neural Network Processing Assist Facility (NNPA) is implemented on processor units of IBM z16. Onnx-mlir can use it via  [IBM Z Deep Neural Network Library (zDNN)](https://github.com/IBM/zDNN). Building and lit tests runs on other IBM Z systems(eg. z15), but numerical tests need to run on z16.

## Build

Add following CMake option to build onnx-mlir for NNPA. Regarding build command for Linux OS, see [here](BuildOnLinuxOSX.md/#build)

- `-DONNX_MLIR_ACCELERATORS=NNPA`

## Test

### Lit tests

The lit tests for NNPA are included in `test/mlir/accelerators/nnpa`. When building onnx-mlir for NNPA, these lit tests also run with the following same command with CPU.

```
cmake --build . --target check-onnx-lit
```

### Numerical tests

Numerical tests for NNPA are provided in `test/accelerators/NNPA/numerical`. Currently tests for Conv2D, MatMul2D, Gemm, LSTM, and GRU are provided and run using following command. These tests can check if a zDNN instruction is included in the generated shared library using an environment variable `TEST_INSTRUCTION`. Also, to check the accuracy of the results, ATOL and RTOL can be set by using environment `TEST_ATOL` and `TEST_RTOL`. An environment variable `TEST_DATARANGE` are provided to set lower and upper bound of data range. They can be set "<lower bound>,<upper bound>" such as "-0.1,0.1". To configure the test cases, an environment variable `TEST_CONFIG` are provided. Current configurations are written in section of each test below.

```
cmake --build . --config Release --target check-onnx-numerical-nnpa
```

These tests uses the same test code with numerical tests for CPU (`test/modellib` and `test/numerial`), but uses different cmake file(`test/accelerator/NNPA/numerical/CMakeLists.txt`).

##### Conv2D
Since Conv2D in zDNN library only supports the case where dilation equals to one, dilation is always set to one in the test. Also, padding types are set as VALID and SAME_UPPER since they are only suppored. All dimensions are static since dynamic height and weight dimension are currently not supported. These configurations are set automatically when using `--maccel=NNPA`, which are equivalent to manually setting the environment variable `TEST_CONFIG` to "-dim=static -dilation=1 -padding=valid_upper".

##### Gemm
`alpha` and `beta` in Gemm are always one, which are supported case by zDNN library. These configurations are set automatically when using `--maccel=NNPA`, which are equivalent to manually setting the environment variable `TEST_CONFIG` to "-alpha=1 -beta=1".

##### LSTM
Peephole tensor is not tested since LSTM in zDNN library does not support it. These configurations are set automatically when using `--maccel=NNPA`, which are equivalent to manually setting the environment variable `TEST_CONFIG` to "-peephole=0".

##### GRU
GRU of zDNN library supports only the case where the linear transformation is applied before multiplying by the output of the reset gata. It is configured automatically when using `--maccel=NNPA`, which are equivalent to manually setting the environment variable `TEST_CONFIG` to "-linearBeforeReset=1".

### Backend tests

Backend tests for NNPA are provided in `test/accelerators/NNPA/backend`. It can be run with following command. Only test cases supported by zDNN runs as listed in `test/accelerators/NNPA/backend/CMakeLists.txt`.

```
cmake --build . --config Release --target check-onnx-backend-nnpa
```

ATOL and RTOL for NNPA are set using environment variables `TEST_ATOL` and `TEST_RTOL` in the `CMakeLists.txt`.
Also, the environment variables `TEST_INSTRUCTION_CHECK` and `TEST_CASE_BY_USER` allow you to check if the NNPA instruction is generated in the shared library. In `CMakeLists.txt`, `TEST_INSTRUCTION_CHECK` is set to true and `TEST_CASE_BY_USER` contains the test case and instruction name. If the instruction name is not found in the shared library, the test will fail.


[./ErrorHandling.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Handling errors in MLIR

Three are two different kinds of errors: errors that comes from user inputs, and compiler errors. We should provide meaningful user feedback for user input errors and we should use the `emitError` functions. Compiler errors should be reported using `asserts` or `llvm_unreachable` calls. In practice, if there are functions where errors are checked, and there is the ability to return "failure," the preferred way is to use `emitError` and return failure.  If, on the other hand, the function does not allow to return failure, then an assert or unreachable call should be used. Returning error is important for passes that check user inputs, e.g. such as during the ingestion of the ONNX model.

## User errors 

MLIR provides for 3 distinct calls depending on the severity: `emitError`, `emitWarning`, and 'emitRemark`. Errors should typically be reported to calling functions for proper handling. Typical use is as depicted below.

```cpp
  return op->emitError("message");
  return op->emitError() << "message";
```

Above calls will include the location of the operation. It returns a `LogicalResult` which can be set/tested as below. Note that the `emitError` calls return a `failure()` value;
```cpp
  LogicalResult isEven(int i) { return (i%2 == 0) success() : failure(); }

  if (succeeded(isEven(0)) && failed(isEven(1))) printf("It is all good.\n");
```

Errors can also be reported outside of the context of an operation. In this case, a location must be provided. To report a warning or a remark, just substitute "Warning" or "Remark" instead of "Error" in the above examples.

## Compiler errors

Once an ONNX graph has been validated, every subsequent erroneous  situations should be reported with an assert to stop the compilation, as this is a compiler error that needs to be properly handled. There are two calls that can be used:

```cpp
  assert(condition-that-should-hold-true && "error message");
  llvm_unreachable("error message");
```

The unreachable call is useful in functions that should return a value, as the compiler will not report warnings if there is no dummy-value return statement along that path. Otherwise, in `void` functions, using an assert is perfectly fine.


## References

Additional relevant information is found in the LLVM and MLIR documentation  referred below.
  
* [LLVM Docs on assert](https://llvm.org/docs/CodingStandards.html#assert-liberally)
* [MLIR Docs on diagnostic](https://mlir.llvm.org/docs/Diagnostics/)
  

[./Dialects/zhigh.md]:

<!-- Autogenerated by mlir-tblgen; don't manually edit -->
### `zhigh.Add` (::onnx_mlir::zhigh::ZHighAddOp)

_ZHigh Add operation_

ZHigh operation to perform an Add.
This operation does not support broadcasting.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH
| `Y` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.AvgPool2D` (::onnx_mlir::zhigh::ZHighAvgPool2DOp)

_ZHigh 2D average pooling operation_

ZHigh operation to perform 2D average pooling.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>padding_type</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC

### `zhigh.BatchNorm` (::onnx_mlir::zhigh::ZHighBatchNormOp)

_ZHigh batchnorm operation_

ZHigh operation to perform batchnorm.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC
| `a` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D
| `b` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC

### `zhigh.Conv2D` (::onnx_mlir::zhigh::ZHighConv2DOp)

_ZHigh 2D convolution operation_

ZHigh operation to perform 2D convolution.
* input: `[num_batches, height_in, width_in, channels_in]`
* input_kernel: `[kernel_height, kernel_width, channels_in, channels_out]`
* input_bias: `[channels_out] `
* kernel_shape: 1D array of kernel height and width
* strides: 1D array of stride height and width
* padding_type: SAME_PADDING or VALID_PADDING
* act_func: ACT_NONE or ACT_RELU
* output: `[num_batches, height_out, width_out, channels_out]`

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>padding_type</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>act_func</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC
| `input_kernel` | unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK
| `input_bias` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC

### `zhigh.DLF16ToF32` (::onnx_mlir::zhigh::ZHighDLF16ToF32Op)

_ZHigh DLF16ToF32 operation_

ZHigh operation to convert a tensor of dlfloat16 to a tensor of f32.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `In` | tensor of 16-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | tensor of 32-bit float values

### `zhigh.Div` (::onnx_mlir::zhigh::ZHighDivOp)

_ZHigh Div operation_

ZHigh operation to perform a Div.
This operation does not support broadcasting.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH
| `Y` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.Exp` (::onnx_mlir::zhigh::ZHighExpOp)

_ZHigh Exp operation_

ZHigh operation to perform a Exp.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.F32ToDLF16` (::onnx_mlir::zhigh::ZHighF32ToDLF16Op)

_ZHigh F32ToDLF16 operation_

ZHigh operation to convert a tensor of f32 to a tensor of dlfloat16.

Optional `saturation` indicates whether the CPU tensor is saturated before stickification
or not. If it is saturated, the dlfloat16 range would be used.
Saturation if off if `saturation == 0` or it is not given. Otherwise, it is on.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>saturation</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `In` | tensor of 32-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | tensor of 16-bit float values

### `zhigh.FixGRUY` (::onnx_mlir::zhigh::ZHighFixGRUYOp)

_Fix Y result of GRU for sequence_lens_

Fix Y result of GRU by padding value after sequence_lens.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `sequence_lens` | tensor of 32-bit signless integer values or none type
| `initial_h` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

### `zhigh.FixGRUYh` (::onnx_mlir::zhigh::ZHighFixGRUYhOp)

_Fix Yh result of GRU for sequence_lens_

Fix Yh result of GRU by picking the value in Y according to sequence_lens.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `sequence_lens` | tensor of 32-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

### `zhigh.GRU` (::onnx_mlir::zhigh::ZHighGRUOp)

_ZHigh GRU operation_

* zHigh operation to perform a GRU.
* Shape for input is `[S, B, I]`. Shape for h0 is `[D, B, H]`.
* Shape for input_weights is `[D, I, 3*H]`.
* Shape for hidden_weights is `[D, H, 3*H]`.
* Shape for input_bias and hidden_bias is `[D, 3*H]`.
* Shape for hn_output is `[S, D, B, H]` if return all timesteps
  and `[1, D, B, H]` if return the final step only.
* S is timesteps, D is the number of directions (1 for unidirectional and
* 2 for bidirectional), B is batch size, I is input size, and
* H is hidden size.
* direction accepts "forward", "reverse", or "bidirectional
* return_all_steps: -1 returns all timesteps, 0: returns only the last timestep."

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>hidden_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>direction</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>return_all_steps</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS
| `h0` | unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or none type
| `input_weights` | unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH
| `input_bias` | unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or none type
| `hidden_weights` | unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH
| `hidden_bias` | unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `hn_output` | unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS

### `zhigh.Gelu` (::onnx_mlir::zhigh::ZHighGeluOp)

_ZHigh Gelu operation_

"ZHigh operation to perform a Gelu."

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>approximate</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.InvSqrt` (::onnx_mlir::zhigh::ZHighInvSqrtOp)

_ZHigh InvSqrt operation_

ZHigh operation to perform a InvSqrt.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.LSTM` (::onnx_mlir::zhigh::ZHighLSTMOp)

_ZHigh LSTM operation_

zHigh operation to perform a LSTM.
* Shape for input is `[S, B, I]`. Shape for `h0` and `c0` is `[D, B, H]`.
* Shape for input_weights is  `[D, I, 4*H]`.
* Shape for hidden_weights is  `[D, H, 4*H]`.
* Shape for input_bias and hidden_bias is `[D, 4*H]`.
* Shape for hn_output is `[S, D, B, H]` if return all timesteps
  and `[1, D, B, H]` if return the final step only.
* Shape for cf_output is `[1, D, B, H]`.
* S is timesteps, D is the number of directions (1 for unidirectional and
* 2 for bidirectional), B is batch size, I is input size, and
* H is hidden size.
* direction accepts "forward", "reverse", or "bidirectional
* return_all_steps: -1 returns all timesteps, 0: returns only the last timestep.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>hidden_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>direction</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>return_all_steps</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS
| `h0` | unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or none type
| `c0` | unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or none type
| `input_weights` | unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO
| `input_bias` | unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or none type
| `hidden_weights` | unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO
| `hidden_bias` | unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `hn_output` | unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS
| `cf_output` | unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS

### `zhigh.LeakyRelu` (::onnx_mlir::zhigh::ZHighLeakyReluOp)

_ZHigh LeakyRelu operation_

"ZHigh operation to perform a LeakyRelu."

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>alpha</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.Log` (::onnx_mlir::zhigh::ZHighLogOp)

_ZHigh Log operation_

ZHigh operation to perform a Log.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.MatMul` (::onnx_mlir::zhigh::ZHighMatMulOp)

_ZHigh MatMul operation_

ZHigh operation to perform a MatMul.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>transposeA</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>transposeB</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS
| `Y` | unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS
| `B` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS

### `zhigh.Max` (::onnx_mlir::zhigh::ZHighMaxOp)

_ZHigh Max operation_

ZHigh operation to perform a Max.
This operation does not support broadcasting.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH
| `Y` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.MaxPool2D` (::onnx_mlir::zhigh::ZHighMaxPool2DOp)

_ZHigh 2D max pooling operation_

ZHigh operation to perform 2D max pooling.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>padding_type</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC

### `zhigh.MeanReduce2d` (::onnx_mlir::zhigh::ZHighMeanReduce2DOp)

_ZHigh 2D mean reduce operation_

ZHigh operation to perform 2D mean reduce. Given an input 4D tensor,
returns a downsampled tensor reducing the middle 2nd and 3rd dimensions
to a size of 1 based on the mean of the original values.
 Input and Output tensors should be in the 3D layout.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC

### `zhigh.Min` (::onnx_mlir::zhigh::ZHighMinOp)

_ZHigh Min operation_

ZHigh operation to perform a Min.
This operation does not support broadcasting.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH
| `Y` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.Mul` (::onnx_mlir::zhigh::ZHighMulOp)

_ZHigh Mul operation_

ZHigh operation to perform a Mul.
This operation does not support broadcasting.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH
| `Y` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.QuantizedMatMul` (::onnx_mlir::zhigh::ZHighQuantizedMatMulOp)

_ZHigh QuantizedMatMul operation_

ZHigh operation to perform a quantized MatMul.

`OutRecScaleIn` and `OutOffsetIn` are recscale and offset for the output.
If `OutRecScaleIn` is given, it will be passed to `OutRecScale`. If it is
None, `OutRescScale` is set to 1.0.
If `OutOffsetIn` is given, it will be passed to `OutOffset`. If it is
None, `OutOffset` is set to 0.0.

* PreComputedBias: -1 bias is re-computed, 0: bias is not pre-computed.

`DequantizeOutput` indicates if the output
is dequantized to real dfloat16 or not. If not, the output is int8 but stored in dlfloat (int8-as-dlfloat).
* DequantizeOutput: -1 output is dequantized, 0: output is not dequantized.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>PreComputedBias</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>DisableClipping</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>DequantizeOutput</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 8-bit signless integer or 16-bit float values or 2D tensor of 8-bit signless integer or 16-bit float values with layout _2D or unranked tensor of 8-bit signless integer or 16-bit float values or 3D tensor of 8-bit signless integer or 16-bit float values with layout _3DS
| `XRecScale` | 0D tensor of 32-bit float values
| `XOffset` | 0D tensor of 32-bit float values
| `Y` | unranked tensor of 8-bit signless integer or 16-bit float values or 2D tensor of 8-bit signless integer or 16-bit float values with layout _2D or unranked tensor of 8-bit signless integer or 16-bit float values or 3D tensor of 8-bit signless integer or 16-bit float values with layout _3DS
| `YRecScale` | 0D tensor of 32-bit float values
| `YOffset` | 0D tensor of 32-bit float values
| `B` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 8-bit signless integer or 16-bit float values or 1D tensor of 8-bit signless integer or 16-bit float values with layout _1D or unranked tensor of 8-bit signless integer or 16-bit float values or 2D tensor of 8-bit signless integer or 16-bit float values with layout _2DS or none type
| `BRecScale` | 0D tensor of 32-bit float values or none type
| `BOffset` | 0D tensor of 32-bit float values or none type
| `OutRecScaleIn` | 0D tensor of 32-bit float values or none type
| `OutOffsetIn` | 0D tensor of 32-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 8-bit signless integer or 16-bit float values or 2D tensor of 8-bit signless integer or 16-bit float values with layout _2D or unranked tensor of 8-bit signless integer or 16-bit float values or 3D tensor of 8-bit signless integer or 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS
| `OutRecScale` | 0D tensor of 32-bit float values
| `OutOffset` | 0D tensor of 32-bit float values

### `zhigh.QuantizedStick` (::onnx_mlir::zhigh::ZHighQuantizedStickOp)

_ZHigh QuantizedStick operation_

ZHigh operation to perform a quantized Stick.
Type is one of values: dlfloat16, int8, and weights.
`sym_mode` indicates whether to use symmetric quantization or not to compute the output rescale and offset.
`sym_mode` is only effective when the input rescale and offset are None.
By default, asymmetric quantization is used.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>quantized_type</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>sym_mode</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `In` | tensor of 32-bit float values or tensor of 8-bit signless integer values or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS
| `InRecScale` | 0D tensor of 32-bit float values or none type
| `InOffset` | 0D tensor of 32-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 8-bit signless integer or 16-bit float values or 1D tensor of 8-bit signless integer or 16-bit float values with layout _1D or unranked tensor of 8-bit signless integer or 16-bit float values or 2D tensor of 8-bit signless integer or 16-bit float values with layout _2D or unranked tensor of 8-bit signless integer or 16-bit float values or 3D tensor of 8-bit signless integer or 16-bit float values with layout _3D or unranked tensor of 8-bit signless integer or 16-bit float values or 2D tensor of 8-bit signless integer or 16-bit float values with layout _2DS or unranked tensor of 8-bit signless integer or 16-bit float values or 3D tensor of 8-bit signless integer or 16-bit float values with layout _3DS or none type
| `RecScale` | 0D tensor of 32-bit float values
| `Offset` | 0D tensor of 32-bit float values

### `zhigh.ReduceMax` (::onnx_mlir::zhigh::ZHighReduceMaxOp)

_ZHigh ReduceMax operation_

ZHigh operation to perform a ReduceMax.
op_type: REDUCE_OP_MAXIMUM or REDUCE_OP_MINIMUM.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.ReduceMin` (::onnx_mlir::zhigh::ZHighReduceMinOp)

_ZHigh ReduceMin operation_

ZHigh operation to perform a ReduceMin.
op_type: REDUCE_OP_MAXIMUM or REDUCE_OP_MINIMUM.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.Relu` (::onnx_mlir::zhigh::ZHighReluOp)

_ZHigh Relu operation_

"ZHigh operation to perform a Relu."

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.Reshape` (::onnx_mlir::zhigh::ZHighReshapeOp)

_ZHigh Reshape operation for Z Tensors_

ZHigh operation to perform a converts a Z Tensor from one type to an equivalent type
with a provided shape. The data is never copied or modified. When no layout is specified,
the output preserve the same layout as the source input.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `source` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH
| `shape` | tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.Sigmoid` (::onnx_mlir::zhigh::ZHighSigmoidOp)

_ZHigh Sigmoid operation_

ZHigh operation to perform a Sigmoid.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.Softmax` (::onnx_mlir::zhigh::ZHighSoftmaxOp)

_ZHigh Softmax operation_

ZHigh operation to perform a Softmax.
act_func: ACT_NONE or ACT_LOG.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>act_func</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS

### `zhigh.Sqrt` (::onnx_mlir::zhigh::ZHighSqrtOp)

_ZHigh Sqrt operation_

ZHigh operation to perform a Sqrt.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.StickForGRU` (::onnx_mlir::zhigh::ZHighStickForGRUOp)

_ZHigh stick operation for GRU_

ZHigh operation to perform a stick for GRU.
Variadic: list of pointers for input data to be transformed:
  - GRU concatenated: 3 data pointers, one for each input gate in
(Z)update, Reset, Hidden, (ZRH) gate order

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `z_gate` | tensor of 32-bit float values
| `r_gate` | tensor of 32-bit float values
| `h_gate` | tensor of 32-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `out` | unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH

### `zhigh.StickForLSTM` (::onnx_mlir::zhigh::ZHighStickForLSTMOp)

_ZHigh stick operation for LSTM_

ZHigh operation to perform a stick for LSTM.
Variadic: list of pointers for input data to be transformed:
  - LSTM concatenated: 4 data pointers, one for each input gate in
Forget, Input, Cell, Output (FICO) order,

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `f_gate` | tensor of 32-bit float values
| `i_gate` | tensor of 32-bit float values
| `c_gate` | tensor of 32-bit float values
| `o_gate` | tensor of 32-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `out` | unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO

### `zhigh.Stick` (::onnx_mlir::zhigh::ZHighStickOp)

_ZHigh Stick operation_

ZHigh operation to perform a Stick."

If `layout`=`NHWC`, input must be in `NCHW` and output will be in `NHWC`.

Optional `saturation` indicates whether the CPU tensor is saturated before stickification
or not. If it is saturated, the dlfloat16 range would be used.
Saturation if off if `saturation == 0` or it is not given. Otherwise, it is on.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>saturation</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `In` | tensor of 32-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or none type

### `zhigh.StickifiedConstantOfShape` (::onnx_mlir::zhigh::ZHighStickifiedConstantOfShapeOp)

_ZHigh Stickified Constant operation for a dynamic shape_

This operator produces a constant tensor to store stickified data.
The stickified data is defined by a f32 scalar value, a dynamic shape
and a layout. Stickified data is 4K-aligned.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>value</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `shape` | tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.StickifiedConstant` (::onnx_mlir::zhigh::ZHighStickifiedConstantOp)

_ZHigh Stickified Constant operation_

This operator produces a constant tensor to store stickified data.
Stickified data is opaque and must be 4K-aligned. One who produces
the stickified data must make sure its size in bytes consistent with
the output tensor's size.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>value</code></td><td>::mlir::Attribute</td><td>any attribute</td></tr>
<tr><td><code>alignment</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH or unranked tensor of 8-bit signless integer or 16-bit float values or 1D tensor of 8-bit signless integer or 16-bit float values with layout _1D or unranked tensor of 8-bit signless integer or 16-bit float values or 2D tensor of 8-bit signless integer or 16-bit float values with layout _2D or unranked tensor of 8-bit signless integer or 16-bit float values or 3D tensor of 8-bit signless integer or 16-bit float values with layout _3D or unranked tensor of 8-bit signless integer or 16-bit float values or 2D tensor of 8-bit signless integer or 16-bit float values with layout _2DS or unranked tensor of 8-bit signless integer or 16-bit float values or 3D tensor of 8-bit signless integer or 16-bit float values with layout _3DS

### `zhigh.Sub` (::onnx_mlir::zhigh::ZHighSubOp)

_ZHigh Sub operation_

ZHigh operation to perform a Sub.
This operation does not support broadcasting.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH
| `Y` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.Tanh` (::onnx_mlir::zhigh::ZHighTanhOp)

_ZHigh Tanh operation_

ZHigh operation to perform a Tanh.

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultLayout`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout FICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout ZRH or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BFICO or unranked tensor of 16-bit float values or 2D/3D tensor of 16-bit float values with layout BZRH

### `zhigh.Unstick` (::onnx_mlir::zhigh::ZHighUnstickOp)

_ZHigh Unstick operation_

ZHigh operation to perform a Unstick.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `In` | unranked tensor of 16-bit float values or 1D tensor of 16-bit float values with layout _1D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2D or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3D or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4D or unranked tensor of 16-bit float values or 2D tensor of 16-bit float values with layout _2DS or unranked tensor of 16-bit float values or 3D tensor of 16-bit float values with layout _3DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout _4DS or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NHWC or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout NCHW or unranked tensor of 16-bit float values or 4D tensor of 16-bit float values with layout HWCK

#### Results:

| Result | Description |
| :----: | ----------- |
| `Out` | tensor of 32-bit float values



[./Dialects/zlow.md]:

<!-- Autogenerated by mlir-tblgen; don't manually edit -->
### `zlow.add` (::onnx_mlir::zlow::ZLowAddOp)

_ZLow add operation_

ZLow operation to perform an add.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `Y` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.avgpool2d` (::onnx_mlir::zlow::ZLowAvgPool2DOp)

_ZLow 2D average pooling operation_

ZLow operation to perform 2D average pooling.
* shape is a 1D MemRef (memref<6xi64>) whose items are:
  * 1st item: batch size
  * 2nd item: channel
  * 3rd item: height in
  * 4th item: width in
  * 5th item: height out
  * 6th item: width out
* kernel_shape: 1D array of kernel height and width
* strides: 1D array of stride height and width
* padding_type: SAME_PADDING or VALID_PADDING.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>padding_type</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `output` | memref of dlfloat16 type values

### `zlow.batchnorm` (::onnx_mlir::zlow::ZLowBatchNormOp)

_ZLow batchnorm operation_

ZLow operation to perform batchnorm.
* shape is a 1D MemRef (memref<4xi64>) whose items are:
  * 1st item: batch size
  * 2nd item: height
  * 3rd item: width
  * 4th item: channel

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | memref of dlfloat16 type values
| `A` | memref of dlfloat16 type values
| `B` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `output` | memref of dlfloat16 type values

### `zlow.conv2d` (::onnx_mlir::zlow::ZLowConv2DOp)

_ZLow 2D convolution operation_

ZLow operation to perform 2D convolution.
* shape is a 1D MemRef (memref<7xi64>) whose items are:
  * 1st item: batch size
  * 2nd item: channel in
  * 3rd item: height in
  * 4th item: width in
  * 5th item: channel out
  * 6th item: height out
  * 7th item: width out
* kernel_shape: 1D array of kernel height and width
* strides: 1D array of stride height and width
* padding_type: SAME_PADDING or VALID_PADDING.
* act_func: ACT_NONE or ACT_RELU.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>padding_type</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>act_func</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | memref of dlfloat16 type values
| `input_kernel` | memref of dlfloat16 type values
| `input_bias` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `output` | memref of dlfloat16 type values

### `zlow.dlf16_to_f32` (::onnx_mlir::zlow::ZLowConvertDLF16ToF32Op)

_Convert a dlfloat16 value to a float32 value_

This operation converts a dlfloat16 value to a float32 value. 

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | dlfloat16 type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | 32-bit float

### `zlow.vec_dlf16_to_f32` (::onnx_mlir::zlow::ZLowConvertDLF16ToF32VectorOp)

_Convert dlfloat16 values to float32 values_

This operation converts dlfloat16 values to float32 values. 

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | vector of 16-bit float values of length 8

#### Results:

| Result | Description |
| :----: | ----------- |
| `output1` | vector of 32-bit float values of length 4
| `output2` | vector of 32-bit float values of length 4

### `zlow.f32_to_dlf16` (::onnx_mlir::zlow::ZLowConvertF32ToDLF16Op)

_Convert a float32 value to a dlfloat16 value_

This operation converts a float32 value to a dlfloat16 value. 

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | 32-bit float

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | dlfloat16 type

### `zlow.vec_f32_to_dlf16` (::onnx_mlir::zlow::ZLowConvertF32ToDLF16VectorOp)

_Convert float32 values to dlfloat16 values_

This operation converts float32 values to dlfloat16 values. 

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input1` | vector of 32-bit float values of length 4
| `input2` | vector of 32-bit float values of length 4

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | vector of 16-bit float values of length 8

### `zlow.div` (::onnx_mlir::zlow::ZLowDivOp)

_ZLow div operation_

ZLow operation to perform a div.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `Y` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.dummy` (::onnx_mlir::zlow::ZLowDummyOp)

_ZLow dummy operation that behaves like identity_

ZLow operation to forward the input value to the output value.
It will be removed if canonicalization is called.

Traits: `MemRefsNormalizable`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | any type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | any type

### `zlow.exp` (::onnx_mlir::zlow::ZLowExpOp)

_ZLow exp operation_

ZLow operation to perform a exp.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.gru` (::onnx_mlir::zlow::ZLowGRUOp)

_ZLow gru operation_

ZLow operation to perform a gru.
* work_area: a 4K-aligned buffer.
* shape is a 1D MemRef (memref<5xi64>) whose items are:;
  * 1st item: direction
  * 2nd item: timestep
  * 3rd item: batchSize
  * 4th item: featureSize
  * 5th item: hiddenSize
* direction accepts "forward", "reverse", or "bidirectional"
* return_all_steps: -1 returns all timesteps, 0: returns only the last timestep.
* prev_layer for where input comes is "none", "uni", or "bidir"

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>direction</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>return_all_steps</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>prev_layer</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | memref of dlfloat16 type values
| `h0` | memref of dlfloat16 type values
| `input_weights` | memref of dlfloat16 type values
| `input_bias` | memref of dlfloat16 type values
| `hidden_weights` | memref of dlfloat16 type values
| `hidden_bias` | memref of dlfloat16 type values
| `work_area` | memref of 8-bit signless integer values
| `shape` | memref of 64-bit signless integer values
| `hn_output` | memref of dlfloat16 type values

### `zlow.gelu` (::onnx_mlir::zlow::ZLowGeluOp)

_ZLow gelu operation_

ZLow operation to perform a gelu.

Traits: `MemRefsNormalizable`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.invsqrt` (::onnx_mlir::zlow::ZLowInvSqrtOp)

_ZLow invsqrt operation_

ZLow operation to perform a invsqrt.

Traits: `MemRefsNormalizable`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.lstm` (::onnx_mlir::zlow::ZLowLSTMOp)

_ZLow lstm operation_

ZLow operation to perform a lstm.
work_area: a 4K-aligned buffer.
* shape is a 1D MemRef (memref<5xi64>) whose items are:
  * 1st item: direction
  * 2nd item: timestep
  * 3rd item: batchSize
  * 4th item: featureSize
  * 5th item: hiddenSize
* direction accepts "forward", "reverse", or "bidirectional"
* return_all_steps: -1 returns all timesteps, 0: returns only the last timestep
* prev_layer for where input comes is "none", "uni", or "bidir"

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>direction</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>return_all_steps</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>prev_layer</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | memref of dlfloat16 type values
| `h0` | memref of dlfloat16 type values
| `c0` | memref of dlfloat16 type values
| `input_weights` | memref of dlfloat16 type values
| `input_bias` | memref of dlfloat16 type values
| `hidden_weights` | memref of dlfloat16 type values
| `hidden_bias` | memref of dlfloat16 type values
| `work_area` | memref of 8-bit signless integer values
| `shape` | memref of 64-bit signless integer values
| `hn_output` | memref of dlfloat16 type values
| `cf_output` | memref of dlfloat16 type values

### `zlow.leakyrelu` (::onnx_mlir::zlow::ZLowLeakyReluOp)

_ZLow leakyrelu operation_

ZLow operation to perform a leakyrelu.

Traits: `MemRefsNormalizable`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>alpha</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.log` (::onnx_mlir::zlow::ZLowLogOp)

_ZLow log operation_

ZLow operation to perform a log.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.matmul` (::onnx_mlir::zlow::ZLowMatMulOp)

_ZLow matmul operation_

ZLow operation to perform a matmul.
* In case of unstacked: X(m, n) * Y(n, p) + Bias(p)
shape is a 1D MemRef (memref<3xi64>) whose items are:
  * 1st item: m
  * 2nd item: n
  * 3rd item: p
* In case of stacked: X(s, m, n) * Y(s, n, p) + Bias(s, p)
     or broadcasting1: X(m, n) * Y(s, n, p) + Bias(s, p)
     or broadcasting23: X(s, m, n) * Y(n, p) + Bias(p)
shape is a 1D MemRef (memref<4xi64>) whose items are:
  * 1st item: s
  * 2nd item: m
  * 3rd item: n
  * 4th item: p
* is_bcast1:  -1 broadcasting1, 0: no broadcasting1.
* is_bcast23: -1 broadcasting23, 0: no broadcasting23.
* is_stacked: -1 stacked, 0: unstacked.
* transposeA: !0 transpose A, 0: do not transpose A.
* transposeB: !0 transpose B, 0: do not transpose B.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>is_bcast1</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>is_bcast23</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>is_stacked</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>transposeA</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>transposeB</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `Y` | memref of dlfloat16 type values
| `Bias` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.max` (::onnx_mlir::zlow::ZLowMaxOp)

_ZLow max operation_

ZLow operation to perform a max.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `Y` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.maxpool2d` (::onnx_mlir::zlow::ZLowMaxPool2DOp)

_ZLow 2D max pooling operation_

ZLow operation to perform 2D max pooling.
* shape is a 1D MemRef (memref<6xi64>) whose items are:
  * 1st item: batch size
  * 2nd item: channel
  * 3rd item: height in
  * 4th item: width in
  * 5th item: height out
  * 6th item: width out
* kernel_shape: 1D array of kernel height and width
* strides: 1D array of stride height and width
* padding_type: SAME_PADDING or VALID_PADDING.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>padding_type</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `output` | memref of dlfloat16 type values

### `zlow.meanreduce2d` (::onnx_mlir::zlow::ZLowMeanReduce2DOp)

_ZLow 2D mean reduce operation_

ZLow operation to perform 2D mean reduce.
* shape is a 1D MemRef (memref<4xindex>) whose items are:;
  * 1st item: batch size": 1st dim of input
  * 2rd item: height": 2nd dim of input
  * 3th item: width": 3rd dim of input
  * 4nd item: channel": 4th dim of input

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `output` | memref of dlfloat16 type values

### `zlow.min` (::onnx_mlir::zlow::ZLowMinOp)

_ZLow min operation_

ZLow operation to perform a min.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `Y` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.mul` (::onnx_mlir::zlow::ZLowMulOp)

_ZLow mul operation_

ZLow operation to perform a mul.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `Y` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.quantizedMatmul` (::onnx_mlir::zlow::ZLowQuantizedMatMulOp)

_ZLow quantized matmul operation_

ZLow operation to perform a matmul.
work_area: a 4K-aligned buffer having the same layout as bias but dlfloat16 type.
* In case of unstacked: X(m, n) * Y(n, p) + Bias(p)
shape is a 1D MemRef (memref<3xi64>) whose items are:
  * 1st item: m
  * 2nd item: n
  * 3rd item: p
* In case of stacked: X(s, m, n) * Y(s, n, p) + Bias(s, p)
     or broadcasting: X(s, m, n) * Y(n, p) + Bias(p)
shape is a 1D MemRef (memref<4xi64>) whose items are:
  * 1st item: s
  * 2nd item: m
  * 3rd item: n
  * 4th item: p
* is_bcast: -1 broadcasting, 0: no broadcasting.
* is_stacked: -1 stacked, 0: unstacked.
* DequantizeOutput: -1 output is dequantized, 0: output is not dequantized.
* PreComputedBias: -1 bias is re-computed, 0: bias is not pre-computed.

Values for `q_type` are "DLFLOAT16", "INT8", "WEIGHTS", "UNDEFINED".


Traits: `MemRefsNormalizable`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>x_q_type</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>y_q_type</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>bias_q_type</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>out_q_type</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>is_bcast</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>is_stacked</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>pre_computed_bias</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>disable_clipping</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>dequantize_output</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type or 8-bit signless integer values
| `x_rec_scale` | 0D memref of 32-bit float values
| `x_offset` | 0D memref of 32-bit float values
| `Y` | memref of dlfloat16 type or 8-bit signless integer values
| `y_rec_scale` | 0D memref of 32-bit float values
| `y_offset` | 0D memref of 32-bit float values
| `Bias` | memref of dlfloat16 type or 8-bit signless integer values
| `bias_rec_scale` | 0D memref of 32-bit float values
| `bias_offset` | 0D memref of 32-bit float values
| `work_area` | memref of dlfloat16 type or 8-bit signless integer values or none type
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type or 8-bit signless integer values
| `out_rec_scale` | 0D memref of 32-bit float values
| `out_offset` | 0D memref of 32-bit float values

### `zlow.quantizedStick` (::onnx_mlir::zlow::ZLowQuantizedStickOp)

_ZLow stick operation for quantization_

"ZLow operation to perform a quantization stick."
"Type is one of values: dlfloat16, int8, and weights."

Traits: `MemRefsNormalizable`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>q_type</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of 8-bit signless integer or 32-bit float values
| `rec_scale` | 0D memref of 32-bit float values
| `offset` | 0D memref of 32-bit float values
| `out` | memref of dlfloat16 type or 8-bit signless integer values

### `zlow.reducemax` (::onnx_mlir::zlow::ZLowReduceMaxOp)

_ZLow reducemax operation_

ZLow operation to perform a reducemax.

Traits: `MemRefsNormalizable`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `work_area` | memref of 8-bit signless integer values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.reducemin` (::onnx_mlir::zlow::ZLowReduceMinOp)

_ZLow reducemin operation_

ZLow operation to perform a reducemin.

Traits: `MemRefsNormalizable`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `work_area` | memref of 8-bit signless integer values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.relu` (::onnx_mlir::zlow::ZLowReluOp)

_ZLow relu operation_

ZLow operation to perform a relu.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.reshape` (::onnx_mlir::zlow::ZLowReshapeOp)

_ZLow Reshape operation_

ZLow operation to perform a reshape (no data movement).

Traits: `MemRefsNormalizable`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `Out` | memref of dlfloat16 type values

### `zlow.sigmoid` (::onnx_mlir::zlow::ZLowSigmoidOp)

_ZLow sigmoid operation_

ZLow operation to perform a sigmoid.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.softmax` (::onnx_mlir::zlow::ZLowSoftmaxOp)

_ZLow softmax operation_

ZLow operation to perform a softmax.
work_area: a 4K-aligned buffer.
act_func: ACT_NONE or ACT_LOG.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>act_func</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `work_area` | memref of 8-bit signless integer values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.sqrt` (::onnx_mlir::zlow::ZLowSqrtOp)

_ZLow sqrt operation_

ZLow operation to perform a sqrt.

Traits: `MemRefsNormalizable`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.stickForGRU` (::onnx_mlir::zlow::ZLowStickForGRUOp)

_ZLow stick operation for GRU_

ZLow operation to perform a stick for GRU.
Variadic: list of pointers for input data to be transformed: 
  - GRU concatenated: 3 data pointers, one for each input gate in (Z)update, Reset, Hidden, (ZRH) gate order.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>prev_layer</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `z_gate` | memref of 16-bit float or 32-bit float values
| `r_gate` | memref of 16-bit float or 32-bit float values
| `h_gate` | memref of 16-bit float or 32-bit float values
| `out` | memref of dlfloat16 type values

### `zlow.stickForLSTM` (::onnx_mlir::zlow::ZLowStickForLSTMOp)

_ZLow stick operation for LSTM_

ZLow operation to perform a stick for LSTM.
Variadic: list of pointers for input data to be transformed: 
  - LSTM concatenated: 4 data pointers, one for each input gate in Forget, Input, Cell, Output (FICO) order.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>prev_layer</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `f_gate` | memref of 16-bit float or 32-bit float values
| `i_gate` | memref of 16-bit float or 32-bit float values
| `c_gate` | memref of 16-bit float or 32-bit float values
| `o_gate` | memref of 16-bit float or 32-bit float values
| `out` | memref of dlfloat16 type values

### `zlow.stick` (::onnx_mlir::zlow::ZLowStickOp)

_ZLow stick operation_

"ZLow operation to perform a stick."

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>saturation</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of 16-bit float or 32-bit float values
| `Out` | memref of dlfloat16 type values

### `zlow.sub` (::onnx_mlir::zlow::ZLowSubOp)

_ZLow sub operation_

ZLow operation to perform a sub.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `Y` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.tanh` (::onnx_mlir::zlow::ZLowTanhOp)

_ZLow tanh operation_

ZLow operation to perform a tanh.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `shape` | memref of 64-bit signless integer values
| `Out` | memref of dlfloat16 type values

### `zlow.unstick` (::onnx_mlir::zlow::ZLowUnstickOp)

_ZLow unstick operation_

ZLow operation to perform a unstick.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>layout</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of dlfloat16 type values
| `Out` | memref of 16-bit float or 32-bit float values



[./Dialects/krnl.md]:

<!-- Autogenerated by mlir-tblgen; don't manually edit -->
### `krnl.acos` (KrnlAcosOp)

_Krnl acos scalar operation_

Krnl acos scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
| `out` | floating-point

### `krnl.acosh` (KrnlAcoshOp)

_Krnl acosh scalar operation_

Krnl acosh scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
| `out` | floating-point

### `krnl.asin` (KrnlAsinOp)

_Krnl asin scalar operation_

Krnl asin scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
| `out` | floating-point

### `krnl.asinh` (KrnlAsinhOp)

_Krnl asinh scalar operation_

Krnl asinh scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
| `out` | floating-point

### `krnl.atan` (KrnlAtanOp)

_Krnl atan scalar operation_

Krnl atan scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
| `out` | floating-point

### `krnl.atanh` (KrnlAtanhOp)

_Krnl atanh scalar operation_

Krnl atanh scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
| `out` | floating-point

### `krnl.block` (KrnlBlockOp)

_Krnl block operation_


Syntax:

```
operation ::= `krnl.block` $loop $tile_size attr-dict `:` functional-type($loop, results)
```

Block a single for loop by a constant tile size. For instance,
```
$ib, $il = krnl.block %i, 4
```
means to block the for loop referred to by %i using a tile size of 4.

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>tile_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `loop` | any type

#### Results:

| Result | Description |
| :----: | ----------- |
| `loop_block` | any type
| `loop_local` | any type

### `krnl.call` (KrnlCallOp)

_Call operation_

The call operation provides a generic way to replace an ONNX Op with a call
to an external function at Krnl level.
`funcName` attributes determines which function to call.
`parameters` is the inputs to Krnl.Call. It includes the outputs and inputs
of the ONNX Op. The outputs and inputs are already lowered to MemRefs.
The external function is assumed NOT to allocate or free any memory.
'numOfOutput` attribute to tell how manu outputs Memref in parameters.
mlir::OpTrait::AttrSizedOperandSegments is not used to put outputs and
inputs into separate variadic parameters because I am thinking of mixing
the inputs and outpus as required by external library.

The attributes of the ONNX Op will be copied to KrnlCallOp under the control
of the user.
In Krnl To llvm lowering, the parameters and attributes will be lowered to
parameters of the llvm function call.

Several builder is defined to help translating an ONNX Op to Krnl.Call.
User can provides the allocated MemRefs for outputs and the inputs
separately. The inputs are usually the operands of the ONNX Op.
The attributes of ONNX Op can be copied or not copied based on a bool
parameter in the builder. Builder also provide a mechanism for user
to selectively copy some attributes.

The krnl.call op will be lowered to llvm at krnl-to-llvm conversion in which
OMTensor is used as a container for MemRef arguments. Other representation
of parameters, such as data pointer only, will be supported in future.

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>funcName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>numOfOutput</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `parameters` | variadic of any type

#### Results:

| Result | Description |
| :----: | ----------- |
| `returnValue` | variadic of floating-point or integer

### `krnl.copy_from_tile_buffer` (KrnlCopyFromBufferOp)

_Copy from buffer._


Syntax:

```
operation ::= `krnl.copy_from_tile_buffer` $buffer `,` $dest `[` $starts `]`  attr-dict `:` type($buffer) `,` type($dest)
```

Operation that copy a destination memory from a buffer memory.
Starts indicate where the buffer data starts to go into the destination
memory. Start values must be at multiples of buffer size in all dimensions.
The buffer rank and dimensions are compile time constants.

If the buffer was oversized with respect of the actual data contained
in the tile, the actual tile size can be given using the tileSize
optional attribute. This attributes has the same rank as the buffer size,
and each dimension must be smaller or equal to the actual buffer size.

Traits: `MemRefsNormalizable`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>tileSize</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `buffer` | memref of any type values
| `dest` | memref of any type values
| `starts` | variadic of index

### `krnl.copy_to_tile_buffer` (KrnlCopyToBufferOp)

_Copy to buffer._


Syntax:

```
operation ::= `krnl.copy_to_tile_buffer` $buffer `,` $source `[` $starts `]` `,`  $padValue  attr-dict
              `:` type($buffer) `,` type($source)
```

Operation that copy a source memory to a buffer memory.
Starts indicate where the source data starts to come from within
the source memory. Start values must be at multiples of buffer size
in all dimensions. The buffer rank and dimensions are compile time
constants.

The buffer will be entirely filled with the source data. By default,
the amount of data to copy is given by the size of the buffer.
In some cases, we may want to oversize a buffer for better cache,
simd, or loop unroll and jam reasons. If that is the case, the
actual tile size of the data to be copied over is given by an
optional tileSize attribute. This attributes has the same rank as
the buffer size, and each dimension must be smaller or equal to
the actual buffer size.

If there is not enough data in the source memory to fill the buffer,
because the operation reaches the upper bounds of the source memory,
several actions may happen.

* If`padToNext` attribute is given, the pad value will be copied from
  the last source data of to the next index for which index modulo `padToNext`
  is zero, i.e. to the end of a "cache line" of side `padToLine`. Pad
  of 1 means no padding, pad of buffer size means fully pad the buffer.
  Default is no padding (1). `PadValue` is used to initialized the padded
  areas.

* If `overreadToNext` attribute is given, the copy may read source past
  its upper bound value. This enable optimized code, e.g. using SIMD
  read operations even if going past the last value of the source
  memory, or unrolling and jamming copy loops to reduce memory latency.
  `overreadToNext` is expressed like padToNext: value of 1 means no
  reading past boundary; value of buffer size enables reading
  as many additional source value as needed to fill the full
  buffer. Default is buffer-size.

`padToNext` and `overreadToNex`t are of the same rank as source and memory
memrefs.

Traits: `MemRefsNormalizable`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>tileSize</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>padToNext</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>transpose</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `buffer` | memref of any type values
| `source` | memref of any type values
| `starts` | variadic of index
| `padValue` | any type

### `krnl.define_loops` (KrnlDefineLoopsOp)

_Define_loops operation_

The "krnl.define_loops" operation is used to define input loops,
those are the for loops appearing in the input program that we
intend to optimize.

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Results:

| Result | Description |
| :----: | ----------- |
&laquo;unnamed&raquo; | variadic of any type

### `krnl.entry_point` (KrnlEntryPointOp)

_Indicate ONNX entry point_

The "krnl.entry_point" function indicates the main entry
                           point of ONNX model.

### `krnl.erf` (KrnlErfOp)

_Krnl erf scalar operation_

Krnl erf scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
| `out` | floating-point

### `krnl.find_index` (KrnlFindIndexOp)

_Retrieve an index into a perfect hash table described by G and V._

This operation can be used to generate a call to a runtime function which,
given two arrays of int32_t values (G and V), which are used to represent a perfect
hash table for a dictionary, returns the index corresponding to the input value.
The index returned is valid only if 'input' is in the dictionary described by G and V.

Traits: `AlwaysSpeculatableImplTrait`, `MemRefsNormalizable`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | string type or 64-bit signless integer
| `G` | memref of 32-bit signless integer values
| `V` | memref of 32-bit signless integer values
| `len` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
| `index` | index

### `krnl.get_induction_var_value` (KrnlGetInductionVariableValueOp)

_Krnl_


Syntax:

```
operation ::= `krnl.get_induction_var_value` `(` $loops `)` attr-dict `:` functional-type($loops, results)
```

Krnl operation to convert loop references to corresponding induction
variable values. This is useful for accessing optimized loop induction
variables, as they are not otherwise accessible during Krnl Dialect.

For example, this operation can be applied to loop references corresponding to
inter-tile iterations. The return values will be the starting index of the
current tile being iterated over.

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `loops` | variadic of any type

#### Results:

| Result | Description |
| :----: | ----------- |
| `ind_var_vals` | variadic of any type

### `krnl.get_linear_offset_index` (KrnlGetLinearOffsetIndexOp)

_A Krnl operation to compute a linear offset index from a N-D index._

Given a MemRef and an N-D index (id_1, id_2, ..., id_n), where n is
the rank of the MemRef, this operation computes a linear offset index.

Traits: `MemRefsNormalizable`

Interfaces: `AffineMapAccessInterface`, `AffineReadOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>map</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `memref` | memref of any type values
| `indices` | variadic of index

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | index

### `krnl.global` (KrnlGlobalOp)

_Krnl global operation_

Operation for holding global data values. A global constant can have a
meaningful name recorded as its `name` attribute. Its content is stored
in the `value` dense element attribute.

Traits: `AlwaysSpeculatableImplTrait`, `MemRefsNormalizable`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>shape</code></td><td>::mlir::Attribute</td><td>any attribute</td></tr>
<tr><td><code>name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>value</code></td><td>::mlir::Attribute</td><td>any attribute</td></tr>
<tr><td><code>offset</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>alignment</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | memref of any type values

### `krnl.runtime_instrument` (KrnlInstrumentOp)

_Instrumentation point._

Operation that invokes the runtime instrument utility.
May be used for gdb.

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>opName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>tag</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>nodeName</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

### `krnl.isinf` (KrnlIsInfOp)

_Krnl isinf scalar operation_

Krnl isinf scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
| `out` | 1-bit signless integer

### `krnl.isnan` (KrnlIsNaNOp)

_Krnl isnan scalar operation_

Krnl isnan scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
| `out` | 1-bit signless integer

### `krnl.iterate` (KrnlIterateOp)

_Iterate operation_

The "krnl.iterate" operation is conceptually equivalent to a nested for loops.

For instance, say we have the following two
```
%l0, %l1 = krnl.define_loops 2
%o0, %o1 = krnl.optimize_loops  {
    // Identity schedule.
    krnl.return_loops %l0, %l1
}
```

Then, consider the following krnl.iterate operation:
```
krnl.iterate (%o0, %o1) with (%l0 -> %i0 = 0 to 10, %l1 -> %i1 = 0 to 10) {
  // Some operations.
}
```

It is equivalent to:
```
for (i0 = 0; i0 < 10; i0++)
  for (i1 = 0; i1 < 10; i1++)
    // Some operations.
```

Traits: `RecursiveMemoryEffects`, `SingleBlockImplicitTerminator<KrnlYieldOp>`, `SingleBlock`

Interfaces: `LoopLikeOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
&laquo;unnamed&raquo; | variadic of any type

#### Results:

| Result | Description |
| :----: | ----------- |
| `results` | variadic of any type

### `krnl.load` (KrnlLoadOp)

_A Krnl operation to load data from the memref._


Syntax:

```
operation ::= `krnl.load` $memref `[` $indices `]` attr-dict `:` type($memref)
```

The `krnl.load` op reads an element from a memref specified by an index
list. The output of load is a new value with the same type as the elements
of the memref. The arity of indices is the rank of the memref (i.e., if the
memref loaded from is of rank 3, then 3 indices are required for the load
following the memref identifier).

Traits: `MemRefsNormalizable`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `memref` | memref of any type values
| `indices` | variadic of index

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | any type

### `krnl.matmul` (KrnlMatMulOp)

_Matmul operation for a single pannel._


Syntax:

```
operation ::= `krnl.matmul` $A `[` $aGlobalIndexMemStart `]` `,`
              $B `[` $bGlobalIndexMemStart `]` `,`
              $C `[` $cGlobalIndexMemStart `]` `,`
              `(` $loops `)` `,`
              `(` $iGlobalIndexComputeStart `,` $jGlobalIndexComputeStart `,`
              $kGlobalIndexComputeStart `)` `,`
              `(` $iGlobalUB `,` $jGlobalUB `,` $kGlobalUB `)`
              attr-dict `:` type($A) `,` type($B)`,` type($C) `,` `(` type($loops) `)`
```

    Perform a matrix multiplication AA * BB + CC with sizes `[IxK] * [KxJ] + [IxJ]`.
    The original matrices AA, BB, and CC can be buffered in buffered arrays
    which may be padded. The original matrices and the padded array might
    have a higher rank than 2, but the actual matrix multiplication operation
    only deal with the innermost 2 ranks of the matrices to perform its matrix
    multiplication operations.

    The computations may also compute only a sub-tile of the buffered arrays.
    This region is depicted using stars '*' below.

    All indices passed to this operation are the global indices in the original
    computation, so as to better know if we have boundary conditions.

    ORIGINAL ARRAY: denoted as AA, BB, CC with sizes AA: `*xIxK`; BB: `*xKxJ`; CC: `*xI*J`).

    BUFFER ARRAYS: denoted as A, B, and C. Note that this operation does
      not require the use of buffers arrays. If none are used, then A=AA,
      B=BB, C=CC. If buffers are used, it is the responsibility of the caller
      to properly fill the buffers with the appropriate data. Buffers are
      typically used for cache tiling.

     ORIGINAL ARRAY

```
     -------------------------------------------------
     |                                               ]
     |                                               ]
     |             buffer array       buffer pad     ]
     |            (3)---------------- ++++           ]
     |             |                 |   +           ]
     |             |     (1)****     |   +           ]
     |             |      *    *     |   +           ]
     |             |      *    *     |   +           ]
     |             |      ****(5)    |   +           ]
     |             |                 |   +           ]
     |             |                 |   +           ]
     |             ------------------|   +           ]
     |             +                     +           ]
     |             +++++++++++++++++++++(4)          ]
     |                                               ]
     -----------------------------------------------(2)
```

* (1) `iGlobalIndexComputeStart`/`jGlobalIndexComputeStart`/`kGlobalIndexComputeStart`,
   required, each three are global 1D indices.
* (2) `iGlobalUB`/`jGlobalUB`/`jGlobalUB`, required, each three are global 1D indices.
* (3) `aGlobalIndexMemStart`/`bGlobalIndexMemStart`/`cGlobalIndexMemStart`,
   required, global nD indices with the same rank as the buffers A, B, and C.
* (4) `aTileSize`/`bTileSize`/`cTileSize`, required when padding, each 2D sizes.
* (5) `computeTileSizes`, required when tiled computation within buffer, 3D sizes (I, J, K).

    The `iGlobalIndexComputeStart`/`jGlobalIndexComputeStart`/
    `kGlobalIndexComputeStart` (1) indicate the global indices of the
    first element of a tile to be computed in the original computations.

    The `iGlobalUB`/`jGlobalUB`/`kGlobalUB` (2) indicate the global upper bounds
    in the original computations.

    We provide 3 buffers for matrix multiply: A, B, and C. For each buffer,
    we indicate the global indices pointing the beginning of the buffer:
    `aGlobalIndexMemStart`, `bGlobalIndexMemStart`, and `cGlobalIndexMemStart` (3).
    If no buffers are used, i.e. the computation starts directly in the
    original memory, the global index is 0. If a buffer for AA is used to
    put data into it starting at indices `[i1, k1]`, where `i1` & `k1` are the
    global indices in the original computations, then `aGlobalIndexMemStart0`
    and `aGlobalIndexMemStart1` are `i1` & `k1`, respectively.

    If the A, B, or C buffers are larger than the actual data tile they
    contain (see `copy_to_tile_buffer`), then the actual tile size must be
    given using an optional attribute: `aTileSize`, `bTileSize`, or `cTileSize` (4).
    These optional tile size have a rank of 2, and their values must be
    equal or smaller than their corresponding buffer memrefs.

    If the computation are further tiled with respect to the size of the
    buffers A, B, or C, then the actual computation tile is given by
    the optional tile attribute `computeTileSize` (5). Its rank is 3, for the
    I, J, and K dimension. The actual A, B, and C buffer tile size
    (possibly specified by the optional parameters) must be a multiple of
    the I, J, and K `computeTileSizes`, in their respective
    dimensions (A: `[IxK]`, B: `[KxJ]`, C: `[IxJ]`).

    Note that the buffers A, B, and C can be of higher dimensionality than
    the traditional 2D mentioned up to now, because of broadcasting rules.
    At this time, we only support broadcast of arrays having ranks of 2 or
    more. Because of the broadcast rules, the higher dimensions have a
    constant index during one matrix multiply. These fixed indices are
    given as prefix dimensions in the starting indices for AA, BB, and CC
    as described above. E.g. if AA has a rank of 3, and BB has a rank of 2,
    the starting indices for AA are `[d, i1, k1]` where `i1` and `k1` are as
    above, and d is index pointing to the current instance of the `IxK`
    AA matrix to be computed. B start indices would be unchanged at `[k1, j1]`.

    Simdize is used to state if simdization is requested.
    Unrolling is used to unroll and jam loops as warranted.

    Below is an example calculating a matrix multiply with pre-zeroed
    C matrix with the sizes below.

```
    %A: memref<40x60xf32>, %B: memref<60x80xf32>, %C: memref<40x80xf32>

    // 3 tiled loops.
    %ii, %jj, %kk = krnl.define_loops 3
    %ib, %il = krnl.block %ii 10 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    %jb, %jl = krnl.block %jj 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    %kb, %kl = krnl.block %kk 10 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    // 3 subtiles.
    %ilb, %ill = krnl.block %il 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    %jlb, %jll = krnl.block %jl 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    %klb, %kll = krnl.block %kl 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    // Permute.
    krnl.permute(%ib, %ilb, %ill, %jb, %jlb, %jll, %kb, %klb, %kll)
        [0, 3, 6, 1, 4, 7, 2, 5, 8] :
        !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop,
        !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
    // Outer 2 for i, j.
    krnl.iterate(%ib, %jb) with (%ii -> %i = 0 to 40,
                                 %jj -> %j = 0 to 80,
                                 %kk -> %k = 0 to 60) {
        %i1, %j1 = krnl.get_induction_var_value(%ib, %jb) :
          (!krnl.loop,!krnl.loop) -> (index, index)
        // Fill C buffer.
        %Cbuff = alloca(): memref<10x8xf32>  // n x m_simd
        krnl.copy_to_tile_buffer %Cbuff, %C[%i1, %j1], %f0 :
          memref<10x8xf32>, memref<40x80xf32>
        // Outer 1 for k.
        krnl.iterate(%kb) with () {
            %k1 = krnl.get_induction_var_value(%kb) : (!krnl.loop) -> (index)
            // Fill A and B buffer
            %Abuff = alloca(): memref<10x10xf32> // i x k
            %Bbuff = alloca(): memref<10x8xf32>  // k x j_simd
            krnl.copy_to_tile_buffer %Abuff, %A[%i1, %k1], %f0 :
              memref<10x10xf32>, memref<40x60xf32>
            krnl.copy_to_tile_buffer %Bbuff, %B[%k1, %j1], %f0 :
              memref<10x8xf32>, memref<60x80xf32>

            // Inner iterations for subtiles.
            krnl.iterate(%ilb, %jlb, %klb) with () {
                %i2, %j2, %k2 = krnl.get_induction_var_value(%ilb, %jlb, %klb) :
                (!krnl.loop,!krnl.loop,!krnl.loop) -> (index,index,index)

                krnl.matmul %Abuff[%i1, %k1], %Bbuff[%k1, %j1], %Cbuff[%i1, %j1],
                    (%ill, %jll, %kll), (%i2, %j2, %k2), (%c40, %c80, %c60)
                    { computeTileSize=[5,4,5], simdize=false, unroll=false } :
                    memref<10x10xf32>, memref<10x8xf32>, memref<10x8xf32>,
                    (!krnl.loop,!krnl.loop,!krnl.loop)
            }
        }
        // Copy back the data into C.
        krnl.copy_from_tile_buffer %Cbuff, %C[%i1, %j1] :
          memref<10x8xf32>, memref<40x80xf32>
    }
```

    Note that code is simdized along the J dim (last dim of B and C matrices).
    For simd to be enabled, the simdized flag must be set to true, and the
    following condition must be true:
    1) The vector length is the second entry of (i, j, k) compute tile size.
       The vector length must be a compile time constant.

Traits: `AttrSizedOperandSegments`, `MemRefsNormalizable`

Interfaces: `SpecializedKernelOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>computeTileSize</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>aTileSize</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>bTileSize</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>cTileSize</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>simdize</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>unroll</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>overcompute</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | memref of any type values
| `aGlobalIndexMemStart` | variadic of index
| `B` | memref of any type values
| `bGlobalIndexMemStart` | variadic of index
| `C` | memref of any type values
| `cGlobalIndexMemStart` | variadic of index
| `loops` | variadic of any type
| `iGlobalIndexComputeStart` | index
| `jGlobalIndexComputeStart` | index
| `kGlobalIndexComputeStart` | index
| `iGlobalUB` | index
| `jGlobalUB` | index
| `kGlobalUB` | index

### `krnl.memcpy` (KrnlMemcpyOp)

_Krnl memcpy operation_

Copy `num_elems` elements from `src` to `dest` MemRef.

Starting positions for `src` and `dest` are defined by `src_offset` and
`dest_offset`, respectively.

It is the users' responsibility to make sure there is no out-of-bound read/write.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `dest` | memref of any type values
| `src` | memref of any type values
| `num_elems` | 64-bit signless integer
| `dest_offset` | index
| `src_offset` | index

### `krnl.memset` (KrnlMemsetOp)

_Set buffer to a given value._


Syntax:

```
operation ::= `krnl.memset` $dest `,` $value attr-dict `:` type($dest)
```

Krnl operation that sets a buffer to a given value.
In case that the buffer is a MemRef with affine_map, `delayed` indicates
whether we set values along original or extended iteration space.

For example, given
- an affine_map `#tile = affine_map < (i)->(i floordiv 4, i mod 4) >`, and
- a buffer of type `memref<5xf32, #tile>`

Original iteration space is along the first axis that has 5 elements.

If we do normalization, the memref becomes `memref<2x4xf32>`. Now we have
an extended iteration space along two axes of sizes 2 and 4, respectively.
This extended iteration space has 8 elements in total.

If `delayed = false`, the original iteration space is used to set values.
In the above example, only 5 out of 8 elementes will be set to the given value.

If `delayed = true`, the extended iteration space is used to set values.
In the above example, all 8 elements will be set to the given value.


Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>delayed</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `dest` | memref of any type values
| `value` | any type

### `krnl.movable` (KrnlMovableOp)

_Krnl movable operation_


Syntax:

```
operation ::= `krnl.movable` $region attr-dict
```

Encapsulates a list of operations, which should be moved under a newly lowered
affine for operation eventually, but cannot presently because the destination
affine for operation is not materialized yet.

This operation is automatically generated by the lowering of Krnl to affine dialect
to assist with maintaining the relative positioning of loop and inner-loop statements.
This construct is particularly helpful, for example, for lowering statements that
are nested imperfectly between an "eager" and a "lazy" loop.

Traits: `SingleBlockImplicitTerminator<KrnlTerminatorOp>`, `SingleBlock`

### `krnl.noValue` (KrnlNoneOp)

_An operation representing the absence of a value._

This operation can be used to represent the absence of a value. It is
typically used as an argument to operators that have optional parameters,
and converted into nullptr while krnl to llvm lowering.
Typically it is used for optional arguments used in KrnlCallop.

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>value</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `none_val` | none type

### `krnl.parallel_clause` (KrnlParallelClauseOp)

_Attach OpenMP clauses to an index varialbe_


Syntax:

```
operation ::= `krnl.parallel_clause` `(` $parallel_loop_index `)` (`,` `num_threads` `(` $num_threads^ `)`)?
              attr-dict `:` type($parallel_loop_index)
```

Attach OpenMP clauses to an index variable. That index variable
is used to uniquely associate a parallel loop with its clauses.

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>proc_bind</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `parallel_loop_index` | index
| `num_threads` | 32-bit signless integer

### `krnl.parallel` (KrnlParallelOp)

_Mark Krnl loops as parallel loops_


Syntax:

```
operation ::= `krnl.parallel` `(` $loops `)` (`,` `num_threads` `(` $num_threads^ `)`)? attr-dict `:` type($loops)
```

Parallelize the specified loops. When multiple loop specifiers are passed
as parameters, there loops can be parallelized as a collapsed loop.
krnl.parallel should be placed as the last operator before krnl.iterate,
Since we do not want to parallelize the loop until we interpret krnl.block,
krnl.permute and krnl.unroll.

Optionally, a value may specifiy the number of threads requested for the
parallel loop. A proc_bind string may also be specified; valid values are
"primary", "close", or "spread". Default values are used when not specified.

```
krnl.parallel (%i0, %i1) : !Krnl.loop, !Krnl.loop
```

Traits: `AttrSizedOperandSegments`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>proc_bind</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `loops` | variadic of any type
| `num_threads` | 32-bit signless integer

### `krnl.permute` (KrnlPermuteOp)

_Krnl permute operation_


Syntax:

```
operation ::= `krnl.permute` `(` $loops `)` $map attr-dict `:` type($loops)
```

Permute a set of affine for loops using a specified permutation map.
The permutation map `map` should be constructed in such way that the
for loop referred to by the i-th operand to permute operation is sent
to the `map[i]`-th position.

For example, the following krnl dialect IR:
```
%ii, %jj, %kk = krnl.define_loops 3
krnl.permute(%ii, %jj, %kk) [1, 2, 0] : !krnl.loop, !krnl.loop, !krnl.loop
krnl.iterate (%ii, %jj, %kk) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20, %kk -> %k = 0 to 30) {}
```
will be lowered to:
```
// Referenced by %kk
affine.for %arg0 = 0 to 30 {
  // Referenced by %ii
  affine.for %arg1 = 0 to 10 {
    // Referenced by %jj
    affine.for %arg2 = 0 to 20 {
    }
  }
}
```

For a more complicated example, we demonstrate 3-D tiling using krnl.block in
conjunction with krnl.permute:
```
%ii, %jj, %kk = krnl.define_loops 3
// Blocking each loop by a factor of 4.
%ib, %il = krnl.block %ii 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
%jb, %jl = krnl.block %jj 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
%kb, %kl = krnl.block %kk 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// Move iteration over tile coordinates to be the outer loops and iterateion over
// the inter-tile elements to be the inner loops.
krnl.permute(%ib, %il, %jb, %jl, %kb, %kl) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
krnl.iterate(%ib, %il, %jb, %jl, %kb, %kl) with (%ii -> %i = 0 to 1024, %jj -> %j = 0 to 2048, %kk -> %k = 0 to 4096)  {
}
```

The above IR gets lowered to:
```
affine.for %arg0 = 0 to 1024 step 4 {
  affine.for %arg1 = 0 to 2048 step 4 {
    affine.for %arg2 = 0 to 4096 step 4 {
      affine.for %arg3 = #map0(%arg0) to #map1(%arg0) {
        affine.for %arg4 = #map0(%arg1) to #map1(%arg1) {
          affine.for %arg5 = #map0(%arg2) to #map1(%arg2) {
          }
        }
      }
    }
  }
}
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>map</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `loops` | variadic of any type

### `krnl.prefetch` (KrnlPrefetchOp)

_A Krnl operation to compute a linear offset index from a N-D index._

Given a MemRef and an N-D index (id_1, id_2, ..., id_n), prefetch the memory
location pointed by this memory reference.

Traits: `MemRefsNormalizable`

Interfaces: `AffineMapAccessInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>isWrite</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>localityHint</code></td><td>::mlir::IntegerAttr</td><td>32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3</td></tr>
<tr><td><code>isDataCache</code></td><td>::mlir::BoolAttr</td><td>bool attribute</td></tr>
<tr><td><code>map</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `memref` | memref of any type values
| `indices` | variadic of index

### `krnl.print` (KrnlPrintOp)

_Print a value._

This operation can be used to print the input value. The user needs to provide a
format string (à la printf) to specify how to print the input value.
If the input value is not specified the operator will print the format string.

Traits: `MemRefsNormalizable`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>format</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | any type

### `krnl.print_tensor` (KrnlPrintTensorOp)

_Print a tensor._

This operation can be used to generate a call to a runtime function which prints a tensor.
At the beginning of the msg string, user can add formatting instructions. The flags are:

*  `%s`: detailed signature (including shape, type, offsets),
*  `%t`: compact type (ala MLIR: `32x16xfloat`),
*  `%d`: data values.

When no formatting is provided, `%s%d` is used (detailed signature and data) by default.
Print operation ends with a newline, except when only requesting a compact types (`%t`).

Traits: `MemRefsNormalizable`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>msg</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | memref of any type values

### `krnl.random_normal` (KrnlRandomNormalOp)

_Generate a random normal tensor._

Operation that generates a random normally distributed tensor.

Traits: `MemRefsNormalizable`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `output` | memref of any type values
| `numberOfValues` | index
| `mean` | floating-point
| `scale` | floating-point
| `seed` | floating-point

### `krnl.region` (KrnlRegionOp)

_Affine boundary for krnl loops_

This Op has a region with AffineScope trait and is used to limit the
scope of `affine.for`. The loop inside `krnl.region` can be affined if
its boundary is defined at the level of `krnl.region`. The `krnl.region` does
not guarantee or require the loops inside it to be affine.
With `krnl.region`, a krnl loop may not be  affine if its boundary symbol
is not defined inside a enclosing region without AffineScope trait.
In MLIR, FuncOp has the AffineScope trait.
The `krnl.region` will be removed after affine.for is lowered.
ToFix: current `krnl.region` does not have input and output. You cannot
create a new memref inside the region and use it outside of the region.

Traits: `AffineScope`, `NoTerminator`, `SingleBlock`

### `krnl.round_even` (KrnlRoundEvenOp)

_Krnl round to nearest even operation_

Krnl round to nearest even operation.  Accept scalar or vector float values.
Vector must be 1D of a size that is a multiple of the hardware vector size.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `in` | floating-point-like

#### Results:

| Result | Description |
| :----: | ----------- |
| `out` | floating-point-like

### `krnl.seqalloc` (KrnlSeqAllocOp)

_Krnl create a sequence_

This op allocates a memref for a new sequence according to the input Type and length.
The output is tagged with Allocate side effect, and a deallocation is defined for
sequence. This deallocation will free all the elements in the sequence as well as
the sequence itself.

Traits: `MemRefsNormalizable`

Interfaces: `AllocationOpInterface`, `MemoryEffectOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `length` | variadic of index

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | memref of any type values

### `krnl.seqdealloc` (KrnlSeqDeallocOp)

_Krnl dealloc a sequence_

This op deallocate the elements in the sequence and the sequence itself
with memref::dealloc. This Op is a deep dealloc for sequence type.

Traits: `MemRefsNormalizable`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input_sequence` | memref of any type values

### `krnl.seqextract` (KrnlSeqExtractOp)

_Krnl load from a sequence_

This op loads an element from the input sequence 'seq' at position 'index'.
The loaded element is copied and then return.
The position value is guaranteed to be positive. Negative position allowed
by ONNX Op definition should be handled before lowered to KrnlSeqExtract.

Attribute 'copy' provides an optimization for copying.
When the attribute 'copy' is 1 (default value): the extracted element is copied and then return.
When the attribute 'copy' is 0: the extracted element is directly returned
without copy.

The returned element is marked as allocated by this Op with the bufferation
interface so that deallocation can be generated correctly through the
Bufferization::Deallocation pass.

Traits: `MemRefsNormalizable`

Interfaces: `AllocationOpInterface`, `MemoryEffectOpInterface`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>copy</code></td><td>::mlir::IntegerAttr</td><td>1-bit unsigned integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `seq` | memref of any type values
| `index` | index

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | any type

### `krnl.seqstore` (KrnlSeqStoreOp)

_Krnl store into a seq_

This op is similar to KrnSeqInsertOp but assumes that the input seq has
the space for the new element and
only need to copy the element and store it into the sequence.
There is no return of a new seq, different from KrnlSeqInsertOp.
This Op is introduced to accumulate a dynamic tensor in a LoopOp with
statically known iteration count.

Traits: `MemRefsNormalizable`

Interfaces: `MemoryEffectOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | any type
| `seq` | memref of any type values
| `index` | index

### `krnl.specialized_kernel` (KrnlSpecializedKernel)

_Krnl specialized kernel op_


Syntax:

```
operation ::= `krnl.specialized_kernel` `(` $loops `)` attr-dict `:` type($loops)
```

Krnl operation to convert.

Interfaces: `SpecializedKernelOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `loops` | variadic of any type

### `krnl.store` (KrnlStoreOp)

_A Krnl operation to store data to the memref._


Syntax:

```
operation ::= `krnl.store` $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
```

The `krnl.store` stores a value to a memref location given by indices. The
value stored should have the same type as the elemental type of the memref.
The number of arguments provided within brackets need to match the rank of
the memref.

Traits: `MemRefsNormalizable`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `value` | any type
| `memref` | memref of any type values
| `indices` | variadic of index

### `krnl.strlen` (KrnlStrlenOp)

_Compute the length of a string._

Krnl operation that computes the length of a string.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `str` | string type

#### Results:

| Result | Description |
| :----: | ----------- |
| `res` | 64-bit signless integer

### `krnl.strncmp` (KrnlStrncmpOp)

_Perform string comparison up to N bytes._

Krnl operation that performs a string comparison up to N bytes.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `str1` | string type
| `str2` | string type
| `len` | 64-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
| `res` | 32-bit signless integer

### `krnl.tan` (KrnlTanOp)

_Krnl tan scalar operation_

Krnl tan scalar operation.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `in` | floating-point

#### Results:

| Result | Description |
| :----: | ----------- |
| `out` | floating-point

### `krnl.terminate` (KrnlTerminatorOp)

_Krnl terminator operation_

Krnl terminator is a special terminator operation for blocks inside krnl
iterate operations. It unconditionally transmits the control flow to the
successor of the operation enclosing the region.

This operation does _not_ have a custom syntax. However, krnl control
operations omit the terminator in their custom syntax for brevity.

Traits: `ReturnLike`, `Terminator`

Interfaces: `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

### `krnl.unroll` (KrnlUnrollOp)

_Krnl unroll operation_


Syntax:

```
operation ::= `krnl.unroll` $loop attr-dict `:` type($loop)
```

Fully unroll the specified loops.
```
krnl.unroll %i
```
unrolls the loop referred to by %i fully.

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `loop` | any type

### `krnl.vector_type_cast` (KrnlVectorTypeCastOp)

_Vector type cast operation_


Syntax:

```
operation ::= `krnl.vector_type_cast` $source attr-dict `:` type($source) `to` type($result)
```

The "vector_type_cast" operation converts a memref from an non-vector
element type to another memref of a vector elemental type while not changing
the source memref's element type. The last dimension size of the source
dimension is divided (floor division) by the vector size to obtain the
corresponding dimension for target memref type.

```
%MV = vector_type_cast %M : memref<64x16xf32> to memref<64x2xvector<8xf32>>
%AV = vector_type_cast %A : memref<?x?xf32> to memref<?x?xvector<8xf32>>
```

Traits: `AlwaysSpeculatableImplTrait`, `MemRefsNormalizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ViewLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `source` | memref of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | memref of any type values

### `krnl.yield` (KrnlYieldOp)

_Yield values to parent operation_


Syntax:

```
operation ::= `krnl.yield` attr-dict ($operands^ `:` type($operands))?
```

The `krnl.yield` yields zero or more SSA values from an krnl.iterate op region and
terminates the region. The semantics of how the values yielded are used
is defined by the parent operation.
If `krnl.yield` has any operands, the operands must match the parent
operation's results.
If the parent operation defines no values, then the `krnl.yield` may be
left out in the custom syntax and the builders will insert one implicitly.
Otherwise, it has to be present in the syntax to indicate which values are
yielded.

Traits: `AlwaysSpeculatableImplTrait`, `MemRefsNormalizable`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operands` | variadic of any type



[./Dialects/onnx.md]:

<!-- Autogenerated by mlir-tblgen; don't manually edit -->
### `onnx.Abs` (ONNXAbsOp)

_ONNX Abs operation_

Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where absolute value, y = abs(x), is applied to
the tensor elementwise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Acos` (ONNXAcosOp)

_ONNX Acos operation_

Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Acosh` (ONNXAcoshOp)

_ONNX Acosh operation_

Calculates the hyperbolic arccosine of the given input tensor element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Adagrad` (ONNXAdagradOp)

_ONNX Adagrad operation_

Compute one iteration of ADAGRAD, a stochastic gradient based optimization
    algorithm. This operator can conduct the optimization of multiple tensor variables.

    Let's define the behavior of this operator. As you can imagine, ADAGRAD requires
    some parameters:

     - The initial learning-rate \"R\".
     - The update count \"T\". That is, the number of training iterations conducted.
     - A L2-norm regularization coefficient \"norm_coefficient\".
     - A learning-rate decay factor \"decay_factor\".
     - A small constant \"epsilon\" to avoid dividing-by-zero.

    At each ADAGRAD iteration, the optimized tensors are moved along a direction
    computed based on their estimated gradient and accumulated squared gradient. Assume
    that only a single tensor \"X\" is updated by this operator. We need the value of \"X\",
    its gradient \"G\", and its accumulated squared gradient \"H\". Therefore, variables in
    this operator's input list are sequentially \"R\", \"T\", \"X\", \"G\", and \"H\". Other
    parameters are given as attributes because they are usually constants. Also, the
    corresponding output tensors are the new value of \"X\" (called \"X_new\"), and then
    the new accumulated squared gradient (called \"H_new\"). Those outputs are computed
    from the given inputs following the pseudo code below.

    Let \"+\", \"-\", \"*\", and \"/\" are all element-wise arithmetic operations with
    numpy-style broadcasting support. The pseudo code to compute those outputs is:

      // Compute a scalar learning-rate factor. At the first update of X, T is generally
      // 0 (0-based update index) or 1 (1-based update index).
      r = R / (1 + T * decay_factor);

      // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
      G_regularized = norm_coefficient * X + G;

      // Compute new accumulated squared gradient.
      H_new = H + G_regularized * G_regularized;

      // Compute the adaptive part of per-coordinate learning rate. Note that Sqrt(...)
      // computes element-wise square-root.
      H_adaptive = Sqrt(H_new) + epsilon

      // Compute the new value of \"X\".
      X_new = X - r * G_regularized / H_adaptive;

    If one assign this operators to optimize multiple inputs, for example, \"X_1\" and \"X_2\", the same
    pseudo code may be extended to handle all tensors jointly. More specifically, we can view \"X\" as a
    concatenation of \"X_1\" and \"X_2\" (of course, their gradient and accumulate gradient should
    be concatenated too) and then just reuse the entire pseudo code.

    Note that ADAGRAD was first proposed in http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.
    In that reference paper, this operator is a special case of the Figure 1's composite mirror
    descent update.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>decay_factor</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>epsilon</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>norm_coefficient</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `R` | tensor of 32-bit float values or tensor of 64-bit float values
| `T` | tensor of 64-bit signless integer values
| `inputs` | variadic of tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `outputs` | variadic of tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Adam` (ONNXAdamOp)

_ONNX Adam operation_

Compute one iteration of Adam, a stochastic gradient based optimization
    algorithm. This operator can conduct the optimization of multiple tensor variables.

    Let's define the behavior of this operator. First of all, Adam requires
    some parameters:

     - The learning-rate \"R\".
     - The update count \"T\". That is, the number of training iterations conducted.
     - A L2-norm regularization coefficient \"norm_coefficient\".
     - A small constant \"epsilon\" to avoid dividing-by-zero.
     - Two coefficients, \"alpha\" and \"beta\".

    At each Adam iteration, the optimized tensors are moved along a direction
    computed based on their exponentially-averaged historical gradient and
    exponentially-averaged historical squared gradient. Assume that only a tensor
    \"X\" is being optimized. The rest of required information is

     - the value of \"X\",
     - \"X\"'s gradient (denoted by \"G\"),
     - \"X\"'s exponentially-averaged historical gradient (denoted by \"V\"), and
     - \"X\"'s exponentially-averaged historical squared gradient (denoted by \"H\").

    Some of those parameters are passed into this operator as input tensors and others
    are stored as this operator's attributes. Specifically, this operator's input tensor
    list is [\"R\", \"T\", \"X\", \"G\", \"V\", \"H\"]. That is, \"R\" is the first input, \"T\" is
    the second input, and so on. Other parameters are given as attributes because they
    are constants. Moreover, the corresponding output tensors are

     - the new value of \"X\" (called \"X_new\"),
     - the new exponentially-averaged historical gradient (denoted by \"V_new\"), and
     - the new exponentially-averaged historical squared gradient (denoted by \"H_new\").

    Those outputs are computed following the pseudo code below.

    Let \"+\", \"-\", \"*\", and \"/\" are all element-wise arithmetic operations with
    numpy-style broadcasting support. The pseudo code to compute those outputs is:

      // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
      G_regularized = norm_coefficient * X + G

      // Update exponentially-averaged historical gradient.
      V_new = alpha * V + (1 - alpha) * G_regularized

      // Update exponentially-averaged historical squared gradient.
      H_new = beta * H + (1 - beta) * G_regularized * G_regularized

      // Compute the element-wise square-root of H_new. V_new will be element-wisely
      // divided by H_sqrt for a better update direction.
      H_sqrt = Sqrt(H_new) + epsilon

      // Compute learning-rate. Note that \"alpha**T\"/\"beta**T\" is alpha's/beta's T-th power.
      R_adjusted = T > 0 ? R * Sqrt(1 - beta**T) / (1 - alpha**T) : R

      // Compute new value of \"X\".
      X_new = X - R_adjusted * V_new / H_sqrt

      // Post-update regularization.
      X_final = (1 - norm_coefficient_post) * X_new

    If there are multiple inputs to be optimized, the pseudo code will be applied
    independently to each of them.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>alpha</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>beta</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>epsilon</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>norm_coefficient</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>norm_coefficient_post</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `R` | tensor of 32-bit float values or tensor of 64-bit float values
| `T` | tensor of 64-bit signless integer values
| `inputs` | variadic of tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `outputs` | variadic of tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Add` (ONNXAddOp)

_ONNX Add operation_

Performs element-wise binary addition (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `B` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.And` (ONNXAndOp)

_ONNX And operation_

Returns the tensor resulted from performing the `and` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 1-bit signless integer values
| `B` | tensor of 1-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 1-bit signless integer values

### `onnx.ArgMax` (ONNXArgMaxOp)

_ONNX ArgMax operation_

Computes the indices of the max elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the max
is selected if the max appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>select_last_index</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 64-bit signless integer values

### `onnx.ArgMin` (ONNXArgMinOp)

_ONNX ArgMin operation_

Computes the indices of the min elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the min
is selected if the min appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>select_last_index</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 64-bit signless integer values

### `onnx.ArrayFeatureExtractor` (ONNXArrayFeatureExtractorOp)

_ONNX ArrayFeatureExtractor operation_

Select elements of the input tensor based on the indices passed.<br>
    The indices are applied to the last axes of the tensor.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values or tensor of string type values
| `Y` | tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Z` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values or tensor of string type values

### `onnx.Asin` (ONNXAsinOp)

_ONNX Asin operation_

Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Asinh` (ONNXAsinhOp)

_ONNX Asinh operation_

Calculates the hyperbolic arcsine of the given input tensor element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Atan` (ONNXAtanOp)

_ONNX Atan operation_

Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Atanh` (ONNXAtanhOp)

_ONNX Atanh operation_

Calculates the hyperbolic arctangent of the given input tensor element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.AveragePool` (ONNXAveragePoolOp)

_ONNX AveragePool operation_

AveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape is calculated differently
 depending on whether explicit padding is used, where pads is employed, or auto padding is used, where auto_pad is utilized.
 With explicit padding (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled. `pad_shape[i]` is the sum of pads along axis `i`. Sliding windows that would start in the right padded region are ignored.

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 or when ceil_mode is disabled (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):
 ```
 VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
 ```
 The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).


Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>auto_pad</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>ceil_mode</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>count_include_pad</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>dilations</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>pads</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.BatchNormalizationInferenceMode` (ONNXBatchNormalizationInferenceModeOp)

_ONNX BatchNormalization operation in test mode_

Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)"

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C*D1*D2 ..*Dn) before a BatchNormalization Op.
This operator has **optional** inputs/outputs. See [the doc](IR.md)
for more details about the representation of optional arguments.
An empty string may be used in the place of an actual argument's name to
indicate a missing argument. Trailing optional arguments (those not followed
by an argument that is present) may also be simply omitted.

This operation is not part of the standard and was added to assist onnx-mlir.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>epsilon</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>momentum</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of any type values or tensor of any type values
| `scale` | memref of any type values or tensor of any type values
| `B` | memref of any type values or tensor of any type values
| `mean` | memref of any type values or tensor of any type values
| `var` | memref of any type values or tensor of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `o_Y` | memref of any type values or tensor of any type values

### `onnx.BatchNormalization` (ONNXBatchNormalizationOp)

_ONNX BatchNormalization operation_

Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
There are five required inputs 'X', 'scale', 'B', 'input_mean' and
'input_var'.
Note that 'input_mean' and 'input_var' are expected to be the estimated
statistics in inference mode (training_mode=False, default),
and the running statistics in training mode (training_mode=True).
There are multiple cases for the number of outputs, which we list below:

* Output case #1: Y, running_mean, running_var (training_mode=True)
* Output case #2: Y (training_mode=False)

When training_mode=False, extra outputs are invalid.
The outputs are updated as follows when training_mode=True:
```
running_mean = input_mean * momentum + current_mean * (1 - momentum)
running_var = input_var * momentum + current_var * (1 - momentum)

Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B
```
where:
```
current_mean = ReduceMean(X, axis=all_except_channel_index)
current_var =  ReduceVar(X, axis=all_except_channel_index)
```
Notice that `ReduceVar` refers to the population variance, and it equals to
`sum(sqrd(x_i - x_avg)) / N`
where `N` is the population size (this formula does not use sample size `N - 1`).

The computation of ReduceMean and ReduceVar uses float to avoid overflow for float16 inputs.

When training_mode=False:
```
Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
```

For previous (depreciated) non-spatial cases, implementors are suggested
to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>epsilon</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>momentum</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>training_mode</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `scale` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `B` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `input_mean` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `input_var` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `running_mean` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or none type
| `running_var` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or none type

### `onnx.Bernoulli` (ONNXBernoulliOp)

_ONNX Bernoulli operation_

Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor
containing probabilities p (a value in the range [0,1]) to be used for drawing the binary random number,
where an output of 1 is produced with probability p and an output of 0 is produced with probability (1-p).

This operator is non-deterministic and may not produce the same values in different
implementations (even if a seed is specified).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dtype</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>seed</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 1-bit signless integer values

### `onnx.Binarizer` (ONNXBinarizerOp)

_ONNX Binarizer operation_

Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>threshold</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values

### `onnx.BitShift` (ONNXBitShiftOp)

_ONNX BitShift operation_

Bitwise shift operator performs element-wise operation. For each input element, if the
attribute \"direction\" is \"RIGHT\", this operator moves its binary representation toward
the right side so that the input value is effectively decreased. If the attribute \"direction\"
is \"LEFT\", bits of binary representation moves toward the left side, which results the
increase of its actual value. The input X is the tensor to be shifted and another input
Y specifies the amounts of shifting. For example, if \"direction\" is \"Right\", X is [1, 4],
and S is [1, 1], the corresponding output Z would be [0, 2]. If \"direction\" is \"LEFT\" with
X=[1, 2] and S=[1, 2], the corresponding output Y would be [2, 8].

Because this operator supports Numpy-style broadcasting, X's and Y's shapes are
not necessarily identical.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>direction</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Z` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values

### `onnx.BitwiseAnd` (ONNXBitwiseAndOp)

_ONNX BitwiseAnd operation_

Returns the tensor resulting from performing the bitwise `and` operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `B` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

### `onnx.BitwiseNot` (ONNXBitwiseNotOp)

_ONNX BitwiseNot operation_

Returns the bitwise not of the input tensor element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

### `onnx.BitwiseOr` (ONNXBitwiseOrOp)

_ONNX BitwiseOr operation_

Returns the tensor resulting from performing the bitwise `or` operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `B` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

### `onnx.BitwiseXor` (ONNXBitwiseXorOp)

_ONNX BitwiseXor operation_

Returns the tensor resulting from performing the bitwise `xor` operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `B` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

### `onnx.BlackmanWindow` (ONNXBlackmanWindowOp)

_ONNX BlackmanWindow operation_

Generates a Blackman window as described in the paper https://ieeexplore.ieee.org/document/1455106.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>output_datatype</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>periodic</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `size` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.CastLike` (ONNXCastLikeOp)

_ONNX CastLike operation_

The operator casts the elements of a given input tensor (the first input) to
the same data type as the elements of the second input tensor.
See documentation of the Cast operator for further details.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>saturate</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 1-bit signless integer values or tensor of string type values or tensor of bfloat16 type values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values
| `target_type` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 1-bit signless integer values or tensor of string type values or tensor of bfloat16 type values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 1-bit signless integer values or tensor of string type values or tensor of bfloat16 type values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values

### `onnx.CastMap` (ONNXCastMapOp)

_ONNX CastMap operation_

Converts a map to a tensor.<br>The map key must be an int64 and the values will be ordered
    in ascending order based on this key.<br>The operator supports dense packing or sparse packing.
    If using sparse packing, the key cannot exceed the max_map-1 value.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>cast_to</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>map_form</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>max_map</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tuple with any combination of 64-bit signless integer or string type values or tuple with any combination of 64-bit signless integer or 32-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of string type values or tensor of 32-bit float values or tensor of 64-bit signless integer values

### `onnx.Cast` (ONNXCastOp)

_ONNX Cast operation_

The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.

Casting from string tensor in plain (e.g., \"3.14\" and \"1000\") and scientific numeric representations
(e.g., \"1e-5\" and \"1E8\") to float types is supported. For example, converting string \"100.5\" to an integer may
yield result 100. There are some string literals reserved for special floating-point values;
\"+INF\" (and \"INF\"), \"-INF\", and \"NaN\" are positive infinity, negative infinity, and not-a-number, respectively.
Any string which can exactly match \"+INF\" in a case-insensitive way would be mapped to positive infinite. Similarly,
this case-insensitive rule is applied to \"INF\" and \"NaN\". When casting from numeric tensors
to string tensors, plain floating-point representation (such as \"314.15926\") would be used.
Converting non-numerical-literal string such as \"Hello World!\" is an undefined behavior. Cases
of converting string representing floating-point arithmetic value, such as \"2.718\", to INT is an undefined behavior.

Conversion from a numerical type to any numerical type is always allowed.
User must be aware of precision loss and value change caused by range difference between two types.
For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.

In more detail, the conversion among numerical types should follow these rules
if the destination type is not a float 8 type.

* Casting from floating point to:
  * floating point: +/- infinity if OOR (out of range).
  * fixed point: undefined if OOR.
  * bool: +/- 0.0 to False; all else to True.
* Casting from fixed point to:
  * floating point: +/- infinity if OOR. (+ infinity in the case of uint)
  * fixed point: when OOR, discard higher bits and reinterpret (with respect to two's complement representation for
    signed types). For example, 200 (int16) -> -56 (int8).
  * bool: zero to False; nonzero to True.
* Casting from bool to:
  * floating point: `{1.0, 0.0}`.
  * fixed point: `{1, 0}`.
  * bool: no change.

Float 8 type were introduced to speed up the training of
deep models. By default the conversion of a float *x* obeys
to the following rules. `[x]` means the value rounded to
the target mantissa width.

| x | E4M3FN | E4M3FNUZ | E5M2 | E5M2FNUZ |
|------|----|----|----|----|
| 0 | 0 | 0 | 0 | 0 |
|-0 | -0 | 0 | -0 | 0 |
| NaN | NaN | NaN | NaN | NaN |
| +/- Inf | +/- FLT_MAX | NaN | FLT_MAX | NaN |
| [x] > FLT_MAX | FLT_MAX | FLT_MAX | FLT_MAX | FLT_MAX |
| [x] < -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX | -FLT_MAX |
| else | RNE | RNE | RNE | RNE |

The behavior changes if the parameter 'saturate' is set to False.
The rules then become:

| x | E4M3FN | E4M3FNUZ | E5M2 | E5M2FNUZ |
|------|----|----|----|----|
| 0 | 0 | 0 | 0 | 0 |
|-0 | -0 | 0 | -0 | 0 |
| NaN | NaN | NaN | NaN | NaN |
| +/- Inf | NaN | NaN | +/- Inf | NaN |
| [x] > FLT_MAX | NaN | NaN | Inf | NaN |
| [x] < -FLT_MAX | NaN | NaN | -Inf | NaN |
| else | RNE | RNE | RNE | RNE |

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ResultTypeInferenceOpInterface`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>saturate</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>to</code></td><td>::mlir::TypeAttr</td><td>any type attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 1-bit signless integer values or tensor of string type values or tensor of bfloat16 type values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 1-bit signless integer values or tensor of string type values or tensor of bfloat16 type values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values

### `onnx.CategoryMapper` (ONNXCategoryMapperOp)

_ONNX CategoryMapper operation_

Converts strings to integers and vice versa.<br>
    Two sequences of equal length are used to map between integers and strings,
    with strings and integers at the same index detailing the mapping.<br>
    Each operator converts either integers to strings or strings to integers, depending
    on which default value attribute is provided. Only one default value attribute
    should be defined.<br>
    If the string default value is set, it will convert integers to strings.
    If the int default value is set, it will convert strings to integers.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>cats_int64s</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>cats_strings</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>default_int64</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>default_string</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of string type values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of string type values or tensor of 64-bit signless integer values

### `onnx.Ceil` (ONNXCeilOp)

_ONNX Ceil operation_

Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Celu` (ONNXCeluOp)

_ONNX Celu operation_

Continuously Differentiable Exponential Linear Units:
Perform the linear unit element-wise on the input tensor X
using formula:

```
max(0,x) + min(0,alpha*(exp(x/alpha)-1))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>alpha</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 32-bit float values

### `onnx.CenterCropPad` (ONNXCenterCropPadOp)

_ONNX CenterCropPad operation_

Center crop or pad an input to given dimensions.

The crop/pad dimensions can be specified for a subset of the `axes`. Non-specified dimensions will not be
cropped or padded.

If the input dimensions are bigger than the crop shape, a centered cropping window is extracted from the input.
If the input dimensions are smaller than the crop shape, the input is padded on each side equally,
so that the input is centered in the output.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input_data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `shape` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output_data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Clip` (ONNXClipOp)

_ONNX Clip operation_

Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `min` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or none type
| `max` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ClipV11` (ONNXClipV11Op)

_ONNX Clip operation_

Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `min` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `max` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.ClipV12` (ONNXClipV12Op)

_ONNX Clip operation_

Clip operator limits the given input within an interval. The interval is
specified by the inputs 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max(), respectively.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `min` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `max` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.ClipV6` (ONNXClipV6Op)

_ONNX Clip operation_

Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>max</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>min</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Col2Im` (ONNXCol2ImOp)

_ONNX Col2Im operation_

The operator rearranges column blocks back into a multidimensional image

Col2Im behaves similarly to PyTorch's fold https://pytorch.org/docs/stable/generated/torch.nn.Fold.html,
but it only supports *batched* multi-dimensional image tensors.
Another implementation in Python with N-dimension support can be found at https://github.com/f-dangel/unfoldNd/.

NOTE:
  Although specifying image_shape looks redundant because it could be calculated from
  convolution formulas, it is required as input for more advanced scenarios as explained
  at PyTorch's implementation (https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Col2Im.cpp#L10)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dilations</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>pads</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `image_shape` | tensor of 64-bit signless integer values
| `block_shape` | tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Compress` (ONNXCompressOp)

_ONNX Compress operation_

Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
    In case axis is not provided, input is flattened before elements are selected.
    Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html


Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `condition` | tensor of 1-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.ConcatFromSequence` (ONNXConcatFromSequenceOp)

_ONNX ConcatFromSequence operation_

Concatenate a sequence of tensors into a single tensor.
All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
By default 'new_axis' is 0, the behavior is similar to numpy.concatenate.
When 'new_axis' is 1, the behavior is similar to numpy.stack.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>new_axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input_sequence` | SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values

#### Results:

| Result | Description |
| :----: | ----------- |
| `concat_result` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Concat` (ONNXConcatOp)

_ONNX Concat operation_

Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `inputs` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `concat_result` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.ConcatShapeTranspose` (ONNXConcatShapeTransposeOp)

_ONNX merged operation_

Merge the following sequence of ops into one op
v1 = onnx.concat
v2 = onnx.shape(v1)
v3 = onnx.transpose(v1)

This operation is not part of the standard and was added to assist onnx-mlir.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>end</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>start</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>perm</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `inputs` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `shape` | tensor of 64-bit signless integer values
| `transposed` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.ConstantOfShape` (ONNXConstantOfShapeOp)

_ONNX ConstantOfShape operation_

Generate a tensor with given value and shape.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ResultTypeInferenceOpInterface`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>value</code></td><td>::mlir::Attribute</td><td>any attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 1-bit signless integer values or tensor of bfloat16 type values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values

### `onnx.Constant` (ONNXConstantOp)

_ONNX Constant operation_

This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ResultTypeInferenceOpInterface`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>sparse_value</code></td><td>::mlir::Attribute</td><td>any attribute</td></tr>
<tr><td><code>value</code></td><td>::mlir::Attribute</td><td>any attribute</td></tr>
<tr><td><code>value_float</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>value_floats</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>value_int</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>value_ints</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>value_string</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>value_strings</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values

### `onnx.ConvInteger` (ONNXConvIntegerOp)

_ONNX ConvInteger operation_

The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>auto_pad</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>dilations</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>group</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>pads</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `x` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values
| `w` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values
| `x_zero_point` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values or none type
| `w_zero_point` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `y` | tensor of 32-bit signless integer values

### `onnx.Conv` (ONNXConvOp)

_ONNX Conv operation_

The convolution operator consumes an input tensor and a filter, and
computes the output.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>auto_pad</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>dilations</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>group</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>pads</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `W` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `B` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.ConvTranspose` (ONNXConvTransposeOp)

_ONNX ConvTranspose operation_

The convolution transpose operator consumes an input tensor and a filter,
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).



Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>auto_pad</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>dilations</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>group</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>output_padding</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>output_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>pads</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `W` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `B` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Cos` (ONNXCosOp)

_ONNX Cos operation_

Calculates the cosine of the given input tensor, element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Cosh` (ONNXCoshOp)

_ONNX Cosh operation_

Calculates the hyperbolic cosine of the given input tensor element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.CumSum` (ONNXCumSumOp)

_ONNX CumSum operation_

Performs cumulative sum of the input elements along the given axis.
By default, it will do the sum inclusively meaning the first element is copied as is.
Through an `exclusive` attribute, this behavior can change to exclude the first element.
It can also perform summation in the opposite direction of the axis. For that, set `reverse` attribute to 1.

Example:
```
input_x = [1, 2, 3]
axis=0
output = [1, 3, 6]
exclusive=1
output = [0, 1, 3]
exclusive=0
reverse=1
output = [6, 5, 3]
exclusive=1
reverse=1
output = [5, 3, 0]
```


Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>exclusive</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>reverse</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `x` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `axis` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `y` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Custom` (ONNXCustomOp)

_ONNX Custom operation_

CustomOp is not an Op defined in onnx standard and was added to support
extention of Op that can be transformed or finally call a user-defined
external function."

It allows for calling a user-defined operation, with a single required
attribute being a string that names the operation. Other inputs are passed
to the user operation.

The number of inputs and outputs can vary.

NoneType is allowed for both input and output, as the CustomOp may require
a fixed number of inputs/outputs for the external function call.

In addition to the values passed to the user-defined operation, certain
attributes are introduced to facilitate the analysis and transformation of
CustomOp.

Since the compiler does not define the semantics of CustomOp, onnx-mlir
cannot infer the shape of its output. Consequently, specific attributes are
introduced to specify how shape inference should be performed on a CustomOp.
These attributes are:
  'inputs_for_infer':
       Optional. The index of inputs used for shape inference.
       The value of index should be [0, the number of inputs).
       If not specified, all the inputs of the CustomOp will be used for
       shape inference.
  'shape_infer_pattern':
       Optional. Specify how to propagate the shape info from the inputs
       (may be limited by inputs_for_infer) to output. Current supported
       patterns are `SameAs`, `MDBroadcast`.
  'output_element_type':
       Optional. The element type for the output tensor. If not specified,
       follow the shape infer pattern behavior. Usually the element type of
       the first input is used.
Each instance of CustomOp can have its own attributes for shape inference,
allowing for customization. However, CustomOps with the same function_name
typically behave similarly in terms of shape inference, and therefore have
the same attributes.

The existing shape inference patterns for ONNX ops are reused for CustomOp,
with the polymorphism in shape inference based on its attribute values.
Due to the current implementation for ONNX Ops, a CustomOp with specified
shape inference attributes supports only a single output, rather than
variadic outputs.

When attributes for shape inference are not provided, the shape inference
for CustomOp will simply pass through.

All of these additional attributes are optional, designed to be less
intrusive. The .mlir file can remain the same when a new attribute is
added.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>function_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>output_element_type</code></td><td>::mlir::TypeAttr</td><td>any type attribute</td></tr>
<tr><td><code>shape_infer_pattern</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>inputs_for_infer</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `inputs` | variadic of tensor of any type values or memref of any type values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `outputs` | variadic of tensor of any type values or memref of any type values or none type

### `onnx.DFT` (ONNXDFTOp)

_ONNX DFT operation_

Computes the discrete Fourier Transform (DFT) of the input.

Assuming the input has shape `[M, N]`, where `N` is the dimension over which the
DFT is computed and `M` denotes the conceptual \"all other dimensions,\"
the DFT `y[m, k]` of shape `[M, N]` is defined as

$$y[m, k] = \sum_{n=0}^{N-1} e^{-2 \pi j \frac{k n}{N} } x[m, n] ,$$

and the inverse transform is defined as

$$x[m, n] = \frac{1}{N} \sum_{k=0}^{N-1} e^{2 \pi j \frac{k n}{N} } y[m, k] ,$$

where $j$ is the imaginary unit.

The actual shape of the output is specified in the \"output\" section.

Reference: https://docs.scipy.org/doc/scipy/tutorial/fft.html

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>inverse</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>onesided</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `dft_length` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or none type
| `axis` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.DFTV17` (ONNXDFTV17Op)

_ONNX DFT operation_

Computes the discrete Fourier transform of input.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>inverse</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>onesided</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `dft_length` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.DeformConv` (ONNXDeformConvOp)

_ONNX DeformConv operation_

Performs deformable convolution as described in https://arxiv.org/abs/1703.06211 and https://arxiv.org/abs/1811.11168.
This operator specification supports the general N-D case. Note that most common use cases have 2D or 3D data.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dilations</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>group</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>offset_group</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>pads</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `W` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `offset` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `B` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `mask` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.DepthToSpace` (ONNXDepthToSpaceOp)

_ONNX DepthToSpace operation_

DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions. By default, `mode` = `DCR`.
In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
following order: depth, column, and then row. The output y is computed from the input x as below:

```
b, c, h, w = x.shape
tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
```

In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the
following order: column, row, and the depth. The output y is computed from the input x as below:

```
b, c, h, w = x.shape
tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>blocksize</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.DequantizeLinear` (ONNXDequantizeLinearOp)

_ONNX DequantizeLinear operation_

The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is `y = (x - x_zero_point) * x_scale`. `x_scale` and `x_zero_point` must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
`x_zero_point` and `x` must have same type. `x` and `y` must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).
`zero-point` is usually not used in the case of float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz quantization,
but the dequantization formula remains the same for consistency and 'x_scale' still determines the output type.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `x` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values
| `x_scale` | tensor of 32-bit float values or tensor of 16-bit float values or tensor of bfloat16 type values
| `x_zero_point` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `y` | tensor of 32-bit float values or tensor of 16-bit float values or tensor of bfloat16 type values

### `onnx.Det` (ONNXDetOp)

_ONNX Det operation_

Det calculates determinant of a square matrix or batches of square matrices.
Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
and the inner-most 2 dimensions form square matrices.
The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.DictVectorizer` (ONNXDictVectorizerOp)

_ONNX DictVectorizer operation_

Uses an index mapping to convert a dictionary to an array.<br>
    Given a dictionary, each key is looked up in the vocabulary attribute corresponding to
    the key type. The index into the vocabulary array at which the key is found is then
    used to index the output 1-D tensor 'Y' and insert into it the value found in the dictionary 'X'.<br>
    The key type of the input map must correspond to the element type of the defined vocabulary attribute.
    Therefore, the output array will be equal in length to the index mapping vector parameter.
    All keys in the input dictionary must be present in the index mapping vector.
    For each item in the input dictionary, insert its value in the output array.
    Any keys not present in the input dictionary, will be zero in the output array.<br>
    For example: if the ``string_vocabulary`` parameter is set to ``[\"a\", \"c\", \"b\", \"z\"]``,
    then an input of ``{\"a\": 4, \"c\": 8}`` will produce an output of ``[4, 8, 0, 0]``.


Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>int64_vocabulary</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>string_vocabulary</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tuple with any combination of string type or 64-bit signless integer values or tuple with any combination of 64-bit signless integer or string type values or tuple with any combination of 64-bit signless integer or 32-bit float values or tuple with any combination of 64-bit signless integer or 64-bit float values or tuple with any combination of string type or 32-bit float values or tuple with any combination of string type or 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 64-bit signless integer values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values

### `onnx.DimGroup` (ONNXDimGroupOp)

_ONNX dimension group operation._

This operation is to link a compile-time unknown dimension of a Tensor
to a group id. Two dimensions that have the same group id are expected
to be equal at runtime.

```
"onnx.DimGroup"(%tensor) {axis = 0 : si64, group_id = 1: si64} : (tensor<?x3x5xf32>) -> ()
```

`axis` identifies the dimension position in the tensor.

`group_id` identifies the group id of the dimension. It is non-negative.
Value -1 for `group_id` means the dimension does not belong to any group.

This operation is currently used in the pass `--onnx-dim-analysis`
for testing the unknown dimension analysis class.

This operation is not part of the standard and was added to assist onnx-mlir.

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>group_id</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Dim` (ONNXDimOp)

_ONNX dimensions operation._

This operation is to obtain the dimension of a Tensor;

```
"onnx.Dim"(%tensor) {axis = 0 : si64} : (tensor<?x3x5xf32>) -> tensor<1xi64>
```

The axis identifies the dimension within the shape which is going to be obtained.

This operation is not part of the standard and was added to assist onnx-mlir.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `dim` | tensor of 64-bit signless integer values

### `onnx.Div` (ONNXDivOp)

_ONNX Div operation_

Performs element-wise binary division (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `B` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Dropout` (ONNXDropoutOp)

_ONNX Dropout operation_

Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
the user can simply not pass `training_mode` input or set it to false.
```
output = scale * data * mask,
```
where
```
scale = 1. / (1. - ratio).
```
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>seed</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values
| `ratio` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or none type
| `training_mode` | tensor of 1-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values
| `mask` | tensor of 1-bit signless integer values or none type

### `onnx.DynamicQuantizeLinear` (ONNXDynamicQuantizeLinearOp)

_ONNX DynamicQuantizeLinear operation_

A Function to fuse calculation for Scale, Zero Point and FP32->8Bit conversion of FP32 Input data.
Outputs Scale, ZeroPoint and Quantized Input for a given FP32 Input.
Scale is calculated as:
```
y_scale = (maximum(0, max(x)) - minimum(0, min(x))) / (qmax - qmin)
```

* where qmax and qmin are max and min values for quantization range i.e. [0, 255] in case of uint8
* data range is adjusted to include 0.

Zero point is calculated as:
```
intermediate_zero_point = qmin - min(x)/y_scale
y_zero_point = cast(round(saturate(itermediate_zero_point)))
```

* where qmax and qmin are max and min values for quantization range .i.e [0, 255] in case of uint8
* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.

Data quantization formula is:
```
y = saturate (round (x / y_scale) + y_zero_point)
```

* for saturation, it saturates to [0, 255] if it's uint8, or [-127, 127] if it's int8. Right now only uint8 is supported.
* rounding to nearest ties to even.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `x` | tensor of 32-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `y` | tensor of 8-bit unsigned integer values
| `y_scale` | tensor of 32-bit float values
| `y_zero_point` | tensor of 8-bit unsigned integer values

### `onnx.Einsum` (ONNXEinsumOp)

_ONNX Einsum operation_

An einsum of the form `term1, term2 -> output-term` produces an output tensor using the following equation

```
output[output-term] = reduce-sum( input1[term1] * input2[term2] )
```

where the reduce-sum performs a summation over all the indices occurring in the input terms (term1, term2)
that do not occur in the output-term.

The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation
convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to
an operand tensor, and the characters within the terms correspond to operands dimensions.

This sequence may be followed by \"->\" to separate the left and right hand side of the equation.
If the equation contains \"->\" followed by the right-hand side, the explicit (not classical) form of the Einstein
summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases,
output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the
equation.

When a dimension character is repeated in the left-hand side, it represents summation along the dimension.

The equation may contain ellipsis (\"...\") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions.
Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions.
The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the
beginning of the output. The equation string may contain space (U+0020) character.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>equation</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `Inputs` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Elu` (ONNXEluOp)

_ONNX Elu operation_

Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.


Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>alpha</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.EntryPoint` (ONNXEntryPointOp)

_Indicate ONNX entry point_

The "onnx.EntryPoint" function indicates the main entry point of ONNX model.

This operation is not part of the standard and was added to assist onnx-mlir.

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>func</code></td><td>::mlir::SymbolRefAttr</td><td>symbol reference attribute</td></tr>
</table>

### `onnx.Equal` (ONNXEqualOp)

_ONNX Equal operation_

Returns the tensor resulted from performing the `equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 1-bit signless integer values or tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or tensor of string type values
| `B` | tensor of 1-bit signless integer values or tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or tensor of string type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 1-bit signless integer values

### `onnx.Erf` (ONNXErfOp)

_ONNX Erf operation_

Computes the error function of the given input tensor element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Exp` (ONNXExpOp)

_ONNX Exp operation_

Calculates the exponential of the given input tensor, element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Expand` (ONNXExpandOp)

_ONNX Expand operation_

Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimensions must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `shape` | tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.EyeLike` (ONNXEyeLikeOp)

_ONNX EyeLike operation_

Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
same as the input tensor. The data type can be specified by the 'dtype' argument. If
'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dtype</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>k</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 1-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 1-bit signless integer values

### `onnx.FeatureVectorizer` (ONNXFeatureVectorizerOp)

_ONNX FeatureVectorizer operation_

Concatenates input tensors into one continuous output.<br>
    All input shapes are 2-D and are concatenated along the second dimension. 1-D tensors are treated as [1,C].
    Inputs are copied to the output maintaining the order of the input arguments.<br>
    All inputs must be integers or floats, while the output will be all floating point values.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>inputdimensions</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | variadic of tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 32-bit float values

### `onnx.Flatten` (ONNXFlattenOp)

_ONNX Flatten operation_

Flattens the input tensor into a 2D matrix. If input tensor has shape
(d_0, d_1, ... d_n) then the output will have shape
(d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values

### `onnx.Floor` (ONNXFloorOp)

_ONNX Floor operation_

Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.GRU` (ONNXGRUOp)

_ONNX GRU operation_

Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

* `X` - input tensor
* `z` - update gate
* `r` - reset gate
* `h` - hidden gate
* `t` - time step (t-1 means previous time step)
* `W[zrh]` - W parameter weight matrix for update, reset, and hidden gates
* `R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates
* `Wb[zrh]` - W bias vectors for update, reset, and hidden gates
* `Rb[zrh]` - R bias vectors for update, reset, and hidden gates
* `WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates
* `RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates
* `WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates
* `RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE:
  Below are optional

* Affine(x)              - alpha * x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha * Tanh(beta * x)
* HardSigmoid(x)         - min(max(alpha * x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha * (e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

* zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
* rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
* ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
* ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
* Ht = (1 - zt) (.) ht + zt (.) Ht-1
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>activation_alpha</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>activation_beta</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>activations</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>clip</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>direction</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>hidden_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>layout</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>linear_before_reset</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `W` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `R` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `B` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `sequence_lens` | tensor of 32-bit signless integer values or none type
| `initial_h` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `Y_h` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

### `onnx.GatherElements` (ONNXGatherElementsOp)

_ONNX GatherElements operation_

GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
and an optional attribute `axis` that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). It is an indexing operation
that produces its output by indexing into the input data tensor at index
positions determined by elements of the `indices` tensor.
Its output shape is the same as the shape of `indices` and consists of one value
(gathered from the `data`) for each element in `indices`.

For instance, in the 3-D case (r = 3), the output produced is determined
by the following equations:
```
out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,
```

This operator is also the inverse of ScatterElements. It is similar to Torch's gather operation.

Example 1:
```
data = [
    [1, 2],
    [3, 4],
]
indices = [
    [0, 0],
    [1, 0],
]
axis = 1
output = [
    [1, 1],
    [4, 3],
]
```
Example 2:
```
data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
indices = [
    [1, 2, 0],
    [2, 0, 0],
]
axis = 0
output = [
    [4, 8, 3],
    [7, 2, 3],
]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `indices` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.GatherND` (ONNXGatherNDOp)

_ONNX GatherND operation_

Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.

`indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
where each element defines a slice of `data`

`batch_dims` (denoted as `b`) is an integer indicating the number of batch dimensions, i.e the leading `b` number of dimensions of
`data` tensor and `indices` are representing the batches, and the gather starts from the `b+1` dimension.

Some salient points about the inputs' rank and shape:

1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`

2) The first `b` dimensions of the shape of `indices` tensor and `data` tensor must be equal.

3) b < min(q, r) is to be honored.

4) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r-b` (inclusive)

5) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
   It is an error if any of the index values are out of bounds.

The output is computed as follows:

The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.

1) If `indices_shape[-1] > r-b` => error condition

2) If `indices_shape[-1] == r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensors
   containing 1-D tensors of dimension `r-b`, where `N` is an integer equals to the product of 1 and all the elements in the batch dimensions
   of the indices_shape. Let us think of each such `r-b` ranked tensor as `indices_slice`. Each *scalar value* corresponding to `data[0:b-1,indices_slice]`
   is filled into the corresponding location of the `(q-b-1)`-dimensional tensor to form the `output` tensor (Example 1 below)

3) If `indices_shape[-1] < r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensor
   containing 1-D tensors of dimension `< r-b`. Let us think of each such tensors as `indices_slice`. Each *tensor slice* corresponding
   to `data[0:b-1, indices_slice , :]` is filled into the corresponding location of the `(q-b-1)`-dimensional tensor
   to form the `output` tensor (Examples 2, 3, 4 and 5 below)

This operator is the inverse of `ScatterND`.

**Example 1**

```
batch_dims = 0
data    = [[0,1],[2,3]]   # data_shape    = [2, 2]
indices = [[0,0],[1,1]]   # indices_shape = [2, 2]
output  = [0,3]           # output_shape  = [2]
```

**Example 2**

```
batch_dims = 0
data    = [[0,1],[2,3]]  # data_shape    = [2, 2]
indices = [[1],[0]]      # indices_shape = [2, 1]
output  = [[2,3],[0,1]]  # output_shape  = [2, 2]
```

**Example 3**

```
batch_dims = 0
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]
output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
```

**Example 4**

```
batch_dims = 0
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]
output  = [[[2,3]],[[4,5]]]             # output_shape  = [2, 1, 2]
```

**Example 5**

```
batch_dims = 1
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[1],[0]]                     # indices_shape = [2, 1]
output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>batch_dims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `indices` | tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Gather` (ONNXGatherOp)

_ONNX Gather operation_

Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).

If `axis = 0`, let `k = indices[i_{0}, ..., i_{q-1\}\]`
then `output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2\}\] = input[k , j_{0}, ..., j_{r-2\}\]`:

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]
indices = [
    [0, 1],
    [1, 2],
]
output = [
    [
        [1.0, 1.2],
        [2.3, 3.4],
    ],
    [
        [2.3, 3.4],
        [4.5, 5.7],
    ],
]
```

If `axis = 1`, let `k = indices[i_{0}, ..., i_{q-1\}\]`
then `output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2\}\] = input[j_{0}, k, j_{1}, ..., j_{r-2\}\]`:

```
data = [
    [1.0, 1.2, 1.9],
    [2.3, 3.4, 3.9],
    [4.5, 5.7, 5.9],
]
indices = [
    [0, 2],
]
axis = 1,
output = [
        [[1.0, 1.9]],
        [[2.3, 3.9]],
        [[4.5, 5.9]],
]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `indices` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Gelu` (ONNXGeluOp)

_ONNX Gelu operation_

Gelu takes one input data (Tensor<T>) and produces one
output data (Tensor<T>) where the gaussian error linear units function,
$y = 0.5 * x * (1 + erf(x/sqrt(2)))$ is applied to the tensor elementwise.
If the attribute \"approximate\" is set to \"tanh\", the function estimation,
$y = 0.5 * x * (1 + Tanh(sqrt(2/\pi) * (x + 0.044715 * x^3)))$ is used and applied
to the tensor elementwise.


Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>approximate</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Gemm` (ONNXGemmOp)

_ONNX Gemm operation_

General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

* A' = transpose(A) if transA else A
* B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>alpha</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>beta</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>transA</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>transB</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values
| `B` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values
| `C` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values

### `onnx.GlobalAveragePool` (ONNXGlobalAveragePoolOp)

_ONNX GlobalAveragePool operation_

GlobalAveragePool consumes an input tensor X and applies average pooling across
 the values in the same channel. This is equivalent to AveragePool with kernel size
 equal to the spatial dimension of input tensor.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.GlobalLpPool` (ONNXGlobalLpPoolOp)

_ONNX GlobalLpPool operation_

GlobalLpPool consumes an input tensor X and applies lp pool pooling across
 the values in the same channel. This is equivalent to LpPool with kernel size
 equal to the spatial dimension of input tensor.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>p</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.GlobalMaxPool` (ONNXGlobalMaxPoolOp)

_ONNX GlobalMaxPool operation_

GlobalMaxPool consumes an input tensor X and applies max pooling across
 the values in the same channel. This is equivalent to MaxPool with kernel size
 equal to the spatial dimension of input tensor.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Gradient` (ONNXGradientOp)

_ONNX Gradient operation_

Gradient operator computes the partial derivatives of a specific tensor w.r.t.
some other tensors. This operator is widely used in gradient-based training
algorithms. To illustrate its use, let's consider a computation graph,

```
X -----.
       |
       v
W --> Conv --> H --> Gemm --> Y
                      ^
                      |
                      Z
```

, where W and Z are trainable tensors. Note that operators' attributes are
omitted for the sake of simplicity. Let dY/dW (dY/dZ) be the gradient of
Y with respect to W (Z). The user can compute gradient by inserting Gradient
operator to form another graph shown below.

```
W --> Conv --> H --> Gemm --> Y
|      ^              ^
|      |              |
|      X              Z
|      |              |
|      |   .----------'
|      |   |  (W/Z/X is the 1st/2nd/3rd input of Gradient as shown in
|      |   |   \"xs\" followed by \"zs\")
|      v   v
'---> Gradient(xs=[\"W\", \"Z\"], zs=[\"X\"], y=\"Y\")
       |   |
       |   '-----------------------------------> dY/dW (1st output of Gradient)
       |
       '---------------------------------------> dY/dZ (2nd output of Gradient)
```

By definition, the tensor \"y\" is a function of independent variables in \"xs\"
and \"zs\". Since we only compute the gradient of \"y\" w.r.t. the differentiable
variables in \"xs\", this Gradient only outputs dY/dW and dY/dZ. Note that \"H\"
cannot appear in \"xs\" and \"zs\". The reason is that \"H\" can be determined by
tensors \"W\" and \"X\" and therefore \"H\" is not an independent variable.

All outputs are optional. If needed, for example, user can assign an empty
string to the 1st output name of that Gradient to skip the generation of dY/dW.
Note that the concept of optional outputs can also be found in ONNX's RNN, GRU,
and LSTM.

Gradient operator can compute derivative against intermediate tensors. For
example, the gradient of Y with respect to H can be done via

```
W --> Conv --> H --> Gemm --> Y
       ^       |      ^
       |       |      |
       X       |      Z
       .-------'      |
       |   .----------'
       |   | (H/Z is the 1st/2nd input of Gradient as shown in \"xs\")
       v   v
      Gradient(xs=[\"H\", \"Z\"], y=\"Y\")
       |   |
       |   '-----------------------------------> dY/dH (1st output of Gradient)
       |
       '---------------------------------------> dY/dZ (2nd output of Gradient)
```

It is possible to represent high-order differentiation using Gradient operators.
For example, given the following linear model:

```
W --> Gemm --> Y --> Loss --> O
       ^              ^
       |              |
       X              L
```

To compute the 2nd order derivative of O with respect to W (denoted by
d^2O/dW^2), one can do

```
W --> Gemm --> Y --> Loss --> O
|      ^              ^
|      |              |
|      X .------------L
|      | |            |
|      | |            v
+------+-+> Gradient(xs=[\"X\", \"W\"], zs=[\"L\"], y=\"O\") ---> dO/dX (1st output of Gradient)
|      | |    |
|      | |    '---> dO/dW (2nd output of Gradient)
|      v v
'---> Gradient(xs=[\"X\", \"W\"], zs=[\"L\"], y=\"dO/dW\") ---> d(dO/dW)dX (1st output of
       |                                                  Gradient)
       |
       |
       '---> d^2O/dW^2 (2nd output of Gradient)
```

The tensors named in attributes \"xs\", \"zs\", and \"y\" define the differentiated
computation graph, and the inputs to Gradient node define the values at
which the gradient is computed. We can feed different tensors to the identified
graph. For example, one can compute the gradient of Y with respect to H at
a specific value of H, H_1, by providing that value as an input to the Gradient
node.

```
W --> Conv --> H --> Gemm --> Y
       ^              ^
       |              |
       X              Z

          Z_1 (2nd input of Gradient)
           |
           v
H_1 --> Gradient(xs=[\"H\", \"Z\"], y=\"Y\") ---> dY/dH when H = H_1 and Y = Y_1.
           |
           '------------------------------> dY/dZ (2nd output of Gradient)
```

When the inputs of Gradient are the tensors named in \"xs\" and \"zs\", the
computation can be optimized. More specifically, intermediate variables in
forward pass can be reused if the gradient is computed via reverse-mode
auto-differentiation.


Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>xs</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>y</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>zs</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `Inputs` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Outputs` | variadic of tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Greater` (ONNXGreaterOp)

_ONNX Greater operation_

Returns the tensor resulted from performing the `greater` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `B` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 1-bit signless integer values

### `onnx.GreaterOrEqual` (ONNXGreaterOrEqualOp)

_ONNX GreaterOrEqual operation_

Returns the tensor resulted from performing the `greater_equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `B` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 1-bit signless integer values

### `onnx.GridSample` (ONNXGridSampleOp)

_ONNX GridSample operation_

Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from the `grid`.
For spatial input `X` with shape (N, C, H, W), the `grid` will have shape (N, H_out, W_out, 2),
the output `Y` will have shape (N, C, H_out, W_out). For volumetric input `X` with shape (N, C, D, H, W),
the `grid` will have shape (N, D_out, H_out, W_out, 3), the output `Y` will have shape (N, C, D_out, H_out, W_out).
More generally, for an input `X` of rank r+2 with shape (N, C, d1, d2, ..., dr),
the `grid` will have shape (N, D1_out, D2_out, ..., Dr_out, r), the output `Y` will have shape (N, C, D1_out, D2_out, ..., Dr_out).

The tensor `X` contains values at centers of square pixels (voxels, etc) locations such as (n, c, d1_in, d2_in, ..., dr_in).
The (n, d1_out, d2_out, ..., dr_out, :) values from the tensor `grid` are the normalized positions for interpolating the values
at the (n, c, d1_out, d2_out, ..., dr_out) locations from the output tensor `Y` using a specified interpolation method (the mode)
and a padding mode (for `grid` positions falling outside the 2-dimensional image).

For example, the values in `grid[n, h_out, w_out, :]` are size-2 vectors specifying normalized positions in the 2-dimensional space of `X`.
They are used to interpolate output values of `Y[n, c, h_out, w_out]`.

The GridSample operator is often used in doing grid generator and sampler in the
[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).
See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>align_corners</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>padding_mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `grid` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.GridSampleV16` (ONNXGridSampleV16Op)

_ONNX GridSample operation_

Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from `grid`.
Currently, only spatial (4-D) inputs are supported. For input `X` with shape (N, C, H, W) and `grid` with shape (N, H_out, W_out, 2),
the output `Y` will have shape (N, C, H_out, W_out).

The tensor `X` contains values at centers of square pixels in a H by W 2-dimensional image.
The tensor `grid` describes normalized positions where the output `Y` is to be computed
using a specified interpolation method (the mode) and a padding mode (for grid positions falling outside the 2-dimensional image).

Elements in `grid[N, H_out, W_out]` are size-2 vectors specifying positions in the 2-dimensional space of `X`.
They are used to interpolate output values of `Y[N, C, H_out, W_out]`.

The GridSample operator is often used in doing grid generator and sampler in the [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).
See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/master/generated/torch.nn.functional.grid_sample.html#torch-nn-functional-grid-sample).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>align_corners</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>padding_mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `grid` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.GroupNormalization` (ONNXGroupNormalizationOp)

_ONNX GroupNormalization operation_

A GroupNormalization function. Carries out group normalization as described in
the paper https://arxiv.org/abs/1803.08494

This operator transforms input according to
```
y = scale * (x - mean) / sqrt(variance + epsilon) + bias,
```
where the mean and variance are computed per instance per group of channels, and
`scale` and `bias` should be specified for each group of channels. The number of
groups `num_groups` should be divisible by the number of channels so that there are
an equal number of channels per group.

The overall computation has two stages: the first stage normalizes the elements to
have zero mean and unit variance for each instance in each group, and the second
stage scales and shifts the results of the first stage. The floating-point precision
used in the first stage is determined by the `stash_type` attribute. For example,
if `stash_type` is 1, the operator casts all input variables to 32-bit float,
performs the computation, and finally casts the normalized results back to the
original type of `X`. The second stage does not depend on `stash_type`.

When the number of groups is the same as the number of channels, this operator is
equivalent to InstanceNormalization. When there is only one group, this operator
is equivalent to LayerNormalization.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>epsilon</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>num_groups</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>stash_type</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `scale` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `bias` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.GroupNormalizationV18` (ONNXGroupNormalizationV18Op)

_ONNX GroupNormalization operation_

A GroupNormalization function. Carries out group normalization as described in
the paper https://arxiv.org/abs/1803.08494

This operator transforms input according to
```
y = scale * (x - mean) / sqrt(variance + epsilon) + bias,
```
where the mean and variance are computed per instance per group of channels, and
`scale` and `bias` should be specified for each group of channels. The number of
groups `num_groups` should be divisible by the number of channels so that there are
an equal number of channels per group.

When the number of groups is the same as the number of channels, this operator is
equivalent to InstanceNormalization. When there is only one group, this operator
is equivalent to LayerNormalization.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>epsilon</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>num_groups</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `scale` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `bias` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.HammingWindow` (ONNXHammingWindowOp)

_ONNX HammingWindow operation_

Generates a Hamming window as described in the paper https://ieeexplore.ieee.org/document/1455106.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>output_datatype</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>periodic</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `size` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.HannWindow` (ONNXHannWindowOp)

_ONNX HannWindow operation_

Generates a Hann window as described in the paper https://ieeexplore.ieee.org/document/1455106.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>output_datatype</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>periodic</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `size` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.HardSigmoid` (ONNXHardSigmoidOp)

_ONNX HardSigmoid operation_

HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>alpha</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>beta</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.HardSwish` (ONNXHardSwishOp)

_ONNX HardSwish operation_

HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Hardmax` (ONNXHardmaxOp)

_ONNX Hardmax operation_

The operator computes the hardmax values for the given input:

 Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise

The \"axis\" attribute indicates the dimension along which Hardmax
will be performed. The output tensor has the same shape
and contains the Hardmax values of the corresponding input.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Identity` (ONNXIdentityOp)

_ONNX Identity operation_

Identity operator

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values or SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values or OptType of SeqType of tensor of 8-bit unsigned integer values values values or OptType of SeqType of tensor of 16-bit unsigned integer values values values or OptType of SeqType of tensor of 32-bit unsigned integer values values values or OptType of SeqType of tensor of 64-bit unsigned integer values values values or OptType of SeqType of tensor of 8-bit signless integer values values values or OptType of SeqType of tensor of 16-bit signless integer values values values or OptType of SeqType of tensor of 32-bit signless integer values values values or OptType of SeqType of tensor of 64-bit signless integer values values values or OptType of SeqType of tensor of 16-bit float values values values or OptType of SeqType of tensor of 32-bit float values values values or OptType of SeqType of tensor of 64-bit float values values values or OptType of SeqType of tensor of string type values values values or OptType of SeqType of tensor of 1-bit signless integer values values values or OptType of SeqType of tensor of complex type with 32-bit float elements values values values or OptType of SeqType of tensor of complex type with 64-bit float elements values values values or OptType of tensor of 8-bit unsigned integer values values or OptType of tensor of 16-bit unsigned integer values values or OptType of tensor of 32-bit unsigned integer values values or OptType of tensor of 64-bit unsigned integer values values or OptType of tensor of 8-bit signless integer values values or OptType of tensor of 16-bit signless integer values values or OptType of tensor of 32-bit signless integer values values or OptType of tensor of 64-bit signless integer values values or OptType of tensor of 16-bit float values values or OptType of tensor of 32-bit float values values or OptType of tensor of 64-bit float values values or OptType of tensor of string type values values or OptType of tensor of 1-bit signless integer values values or OptType of tensor of complex type with 32-bit float elements values values or OptType of tensor of complex type with 64-bit float elements values values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values or SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values or OptType of SeqType of tensor of 8-bit unsigned integer values values values or OptType of SeqType of tensor of 16-bit unsigned integer values values values or OptType of SeqType of tensor of 32-bit unsigned integer values values values or OptType of SeqType of tensor of 64-bit unsigned integer values values values or OptType of SeqType of tensor of 8-bit signless integer values values values or OptType of SeqType of tensor of 16-bit signless integer values values values or OptType of SeqType of tensor of 32-bit signless integer values values values or OptType of SeqType of tensor of 64-bit signless integer values values values or OptType of SeqType of tensor of 16-bit float values values values or OptType of SeqType of tensor of 32-bit float values values values or OptType of SeqType of tensor of 64-bit float values values values or OptType of SeqType of tensor of string type values values values or OptType of SeqType of tensor of 1-bit signless integer values values values or OptType of SeqType of tensor of complex type with 32-bit float elements values values values or OptType of SeqType of tensor of complex type with 64-bit float elements values values values or OptType of tensor of 8-bit unsigned integer values values or OptType of tensor of 16-bit unsigned integer values values or OptType of tensor of 32-bit unsigned integer values values or OptType of tensor of 64-bit unsigned integer values values or OptType of tensor of 8-bit signless integer values values or OptType of tensor of 16-bit signless integer values values or OptType of tensor of 32-bit signless integer values values or OptType of tensor of 64-bit signless integer values values or OptType of tensor of 16-bit float values values or OptType of tensor of 32-bit float values values or OptType of tensor of 64-bit float values values or OptType of tensor of string type values values or OptType of tensor of 1-bit signless integer values values or OptType of tensor of complex type with 32-bit float elements values values or OptType of tensor of complex type with 64-bit float elements values values

### `onnx.If` (ONNXIfOp)

_ONNX If operation_

If conditional

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasOnnxSubgraphOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ResultTypeInferenceOpInterface`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `cond` | tensor of 1-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `outputs` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values or SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of bfloat16 type values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values or SeqType of tensor of f8E4M3FN type values values or SeqType of tensor of f8E4M3FNUZ type values values or SeqType of tensor of f8E5M2 type values values or SeqType of tensor of f8E5M2FNUZ type values values or SeqType of tensor of 4-bit unsigned integer values values or SeqType of tensor of 4-bit signless integer values values or OptType of SeqType of tensor of 8-bit unsigned integer values values values or OptType of SeqType of tensor of 16-bit unsigned integer values values values or OptType of SeqType of tensor of 32-bit unsigned integer values values values or OptType of SeqType of tensor of 64-bit unsigned integer values values values or OptType of SeqType of tensor of 8-bit signless integer values values values or OptType of SeqType of tensor of 16-bit signless integer values values values or OptType of SeqType of tensor of 32-bit signless integer values values values or OptType of SeqType of tensor of 64-bit signless integer values values values or OptType of SeqType of tensor of bfloat16 type values values values or OptType of SeqType of tensor of 16-bit float values values values or OptType of SeqType of tensor of 32-bit float values values values or OptType of SeqType of tensor of 64-bit float values values values or OptType of SeqType of tensor of string type values values values or OptType of SeqType of tensor of 1-bit signless integer values values values or OptType of SeqType of tensor of complex type with 32-bit float elements values values values or OptType of SeqType of tensor of complex type with 64-bit float elements values values values or OptType of tensor of 8-bit unsigned integer values values or OptType of tensor of 16-bit unsigned integer values values or OptType of tensor of 32-bit unsigned integer values values or OptType of tensor of 64-bit unsigned integer values values or OptType of tensor of 8-bit signless integer values values or OptType of tensor of 16-bit signless integer values values or OptType of tensor of 32-bit signless integer values values or OptType of tensor of 64-bit signless integer values values or OptType of tensor of bfloat16 type values values or OptType of tensor of 16-bit float values values or OptType of tensor of 32-bit float values values or OptType of tensor of 64-bit float values values or OptType of tensor of string type values values or OptType of tensor of 1-bit signless integer values values or OptType of tensor of complex type with 32-bit float elements values values or OptType of tensor of complex type with 64-bit float elements values values or OptType of tensor of f8E4M3FN type values values or OptType of tensor of f8E4M3FNUZ type values values or OptType of tensor of f8E5M2 type values values or OptType of tensor of f8E5M2FNUZ type values values or OptType of tensor of 4-bit unsigned integer values values or OptType of tensor of 4-bit signless integer values values

### `onnx.Imputer` (ONNXImputerOp)

_ONNX Imputer operation_

Replaces inputs that equal one value with another, leaving all other elements alone.<br>
    This operator is typically used to replace missing values in situations where they have a canonical
    representation, such as -1, 0, NaN, or some extreme value.<br>
    One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
    holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
    width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
    which one depends on whether floats or integers are being processed.<br>
    The imputed_value attribute length can be 1 element, or it can have one element per input feature.<br>In other words, if the input tensor has the shape [*,F], then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>imputed_value_floats</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>imputed_value_int64s</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>replaced_value_float</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>replaced_value_int64</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values

### `onnx.InstanceNormalization` (ONNXInstanceNormalizationOp)

_ONNX InstanceNormalization operation_

Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.


Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>epsilon</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `scale` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `B` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.IsInf` (ONNXIsInfOp)

_ONNX IsInf operation_

Map infinity to true and other values to false.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>detect_negative</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>detect_positive</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 1-bit signless integer values

### `onnx.IsNaN` (ONNXIsNaNOp)

_ONNX IsNaN operation_

Returns which elements of the input are NaN.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 1-bit signless integer values

### `onnx.LRN` (ONNXLRNOp)

_ONNX LRN operation_

Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element `X[n, c, d1, ..., dk]` in a tensor
of shape `(N x C x D1 x D2, ..., Dk)`, its region is
`{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}`.

`square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2)`,
where `max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))`.

`Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta`

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>alpha</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>beta</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>bias</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.LSTM` (ONNXLSTMOp)

_ONNX LSTM operation_

Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

* `X` - input tensor
* `i` - input gate
* `o` - output gate
* `f` - forget gate
* `c` - cell gate
* `t` - time step (t-1 means previous time step)
* `W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates
* `R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates
* `Wb[iofc]` - W bias vectors for input, output, forget, and cell gates
* `Rb[iofc]` - R bias vectors for input, output, forget, and cell gates
* `P[iof]`  - P peephole weight vector for input, output, and forget gates
* `WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates
* `RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates
* `WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates
* `RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates
* `PB[iof]`  - P peephole weight vector for backward input, output, and forget gates
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE: Below are optional

* Affine(x)              - alpha*x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha*Tanh(beta*x)
* HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

* it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
* ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
* ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
* Ct = ft (.) Ct-1 + it (.) ct
* ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
* Ht = ot (.) h(Ct)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>activation_alpha</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>activation_beta</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>activations</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>clip</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>direction</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>hidden_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>input_forget</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>layout</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `W` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `R` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `B` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `sequence_lens` | tensor of 32-bit signless integer values or none type
| `initial_h` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `initial_c` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `P` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `Y_h` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `Y_c` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

### `onnx.LabelEncoder` (ONNXLabelEncoderOp)

_ONNX LabelEncoder operation_

Maps each element in the input tensor to another value.<br>
    The mapping is determined by the two parallel attributes, 'keys_*' and
    'values_*' attribute. The i-th value in the specified 'keys_*' attribute
    would be mapped to the i-th value in the specified 'values_*' attribute. It
    implies that input's element type and the element type of the specified
    'keys_*' should be identical while the output type is identical to the
    specified 'values_*' attribute. If an input element can not be found in the
    specified 'keys_*' attribute, the 'default_*' that matches the specified
    'values_*' attribute may be used as its output value.<br>
    Let's consider an example which maps a string tensor to an integer tensor.
    Assume and 'keys_strings' is [\"Amy\", \"Sally\"], 'values_int64s' is [5, 6],
    and 'default_int64' is '-1'.  The input [\"Dori\", \"Amy\", \"Amy\", \"Sally\",
    \"Sally\"] would be mapped to [-1, 5, 5, 6, 6].<br>
    Since this operator is an one-to-one mapping, its input and output shapes
    are the same. Notice that only one of 'keys_*'/'values_*' can be set.<br>
    For key look-up, bit-wise comparison is used so even a float NaN can be
    mapped to a value in 'values_*' attribute.<br>

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>default_float</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>default_int64</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>default_string</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>keys_floats</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>keys_int64s</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>keys_strings</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>values_floats</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>values_int64s</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>values_strings</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of string type values or tensor of 64-bit signless integer values or tensor of 32-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of string type values or tensor of 64-bit signless integer values or tensor of 32-bit float values

### `onnx.LayerNormalization` (ONNXLayerNormalizationOp)

_ONNX LayerNormalization operation_

This is layer normalization defined in ONNX as function.
      The overall computation can be split into two stages.
      The first stage is standardization, which makes the
      normalized elements have zero mean and unit variances.
      The computation required by standardization can be
      described by the following equations.
      ```
      Mean = ReduceMean<axes=normalized_axes>(X)
      D = Sub(X, Mean)
      DD = Mul(D, D)
      Var = ReduceMean<axes=normalized_axes>(DD)
      VarEps = Add(Var, epsilon)
      StdDev = Sqrt(VarEps)
      InvStdDev = Reciprocal(StdDev)
      Normalized = Mul(D, InvStdDev)
      ```
      where `normalized_axes` is `[axis, ..., rank of X - 1]`.
      The variables `Var` and `StdDev` stand for variance and
      standard deviation, respectively. The second output is
      `Mean` and the last one is `InvStdDev`.
      Depending on `stash_type` attribute, the actual computation
      must happen in different floating-point precision.
      For example, if `stash_type` is 1, this operator casts
      all input variables to 32-bit float, perform the computation, and
      finally cast `Normalized` back to the original type of `X`.
      The second stage then scales and shifts the outcome of the
      first stage using
      ```
      NormalizedScaled = Mul(Normalized, Scale)
      Y = Add(NormalizedScaled, B)
      ```
      The second stage doesn't depends on `stash_type`.
      All equations are in [this syntax](https://github.com/onnx/onnx/blob/main/docs/Syntax.md).
      The same variable (i.e., input, output, and attribute) uses
      the same name in the equations above and this operator's definition.
      Let `d[i]` indicate the i-th dimension of `X`.
      If `X`'s shape is `[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]`,
      the shape of `Mean` and `InvStdDev` is `[d[0], ..., d[axis-1], 1, ..., 1]`.
      `Y` and `X` have the same shape. This operator supports unidirectional broadcasting
      (tensors `Scale` and `B` should be unidirectional broadcastable to tensor `X`);
      for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>epsilon</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>stash_type</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `Scale` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `B` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `Mean` | tensor of 32-bit float values or tensor of bfloat16 type values or none type
| `InvStdDev` | tensor of 32-bit float values or tensor of bfloat16 type values or none type

### `onnx.LayoutTransform` (ONNXLayoutTransformOp)

_An operation that transforms data between different layout formats_

An operation that transforms a tensor from a layout to another layout. 
A layout is defined by an attribute, i.e. `target_layout`, which allows this
operation work with an arbitrary layout (e.g. a layout used for accelerators).

`target_layout` is optional. If it is not given, the input tensor will be
transformed to a normal tensor that does not have layout.

If `target_layout` is the same as the input's layout, this operation will
become an no-op by canonicalization. 

The input and output tensors must have the same shape.

This operation is not part of the standard and was added to assist onnx-mlir.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>target_layout</code></td><td>::mlir::Attribute</td><td>layout attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 16-bit float or 32-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float or 32-bit float values

### `onnx.LeakyRelu` (ONNXLeakyReluOp)

_ONNX LeakyRelu operation_

LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`, is applied to the data tensor elementwise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>alpha</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Less` (ONNXLessOp)

_ONNX Less operation_

Returns the tensor resulted from performing the `less` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `B` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 1-bit signless integer values

### `onnx.LessOrEqual` (ONNXLessOrEqualOp)

_ONNX LessOrEqual operation_

Returns the tensor resulted from performing the `less_equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `B` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 1-bit signless integer values

### `onnx.LinearClassifier` (ONNXLinearClassifierOp)

_ONNX LinearClassifier operation_

Linear classifier

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>classlabels_ints</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>classlabels_strings</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>coefficients</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>intercepts</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>multi_class</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>post_transform</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of string type values or tensor of 64-bit signless integer values
| `Z` | tensor of 32-bit float values

### `onnx.LinearRegressor` (ONNXLinearRegressorOp)

_ONNX LinearRegressor operation_

Generalized linear regression evaluation.<br>
    If targets is set to 1 (default) then univariate regression is performed.<br>
    If targets is set to M then M sets of coefficients must be passed in as a sequence
    and M results will be output for each input n in N.<br>
    The coefficients array is of length n, and the coefficients for each target are contiguous.
    Intercepts are optional but if provided must match the number of targets.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>coefficients</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>intercepts</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>post_transform</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>targets</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 32-bit float values

### `onnx.Log` (ONNXLogOp)

_ONNX Log operation_

Calculates the natural log of the given input tensor, element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.LogSoftmax` (ONNXLogSoftmaxOp)

_ONNX LogSoftmax operation_

The operator computes the log of softmax values for the given input:

 LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))

The \"axis\" attribute indicates the dimension along which LogSoftmax
will be performed. The output tensor has the same shape
and contains the LogSoftmax values of the corresponding input.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Loop` (ONNXLoopOp)

_ONNX Loop operation_

Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

Operator inputs defined as (max_trip_count, condition_var).

* input (\"\", \"\"):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

* input (\"\", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

* input (\"\", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

* input (trip_count, \"\") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

* input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }


*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]           // iteration number
      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
    ) {
      %my_local = Add(%a, %b_in)
      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
      return %keepgoing_out, %b_out, %user_defined_val
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      /* initialize loop-carried variables and scan-output variables */
      bool keepgoing_out = keepgoing
      int b_out = b

      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
        /* Implicitly-defined code: bind actual parameter values
           to formal parameter variables of loop-body */
        bool keepgoing_in = keepgoing_out;
        bool b_in = b_out;

        /* User-defined code (loop body) */
        int my_local = a + b_in; // Reading value \"a\" from the enclosing scope is fine
        b_out = a - b_in;
        keepgoing_out = my_local > b_out;
        user_defined_val = b_in + b_in; // b_in and b_out are different variables
        /* End user-defined code */

        /* Implicitly defined-code */
        user_defined_vals[i] = user_defined_val // accumulate scan-output values
      }
      // int t = my_local; // Can't do this. my_local is not accessible here.

      // The values below are bound to the output variables of the loop and therefore accessible
      // b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable \"a\" here) are in scope and can
   be referenced in the inputs of the loop.
2) Any values computed in the loop body that needs to be used in a subsequent
   iteration or after the loop are modelled using a pair of variables in the loop-body,
   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
   These are referred to as loop-carried dependences. The loop operation node
   supplies the input value of the input variable for the first iteration, and
   returns the output value of the output variable produced by the final
   iteration.
3) Scan_output variables are used to implicitly concatenate values computed across
   all the iterations. In the above example, the value of user_defined_val computed
   over all iterations are concatenated and returned as the value of user_defined_vals
   after the loop.
4) Values created in the body cannot be accessed in the enclosing scope,
   except using the mechanism described above.

Note that the semantics of this op support \"diagonal\" or \"wavefront\" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).

The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasOnnxSubgraphOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ResultTypeInferenceOpInterface`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `M` | tensor of 64-bit signless integer values or none type
| `cond` | tensor of 1-bit signless integer values or none type
| `v_initial` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values or SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of bfloat16 type values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values or SeqType of tensor of f8E4M3FN type values values or SeqType of tensor of f8E4M3FNUZ type values values or SeqType of tensor of f8E5M2 type values values or SeqType of tensor of f8E5M2FNUZ type values values or SeqType of tensor of 4-bit unsigned integer values values or SeqType of tensor of 4-bit signless integer values values or OptType of SeqType of tensor of 8-bit unsigned integer values values values or OptType of SeqType of tensor of 16-bit unsigned integer values values values or OptType of SeqType of tensor of 32-bit unsigned integer values values values or OptType of SeqType of tensor of 64-bit unsigned integer values values values or OptType of SeqType of tensor of 8-bit signless integer values values values or OptType of SeqType of tensor of 16-bit signless integer values values values or OptType of SeqType of tensor of 32-bit signless integer values values values or OptType of SeqType of tensor of 64-bit signless integer values values values or OptType of SeqType of tensor of bfloat16 type values values values or OptType of SeqType of tensor of 16-bit float values values values or OptType of SeqType of tensor of 32-bit float values values values or OptType of SeqType of tensor of 64-bit float values values values or OptType of SeqType of tensor of string type values values values or OptType of SeqType of tensor of 1-bit signless integer values values values or OptType of SeqType of tensor of complex type with 32-bit float elements values values values or OptType of SeqType of tensor of complex type with 64-bit float elements values values values or OptType of tensor of 8-bit unsigned integer values values or OptType of tensor of 16-bit unsigned integer values values or OptType of tensor of 32-bit unsigned integer values values or OptType of tensor of 64-bit unsigned integer values values or OptType of tensor of 8-bit signless integer values values or OptType of tensor of 16-bit signless integer values values or OptType of tensor of 32-bit signless integer values values or OptType of tensor of 64-bit signless integer values values or OptType of tensor of bfloat16 type values values or OptType of tensor of 16-bit float values values or OptType of tensor of 32-bit float values values or OptType of tensor of 64-bit float values values or OptType of tensor of string type values values or OptType of tensor of 1-bit signless integer values values or OptType of tensor of complex type with 32-bit float elements values values or OptType of tensor of complex type with 64-bit float elements values values or OptType of tensor of f8E4M3FN type values values or OptType of tensor of f8E4M3FNUZ type values values or OptType of tensor of f8E5M2 type values values or OptType of tensor of f8E5M2FNUZ type values values or OptType of tensor of 4-bit unsigned integer values values or OptType of tensor of 4-bit signless integer values values

#### Results:

| Result | Description |
| :----: | ----------- |
| `v_final_and_scan_outputs` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values or SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of bfloat16 type values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values or SeqType of tensor of f8E4M3FN type values values or SeqType of tensor of f8E4M3FNUZ type values values or SeqType of tensor of f8E5M2 type values values or SeqType of tensor of f8E5M2FNUZ type values values or SeqType of tensor of 4-bit unsigned integer values values or SeqType of tensor of 4-bit signless integer values values or OptType of SeqType of tensor of 8-bit unsigned integer values values values or OptType of SeqType of tensor of 16-bit unsigned integer values values values or OptType of SeqType of tensor of 32-bit unsigned integer values values values or OptType of SeqType of tensor of 64-bit unsigned integer values values values or OptType of SeqType of tensor of 8-bit signless integer values values values or OptType of SeqType of tensor of 16-bit signless integer values values values or OptType of SeqType of tensor of 32-bit signless integer values values values or OptType of SeqType of tensor of 64-bit signless integer values values values or OptType of SeqType of tensor of bfloat16 type values values values or OptType of SeqType of tensor of 16-bit float values values values or OptType of SeqType of tensor of 32-bit float values values values or OptType of SeqType of tensor of 64-bit float values values values or OptType of SeqType of tensor of string type values values values or OptType of SeqType of tensor of 1-bit signless integer values values values or OptType of SeqType of tensor of complex type with 32-bit float elements values values values or OptType of SeqType of tensor of complex type with 64-bit float elements values values values or OptType of tensor of 8-bit unsigned integer values values or OptType of tensor of 16-bit unsigned integer values values or OptType of tensor of 32-bit unsigned integer values values or OptType of tensor of 64-bit unsigned integer values values or OptType of tensor of 8-bit signless integer values values or OptType of tensor of 16-bit signless integer values values or OptType of tensor of 32-bit signless integer values values or OptType of tensor of 64-bit signless integer values values or OptType of tensor of bfloat16 type values values or OptType of tensor of 16-bit float values values or OptType of tensor of 32-bit float values values or OptType of tensor of 64-bit float values values or OptType of tensor of string type values values or OptType of tensor of 1-bit signless integer values values or OptType of tensor of complex type with 32-bit float elements values values or OptType of tensor of complex type with 64-bit float elements values values or OptType of tensor of f8E4M3FN type values values or OptType of tensor of f8E4M3FNUZ type values values or OptType of tensor of f8E5M2 type values values or OptType of tensor of f8E5M2FNUZ type values values or OptType of tensor of 4-bit unsigned integer values values or OptType of tensor of 4-bit signless integer values values

### `onnx.LpNormalization` (ONNXLpNormalizationOp)

_ONNX LpNormalization operation_

Given a matrix, apply Lp-normalization along the provided axis.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>p</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.LpPool` (ONNXLpPoolOp)

_ONNX LpPool operation_

LpPool consumes an input tensor X and applies Lp pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 Lp pooling consisting of computing the Lp norm on all values of a subset
 of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - {kernelSpatialShape}) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled `pad_shape[i]` is the sum of pads along axis `i`.

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - {kernelSpatialShape} + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + {kernelSpatialShape} - input_spatial_shape[i]
 ```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>auto_pad</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>ceil_mode</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>dilations</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>p</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>pads</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.MatMulInteger` (ONNXMatMulIntegerOp)

_ONNX MatMulInteger operation_

Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values
| `B` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values
| `a_zero_point` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values or none type
| `b_zero_point` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 32-bit signless integer values

### `onnx.MatMul` (ONNXMatMulOp)

_ONNX MatMul operation_

Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values
| `B` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values

### `onnx.Max` (ONNXMaxOp)

_ONNX Max operation_

Element-wise max of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data_0` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `max` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.MaxPool` (ONNXMaxPoolOp)

_ONNX MaxPool operation_

MaxPool consumes an input tensor X and applies max pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 max pooling consisting of computing the max on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape is calculated differently
 depending on whether explicit padding is used, where pads is employed, or auto padding is used, where auto_pad is utilized.
 With explicit padding (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled. `pad_shape[i]` is the sum of pads along axis `i`. Sliding windows that would start in the right padded region are ignored.

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 or when ceil_mode is disabled (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):
 ```
 VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
 ```
 The output of each pooling window is maximum number of elements exclude pad. 


Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>auto_pad</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>ceil_mode</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>dilations</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>pads</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>storage_order</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values
| `Indices` | tensor of 64-bit signless integer values or none type

### `onnx.MaxPoolSingleOut` (ONNXMaxPoolSingleOutOp)

_ONNX MaxPool operation with a single output._

ONNX MaxPool operation with a single output.
See ONNXMaxPoolOp for a full description of the MaxPool semantics.

This operation is not part of the standard and was added to assist onnx-mlir.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>auto_pad</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>ceil_mode</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>dilations</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>pads</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>storage_order</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | memref of any type values or tensor of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `o_Y` | memref of any type values or tensor of any type values

### `onnx.MaxRoiPool` (ONNXMaxRoiPoolOp)

_ONNX MaxRoiPool operation_

ROI max pool consumes an input tensor X and region of interests (RoIs) to
 apply max pooling across each RoI, to produce output 4-D tensor of shape
 (num_rois, channels, pooled_shape[0], pooled_shape[1]).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>pooled_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>spatial_scale</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `rois` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.MaxUnpool` (ONNXMaxUnpoolOp)

_ONNX MaxUnpool operation_

MaxUnpool essentially computes the partial inverse of the MaxPool op.
 The input information to this op is typically the output information from a MaxPool op. The first
 input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
 from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corresponding
 to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
 The third (optional) input is a tensor that specifies the output size of the unpooling operation.

MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
 values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
 the result of an unpooling operation should give back the original input to the unpooling op.

MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
 The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
 known/predictable size.

In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
 which define the exact unpooling op. The attributes typically have the same values as the corresponding
 pooling op that the unpooling op is trying to invert.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>pads</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `I` | tensor of 64-bit signless integer values
| `output_shape` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Mean` (ONNXMeanOp)

_ONNX Mean operation_

Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data_0` | variadic of tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `mean` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.MeanVarianceNormalization` (ONNXMeanVarianceNormalizationOp)

_ONNX MeanVarianceNormalization operation_

A MeanVarianceNormalization Function: Perform mean variance normalization
      on the input tensor X using formula: `(X-EX)/sqrt(E(X-EX)^2)`

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.MelWeightMatrix` (ONNXMelWeightMatrixOp)

_ONNX MelWeightMatrix operation_

Generate a MelWeightMatrix that can be used to re-weight a Tensor containing a linearly sampled frequency spectra (from DFT or STFT) into num_mel_bins frequency information based on the [lower_edge_hertz, upper_edge_hertz] range on the mel scale.
This function defines the mel scale in terms of a frequency in hertz according to the following formula:

    mel(f) = 2595 * log10(1 + f/700)

In the returned matrix, all the triangles (filterbanks) have a peak value of 1.0.

The returned MelWeightMatrix can be used to right-multiply a spectrogram S of shape [frames, num_spectrogram_bins] of linear scale spectrum values (e.g. STFT magnitudes) to generate a \"mel spectrogram\" M of shape [frames, num_mel_bins].

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>output_datatype</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `num_mel_bins` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `dft_length` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `sample_rate` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `lower_edge_hertz` | tensor of 32-bit float values or tensor of 16-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `upper_edge_hertz` | tensor of 32-bit float values or tensor of 16-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Min` (ONNXMinOp)

_ONNX Min operation_

Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data_0` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `min` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Mish` (ONNXMishOp)

_ONNX Mish operation_

Mish: A Self Regularized Non-Monotonic Neural Activation Function.

Perform the linear unit element-wise on the input tensor X using formula:

```
mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Mod` (ONNXModOp)

_ONNX Mod operation_

Performs element-wise binary modulus (with Numpy-style broadcasting support).
  The sign of the remainder is the same as that of the Divisor.

  Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend
  (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.
  This attribute is set to 0 by default causing the behavior to be like integer mod.
  Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().

  If the input type is floating point, then `fmod` attribute must be set to 1.

  In case of dividend being zero, the results will be platform dependent.

  This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>fmod</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `B` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Momentum` (ONNXMomentumOp)

_ONNX Momentum operation_

Compute one iteration of stochastic gradient update with momentum.
    This operator can conduct the optimization of multiple tensor variables.

    Let's define the behavior of this operator. As you can imagine, SG with momentum requires
    several parameters:

     - The learning-rate \"R\".
     - The update count \"T\". That is, the number of conducted training iterations. It should
       be zero in the first training iteration.
     - A L2-norm regularization coefficient \"norm_coefficient\".
     - A decay coefficient of previous accumulated gradient (i.e., momentum) \"alpha\".
     - The scaling coefficient of current gradient \"beta\".
     - An attribute to choose either standard momentum or Nesterov's momentum \"mode\" should
       be used.

    For the sake of simplicity, assume that there is only one tensor (called \"X\") to be optimized.
    Other necessary inputs are \"X\"'s gradient (called \"G\") and \"X\"'s momentum (called \"V\"). This
    Momentum operator maps all these inputs to the new value of \"X\" (called \"X_new\") and its new
    momentum (called \"V_new\").

    This operator supports two different momentum algorithms. Set the attribute \"mode\" to
    \"nesterov\" if Nesterov's momentum is desired. Otherwise, set the attribute \"model\" to
    \"standard\" to use standard momentum. Computation details are described subsequently.

    Let \"+\", \"-\", \"*\", and \"/\" are all element-wise operations with numpy-style broadcasting.

    Pseudo code for SG with standard momentum:

      // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
      // values of all elements in X.
      G_regularized = norm_coefficient * X + G

      // In the first training iteration, beta should always be 1.
      beta_adjusted = T > 0 ? beta : 1

      // Compute the current momentum based on previous momentum and the current gradient.
      V_new = alpha * V + beta_adjusted * G_regularized

      // Update X.
      X_new = X - R * V_new

    Pseudo code for SG with Nesterov's momentum:

      // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
      // values of all elements in X.
      G_regularized = norm_coefficient * X + G;

      // In the first training iteration, beta should always be 1.
      beta_adjusted = T > 0 ? beta : 1

      // Compute the current momentum based on previous momentum and the current gradient.
      V_new = alpha * V + beta_adjusted * G_regularized;

      // Compute final update direction and then update X.
      X_new = X - R * (G_regularized + alpha * V_new)

    If one assign this operators to optimize multiple inputs, for example, \"X_1\" and \"X_2\". The same
    pseudo code would be extended to handle all tensors jointly. More specifically, we can view \"X\" as a
    concatenation of \"X_1\" and \"X_2\" (of course, their gradient and accumulate gradient should
    be concatenated too) and then our pseudo code becomes applicable.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>alpha</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>beta</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>norm_coefficient</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `R` | tensor of 32-bit float values or tensor of 64-bit float values
| `T` | tensor of 64-bit signless integer values
| `inputs` | variadic of tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `outputs` | variadic of tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Mul` (ONNXMulOp)

_ONNX Mul operation_

Performs element-wise binary multiplication (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `B` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Multinomial` (ONNXMultinomialOp)

_ONNX Multinomial operation_

Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dtype</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>sample_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>seed</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

### `onnx.Neg` (ONNXNegOp)

_ONNX Neg operation_

Neg takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where each element flipped sign, y = -x, is applied to
the tensor elementwise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values or tensor of 32-bit signless integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 32-bit float values or tensor of 32-bit signless integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.NegativeLogLikelihoodLoss` (ONNXNegativeLogLikelihoodLossOp)

_ONNX NegativeLogLikelihoodLoss operation_

A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
Its \"input\" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
The \"input\" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
The operator's \"target\" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:

```
loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].
```

When an optional \"weight\" is provided, the sample loss is calculated as:

```
loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].
```

loss is zero for the case when target-value equals ignore_index.

```
loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index
```

If \"reduction\" attribute is set to \"none\", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
If \"reduction\" attribute is set to \"mean\" (the default attribute value), the output loss is (weight) averaged:

```
mean(loss), if \"weight\" is not provided,
```

or if weight is provided,

```
sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.
```

If \"reduction\" attribute is set to \"sum\", the output is a scalar: `sum(loss)`.

See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.

Example 1:

```
// negative log likelihood loss, \"none\" reduction
N, C, d1 = 2, 3, 2
input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
          [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
target = [[2, 1], [0, 2]]

loss = np.zeros((N, d1))
for n in range(N):
    for d_1 in range(d1):
        c = target[n][d_1]
        loss[n][d_1] = -input[n][c][d_1]

// print(loss)
// [[-3. -2.]
//  [-0. -2.]]
```

Example 2:

```
// weighted negative log likelihood loss, sum reduction
N, C, d1 = 2, 3, 2
input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
        [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
target = [[2, 1], [0, 2]]
weight = [0.2, 0.3, 0.1]
loss = np.zeros((N, d1))
for n in range(N):
    for d_1 in range(d1):
        c = target[n][d_1]
        loss[n][d_1] = -input[n][c][d_1] * weight[c]

loss = np.sum(loss)
// print(loss)
// -1.1
```

Example 3:

```
// weighted negative log likelihood loss, mean reduction
N, C, d1 = 2, 3, 2
input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
        [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
target = [[2, 1], [0, 2]]
weight = [0.2, 0.3, 0.1]
loss = np.zeros((N, d1))
weight_total = 0
for n in range(N):
    for d_1 in range(d1):
        c = target[n][d_1]
        loss[n][d_1] = -input[n][c][d_1] * weight[c]
        weight_total = weight_total + weight[c]

loss = np.sum(loss) / weight_total
// print(loss)
// -1.57
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>ignore_index</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>reduction</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `target` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `weight` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `loss` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.NonMaxSuppression` (ONNXNonMaxSuppressionOp)

_ONNX NonMaxSuppression operation_

Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>center_point_box</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `boxes` | tensor of 32-bit float values
| `scores` | tensor of 32-bit float values
| `max_output_boxes_per_class` | tensor of 64-bit signless integer values or none type
| `iou_threshold` | tensor of 32-bit float values or none type
| `score_threshold` | tensor of 32-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `selected_indices` | tensor of 64-bit signless integer values

### `onnx.NonZero` (ONNXNonZeroOp)

_ONNX NonZero operation_

Returns the indices of the elements that are non-zero
    (in row-major order - by dimension).
    NonZero behaves similar to numpy.nonzero:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html,
    but for scalar input, NonZero produces output shape (0, N) instead of (1, N), which is different from Numpy's behavior.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 64-bit signless integer values

### `onnx.NoValue` (ONNXNoneOp)

_An operation representing the absence of a value._

This operation can be used to represent the absence of a value. It is typically
used as an argument to operators that have optional parameters.

Example:
```MLIR
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis=1 : si64 } : (tensor<?xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
```

This operation is not part of the standard and was added to assist onnx-mlir.

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>value</code></td><td>::mlir::UnitAttr</td><td>unit attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `none_val` | none type

### `onnx.Normalizer` (ONNXNormalizerOp)

_ONNX Normalizer operation_

Normalize the input.  There are three normalization modes, which have the corresponding formulas,
    defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':<br>
<br>
    Max: Y = X / max(X)<br>
    L1:  Y = X / sum(X)<br>
    L2:  Y = sqrt(X^2 / sum(X^2)}<br>
    In all modes, if the divisor is zero, Y == X.
<br>
    For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row
    of the batch is normalized independently.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>norm</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 32-bit float values

### `onnx.Not` (ONNXNotOp)

_ONNX Not operation_

Returns the negation of the input tensor element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 1-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 1-bit signless integer values

### `onnx.OneHotEncoder` (ONNXOneHotEncoderOp)

_ONNX OneHotEncoder operation_

Replace each input element with an array of ones and zeros, where a single
    one is placed at the index of the category that was passed in. The total category count
    will determine the size of the extra dimension of the output array Y.<br>
    For example, if we pass a tensor with a single value of 4, and a category count of 8,
    the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.<br>
    This operator assumes every input feature is from the same set of categories.<br>
    If the input is a tensor of float, int32, or double, the data will be cast
    to integers and the cats_int64s category list will be used for the lookups.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>cats_int64s</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>cats_strings</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>zeros</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of string type values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 32-bit float values

### `onnx.OneHot` (ONNXOneHotOp)

_ONNX OneHot operation_

Produces a one-hot tensor based on inputs.
    The locations represented by the index values in the 'indices' input tensor will have 'on_value'
    and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
    are specified as part of required input argument 'values', which is a two-element tensor of format
    [off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
    input tensor. The additional dimension is for one-hot representation. The additional dimension will
    be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional
    dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
    dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
    as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
    the range [-depth, depth-1] will result in one-hot representation with all 'off_value' values in the
    output tensor.

    when axis = 0:
    output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.

    when axis = -1:
    output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.


Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `indices` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `depth` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `values` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.OptionalGetElement` (ONNXOptionalGetElementOp)

_ONNX OptionalGetElement operation_

If the input is a tensor or sequence type, it returns the input.
If the input is an optional type, it outputs the element in the input.
It is an error if the input is an empty optional-type (i.e. does not have an element) and the behavior is undefined in this case.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | OptType of SeqType of tensor of 8-bit unsigned integer values values values or OptType of SeqType of tensor of 16-bit unsigned integer values values values or OptType of SeqType of tensor of 32-bit unsigned integer values values values or OptType of SeqType of tensor of 64-bit unsigned integer values values values or OptType of SeqType of tensor of 8-bit signless integer values values values or OptType of SeqType of tensor of 16-bit signless integer values values values or OptType of SeqType of tensor of 32-bit signless integer values values values or OptType of SeqType of tensor of 64-bit signless integer values values values or OptType of SeqType of tensor of 16-bit float values values values or OptType of SeqType of tensor of 32-bit float values values values or OptType of SeqType of tensor of 64-bit float values values values or OptType of SeqType of tensor of string type values values values or OptType of SeqType of tensor of 1-bit signless integer values values values or OptType of SeqType of tensor of complex type with 32-bit float elements values values values or OptType of SeqType of tensor of complex type with 64-bit float elements values values values or OptType of tensor of 8-bit unsigned integer values values or OptType of tensor of 16-bit unsigned integer values values or OptType of tensor of 32-bit unsigned integer values values or OptType of tensor of 64-bit unsigned integer values values or OptType of tensor of 8-bit signless integer values values or OptType of tensor of 16-bit signless integer values values or OptType of tensor of 32-bit signless integer values values or OptType of tensor of 64-bit signless integer values values or OptType of tensor of 16-bit float values values or OptType of tensor of 32-bit float values values or OptType of tensor of 64-bit float values values or OptType of tensor of string type values values or OptType of tensor of 1-bit signless integer values values or OptType of tensor of complex type with 32-bit float elements values values or OptType of tensor of complex type with 64-bit float elements values values or tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values

### `onnx.OptionalHasElement` (ONNXOptionalHasElementOp)

_ONNX OptionalHasElement operation_

Returns true if (1) the input is an optional-type and contains an element,
or, (2) the input is a tensor or sequence type.
If the input is not provided or is an empty optional-type, this op returns false.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | OptType of SeqType of tensor of 8-bit unsigned integer values values values or OptType of SeqType of tensor of 16-bit unsigned integer values values values or OptType of SeqType of tensor of 32-bit unsigned integer values values values or OptType of SeqType of tensor of 64-bit unsigned integer values values values or OptType of SeqType of tensor of 8-bit signless integer values values values or OptType of SeqType of tensor of 16-bit signless integer values values values or OptType of SeqType of tensor of 32-bit signless integer values values values or OptType of SeqType of tensor of 64-bit signless integer values values values or OptType of SeqType of tensor of 16-bit float values values values or OptType of SeqType of tensor of 32-bit float values values values or OptType of SeqType of tensor of 64-bit float values values values or OptType of SeqType of tensor of string type values values values or OptType of SeqType of tensor of 1-bit signless integer values values values or OptType of SeqType of tensor of complex type with 32-bit float elements values values values or OptType of SeqType of tensor of complex type with 64-bit float elements values values values or OptType of tensor of 8-bit unsigned integer values values or OptType of tensor of 16-bit unsigned integer values values or OptType of tensor of 32-bit unsigned integer values values or OptType of tensor of 64-bit unsigned integer values values or OptType of tensor of 8-bit signless integer values values or OptType of tensor of 16-bit signless integer values values or OptType of tensor of 32-bit signless integer values values or OptType of tensor of 64-bit signless integer values values or OptType of tensor of 16-bit float values values or OptType of tensor of 32-bit float values values or OptType of tensor of 64-bit float values values or OptType of tensor of string type values values or OptType of tensor of 1-bit signless integer values values or OptType of tensor of complex type with 32-bit float elements values values or OptType of tensor of complex type with 64-bit float elements values values or tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 1-bit signless integer values

### `onnx.Optional` (ONNXOptionalOp)

_ONNX Optional operation_

Constructs an optional-type value containing either an empty optional of a certain type specified by the attribute,
or a non-empty value containing the input element.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>type</code></td><td>::mlir::TypeAttr</td><td>any type attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | OptType of SeqType of tensor of 8-bit unsigned integer values values values or OptType of SeqType of tensor of 16-bit unsigned integer values values values or OptType of SeqType of tensor of 32-bit unsigned integer values values values or OptType of SeqType of tensor of 64-bit unsigned integer values values values or OptType of SeqType of tensor of 8-bit signless integer values values values or OptType of SeqType of tensor of 16-bit signless integer values values values or OptType of SeqType of tensor of 32-bit signless integer values values values or OptType of SeqType of tensor of 64-bit signless integer values values values or OptType of SeqType of tensor of 16-bit float values values values or OptType of SeqType of tensor of 32-bit float values values values or OptType of SeqType of tensor of 64-bit float values values values or OptType of SeqType of tensor of string type values values values or OptType of SeqType of tensor of 1-bit signless integer values values values or OptType of SeqType of tensor of complex type with 32-bit float elements values values values or OptType of SeqType of tensor of complex type with 64-bit float elements values values values or OptType of tensor of 8-bit unsigned integer values values or OptType of tensor of 16-bit unsigned integer values values or OptType of tensor of 32-bit unsigned integer values values or OptType of tensor of 64-bit unsigned integer values values or OptType of tensor of 8-bit signless integer values values or OptType of tensor of 16-bit signless integer values values or OptType of tensor of 32-bit signless integer values values or OptType of tensor of 64-bit signless integer values values or OptType of tensor of 16-bit float values values or OptType of tensor of 32-bit float values values or OptType of tensor of 64-bit float values values or OptType of tensor of string type values values or OptType of tensor of 1-bit signless integer values values or OptType of tensor of complex type with 32-bit float elements values values or OptType of tensor of complex type with 64-bit float elements values values

### `onnx.Or` (ONNXOrOp)

_ONNX Or operation_

Returns the tensor resulted from performing the `or` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 1-bit signless integer values
| `B` | tensor of 1-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 1-bit signless integer values

### `onnx.PRelu` (ONNXPReluOp)

_ONNX PRelu operation_

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `slope` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

### `onnx.Pad` (ONNXPadOp)

_ONNX Pad operation_

Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array

4) `wrap` - wrap-around padding as if the data tensor forms a torus


Example 1 (`constant` mode):

Insert 0 pads to the beginning of the second dimension.

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'constant'

constant_value = 0.0

output = [
    [0.0, 0.0, 1.0, 1.2],
    [0.0, 0.0, 2.3, 3.4],
    [0.0, 0.0, 4.5, 5.7],
]
```

Example 2 (`reflect` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'reflect'

output = [
    [1.0, 1.2, 1.0, 1.2],
    [2.3, 3.4, 2.3, 3.4],
    [4.5, 5.7, 4.5, 5.7],
]
```

Example 3 (`edge` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'edge'

output = [
    [1.0, 1.0, 1.0, 1.2],
    [2.3, 2.3, 2.3, 3.4],
    [4.5, 4.5, 4.5, 5.7],
]
```

Example 4 (`wrap` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [2, 1, 1, 1]

mode = 'wrap'

output = [
    [3.4, 2.3, 3.4, 2.3],
    [5.7, 4.5, 5.7, 4.5],
    [1.2, 1.0, 1.2, 1.0],
    [3.4, 2.3, 3.4, 2.3],
    [5.7, 4.5, 5.7, 4.5],
    [1.2, 1.0, 1.2, 1.0],
]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values
| `pads` | tensor of 64-bit signless integer values
| `constant_value` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values or none type
| `axes` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values

### `onnx.PadV11` (ONNXPadV11Op)

_ONNX Pad operation_

Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array


Example 1 (`constant` mode):
  Insert 0 pads to the beginning of the second dimension.

  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'constant'

  constant_value = 0.0

  output =
  [
      [0.0, 0.0, 1.0, 1.2],
      [0.0, 0.0, 2.3, 3.4],
      [0.0, 0.0, 4.5, 5.7],
  ]


Example 2 (`reflect` mode):
  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'reflect'

  output =
  [
      [1.0, 1.2, 1.0, 1.2],
      [2.3, 3.4, 2.3, 3.4],
      [4.5, 5.7, 4.5, 5.7],
  ]


Example 3 (`edge` mode):
  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'edge'

  output =
  [
      [1.0, 1.0, 1.0, 1.2],
      [2.3, 2.3, 2.3, 3.4],
      [4.5, 4.5, 4.5, 5.7],
  ]


Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `pads` | tensor of 64-bit signless integer values
| `constant_value` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.PadV13` (ONNXPadV13Op)

_ONNX Pad operation_

Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array


Example 1 (`constant` mode):
  Insert 0 pads to the beginning of the second dimension.

  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'constant'

  constant_value = 0.0

  output =
  [
      [0.0, 0.0, 1.0, 1.2],
      [0.0, 0.0, 2.3, 3.4],
      [0.0, 0.0, 4.5, 5.7],
  ]


Example 2 (`reflect` mode):
  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'reflect'

  output =
  [
      [1.0, 1.2, 1.0, 1.2],
      [2.3, 3.4, 2.3, 3.4],
      [4.5, 5.7, 4.5, 5.7],
  ]


Example 3 (`edge` mode):
  data =
  [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]

  pads = [0, 2, 0, 0]

  mode = 'edge'

  output =
  [
      [1.0, 1.0, 1.0, 1.2],
      [2.3, 2.3, 2.3, 3.4],
      [4.5, 4.5, 4.5, 5.7],
  ]


Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `pads` | tensor of 64-bit signless integer values
| `constant_value` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.PadV18` (ONNXPadV18Op)

_ONNX Pad operation_

Given a tensor containing the data to be padded (`data`), a tensor containing the number of start and end pad values for axis (`pads`), (optionally) a `mode`, and (optionally) `constant_value`,
a padded tensor (`output`) is generated.

The three supported `modes` are (similar to corresponding modes supported by `numpy.pad`):

1) `constant`(default) - pads with a given constant value as specified by `constant_value` (which defaults to 0, empty string, or False)

2) `reflect` - pads with the reflection of the vector mirrored on the first and last values of the vector along each axis

3) `edge` - pads with the edge values of array


Example 1 (`constant` mode):

Insert 0 pads to the beginning of the second dimension.

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'constant'

constant_value = 0.0

output = [
    [0.0, 0.0, 1.0, 1.2],
    [0.0, 0.0, 2.3, 3.4],
    [0.0, 0.0, 4.5, 5.7],
]
```

Example 2 (`reflect` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'reflect'

output = [
    [1.0, 1.2, 1.0, 1.2],
    [2.3, 3.4, 2.3, 3.4],
    [4.5, 5.7, 4.5, 5.7],
]
```

Example 3 (`edge` mode):

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]

pads = [0, 2, 0, 0]

mode = 'edge'

output = [
    [1.0, 1.0, 1.0, 1.2],
    [2.3, 2.3, 2.3, 3.4],
    [4.5, 4.5, 4.5, 5.7],
]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `pads` | tensor of 64-bit signless integer values
| `constant_value` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or none type
| `axes` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.PadV2` (ONNXPadV2Op)

_ONNX Pad operation_

Given `data` tensor, pads, mode, and value.
Example:
  Insert 0 pads to the beginning of the second dimension.
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  pads = [0, 2, 0, 0]
  output = [
      [
          [0.0, 0.0, 1.0, 1.2],
          [0.0, 0.0, 2.3, 3.4],
          [0.0, 0.0, 4.5, 5.7],
      ],
  ]

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>pads</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>value</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Pow` (ONNXPowOp)

_ONNX Pow operation_

Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Z` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.PrintSignature` (ONNXPrintSignatureOp)

_ONNX Op to print type signature or data of its input operands_

Print type signature or data of the input operands of this op.
The parameter op_name specifies a string to be printed before the tensors.
and usually the op_name and onnx_node_name are used.
This operation is introduced early so as to preserve the name of the original ONNX op.
The argument print_data control whether the data of the tensors to be printed.
When print_data == 1, the data of the tensor will be printed. Otherwise, just shape.
The argument input specifies the tensor to be printed. They could be a list
of the inputs and outputs of an ONNX op.

This operation is not part of the standard and was added to assist onnx-mlir.

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>op_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>print_data</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | variadic of tensor of any type values or none type

### `onnx.QLinearConv` (ONNXQLinearConvOp)

_ONNX QLinearConv operation_

The convolution operator consumes a quantized input tensor, its scale and zero point,
a quantized filter, its scale and zero point, and output's scale and zero point,
and computes the quantized output. Each scale and zero-point pair must have same shape.
It means they must be either scalars (per tensor) or 1-D tensors (per output channel).
Each input or output and its related zero point must have same type.
When bias is present it must be quantized using scale = input scale * weight scale and
zero point as 0.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>auto_pad</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>dilations</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>group</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>kernel_shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>pads</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>strides</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `x` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values
| `x_scale` | tensor of 32-bit float values
| `x_zero_point` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values
| `w` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values
| `w_scale` | tensor of 32-bit float values
| `w_zero_point` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values
| `y_scale` | tensor of 32-bit float values
| `y_zero_point` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values
| `B` | tensor of 32-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `y` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values

### `onnx.QLinearMatMul` (ONNXQLinearMatMulOp)

_ONNX QLinearMatMul operation_

Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
(per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
or per column quantization. If the input is 2D of shape [M, K] then zero point and scale tensor may be
an M element vector [v_1, v_2, ..., v_M] for per row quantization and K element vector of shape [v_1, v_2, ..., v_K]
for per column quantization. If the input is N-D tensor with shape [D1, D2, M, K] then zero point and scale tensor may
have shape [D1, D2, M, 1] for per row quantization and shape [D1, D2, 1, K] for per column quantization.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `a` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values
| `a_scale` | tensor of 32-bit float values
| `a_zero_point` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values
| `b` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values
| `b_scale` | tensor of 32-bit float values
| `b_zero_point` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values
| `y_scale` | tensor of 32-bit float values
| `y_zero_point` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `y` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values

### `onnx.QuantizeLinear` (ONNXQuantizeLinearOp)

_ONNX QuantizeLinear operation_

The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is `y = saturate ((x / y_scale) + y_zero_point)`.
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to the nearest even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
'y_zero_point' and 'y' must have same type.
'y_zero_point' is usually not used for quantization to float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz,
but the quantization formula remains the same for consistency and
the type of the attribute 'y_zero_point' still determines the quantization type.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>saturate</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `x` | tensor of 32-bit float values or tensor of 16-bit float values or tensor of bfloat16 type values or tensor of 32-bit signless integer values
| `y_scale` | tensor of 32-bit float values or tensor of 16-bit float values or tensor of bfloat16 type values or tensor of 32-bit signless integer values
| `y_zero_point` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `y` | tensor of 8-bit signless integer values or tensor of 8-bit unsigned integer values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values

### `onnx.RMSLayerNormalization` (ONNXRMSLayerNormalizationOp)

_ONNX RMSLayerNormalization operation_

This is RMS layer normalization defined in ONNX as function.
      The overall computation can be split into two stages.
      The first stage is an approximate standardization, which makes the
      normalized elements have zero mean and unit variances.
      See Equation (4) in [this paper](https://arxiv.org/pdf/1910.07467.pdf).
      The computation required by standardization can be
      described by the following equations.
      ```
      DD = Mul(X, X)
      Var = ReduceMean<axes=normalized_axes>(DD)
      VarEps = Add(Var, epsilon)
      StdDev = Sqrt(VarEps)
      InvStdDev = Reciprocal(StdDev)
      Normalized = Mul(X, InvStdDev)
      ```
      where `normalized_axes` is `[axis, ..., rank of X - 1]`.
      The variables `Var` and `StdDev` stand for approximate variance and
      standard deviation, respectively.
      Depending on `stash_type` attribute, the actual computation
      must happen in different floating-point precision.
      For example, if `stash_type` is 1, this operator casts
      all input variables to 32-bit float, perform the computation, and
      finally cast `Normalized` back to the original type of `X`.
      The second stage then scales and shifts the outcome of the
      first stage using
      ```
      NormalizedScaled = Mul(Normalized, Scale)
      Y = Add(NormalizedScaled, B)
      ```
      The second stage doesn't depends on `stash_type`.
      All equations are in [this syntax](https://github.com/onnx/onnx/blob/main/docs/Syntax.md).
      The same variable (i.e., input, output, and attribute) uses
      the same name in the equations above and this operator's definition.
      Let `d[i]` indicate the i-th dimension of `X`.
      If `X`'s shape is `[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]`,
      the shape of `Mean` and `InvStdDev` is `[d[0], ..., d[axis-1], 1, ..., 1]`.
      `Y` and `X` have the same shape.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>epsilon</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>stash_type</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `Scale` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `B` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `InvStdDev` | tensor of 32-bit float values or tensor of bfloat16 type values or none type

### `onnx.RNN` (ONNXRNNOp)

_ONNX RNN operation_

Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

* `X` - input tensor
* `i` - input gate
* `t` - time step (t-1 means previous time step)
* `Wi` - W parameter weight matrix for input gate
* `Ri` - R recurrence weight matrix for input gate
* `Wbi` - W parameter bias vector for input gate
* `Rbi` - R parameter bias vector for input gate
* `WBi` - W parameter weight matrix for backward input gate
* `RBi` - R recurrence weight matrix for backward input gate
* `WBbi` - WR bias vectors for backward input gate
* `RBbi` - RR bias vectors for backward input gate
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE: Below are optional

* Affine(x)              - alpha*x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha*Tanh(beta*x)
* HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Tanh):

* Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>activation_alpha</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>activation_beta</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>activations</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>clip</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>direction</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>hidden_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>layout</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `W` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `R` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `B` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `sequence_lens` | tensor of 32-bit signless integer values or none type
| `initial_h` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `Y_h` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type

### `onnx.RandomNormalLike` (ONNXRandomNormalLikeOp)

_ONNX RandomNormalLike operation_

Generate a tensor with random values drawn from a normal distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the normal distribution are specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message, and be valid as an output type.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ResultTypeInferenceOpInterface`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dtype</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>mean</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>scale</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>seed</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.RandomNormal` (ONNXRandomNormalOp)

_ONNX RandomNormal operation_

Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ResultTypeInferenceOpInterface`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dtype</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>mean</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>scale</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>seed</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.RandomUniformLike` (ONNXRandomUniformLikeOp)

_ONNX RandomUniformLike operation_

Generate a tensor with random values drawn from a uniform distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the uniform distribution are specified by `low` and `high`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dtype</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>high</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>low</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>seed</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.RandomUniform` (ONNXRandomUniformOp)

_ONNX RandomUniform operation_

Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dtype</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>high</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>low</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>seed</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>shape</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Range` (ONNXRangeOp)

_ONNX Range operation_

Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
up to `limit` (exclusive).

The number of elements in the output of range is computed as below:

```
number_of_elements = max( ceil( (limit - start) / delta ) , 0 )
```

The pseudocode determining the contents of the output is shown below:

```
for(int i=0; i<number_of_elements; ++i) {
  output[i] =  start + (i * delta);
}
```

Example 1

```
Inputs: start = 3, limit = 9, delta = 3
Output: [3, 6]
```

Example 2

```
Inputs: start = 10, limit = 4, delta = -2
Output: [10, 8, 6]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `start` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `limit` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `delta` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

### `onnx.Reciprocal` (ONNXReciprocalOp)

_ONNX Reciprocal operation_

Reciprocal takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the reciprocal is, y = 1/x, is applied to
the tensor elementwise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceL1` (ONNXReduceL1Op)

_ONNX ReduceL1 operation_

Computes the L1 norm of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>noop_with_empty_axes</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `axes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceL1V13` (ONNXReduceL1V13Op)

_ONNX ReduceL1 operation_

Computes the L1 norm of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceL2` (ONNXReduceL2Op)

_ONNX ReduceL2 operation_

Computes the L2 norm of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>noop_with_empty_axes</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `axes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceL2V13` (ONNXReduceL2V13Op)

_ONNX ReduceL2 operation_

Computes the L2 norm of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceLogSumExp` (ONNXReduceLogSumExpOp)

_ONNX ReduceLogSumExp operation_

Computes the log sum exponent of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>noop_with_empty_axes</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `axes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceLogSumExpV13` (ONNXReduceLogSumExpV13Op)

_ONNX ReduceLogSumExp operation_

Computes the log sum exponent of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceLogSum` (ONNXReduceLogSumOp)

_ONNX ReduceLogSum operation_

Computes the log sum of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>noop_with_empty_axes</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `axes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceLogSumV13` (ONNXReduceLogSumV13Op)

_ONNX ReduceLogSum operation_

Computes the log sum of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or undefined otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceMax` (ONNXReduceMaxOp)

_ONNX ReduceMax operation_

Computes the max of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise.


If the input data type is Boolean, the comparison should consider `False < True`.

The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>noop_with_empty_axes</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or tensor of 8-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 1-bit signless integer values
| `axes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or tensor of 8-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 1-bit signless integer values

### `onnx.ReduceMaxV13` (ONNXReduceMaxV13Op)

_ONNX ReduceMax operation_

Computes the max of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or tensor of 8-bit unsigned integer values or tensor of 8-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or tensor of 8-bit unsigned integer values or tensor of 8-bit signless integer values

### `onnx.ReduceMaxV18` (ONNXReduceMaxV18Op)

_ONNX ReduceMax operation_

Computes the max of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields minus infinity (if supported by the datatype) or the minimum value of the data type otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>noop_with_empty_axes</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or tensor of 8-bit unsigned integer values or tensor of 8-bit signless integer values
| `axes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or tensor of 8-bit unsigned integer values or tensor of 8-bit signless integer values

### `onnx.ReduceMean` (ONNXReduceMeanOp)

_ONNX ReduceMean operation_

Computes the mean of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields undefined.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>noop_with_empty_axes</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `axes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceMeanV13` (ONNXReduceMeanV13Op)

_ONNX ReduceMean operation_

Computes the mean of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields undefined.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceMin` (ONNXReduceMinOp)

_ONNX ReduceMin operation_

Computes the min of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields plus infinity (if supported by the datatype) or the maximum value of the data type otherwise.


If the input data type is Boolean, the comparison should consider `False < True`.

The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>noop_with_empty_axes</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or tensor of 8-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 1-bit signless integer values
| `axes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or tensor of 8-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 1-bit signless integer values

### `onnx.ReduceMinV13` (ONNXReduceMinV13Op)

_ONNX ReduceMin operation_

Computes the min of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields plus infinity (if supported by the datatype) or the maximum value of the data type otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or tensor of 8-bit unsigned integer values or tensor of 8-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or tensor of 8-bit unsigned integer values or tensor of 8-bit signless integer values

### `onnx.ReduceMinV18` (ONNXReduceMinV18Op)

_ONNX ReduceMin operation_

Computes the min of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields plus infinity (if supported by the datatype) or the maximum value of the data type otherwise.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>noop_with_empty_axes</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or tensor of 8-bit unsigned integer values or tensor of 8-bit signless integer values
| `axes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or tensor of 8-bit unsigned integer values or tensor of 8-bit signless integer values

### `onnx.ReduceProd` (ONNXReduceProdOp)

_ONNX ReduceProd operation_

Computes the product of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 1.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>noop_with_empty_axes</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `axes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceProdV13` (ONNXReduceProdV13Op)

_ONNX ReduceProd operation_

Computes the product of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 1.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceSum` (ONNXReduceSumOp)

_ONNX ReduceSum operation_

Computes the sum of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>noop_with_empty_axes</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `axes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceSumSquare` (ONNXReduceSumSquareOp)

_ONNX ReduceSumSquare operation_

Computes the sum square of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>noop_with_empty_axes</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `axes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceSumSquareV13` (ONNXReduceSumSquareV13Op)

_ONNX ReduceSumSquare operation_

Computes the sum square of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.ReduceSumV11` (ONNXReduceSumV11Op)

_ONNX ReduceSum operation_

Computes the sum of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `reduced` | tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Relu` (ONNXReluOp)

_ONNX Relu operation_

Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = max(0, x), is applied to
the tensor elementwise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values or tensor of 32-bit signless integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 32-bit float values or tensor of 32-bit signless integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Reshape` (ONNXReshapeOp)

_ONNX Reshape operation_

Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor). If 'allowzero' is set, and the new shape includes 0, the
dimension will be set explicitly to zero (i.e. not taken from input tensor).
Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.

If the attribute 'allowzero' is set, it is invalid for the specified shape to
contain both a zero value and -1, as the value of the dimension corresponding
to -1 cannot be determined uniquely.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>allowzero</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values
| `shape` | tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `reshaped` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values

### `onnx.Resize` (ONNXResizeOp)

_ONNX Resize operation_

Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
```
output_dimension = floor(input_dimension * (roi_end - roi_start) * scale)
```
if input \\"sizes\\" is not specified.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>antialias</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>coordinate_transformation_mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>cubic_coeff_a</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>exclude_outside</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>extrapolation_value</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>keep_aspect_ratio_policy</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>nearest_mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `roi` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `scales` | tensor of 32-bit float values or none type
| `sizes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.ResizeV10` (ONNXResizeV10Op)

_ONNX Resize operation_

Resize the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `scales` | tensor of 32-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.ResizeV11` (ONNXResizeV11Op)

_ONNX Resize operation_

Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \\"sizes\\" is not specified.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>coordinate_transformation_mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>cubic_coeff_a</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>exclude_outside</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>extrapolation_value</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>nearest_mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `roi` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `scales` | tensor of 32-bit float values
| `sizes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.ResizeV13` (ONNXResizeV13Op)

_ONNX Resize operation_

Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \\"sizes\\" is not specified.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>coordinate_transformation_mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>cubic_coeff_a</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>exclude_outside</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>extrapolation_value</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>nearest_mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `roi` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `scales` | tensor of 32-bit float values or none type
| `sizes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.ResizeV18` (ONNXResizeV18Op)

_ONNX Resize operation_

Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is: <br/>
  `output_dimension = floor(input_dimension * (roi_end - roi_start) * scale)` <br/>
if input \\"sizes\\" is not specified.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>antialias</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>coordinate_transformation_mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>cubic_coeff_a</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>exclude_outside</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>extrapolation_value</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>keep_aspect_ratio_policy</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>nearest_mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `roi` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or none type
| `scales` | tensor of 32-bit float values or none type
| `sizes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Return` (ONNXReturnOp)

_Function return operation_


Syntax:

```
operation ::= `onnx.Return` attr-dict ($operands^ `:` type($operands))?
```

The `onnx.Return` operation represents a return operation within a function.
The operation takes variable number of operands and produces no results.
The operand number and types must match the signature of the function
that contains the operation, with the exception that shaped types may have
more specific shapes than the function signature result types, which allows
rewrites of defining ops of operands to make their result shapes more specific.
This operation terminates a func::FuncOp in the ONNX dialect and is replaced
by func::ReturnOp in StandardFuncReturnPass before lowering to Krnl or other
dialects.

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<func::FuncOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operands` | variadic of any type

### `onnx.ReverseSequence` (ONNXReverseSequenceOp)

_ONNX ReverseSequence operation_

Reverse batch of sequences having different lengths specified by `sequence_lens`.

For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.

Example 1:
  input = [[0.0, 4.0, 8.0,  12.0],
           [1.0, 5.0, 9.0,  13.0],
           [2.0, 6.0, 10.0, 14.0],
           [3.0, 7.0, 11.0, 15.0]]
  sequence_lens = [4, 3, 2, 1]
  time_axis = 0
  batch_axis = 1

  output = [[3.0, 6.0, 9.0,  12.0],
            [2.0, 5.0, 8.0,  13.0],
            [1.0, 4.0, 10.0, 14.0],
            [0.0, 7.0, 11.0, 15.0]]

Example 2:
  input = [[0.0,  1.0,  2.0,  3.0 ],
           [4.0,  5.0,  6.0,  7.0 ],
           [8.0,  9.0,  10.0, 11.0],
           [12.0, 13.0, 14.0, 15.0]]
  sequence_lens = [1, 2, 3, 4]
  time_axis = 1
  batch_axis = 0

  output = [[0.0,  1.0,  2.0,  3.0 ],
            [5.0,  4.0,  6.0,  7.0 ],
            [10.0, 9.0,  8.0,  11.0],
            [15.0, 14.0, 13.0, 12.0]]

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>batch_axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>time_axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `sequence_lens` | tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.RoiAlign` (ONNXRoiAlignOp)

_ONNX RoiAlign operation_

Region of Interest (RoI) align operation described in the
[Mask R-CNN paper](https://arxiv.org/abs/1703.06870).
RoiAlign consumes an input tensor X and region of interests (rois)
to apply pooling across each RoI; it produces a 4-D tensor of shape
(num_rois, C, output_height, output_width).

RoiAlign is proposed to avoid the misalignment by removing
quantizations while converting from original image into feature
map and from feature map into RoI feature; in each ROI bin,
the value of the sampled locations are computed directly
through bilinear interpolation.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>coordinate_transformation_mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>output_height</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>output_width</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>sampling_ratio</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>spatial_scale</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `rois` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `batch_indices` | tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Round` (ONNXRoundOp)

_ONNX Round operation_

Round takes one input Tensor and rounds the values, element-wise, meaning
it finds the nearest integer for each value.
In case of halves, the rule is to round them to the nearest even integer.
If input x is integral, +0, -0, NaN,  or infinite, x itself is returned.
The output tensor has the same shape and type as the input.

Examples:
```
round([0.9]) = [1.0]
round([2.5]) = [2.0]
round([2.3]) = [2.0]
round([1.5]) = [2.0]
round([-4.5]) = [-4.0]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.STFT` (ONNXSTFTOp)

_ONNX STFT operation_

Computes the Short-time Fourier Transform of the signal.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>onesided</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `signal` | tensor of 32-bit float values or tensor of 16-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `frame_step` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `window` | tensor of 32-bit float values or tensor of 16-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or none type
| `frame_length` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 32-bit float values or tensor of 16-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.SVMClassifier` (ONNXSVMClassifierOp)

_ONNX SVMClassifier operation_

Support Vector Machine classifier

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>classlabels_ints</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>classlabels_strings</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>coefficients</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>kernel_params</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>kernel_type</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>post_transform</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>prob_a</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>prob_b</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>rho</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>support_vectors</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>vectors_per_class</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of string type values or tensor of 64-bit signless integer values
| `Z` | tensor of 32-bit float values

### `onnx.SVMRegressor` (ONNXSVMRegressorOp)

_ONNX SVMRegressor operation_

Support Vector Machine regression prediction and one-class SVM anomaly detection.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>coefficients</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>kernel_params</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>kernel_type</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>n_supports</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>one_class</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>post_transform</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>rho</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>support_vectors</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 32-bit float values

### `onnx.Scaler` (ONNXScalerOp)

_ONNX Scaler operation_

Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>offset</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>scale</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 32-bit float values

### `onnx.Scan` (ONNXScanOp)

_ONNX Scan operation_

Scan can be used to iterate over one or more scan_input tensors,
constructing zero or more scan_output tensors. It combines ideas from general recurrences,
functional programming constructs such as scan, fold, map, and zip, and is intended to enable
generalizations of RNN-like constructs for sequence-to-sequence processing.
Other tensors (referred to as state_variables here) can be used to carry a state
when iterating from one element to another (similar to hidden-state in RNNs, also referred
to as loop-carried dependences in the context of loops).
Many common usages involve a single scan_input tensor (where functionality
similar to scan, fold and map can be obtained). When more than one scan_input is used,
a behavior similar to zip is obtained.

The attribute body must be a graph, specifying the computation to be performed in
every iteration. It takes as input the current values of the state_variables and
the current iterated element of the scan_inputs. It must return the (updated) values
of the state_variables and zero or more scan_output_element tensors. The values of the
scan_output_element tensors are concatenated over all the iterations to produce the
scan_output values of the scan construct (similar to the concatenated intermediate
hidden-state values of RNN-like constructs). All the output tensors (state_variables as
well as scan_output_element tensors) are required to have the same shape in each iteration
of the loop (a restriction imposed to enable efficient memory allocation).

Note that the iterated element passed to the body subgraph does not have a sequence
axis. It will have a rank one less than the rank of the corresponding scan_input.

The scan operation returns the final values of the state_variables as well as the
scan_outputs.

The optional attribute scan_input_directions specifies the direction (forward or backward)
for each scan input. If this attribute is omitted, all sequences are scanned in the forward
direction. A bidirectional scan may be performed by specifying the same tensor input twice
in the scan_inputs, once with a forward direction, and once with a backward direction.

The scan_output of the operation is produced by concatenating the scan_output_element
values produced by the body in each iteration.  The optional attribute scan_output_directions
specifies the direction in which scan_output is constructed (by appending or prepending the
scan_output_element to scan_output in each iteration) for each scan_output. If this attribute
is omitted, the scan_output_element is appended to the scan_output in each iteration.

The optional attribute scan_input_axes specifies the axis to be scanned for each scan_input.
If omitted, every scan_input will be scanned in axis 0. For example, if axis 0 is the
batch axis and axis 1 is the time axis (to be scanned), specify an axis value of 1.
Note that scanning a non-zero axis may be less efficient than scanning axis zero.

The optional attribute scan_output_axes specifies the axis along which the scan_outputs
are accumulated for each scan_output. For example, if axis 1 is the time axis (to be
scanned) for both inputs and outputs, specify a scan_input axis and scan_output axis
value of 1.

Note that because of the ONNX restriction that only the last parameter of an operator can
be variadic, the initial-states and scan-inputs are listed together as one input parameter.
Similarly, the final-states and scan-outputs are listed together as one output parameter.
The attribute num_scan_inputs indicates the number M of scan-inputs.

The behavior of

    Scan <
        num_scan_inputs = m,
        body = loop-body,
        scan_input_axes = [axis_1, ..., axis_m]
    > (init_1, ..., init_n, scan_1, ..., scan_m)

is equivalent to the following pseudo-code:

    // scan_i.shape[axis_i] denotes the (max) sequence-length of scan_i
    // scan_i.shape[axis_i] is required to be equal to scan_j.shape[axis_j] for all i,j.
    sequence_length = scan_1.shape[axis_1];

    // initialize state-variables
    st_1 = init_1; ... st_n = init_n;
    // initialize scan-output variables: [] denotes an empty tensor
    scan_out_1 = []; ...; scan_out_k = [];
    // identify number of iterations:

    // execute loop
    for (int t = 0; t < sequence_length; ++t) {
        // generate the scan-input elements: the notation T<axis=k>[t] indicates the sub-tensor
        // of rank one less than T obtained by indexing T at position t along axis k.
        si_1 = scan_1<axis=axis_1>[t];
        ... ;
        si_m = scan_m<axis=axis_m>[t];
        // execute loop-body
        st_1, ..., st_n, so_1, ..., so_k = loop-body(st_1, ..., st_n, si_1, ..., si_m)
        // accumulate the scan-output elements
        scan_out_1 = Concat<axis=0>(scan_out_1, so_1); ... ; scan_out_k = Concat<axis=0>(scan_out_k, so_k);
    }

    return st_1, ..., st_n, scan_out_1, ..., scan_out_k;

*Sample usage: Encoding RNN using a Scan*

The following example shows how a simple RNN over an input tensor %X, with weight tensor %Wi,
recurrence weight tensor %Ri, bias tensors %Wbi and %Rbi, and initial hidden-state %H_0 can
be encoded as a ScanLoop. Note that the loop-body is a nested graph, and it directly computes
%Wi, %Ri, %Wbi, and %Rbi (typically constants or initializers in the body graph). If these
values are computed in the outer graph, they need to be passed in as extra state_variables.

    graph rnn-encoding {
      %H_0 = ...
      %X = ...
      %Y_h, %Y = Scan[body = <graph rnn-cell-1>, num_scan_inputs=1](%H_0, %X)
      return %Y, %Y_h
    }

    graph rnn-cell-1 (
      %H_tminus1[FLOAT, tensor]
      %X_t[FLOAT, tensor]
    ) {
      %Wi = ...
      %Ri = ...
      %Wbi = ...
      %Rbi = ...
      %t1 = X_t * (Wi^T)
      %t2 = H_tminus1*(Ri^T)
      %t3 = Add(%t1, %t2)
      %t4 = Add(%t3, %Wbi)
      %t5 = Add(%t4, %Rbi)
      %Ht = Tanh(%t5)
      %Accumulate = Identity(%Ht)
      return %Ht, %Accumulate
    }


Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasOnnxSubgraphOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ResultTypeInferenceOpInterface`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>num_scan_inputs</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>scan_input_axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>scan_input_directions</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>scan_output_axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>scan_output_directions</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `initial_state_and_scan_inputs` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `final_state_and_scan_outputs` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values

### `onnx.ScatterElements` (ONNXScatterElementsOp)

_ONNX ScatterElements operation_

ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

`reduction` allows specification of an optional reduction operation, which is applied to all values in `updates`
tensor into `output` at the specified `indices`.
In cases where `reduction` is set to \"none\", indices should not have duplicate entries: that is, if idx1 != idx2,
then indices[idx1] != indices[idx2]. For instance, in a 2-D tensor case, the update
corresponding to the [i][j] entry is performed as below:
```
output[indices[i][j]][j] = updates[i][j] if axis = 0,
output[i][indices[i][j]] = updates[i][j] if axis = 1,
```
When `reduction` is set to some reduction function `f`, the update corresponding to the [i][j] entry is performed as below:
```
output[indices[i][j]][j] = f(output[indices[i][j]][j], updates[i][j]) if axis = 0,
output[i][indices[i][j]] = f(output[i][indices[i][j]], updates[i][j]) if axis = 1,
```
where the `f` is `+`, `*`, `max` or `min` as specified.

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

(Opset 18 change): Adds max/min to the set of allowed reduction ops.

Example 1:
```
data = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
]
indices = [
    [1, 0, 2],
    [0, 2, 1],
]
updates = [
    [1.0, 1.1, 1.2],
    [2.0, 2.1, 2.2],
]
output = [
    [2.0, 1.1, 0.0]
    [1.0, 0.0, 2.2]
    [0.0, 2.1, 1.2]
]
```
Example 2:
```
data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
indices = [[1, 3]]
updates = [[1.1, 2.1]]
axis = 1
output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>reduction</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `indices` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `updates` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.ScatterND` (ONNXScatterNDOp)

_ONNX ScatterND operation_

ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1,
and `updates` tensor of rank q + r - indices.shape[-1] - 1. The output of the operation
is produced by creating a copy of the input `data`, and then updating its value to values
specified by `updates` at specific index positions specified by `indices`. Its output shape
is the same as the shape of `data`.

`indices` is an integer tensor. Let k denote indices.shape[-1], the last dimension in the shape of `indices`.
`indices` is treated as a (q-1)-dimensional tensor of k-tuples, where each k-tuple is a partial-index into `data`.
Hence, k can be a value at most the rank of `data`. When k equals rank(data), each update entry specifies an
update to a single element of the tensor. When k is less than rank(data) each update entry specifies an
update to a slice of the tensor. Index values are allowed to be negative, as per the usual
convention for counting backwards from the end, but are expected in the valid range.

`updates` is treated as a (q-1)-dimensional tensor of replacement-slice-values. Thus, the
first (q-1) dimensions of updates.shape must match the first (q-1) dimensions of indices.shape.
The remaining dimensions of `updates` correspond to the dimensions of the
replacement-slice-values. Each replacement-slice-value is a (r-k) dimensional tensor,
corresponding to the trailing (r-k) dimensions of `data`.  Thus, the shape of `updates`
must equal indices.shape[0:q-1] ++ data.shape[k:r-1], where ++ denotes the concatenation
of shapes.

The `output` is calculated via the following equation:

```
output = np.copy(data)
update_indices = indices.shape[:-1]
for idx in np.ndindex(update_indices):
    output[indices[idx]] = updates[idx]
```

The order of iteration in the above loop is not specified.
In particular, indices should not have duplicate entries: that is, if idx1 != idx2, then indices[idx1] != indices[idx2].
This ensures that the output value does not depend on the iteration order.

`reduction` allows specification of an optional reduction operation, which is applied to all values in `updates`
tensor into `output` at the specified `indices`.
In cases where `reduction` is set to \"none\", indices should not have duplicate entries: that is, if idx1 != idx2,
then indices[idx1] != indices[idx2]. This ensures that the output value does not depend on the iteration order.
When `reduction` is set to some reduction function `f`, `output` is calculated as follows:

```
output = np.copy(data)
update_indices = indices.shape[:-1]
for idx in np.ndindex(update_indices):
    output[indices[idx]] = f(output[indices[idx]], updates[idx])
```

where the `f` is `+`, `*`, `max` or `min` as specified.

This operator is the inverse of GatherND.

(Opset 18 change): Adds max/min to the set of allowed reduction ops.

Example 1:
```
data    = [1, 2, 3, 4, 5, 6, 7, 8]
indices = [[4], [3], [1], [7]]
updates = [9, 10, 11, 12]
output  = [1, 11, 3, 10, 9, 6, 7, 12]
```

Example 2:
```
data    = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
indices = [[0], [2]]
updates = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]
output  = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>reduction</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `indices` | tensor of 64-bit signless integer values
| `updates` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Scatter` (ONNXScatterOp)

_ONNX Scatter operation_

This operator is deprecated. Please use ScatterElements, which provides the same functionality.

Scatter takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry
is performed as below:
```
  output[indices[i][j]][j] = updates[i][j] if axis = 0,
  output[i][indices[i][j]] = updates[i][j] if axis = 1,
```

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

Example 1:
```
  data = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
  ]
  indices = [
      [1, 0, 2],
      [0, 2, 1],
  ]
  updates = [
      [1.0, 1.1, 1.2],
      [2.0, 2.1, 2.2],
  ]
  output = [
      [2.0, 1.1, 0.0]
      [1.0, 0.0, 2.2]
      [0.0, 2.1, 1.2]
  ]
```
Example 2:
```
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `indices` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `updates` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Selu` (ONNXSeluOp)

_ONNX Selu operation_

Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>alpha</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>gamma</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.SequenceAt` (ONNXSequenceAtOp)

_ONNX SequenceAt operation_

Outputs a tensor copy from the tensor at 'position' in 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input_sequence` | SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values
| `position` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `tensor` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.SequenceConstruct` (ONNXSequenceConstructOp)

_ONNX SequenceConstruct operation_

Construct a tensor sequence containing 'inputs' tensors.
All tensors in 'inputs' must have the same data type.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `inputs` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output_sequence` | SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values

### `onnx.SequenceEmpty` (ONNXSequenceEmptyOp)

_ONNX SequenceEmpty operation_

Construct an empty tensor sequence, with given data type.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>dtype</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values

### `onnx.SequenceErase` (ONNXSequenceEraseOp)

_ONNX SequenceErase operation_

Outputs a tensor sequence that removes the tensor at 'position' from 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it erases the last tensor from 'input_sequence'.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input_sequence` | SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values
| `position` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output_sequence` | SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values

### `onnx.SequenceInsert` (ONNXSequenceInsertOp)

_ONNX SequenceInsert operation_

Outputs a tensor sequence that inserts 'tensor' into 'input_sequence' at 'position'.
'tensor' must have the same data type as 'input_sequence'.
Accepted range for 'position' is in `[-n, n]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it inserts 'tensor' to the back of 'input_sequence'.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input_sequence` | SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values
| `tensor` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `position` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output_sequence` | SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values

### `onnx.SequenceLength` (ONNXSequenceLengthOp)

_ONNX SequenceLength operation_

Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input_sequence` | SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values

#### Results:

| Result | Description |
| :----: | ----------- |
| `length` | tensor of 64-bit signless integer values

### `onnx.SequenceMap` (ONNXSequenceMapOp)

_ONNX SequenceMap operation_

Applies a sub-graph to each sample in the input sequence(s).

Inputs can be either tensors or sequences, with the exception of the first input which must
be a sequence. The length of the first input sequence will determine the number of samples in the
outputs. Any other sequence inputs should have the same number of samples. The number of inputs
and outputs, should match the one of the subgraph.

For each i-th element in the output, a sample will be extracted from the input sequence(s) at
the i-th position and the sub-graph will be applied to it.
The outputs will contain the outputs of the sub-graph for each sample, in the same order as in
the input.

This operator assumes that processing each sample is independent and could executed in parallel
or in any order. Users cannot expect any specific ordering in which each subgraph is computed.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `HasOnnxSubgraphOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input_sequence` | SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values
| `additional_inputs` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values

#### Results:

| Result | Description |
| :----: | ----------- |
| `out_sequence` | variadic of SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values

### `onnx.Shape` (ONNXShapeOp)

_ONNX Shape operation_

Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
Optional attributes start and end can be used to compute a slice of the input tensor's shape.
If start axis is omitted, the slice starts from axis 0.
The end axis, if specified, is exclusive (and the returned value will not include the size of that axis).
If the end axis is omitted, the axes upto the last one will be included.
Negative axes indicate counting back from the last axis.
Note that axes will be clamped to the range [0, r-1], where r is the
rank of the input tensor if they are out-of-range (after adding r in the case of
negative axis). Thus, specifying any end value > r is equivalent to specifying an end
value of r, and specifying any start value < -r is equivalent to specifying a start
value of 0.

Examples:

```
Input tensor with shape: [2, 3, 4]
No attributes specified.
Output: [2, 3, 4]
```

```
Input tensor with shape: [2, 3, 4]
start: -1
Output: [4]
```

```
Input tensor with shape: [2, 3, 4]
end: -1
Output: [2, 3]
```

```
Input tensor with shape: [2, 3, 4]
start: 1
end: 2
Output: [3]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>end</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>start</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `shape` | tensor of 64-bit signless integer values

### `onnx.ShapeTransform` (ONNXShapeTransformOp)

_ONNX Element-wise shape transformation operation_

This operator transforms a tensor into another tensor whose shape is changed
by a given affine map. This is elemement-wise transformation, so each element
in the input will be copied to an element in the output via the affine map.
The affine map must be bijective.

For example, the following code is using `onnx.ShapeTransform` to reshape
a tensor from 2D to 4D.
```mlir
#reshape = affine_map(d0, d1) -> (d0/32, d0%32, d1/64, d1%64)
%Y = onnx.ShapeTransform(%arg0) {index_map = #reshape} :  (tensor<128x128xf32>) -> tensor<4x32x2x64xf32>
```

`onnx.ShapeTransform` will be finally materialized into an `affine.for` via
lowering to `krnl` dialect, e.g.
```mlir
%alloc = memref.alloc() {alignment = 16 : i64} : memref<4x32x2x64xf32>
affine.for %arg1 = 0 to 128 {
  affine.for %arg2 = 0 to 128 {
    %0 = affine.load %arg0[%arg1, %arg2] : memref< 128x128xf32 >
    affine.store %0, %alloc[%arg1 / 32, %arg1 % 32, %arg2 / 64, %arg2 % 64] : memref<4x32x2x64xf32>
  }
}
```

When being canonicalized, ShapeTransform operations are composed into
a new ShapeTransform operation by composing their affine maps.

At this moment, this operation only supports static dimensions.

This operation is not part of the standard and was added to assist onnx-mlir.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>index_map</code></td><td>::mlir::AffineMapAttr</td><td>AffineMap attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 32-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 32-bit float values

### `onnx.Shrink` (ONNXShrinkOp)

_ONNX Shrink operation_

Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
having same datatype and shape with input. It has two attributes, lambd and
bias. The formula of this operator is: If x < -lambd, y = x + bias;
If x > lambd, y = x - bias; Otherwise, y = 0.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>bias</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
<tr><td><code>lambd</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Sigmoid` (ONNXSigmoidOp)

_ONNX Sigmoid operation_

Sigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the
tensor elementwise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Sign` (ONNXSignOp)

_ONNX Sign operation_

Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Sin` (ONNXSinOp)

_ONNX Sin operation_

Calculates the sine of the given input tensor, element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Sinh` (ONNXSinhOp)

_ONNX Sinh operation_

Calculates the hyperbolic sine of the given input tensor element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Size` (ONNXSizeOp)

_ONNX Size operation_

Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `size` | tensor of 64-bit signless integer values

### `onnx.Slice` (ONNXSliceOp)

_ONNX Slice operation_

Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slice#slicing-and-striding

Slice uses the `starts`, `ends`, `axes` and `steps` inputs to select a sub-tensor
of its input `data` tensor.

An effective `starts[i]`, `ends[i]`, and `steps[i]` must be computed for each `i`
in `[0, ... r-1]` where `r = rank(input)` as follows:

If `axes` are omitted, they are set to `[0, ..., r-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`

The effective values are initialized as `start[i] = 0`, `ends[i] = dims[i]` where
`dims` are the dimensions of `input` and `steps[i] = 1`.

All negative elements of `axes` are made non-negative by adding `r` to them, where
`r =rank(input)`.

All negative values in `starts[i]` and `ends[i]` have `dims[axes[i]]` added to them,
where `dims` are the dimensions of `input`. Then `start[axes[i]]` is the adjusted
`starts[i]` is clamped into the range `[0, dims[axes[i]]]` for positive stepping
and `[0, dims[axes[i]]-1]` for negative stepping.

The clamping for the adjusted `ends[i]` depends on the sign of `steps[i]` and must
accommodate copying 0 through `dims[axes[i]]` elements, so for positive stepping
`ends[axes[i]]` is clamped to `[0, dims[axes[i]]]`, while for negative stepping it
is clamped to `[-1, dims[axes[i]]-1]`.

Finally, `steps[axes[i]] = steps[i]`.

For slicing to the end of a dimension with unknown size, it is recommended to pass
in `INT_MAX` when slicing forward and 'INT_MIN' when slicing backward.

Example 1:

```
data = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]
axes = [0, 1]
starts = [1, 0]
ends = [2, 3]
steps = [1, 2]
result = [
    [5, 7],
]
```

Example 2:

```
data = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]
starts = [0, 1]
ends = [-1, 1000]
result = [
    [2, 3, 4],
]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `starts` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `ends` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `axes` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or none type
| `steps` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.SoftmaxCrossEntropyLoss` (ONNXSoftmaxCrossEntropyLossOp)

_ONNX SoftmaxCrossEntropyLoss operation_

Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

* shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
  with K >= 1 in case of K-dimensional loss.
* shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
  with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can calculated as follows:
```
l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
```
or
```
l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.
```

loss is zero for the case when label-value equals ignore_index.
```
l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index
```

where:
```
p = Softmax(scores)
y = Log(p)
c = labels[i][d1][d2]...[dk]
```

Finally, L is optionally reduced:

* If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
* If reduction = 'sum', the output is scalar: Sum(L).
* If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: `ReduceSum(L) / ReduceSum(W)`,
  where tensor W is of shape `(N, D1, D2, ..., Dk)` and `W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]]`.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>ignore_index</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>reduction</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `scores` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `labels` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values
| `weights` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `log_prob` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values or none type

### `onnx.Softmax` (ONNXSoftmaxOp)

_ONNX Softmax operation_

The operator computes the normalized exponential values for the given input:

 Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1) 

The \"axis\" attribute indicates the dimension along which Softmax
will be performed. The output tensor has the same shape
and contains the Softmax values of the corresponding input.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.SoftmaxV11` (ONNXSoftmaxV11Op)

_ONNX Softmax operation_

The operator computes the softmax (normalized exponential) values for each layer in the batch
 of the given input.

The input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1\}\] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1\}\]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1\}\], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors. The output tensor has the same shape
and contains the softmax values of the corresponding input.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Softplus` (ONNXSoftplusOp)

_ONNX Softplus operation_

Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Softsign` (ONNXSoftsignOp)

_ONNX Softsign operation_

Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.SpaceToDepth` (ONNXSpaceToDepthOp)

_ONNX SpaceToDepth operation_

SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>blocksize</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Split` (ONNXSplitOp)

_ONNX Split operation_

Split a tensor into a list of tensors, along the specified 'axis'.
Either input 'split' or the attribute 'num_outputs' should be specified, but not both.
If the attribute 'num_outputs' is specified, then the tensor is split into equal sized parts.
If the tensor is not evenly splittable into `num_outputs`, the last chunk will be smaller.
If the input 'split' is specified, it indicates the sizes of each output in the split.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>num_outputs</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `split` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `outputs` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.SplitToSequence` (ONNXSplitToSequenceOp)

_ONNX SplitToSequence operation_

Split a tensor into a sequence of tensors, along the specified 'axis'.
Lengths of the parts can be specified using the optional argument 'split'.
If the argument `split' is not specified, a default scalar value of 1
is used as the value of `split'.
'split' must contain only positive numbers.
'split' is either a scalar (tensor of empty shape), or a 1-D tensor.
If 'split' is a scalar, then 'input' will be split into chunks all of size 'split'
if possible. The last chunk alone may be smaller than 'split' if the 'input' size
along the given axis 'axis' is not divisible by 'split'.
If 'split' is a 1-dimensional tensor, the input tensor is split into 'size(split)' chunks,
with lengths of the parts on 'axis' specified in 'split'. In this scenario, the sum of entries
in 'split' must be equal to the dimension size of input tensor on 'axis'.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>keepdims</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `split` | tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output_sequence` | SeqType of tensor of 8-bit unsigned integer values values or SeqType of tensor of 16-bit unsigned integer values values or SeqType of tensor of 32-bit unsigned integer values values or SeqType of tensor of 64-bit unsigned integer values values or SeqType of tensor of 8-bit signless integer values values or SeqType of tensor of 16-bit signless integer values values or SeqType of tensor of 32-bit signless integer values values or SeqType of tensor of 64-bit signless integer values values or SeqType of tensor of 16-bit float values values or SeqType of tensor of 32-bit float values values or SeqType of tensor of 64-bit float values values or SeqType of tensor of string type values values or SeqType of tensor of 1-bit signless integer values values or SeqType of tensor of complex type with 32-bit float elements values values or SeqType of tensor of complex type with 64-bit float elements values values

### `onnx.SplitV11` (ONNXSplitV11Op)

_ONNX Split operation_

Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>split</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `outputs` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.SplitV13` (ONNXSplitV13Op)

_ONNX Split operation_

Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using input 'split'.
Otherwise, the tensor is split to equal sized parts.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `split` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `outputs` | variadic of tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Sqrt` (ONNXSqrtOp)

_ONNX Sqrt operation_

Square root takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the square root is, y = x^0.5, is applied to
the tensor elementwise. If x is negative, then it will return NaN.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Squeeze` (ONNXSqueezeOp)

_ONNX Squeeze operation_

Remove single-dimensional entries from the shape of a tensor.
Takes an input `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values
| `axes` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `squeezed` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values

### `onnx.SqueezeV11` (ONNXSqueezeV11Op)

_ONNX Squeeze operation_

Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `squeezed` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.StringNormalizer` (ONNXStringNormalizerOp)

_ONNX StringNormalizer operation_

StringNormalization performs string operations for basic cleaning.
This operator has only one input (denoted by X) and only one output
(denoted by Y). This operator first examines the elements in the X,
and removes elements specified in \"stopwords\" attribute.
After removing stop words, the intermediate result can be further lowercased,
uppercased, or just returned depending the \"case_change_action\" attribute.
This operator only accepts [C]- and [1, C]-tensor.
If all elements in X are dropped, the output will be the empty value of string tensor with shape [1]
if input shape is [C] and shape [1, 1] if input shape is [1, C].

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>case_change_action</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>is_case_sensitive</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>locale</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>stopwords</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of string type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of string type values

### `onnx.Sub` (ONNXSubOp)

_ONNX Sub operation_

Performs element-wise binary subtraction (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values
| `B` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Sum` (ONNXSumOp)

_ONNX Sum operation_

Element-wise sum of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data_0` | variadic of tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

#### Results:

| Result | Description |
| :----: | ----------- |
| `sum` | tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of bfloat16 type values

### `onnx.Tan` (ONNXTanOp)

_ONNX Tan operation_

Calculates the tangent of the given input tensor, element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Tanh` (ONNXTanhOp)

_ONNX Tanh operation_

Calculates the hyperbolic tangent of the given input tensor element-wise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.TfIdfVectorizer` (ONNXTfIdfVectorizerOp)

_ONNX TfIdfVectorizer operation_

This transform extracts n-grams from the input sequence and save them as a vector. Input can
be either a 1-D or 2-D tensor. For 1-D input, output is the n-gram representation of that input.
For 2-D input, the output is also a  2-D tensor whose i-th row is the n-gram representation of the i-th input row.
More specifically, if input shape is [C], the corresponding output shape would be [max(ngram_indexes) + 1].
If input shape is [N, C], this operator produces a [N, max(ngram_indexes) + 1]-tensor.

In contrast to standard n-gram extraction, here, the indexes of extracting an n-gram from the original
sequence are not necessarily consecutive numbers. The discontinuity between indexes are controlled by the number of skips.
If the number of skips is 2, we should skip two tokens when scanning through the original sequence.
Let's consider an example. Assume that input sequence is [94, 17, 36, 12, 28] and the number of skips is 2.
The associated 2-grams are [94, 12] and [17, 28] respectively indexed by [0, 3] and [1, 4].
If the number of skips becomes 0, the 2-grams generated are [94, 17], [17, 36], [36, 12], [12, 28]
indexed by [0, 1], [1, 2], [2, 3], [3, 4], respectively.

The output vector (denoted by Y) stores the count of each n-gram;
Y[ngram_indexes[i]] indicates the times that the i-th n-gram is found. The attribute ngram_indexes is used to determine the mapping
between index i and the corresponding n-gram's output coordinate. If pool_int64s is [94, 17, 17, 36], ngram_indexes is [1, 0],
ngram_counts=[0, 0], then the Y[0] (first element in Y) and Y[1] (second element in Y) are the counts of [17, 36] and [94, 17],
respectively. An n-gram which cannot be found in pool_strings/pool_int64s should be ignored and has no effect on the output.
Note that we may consider all skips up to S when generating the n-grams.

The examples used above are true if mode is \"TF\". If mode is \"IDF\", all the counts larger than 1 would be truncated to 1 and
the i-th element in weights would be used to scale (by multiplication) the count of the i-th n-gram in pool. If mode is \"TFIDF\",
this operator first computes the counts of all n-grams and then scale them by the associated values in the weights attribute.

Only one of pool_strings and pool_int64s can be set. If pool_int64s is set, the input should be an integer tensor.
If pool_strings is set, the input must be a string tensor.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>max_gram_length</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>max_skip_count</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>min_gram_length</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>ngram_counts</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>ngram_indexes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>pool_int64s</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>pool_strings</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>weights</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of string type values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 32-bit float values

### `onnx.ThresholdedRelu` (ONNXThresholdedReluOp)

_ONNX ThresholdedRelu operation_

ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>alpha</code></td><td>::mlir::FloatAttr</td><td>32-bit float attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values

### `onnx.Tile` (ONNXTileOp)

_ONNX Tile operation_

Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `repeats` | tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.TopK` (ONNXTopKOp)

_ONNX TopK operation_

Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
shape [a_0, a_1, ..., a_{n-1\}\] and integer argument k, return two outputs:

* Value tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1\}\]
  which contains the values of the top k elements along the specified axis
* Index tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1\}\] which
  contains the indices of the top k elements (original indices from the input
  tensor).

* If \"largest\" is 1 (the default value) then the k largest elements are returned.
* If \"sorted\" is 1 (the default value) then the resulting k elements will be sorted.
* If \"sorted\" is 0, order of returned 'Values' and 'Indices' are undefined.

Given two equivalent values, this operator uses the indices along the axis as
a tiebreaker. That is, the element with the lower index will appear first.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>largest</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>sorted</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `K` | tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Values` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values
| `Indices` | tensor of 64-bit signless integer values

### `onnx.Transpose` (ONNXTransposeOp)

_ONNX Transpose operation_

Transpose the input tensor similar to numpy.transpose. For example, when
perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>perm</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `transposed` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values

### `onnx.TreeEnsembleClassifier` (ONNXTreeEnsembleClassifierOp)

_ONNX TreeEnsembleClassifier operation_

Tree Ensemble classifier.  Returns the top class for each of N inputs.<br>
    The attributes named 'nodes_X' form a sequence of tuples, associated by
    index into the sequences, which must all be of equal length. These tuples
    define the nodes.<br>
    Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
    A leaf may have multiple votes, where each vote is weighted by
    the associated class_weights index.<br>
    One and only one of classlabels_strings or classlabels_int64s
    will be defined. The class_ids are indices into this list.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>base_values</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>class_ids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>class_nodeids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>class_treeids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>class_weights</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>classlabels_int64s</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>classlabels_strings</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>nodes_falsenodeids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>nodes_featureids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>nodes_hitrates</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>nodes_missing_value_tracks_true</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>nodes_modes</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>nodes_nodeids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>nodes_treeids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>nodes_truenodeids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>nodes_values</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>post_transform</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of string type values or tensor of 64-bit signless integer values
| `Z` | tensor of 32-bit float values

### `onnx.TreeEnsembleRegressor` (ONNXTreeEnsembleRegressorOp)

_ONNX TreeEnsembleRegressor operation_

Tree Ensemble regressor.  Returns the regressed values for each input in N.<br>
    All args with nodes_ are fields of a tuple of tree nodes, and
    it is assumed they are the same length, and an index i will decode the
    tuple across these inputs.  Each node id can appear only once
    for each tree id.<br>
    All fields prefixed with target_ are tuples of votes at the leaves.<br>
    A leaf may have multiple votes, where each vote is weighted by
    the associated target_weights index.<br>
    All trees must have their node ids start at 0 and increment by 1.<br>
    Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>aggregate_function</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>base_values</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>n_targets</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>nodes_falsenodeids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>nodes_featureids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>nodes_hitrates</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>nodes_missing_value_tracks_true</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>nodes_modes</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
<tr><td><code>nodes_nodeids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>nodes_treeids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>nodes_truenodeids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>nodes_values</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
<tr><td><code>post_transform</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>target_ids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>target_nodeids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>target_treeids</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>target_weights</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values or tensor of 64-bit float values or tensor of 64-bit signless integer values or tensor of 32-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 32-bit float values

### `onnx.Trilu` (ONNXTriluOp)

_ONNX Trilu operation_

Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower triangular part of the tensor(s).
The attribute \"upper\" determines whether the upper or lower part is retained. If set to true,
the upper triangular matrix is retained. Lower triangular matrix is retained otherwise.
Default value for the \"upper\" attribute is true.
Trilu takes one input tensor of shape [*, N, M], where * is zero or more batch dimensions. The upper triangular part consists
of the elements on and above the given diagonal (k). The lower triangular part consists of elements on and below the diagonal.
All other elements in the matrix are set to zero.
If k = 0, the triangular part on and above/below the main diagonal is retained.
If upper is set to true, a positive k retains the upper triangular matrix excluding the main diagonal and (k-1) diagonals above it.
A negative k value retains the main diagonal and |k| diagonals below it.
If upper is set to false, a positive k retains the lower triangular matrix including the main diagonal and k diagonals above it.
A negative k value excludes the main diagonal and (|k|-1) diagonals below it.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>upper</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `k` | tensor of 64-bit signless integer values or none type

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Unique` (ONNXUniqueOp)

_ONNX Unique operation_

Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned.
Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.

This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs.
The first output tensor 'Y' contains all unique values or subtensors of the input.
The second optional output tensor 'indices' contains indices of 'Y' elements' first occurrence in 'X'.
The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'.
The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input.

Outputs are either sorted in ascending order or optionally in the order of the first occurrence of the values in the input.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html

Example 1:
```
input_X = [2, 1, 1, 3, 4, 3]
attribute_sorted = 0
attribute_axis = None
output_Y = [2, 1, 3, 4]
output_indices = [0, 1, 3, 4]
output_inverse_indices = [0, 1, 1, 2, 3, 2]
output_counts = [1, 2, 2, 1]
```

Example 2:
```
input_X = [[1, 3], [2, 3]]
attribute_sorted = 1
attribute_axis = None
output_Y = [1, 2, 3]
output_indices = [0, 2, 1]
output_inverse_indices = [0, 2, 1, 2]
output_counts = [1, 1, 2]
```

Example 3:
```
input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
attribute_sorted = 1
attribute_axis = 0
output_Y = [[1, 0, 0], [2, 3, 4]]
output_indices = [0, 2]
output_inverse_indices = [0, 0, 1]
output_counts = [2, 1]
```

Example 4:
```
input_x = [[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
            [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]]
attribute_sorted = 1
attribute_axis = 1
```

intermediate data are presented below for better understanding:
there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
```
A: [[1, 1], [1, 1]],
   [[0, 1], [0, 1]],
   [[2, 1], [2, 1]],
   [[0, 1], [0, 1]].
```

there are 3 unique subtensors:
```
[[1, 1], [1, 1]],
[[0, 1], [0, 1]],
[[2, 1], [2, 1]].
```

sorted unique subtensors:
```
B: [[0, 1], [0, 1]],
   [[1, 1], [1, 1]],
   [[2, 1], [2, 1]].
```

output_Y is constructed from B:
```
[[[0. 1.], [1. 1.], [2. 1.]],
 [[0. 1.], [1. 1.], [2. 1.]]]
```

output_indices is to map from B to A:
```
[1, 0, 2]
```

output_inverse_indices is to map from A to B:
```
[1, 0, 2, 0]
```

output_counts:
```
[2, 1, 1]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axis</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
<tr><td><code>sorted</code></td><td>::mlir::IntegerAttr</td><td>64-bit signed integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `indices` | tensor of 64-bit signless integer values or none type
| `inverse_indices` | tensor of 64-bit signless integer values or none type
| `counts` | tensor of 64-bit signless integer values or none type

### `onnx.Unsqueeze` (ONNXUnsqueezeOp)

_ONNX Unsqueeze operation_

Insert single-dimensional entries to the shape of an input tensor (`data`).
Takes one required input `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).

For example, given an input tensor (`data`) of shape [3, 4, 5], then
Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing same data as `data` but with shape [1, 3, 4, 5, 1].

The input `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1].
The order of values in `axes` does not matter and can come in any order.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values
| `axes` | tensor of 64-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `expanded` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values or tensor of f8E4M3FN type values or tensor of f8E4M3FNUZ type values or tensor of f8E5M2 type values or tensor of f8E5M2FNUZ type values or tensor of 4-bit unsigned integer values or tensor of 4-bit signless integer values

### `onnx.UnsqueezeV11` (ONNXUnsqueezeV11Op)

_ONNX Unsqueeze operation_

Insert single-dimensional entries to the shape of an input tensor (`data`).
Takes one required argument `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).

For example:
  Given an input tensor (`data`) of shape [3, 4, 5], then
  Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing same data as `data` but with shape [1, 3, 4, 5, 1].

The attribute `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1].
The order of values in `axes` does not matter and can come in any order.


Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>axes</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `expanded` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Upsample` (ONNXUpsampleOp)

_ONNX Upsample operation_

Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `scales` | tensor of 32-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.UpsampleV7` (ONNXUpsampleV7Op)

_ONNX Upsample operation_

Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>mode</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>scales</code></td><td>::mlir::ArrayAttr</td><td>32-bit float array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Where` (ONNXWhereOp)

_ONNX Where operation_

Return elements, either from X or Y, depending on condition.
Where behaves like
[numpy.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
with three parameters.

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `condition` | tensor of 1-bit signless integer values
| `X` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values
| `Y` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

#### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit unsigned integer values or tensor of 16-bit unsigned integer values or tensor of 32-bit unsigned integer values or tensor of 64-bit unsigned integer values or tensor of 8-bit signless integer values or tensor of 16-bit signless integer values or tensor of 32-bit signless integer values or tensor of 64-bit signless integer values or tensor of bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit float values or tensor of 64-bit float values or tensor of string type values or tensor of 1-bit signless integer values or tensor of complex type with 32-bit float elements values or tensor of complex type with 64-bit float elements values

### `onnx.Xor` (ONNXXorOp)

_ONNX Xor operation_

Returns the tensor resulted from performing the `xor` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `A` | tensor of 1-bit signless integer values
| `B` | tensor of 1-bit signless integer values

#### Results:

| Result | Description |
| :----: | ----------- |
| `C` | tensor of 1-bit signless integer values

### `onnx.Yield` (ONNXYieldOp)

_ONNX yield operation_


Syntax:

```
operation ::= `onnx.Yield` attr-dict ($operands^ `:` type($operands))?
```

The `onnx.Yield` operation represents a yield operation within an ONNX subgraph.
The operation takes variable number of operands and produces no results.

This operation is not part of the standard and was added to assist onnx-mlir.
It terminates a ONNXLoop/Scan/IfOp region.

Traits: `AlwaysSpeculatableImplTrait`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operands` | variadic of any type

### `onnx.ZipMap` (ONNXZipMapOp)

_ONNX ZipMap operation_

Creates a map from the input and the attributes.<br>
    The values are provided by the input tensor, while the keys are specified by the attributes.
    Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
    The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ShapeHelperOpInterface`, `ShapeInferenceOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>classlabels_int64s</code></td><td>::mlir::ArrayAttr</td><td>64-bit integer array attribute</td></tr>
<tr><td><code>classlabels_strings</code></td><td>::mlir::ArrayAttr</td><td>string array attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `X` | tensor of 32-bit float values

#### Results:

| Result | Description |
| :----: | ----------- |
| `Z` | SeqType of tuple with any combination of string type or 32-bit float values values or SeqType of tuple with any combination of 64-bit signless integer or 32-bit float values values



[./doc_check/README.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

# DocCheck

### Goal

It is always desirable to ensure that every piece of knowledge has a single, unambiguous, authoritative representation 
in our codebase. However, sometimes violating such principle can result in improved overall quality of the software
project. For instance, when we write documentation containing example code snippets, it is desirable to write tests
for them - however, if we do so, the same code example will exist both in documentation and in tests! Such duplication
of knowledge has tangible adverse consequences - when documentation is updated to new examples, tests become obsolete. 
Moreover, the discrepancy between multiple copies of the same knowledge (e.g., code example) can only be spotted with 
manual inspection.

Under such circumstances, to establish a single source of trough in an enforceable manner, we can turn to the DocCheck
tool. Simply put, DocCheck enforces the consistency constraints as specified by the users between textual artifacts in 
our codebase. Textual artifacts can be:
- Sections in documentation
- Content of a file
- Output of command execution
- ...

Specifically, DocCheck allows us to precisely specify how a textual artifact is derived from another. Such
specification is then parsed and verified by our software testing infrastructure to ensure the consistency between
derived textual artifact and the original one. This overall workflow provides an enforceable way to establish a single,
unambiguous and authoritative representation of knowledge in our codebase.

### Directives

Directives can be used to communicate the relationship between derived and original textual artifacts to DocCheck. 
DocCheck will perform consistency constraints checking according to the specification. In this section, supported 
directives are explained in details.

Currently, directives can be specified either in a Markdown file or in a standalone DocCheck configuration file (a file 
ending with `.dc` suffix). For markdown file, specify directive using the following syntax:

```markdown
[{directive}]: <> ({configuration})
```

For standalone DocCheck configuration file, use the following syntax:
```
{directive}({configuration})
```

where `{directive}` is the name of the directive and `{configuration}` expresses the specific 
parameters of this directive. In general, a directive configuration is expressed using a python dictionary literal, 
with supported configuration parameter name as keys and the desired state of configuration as values.

Special shorthands exist for each directive individually.

##### `same-as-file`:

Use `same-as-file` directive to ensure that the code section following this directive is the same as a source file.
This is useful primarily because testing code snippet in documentation directly is often impossible. However,
unit tests can be written utilizing an exact copy of the code snippet content. We can use `same-as-file` directive 
to ensure the code snippet is always the same as its copy used in some unit tests. 

`same-as-file` directive supports a convenient short-hand configuration format where the directive configuration can 
be fully specified using the name of the reference file to check against. For example, to ensure a code snippet is the 
same as a unit-tested file `reference.cpp`, use the following directive as shown in the documentation snippet:

[same-as-file]: <> ({"ref": "docs/doc_check/test/same-as-file/simple/README.md", "skip-ref": 2})
````markdown
Lorem ipsum dolor sit amet, consectetur adipiscing elit.

[same-as-file]: <> (reference.cpp)
```cpp
#include<iostream>

using namespace std;

int main() {
    cout<<"Hello World";
    return 0;
}
```
````

In the canonical form of directive configuration (as a python dictionary literal), this directive supports these parameters in it:

`ref` (string): reference file to check against.

`skip-doc` (int): number of lines to skip when checking the documentation.

`skip-ref` (int): number of lines to skip when scanning the reference file.

For example, to ensure the following code snippet is the same as a unit-tested file `reference.cpp`, except for the first 2 lines of the code used in documentation, and the first 3 lines of code used in the reference file, the following directive configuration can be used:

[same-as-file]: <> ({"ref": "docs/doc_check/test/same-as-file/skip-doc-ref/README.md", "skip-ref": 2})
````markdown
Lorem ipsum dolor sit amet, consectetur adipiscing elit.

[same-as-file]: <> ({"ref": "reference.cpp", "skip-doc": 2, "skip-ref": 3})
```cpp
// First line unique to documentation
// Second line unique to documentation
#include<iostream>

using namespace std;

int main() {
    cout<<"Hello World";
    return 0;
}
```
````

#### `file-same-as-stdout`

Use `file-same-as-stdout` to ensure that file content is the same as the output of executing a command.
This directive supports these parameters in it:

`file` (string): file to compare with.

`cmd` (List[str]): the command (expressed as a list of command components), e.g. `["ls", "-l"]`.

For example, to ensure that the content of a file `test.in`:

[same-as-file]: <> (docs/doc_check/test/file-same-as-stdout/success/test.in)
```
dog
```

is exactly the same as the output of command execution `echo dog`, one can use the following directive:
[same-as-file]: <> (docs/doc_check/test/file-same-as-stdout/success/test.in.dc)
```
file-same-as-stdout({"file": "test.in", "cmd": ["echo", "dog"]})
```

[./doc_check/test/same-as-file/simple/README.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

Lorem ipsum dolor sit amet, consectetur adipiscing elit.

[same-as-file]: <> (reference.cpp)
```cpp
#include<iostream>

using namespace std;

int main() {
    cout<<"Hello World";
    return 0;
}
```

[./doc_check/test/same-as-file/error-doc-different-from-ref/README.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

Lorem ipsum dolor sit amet, consectetur adipiscing elit.

[same-as-file]: <> (reference.cpp)
```cpp
#include<iostream>

int main() {
    cout<<"Hello World";
    return 0;
}
```

[./doc_check/test/same-as-file/skip-doc-ref/README.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

Lorem ipsum dolor sit amet, consectetur adipiscing elit.

[same-as-file]: <> ({"ref": "reference.cpp", "skip-doc": 2, "skip-ref": 3})
```cpp
// First line unique to documentation
// Second line unique to documentation
#include<iostream>

using namespace std;

int main() {
    cout<<"Hello World";
    return 0;
}
```

[./doc_check/test/same-as-file/error-doc-shorter-than-ref/README.md]:

<!--- SPDX-License-Identifier: Apache-2.0 -->

Lorem ipsum dolor sit amet, consectetur adipiscing elit.

[same-as-file]: <> (reference.cpp)
```cpp
#include<iostream>

```

[./mnist_example/README.md]:

# Table of Contents

- [Train Model in PyTorch, Compile using ONNX-MLIR](#train-model-in-pytorch-compile-using-onnx-mlir)
  - [Training the Model](#training-the-model)
  - [Environment Variables Setup](#environment-variables-setup)
  - [Compile Model](#compile-model)
  - [Write a C Driver Code](#write-a-C-driver-code)
    - [Inference Entry Point](#inference-entry-point)
    - [Feeding Inputs and Retrieving Results](#feeding-inputs-and-retrieving-results)
  - [Write a Python Driver Code](#write-a-Python-driver-code)
  - [Write a Java Driver Code](#write-a-java-driver-code)

# Train Model in PyTorch, Compile using ONNX-MLIR

In this example, we will demonstrate training a mnist model in PyTorch and compile, run it using only C++.

## Training the Model

An already trained [mnist.onnx](mnist.onnx) model is provided for your convenience.

If you want to train the model yourself, make sure that dependent python packages specified in `requirements.txt` are installed.
Run the training script using the following command:

```bash
./gen_mnist_onnx.py --epochs=1 --batch-size=128 --export-onnx --save-model
```

Which basically says, train the model for 1 epoch using a batch size of 128. Such configuration encourages a speedy training process.
The flag `--export-onnx` will export the trained model to an ONNX protobuf object.
The flag `--save-model` will save a snapshot of the trained model.

The model is a simple neural network defined as such:

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(14*14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(x, 2)
        x = x.reshape(-1, 1*14*14)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output
```

After training is complete, an onnx model named `mnist.onnx` should appear.
If you are interested in knowing how to export a pytorch model, here's the relevant code snippet:

```python
  model = Net()
#...
#Train...
#...
  input_names = ["image"]
  output_names = ["prediction"]
  dummy_input = torch.randn(1, 1, 28, 28)
  torch.onnx.export(model,
                    dummy_input,
                    "mnist.onnx",
                    verbose=True,
                    input_names=input_names,
                    output_names=output_names)
```

Upon inspection, it should look like:

![alt text](mnist-simple.png "Simple MNIST Model")

## Environment Variables Setup

Now we are ready to compile the model! To make it easier to invoke commands and include header files, I updated my environment variables as such:

```bash
#ONNX_MLIR_ROOT points to the root of the onnx-mlir,
#under which the include and the build directory lies.
export ONNX_MLIR_ROOT=$(pwd)/../..
#Define the bin directory where onnx-mlir binary resides.Change only if you
#have a non - standard install.
export ONNX_MLIR_BIN=$ONNX_MLIR_ROOT/build/Debug/bin
#Define the include directory where onnx-mlir runtime include files resides.
#Change only if you have a non - standard install.
export ONNX_MLIR_INCLUDE=$ONNX_MLIR_ROOT/include

#Include ONNX-MLIR executable directories part of $PATH.
export PATH=$ONNX_MLIR_ROOT/build/Debug/bin:$PATH

#Compiler needs to know where to find its                                      \
    runtime.Set ONNX_MLIR_RUNTIME_DIR to proper path.
export ONNX_MLIR_RUNTIME_DIR=../../build/Debug/lib
```

You may also simply execute `chmod +x update_env.sh` and `./update_env.sh` for the above commands directly in the docs/docs/mnist_example and everything should work fine.

## Compile Model

To compile the model into a shared library that can be used with C/C++ and Python drivers, we invoke `onnx-mlir` with the `-EmitLib` option (it can be omitted since it's the default):

```bash
onnx-mlir -O3 [-EmitLib] mnist.onnx
```

A `mnist.so` should appear, which corresponds to the compiled model object file. An example to compile the model via Python interface is also provided. You could also run `python3 mnist-compile.py` and you will see a `mnist.so` appears as well.

To compile the model into a jar archive that can be used with Java drivers, we invoke `onnx-mlir` with the `-EmitJNI` option:

```bash
onnx-mlir -O3 -EmitJNI mnist.onnx
```

A `mnist.jar` should appear, which corresponds to the compiled model object file along with Java API classes.

## Multi-threading

onnx-mlir provides a multi-thread safe parallel compilation mode. Whether each thread is given a name or not by the user, onnx-mlir is multi-threaded safe. If you would like to give a name to a thread, use the `-customEnvFlags` keyword and an example can be found as follows.

```bash
export MNIST_WITH_O3="-O3"
onnx-mlir -O3 -customEnvFlags=MNIST_WITH_O3 [--EmitLib] mnist.onnx -o mnist03
```

A multi-threaded experiment from command line written in Python is provided.

```python
import datetime
import os
import threading

def execCmd(cmd):
    try:
        print("command " + cmd + " starts at " + str(datetime.datetime.now()))
        os.system(cmd)
        print("command " + cmd + " is finished at " + str(datetime.datetime.now()))
    except:
        print("command " + cmd + " meets errors")

if __name__ == '__main__':

#define 2 different commands
    cmds = ['onnx-mlir -O3 mnist.onnx -o mnist03','onnx-mlir -O1 mnist.onnx -o mnist01']

    threads = []

    print("program starts at " + str(datetime.datetime.now()))

#run the commands
    for cmd in cmds:
        th = threading.Thread(target=execCmd, args=(cmd,))
        th.start()
        threads.append(th)

#wait for all the commands finish
    for th in threads:
        th.join()

    print("program is finished at " + str(datetime.datetime.now()))
```

You can execute `python3 multi-threading-test.py` under the current directory to test.

## Write a C Driver Code

Documentation of the APIs are found [here](https://onnx.ai/onnx-mlir), with the C interface for Tensor [here](https://onnx.ai/onnx-mlir/doxygen_html/OMTensor_h/_o_m_tensor_8h.html) and TensorList [here](https://onnx.ai/onnx-mlir/doxygen_html/OMTensorList_h/_o_m_tensor_list_8h.html).

To invoke the compiled model, we need to know the entry point signature with which to call into the model inference function, and based on it, engineer a C++ driver that feeds test data into this inference function and retrieve the prediction results.

### Inference Entry Point

The signature of the model inference function for all models is:

```cpp
extern "C" OMTensorList *run_main_graph(OMTensorList *);
```

I.e., all models ingests an `OMTensorList*`, and returns an `OMTensorList*`. Documentation of the APIs are found [here](https://onnx.ai/onnx-mlir/doxygen_html/OnnxMlirRuntime/index.html), with the C interface for Tensor [here](https://onnx.ai/onnx-mlir/doxygen_html/OMTensor_h/_o_m_tensor_8h.html) and TensorList [here](https://onnx.ai/onnx-mlir/doxygen_html/OMTensorList_h/_o_m_tensor_list_8h.html).

### Feeding Inputs and Retrieving Results

To invoke the inference function, we use the following code to communicate with the compiled model inference function.

```cpp
#include <iostream>
#include <vector>

#include "OnnxMlirRuntime.h"

// Declare the inference entry point.
extern "C" OMTensorList *run_main_graph(OMTensorList *);

static float img_data[] = {...};

int main() {
  // Create an input tensor list of 1 tensor.
  int inputNum = 1;
  OMTensor *inputTensors[inputNum];
  // The first input is of tensor<1x1x28x28xf32>.
  int64_t rank = 4;
  int64_t shape[] = {1, 1, 28, 28};

  // Create a tensor using omTensorCreateWithOwnership (returns a pointer to the OMTensor).
  // When the parameter, owning is set to "true", the OMTensor will free the data
  // pointer (img_data) upon destruction. If owning is set to false, the data pointer will
  // not be freed upon destruction.
  OMTensor *tensor = omTensorCreateWithOwnership(img_data, shape, rank, ONNX_TYPE_FLOAT, /*owning=*/false);

  // Create a tensor list using omTensorListCreate (returns a pointer to the OMTensorList).
  inputTensors[0] = tensor;
  OMTensorList *tensorListIn = omTensorListCreate(inputTensors, inputNum);

  // Compute outputs.
  OMTensorList *tensorListOut = run_main_graph(tensorListIn);

  // Extract the output. The model defines one output of type tensor<1x10xf32>.
  OMTensor *y = omTensorListGetOmtByIndex(tensorListOut, 0);
  float *prediction = (float *)omTensorGetDataPtr(y);

  // Analyze the output.
  int digit = -1;
  float prob = 0.;
  for (int i = 0; i < 10; i++) {
    printf("prediction[%d] = %f\n", i, prediction[i]);
    if (prediction[i] > prob) {
      digit = i;
      prob = prediction[i];
    }
  }
  // The OMTensorListDestroy will free all tensors in the OMTensorList
  // upon destruction. It is important to note, that every tensor will
  // be destroyed. To free the OMTensorList data structure but leave the
  // tensors as is, use OMTensorListDestroyShallow instead.
  omTensorListDestroy(tensorListOut);
  omTensorListDestroy(tensorListIn);

  printf("The digit is %d\n", digit);
  return 0;
}
```

Now, putting everything together, we invoke g++ to compile and link together the driver code, C runtime API and the compiled model inference function:

```bash
g++ --std=c++11 -O3 mnist.cpp ./mnist.so -o mnist -I $ONNX_MLIR_INCLUDE
```

Now run it by calling `./mnist`! It outputs the following for the image in the test:

```shell
prediction[0] = 1.000000
prediction[1] = 0.000000
prediction[2] = 0.000000
prediction[3] = 0.000000
prediction[4] = 0.000000
prediction[5] = 0.000000
prediction[6] = 0.000000
prediction[7] = 0.000000
prediction[8] = 0.000000
prediction[9] = 0.000000
The digit is 0.
```

The full code is available [here](mnist.cpp).

## Write a Python Driver Code

You will find most of the details of the Python driver interface described [here](https://onnx.ai/onnx-mlir/UsingPyRuntime.html). We summarize here quickly how to execute mnist in python.

First, we include the necessary Python runtime library. The library path can be set by using the PYTHONPATH or simply creating a soft link in the current directory to the Python shared library (typically: `build/Debug/lib/PyRuntime.cpython-<target>.so`).

```Python
import numpy as np
from PyRuntime import OMExecutionSession
```

The runtime use an `OMExecutionSession` object to hold a specific model and entry point. On this object, we can perform in inference using the `run(input)` call where `input` is a list of numpy arrays. The signature of the input and output model can be extracted using, respectively, the `input_signature()` and `output_signature()` formatted as JSON strings. The code is shown below.

```Python
#Load the model mnist.so compiled with onnx-mlir.
session = OMExecutionSession('mnist.so')
#Print the models input / output signature, for display.
#If there are problems with the signature functions,                           \
    they can be simply commented out.
print("input signature in json", session.input_signature())
print("output signature in json", session.output_signature())
#Create an input arbitrarily filled of 1.0 values(file has the actual values).
input = np.full((1, 1, 28, 28), 1, np.dtype(np.float32))
#Run the model.It is best to always use the[] around the inputs as the inputs
#are an vector of numpy arrays.
outputs = session.run([input])
```

The outputs can then be analyzed by inspecting the values inside the `output` list of numpy arrays.

The full code is available [here](mnist-runPyRuntime.py). It finds that `0` is the most likely digit for the given input. The command is:

```shell
./mnist-runPyRuntime.py
```

and produces an output similar to the following (you may see slightly different prediction numbers if you train the model yourself):

```shell
input signature in json [    { "type" : "f32" , "dims" : [1 , 1 , 28 , 28] , "name" : "image" }
]
output signature in json [   { "type" : "f32" , "dims" : [1 , 10] , "name" : "prediction" }
]
prediction  0 = 0.9999999
prediction  1 = 6.470239e-17
prediction  2 = 5.3113327e-09
prediction  3 = 2.3830837e-10
prediction  4 = 1.54674e-15
prediction  5 = 1.6361314e-07
prediction  6 = 2.7768482e-11
prediction  7 = 8.211209e-13
prediction  8 = 2.9605862e-09
prediction  9 = 8.650948e-15
The digit is 0
```

We provide two additional Python interfaces. 
The second interface extends the above execution session by simply compiling a model before loading it for execution (see [here](mnist-runPyCompileAndRuntime.py)). 
The user simply passes the `.onnx` model and the flags needed to compile the model.
Unless explicitly disabled by the `reuse_compiled_model=0`, the execution session will reuse a previously compiled model whose name matches the name the output file generated by the compiler. 
Note that the execution session does not check if the cached version was compiled using identical compiler flags; it is the responsibility of the user to then clear the cached version, or disable the reuse using the provided optional flag.

For example, the code below will compile and load the `mnist.onnx` model, compiling only when the `mnist2.so` binary file cannot be located. Model inference can then proceed using the `session.run(...)` command.

```Python
# Load onnx model and create CompileExecutionSession object,
# by first compiling the mnist.onnx model with the "-O3" options.
session = OMCompileExecutionSession("./mnist.onnx" ,"-O3 -o=mnist2")
```

The third interface provides a simple interface to explicitly compile an onnx model (see [here](mnist-compile.py)). 

## Write a Java Driver Code

Inference APIs and data structures for Java closely mirror those for C/C++. Documentation of the APIs are found [here](https://onnx.ai/onnx-mlir/doxygen_html/OMModel_java/classcom_1_1ibm_1_1onnxmlir_1_1_o_m_model.html), with the Java interface for Tensor [here](https://onnx.ai/onnx-mlir/doxygen_html/OMTensor_java/classcom_1_1ibm_1_1onnxmlir_1_1_o_m_tensor.html) and TensorList [here](https://onnx.ai/onnx-mlir/doxygen_html/OMTensorList_java/classcom_1_1ibm_1_1onnxmlir_1_1_o_m_tensor_list.html).

An example Java driver for the mnist model is given below.

```java
import com.ibm.onnxmlir.OMModel;
import com.ibm.onnxmlir.OMTensor;
import com.ibm.onnxmlir.OMTensorList;

public class Mnist {
  static float[] img_data = {...};

public
  static void main(String[] args) {
    // Create an input tensor list of 1 tensor.
    // The first input is of tensor<1x1x28x28xf32>.
    OMTensor tensor = new OMTensor(img_data, new long[]{1, 1, 28, 28});
    OMTensor[] inputTensors = new OMTensor[]{tensor};
    // Create a tensor list
    OMTensorList tensorListIn = new OMTensorList(inputTensors);

    // Compute outputs.
    OMTensorList tensorListOut = OMModel.mainGraph(tensorListIn);

    // Extract the output. The model defines one output of type
    // tensor<1x10xf32>.
    OMTensor y = tensorListOut.getOmtByIndex(0);
    float[] prediction = y.getFloatData();

    // Analyze the output.
    int digit = -1;
    float prob = 0.f;
    for (int i = 0; i < prediction.length; i++) {
      System.out.println("prediction[" + i + "] = " + prediction[i]);
      if (prediction[i] > prob) {
        digit = i;
        prob = prediction[i];
      }
    }

    System.out.println("The digit is " + digit);
  }
}
```

The full code is [here](Mnist.java).

To compile the driver:

```
javac -cp mnist.jar Mnist.java
```

To run the driver:

```
java -cp .:mnist.jar Mnist
```

It should produce an output similar to the following:

```
prediction[0] = 0.9999999
prediction[1] = 6.470239E-17
prediction[2] = 5.3113327E-9
prediction[3] = 2.3830837E-10
prediction[4] = 1.54674E-15
prediction[5] = 1.6361314E-7
prediction[6] = 2.7768482E-11
prediction[7] = 8.211209E-13
prediction[8] = 2.9605862E-9
prediction[9] = 8.650948E-15
The digit is 0
```


