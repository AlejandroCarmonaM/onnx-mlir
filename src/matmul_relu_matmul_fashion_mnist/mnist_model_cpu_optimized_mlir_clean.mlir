module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "mnist_model_cpu_optimized"} {
  func.func @main_graph(%arg0: memref<1x784xf32> {onnx.name = "input"}) -> (memref<1x10xf32> {onnx.name = "output"}) attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = "krnl.global"() {name = "constant_0", shape = [128, 784], value = [...] : tensor<128x784xf32>} : () -> memref<128x784xf32>
    %1 = "krnl.global"() {name = "constant_1", shape = [128], value = [...]  : tensor<128xf32>} : () -> memref<128xf32>
    %2 = "krnl.global"() {name = "constant_2", shape = [10, 128], value = [...] : tensor<10x128xf32>} : () -> memref<10x128xf32>
    %3 = "krnl.global"() {name = "constant_3", shape = [10], value = dense<[0.0835401713, -0.373944432, 0.206879765, 0.180731401, -0.428273022, 0.313445568, 0.191657051, 0.136042759, -0.325663328, -0.49137786]> : tensor<10xf32>} : () -> memref<10xf32>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x128xf32>
    "krnl.call"(%alloc, %arg0, %0, %1) {activation = "Relu", alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, domain_name = "com.microsoft", funcName = "FusedGemm", numOfOutput = 1 : si64, onnx_node_name = "fused /fc1/Gemm", transA = 0 : si64, transB = 1 : si64} : (memref<1x128xf32>, memref<1x784xf32>, memref<128x784xf32>, memref<128xf32>) -> ()
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<1x10xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst, %alloca[] : memref<f32>
        affine.for %arg3 = 0 to 128 {
          %7 = affine.load %alloc[%arg1, %arg3] : memref<1x128xf32>
          %8 = affine.load %2[%arg2, %arg3] : memref<10x128xf32>
          %9 = arith.mulf %7, %8 : f32
          %10 = affine.load %alloca[] : memref<f32>
          %11 = arith.addf %9, %10 : f32
          affine.store %11, %alloca[] : memref<f32>
        }
        %4 = affine.load %alloca[] : memref<f32>
        %5 = affine.load %3[%arg2] : memref<10xf32>
        %6 = arith.addf %4, %5 : f32
        affine.store %6, %alloc_0[%arg1, %arg2] : memref<1x10xf32>
      }
    }
    return %alloc_0 : memref<1x10xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 784] , \22name\22 : \22input\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22output\22 }\0A\0A]\00"} : () -> ()
}