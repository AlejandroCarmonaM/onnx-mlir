module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "mnist_model_cpu_optimized"} {
  func.func @main_graph(%arg0: tensor<1x784xf32> {onnx.name = "input"}) -> (tensor<1x10xf32> {onnx.name = "output"}) {
    %0 = onnx.Constant dense<"[...]"> : tensor<128x784xf32>
    %1 = onnx.Constant dense<"[...]"> : tensor<128xf32>
    %2 = onnx.Constant dense<"[...]"> : tensor<10x128xf32>
    %3 = onnx.Constant dense<[0.0835401713, -0.373944432, 0.206879765, 0.180731401, -0.428273022, 0.313445568, 0.191657051, 0.136042759, -0.325663328, -0.49137786]> : tensor<10xf32>
    %4 = "onnx.Custom"(%arg0, %0, %1) {activation = "Relu", alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, domain_name = "com.microsoft", function_name = "FusedGemm", onnx_node_name = "fused /fc1/Gemm", transA = 0 : si64, transB = 1 : si64} : (tensor<1x784xf32>, tensor<128x784xf32>, tensor<128xf32>) -> tensor<1x128xf32>
    %5 = "onnx.Gemm"(%4, %2, %3) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "/fc2/Gemm", transA = 0 : si64, transB = 1 : si64} : (tensor<1x128xf32>, tensor<10x128xf32>, tensor<10xf32>) -> tensor<1x10xf32>
    return %5 : tensor<1x10xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
