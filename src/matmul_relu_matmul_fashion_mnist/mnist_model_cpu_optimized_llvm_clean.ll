module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "mnist_model_cpu_optimized"} {
  llvm.func @strncmp(!llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.mlir.global external constant @_entry_point_1_mnist_model_cpu_optimized("run_main_graph_mnist_model_cpu_optimized\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_1_in_sig_mnist_model_cpu_optimized("[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 784] , \22name\22 : \22input\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_1_out_sig_mnist_model_cpu_optimized("[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22output\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_mnist_model_cpu_optimized("run_main_graph\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_in_sig_mnist_model_cpu_optimized("[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 784] , \22name\22 : \22input\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.mlir.global external constant @_entry_point_0_out_sig_mnist_model_cpu_optimized("[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \22output\22 }\0A\0A]\00") {addr_space = 0 : i32}
  llvm.func @free(!llvm.ptr)
  llvm.func @FusedGemm(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, !llvm.ptr, !llvm.ptr, i64, i64)
  llvm.mlir.global internal constant @"om_fused /fc1/Gemm_mnist_model_cpu_optimized"("fused /fc1/Gemm") {addr_space = 0 : i32, alignment = 16 : i64}
  llvm.mlir.global internal constant @om_com.microsoft_mnist_model_cpu_optimized("com.microsoft") {addr_space = 0 : i32, alignment = 16 : i64}
  llvm.mlir.global internal constant @om_Relu_mnist_model_cpu_optimized("Relu") {addr_space = 0 : i32, alignment = 16 : i64}
  llvm.func @omGetExternalConstantAddr(!llvm.ptr, !llvm.ptr, i64)
  llvm.func @omMMapBinaryFile(!llvm.ptr, !llvm.ptr, i64, i64) -> i1
  llvm.func @omTensorListGetSize(!llvm.ptr) -> i64
  llvm.func @omTensorPrint(!llvm.ptr, !llvm.ptr)
  llvm.func @omTensorListGetOmtArray(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorSetDataType(!llvm.ptr, i64)
  llvm.func @omTensorGetDataType(!llvm.ptr) -> i64
  llvm.func @omTensorGetStrides(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorGetShape(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorGetRank(!llvm.ptr) -> i64
  llvm.func @omTensorSetDataPtr(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr)
  llvm.func @omTensorGetDataPtr(!llvm.ptr) -> !llvm.ptr
  llvm.func @omTensorDestroy(!llvm.ptr)
  llvm.func @omTensorCreateUntyped(i64) -> !llvm.ptr
  llvm.func @omTensorListCreate(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global internal constant @constant_3_mnist_model_cpu_optimized(dense<[0.0835401713, -0.373944432, 0.206879765, 0.180731401, -0.428273022, 0.313445568, 0.191657051, 0.136042759, -0.325663328, -0.49137786]> : tensor<10xf32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<10 x f32>
  llvm.mlir.global internal constant @constant_2_mnist_model_cpu_optimized("[...]")
  llvm.mlir.global internal constant @constant_1_mnist_model_cpu_optimized(dense<"[...]") {addr_space = 0 : i32, alignment = 16 : i64}
  llvm.func @main_graph_mnist_model_cpu_optimized(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> (!llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {onnx.name = "output"}) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : si64) : i64
    %1 = llvm.mlir.constant(0 : si64) : i64
    %2 = llvm.mlir.addressof @"om_fused /fc1/Gemm_mnist_model_cpu_optimized" : !llvm.ptr
    %3 = llvm.mlir.addressof @om_com.microsoft_mnist_model_cpu_optimized : !llvm.ptr
    %4 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %5 = llvm.mlir.addressof @om_Relu_mnist_model_cpu_optimized : !llvm.ptr
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(0 : i64) : i64
    %8 = llvm.mlir.constant(2 : i64) : i64
    %9 = llvm.mlir.constant(16 : index) : i64
    %10 = llvm.mlir.zero : !llvm.ptr
    %11 = llvm.mlir.addressof @constant_3_mnist_model_cpu_optimized : !llvm.ptr
    %12 = llvm.mlir.addressof @constant_2_mnist_model_cpu_optimized : !llvm.ptr
    %13 = llvm.mlir.addressof @constant_1_mnist_model_cpu_optimized : !llvm.ptr
    %14 = llvm.mlir.constant(784 : index) : i64
    %15 = llvm.mlir.addressof @constant_0_mnist_model_cpu_optimized : !llvm.ptr
    %16 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    %19 = llvm.mlir.constant(10 : index) : i64
    %20 = llvm.mlir.constant(128 : index) : i64
    %21 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.getelementptr %10[128] : (!llvm.ptr) -> !llvm.ptr, f32
    %23 = llvm.ptrtoint %22 : !llvm.ptr to i64
    %24 = llvm.add %23, %9 : i64
    %25 = llvm.call @malloc(%24) : (i64) -> !llvm.ptr
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.sub %9, %18 : i64
    %28 = llvm.add %26, %27 : i64
    %29 = llvm.urem %28, %9 : i64
    %30 = llvm.sub %28, %29 : i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    %32 = llvm.call @omTensorCreateUntyped(%8) : (i64) -> !llvm.ptr
    llvm.call @omTensorSetDataPtr(%32, %7, %25, %31) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @omTensorSetDataType(%32, %6) : (!llvm.ptr, i64) -> ()
    %33 = llvm.call @omTensorGetShape(%32) : (!llvm.ptr) -> !llvm.ptr
    %34 = llvm.call @omTensorGetStrides(%32) : (!llvm.ptr) -> !llvm.ptr
    llvm.store %18, %33 : i64, !llvm.ptr
    llvm.store %20, %34 : i64, !llvm.ptr
    %35 = llvm.getelementptr %33[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %20, %35 : i64, !llvm.ptr
    %36 = llvm.getelementptr %34[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %18, %36 : i64, !llvm.ptr
    %37 = llvm.call @omTensorCreateUntyped(%8) : (i64) -> !llvm.ptr
    llvm.call @omTensorSetDataPtr(%37, %7, %arg0, %arg1) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @omTensorSetDataType(%37, %6) : (!llvm.ptr, i64) -> ()
    %38 = llvm.call @omTensorGetShape(%37) : (!llvm.ptr) -> !llvm.ptr
    %39 = llvm.call @omTensorGetStrides(%37) : (!llvm.ptr) -> !llvm.ptr
    llvm.store %arg3, %38 : i64, !llvm.ptr
    llvm.store %arg5, %39 : i64, !llvm.ptr
    %40 = llvm.getelementptr %38[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %arg4, %40 : i64, !llvm.ptr
    %41 = llvm.getelementptr %39[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %arg6, %41 : i64, !llvm.ptr
    %42 = llvm.call @omTensorCreateUntyped(%8) : (i64) -> !llvm.ptr
    llvm.call @omTensorSetDataPtr(%42, %7, %15, %15) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @omTensorSetDataType(%42, %6) : (!llvm.ptr, i64) -> ()
    %43 = llvm.call @omTensorGetShape(%42) : (!llvm.ptr) -> !llvm.ptr
    %44 = llvm.call @omTensorGetStrides(%42) : (!llvm.ptr) -> !llvm.ptr
    llvm.store %20, %43 : i64, !llvm.ptr
    llvm.store %14, %44 : i64, !llvm.ptr
    %45 = llvm.getelementptr %43[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %14, %45 : i64, !llvm.ptr
    %46 = llvm.getelementptr %44[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %18, %46 : i64, !llvm.ptr
    %47 = llvm.call @omTensorCreateUntyped(%6) : (i64) -> !llvm.ptr
    llvm.call @omTensorSetDataPtr(%47, %7, %13, %13) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @omTensorSetDataType(%47, %6) : (!llvm.ptr, i64) -> ()
    %48 = llvm.call @omTensorGetShape(%47) : (!llvm.ptr) -> !llvm.ptr
    %49 = llvm.call @omTensorGetStrides(%47) : (!llvm.ptr) -> !llvm.ptr
    llvm.store %20, %48 : i64, !llvm.ptr
    llvm.store %18, %49 : i64, !llvm.ptr
    llvm.call @FusedGemm(%32, %37, %42, %47, %5, %4, %4, %3, %2, %1, %0) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
    llvm.call @omTensorDestroy(%32) : (!llvm.ptr) -> ()
    llvm.call @omTensorDestroy(%37) : (!llvm.ptr) -> ()
    llvm.call @omTensorDestroy(%42) : (!llvm.ptr) -> ()
    llvm.call @omTensorDestroy(%47) : (!llvm.ptr) -> ()
    %50 = llvm.getelementptr %10[10] : (!llvm.ptr) -> !llvm.ptr, f32
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.add %51, %20 : i64
    %53 = llvm.call @malloc(%52) : (i64) -> !llvm.ptr
    %54 = llvm.ptrtoint %53 : !llvm.ptr to i64
    %55 = llvm.sub %20, %18 : i64
    %56 = llvm.add %54, %55 : i64
    %57 = llvm.urem %56, %20 : i64
    %58 = llvm.sub %56, %57 : i64
    %59 = llvm.inttoptr %58 : i64 to !llvm.ptr
    %60 = llvm.insertvalue %53, %21[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.insertvalue %59, %60[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.insertvalue %17, %61[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.insertvalue %18, %62[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.insertvalue %19, %63[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %19, %64[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.insertvalue %18, %65[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.alloca %18 x f32 : (i64) -> !llvm.ptr
    llvm.br ^bb1(%17 : i64)
  ^bb1(%68: i64):  // 2 preds: ^bb0, ^bb8
    %69 = llvm.icmp "slt" %68, %18 : i64
    llvm.cond_br %69, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%17 : i64)
  ^bb3(%70: i64):  // 2 preds: ^bb2, ^bb7
    %71 = llvm.icmp "slt" %70, %19 : i64
    llvm.cond_br %71, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.store %16, %67 : f32, !llvm.ptr
    llvm.br ^bb5(%17 : i64)
  ^bb5(%72: i64):  // 2 preds: ^bb4, ^bb6
    %73 = llvm.icmp "slt" %72, %20 : i64
    llvm.cond_br %73, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %74 = llvm.mul %68, %20 : i64
    %75 = llvm.add %74, %72 : i64
    %76 = llvm.getelementptr %31[%75] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %77 = llvm.load %76 : !llvm.ptr -> f32
    %78 = llvm.mul %70, %20 : i64
    %79 = llvm.add %78, %72 : i64
    %80 = llvm.getelementptr %12[%79] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %81 = llvm.load %80 : !llvm.ptr -> f32
    %82 = llvm.fmul %77, %81 : f32
    %83 = llvm.load %67 : !llvm.ptr -> f32
    %84 = llvm.fadd %82, %83 : f32
    llvm.store %84, %67 : f32, !llvm.ptr
    %85 = llvm.add %72, %18 : i64
    llvm.br ^bb5(%85 : i64)
  ^bb7:  // pred: ^bb5
    %86 = llvm.load %67 : !llvm.ptr -> f32
    %87 = llvm.getelementptr %11[%70] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %88 = llvm.load %87 : !llvm.ptr -> f32
    %89 = llvm.fadd %86, %88 : f32
    %90 = llvm.mul %68, %19 : i64
    %91 = llvm.add %90, %70 : i64
    %92 = llvm.getelementptr %59[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %89, %92 : f32, !llvm.ptr
    %93 = llvm.add %70, %18 : i64
    llvm.br ^bb3(%93 : i64)
  ^bb8:  // pred: ^bb3
    %94 = llvm.add %68, %18 : i64
    llvm.br ^bb1(%94 : i64)
  ^bb9:  // pred: ^bb1
    llvm.call @free(%25) : (!llvm.ptr) -> ()
    llvm.return %66 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  llvm.func @_mlir_ciface_main_graph_mnist_model_cpu_optimized(%arg0: !llvm.ptr, %arg1: !llvm.ptr {onnx.name = "input"}) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.call @main_graph_mnist_model_cpu_optimized(%1, %2, %3, %4, %5, %6, %7) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.store %8, %arg0 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.return
  }
  llvm.func @run_main_graph_mnist_model_cpu_optimized(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.constant(2 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.call @omTensorListGetOmtArray(%arg0) : (!llvm.ptr) -> !llvm.ptr
    %5 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    %6 = llvm.load %4 : !llvm.ptr -> !llvm.ptr
    %7 = llvm.alloca %3 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    %8 = llvm.call @omTensorGetDataPtr(%6) : (!llvm.ptr) -> !llvm.ptr
    %9 = llvm.insertvalue %8, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %8, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %1, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.call @omTensorGetShape(%6) : (!llvm.ptr) -> !llvm.ptr
    %13 = llvm.call @omTensorGetStrides(%6) : (!llvm.ptr) -> !llvm.ptr
    %14 = llvm.load %12 : !llvm.ptr -> i64
    %15 = llvm.insertvalue %14, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.load %13 : !llvm.ptr -> i64
    %17 = llvm.insertvalue %16, %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.getelementptr %12[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %19 = llvm.load %18 : !llvm.ptr -> i64
    %20 = llvm.insertvalue %19, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.getelementptr %13[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %22 = llvm.load %21 : !llvm.ptr -> i64
    %23 = llvm.insertvalue %22, %20[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.store %23, %7 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.call @_mlir_ciface_main_graph_mnist_model_cpu_optimized(%5, %7) : (!llvm.ptr, !llvm.ptr) -> ()
    %24 = llvm.load %5 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.alloca %3 x !llvm.ptr : (i64) -> !llvm.ptr
    %26 = llvm.call @omTensorCreateUntyped(%0) : (i64) -> !llvm.ptr
    %27 = llvm.extractvalue %24[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.extractvalue %24[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @omTensorSetDataPtr(%26, %3, %27, %28) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @omTensorSetDataType(%26, %3) : (!llvm.ptr, i64) -> ()
    %29 = llvm.call @omTensorGetShape(%26) : (!llvm.ptr) -> !llvm.ptr
    %30 = llvm.call @omTensorGetStrides(%26) : (!llvm.ptr) -> !llvm.ptr
    %31 = llvm.extractvalue %24[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.store %31, %29 : i64, !llvm.ptr
    %32 = llvm.extractvalue %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.store %32, %30 : i64, !llvm.ptr
    %33 = llvm.extractvalue %24[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.getelementptr %29[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %33, %34 : i64, !llvm.ptr
    %35 = llvm.extractvalue %24[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.getelementptr %30[1] : (!llvm.ptr) -> !llvm.ptr, i64
    llvm.store %35, %36 : i64, !llvm.ptr
    llvm.store %26, %25 : !llvm.ptr, !llvm.ptr
    %37 = llvm.call @omTensorListCreate(%25, %3) : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.return %37 : !llvm.ptr
  }
  llvm.func @run_main_graph(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @run_main_graph_mnist_model_cpu_optimized(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global internal constant @_entry_point_arrays_mnist_model_cpu_optimized() {addr_space = 0 : i32} : !llvm.array<3 x ptr> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_mnist_model_cpu_optimized : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %3 = llvm.mlir.addressof @_entry_point_0_mnist_model_cpu_optimized : !llvm.ptr
    %4 = llvm.insertvalue %3, %2[0] : !llvm.array<3 x ptr> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.array<3 x ptr> 
    %6 = llvm.insertvalue %0, %5[2] : !llvm.array<3 x ptr> 
    llvm.return %6 : !llvm.array<3 x ptr>
  }
  llvm.func @omQueryEntryPoints_mnist_model_cpu_optimized(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.addressof @_entry_point_arrays_mnist_model_cpu_optimized : !llvm.ptr
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.icmp "ne" %arg0, %2 : !llvm.ptr
    llvm.cond_br %3, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.store %1, %arg0 : i64, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omQueryEntryPoints(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omQueryEntryPoints_mnist_model_cpu_optimized(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omInputSignature_mnist_model_cpu_optimized(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_in_sig_mnist_model_cpu_optimized : !llvm.ptr
    %2 = llvm.mlir.constant(41 : i64) : i64
    %3 = llvm.mlir.addressof @_entry_point_1_mnist_model_cpu_optimized : !llvm.ptr
    %4 = llvm.mlir.addressof @_entry_point_0_in_sig_mnist_model_cpu_optimized : !llvm.ptr
    %5 = llvm.mlir.constant(15 : i64) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.addressof @_entry_point_0_mnist_model_cpu_optimized : !llvm.ptr
    %8 = llvm.call @strncmp(%arg0, %7, %5) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %9 = llvm.icmp "eq" %8, %6 : i32
    llvm.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.return %4 : !llvm.ptr
  ^bb2:  // pred: ^bb0
    %10 = llvm.call @strncmp(%arg0, %3, %2) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %11 = llvm.icmp "eq" %10, %6 : i32
    llvm.cond_br %11, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.return %1 : !llvm.ptr
  ^bb4:  // pred: ^bb2
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omInputSignature(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omInputSignature_mnist_model_cpu_optimized(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omOutputSignature_mnist_model_cpu_optimized(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.addressof @_entry_point_1_out_sig_mnist_model_cpu_optimized : !llvm.ptr
    %2 = llvm.mlir.constant(41 : i64) : i64
    %3 = llvm.mlir.addressof @_entry_point_1_mnist_model_cpu_optimized : !llvm.ptr
    %4 = llvm.mlir.addressof @_entry_point_0_out_sig_mnist_model_cpu_optimized : !llvm.ptr
    %5 = llvm.mlir.constant(15 : i64) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.addressof @_entry_point_0_mnist_model_cpu_optimized : !llvm.ptr
    %8 = llvm.call @strncmp(%arg0, %7, %5) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %9 = llvm.icmp "eq" %8, %6 : i32
    llvm.cond_br %9, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.return %4 : !llvm.ptr
  ^bb2:  // pred: ^bb0
    %10 = llvm.call @strncmp(%arg0, %3, %2) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %11 = llvm.icmp "eq" %10, %6 : i32
    llvm.cond_br %11, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.return %1 : !llvm.ptr
  ^bb4:  // pred: ^bb2
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @omOutputSignature(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.call @omOutputSignature_mnist_model_cpu_optimized(%arg0) : (!llvm.ptr) -> !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}
