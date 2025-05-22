// Filename: fused_gemm_test.c
// /* Description:
//  * This file implements a test for the FusedGemm operator using ONNX Runtime C API.
//  * It creates input tensors, sets up the operator attributes, and invokes the operator.
//  * The output tensor is printed to the console.
//  * It does compile but has not been tested in runtime yet. So IT MAY NOT WORK (~99% doesn't work, but it 
//  * shows how to link the ONNX Runtime C API in compile_onnxrt_program.sh and use some of the C API funcs).

#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_c_api.h>

int main() {
    // 1. Initialize ONNX Runtime environment and get API pointer
    OrtEnv* env = NULL;
    const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtStatus* status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "fusedgemm", &env);
    if (status) { fprintf(stderr, "Env error\n"); return 1; }

    // 2. Create memory info for CPU tensors
    OrtMemoryInfo* memory_info = NULL;
    status = ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status) { fprintf(stderr, "MemInfo error\n"); return 1; }

    // 3. Prepare input tensors (A: [2,3], B: [3,4], C: [2,4])
    float A[6] = {1, 2, 3, 4, 5, 6};
    float B[12] = {1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0};
    float C[8] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    int64_t shape_A[2] = {2, 3}, shape_B[2] = {3, 4}, shape_C[2] = {2, 4};

    OrtValue *input_A = NULL, *input_B = NULL, *input_C = NULL;
    status = ort->CreateTensorWithDataAsOrtValue(memory_info, A, sizeof(A), shape_A, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_A);
    if (status) { fprintf(stderr, "Tensor A error\n"); return 1; }
    status = ort->CreateTensorWithDataAsOrtValue(memory_info, B, sizeof(B), shape_B, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_B);
    if (status) { fprintf(stderr, "Tensor B error\n"); return 1; }
    status = ort->CreateTensorWithDataAsOrtValue(memory_info, C, sizeof(C), shape_C, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_C);
    if (status) { fprintf(stderr, "Tensor C error\n"); return 1; }

    // 4. Create operator attributes (alpha, beta, transA, transB)
    float alpha = 1.0f, beta = 1.0f;
    int64_t transA = 0, transB = 0;
    OrtOpAttr *attr_alpha = NULL, *attr_beta = NULL, *attr_transA = NULL, *attr_transB = NULL;
    status = ort->CreateOpAttr("alpha", &alpha, sizeof(float), ORT_OP_ATTR_FLOAT, &attr_alpha);
    if (status) { fprintf(stderr, "Attr alpha error\n"); return 1; }
    status = ort->CreateOpAttr("beta", &beta, sizeof(float), ORT_OP_ATTR_FLOAT, &attr_beta);
    if (status) { fprintf(stderr, "Attr beta error\n"); return 1; }
    status = ort->CreateOpAttr("transA", &transA, sizeof(int64_t), ORT_OP_ATTR_INT, &attr_transA);
    if (status) { fprintf(stderr, "Attr transA error\n"); return 1; }
    status = ort->CreateOpAttr("transB", &transB, sizeof(int64_t), ORT_OP_ATTR_INT, &attr_transB);
    if (status) { fprintf(stderr, "Attr transB error\n"); return 1; }

    // 5. Create the FusedGemm operator
    OrtOp* fused_gemm_op = NULL;
    const OrtOpAttr* attrs[] = {attr_alpha, attr_beta, attr_transA, attr_transB};
    /*CreateOp()
    OrtStatus * OrtApi::CreateOp	(	const OrtKernelInfo * 	info,
    const char * 	op_name,
    const char * 	domain,
    int 	version,
    const char ** 	type_constraint_names,
    const ONNXTensorElementDataType * 	type_constraint_values,
    int 	type_constraint_count,
    const OrtOpAttr *const * 	attr_values,
    int 	attr_count,
    int 	input_count,
    int 	output_count,
    OrtOp ** 	ort_op 
    )		
    : Create onnxruntime native operator

    Parameters
    [in]	info	Kernel info
    [in]	op_name	Operator name
    [in]	domain	Operator domain
    [in]	version	Operator opset version
    [in]	type_constraint_names	Name of the type contraints, such as "T" or "T1"
    [in]	type_constraint_values	Type of each contraints
    [in]	type_constraint_count	Number of contraints
    [in]	attr_values	Attributes used to initialize the operator
    [in]	attr_count	Number of the attributes
    [in]	input_count	Number of inputs
    [in]	output_count	Number of outputs
    [out]	ort_op	Operator that has been created*/

    const char* type_constraint_names[] = {"T"};
    ONNXTensorElementDataType type_constraint_values[] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
    status = ort->CreateOp(
        NULL, // Kernel info
        "FusedGemm",
        "com.microsoft",
        14, // opset version
        type_constraint_names,
        type_constraint_values,
        1, // type_constraint_count
        attrs,
        4, // attr_count
        3, // input_count
        1, // output_count
        &fused_gemm_op
    );
    if (status) { fprintf(stderr, "CreateOp error\n"); return 1; }

    // 6. Prepare input and output arrays
    OrtValue* inputs[] = {input_A, input_B, input_C};
    OrtValue* output_Y = NULL;
    OrtValue* outputs[] = {output_Y};

    // 7. Invoke the operator
    status = ort->InvokeOp(
        NULL, // context
        fused_gemm_op,
        (const OrtValue* const*)inputs,
        3,
        outputs,
        1
    );
    if (status) { fprintf(stderr, "InvokeOp error\n"); return 1; }

    // 8. Retrieve and print output tensor
    float* out_data = NULL;
    status = ort->GetTensorMutableData(outputs[0], (void**)&out_data);
    if (status) { fprintf(stderr, "GetTensorMutableData error\n"); return 1; }
    printf("Output Y:\n");
    for (int i = 0; i < 8; ++i) printf("%f ", out_data[i]);
    printf("\n");

    // 9. Release all resources
    ort->ReleaseValue(input_A);
    ort->ReleaseValue(input_B);
    ort->ReleaseValue(input_C);
    ort->ReleaseValue(outputs[0]);
    ort->ReleaseOp(fused_gemm_op);
    ort->ReleaseOpAttr(attr_alpha);
    ort->ReleaseOpAttr(attr_beta);
    ort->ReleaseOpAttr(attr_transA);
    ort->ReleaseOpAttr(attr_transB);
    ort->ReleaseMemoryInfo(memory_info);
    ort->ReleaseEnv(env);

    return 0;
}