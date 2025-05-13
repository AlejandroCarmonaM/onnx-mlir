/**********************************************
 * IMPORT LIBRARIES
 **********************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For memcpy, strcmp
#include <assert.h> // For assert
#include <math.h> // For fabs
#include "onnxruntime_c_api.h" // ONNX Runtime C API

/**********************************************
 * CONSTANTS & PARAMETERS
 **********************************************/
const char* MODEL_PATH = "model_with_mycallerop.onnx";
const char* CUSTOM_OP_DOMAIN_NAME = "my.custom.domain";

// Global OrtApi pointer
const OrtApi* g_ort = NULL;

/**********************************************
 * HELPER FUNCTIONS & STRUCTS
 **********************************************/

// Helper to check ORT status
void CheckStatus(OrtStatus* status, const char* operation_name) {
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "ERROR during %s: %s\n", operation_name, msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

// Kernel structure for MyCallerOp
typedef struct {
    OrtKernelInfo* kernel_info; // To store a copy of kernel_info for MyCallerOp (non-const)
    OrtAllocator* allocator;
    // FusedGemm attributes (can be configured if needed)
    float alpha;
    float beta;
    int64_t transA;
    int64_t transB;
} MyCallerOpKernel;

/**********************************************
 * CUSTOM OPERATOR: MyCallerOp DEFINITIONS
 **********************************************/

// MyCallerOp: KernelCreate
void* ORT_API_CALL MyCallerOp_CreateKernel(const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    (void)op; 
    MyCallerOpKernel* op_kernel = (MyCallerOpKernel*)malloc(sizeof(MyCallerOpKernel));
    if (!op_kernel) {
        if (api) { // Use the passed 'api'
            OrtStatus* status = api->CreateStatus(ORT_FAIL, "Failed to allocate memory for MyCallerOpKernel");
            // Optionally, log the error message from 'status' if desired, then release it.
            // api->ReleaseStatus(status); // If you were to use the status object
        }
        fprintf(stderr, "ERROR: Failed to allocate memory for MyCallerOpKernel\n");
        return NULL; 
    }

    // Use the 'api' pointer passed to the function
    OrtStatus* status1 = api->GetAllocatorWithDefaultOptions(&op_kernel->allocator);
    if (status1 != NULL) { // Manual check if not using CheckStatus which uses global g_ort
        fprintf(stderr, "ERROR during GetAllocatorWithDefaultOptions: %s\n", api->GetErrorMessage(status1));
        api->ReleaseStatus(status1);
        free(op_kernel);
        return NULL;
    }

    OrtStatus* status2 = api->CopyKernelInfo(info, &op_kernel->kernel_info);
    if (status2 != NULL) { // Manual check
        fprintf(stderr, "ERROR during CopyKernelInfo: %s\n", api->GetErrorMessage(status2));
        api->ReleaseStatus(status2);
        api->ReleaseAllocator(op_kernel->allocator); // Clean up previously acquired allocator
        free(op_kernel);
        return NULL;
    }
    
    op_kernel->alpha = 1.0f;
    op_kernel->beta = 0.0f; 
    op_kernel->transA = 0;
    op_kernel->transB = 0;

    return op_kernel;
}

// MyCallerOp: KernelCompute
void ORT_API_CALL MyCallerOp_Compute(void* op_kernel_void, OrtKernelContext* context) {
    MyCallerOpKernel* op_kernel = (MyCallerOpKernel*)op_kernel_void;

    // 1. Get inputs for MyCallerOp (which will be A and B for FusedGemm)
    const OrtValue* input_A_val;
    CheckStatus(g_ort->KernelContext_GetInput(context, 0, &input_A_val), "GetInput A for MyCallerOp");
    const OrtValue* input_B_val;
    CheckStatus(g_ort->KernelContext_GetInput(context, 1, &input_B_val), "GetInput B for MyCallerOp");

    // 2. Prepare attributes for FusedGemm
    OrtOpAttr* attr_alpha = NULL;
    OrtOpAttr* attr_beta = NULL;
    OrtOpAttr* attr_transA = NULL;
    OrtOpAttr* attr_transB = NULL;

    CheckStatus(g_ort->CreateOpAttr("alpha", &op_kernel->alpha, sizeof(float), ORT_OP_ATTR_FLOAT, &attr_alpha), "CreateOpAttr alpha");
    CheckStatus(g_ort->CreateOpAttr("beta", &op_kernel->beta, sizeof(float), ORT_OP_ATTR_FLOAT, &attr_beta), "CreateOpAttr beta");
    CheckStatus(g_ort->CreateOpAttr("transA", &op_kernel->transA, sizeof(int64_t), ORT_OP_ATTR_INT, &attr_transA), "CreateOpAttr transA");
    CheckStatus(g_ort->CreateOpAttr("transB", &op_kernel->transB, sizeof(int64_t), ORT_OP_ATTR_INT, &attr_transB), "CreateOpAttr transB");
    
    const OrtOpAttr* fused_gemm_attrs[] = {attr_alpha, attr_beta, attr_transA, attr_transB};

    // 3. Create the FusedGemm operator instance
    OrtOp* fused_gemm_op = NULL;
    // Use the kernel_info of MyCallerOp itself for CreateOp's first argument
    CheckStatus(g_ort->CreateOp(op_kernel->kernel_info, "FusedGemm", "com.microsoft", 1, 
                               NULL, NULL, 0, // Type constraints (none for FusedGemm)
                               fused_gemm_attrs, 4, // Attributes
                               2, 1, // Input count (A, B), Output count (Y)
                               &fused_gemm_op), "CreateOp FusedGemm");

    // 4. Prepare inputs for FusedGemm
    const OrtValue* fused_gemm_inputs[] = {input_A_val, input_B_val};

    // 5. Prepare output for FusedGemm (this will be MyCallerOp's output)
    // Get tensor info for input A to determine M and K
    OrtTensorTypeAndShapeInfo* input_A_info;
    CheckStatus(g_ort->GetTensorTypeAndShape(input_A_val, &input_A_info), "GetTensorTypeAndShape for A");
    size_t num_dims_A;
    CheckStatus(g_ort->GetDimensionsCount(input_A_info, &num_dims_A), "GetDimensionsCount for A");
    assert(num_dims_A == 2); // Expecting 2D matrix
    int64_t dims_A[2];
    CheckStatus(g_ort->GetDimensions(input_A_info, dims_A, num_dims_A), "GetDimensions for A");
    g_ort->ReleaseTensorTypeAndShapeInfo(input_A_info);

    // Get tensor info for input B to determine K and N
    OrtTensorTypeAndShapeInfo* input_B_info;
    CheckStatus(g_ort->GetTensorTypeAndShape(input_B_val, &input_B_info), "GetTensorTypeAndShape for B");
    size_t num_dims_B;
    CheckStatus(g_ort->GetDimensionsCount(input_B_info, &num_dims_B), "GetDimensionsCount for B");
    assert(num_dims_B == 2); // Expecting 2D matrix
    int64_t dims_B[2];
    CheckStatus(g_ort->GetDimensions(input_B_info, dims_B, num_dims_B), "GetDimensions for B");
    g_ort->ReleaseTensorTypeAndShapeInfo(input_B_info);

    // Output dimensions for Y (M_A x N_B)
    int64_t output_dims_Y[] = {dims_A[0], dims_B[1]}; // Assuming no transpose for simplicity of shape calc
    
    OrtValue* output_Y_val = NULL; // This will be MyCallerOp's output tensor
    CheckStatus(g_ort->KernelContext_GetOutput(context, 0, output_dims_Y, 2, &output_Y_val), "GetOutput Y for MyCallerOp");
    
    const OrtValue* fused_gemm_outputs[] = {output_Y_val}; // InvokeOp will write into this

    // 6. Invoke FusedGemm
    CheckStatus(g_ort->InvokeOp(context, fused_gemm_op, fused_gemm_inputs, 2, 
                               (OrtValue* const*)fused_gemm_outputs, 1), "InvokeOp FusedGemm");
    
    // 7. Cleanup FusedGemm specific resources
    g_ort->ReleaseOp(fused_gemm_op);
    g_ort->ReleaseOpAttr(attr_alpha);
    g_ort->ReleaseOpAttr(attr_beta);
    g_ort->ReleaseOpAttr(attr_transA);
    g_ort->ReleaseOpAttr(attr_transB);
}

// MyCallerOp: KernelDestroy
void ORT_API_CALL MyCallerOp_DestroyKernel(void* op_kernel_void) {
    MyCallerOpKernel* op_kernel = (MyCallerOpKernel*)op_kernel_void;
    if (op_kernel) {
        if (op_kernel->kernel_info) { // Check if kernel_info was successfully copied
            g_ort->ReleaseKernelInfo(op_kernel->kernel_info); // Release the copied info
        }
        if (op_kernel->allocator) { // Check if allocator was successfully obtained
            g_ort->ReleaseAllocator(op_kernel->allocator);
        }
        free(op_kernel);
    }
}

// MyCallerOp: Getters for op definition
const char* ORT_API_CALL MyCallerOp_GetName(const OrtCustomOp* op) { (void)op; return "MyCallerOp"; }
const char* ORT_API_CALL MyCallerOp_GetExecutionProviderType(const OrtCustomOp* op) { (void)op; return NULL; } // NULL for CPU
ONNXTensorElementDataType ORT_API_CALL MyCallerOp_GetInputType(const OrtCustomOp* op, size_t index) { (void)op; (void)index; return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
size_t ORT_API_CALL MyCallerOp_GetInputTypeCount(const OrtCustomOp* op) { (void)op; return 2; } // A, B
ONNXTensorElementDataType ORT_API_CALL MyCallerOp_GetOutputType(const OrtCustomOp* op, size_t index) { (void)op; (void)index; return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
size_t ORT_API_CALL MyCallerOp_GetOutputTypeCount(const OrtCustomOp* op) { (void)op; return 1; } // Y


/**********************************************
 * MAIN PROGRAM
 **********************************************/
int main() {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "Failed to get ONNX Runtime API\n");
        return 1;
    }

    OrtEnv* env;
    CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "FusedGemmInvokeTest", &env), "CreateEnv");

    OrtSessionOptions* session_options;
    CheckStatus(g_ort->CreateSessionOptions(&session_options), "CreateSessionOptions");
    CheckStatus(g_ort->SetIntraOpNumThreads(session_options, 1), "SetIntraOpNumThreads"); // For simplicity
    
    // Enable contrib ops, FusedGemm is often a contrib op.
    // This might not be strictly necessary if FusedGemm is found via com.microsoft domain directly,
    // but good to have if it falls under general contrib op handling.
    // CheckStatus(g_ort->EnableOrtCustomOps(session_options), "EnableOrtCustomOps"); // <--- COMMENT OUT THIS LINE


    /******************************************
     * REGISTER CUSTOM OPERATOR
     ******************************************/
    OrtCustomOpDomain* custom_op_domain;
    CheckStatus(g_ort->CreateCustomOpDomain(CUSTOM_OP_DOMAIN_NAME, &custom_op_domain), "CreateCustomOpDomain");

    OrtCustomOp my_caller_op;
    my_caller_op.version = ORT_API_VERSION; // Use ORT_API_VERSION if ORT_CUSTOM_OP_API_VERSION is not defined
    my_caller_op.CreateKernel = MyCallerOp_CreateKernel;
    my_caller_op.GetName = MyCallerOp_GetName;
    my_caller_op.GetExecutionProviderType = MyCallerOp_GetExecutionProviderType;
    my_caller_op.GetInputType = MyCallerOp_GetInputType;
    my_caller_op.GetInputTypeCount = MyCallerOp_GetInputTypeCount;
    my_caller_op.GetOutputType = MyCallerOp_GetOutputType;
    my_caller_op.GetOutputTypeCount = MyCallerOp_GetOutputTypeCount;
    my_caller_op.KernelCompute = MyCallerOp_Compute;
    my_caller_op.KernelDestroy = MyCallerOp_DestroyKernel;
    // Optional fields can be NULL if not used
    my_caller_op.GetInputCharacteristic = NULL;
    my_caller_op.GetOutputCharacteristic = NULL;
    my_caller_op.GetInputMemoryType = NULL; 
    my_caller_op.GetVariadicInputMinArity = NULL;
    my_caller_op.GetVariadicInputHomogeneity = NULL;
    my_caller_op.GetVariadicOutputMinArity = NULL;
    my_caller_op.GetVariadicOutputHomogeneity = NULL;
    // Remove V2 fields if they are not in your OrtCustomOp struct version
    // my_caller_op.CreateKernelV2 = NULL; 
    // my_caller_op.ComputeV2 = NULL;
    // my_caller_op.DestroyV2 = NULL;
    my_caller_op.InferOutputShapeFn = NULL; // Corrected based on compiler suggestion, was InferOutputShape


    CheckStatus(g_ort->CustomOpDomain_Add(custom_op_domain, &my_caller_op), "CustomOpDomain_Add MyCallerOp");

    CheckStatus(g_ort->AddCustomOpDomain(session_options, custom_op_domain), "AddCustomOpDomain to SessionOptions");

    /******************************************
     * CREATE SESSION & LOAD MODEL
     ******************************************/
    OrtSession* session;
    printf("Loading model: %s\n", MODEL_PATH);
    CheckStatus(g_ort->CreateSession(env, MODEL_PATH, session_options, &session), "CreateSession");
    printf("Model loaded successfully.\n");

    /******************************************
     * PREPARE INPUT DATA
     ******************************************/
    OrtAllocator* allocator;
    CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator), "GetAllocatorWithDefaultOptions for main");

    float matrix_A_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int64_t matrix_A_dims[] = {2, 2};
    size_t matrix_A_data_len = sizeof(matrix_A_data);

    float matrix_B_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    int64_t matrix_B_dims[] = {2, 2};
    size_t matrix_B_data_len = sizeof(matrix_B_data);

    OrtValue* input_tensors[2];
    OrtMemoryInfo* memory_info_cpu;
    CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_cpu), "CreateCpuMemoryInfo");

    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info_cpu, matrix_A_data, matrix_A_data_len,
                                                     matrix_A_dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                     &input_tensors[0]), "CreateTensor A");
    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info_cpu, matrix_B_data, matrix_B_data_len,
                                                     matrix_B_dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                     &input_tensors[1]), "CreateTensor B");
    g_ort->ReleaseMemoryInfo(memory_info_cpu); // Released after use

    const char* input_names[] = {"XA", "XB"};
    const char* output_names[] = {"XY"};
    OrtValue* output_tensor = NULL;

    /******************************************
     * PERFORM ACTIONS (RUN SESSION)
     ******************************************/
    printf("Running inference...\n");
    CheckStatus(g_ort->Run(session, NULL, input_names, (const OrtValue* const*)input_tensors, 2,
                           output_names, 1, &output_tensor), "Run Session");
    printf("Inference completed.\n");

    /******************************************
     * OUTPUT RESULTS & VERIFY
     ******************************************/
    float* output_data;
    CheckStatus(g_ort->GetTensorMutableData(output_tensor, (void**)&output_data), "GetTensorMutableData for output");

    printf("Output tensor Y:\n");
    printf("[%.1f, %.1f]\n", output_data[0], output_data[1]);
    printf("[%.1f, %.1f]\n", output_data[2], output_data[3]);

    float expected_Y_data[] = {19.0f, 22.0f, 43.0f, 50.0f}; // (1*5+2*7), (1*6+2*8), (3*5+4*7), (3*6+4*8)
    int correct = 1;
    for (size_t i = 0; i < 4; ++i) {
        if (fabs(output_data[i] - expected_Y_data[i]) > 1e-5) {
            correct = 0;
            break;
        }
    }
    if (correct) {
        printf("Verification PASSED!\n");
    } else {
        printf("Verification FAILED!\n");
        printf("Expected Y:\n[%.1f, %.1f]\n[%.1f, %.1f]\n",
               expected_Y_data[0], expected_Y_data[1], expected_Y_data[2], expected_Y_data[3]);
    }

    /******************************************
     * CLEANUP
     ******************************************/
    g_ort->ReleaseValue(input_tensors[0]);
    g_ort->ReleaseValue(input_tensors[1]);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    // CustomOpDomain is owned by session_options after AddCustomOpDomain,
    // but if AddCustomOpDomain fails or for standalone domains, it needs release.
    // Here, it's tied to session_options, which handles its lifecycle.
    // If CreateCustomOpDomain was called but not added, it would need g_ort->ReleaseCustomOpDomain(custom_op_domain);
    g_ort->ReleaseEnv(env);
    g_ort->ReleaseAllocator(allocator); // Release allocator obtained in main

    printf("Program finished.\n");
    return 0;
}