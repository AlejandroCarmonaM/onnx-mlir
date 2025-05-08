/**********************************************
 * IMPORT LIBRARIES
 **********************************************/
#include <cstdint>
#include <vector>
#include <cmath>    // For std::max
#include <cstdio>   // For fprintf, stderr, fflush
#include <cstring>  // For memcpy
#include <numeric>  // For omTensorGetNumElems helper (std::accumulate)

// ONNX-MLIR Runtime API
#include "OnnxMlirRuntime.h"
// ONNX Runtime C API
#include "onnxruntime_c_api.h"

/**********************************************
 * CONSTANTS & PARAMETERS
 **********************************************/
// Using fixed alpha and beta as per "Mimics ONNX Gemm (alpha=1, beta=1)"
const float GEMM_ALPHA = 1.0f;
const float GEMM_BETA = 1.0f; // Beta applies to the C/Bias input

/**********************************************
 * HELPER FUNCTION DEFINITIONS
 **********************************************/

/*
 * Purpose: Basic ReLU activation function.
 */
inline float relu(float x) {
    return std::max(0.0f, x);
}

/*
 * Purpose: Calculate the total number of elements in an OMTensor.
 */
static size_t omTensorGetNumElems(OMTensor* omTensor) {
    if (!omTensor) return 0;
    int32_t rank = omTensorGetRank(omTensor);
    if (rank == 0 && omTensorGetDataPtr(omTensor) != nullptr) return 1;
    if (rank <= 0) return 0; // Also handles negative/invalid rank

    const int64_t* dims = omTensorGetShape(omTensor);
    if (!dims) return 0;

    size_t num_elems = 1;
    for (int32_t i = 0; i < rank; ++i) {
        if (dims[i] < 0) return 0; // Invalid dimension
        num_elems *= static_cast<size_t>(dims[i]);
    }
    return num_elems;
}

/*
 * Purpose: Helper to check ORT status and print errors.
 */
static void CheckOrtStatus(const OrtApi* ort_api, OrtStatus* status, const char* operation_name) {
    if (status != NULL) {
        const char* msg = ort_api->GetErrorMessage(status);
        fprintf(stderr, "    ONNX Runtime ERROR during %s: %s\n", operation_name, msg);
        ort_api->ReleaseStatus(status);
        // In experimental code, we might not exit, to see if cleanup can proceed
        // or to allow the caller to handle the error if this function returned a status.
    }
}


/**********************************************
 * MAIN RUNTIME FUNCTION DEFINITION
 **********************************************/

/*
 * Purpose: Wraps ONNX Runtime's FusedGemm operator and applies ReLU.
 *          Attempts a model-less, context-less invocation.
 * Parameters:
 *    - A_omTensor (OMTensor*): Input tensor A.
 *    - B_omTensor (OMTensor*): Input tensor B.
 *    - Bias_omTensor (OMTensor*): Optional input tensor C/Bias.
 *    - Y_omTensor (OMTensor*): Output tensor Y (result of FusedGemm + ReLU).
 *    - M (int64_t): Dimension M of the output.
 *    - N (int64_t): Dimension N of the output.
 *    - K (int64_t): Dimension K (shared dimension).
 *    - transA_param (int64_t): Flag indicating if A should be transposed.
 *    - transB_param (int64_t): Flag indicating if B should be transposed.
 * Returns:
 *    - void: Output Y_omTensor is modified in place.
 */
extern "C" void ort_cpu_ep_fused_gemm(
    OMTensor* A_omTensor,
    OMTensor* B_omTensor,
    OMTensor* Bias_omTensor,
    OMTensor* Y_omTensor,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t transA_param, // Renamed to avoid conflict with OrtOpAttr variable
    int64_t transB_param  // Renamed to avoid conflict with OrtOpAttr variable
) {
    /******************************************
     * INITIAL LOGGING & BASIC VALIDATION
     ******************************************/
    fprintf(stdout, "##########################################\n C++ ort_cpu_ep_fused_gemm (ORT Direct Call Wrapper) called:\n#######################\n");
    
    // As a verification step to verify that the call is happening, we exit now incondionally
    exit(0);
    
    fprintf(stderr, ">>> C++ ort_cpu_ep_fused_gemm (ORT Direct Call Wrapper) called:\n");
    fprintf(stderr, "    M=%lld, N=%lld, K=%lld, transA=%lld, transB=%lld\n",
            (long long)M, (long long)N, (long long)K, (long long)transA_param, (long long)transB_param);
    fprintf(stderr, "    A OMTensor*: %p, B OMTensor*: %p, Bias OMTensor*: %p, Y OMTensor*: %p\n",
            (void*)A_omTensor, (void*)B_omTensor, (void*)Bias_omTensor, (void*)Y_omTensor);
    fflush(stderr);

    if (!A_omTensor || !B_omTensor || !Y_omTensor ||
        !omTensorGetDataPtr(A_omTensor) || !omTensorGetDataPtr(B_omTensor) || !omTensorGetDataPtr(Y_omTensor)) {
         fprintf(stderr, "    ERROR: Received NULL OMTensor or NULL data pointer for A, B, or Y!\n");
         fflush(stderr);
         return;
    }
    if (omTensorGetDataType(A_omTensor) != ONNX_TYPE_FLOAT ||
        omTensorGetDataType(B_omTensor) != ONNX_TYPE_FLOAT ||
        (Bias_omTensor && omTensorGetDataPtr(Bias_omTensor) && omTensorGetDataType(Bias_omTensor) != ONNX_TYPE_FLOAT) ||
        omTensorGetDataType(Y_omTensor) != ONNX_TYPE_FLOAT) {
        fprintf(stderr, "    ERROR: All provided tensors must be of type float!\n");
        fflush(stderr);
        return;
    }

    /******************************************
     * INITIALIZE ONNX RUNTIME API & ENV
     ******************************************/
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort_api) {
        fprintf(stderr, "    ERROR: Failed to get ONNX Runtime API base!\n");
        fflush(stderr);
        return;
    }

    // ...existing code...
    OrtEnv* ort_env = nullptr;
    OrtStatus* current_status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ort_direct_fusedgemm_env", &ort_env);
    CheckOrtStatus(ort_api, current_status, "CreateEnv");
    if (!ort_env || current_status !=NULL) { 
        fprintf(stderr, "    ERROR: Failed to create ONNX Runtime Environment!\n");
        fflush(stderr);
        if (current_status) ort_api->ReleaseStatus(current_status); 
        return;
    }

    OrtAllocator* allocator = nullptr;
    current_status = ort_api->GetAllocatorWithDefaultOptions(&allocator); // Removed ort_env
    CheckOrtStatus(ort_api, current_status, "GetAllocatorWithDefaultOptions");
    if (!allocator || current_status != NULL) {
        fprintf(stderr, "    ERROR: Failed to get default allocator.\n");
// ...existing code...
        if (current_status) ort_api->ReleaseStatus(current_status);
        ort_api->ReleaseEnv(ort_env);
        return;
    }

    /******************************************
     * PREPARE INPUT OrtValueS
     ******************************************/
    OrtMemoryInfo* cpu_memory_info = nullptr;
    current_status = ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_memory_info);
    CheckOrtStatus(ort_api, current_status, "CreateCpuMemoryInfo");
    if (!cpu_memory_info || current_status != NULL) {
        fprintf(stderr, "    ERROR: Failed to create CPU Memory Info.\n");
        if (current_status) ort_api->ReleaseStatus(current_status);
        if (allocator) ort_api->ReleaseAllocator(allocator);
        ort_api->ReleaseEnv(ort_env);
        return;
    }

    OrtValue* input_A_val = nullptr;
    OrtValue* input_B_val = nullptr;
    OrtValue* input_C_val = nullptr; // For Bias
    float dummy_c_scalar_data = 0.0f;
    int64_t dummy_c_dims[] = {}; // Scalar shape

    // Input A
    current_status = ort_api->CreateTensorWithDataAsOrtValue(cpu_memory_info,
                                       const_cast<void*>(omTensorGetDataPtr(A_omTensor)),
                                       omTensorGetNumElems(A_omTensor) * sizeof(float),
                                       omTensorGetShape(A_omTensor),
                                       omTensorGetRank(A_omTensor),
                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_A_val);
    CheckOrtStatus(ort_api, current_status, "CreateTensorWithDataAsOrtValue for A");

    // Input B
    current_status = ort_api->CreateTensorWithDataAsOrtValue(cpu_memory_info,
                                       const_cast<void*>(omTensorGetDataPtr(B_omTensor)),
                                       omTensorGetNumElems(B_omTensor) * sizeof(float),
                                       omTensorGetShape(B_omTensor),
                                       omTensorGetRank(B_omTensor),
                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_B_val);
    CheckOrtStatus(ort_api, current_status, "CreateTensorWithDataAsOrtValue for B");

    // Input C (Bias)
    if (Bias_omTensor && omTensorGetDataPtr(Bias_omTensor)) {
        current_status = ort_api->CreateTensorWithDataAsOrtValue(cpu_memory_info,
                                           const_cast<void*>(omTensorGetDataPtr(Bias_omTensor)),
                                           omTensorGetNumElems(Bias_omTensor) * sizeof(float),
                                           omTensorGetShape(Bias_omTensor),
                                           omTensorGetRank(Bias_omTensor),
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_C_val);
        CheckOrtStatus(ort_api, current_status, "CreateTensorWithDataAsOrtValue for Bias/C");
    } else {
        fprintf(stderr, "    Bias_omTensor is NULL or has no data, providing dummy scalar 0.0f for C input.\n");
        current_status = ort_api->CreateTensorWithDataAsOrtValue(cpu_memory_info,
                                           &dummy_c_scalar_data, sizeof(float),
                                           dummy_c_dims, 0, // Scalar has 0 dimensions
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_C_val);
        CheckOrtStatus(ort_api, current_status, "CreateTensorWithDataAsOrtValue for dummy C");
    }

    if (!input_A_val || !input_B_val || !input_C_val) {
        fprintf(stderr, "    ERROR: Failed to create one or more input OrtValues.\n");
        if (input_A_val) ort_api->ReleaseValue(input_A_val);
        if (input_B_val) ort_api->ReleaseValue(input_B_val);
        if (input_C_val) ort_api->ReleaseValue(input_C_val);
        ort_api->ReleaseMemoryInfo(cpu_memory_info);
        if (allocator) ort_api->ReleaseAllocator(allocator);
        ort_api->ReleaseEnv(ort_env);
        return;
    }
    std::vector<const OrtValue*> ort_inputs = {input_A_val, input_B_val, input_C_val};

    /******************************************
     * PREPARE FusedGemm ATTRIBUTES
     ******************************************/
    OrtOpAttr* attr_alpha = nullptr;
    OrtOpAttr* attr_beta = nullptr;
    OrtOpAttr* attr_transA = nullptr;
    OrtOpAttr* attr_transB = nullptr;
    float alpha_val = GEMM_ALPHA; 
    float beta_val = GEMM_BETA;   

    current_status = ort_api->CreateOpAttr("alpha", &alpha_val, sizeof(float), ORT_OP_ATTR_FLOAT, &attr_alpha);
    CheckOrtStatus(ort_api, current_status, "CreateOpAttr alpha");
    current_status = ort_api->CreateOpAttr("beta", &beta_val, sizeof(float), ORT_OP_ATTR_FLOAT, &attr_beta);
    CheckOrtStatus(ort_api, current_status, "CreateOpAttr beta");
    current_status = ort_api->CreateOpAttr("transA", &transA_param, sizeof(int64_t), ORT_OP_ATTR_INT, &attr_transA);
    CheckOrtStatus(ort_api, current_status, "CreateOpAttr transA");
    current_status = ort_api->CreateOpAttr("transB", &transB_param, sizeof(int64_t), ORT_OP_ATTR_INT, &attr_transB);
    CheckOrtStatus(ort_api, current_status, "CreateOpAttr transB");

    if (!attr_alpha || !attr_beta || !attr_transA || !attr_transB) {
        fprintf(stderr, "    ERROR: Failed to create one or more OrtOpAttrs.\n");
        if(attr_alpha) ort_api->ReleaseOpAttr(attr_alpha);
        if(attr_beta) ort_api->ReleaseOpAttr(attr_beta);
        if(attr_transA) ort_api->ReleaseOpAttr(attr_transA);
        if(attr_transB) ort_api->ReleaseOpAttr(attr_transB);
        ort_api->ReleaseValue(input_A_val);
        ort_api->ReleaseValue(input_B_val);
        ort_api->ReleaseValue(input_C_val);
        ort_api->ReleaseMemoryInfo(cpu_memory_info);
        if (allocator) ort_api->ReleaseAllocator(allocator);
        ort_api->ReleaseEnv(ort_env);
        return;
    }
    std::vector<const OrtOpAttr*> op_attrs = {attr_alpha, attr_beta, attr_transA, attr_transB};

    /******************************************
     * CREATE FusedGemm OPERATOR & OUTPUT VALUE
     ******************************************/
    OrtOp* fused_gemm_op = nullptr;
    OrtValue* output_Y_gemm_val = nullptr;
    std::vector<int64_t> output_dims = {M, N}; // Define output_dims here

    current_status = ort_api->CreateOp(nullptr, 
                                     "FusedGemm", "com.microsoft", 1,
                                     nullptr, nullptr, 0, 
                                     op_attrs.data(), op_attrs.size(),
                                     3, 1, 
                                     &fused_gemm_op);
    CheckOrtStatus(ort_api, current_status, "CreateOp FusedGemm");

    if (fused_gemm_op && current_status == NULL) {
        current_status = ort_api->CreateTensorAsOrtValue(allocator, // Use the obtained allocator
                                         output_dims.data(), output_dims.size(),
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                         &output_Y_gemm_val);
        CheckOrtStatus(ort_api, current_status, "CreateTensorAsOrtValue for FusedGemm output Y_gemm");
    }


    if (!fused_gemm_op || !output_Y_gemm_val || current_status != NULL) {
        fprintf(stderr, "    ERROR: Failed to create FusedGemm OrtOp or its output OrtValue.\n");
        if (fused_gemm_op) ort_api->ReleaseOp(fused_gemm_op);
        if (output_Y_gemm_val) ort_api->ReleaseValue(output_Y_gemm_val);
    } else {
        /******************************************
         * INVOKE FusedGemm OPERATOR
         ******************************************/
        fprintf(stderr, "    Invoking ORT FusedGemm operator...\n");
        fflush(stderr);
        std::vector<OrtValue*> ort_outputs = {output_Y_gemm_val}; 
        current_status = ort_api->InvokeOp(nullptr, 
                                         fused_gemm_op,
                                         ort_inputs.data(), ort_inputs.size(),
                                         ort_outputs.data(), ort_outputs.size());
        CheckOrtStatus(ort_api, current_status, "InvokeOp FusedGemm");

        /******************************************
         * PROCESS OUTPUT & MANUAL RELU
         ******************************************/
        if (current_status == NULL) {
            float* gemm_result_data_ptr = nullptr;
            current_status = ort_api->GetTensorMutableData(output_Y_gemm_val, (void**)&gemm_result_data_ptr);
            CheckOrtStatus(ort_api, current_status, "GetTensorMutableData for FusedGemm output");

            if (current_status == NULL && gemm_result_data_ptr) {
                float* y_target_data_ptr = static_cast<float*>(omTensorGetDataPtr(Y_omTensor));
                size_t num_output_elements = omTensorGetNumElems(Y_omTensor); 

                size_t gemm_out_elems = 1;
                for(int64_t dim_val : output_dims) gemm_out_elems *= static_cast<size_t>(dim_val);

                if (num_output_elements != gemm_out_elems) {
                    fprintf(stderr, "    ERROR: Mismatch between Y_omTensor elements (%zu) and FusedGemm output elements (%zu).\n", num_output_elements, gemm_out_elems);
                } else {
                    memcpy(y_target_data_ptr, gemm_result_data_ptr, num_output_elements * sizeof(float));
                    fprintf(stderr, "    Applying ReLU manually to FusedGemm output...\n");
                    for (size_t i = 0; i < num_output_elements; ++i) {
                        y_target_data_ptr[i] = relu(y_target_data_ptr[i]);
                    }
                }
            }
        } else {
            fprintf(stderr, "    Skipping output processing and ReLU due to FusedGemm InvokeOp failure.\n");
        }
    }
    fflush(stderr);

    /******************************************
     * CLEANUP ONNX RUNTIME RESOURCES
     ******************************************/
    if (fused_gemm_op) ort_api->ReleaseOp(fused_gemm_op);
    if (output_Y_gemm_val) ort_api->ReleaseValue(output_Y_gemm_val);
    if (attr_alpha) ort_api->ReleaseOpAttr(attr_alpha);
    if (attr_beta) ort_api->ReleaseOpAttr(attr_beta);
    if (attr_transA) ort_api->ReleaseOpAttr(attr_transA);
    if (attr_transB) ort_api->ReleaseOpAttr(attr_transB);
    if (input_A_val) ort_api->ReleaseValue(input_A_val);
    if (input_B_val) ort_api->ReleaseValue(input_B_val);
    if (input_C_val) ort_api->ReleaseValue(input_C_val);
    if (cpu_memory_info) ort_api->ReleaseMemoryInfo(cpu_memory_info);
    if (allocator) ort_api->ReleaseAllocator(allocator);
    if (ort_env) ort_api->ReleaseEnv(ort_env);

    fprintf(stderr, ">>> Finished C++ ort_cpu_ep_fused_gemm (ORT Direct Call Wrapper) <<<\n");
    fflush(stderr);
}