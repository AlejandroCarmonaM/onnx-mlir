/**********************************************
 * IMPORT LIBRARIES
 **********************************************/
#include <stdio.h>
#include <stdlib.h>
#include "onnxruntime_c_api.h"

/**********************************************
 * CONSTANTS & PARAMETERS
 **********************************************/
// No specific constants for this minimal check

/**********************************************
 * HELPER FUNCTIONS & STRUCTS
 **********************************************/
// Minimal status check to avoid early exit for this test
void CheckOrtStatus(const OrtApi* ort_api, OrtStatus* status, const char* operation_name) {
    if (status != NULL) {
        const char* msg = ort_api->GetErrorMessage(status);
        fprintf(stderr, "ERROR during %s: %s\n", operation_name, msg);
        ort_api->ReleaseStatus(status);
        // For this minimal test, we might not want to exit immediately
        // to see if other steps can proceed or also fail.
        // exit(1); 
    } else {
        printf("SUCCESS: %s\n", operation_name);
    }
}

/**********************************************
 * MAIN PROGRAM / EXECUTION LOGIC
 **********************************************/
int main() {
    /******************************************
     * INITIALIZE ONNX RUNTIME API
     ******************************************/
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "Failed to get ONNX Runtime API base.\n");
        return 1;
    }
    printf("Successfully got ONNX Runtime API base.\n");
    printf("ORT_API_VERSION from header: %d\n", ORT_API_VERSION);

    /******************************************
     * CREATE ENVIRONMENT
     ******************************************/
    OrtEnv* env;
    OrtStatus* env_status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "MinimalVersionCheck", &env);
    CheckOrtStatus(g_ort, env_status, "CreateEnv");
    if (env_status != NULL) { // If CreateEnv failed, can't proceed
        fprintf(stderr, "Aborting due to CreateEnv failure.\n");
        return 1;
    }

    /******************************************
     * GET AND PRINT LIBRARY VERSION
     ******************************************/
    const char* build_info_string = g_ort->GetBuildInfoString();
    if (build_info_string) {
        printf("ONNX Runtime Build Info String: %s\n", build_info_string);
    } else {
        fprintf(stderr, "Failed to get ONNX Runtime build info string from the library.\n");
    }

    /******************************************
     * CLEANUP
     ******************************************/
    if (env) {
        g_ort->ReleaseEnv(env);
        printf("Released Env.\n");
    }

    printf("Minimal ONNX Runtime check finished.\n");
    return 0;
}