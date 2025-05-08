diff --git a/.github/copilot-instructions.md b/.github/copilot-instructions.md
new file mode 100644
index 0000000..b50d90e
--- /dev/null
+++ b/.github/copilot-instructions.md
@@ -0,0 +1,83 @@
+**Context:**  
+You are integrating a custom ONNX `FusedGemm` operator into the ONNX-MLIR pipeline. Your workflow involves lowering an ONNX `Custom` op through MLIR, generating a `KrnlCallOp` that ultimately calls your C++ function `ort_cpu_ep_fused_gemm`.
+
+**Goal:**  
+Your code must be behaviour oriented, well separated and divided into sections with both func defs and comments separating sections for easier code dissecting. Try and follow this boilerplate for c++:
+
+"/**********************************************
+ * IMPORT LIBRARIES
+ **********************************************/
+
+/*
+Libraries and tools used in this script, along with version info when applicable.
+Example:
+#include <vector>   // C++ Standard Library
+#include <iostream> // For input/output
+*/
+
+#include <iostream>  // For std::cout, std::endl
+#include <string>    // For std::string
+// Add additional libraries as needed
+
+/**********************************************
+ * CONSTANTS & PARAMETERS
+ **********************************************/
+
+/*
+Constants and parameters used in this script.
+Example:
+const std::string DATA_PATH = "/path/to/data";
+const double THRESHOLD = 0.5;
+*/
+
+const int EXAMPLE_CONSTANT = 42;
+
+/**********************************************
+ * FUNCTION DEFINITIONS
+ **********************************************/
+
+/*
+Function documentation:
+Purpose: Example function to demonstrate structure.
+Parameters:
+    - param1 (int): Description of param1.
+    - param2 (int): Description of param2.
+Returns:
+    - int: Description of the return value.
+*/
+
+int exampleFunction(int param1, int param2) {
+    // Function logic here
+    return param1 + param2;
+}
+
+/**********************************************
+ * MAIN PROGRAM
+ **********************************************/
+
+int main() {
+    /******************************************
+     * INITIALIZE DATA
+     ******************************************/
+    int a = 1;
+    int b = 2;
+
+    /******************************************
+     * PERFORM ACTIONS
+     ******************************************/
+    int result = exampleFunction(a, b);
+
+    /******************************************
+     * OUTPUT RESULTS
+     ******************************************/
+    std::cout << "Result: " << result << std::endl;
+
+    /******************************************
+     * CLEANUP (if necessary)
+     ******************************************/
+    // Nothing to clean up in this example
+
+    return 0;
+}
+"
+Adapt if necesary to other languages when necessary
\ No newline at end of file
diff --git a/onnxruntime_docs/ortAPI.md b/onnxruntime_docs/ortAPI.md
new file mode 100644
index 0000000..31edd14
--- /dev/null
+++ b/onnxruntime_docs/ortAPI.md
@@ -0,0 +1,849 @@
+OrtApi Struct Reference
+
+#include <onnxruntime_c_api.h>
+
+Public Member Functions
+OrtStatus * 	SynchronizeBoundInputs (OrtIoBinding *binding_ptr)
+ 	Synchronize bound inputs. The call may be necessary for some providers, such as cuda, in case the system that allocated bound memory operated on a different stream. However, the operation is provider specific and could be a no-op.
+ 
+OrtStatus * 	SynchronizeBoundOutputs (OrtIoBinding *binding_ptr)
+ 	Synchronize bound outputs. The call may be necessary for some providers, such as cuda, in case the system that allocated bound memory operated on a different stream. However, the operation is provider specific and could be a no-op.
+ 
+OrtStatus * 	SessionOptionsAppendExecutionProvider_MIGraphX (OrtSessionOptions *options, const OrtMIGraphXProviderOptions *migraphx_options)
+ 	Append MIGraphX provider to session options.
+ 
+OrtStatus * 	AddExternalInitializers (OrtSessionOptions *options, const char *const *initializer_names, const OrtValue *const *initializers, size_t num_initializers)
+ 	Replace initialized Tensors with external data with the data provided in initializers.
+ 
+OrtStatus * 	CreateOpAttr (const char *name, const void *data, int len, OrtOpAttrType type, OrtOpAttr **op_attr)
+ 	: Create attribute of onnxruntime operator
+ 
+void 	ReleaseOpAttr (OrtOpAttr *input)
+ 
+OrtStatus * 	CreateOp (const OrtKernelInfo *info, const char *op_name, const char *domain, int version, const char **type_constraint_names, const ONNXTensorElementDataType *type_constraint_values, int type_constraint_count, const OrtOpAttr *const *attr_values, int attr_count, int input_count, int output_count, OrtOp **ort_op)
+ 	: Create onnxruntime native operator
+ 
+OrtStatus * 	InvokeOp (const OrtKernelContext *context, const OrtOp *ort_op, const OrtValue *const *input_values, int input_count, OrtValue *const *output_values, int output_count)
+ 	: Invoke the operator created by OrtApi::CreateOp The inputs must follow the order as specified in onnx specification
+ 
+void 	ReleaseOp (OrtOp *input)
+ 
+OrtStatus * 	SessionOptionsAppendExecutionProvider (OrtSessionOptions *options, const char *provider_name, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
+ 	: Append execution provider to the session options.
+ 
+OrtStatus * 	CopyKernelInfo (const OrtKernelInfo *info, OrtKernelInfo **info_copy)
+ 
+void 	ReleaseKernelInfo (OrtKernelInfo *input)
+ 
+OrtStatus * 	SessionOptionsAppendExecutionProvider_CANN (OrtSessionOptions *options, const OrtCANNProviderOptions *cann_options)
+ 	Append CANN provider to session options.
+ 
+OrtStatus * 	CreateCANNProviderOptions (OrtCANNProviderOptions **out)
+ 	Create an OrtCANNProviderOptions.
+ 
+OrtStatus * 	UpdateCANNProviderOptions (OrtCANNProviderOptions *cann_options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
+ 	Set options in a CANN Execution Provider.
+ 
+OrtStatus * 	GetCANNProviderOptionsAsString (const OrtCANNProviderOptions *cann_options, OrtAllocator *allocator, char **ptr)
+ 	Get serialized CANN provider options string.
+ 
+OrtStatus * 	UpdateEnvWithCustomLogLevel (OrtEnv *ort_env, OrtLoggingLevel log_severity_level)
+ 
+OrtStatus * 	SetGlobalIntraOpThreadAffinity (OrtThreadingOptions *tp_options, const char *affinity_string)
+ 
+OrtStatus * 	RegisterCustomOpsLibrary_V2 (OrtSessionOptions *options, const char *library_name)
+ 	Register custom ops from a shared library.
+ 
+OrtStatus * 	RegisterCustomOpsUsingFunction (OrtSessionOptions *options, const char *registration_func_name)
+ 	Register custom ops by calling a RegisterCustomOpsFn function.
+ 
+OrtStatus * 	SessionOptionsAppendExecutionProvider_Dnnl (OrtSessionOptions *options, const OrtDnnlProviderOptions *dnnl_options)
+ 	Append dnnl provider to session options.
+ 
+OrtStatus * 	CreateDnnlProviderOptions (OrtDnnlProviderOptions **out)
+ 	Create an OrtDnnlProviderOptions.
+ 
+OrtStatus * 	UpdateDnnlProviderOptions (OrtDnnlProviderOptions *dnnl_options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
+ 	Set options in a oneDNN Execution Provider.
+ 
+OrtStatus * 	GetDnnlProviderOptionsAsString (const OrtDnnlProviderOptions *dnnl_options, OrtAllocator *allocator, char **ptr)
+ 
+OrtStatus * 	KernelInfoGetConstantInput_tensor (const OrtKernelInfo *info, size_t index, int *is_constant, const OrtValue **out)
+ 	Get a OrtValue tensor stored as a constant initializer in the graph node.
+ 
+OrtStatus * 	CastTypeInfoToOptionalTypeInfo (const OrtTypeInfo *type_info, const OrtOptionalTypeInfo **out)
+ 	Get Optional Type information from an OrtTypeInfo.
+ 
+OrtStatus * 	GetOptionalContainedTypeInfo (const OrtOptionalTypeInfo *optional_type_info, OrtTypeInfo **out)
+ 	Get OrtTypeInfo for the allowed contained type from an OrtOptionalTypeInfo.
+ 
+OrtStatus * 	GetResizedStringTensorElementBuffer (OrtValue *value, size_t index, size_t length_in_bytes, char **buffer)
+ 	Set a single string in a string tensor Do not zero terminate the string data.
+ 
+OrtStatus * 	KernelContext_GetAllocator (const OrtKernelContext *context, const OrtMemoryInfo *mem_info, OrtAllocator **out)
+ 	Get Allocator from KernelContext for a specific memoryInfo. Please use C API ReleaseAllocator to release out object.
+ 
+Public Attributes
+void(* 	ReleaseCANNProviderOptions )(OrtCANNProviderOptions *input)
+ 	Release an OrtCANNProviderOptions.
+ 
+void(* 	MemoryInfoGetDeviceType )(const OrtMemoryInfo *ptr, OrtMemoryInfoDeviceType *out)
+ 
+void(* 	ReleaseDnnlProviderOptions )(OrtDnnlProviderOptions *input)
+ 	Release an OrtDnnlProviderOptions.
+ 
+const char *(* 	GetBuildInfoString )(void)
+ 	Returns a null terminated string of the build info including git info and cxx flags.
+ 
+OrtStatus
+OrtStatus *(* 	CreateStatus )(OrtErrorCode code, const char *msg) __attribute__((nonnull))
+ 	Create an OrtStatus from a null terminated string.
+ 
+OrtErrorCode(* 	GetErrorCode )(const OrtStatus *status) __attribute__((nonnull))
+ 	Get OrtErrorCode from OrtStatus.
+ 
+const char *(* 	GetErrorMessage )(const OrtStatus *status) __attribute__((nonnull))
+ 	Get error string from OrtStatus.
+ 
+void 	ReleaseStatus (OrtStatus *input)
+ 
+OrtIoBinding
+void(* 	ClearBoundInputs )(OrtIoBinding *binding_ptr) __attribute__((nonnull))
+ 	Clears any previously set Inputs for an OrtIoBinding.
+ 
+void(* 	ClearBoundOutputs )(OrtIoBinding *binding_ptr) __attribute__((nonnull))
+ 	Clears any previously set Outputs for an OrtIoBinding.
+ 
+void 	ReleaseIoBinding (OrtIoBinding *input)
+ 	Release an OrtIoBinding obtained from OrtApi::CreateIoBinding.
+ 
+OrtStatus * 	BindInput (OrtIoBinding *binding_ptr, const char *name, const OrtValue *val_ptr)
+ 	Bind an OrtValue to an OrtIoBinding input.
+ 
+OrtStatus * 	BindOutput (OrtIoBinding *binding_ptr, const char *name, const OrtValue *val_ptr)
+ 	Bind an OrtValue to an OrtIoBinding output.
+ 
+OrtStatus * 	BindOutputToDevice (OrtIoBinding *binding_ptr, const char *name, const OrtMemoryInfo *mem_info_ptr)
+ 	Bind an OrtIoBinding output to a device.
+ 
+OrtStatus * 	GetBoundOutputNames (const OrtIoBinding *binding_ptr, OrtAllocator *allocator, char **buffer, size_t **lengths, size_t *count)
+ 	Get the names of an OrtIoBinding's outputs.
+ 
+OrtStatus * 	GetBoundOutputValues (const OrtIoBinding *binding_ptr, OrtAllocator *allocator, OrtValue ***output, size_t *output_count)
+ 	Get the output OrtValue objects from an OrtIoBinding.
+ 
+OrtTensorRTProviderOptionsV2
+void(* 	ReleaseTensorRTProviderOptions )(OrtTensorRTProviderOptionsV2 *input)
+ 	Release an OrtTensorRTProviderOptionsV2.
+ 
+OrtStatus * 	CreateTensorRTProviderOptions (OrtTensorRTProviderOptionsV2 **out)
+ 	Create an OrtTensorRTProviderOptionsV2.
+ 
+OrtStatus * 	UpdateTensorRTProviderOptions (OrtTensorRTProviderOptionsV2 *tensorrt_options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
+ 	Set options in a TensorRT Execution Provider.
+ 
+OrtStatus * 	GetTensorRTProviderOptionsAsString (const OrtTensorRTProviderOptionsV2 *tensorrt_options, OrtAllocator *allocator, char **ptr)
+ 	Get serialized TensorRT provider options string.
+ 
+OrtCUDAProviderOptionsV2
+void(* 	ReleaseCUDAProviderOptions )(OrtCUDAProviderOptionsV2 *input)
+ 	Release an OrtCUDAProviderOptionsV2.
+ 
+OrtStatus * 	CreateCUDAProviderOptions (OrtCUDAProviderOptionsV2 **out)
+ 	Create an OrtCUDAProviderOptionsV2.
+ 
+OrtStatus * 	UpdateCUDAProviderOptions (OrtCUDAProviderOptionsV2 *cuda_options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
+ 	Set options in a CUDA Execution Provider.
+ 
+OrtStatus * 	GetCUDAProviderOptionsAsString (const OrtCUDAProviderOptionsV2 *cuda_options, OrtAllocator *allocator, char **ptr)
+ 
+Ort Training
+const OrtTrainingApi *(* 	GetTrainingApi )(uint32_t version)
+ 	Gets the Training C Api struct.
+ 
+OrtROCMProviderOptions
+void(* 	ReleaseROCMProviderOptions )(OrtROCMProviderOptions *input)
+ 	Release an OrtROCMProviderOptions.
+ 
+OrtStatus * 	CreateROCMProviderOptions (OrtROCMProviderOptions **out)
+ 	Create an OrtROCMProviderOptions.
+ 
+OrtStatus * 	UpdateROCMProviderOptions (OrtROCMProviderOptions *rocm_options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
+ 	Set options in a ROCm Execution Provider.
+ 
+OrtStatus * 	GetROCMProviderOptionsAsString (const OrtROCMProviderOptions *rocm_options, OrtAllocator *allocator, char **ptr)
+ 
+OrtStatus * 	CreateAndRegisterAllocatorV2 (OrtEnv *env, const char *provider_type, const OrtMemoryInfo *mem_info, const OrtArenaCfg *arena_cfg, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
+ 	Create an allocator with specific type and register it with the OrtEnv This API enhance CreateAndRegisterAllocator that it can create an allocator with specific type, not just CPU allocator Enables sharing the allocator between multiple sessions that use the same env instance. Lifetime of the created allocator will be valid for the duration of the environment. Returns an error if an allocator with the same OrtMemoryInfo is already registered.
+ 
+OrtStatus * 	RunAsync (OrtSession *session, const OrtRunOptions *run_options, const char *const *input_names, const OrtValue *const *input, size_t input_len, const char *const *output_names, size_t output_names_len, OrtValue **output, RunAsyncCallbackFn run_async_callback, void *user_data)
+ 	Run the model asynchronously in a thread owned by intra op thread pool.
+ 
+OrtStatus * 	UpdateTensorRTProviderOptionsWithValue (OrtTensorRTProviderOptionsV2 *tensorrt_options, const char *key, void *value)
+ 
+OrtStatus * 	GetTensorRTProviderOptionsByName (const OrtTensorRTProviderOptionsV2 *tensorrt_options, const char *key, void **ptr)
+ 
+OrtStatus * 	UpdateCUDAProviderOptionsWithValue (OrtCUDAProviderOptionsV2 *cuda_options, const char *key, void *value)
+ 
+OrtStatus * 	GetCUDAProviderOptionsByName (const OrtCUDAProviderOptionsV2 *cuda_options, const char *key, void **ptr)
+ 
+OrtStatus * 	KernelContext_GetResource (const OrtKernelContext *context, int resource_version, int resource_id, void **resource)
+ 
+OrtStatus * 	SetUserLoggingFunction (OrtSessionOptions *options, OrtLoggingFunction user_logging_function, void *user_logging_param)
+ 	Set user logging function.
+ 
+OrtStatus * 	ShapeInferContext_GetInputCount (const OrtShapeInferContext *context, size_t *out)
+ 
+OrtStatus * 	ShapeInferContext_GetInputTypeShape (const OrtShapeInferContext *context, size_t index, OrtTensorTypeAndShapeInfo **info)
+ 
+OrtStatus * 	ShapeInferContext_GetAttribute (const OrtShapeInferContext *context, const char *attr_name, const OrtOpAttr **attr)
+ 
+OrtStatus * 	ShapeInferContext_SetOutputTypeShape (const OrtShapeInferContext *context, size_t index, const OrtTensorTypeAndShapeInfo *info)
+ 
+OrtStatus * 	SetSymbolicDimensions (OrtTensorTypeAndShapeInfo *info, const char *dim_params[], size_t dim_params_length)
+ 
+OrtStatus * 	ReadOpAttr (const OrtOpAttr *op_attr, OrtOpAttrType type, void *data, size_t len, size_t *out)
+ 
+OrtStatus * 	SetDeterministicCompute (OrtSessionOptions *options, bool value)
+ 	Set whether to use deterministic compute.
+ 
+OrtStatus * 	KernelContext_ParallelFor (const OrtKernelContext *context, void(*fn)(void *, size_t), size_t total, size_t num_batch, void *usr_data)
+ 
+OrtStatus * 	SessionOptionsAppendExecutionProvider_OpenVINO_V2 (OrtSessionOptions *options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
+ 	Append OpenVINO execution provider to the session options.
+ 
+OrtStatus * 	SessionOptionsAppendExecutionProvider_VitisAI (OrtSessionOptions *options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
+ 	Append VitisAI provider to session options.
+ 
+OrtStatus * 	KernelContext_GetScratchBuffer (const OrtKernelContext *context, const OrtMemoryInfo *mem_info, size_t count_or_bytes, void **out)
+ 	Get scratch buffer from the corresponding allocator under the sepcific OrtMemoryInfo object. NOTE: callers are responsible to release this scratch buffer from the corresponding allocator.
+ 
+OrtStatus * 	KernelInfoGetAllocator (const OrtKernelInfo *info, OrtMemType mem_type, OrtAllocator **out)
+ 	Get allocator from KernelInfo for a specific memory type. Please use C API ReleaseAllocator to release out object.
+ 
+OrtStatus * 	AddExternalInitializersFromFilesInMemory (OrtSessionOptions *options, const char *const *external_initializer_file_names, char *const *external_initializer_file_buffer_array, const size_t *external_initializer_file_lengths, size_t num_external_initializer_files)
+ 	Replace initialized Tensors with external data with the provided files in memory.
+ 
+OrtStatus * 	CreateLoraAdapter (const char *adapter_file_path, OrtAllocator *allocator, OrtLoraAdapter **out)
+ 	Create an OrtLoraAdapter.
+ 
+OrtStatus * 	CreateLoraAdapterFromArray (const void *bytes, size_t num_bytes, OrtAllocator *allocator, OrtLoraAdapter **out)
+ 	Create an OrtLoraAdapter.
+ 
+void 	ReleaseLoraAdapter (OrtLoraAdapter *input)
+ 	Release an OrtLoraAdapter obtained from OrtApi::CreateLoraAdapter.
+ 
+OrtStatus * 	RunOptionsAddActiveLoraAdapter (OrtRunOptions *options, const OrtLoraAdapter *adapter)
+ 	Add the Lora Adapter to the list of active adapters.
+ 
+OrtEnv
+OrtStatus * 	CreateEnv (OrtLoggingLevel log_severity_level, const char *logid, OrtEnv **out)
+ 	Create an OrtEnv.
+ 
+OrtStatus * 	CreateEnvWithCustomLogger (OrtLoggingFunction logging_function, void *logger_param, OrtLoggingLevel log_severity_level, const char *logid, OrtEnv **out)
+ 	Create an OrtEnv.
+ 
+OrtStatus * 	EnableTelemetryEvents (const OrtEnv *env)
+ 	Enable Telemetry.
+ 
+OrtStatus * 	DisableTelemetryEvents (const OrtEnv *env)
+ 	Disable Telemetry.
+ 
+void 	ReleaseEnv (OrtEnv *input)
+ 
+OrtStatus * 	CreateEnvWithGlobalThreadPools (OrtLoggingLevel log_severity_level, const char *logid, const OrtThreadingOptions *tp_options, OrtEnv **out)
+ 	Create an OrtEnv.
+ 
+OrtStatus * 	CreateAndRegisterAllocator (OrtEnv *env, const OrtMemoryInfo *mem_info, const OrtArenaCfg *arena_cfg)
+ 	Create an allocator and register it with the OrtEnv.
+ 
+OrtStatus * 	SetLanguageProjection (const OrtEnv *ort_env, OrtLanguageProjection projection)
+ 	Set language projection.
+ 
+OrtStatus * 	CreateEnvWithCustomLoggerAndGlobalThreadPools (OrtLoggingFunction logging_function, void *logger_param, OrtLoggingLevel log_severity_level, const char *logid, const struct OrtThreadingOptions *tp_options, OrtEnv **out)
+ 
+OrtSession
+OrtStatus * 	CreateSession (const OrtEnv *env, const char *model_path, const OrtSessionOptions *options, OrtSession **out)
+ 	Create an OrtSession from a model file.
+ 
+OrtStatus * 	CreateSessionFromArray (const OrtEnv *env, const void *model_data, size_t model_data_length, const OrtSessionOptions *options, OrtSession **out)
+ 	Create an OrtSession from memory.
+ 
+OrtStatus * 	Run (OrtSession *session, const OrtRunOptions *run_options, const char *const *input_names, const OrtValue *const *inputs, size_t input_len, const char *const *output_names, size_t output_names_len, OrtValue **outputs)
+ 	Run the model in an OrtSession.
+ 
+OrtStatus * 	SessionGetInputCount (const OrtSession *session, size_t *out)
+ 	Get input count for a session.
+ 
+OrtStatus * 	SessionGetOutputCount (const OrtSession *session, size_t *out)
+ 	Get output count for a session.
+ 
+OrtStatus * 	SessionGetOverridableInitializerCount (const OrtSession *session, size_t *out)
+ 	Get overridable initializer count.
+ 
+OrtStatus * 	SessionGetInputTypeInfo (const OrtSession *session, size_t index, OrtTypeInfo **type_info)
+ 	Get input type information.
+ 
+OrtStatus * 	SessionGetOutputTypeInfo (const OrtSession *session, size_t index, OrtTypeInfo **type_info)
+ 	Get output type information.
+ 
+OrtStatus * 	SessionGetOverridableInitializerTypeInfo (const OrtSession *session, size_t index, OrtTypeInfo **type_info)
+ 	Get overridable initializer type information.
+ 
+OrtStatus * 	SessionGetInputName (const OrtSession *session, size_t index, OrtAllocator *allocator, char **value)
+ 	Get input name.
+ 
+OrtStatus * 	SessionGetOutputName (const OrtSession *session, size_t index, OrtAllocator *allocator, char **value)
+ 	Get output name.
+ 
+OrtStatus * 	SessionGetOverridableInitializerName (const OrtSession *session, size_t index, OrtAllocator *allocator, char **value)
+ 	Get overridable initializer name.
+ 
+void 	ReleaseSession (OrtSession *input)
+ 
+OrtStatus * 	SessionEndProfiling (OrtSession *session, OrtAllocator *allocator, char **out)
+ 	End profiling and return filename of the profile data.
+ 
+OrtStatus * 	SessionGetModelMetadata (const OrtSession *session, OrtModelMetadata **out)
+ 	Get OrtModelMetadata from an OrtSession.
+ 
+OrtStatus * 	RunWithBinding (OrtSession *session, const OrtRunOptions *run_options, const OrtIoBinding *binding_ptr)
+ 	Run a model using Io Bindings for the inputs & outputs.
+ 
+OrtStatus * 	CreateIoBinding (OrtSession *session, OrtIoBinding **out)
+ 	Create an OrtIoBinding instance.
+ 
+OrtStatus * 	SessionGetProfilingStartTimeNs (const OrtSession *session, uint64_t *out)
+ 	Return the time that profiling was started.
+ 
+OrtStatus * 	CreateSessionWithPrepackedWeightsContainer (const OrtEnv *env, const char *model_path, const OrtSessionOptions *options, OrtPrepackedWeightsContainer *prepacked_weights_container, OrtSession **out)
+ 	Create session with prepacked weights container.
+ 
+OrtStatus * 	CreateSessionFromArrayWithPrepackedWeightsContainer (const OrtEnv *env, const void *model_data, size_t model_data_length, const OrtSessionOptions *options, OrtPrepackedWeightsContainer *prepacked_weights_container, OrtSession **out)
+ 	Create session from memory with prepacked weights container.
+ 
+OrtSessionOptions
+Custom operator APIs
+
+OrtStatus * 	CreateSessionOptions (OrtSessionOptions **options)
+ 	Create an OrtSessionOptions object.
+ 
+OrtStatus * 	SetOptimizedModelFilePath (OrtSessionOptions *options, const char *optimized_model_filepath)
+ 	Set filepath to save optimized model after graph level transformations.
+ 
+OrtStatus * 	CloneSessionOptions (const OrtSessionOptions *in_options, OrtSessionOptions **out_options)
+ 	Create a copy of an existing OrtSessionOptions.
+ 
+OrtStatus * 	SetSessionExecutionMode (OrtSessionOptions *options, ExecutionMode execution_mode)
+ 	Set execution mode.
+ 
+OrtStatus * 	EnableProfiling (OrtSessionOptions *options, const char *profile_file_prefix)
+ 	Enable profiling for a session.
+ 
+OrtStatus * 	DisableProfiling (OrtSessionOptions *options)
+ 	Disable profiling for a session.
+ 
+OrtStatus * 	EnableMemPattern (OrtSessionOptions *options)
+ 	Enable the memory pattern optimization.
+ 
+OrtStatus * 	DisableMemPattern (OrtSessionOptions *options)
+ 	Disable the memory pattern optimization.
+ 
+OrtStatus * 	EnableCpuMemArena (OrtSessionOptions *options)
+ 	Enable the memory arena on CPU.
+ 
+OrtStatus * 	DisableCpuMemArena (OrtSessionOptions *options)
+ 	Disable the memory arena on CPU.
+ 
+OrtStatus * 	SetSessionLogId (OrtSessionOptions *options, const char *logid)
+ 	Set session log id.
+ 
+OrtStatus * 	SetSessionLogVerbosityLevel (OrtSessionOptions *options, int session_log_verbosity_level)
+ 	Set session log verbosity level.
+ 
+OrtStatus * 	SetSessionLogSeverityLevel (OrtSessionOptions *options, int session_log_severity_level)
+ 	Set session log severity level.
+ 
+OrtStatus * 	SetSessionGraphOptimizationLevel (OrtSessionOptions *options, GraphOptimizationLevel graph_optimization_level)
+ 	Set the optimization level to apply when loading a graph.
+ 
+OrtStatus * 	SetIntraOpNumThreads (OrtSessionOptions *options, int intra_op_num_threads)
+ 	Sets the number of threads used to parallelize the execution within nodes.
+ 
+OrtStatus * 	SetInterOpNumThreads (OrtSessionOptions *options, int inter_op_num_threads)
+ 	Sets the number of threads used to parallelize the execution of the graph.
+ 
+OrtStatus * 	AddCustomOpDomain (OrtSessionOptions *options, OrtCustomOpDomain *custom_op_domain)
+ 	Add custom op domain to a session options.
+ 
+OrtStatus * 	RegisterCustomOpsLibrary (OrtSessionOptions *options, const char *library_path, void **library_handle)
+ 
+OrtStatus * 	AddFreeDimensionOverride (OrtSessionOptions *options, const char *dim_denotation, int64_t dim_value)
+ 	Override session symbolic dimensions.
+ 
+void 	ReleaseSessionOptions (OrtSessionOptions *input)
+ 
+OrtStatus * 	DisablePerSessionThreads (OrtSessionOptions *options)
+ 	Use global thread pool on a session.
+ 
+OrtStatus * 	AddFreeDimensionOverrideByName (OrtSessionOptions *options, const char *dim_name, int64_t dim_value)
+ 
+OrtStatus * 	AddSessionConfigEntry (OrtSessionOptions *options, const char *config_key, const char *config_value)
+ 	Set a session configuration entry as a pair of strings.
+ 
+OrtStatus * 	AddInitializer (OrtSessionOptions *options, const char *name, const OrtValue *val)
+ 	Add a pre-allocated initializer to a session.
+ 
+OrtStatus * 	SessionOptionsAppendExecutionProvider_CUDA (OrtSessionOptions *options, const OrtCUDAProviderOptions *cuda_options)
+ 	Append CUDA provider to session options.
+ 
+OrtStatus * 	SessionOptionsAppendExecutionProvider_ROCM (OrtSessionOptions *options, const OrtROCMProviderOptions *rocm_options)
+ 	Append ROCM execution provider to the session options.
+ 
+OrtStatus * 	SessionOptionsAppendExecutionProvider_OpenVINO (OrtSessionOptions *options, const OrtOpenVINOProviderOptions *provider_options)
+ 	Append OpenVINO execution provider to the session options.
+ 
+OrtStatus * 	SessionOptionsAppendExecutionProvider_TensorRT (OrtSessionOptions *options, const OrtTensorRTProviderOptions *tensorrt_options)
+ 	Append TensorRT provider to session options.
+ 
+OrtStatus * 	SessionOptionsAppendExecutionProvider_TensorRT_V2 (OrtSessionOptions *options, const OrtTensorRTProviderOptionsV2 *tensorrt_options)
+ 	Append TensorRT execution provider to the session options.
+ 
+OrtStatus * 	EnableOrtCustomOps (OrtSessionOptions *options)
+ 	Enable custom operators.
+ 
+OrtStatus * 	HasValue (const OrtValue *value, int *out)
+ 	Sets out to 1 iff an optional type OrtValue has an element, 0 otherwise (OrtValue is None) Use this API to find if the optional type OrtValue is None or not. If the optional type OrtValue is not None, use the OrtValue just like any other OrtValue. For example, if you get an OrtValue that corresponds to Optional(tensor) and if HasValue() returns true, use it as tensor and so on.
+ 
+OrtStatus * 	SessionOptionsAppendExecutionProvider_CUDA_V2 (OrtSessionOptions *options, const OrtCUDAProviderOptionsV2 *cuda_options)
+ 	Append CUDA execution provider to the session options.
+ 
+OrtStatus * 	HasSessionConfigEntry (const OrtSessionOptions *options, const char *config_key, int *out)
+ 	Checks if the given session configuration entry exists.
+ 
+OrtStatus * 	GetSessionConfigEntry (const OrtSessionOptions *options, const char *config_key, char *config_value, size_t *size)
+ 	Get a session configuration value.
+ 
+OrtCustomOpDomain
+OrtStatus * 	CreateCustomOpDomain (const char *domain, OrtCustomOpDomain **out)
+ 	Create a custom op domain.
+ 
+OrtStatus * 	CustomOpDomain_Add (OrtCustomOpDomain *custom_op_domain, const OrtCustomOp *op)
+ 	Add a custom op to a custom op domain.
+ 
+void 	ReleaseCustomOpDomain (OrtCustomOpDomain *input)
+ 
+OrtRunOptions
+OrtStatus * 	CreateRunOptions (OrtRunOptions **out)
+ 	Create an OrtRunOptions.
+ 
+OrtStatus * 	RunOptionsSetRunLogVerbosityLevel (OrtRunOptions *options, int log_verbosity_level)
+ 	Set per-run log verbosity level.
+ 
+OrtStatus * 	RunOptionsSetRunLogSeverityLevel (OrtRunOptions *options, int log_severity_level)
+ 	Set per-run log severity level.
+ 
+OrtStatus * 	RunOptionsSetRunTag (OrtRunOptions *options, const char *run_tag)
+ 	Set per-run tag.
+ 
+OrtStatus * 	RunOptionsGetRunLogVerbosityLevel (const OrtRunOptions *options, int *log_verbosity_level)
+ 	Get per-run log verbosity level.
+ 
+OrtStatus * 	RunOptionsGetRunLogSeverityLevel (const OrtRunOptions *options, int *log_severity_level)
+ 	Get per-run log severity level.
+ 
+OrtStatus * 	RunOptionsGetRunTag (const OrtRunOptions *options, const char **run_tag)
+ 	Get per-run tag.
+ 
+OrtStatus * 	RunOptionsSetTerminate (OrtRunOptions *options)
+ 	Set terminate flag.
+ 
+OrtStatus * 	RunOptionsUnsetTerminate (OrtRunOptions *options)
+ 	Clears the terminate flag.
+ 
+void 	ReleaseRunOptions (OrtRunOptions *input)
+ 
+OrtStatus * 	AddRunConfigEntry (OrtRunOptions *options, const char *config_key, const char *config_value)
+ 	Set a single run configuration entry as a pair of strings.
+ 
+OrtValue
+OrtStatus * 	CreateTensorAsOrtValue (OrtAllocator *allocator, const int64_t *shape, size_t shape_len, ONNXTensorElementDataType type, OrtValue **out)
+ 	Create a tensor.
+ 
+OrtStatus * 	CreateTensorWithDataAsOrtValue (const OrtMemoryInfo *info, void *p_data, size_t p_data_len, const int64_t *shape, size_t shape_len, ONNXTensorElementDataType type, OrtValue **out)
+ 	Create a tensor backed by a user supplied buffer.
+ 
+OrtStatus * 	IsTensor (const OrtValue *value, int *out)
+ 	Return if an OrtValue is a tensor type.
+ 
+OrtStatus * 	GetTensorMutableData (OrtValue *value, void **out)
+ 	Get a pointer to the raw data inside a tensor.
+ 
+OrtStatus * 	FillStringTensor (OrtValue *value, const char *const *s, size_t s_len)
+ 	Set all strings at once in a string tensor.
+ 
+OrtStatus * 	GetStringTensorDataLength (const OrtValue *value, size_t *len)
+ 	Get total byte length for all strings in a string tensor.
+ 
+OrtStatus * 	GetStringTensorContent (const OrtValue *value, void *s, size_t s_len, size_t *offsets, size_t offsets_len)
+ 	Get all strings from a string tensor.
+ 
+OrtStatus * 	GetTensorTypeAndShape (const OrtValue *value, OrtTensorTypeAndShapeInfo **out)
+ 	Get type and shape information from a tensor OrtValue.
+ 
+OrtStatus * 	GetTypeInfo (const OrtValue *value, OrtTypeInfo **out)
+ 	Get type information of an OrtValue.
+ 
+OrtStatus * 	GetValueType (const OrtValue *value, enum ONNXType *out)
+ 	Get ONNXType of an OrtValue.
+ 
+OrtStatus * 	GetValue (const OrtValue *value, int index, OrtAllocator *allocator, OrtValue **out)
+ 	Get non tensor data from an OrtValue.
+ 
+OrtStatus * 	GetValueCount (const OrtValue *value, size_t *out)
+ 	Get non tensor value count from an OrtValue.
+ 
+OrtStatus * 	CreateValue (const OrtValue *const *in, size_t num_values, enum ONNXType value_type, OrtValue **out)
+ 	Create a map or sequence OrtValue.
+ 
+OrtStatus * 	CreateOpaqueValue (const char *domain_name, const char *type_name, const void *data_container, size_t data_container_size, OrtValue **out)
+ 	Create an opaque (custom user defined type) OrtValue.
+ 
+OrtStatus * 	GetOpaqueValue (const char *domain_name, const char *type_name, const OrtValue *in, void *data_container, size_t data_container_size)
+ 	Get internal data from an opaque (custom user defined type) OrtValue.
+ 
+void 	ReleaseValue (OrtValue *input)
+ 
+OrtStatus * 	GetStringTensorElementLength (const OrtValue *value, size_t index, size_t *out)
+ 	Get the length of a single string in a string tensor.
+ 
+OrtStatus * 	GetStringTensorElement (const OrtValue *value, size_t s_len, size_t index, void *s)
+ 	Get a single string from a string tensor.
+ 
+OrtStatus * 	FillStringTensorElement (OrtValue *value, const char *s, size_t index)
+ 	Set a single string in a string tensor.
+ 
+OrtStatus * 	TensorAt (OrtValue *value, const int64_t *location_values, size_t location_values_count, void **out)
+ 	Direct memory access to a specified tensor element.
+ 
+OrtStatus * 	IsSparseTensor (const OrtValue *value, int *out)
+ 	Sets *out to 1 iff an OrtValue is a SparseTensor, and 0 otherwise.
+ 
+OrtStatus * 	CreateSparseTensorAsOrtValue (OrtAllocator *allocator, const int64_t *dense_shape, size_t dense_shape_len, ONNXTensorElementDataType type, OrtValue **out)
+ 	Create an OrtValue with a sparse tensor that is empty.
+ 
+OrtStatus * 	FillSparseTensorCoo (OrtValue *ort_value, const OrtMemoryInfo *data_mem_info, const int64_t *values_shape, size_t values_shape_len, const void *values, const int64_t *indices_data, size_t indices_num)
+ 
+OrtStatus * 	FillSparseTensorCsr (OrtValue *ort_value, const OrtMemoryInfo *data_mem_info, const int64_t *values_shape, size_t values_shape_len, const void *values, const int64_t *inner_indices_data, size_t inner_indices_num, const int64_t *outer_indices_data, size_t outer_indices_num)
+ 
+OrtStatus * 	FillSparseTensorBlockSparse (OrtValue *ort_value, const OrtMemoryInfo *data_mem_info, const int64_t *values_shape, size_t values_shape_len, const void *values, const int64_t *indices_shape_data, size_t indices_shape_len, const int32_t *indices_data)
+ 
+OrtStatus * 	CreateSparseTensorWithValuesAsOrtValue (const OrtMemoryInfo *info, void *p_data, const int64_t *dense_shape, size_t dense_shape_len, const int64_t *values_shape, size_t values_shape_len, ONNXTensorElementDataType type, OrtValue **out)
+ 
+OrtStatus * 	UseCooIndices (OrtValue *ort_value, int64_t *indices_data, size_t indices_num)
+ 
+OrtStatus * 	UseCsrIndices (OrtValue *ort_value, int64_t *inner_data, size_t inner_num, int64_t *outer_data, size_t outer_num)
+ 
+OrtStatus * 	UseBlockSparseIndices (OrtValue *ort_value, const int64_t *indices_shape, size_t indices_shape_len, int32_t *indices_data)
+ 
+OrtStatus * 	GetSparseTensorFormat (const OrtValue *ort_value, enum OrtSparseFormat *out)
+ 	Returns sparse tensor format enum iff a given ort value contains an instance of sparse tensor.
+ 
+OrtStatus * 	GetSparseTensorValuesTypeAndShape (const OrtValue *ort_value, OrtTensorTypeAndShapeInfo **out)
+ 	Returns data type and shape of sparse tensor values (nnz) iff OrtValue contains a SparseTensor.
+ 
+OrtStatus * 	GetSparseTensorValues (const OrtValue *ort_value, const void **out)
+ 	Returns numeric data for sparse tensor values (nnz). For string values use GetStringTensor*().
+ 
+OrtStatus * 	GetSparseTensorIndicesTypeShape (const OrtValue *ort_value, enum OrtSparseIndicesFormat indices_format, OrtTensorTypeAndShapeInfo **out)
+ 	Returns data type, shape for the type of indices specified by indices_format.
+ 
+OrtStatus * 	GetSparseTensorIndices (const OrtValue *ort_value, enum OrtSparseIndicesFormat indices_format, size_t *num_indices, const void **indices)
+ 	Returns indices data for the type of the indices specified by indices_format.
+ 
+OrtTypeInfo
+OrtStatus * 	CastTypeInfoToTensorInfo (const OrtTypeInfo *type_info, const OrtTensorTypeAndShapeInfo **out)
+ 	Get OrtTensorTypeAndShapeInfo from an OrtTypeInfo.
+ 
+OrtStatus * 	GetOnnxTypeFromTypeInfo (const OrtTypeInfo *type_info, enum ONNXType *out)
+ 	Get ONNXType from OrtTypeInfo.
+ 
+void 	ReleaseTypeInfo (OrtTypeInfo *input)
+ 
+OrtStatus * 	GetDenotationFromTypeInfo (const OrtTypeInfo *type_info, const char **const denotation, size_t *len)
+ 	Get denotation from type information.
+ 
+OrtStatus * 	CastTypeInfoToMapTypeInfo (const OrtTypeInfo *type_info, const OrtMapTypeInfo **out)
+ 	Get detailed map information from an OrtTypeInfo.
+ 
+OrtStatus * 	CastTypeInfoToSequenceTypeInfo (const OrtTypeInfo *type_info, const OrtSequenceTypeInfo **out)
+ 	Cast OrtTypeInfo to an OrtSequenceTypeInfo.
+ 
+OrtTensorTypeAndShapeInfo
+OrtStatus * 	CreateTensorTypeAndShapeInfo (OrtTensorTypeAndShapeInfo **out)
+ 	Create an OrtTensorTypeAndShapeInfo object.
+ 
+OrtStatus * 	SetTensorElementType (OrtTensorTypeAndShapeInfo *info, enum ONNXTensorElementDataType type)
+ 	Set element type in OrtTensorTypeAndShapeInfo.
+ 
+OrtStatus * 	SetDimensions (OrtTensorTypeAndShapeInfo *info, const int64_t *dim_values, size_t dim_count)
+ 	Set shape information in OrtTensorTypeAndShapeInfo.
+ 
+OrtStatus * 	GetTensorElementType (const OrtTensorTypeAndShapeInfo *info, enum ONNXTensorElementDataType *out)
+ 	Get element type in OrtTensorTypeAndShapeInfo.
+ 
+OrtStatus * 	GetDimensionsCount (const OrtTensorTypeAndShapeInfo *info, size_t *out)
+ 	Get dimension count in OrtTensorTypeAndShapeInfo.
+ 
+OrtStatus * 	GetDimensions (const OrtTensorTypeAndShapeInfo *info, int64_t *dim_values, size_t dim_values_length)
+ 	Get dimensions in OrtTensorTypeAndShapeInfo.
+ 
+OrtStatus * 	GetSymbolicDimensions (const OrtTensorTypeAndShapeInfo *info, const char *dim_params[], size_t dim_params_length)
+ 	Get symbolic dimension names in OrtTensorTypeAndShapeInfo.
+ 
+OrtStatus * 	GetTensorShapeElementCount (const OrtTensorTypeAndShapeInfo *info, size_t *out)
+ 	Get total number of elements in a tensor shape from an OrtTensorTypeAndShapeInfo.
+ 
+void 	ReleaseTensorTypeAndShapeInfo (OrtTensorTypeAndShapeInfo *input)
+ 
+OrtMemoryInfo
+OrtStatus * 	CreateMemoryInfo (const char *name, enum OrtAllocatorType type, int id, enum OrtMemType mem_type, OrtMemoryInfo **out)
+ 	Create an OrtMemoryInfo.
+ 
+OrtStatus * 	CreateCpuMemoryInfo (enum OrtAllocatorType type, enum OrtMemType mem_type, OrtMemoryInfo **out)
+ 	Create an OrtMemoryInfo for CPU memory.
+ 
+OrtStatus * 	CompareMemoryInfo (const OrtMemoryInfo *info1, const OrtMemoryInfo *info2, int *out)
+ 	Compare OrtMemoryInfo objects for equality.
+ 
+OrtStatus * 	MemoryInfoGetName (const OrtMemoryInfo *ptr, const char **out)
+ 	Get name from OrtMemoryInfo.
+ 
+OrtStatus * 	MemoryInfoGetId (const OrtMemoryInfo *ptr, int *out)
+ 	Get the id from OrtMemoryInfo.
+ 
+OrtStatus * 	MemoryInfoGetMemType (const OrtMemoryInfo *ptr, OrtMemType *out)
+ 	Get the OrtMemType from OrtMemoryInfo.
+ 
+OrtStatus * 	MemoryInfoGetType (const OrtMemoryInfo *ptr, OrtAllocatorType *out)
+ 	Get the OrtAllocatorType from OrtMemoryInfo.
+ 
+void 	ReleaseMemoryInfo (OrtMemoryInfo *input)
+ 
+OrtAllocator
+OrtStatus * 	AllocatorAlloc (OrtAllocator *ort_allocator, size_t size, void **out)
+ 	Calls OrtAllocator::Alloc function.
+ 
+OrtStatus * 	AllocatorFree (OrtAllocator *ort_allocator, void *p)
+ 	Calls OrtAllocator::Free function.
+ 
+OrtStatus * 	AllocatorGetInfo (const OrtAllocator *ort_allocator, const struct OrtMemoryInfo **out)
+ 	Calls OrtAllocator::Info function.
+ 
+OrtStatus * 	GetAllocatorWithDefaultOptions (OrtAllocator **out)
+ 	Get the default allocator.
+ 
+OrtStatus * 	CreateAllocator (const OrtSession *session, const OrtMemoryInfo *mem_info, OrtAllocator **out)
+ 	Create an allocator for an OrtSession following an OrtMemoryInfo.
+ 
+void 	ReleaseAllocator (OrtAllocator *input)
+ 	Release an OrtAllocator obtained from OrtApi::CreateAllocator.
+ 
+OrtStatus * 	RegisterAllocator (OrtEnv *env, OrtAllocator *allocator)
+ 	Register a custom allocator.
+ 
+OrtStatus * 	UnregisterAllocator (OrtEnv *env, const OrtMemoryInfo *mem_info)
+ 	Unregister a custom allocator.
+ 
+OrtKernelInfo
+Custom operator APIs.
+
+OrtStatus * 	KernelInfoGetAttribute_float (const OrtKernelInfo *info, const char *name, float *out)
+ 	Get a float stored as an attribute in the graph node.
+ 
+OrtStatus * 	KernelInfoGetAttribute_int64 (const OrtKernelInfo *info, const char *name, int64_t *out)
+ 	Fetch a 64-bit int stored as an attribute in the graph node.
+ 
+OrtStatus * 	KernelInfoGetAttribute_string (const OrtKernelInfo *info, const char *name, char *out, size_t *size)
+ 	Fetch a string stored as an attribute in the graph node.
+ 
+OrtStatus * 	KernelInfoGetAttributeArray_float (const OrtKernelInfo *info, const char *name, float *out, size_t *size)
+ 	Fetch an array of int64_t values stored as an attribute in the graph node.
+ 
+OrtStatus * 	KernelInfoGetAttributeArray_int64 (const OrtKernelInfo *info, const char *name, int64_t *out, size_t *size)
+ 	Fetch an array of int64_t values stored as an attribute in the graph node.
+ 
+OrtStatus * 	KernelInfo_GetInputCount (const OrtKernelInfo *info, size_t *out)
+ 	Get the number of inputs from OrtKernelInfo.
+ 
+OrtStatus * 	KernelInfo_GetOutputCount (const OrtKernelInfo *info, size_t *out)
+ 	Get the number of outputs from OrtKernelInfo.
+ 
+OrtStatus * 	KernelInfo_GetInputName (const OrtKernelInfo *info, size_t index, char *out, size_t *size)
+ 	Get the name of a OrtKernelInfo's input.
+ 
+OrtStatus * 	KernelInfo_GetOutputName (const OrtKernelInfo *info, size_t index, char *out, size_t *size)
+ 	Get the name of a OrtKernelInfo's output.
+ 
+OrtStatus * 	KernelInfo_GetInputTypeInfo (const OrtKernelInfo *info, size_t index, OrtTypeInfo **type_info)
+ 	Get the type information for a OrtKernelInfo's input.
+ 
+OrtStatus * 	KernelInfo_GetOutputTypeInfo (const OrtKernelInfo *info, size_t index, OrtTypeInfo **type_info)
+ 	Get the type information for a OrtKernelInfo's output.
+ 
+OrtStatus * 	KernelInfoGetAttribute_tensor (const OrtKernelInfo *info, const char *name, OrtAllocator *allocator, OrtValue **out)
+ 	Get a OrtValue tensor stored as an attribute in the graph node.
+ 
+OrtStatus * 	KernelInfo_GetNodeName (const OrtKernelInfo *info, char *out, size_t *size)
+ 	Get the graph node name from OrtKernelInfo.
+ 
+OrtStatus * 	KernelInfo_GetLogger (const OrtKernelInfo *info, const OrtLogger **logger)
+ 	Get the session logger from OrtKernelInfo.
+ 
+OrtKernelContext
+Custom operator APIs.
+
+OrtStatus * 	KernelContext_GetInputCount (const OrtKernelContext *context, size_t *out)
+ 	Used for custom operators, get the input count of a kernel.
+ 
+OrtStatus * 	KernelContext_GetOutputCount (const OrtKernelContext *context, size_t *out)
+ 	Used for custom operators, get the output count of a kernel.
+ 
+OrtStatus * 	KernelContext_GetInput (const OrtKernelContext *context, size_t index, const OrtValue **out)
+ 	Used for custom operators, get an input of a kernel.
+ 
+OrtStatus * 	KernelContext_GetOutput (OrtKernelContext *context, size_t index, const int64_t *dim_values, size_t dim_count, OrtValue **out)
+ 	Used for custom operators, get an output of a kernel.
+ 
+OrtStatus * 	KernelContext_GetGPUComputeStream (const OrtKernelContext *context, void **out)
+ 	Used for custom operators, gets the GPU compute stream to use to launch the custom a GPU kernel.
+ 
+OrtStatus * 	KernelContext_GetLogger (const OrtKernelContext *context, const OrtLogger **logger)
+ 	Get the runtime logger from OrtKernelContext.
+ 
+OrtMapTypeInfo
+OrtStatus * 	GetMapKeyType (const OrtMapTypeInfo *map_type_info, enum ONNXTensorElementDataType *out)
+ 	Get key type from an OrtMapTypeInfo.
+ 
+OrtStatus * 	GetMapValueType (const OrtMapTypeInfo *map_type_info, OrtTypeInfo **type_info)
+ 	Get the value type from an OrtMapTypeInfo.
+ 
+void 	ReleaseMapTypeInfo (OrtMapTypeInfo *input)
+ 
+OrtSequenceTypeInfo
+OrtStatus * 	GetSequenceElementType (const OrtSequenceTypeInfo *sequence_type_info, OrtTypeInfo **type_info)
+ 	Get element type from an OrtSequenceTypeInfo.
+ 
+void 	ReleaseSequenceTypeInfo (OrtSequenceTypeInfo *input)
+ 
+OrtModelMetadata
+OrtStatus * 	ModelMetadataGetProducerName (const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char **value)
+ 	Get producer name from an OrtModelMetadata.
+ 
+OrtStatus * 	ModelMetadataGetGraphName (const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char **value)
+ 	Get graph name from an OrtModelMetadata.
+ 
+OrtStatus * 	ModelMetadataGetDomain (const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char **value)
+ 	Get domain from an OrtModelMetadata.
+ 
+OrtStatus * 	ModelMetadataGetDescription (const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char **value)
+ 	Get description from an OrtModelMetadata.
+ 
+OrtStatus * 	ModelMetadataLookupCustomMetadataMap (const OrtModelMetadata *model_metadata, OrtAllocator *allocator, const char *key, char **value)
+ 	Return data for a key in the custom metadata map in an OrtModelMetadata.
+ 
+OrtStatus * 	ModelMetadataGetVersion (const OrtModelMetadata *model_metadata, int64_t *value)
+ 	Get version number from an OrtModelMetadata.
+ 
+void 	ReleaseModelMetadata (OrtModelMetadata *input)
+ 
+OrtStatus * 	ModelMetadataGetCustomMetadataMapKeys (const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char ***keys, int64_t *num_keys)
+ 
+OrtStatus * 	ModelMetadataGetGraphDescription (const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char **value)
+ 
+OrtThreadingOptions
+OrtStatus * 	CreateThreadingOptions (OrtThreadingOptions **out)
+ 	Create an OrtThreadingOptions.
+ 
+void 	ReleaseThreadingOptions (OrtThreadingOptions *input)
+ 
+OrtStatus * 	SetGlobalIntraOpNumThreads (OrtThreadingOptions *tp_options, int intra_op_num_threads)
+ 	Set global intra-op thread count.
+ 
+OrtStatus * 	SetGlobalInterOpNumThreads (OrtThreadingOptions *tp_options, int inter_op_num_threads)
+ 	Set global inter-op thread count.
+ 
+OrtStatus * 	SetGlobalSpinControl (OrtThreadingOptions *tp_options, int allow_spinning)
+ 	Set global spin control options.
+ 
+OrtStatus * 	SetGlobalDenormalAsZero (OrtThreadingOptions *tp_options)
+ 	Set threading flush-to-zero and denormal-as-zero.
+ 
+OrtStatus * 	SetGlobalCustomCreateThreadFn (OrtThreadingOptions *tp_options, OrtCustomCreateThreadFn ort_custom_create_thread_fn)
+ 	Set custom thread creation function for global thread pools.
+ 
+OrtStatus * 	SetGlobalCustomThreadCreationOptions (OrtThreadingOptions *tp_options, void *ort_custom_thread_creation_options)
+ 	Set custom thread creation options for global thread pools.
+ 
+OrtStatus * 	SetGlobalCustomJoinThreadFn (OrtThreadingOptions *tp_options, OrtCustomJoinThreadFn ort_custom_join_thread_fn)
+ 	Set custom thread join function for global thread pools.
+ 
+Misc
+OrtStatus * 	GetAvailableProviders (char ***out_ptr, int *provider_length)
+ 	Get the names of all available providers.
+ 
+OrtStatus * 	ReleaseAvailableProviders (char **ptr, int providers_length)
+ 	Release data from OrtApi::GetAvailableProviders. This API will never fail so you can rely on it in a noexcept code.
+ 
+OrtStatus * 	SetCurrentGpuDeviceId (int device_id)
+ 	Set current GPU device ID.
+ 
+OrtStatus * 	GetCurrentGpuDeviceId (int *device_id)
+ 	Get current GPU device ID.
+ 
+OrtArenaCfg
+OrtStatus * 	CreateArenaCfg (size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes, int max_dead_bytes_per_chunk, OrtArenaCfg **out)
+ 
+void 	ReleaseArenaCfg (OrtArenaCfg *input)
+ 
+OrtStatus * 	CreateArenaCfgV2 (const char *const *arena_config_keys, const size_t *arena_config_values, size_t num_keys, OrtArenaCfg **out)
+ 	Create an OrtArenaCfg.
+ 
+OrtPrepackedWeightsContainer
+OrtStatus * 	CreatePrepackedWeightsContainer (OrtPrepackedWeightsContainer **out)
+ 	Create an OrtPrepackedWeightsContainer.
+ 
+void 	ReleasePrepackedWeightsContainer (OrtPrepackedWeightsContainer *input)
+ 	Release OrtPrepackedWeightsContainer instance.
+ 
+GetTensorMemoryInfo
+OrtStatus * 	GetTensorMemoryInfo (const OrtValue *value, const OrtMemoryInfo **mem_info)
+ 	Returns a pointer to the OrtMemoryInfo of a Tensor.
+ 
+GetExecutionProviderApi
+OrtStatus * 	GetExecutionProviderApi (const char *provider_name, uint32_t version, const void **provider_api)
+ 	Get a pointer to the requested version of the Execution Provider specific API extensions to the OrtApi.
+ 
+SessionOptions
+OrtStatus * 	SessionOptionsSetCustomCreateThreadFn (OrtSessionOptions *options, OrtCustomCreateThreadFn ort_custom_create_thread_fn)
+ 	Set custom thread creation function.
+ 
+OrtStatus * 	SessionOptionsSetCustomThreadCreationOptions (OrtSessionOptions *options, void *ort_custom_thread_creation_options)
+ 	Set creation options for custom thread.
+ 
+OrtStatus * 	SessionOptionsSetCustomJoinThreadFn (OrtSessionOptions *options, OrtCustomJoinThreadFn ort_custom_join_thread_fn)
+ 	Set custom thread join function.
+ 
+OrtLogger
+Custom operator APIs.
+
+OrtStatus * 	Logger_LogMessage (const OrtLogger *logger, OrtLoggingLevel log_severity_level, const char *message, const char *file_path, int line_number, const char *func_name)
+ 	Logs a message at the given severity level using the provided OrtLogger.
+ 
+OrtStatus * 	Logger_GetLoggingSeverityLevel (const OrtLogger *logger, OrtLoggingLevel *out)
+ 	Get the logging severity level of the OrtLogger.
+ 
+OrtEpDynamicOptions
+OrtStatus * 	SetEpDynamicOptions (OrtSession *sess, const char *const *keys, const char *const *values, size_t kv_len)
+ 	Set DynamicOptions for EPs (Execution Providers)
+ 
diff --git a/src/Conversion/KrnlToLLVM/KrnlCall.cpp b/src/Conversion/KrnlToLLVM/KrnlCall.cpp
index 3251fd1..2b40775 100644
--- a/src/Conversion/KrnlToLLVM/KrnlCall.cpp
+++ b/src/Conversion/KrnlToLLVM/KrnlCall.cpp
@@ -1,19 +1,11 @@
-/*
- * SPDX-License-Identifier: Apache-2.0
- */
-
-//===-------------- KrnlCall.cpp - Lower KrnlCallOp -----------------------===//
-//
-// Copyright 2022-2024 The IBM Research Authors.
-//
-// =============================================================================
-//
-// This file lowers the KrnlCallOp operator.
-//
-//===----------------------------------------------------------------------===//
+/**********************************************
+ * IMPORT LIBRARIES
+ **********************************************/
 
+/*
+Libraries and tools used in this script, along with version info when applicable.
+*/
 #include "llvm/ADT/TypeSwitch.h"
-
 #include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
 #include "src/Dialect/Krnl/DialectBuilder.hpp"
 
@@ -24,214 +16,441 @@ using namespace mlir;
 namespace onnx_mlir {
 namespace krnl {
 
-class KrnlCallOpLowering : public ConversionPattern {
-public:
-  explicit KrnlCallOpLowering(
-      LLVMTypeConverter &typeConverter, MLIRContext *context)
-      : ConversionPattern(
-            typeConverter, KrnlCallOp::getOperationName(), 1, context) {}
+/**********************************************
+ * FUNCTION DEFINITIONS
+ **********************************************/
+
+/*
+ * Purpose: Helper to check if the current KrnlCallOp is calling the custom
+ *          ort_cpu_ep_fused_gemm function.
+ * Parameters:
+ *    - op (Operation*): The KrnlCallOp being lowered.
+ * Returns:
+ *    - bool: True if this is a call to ort_cpu_ep_fused_gemm, false otherwise.
+ */
+bool isFusedGemmCall(Operation *op) {
+    if (auto callOp = llvm::dyn_cast<KrnlCallOp>(op)) {
+        auto funcNameAttr = callOp.getFuncNameAttr();
+        if (funcNameAttr && funcNameAttr.getValue() == "ort_cpu_ep_fused_gemm")
+            return true;
+    }
+    return false;
+}
+
+/**********************************************
+ * FUSEDGEMM SPECIALIZED LOWERING
+ **********************************************/
+
+/*
+ * Purpose: Lower KrnlCallOp for ort_cpu_ep_fused_gemm.
+ *          Pass OMTensor* wrappers for all tensors, and scalars as int64_t.
+ * Parameters:
+ *    - op (Operation*): The KrnlCallOp being lowered.
+ *    - operands (ArrayRef<Value>): The operands to the call.
+ *    - rewriter (ConversionPatternRewriter&): The MLIR rewriter.
+ *    - llvmTypeConverter (LLVMTypeConverter*): The LLVM type converter.
+ * Returns:
+ *    - LogicalResult: Success or failure.
+ */
+LogicalResult lowerFusedGemmKrnlCallOp(Operation *op, ArrayRef<Value> operands,
+    ConversionPatternRewriter &rewriter, LLVMTypeConverter *llvmTypeConverter) {
 
-  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
-      ConversionPatternRewriter &rewriter) const override {
-    KrnlCallOpAdaptor krnlCallAdaptor(operands);
+    /******************************************
+     * INITIALIZE DATA
+     ******************************************/
     Location loc = op->getLoc();
     KrnlCallOp krnlCallOp = llvm::cast<KrnlCallOp>(op);
+    ModuleOp module = op->getParentOfType<ModuleOp>();
     MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
-    const LLVMTypeConverter *llvmTypeConverter =
-        static_cast<const LLVMTypeConverter *>(getTypeConverter());
+    const auto &apiRegistry =
+        RuntimeAPIRegistry(module, rewriter, *llvmTypeConverter);
 
-    // Get a symbol reference to the function, inserting it if necessary.
-    ModuleOp module = op->getParentOfType<ModuleOp>();
     llvm::SmallVector<Type, 4> parameterTypeList;
     llvm::SmallVector<Value, 4> parameterList;
     llvm::SmallVector<Value, 4> omTensors;
 
-    // Some type of operands has been converted.
-    // It is better to check the type of original operands.
-    // Thus, the two kinds of operands are used together.
-    auto itConverted = krnlCallAdaptor.getParameters().begin();
-    auto itOriginal = krnlCallOp.getParameters().begin();
-    for (; itConverted != krnlCallAdaptor.getParameters().end();
-         itConverted++, itOriginal++) {
-      handleOneParameter(rewriter, op, *itConverted, *itOriginal,
-          parameterTypeList, parameterList, omTensors);
+    llvm::outs() << "[FusedGemm] Lowering KrnlCallOp for ort_cpu_ep_fused_gemm (OMTensor wrapper mode)\n";
+
+    auto origParams = krnlCallOp.getParameters();
+    auto numParams = origParams.size();
+
+    // Defensive: Expecting 9 parameters (A, B, Bias, Y, M, N, K, transA, transB)
+    if (numParams != 9 || operands.size() != 9) {
+        llvm::outs() << "[FusedGemm] ERROR: Expected 9 parameters, got " << numParams << " and " << operands.size() << "\n";
+        return failure();
+    }
+
+    /******************************************
+     * RETRIEVE AND WRAP TENSOR ARGUMENTS AS OMTensor*
+     ******************************************/
+    llvm::SmallVector<std::string, 4> tensorNames = {"A", "B", "Bias", "Y"};
+    for (int i = 0; i < 4; ++i) {
+        Value origVal = origParams[i];
+        Value convVal = operands[i];
+        Type ty = origVal.getType();
+        std::string name = tensorNames[i];
+
+        if (auto memRefTy = mlir::dyn_cast<MemRefType>(ty)) {
+            // Wrap MemRef as OMTensor*
+            auto int64Ty = rewriter.getI64Type();
+            auto memRefRank = memRefTy.getRank();
+            auto memRefRankVal = create.llvm.constant(int64Ty, static_cast<int64_t>(memRefRank));
+            Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
+                RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});
+            Type llvmElemTy = llvmTypeConverter->convertType(memRefTy.getElementType());
+            krnl::fillOMTensorWithMemRef(convVal, llvmElemTy, omTensor,
+                false /*outOwning*/, rewriter, loc, apiRegistry, module);
+            auto int8Ty = IntegerType::get(op->getContext(), 8);
+            auto opaquePtrTy = getPointerType(op->getContext(), int8Ty);
+            parameterTypeList.emplace_back(opaquePtrTy);
+            parameterList.emplace_back(omTensor);
+            omTensors.emplace_back(omTensor);
+            llvm::outs() << "[FusedGemm] " << name << " wrapped as OMTensor*\n";
+        } else if (mlir::isa<NoneType>(ty)) {
+            auto int8Ty = IntegerType::get(op->getContext(), 8);
+            auto opaquePtrTy = getPointerType(op->getContext(), int8Ty);
+            Value nullPtr = create.llvm.null(opaquePtrTy);
+            parameterTypeList.emplace_back(opaquePtrTy);
+            parameterList.emplace_back(nullPtr);
+            llvm::outs() << "[FusedGemm] " << name << " is NoneType, passing nullptr\n";
+        } else {
+            llvm::outs() << "[FusedGemm] Unexpected type for tensor arg '" << name << "'\n";
+            return failure();
+        }
     }
 
-    // Handle the Attributes
-    for (auto namedAttr : op->getAttrs()) {
-      // Avoid the funcName() Attribute
-      if (namedAttr.getName().getValue() == "funcName")
-        continue;
-      if (namedAttr.getName().getValue() == "numOfOutput")
-        continue;
-      handleOneAttribute(
-          rewriter, op, namedAttr.getValue(), parameterTypeList, parameterList);
+    /******************************************
+     * RETRIEVE AND CONVERT SCALAR ARGUMENTS
+     ******************************************/
+    for (int i = 4; i < 9; ++i) {
+        Value origVal = origParams[i];
+        Value convVal = operands[i];
+        Type ty = origVal.getType();
+
+        if (mlir::isa<IndexType>(ty)) {
+            auto int64Ty = rewriter.getI64Type();
+            Value casted = rewriter.create<arith::IndexCastOp>(loc, int64Ty, convVal);
+            parameterTypeList.emplace_back(int64Ty);
+            parameterList.emplace_back(casted);
+            llvm::outs() << "[FusedGemm] Scalar (IndexType) cast to int64.\n";
+        } else if (mlir::isa<IntegerType>(ty)) {
+            Type llvmTy = llvmTypeConverter->convertType(ty);
+            parameterTypeList.emplace_back(llvmTy);
+            parameterList.emplace_back(convVal);
+            llvm::outs() << "[FusedGemm] Scalar (IntegerType) passed directly.\n";
+        } else {
+            llvm::outs() << "[FusedGemm] Unexpected type for scalar arg\n";
+            return failure();
+        }
     }
 
+    /******************************************
+     * PERFORM CALL
+     ******************************************/
     ValueRange returns = op->getResults();
     if (returns.size() == 0) {
-      // There is no return
-      FlatSymbolRefAttr callRef =
-          create.llvm.getOrInsertSymbolRef(module, krnlCallOp.getFuncName(),
-              LLVM::LLVMVoidType::get(module.getContext()), parameterTypeList);
-      create.llvm.call({}, callRef, parameterList);
-
-      rewriter.eraseOp(op);
+        FlatSymbolRefAttr callRef =
+            create.llvm.getOrInsertSymbolRef(module, krnlCallOp.getFuncName(),
+                LLVM::LLVMVoidType::get(module.getContext()), parameterTypeList);
+        create.llvm.call({}, callRef, parameterList);
+        rewriter.eraseOp(op);
+        llvm::outs() << "[FusedGemm] Call to ort_cpu_ep_fused_gemm emitted (void)\n";
     } else {
-      assert(returns.size() == 1 &&
-             "Only one return value is allowed for krnl.call now");
-      Type llvmReturnType =
-          llvmTypeConverter->convertType(returns[0].getType());
-
-      FlatSymbolRefAttr callRef = create.llvm.getOrInsertSymbolRef(
-          module, krnlCallOp.getFuncName(), llvmReturnType, parameterTypeList);
-      auto llvmCall =
-          create.llvm.call({llvmReturnType}, callRef, parameterList);
-      rewriter.replaceOp(op, llvmCall.getDefiningOp()->getResults()[0]);
+        assert(returns.size() == 1 &&
+               "Only one return value is allowed for krnl.call now");
+        Type llvmReturnType =
+            llvmTypeConverter->convertType(returns[0].getType());
+
+        FlatSymbolRefAttr callRef = create.llvm.getOrInsertSymbolRef(
+            module, krnlCallOp.getFuncName(), llvmReturnType, parameterTypeList);
+        auto llvmCall =
+            create.llvm.call({llvmReturnType}, callRef, parameterList);
+        rewriter.replaceOp(op, llvmCall.getDefiningOp()->getResults()[0]);
+        llvm::outs() << "[FusedGemm] Call to ort_cpu_ep_fused_gemm emitted (with return)\n";
     }
 
-    // Destroy OMTensor wrappers of parameters.
-    const auto &apiRegistry =
-        RuntimeAPIRegistry(module, rewriter, *llvmTypeConverter);
+    /******************************************
+     * CLEANUP: Destroy OMTensor wrappers
+     ******************************************/
     for (Value omt : omTensors) {
-      RuntimeAPI::callApi(
-          rewriter, loc, apiRegistry, RuntimeAPI::API::DESTROY_OMTENSOR, {omt});
+        RuntimeAPI::callApi(
+            rewriter, loc, apiRegistry, RuntimeAPI::API::DESTROY_OMTENSOR, {omt});
     }
 
     return success();
-  }
+}
+
+/**********************************************
+ * DEFAULT KrnlCallOp LOWERING
+ **********************************************/
+
+class KrnlCallOpLowering : public ConversionPattern {
+public:
+    explicit KrnlCallOpLowering(
+        LLVMTypeConverter &typeConverter, MLIRContext *context)
+        : ConversionPattern(
+            typeConverter, KrnlCallOp::getOperationName(), 1, context) {}
+
+    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
+        ConversionPatternRewriter &rewriter) const override {
+
+        /******************************************
+         * SPECIAL CASE: FusedGemm
+         ******************************************/
+        if (isFusedGemmCall(op)) {
+            llvm::outs() << "[KrnlCall] Detected FusedGemm KrnlCallOp, using specialized lowering\n";
+            return lowerFusedGemmKrnlCallOp(op, operands, rewriter, (LLVMTypeConverter*)getTypeConverter());
+        }
+
+        /******************************************
+         * DEFAULT KrnlCallOp LOWERING (unchanged)
+         ******************************************/
+        KrnlCallOpAdaptor krnlCallAdaptor(operands);
+        Location loc = op->getLoc();
+        KrnlCallOp krnlCallOp = llvm::cast<KrnlCallOp>(op);
+        MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
+        const LLVMTypeConverter *llvmTypeConverter =
+            static_cast<const LLVMTypeConverter *>(getTypeConverter());
+
+        ModuleOp module = op->getParentOfType<ModuleOp>();
+        llvm::SmallVector<Type, 4> parameterTypeList;
+        llvm::SmallVector<Value, 4> parameterList;
+        llvm::SmallVector<Value, 4> omTensors;
+
+        bool isFusedGemm = isFusedGemmCall(op);
+
+        auto itConverted = krnlCallAdaptor.getParameters().begin();
+        auto itOriginal = krnlCallOp.getParameters().begin();
+        for (; itConverted != krnlCallAdaptor.getParameters().end();
+             itConverted++, itOriginal++) {
+            handleOneParameter(rewriter, op, *itConverted, *itOriginal,
+                parameterTypeList, parameterList, omTensors, isFusedGemm);
+        }
+
+        // Handle the Attributes
+        for (auto namedAttr : op->getAttrs()) {
+            if (namedAttr.getName().getValue() == "funcName")
+                continue;
+            if (namedAttr.getName().getValue() == "numOfOutput")
+                continue;
+            handleOneAttribute(
+                rewriter, op, namedAttr.getValue(), parameterTypeList, parameterList);
+        }
+
+        ValueRange returns = op->getResults();
+        if (returns.size() == 0) {
+            FlatSymbolRefAttr callRef =
+                create.llvm.getOrInsertSymbolRef(module, krnlCallOp.getFuncName(),
+                    LLVM::LLVMVoidType::get(module.getContext()), parameterTypeList);
+            create.llvm.call({}, callRef, parameterList);
+            rewriter.eraseOp(op);
+        } else {
+            assert(returns.size() == 1 &&
+                   "Only one return value is allowed for krnl.call now");
+            Type llvmReturnType =
+                llvmTypeConverter->convertType(returns[0].getType());
+
+            FlatSymbolRefAttr callRef = create.llvm.getOrInsertSymbolRef(
+                module, krnlCallOp.getFuncName(), llvmReturnType, parameterTypeList);
+            auto llvmCall =
+                create.llvm.call({llvmReturnType}, callRef, parameterList);
+            rewriter.replaceOp(op, llvmCall.getDefiningOp()->getResults()[0]);
+        }
+
+        // Destroy OMTensor wrappers of parameters (not used for fused gemm).
+        if (!isFusedGemm) {
+            const auto &apiRegistry =
+                RuntimeAPIRegistry(module, rewriter, *llvmTypeConverter);
+            for (Value omt : omTensors) {
+                RuntimeAPI::callApi(
+                    rewriter, loc, apiRegistry, RuntimeAPI::API::DESTROY_OMTENSOR, {omt});
+            }
+        }
+
+        return success();
+    }
 
 private:
-  void handleOneParameter(PatternRewriter &rewriter, Operation *op,
-      Value parameter, Value original,
-      llvm::SmallVector<Type, 4> &parameterTypeList,
-      llvm::SmallVector<Value, 4> &parameterList,
-      llvm::SmallVector<Value, 4> &omTensors) const {
-    MLIRContext *context = op->getContext();
-    Location loc = op->getLoc();
-    ModuleOp module = op->getParentOfType<ModuleOp>();
-    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
-    const auto *llvmTypeConverter =
-        static_cast<const LLVMTypeConverter *>(getTypeConverter());
-    const auto &apiRegistry =
-        RuntimeAPIRegistry(module, rewriter, *llvmTypeConverter);
+    /*
+     * Purpose: Handle one parameter for KrnlCallOp lowering.
+     *          For ort_cpu_ep_fused_gemm, pass OMTensor wrappers for tensors,
+     *          and pass scalars directly (cast index to i64).
+     */
+    void handleOneParameter(PatternRewriter &rewriter, Operation *op,
+        Value parameter, Value original,
+        llvm::SmallVector<Type, 4> &parameterTypeList,
+        llvm::SmallVector<Value, 4> &parameterList,
+        llvm::SmallVector<Value, 4> &omTensors,
+        bool isFusedGemm) const {
+        MLIRContext *context = op->getContext();
+        Location loc = op->getLoc();
+        ModuleOp module = op->getParentOfType<ModuleOp>();
+        MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
+        const auto *llvmTypeConverter =
+            static_cast<const LLVMTypeConverter *>(getTypeConverter());
+        const auto &apiRegistry =
+            RuntimeAPIRegistry(module, rewriter, *llvmTypeConverter);
 
-    // Check the original type, not after type conversion
-    Type ty = original.getType();
-    if (auto originalMemRef = mlir::dyn_cast<MemRefType>(ty)) {
-      auto int64Ty = IntegerType::get(context, 64);
-      auto memRefTy = mlir::dyn_cast<LLVM::LLVMStructType>(parameter.getType());
-      auto memRefRank = krnl::getRankFromMemRefType(memRefTy);
-      auto memRefRankVal =
-          create.llvm.constant(int64Ty, static_cast<int64_t>(memRefRank));
-      Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
-          RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});
-
-      Type llvmOrigElemTy =
-          llvmTypeConverter->convertType(originalMemRef.getElementType());
-      krnl::fillOMTensorWithMemRef(parameter, llvmOrigElemTy, omTensor,
-          false /*outOwning*/, rewriter, loc, apiRegistry, module);
-      auto int8Ty = IntegerType::get(context, 8);
-      auto opaquePtrTy = getPointerType(context, int8Ty);
-      parameterTypeList.emplace_back(opaquePtrTy);
-      parameterList.emplace_back(omTensor);
-      omTensors.emplace_back(omTensor);
-    } else if (mlir::isa<NoneType>(ty)) {
-      // Generate llvm null pinter for NoneType
-      auto int8Ty = IntegerType::get(context, 8);
-      auto opaquePtrTy = getPointerType(context, int8Ty);
-      parameterTypeList.emplace_back(opaquePtrTy);
-      Value nullPtr = create.llvm.null(opaquePtrTy);
-      parameterList.emplace_back(nullPtr);
-    } else {
-      parameterTypeList.emplace_back(parameter.getType());
-      parameterList.emplace_back(parameter);
+        Type ty = original.getType();
+
+        // Special-case for ort_cpu_ep_fused_gemm
+        if (isFusedGemm) {
+            if (auto memRefTy = mlir::dyn_cast<MemRefType>(ty)) {
+                // TENSOR: Wrap as OMTensor*
+                auto int64Ty = IntegerType::get(context, 64);
+                auto memRefRank = memRefTy.getRank();
+                auto memRefRankVal = create.llvm.constant(int64Ty, static_cast<int64_t>(memRefRank));
+                Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
+                    RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});
+                Type llvmElemTy = llvmTypeConverter->convertType(memRefTy.getElementType());
+                krnl::fillOMTensorWithMemRef(parameter, llvmElemTy, omTensor,
+                    false /*outOwning*/, rewriter, loc, apiRegistry, module);
+                auto int8Ty = IntegerType::get(context, 8);
+                auto opaquePtrTy = getPointerType(context, int8Ty);
+                parameterTypeList.emplace_back(opaquePtrTy);
+                parameterList.emplace_back(omTensor);
+                omTensors.emplace_back(omTensor);
+                return;
+            } else if (ty.isa<IndexType>()) {
+                // SCALAR: index type, cast to i64
+                auto int64Ty = rewriter.getI64Type();
+                Value casted = rewriter.create<arith::IndexCastOp>(loc, int64Ty, parameter);
+                parameterTypeList.emplace_back(int64Ty);
+                parameterList.emplace_back(casted);
+                return;
+            } else if (ty.isa<IntegerType>() || ty.isa<FloatType>()) {
+                // SCALAR: Pass directly (int64_t, float, etc.)
+                Type llvmTy = llvmTypeConverter->convertType(ty);
+                parameterTypeList.emplace_back(llvmTy);
+                parameterList.emplace_back(parameter);
+                return;
+            } else if (mlir::isa<NoneType>(ty)) {
+                // Pass null pointer for NoneType
+                auto int8Ty = IntegerType::get(context, 8);
+                auto opaquePtrTy = getPointerType(context, int8Ty);
+                parameterTypeList.emplace_back(opaquePtrTy);
+                Value nullPtr = create.llvm.null(opaquePtrTy);
+                parameterList.emplace_back(nullPtr);
+                return;
+            }
+            // Add more cases if needed
+        }
+
+        // Default lowering for other calls (OMTensor wrapping, etc.)
+        if (auto originalMemRef = mlir::dyn_cast<MemRefType>(ty)) {
+            auto int64Ty = IntegerType::get(context, 64);
+            auto memRefTy = mlir::dyn_cast<LLVM::LLVMStructType>(parameter.getType());
+            auto memRefRank = krnl::getRankFromMemRefType(memRefTy);
+            auto memRefRankVal =
+                create.llvm.constant(int64Ty, static_cast<int64_t>(memRefRank));
+            Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
+                RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});
+
+            Type llvmOrigElemTy =
+                llvmTypeConverter->convertType(originalMemRef.getElementType());
+            krnl::fillOMTensorWithMemRef(parameter, llvmOrigElemTy, omTensor,
+                false /*outOwning*/, rewriter, loc, apiRegistry, module);
+            auto int8Ty = IntegerType::get(context, 8);
+            auto opaquePtrTy = getPointerType(context, int8Ty);
+            parameterTypeList.emplace_back(opaquePtrTy);
+            parameterList.emplace_back(omTensor);
+            omTensors.emplace_back(omTensor);
+        } else if (mlir::isa<NoneType>(ty)) {
+            auto int8Ty = IntegerType::get(context, 8);
+            auto opaquePtrTy = getPointerType(context, int8Ty);
+            parameterTypeList.emplace_back(opaquePtrTy);
+            Value nullPtr = create.llvm.null(opaquePtrTy);
+            parameterList.emplace_back(nullPtr);
+        } else {
+            parameterTypeList.emplace_back(parameter.getType());
+            parameterList.emplace_back(parameter);
+        }
     }
-  }
 
-  void handleOneAttribute(PatternRewriter &rewriter, Operation *op,
-      Attribute attribute, llvm::SmallVector<Type, 4> &parameterTypeList,
-      llvm::SmallVector<Value, 4> &parameterList) const {
-    auto *context = op->getContext();
-    Location loc = op->getLoc();
-    ModuleOp module = op->getParentOfType<ModuleOp>();
-    MultiDialectBuilder<KrnlBuilder, LLVMBuilder> create(rewriter, loc);
-    const LLVMTypeConverter *llvmTypeConverter =
-        static_cast<const LLVMTypeConverter *>(getTypeConverter());
-    const auto &apiRegistry =
-        RuntimeAPIRegistry(module, rewriter, *llvmTypeConverter);
+    /*
+     * Purpose: Handle one attribute for KrnlCallOp lowering.
+     */
+    void handleOneAttribute(PatternRewriter &rewriter, Operation *op,
+        Attribute attribute, llvm::SmallVector<Type, 4> &parameterTypeList,
+        llvm::SmallVector<Value, 4> &parameterList) const {
+        auto *context = op->getContext();
+        Location loc = op->getLoc();
+        ModuleOp module = op->getParentOfType<ModuleOp>();
+        MultiDialectBuilder<KrnlBuilder, LLVMBuilder> create(rewriter, loc);
+        const LLVMTypeConverter *llvmTypeConverter =
+            static_cast<const LLVMTypeConverter *>(getTypeConverter());
+        const auto &apiRegistry =
+            RuntimeAPIRegistry(module, rewriter, *llvmTypeConverter);
 
-    TypeSwitch<Attribute>(attribute)
-        .Case<StringAttr>([&](StringAttr strAttr) {
-          StringRef attrValue = strAttr.getValue();
-          LLVM::GlobalOp globalStr = krnl::getOrCreateGlobalString(
-              attrValue, loc, rewriter, module, llvmTypeConverter);
-          Value strPtr = krnl::getPtrToGlobalString(globalStr, loc, rewriter);
-          auto int8Ty = IntegerType::get(context, 8);
-          auto opaquePtrTy = getPointerType(context, int8Ty);
-          parameterTypeList.emplace_back(opaquePtrTy);
-          parameterList.emplace_back(strPtr);
-        })
-        .Case<IntegerAttr>([&](IntegerAttr integerAttr) {
-          auto int64Ty = IntegerType::get(context, 64);
-          Value cst =
-              rewriter.create<LLVM::ConstantOp>(loc, int64Ty, integerAttr);
-          parameterTypeList.emplace_back(int64Ty);
-          parameterList.emplace_back(cst);
-        })
-        .Case<FloatAttr>([&](FloatAttr floatAttr) {
-          auto f64Ty = rewriter.getF64Type();
-          Value cst = rewriter.create<LLVM::ConstantOp>(loc, f64Ty,
-              rewriter.getFloatAttr(f64Ty, floatAttr.getValueAsDouble()));
-          parameterTypeList.emplace_back(f64Ty);
-          parameterList.emplace_back(cst);
-        })
-        .Case<DenseElementsAttr>([&](DenseElementsAttr denseAttr) {
-          // Use krnl.global to handle it
-          // Since the attribute is still in tensor type, the code has to cross
-          // onnx to krnl, and krnl to llvm.
-          // In future, the attributes should be converted in krnl.call builder.
-          // This code passed onnx-mlir-opt --convert-krnl-to-llvm test case,
-          // but failed in onnx-milr for the tensor type for the attribute
-          auto tensorTy = mlir::cast<TensorType>(denseAttr.getType());
-          auto memRefTy =
-              MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
-          Value constantGlobal =
-              create.krnl.constant(memRefTy, "constant_", denseAttr);
-          Value convertedConstantGlobal =
-              rewriter
-                  .create<UnrealizedConversionCastOp>(loc,
-                      llvmTypeConverter->convertType(memRefTy), constantGlobal)
-                  .getResult(0);
-
-          auto int64Ty = IntegerType::get(context, 64);
-          auto memRefRank = memRefTy.getRank();
-          auto memRefRankVal =
-              create.llvm.constant(int64Ty, static_cast<int64_t>(memRefRank));
-          Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
-              RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});
-
-          Type llvmElemTy =
-              llvmTypeConverter->convertType(memRefTy.getElementType());
-          krnl::fillOMTensorWithMemRef(convertedConstantGlobal, llvmElemTy,
-              omTensor, false /*outOwning*/, rewriter, loc, apiRegistry,
-              module);
-          auto int8Ty = IntegerType::get(context, 8);
-          auto opaquePtrTy = getPointerType(context, int8Ty);
-          parameterTypeList.emplace_back(opaquePtrTy);
-          parameterList.emplace_back(omTensor);
-        })
-        .Default([&](Attribute attr) {
-          llvm_unreachable("This type of Attribute used by krnl.call is not "
-                           "yet implemented");
-        });
-  }
+        TypeSwitch<Attribute>(attribute)
+            .Case<StringAttr>([&](StringAttr strAttr) {
+                StringRef attrValue = strAttr.getValue();
+                LLVM::GlobalOp globalStr = krnl::getOrCreateGlobalString(
+                    attrValue, loc, rewriter, module, llvmTypeConverter);
+                Value strPtr = krnl::getPtrToGlobalString(globalStr, loc, rewriter);
+                auto int8Ty = IntegerType::get(context, 8);
+                auto opaquePtrTy = getPointerType(context, int8Ty);
+                parameterTypeList.emplace_back(opaquePtrTy);
+                parameterList.emplace_back(strPtr);
+            })
+            .Case<IntegerAttr>([&](IntegerAttr integerAttr) {
+                auto int64Ty = IntegerType::get(context, 64);
+                Value cst =
+                    rewriter.create<LLVM::ConstantOp>(loc, int64Ty, integerAttr);
+                parameterTypeList.emplace_back(int64Ty);
+                parameterList.emplace_back(cst);
+            })
+            .Case<FloatAttr>([&](FloatAttr floatAttr) {
+                auto f64Ty = rewriter.getF64Type();
+                Value cst = rewriter.create<LLVM::ConstantOp>(loc, f64Ty,
+                    rewriter.getFloatAttr(f64Ty, floatAttr.getValueAsDouble()));
+                parameterTypeList.emplace_back(f64Ty);
+                parameterList.emplace_back(cst);
+            })
+            .Case<DenseElementsAttr>([&](DenseElementsAttr denseAttr) {
+                auto tensorTy = mlir::cast<TensorType>(denseAttr.getType());
+                auto memRefTy =
+                    MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
+                Value constantGlobal =
+                    create.krnl.constant(memRefTy, "constant_", denseAttr);
+                Value convertedConstantGlobal =
+                    rewriter
+                        .create<UnrealizedConversionCastOp>(loc,
+                            llvmTypeConverter->convertType(memRefTy), constantGlobal)
+                        .getResult(0);
+
+                auto int64Ty = IntegerType::get(context, 64);
+                auto memRefRank = memRefTy.getRank();
+                auto memRefRankVal =
+                    create.llvm.constant(int64Ty, static_cast<int64_t>(memRefRank));
+                Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
+                    RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});
+
+                Type llvmElemTy =
+                    llvmTypeConverter->convertType(memRefTy.getElementType());
+                krnl::fillOMTensorWithMemRef(convertedConstantGlobal, llvmElemTy,
+                    omTensor, false /*outOwning*/, rewriter, loc, apiRegistry,
+                    module);
+                auto int8Ty = IntegerType::get(context, 8);
+                auto opaquePtrTy = getPointerType(context, int8Ty);
+                parameterTypeList.emplace_back(opaquePtrTy);
+                parameterList.emplace_back(omTensor);
+            })
+            .Default([&](Attribute attr) {
+                llvm_unreachable("This type of Attribute used by krnl.call is not "
+                                 "yet implemented");
+            });
+    }
 };
 
+/**********************************************
+ * PATTERN REGISTRATION
+ **********************************************/
+
 void populateLoweringKrnlCallOpPattern(LLVMTypeConverter &typeConverter,
     RewritePatternSet &patterns, MLIRContext *ctx) {
-  patterns.insert<KrnlCallOpLowering>(typeConverter, ctx);
+    patterns.insert<KrnlCallOpLowering>(typeConverter, ctx);
 }
 
 } // namespace krnl
-} // namespace onnx_mlir
+} // namespace onnx_mlir
\ No newline at end of file
diff --git a/src/Conversion/ONNXToKrnl/Additional/Custom.cpp b/src/Conversion/ONNXToKrnl/Additional/Custom.cpp
index 952760a..bc85b1a 100644
--- a/src/Conversion/ONNXToKrnl/Additional/Custom.cpp
+++ b/src/Conversion/ONNXToKrnl/Additional/Custom.cpp
@@ -22,11 +22,16 @@ namespace onnx_mlir {
 
 struct ONNXCustomOpLowering : public OpConversionPattern<ONNXCustomOp> {
   ONNXCustomOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
-      : OpConversionPattern(typeConverter, ctx) {}
+      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1) {}
 
   LogicalResult matchAndRewrite(ONNXCustomOp customOp,
       ONNXCustomOpAdaptor operandAdaptor,
       ConversionPatternRewriter &rewriter) const final {
+    // Skip FusedGemm operations so our specialized pattern can handle them
+    StringAttr funcNameAttr = customOp.getFunctionNameAttr();
+    if (funcNameAttr && funcNameAttr.getValue() == "FusedGemm") {
+      return failure();
+    }
     Operation *op = customOp.getOperation();
     Location loc = op->getLoc();
     ValueRange operands = operandAdaptor.getOperands();
diff --git a/src/Conversion/ONNXToKrnl/Additional/FusedGemm.cpp b/src/Conversion/ONNXToKrnl/Additional/FusedGemm.cpp
new file mode 100644
index 0000000..5072885
--- /dev/null
+++ b/src/Conversion/ONNXToKrnl/Additional/FusedGemm.cpp
@@ -0,0 +1,150 @@
+/**********************************************
+ * IMPORT LIBRARIES
+ **********************************************/
+
+/*
+Libraries and tools used in this script, along with version info when applicable.
+*/
+#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
+#include "src/Dialect/Krnl/KrnlHelper.hpp"
+#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
+#include "src/Conversion/ONNXToKrnl/Additional/FusedGemm.hpp"
+
+using namespace mlir;
+
+namespace onnx_mlir {
+
+/**********************************************
+ * FUNCTION DEFINITIONS
+ **********************************************/
+
+/*
+ * Purpose: Lower the ONNXCustomOp "FusedGemm" to a KrnlCallOp that calls
+ *          the C++ runtime function ort_cpu_ep_fused_gemm, passing
+ *          raw pointers for tensors and int64_t for scalars.
+ */
+struct ONNXFusedGemmOpLowering : public OpConversionPattern<ONNXCustomOp> {
+  ONNXFusedGemmOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
+      : OpConversionPattern(typeConverter, ctx, /*benefit=*/10) {}
+
+  LogicalResult matchAndRewrite(ONNXCustomOp customOp,
+      ONNXCustomOpAdaptor operandAdaptor,
+      ConversionPatternRewriter &rewriter) const final {
+
+    /******************************************
+     * CHECK OP TYPE
+     ******************************************/
+    StringAttr funcNameAttr = customOp.getFunctionNameAttr();
+    if (!funcNameAttr || funcNameAttr.getValue() != "FusedGemm")
+      return failure();
+
+    Operation *op = customOp.getOperation();
+    Location loc = op->getLoc();
+    ValueRange operands = operandAdaptor.getOperands();
+
+    /******************************************
+     * HELPERS AND SHAPE
+     ******************************************/
+    MultiDialectBuilder<AffineBuilder, IndexExprBuilderForKrnl, KrnlBuilder, MemRefBuilder> create(rewriter, loc);
+    IndexExprScope scope(create.krnlIE);
+
+    ONNXCustomOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
+    shapeHelper.computeShapeAndAssertOnFailure();
+
+    /******************************************
+     * OUTPUT ALLOCATION
+     ******************************************/
+    Type outputType = op->getResultTypes()[0];
+    MemRefType outputMemRefType = mlir::cast<MemRefType>(typeConverter->convertType(outputType));
+
+    /******************************************
+     * EXTRACT INPUTS AND ATTRIBUTES
+     ******************************************/
+    MemRefBuilder memrefBuilder(rewriter, loc);
+    IntegerAttr transAAttr = customOp->getAttrOfType<IntegerAttr>("transA");
+    IntegerAttr transBAttr = customOp->getAttrOfType<IntegerAttr>("transB");
+    int64_t transA = transAAttr ? transAAttr.getValue().getSExtValue() : 0;
+    int64_t transB = transBAttr ? transBAttr.getValue().getSExtValue() : 0;
+    Value A = operands[0];
+    Value B = operands[1];
+
+    /******************************************
+     * COMPUTE DIMS (index type)
+     ******************************************/
+    Value Midx, Nidx, Kidx;
+    if (transA == 0) {
+      Midx = memrefBuilder.dim(A, 0);
+      Kidx = memrefBuilder.dim(A, 1);
+    } else {
+      Kidx = memrefBuilder.dim(A, 0);
+      Midx = memrefBuilder.dim(A, 1);
+    }
+    if (transB == 0) {
+      Nidx = memrefBuilder.dim(B, 1);
+    } else {
+      Nidx = memrefBuilder.dim(B, 0);
+    }
+
+    /******************************************
+     * ALLOCATE OUTPUT BUFFER
+     ******************************************/
+    Value outputAlloc;
+    if (outputMemRefType.getNumDynamicDims() == 2) {
+      outputAlloc = create.mem.alignedAlloc(outputMemRefType, {Midx, Nidx});
+    } else if (outputMemRefType.getNumDynamicDims() == 1) {
+      if (outputMemRefType.isDynamicDim(0))
+        outputAlloc = create.mem.alignedAlloc(outputMemRefType, {Midx});
+      else
+        outputAlloc = create.mem.alignedAlloc(outputMemRefType, {Nidx});
+    } else {
+      outputAlloc = create.mem.alignedAlloc(outputMemRefType);
+    }
+
+    /******************************************
+     * PREPARE CALL OPERANDS
+     ******************************************/
+    Value BiasOperand = (operands.size() > 2) ? operands[2] : Value();
+    Value Y = outputAlloc;
+    Value transAVal = rewriter.create<arith::ConstantOp>(
+        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(transA));
+    Value transBVal = rewriter.create<arith::ConstantOp>(
+        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(transB));
+
+        SmallVector<Value, 9> callOperands{A, B, BiasOperand, Y, Midx, Nidx, Kidx, transAVal, transBVal};
+
+    /******************************************
+     * LOWER TO KrnlCallOp
+     ******************************************/
+    std::vector<std::string> attributeNames; // Add attribute names if you want to copy any
+
+    rewriter.create<KrnlCallOp>(
+        loc,
+        "ort_cpu_ep_fused_gemm", // function name as string
+        SmallVector<Value, 0>{}, // outputs
+        op,                       // original op (for attribute copying)
+        callOperands,             // inputs
+        attributeNames            // attributes to copy
+    );
+
+    rewriter.replaceOp(op, Y);
+    return success();
+  }
+};
+
+/**********************************************
+ * PATTERN REGISTRATION
+ **********************************************/
+
+void populateONNXToKrnlConversionAdditionalPass(RewritePatternSet &patterns,
+    TypeConverter &typeConverter, MLIRContext *ctx) {
+  patterns.insert<ONNXFusedGemmOpLowering>(typeConverter, ctx);
+}
+
+} // namespace onnx_mlir
+
+void onnx_mlir::populateLoweringONNXFusedGemmOpPattern(
+    mlir::RewritePatternSet &patterns,
+    mlir::TypeConverter &typeConverter,
+    mlir::MLIRContext *ctx) {
+  populateONNXToKrnlConversionAdditionalPass(patterns, typeConverter, ctx);
+}
\ No newline at end of file
diff --git a/src/Conversion/ONNXToKrnl/Additional/FusedGemm.hpp b/src/Conversion/ONNXToKrnl/Additional/FusedGemm.hpp
new file mode 100644
index 0000000..95ab9d5
--- /dev/null
+++ b/src/Conversion/ONNXToKrnl/Additional/FusedGemm.hpp
@@ -0,0 +1,27 @@
+/*
+ * SPDX-License-Identifier: Apache-2.0
+ */
+
+//===---------- FusedGemm.hpp - Lowering FusedGemm Custom Op --------------===//
+//
+// Copyright 2023 The IBM Research Authors.
+//
+// =============================================================================
+//
+// This file contains declaration of FusedGemm lowering.
+//
+//===----------------------------------------------------------------------===//
+
+#pragma once
+
+#include "mlir/IR/PatternMatch.h"
+#include "mlir/Transforms/DialectConversion.h"
+
+namespace onnx_mlir {
+
+// Populate the pattern list for lowering ONNX FusedGemm 
+// Custom operation to Krnl
+void populateLoweringONNXFusedGemmOpPattern(mlir::RewritePatternSet &patterns,
+    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx);
+
+} // namespace onnx_mlir
diff --git a/src/Conversion/ONNXToKrnl/CMakeLists.txt b/src/Conversion/ONNXToKrnl/CMakeLists.txt
index 6a68f3c..151bfcd 100644
--- a/src/Conversion/ONNXToKrnl/CMakeLists.txt
+++ b/src/Conversion/ONNXToKrnl/CMakeLists.txt
@@ -6,6 +6,7 @@ add_onnx_mlir_library(OMONNXToKrnl
   ONNXToKrnlCommon.cpp
   PerfectHash.cpp
   Additional/Custom.cpp
+  Additional/FusedGemm.cpp
   Additional/LayoutTransform.cpp
   Additional/ShapeTransform.cpp
   ControlFlow/If.cpp
diff --git a/src/Conversion/ONNXToKrnl/ConvertONNXToKrnl.cpp b/src/Conversion/ONNXToKrnl/ConvertONNXToKrnl.cpp
index f901d67..7914a28 100644
--- a/src/Conversion/ONNXToKrnl/ConvertONNXToKrnl.cpp
+++ b/src/Conversion/ONNXToKrnl/ConvertONNXToKrnl.cpp
@@ -23,6 +23,8 @@
 #include "src/Compiler/OptionUtils.hpp"
 #include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
 #include "src/Dialect/Mlir/VectorMachineSupport.hpp"
+/********************MOD FOR FUSEDGEMM**************************/
+#include "src/Conversion/ONNXToKrnl/Additional/FusedGemm.hpp"
 
 using namespace mlir;
 
@@ -283,6 +285,9 @@ void populateONNXToKrnlConversionPattern(RewritePatternSet &patterns,
   patterns.insert<ONNXEntryPointLowering>(ctx);
   // Additional
   populateLoweringONNXCustomOpPattern(patterns, typeConverter, ctx);
+  /********************MOD FOR FUSEDGEMM**************************/
+  // Add our FusedGemm pattern *after* the generic custom pattern
+  populateLoweringONNXFusedGemmOpPattern(patterns, typeConverter, ctx);
   populateLoweringONNXLayoutTransformOpPattern(patterns, typeConverter, ctx, enableParallel);
   populateLoweringONNXShapeTransformOpPattern(patterns, typeConverter, ctx);
   // clang-format on
diff --git a/src/Dialect/ONNX/ONNXOps/Additional/Custom.cpp b/src/Dialect/ONNX/ONNXOps/Additional/Custom.cpp
index 00863e8..6971b11 100644
--- a/src/Dialect/ONNX/ONNXOps/Additional/Custom.cpp
+++ b/src/Dialect/ONNX/ONNXOps/Additional/Custom.cpp
@@ -12,7 +12,12 @@
 //
 //===----------------------------------------------------------------------===//
 
+// cmake --build . --target OMONNXOps
+
 #include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
+#include "mlir/IR/TypeUtilities.h" // For getElementTypeOrSelf
+#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp" // For shape helper base class
+#include <cassert> // For assert
 
 using namespace mlir;
 using namespace mlir::OpTrait::util;
@@ -24,8 +29,71 @@ using namespace onnx_mlir;
 
 LogicalResult ONNXCustomOp::inferShapes(
     std::function<void(Region &)> doShapeInference) {
-  // ToDo: this check could be refined to the shape related input,
-  // if inputs_for_infer is specified
+
+  // Special handling for FusedGemm
+  if (getFunctionName() == "FusedGemm") {
+    // Check if inputs A and B are RankedTensorType first.
+    if (!mlir::isa<RankedTensorType>(getInputs()[0].getType()) ||
+        !mlir::isa<RankedTensorType>(getInputs()[1].getType())) {
+      // Cannot infer shape if inputs are not ranked tensors yet.
+      return success();
+    }
+
+    // Inputs are ranked, cast them. aType and bType are now in scope.
+    auto aType = mlir::cast<RankedTensorType>(getInputs()[0].getType());
+    auto bType = mlir::cast<RankedTensorType>(getInputs()[1].getType());
+
+    // Check if inputs are 2D.
+    if (aType.getRank() != 2 || bType.getRank() != 2) {
+      // Cannot infer shape if inputs are not 2D.
+      // Let the verifier handle this as a potential error later.
+      return success();
+    }
+
+    // Get Gemm attributes
+    // Use getAttrOfType for safety, provide default if missing? Assume present for now.
+    bool transA = (*this)->getAttrOfType<IntegerAttr>("transA").getValue().getSExtValue() != 0;
+    bool transB = (*this)->getAttrOfType<IntegerAttr>("transB").getValue().getSExtValue() != 0;
+
+    // Get dimensions (handle transpose) - aType and bType are now accessible
+    int64_t M = transA ? aType.getShape()[1] : aType.getShape()[0];
+    int64_t K_A = transA ? aType.getShape()[0] : aType.getShape()[1]; // K from A
+    int64_t K_B = transB ? bType.getShape()[1] : bType.getShape()[0]; // K from B
+    int64_t N = transB ? bType.getShape()[0] : bType.getShape()[1];
+
+    // Validation: K dimensions must match if both are static
+    if (K_A != ShapedType::kDynamic && K_B != ShapedType::kDynamic && K_A != K_B) {
+       // This should ideally be caught by a verifier, but good to check here too.
+       // return emitOptionalError(getLoc(), "FusedGemm K dimensions mismatch");
+       return success(); // Let verifier handle error
+    }
+
+    // Determine output element type
+    // Use attribute if present, otherwise infer from input A (or B, should match)
+    Type outputElementType = getOutputElementType().value_or(aType.getElementType());
+
+    // Define the output shape: [M, N]
+    SmallVector<int64_t, 2> outputShapeVec;
+    outputShapeVec.push_back(M); // M can be dynamic
+    // N must be static for standard Gemm lowering/allocation.
+    // If N is dynamic, it might indicate an issue or require different handling.
+    if (N == ShapedType::kDynamic) {
+        // Let the verifier handle this potential issue later.
+        return success();
+    }
+    outputShapeVec.push_back(N);
+
+    // Manually set the result type to tensor<[M x N], outputElementType>
+    RankedTensorType newResTy =
+        RankedTensorType::get(outputShapeVec, outputElementType);
+
+    // opResult(0) is the first result
+    getResult(0).setType(newResTy);
+
+    return success(); // Successfully inferred shape for FusedGemm
+  }
+
+  // Original logic for other custom ops using shape_infer_pattern
   if (!hasShapeAndRank(getOperation()))
     return success();
 
@@ -38,17 +106,31 @@ LogicalResult ONNXCustomOp::inferShapes(
         "Shape inference pattern for multiple outputs NOT supported");
   }
 
-  // Deterimine the element type of output.
+  // Determine the element type of output.
   // Use output_element_type attribute if specified.
-  // Otherwise,  use the first input in the list of inputs_for_infer.
+  // Otherwise, use the first input in the list of inputs_for_infer.
   std::optional<ArrayAttr> inputIndexAttrs = getInputsForInfer();
   int64_t inputIdx = 0;
-  if (inputIndexAttrs.has_value())
-    inputIdx = mlir::cast<IntegerAttr>(inputIndexAttrs->getValue()[0]).getInt();
+  if (inputIndexAttrs.has_value() && !inputIndexAttrs.value().empty()) {
+    // Ensure the index is valid
+    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(inputIndexAttrs.value().getValue()[0])) {
+        inputIdx = intAttr.getInt();
+        if (inputIdx < 0 || inputIdx >= (int64_t)getInputs().size()) {
+             return emitError("Invalid input index in inputs_for_infer");
+        }
+    } else {
+        return emitError("Non-integer attribute in inputs_for_infer");
+    }
+  } else if (getInputs().empty()) {
+      // Cannot infer element type if there are no inputs and no attribute
+      return emitError("Cannot infer output element type: no inputs and no output_element_type attribute");
+  }
+
 
   Type elementType = getOutputElementType().value_or(
       getElementType(getInputs()[inputIdx].getType()));
 
+  // Use the base ShapeHelper for pattern-based inference
   ONNXCustomOpShapeHelper shapeHelper(getOperation(), {});
   return shapeHelper.computeShapeAndUpdateType(elementType);
-}
+}
\ No newline at end of file
diff --git a/src/Runtime/CMakeLists.txt b/src/Runtime/CMakeLists.txt
index e9e888f..755d83a 100644
--- a/src/Runtime/CMakeLists.txt
+++ b/src/Runtime/CMakeLists.txt
@@ -1,4 +1,3 @@
-
 # SPDX-License-Identifier: Apache-2.0
 
 if (NOT ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT)
@@ -52,6 +51,8 @@ add_onnx_mlir_library(OMTensorUtils
   OMTensorList.cpp
   OMUnique.cpp
   OnnxDataType.cpp
+  # Include the FusedGemm C++ placeholder implementation
+  FusedGemmRuntime.cpp
   ${ONNX_MLIR_SRC_ROOT}/src/Support/SmallFPConversion.c
 
   DEPENDS 
@@ -67,6 +68,10 @@ set_target_properties(OMTensorUtils
   POSITION_INDEPENDENT_CODE TRUE
   )
 
+# No need to add or link FusedGemmRuntime as a separate library.
+# Including FusedGemmRuntime.cpp in OMTensorUtils ensures all onnx-mlir generated model libraries
+# will have access to the fused gemm symbol automatically.
+
 if (ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT)
 add_compile_definitions(ENABLE_PYRUNTIME_LIGHT)
 add_onnx_mlir_library(OMExecutionSession
@@ -91,4 +96,4 @@ endif()
 set_target_properties(OMExecutionSession
   PROPERTIES
   POSITION_INDEPENDENT_CODE TRUE
-  )
+  )
\ No newline at end of file
diff --git a/src/Runtime/FusedGemmRuntime.cpp b/src/Runtime/FusedGemmRuntime.cpp
new file mode 100644
index 0000000..333ee63
--- /dev/null
+++ b/src/Runtime/FusedGemmRuntime.cpp
@@ -0,0 +1,120 @@
+#include <cstdint>
+#include <vector>
+#include <cmath>
+#include <iostream>
+#include <cstdio>
+
+// Include the ONNX-MLIR Runtime header defining OMTensor and API functions
+// Adjust the path based on your build/install location if necessary.
+// Common locations might be include/onnx-mlir/Runtime/ or similar.
+#include "OnnxMlirRuntime.h"
+
+// Basic ReLU activation
+float relu(float x) {
+    return std::max(0.0f, x);
+}
+
+// Modified implementation accepting OMTensor*
+extern "C" void ort_cpu_ep_fused_gemm(
+    OMTensor* A_omTensor,  // OMTensor for Matrix A
+    OMTensor* B_omTensor,  // OMTensor for Matrix B
+    OMTensor* Bias_omTensor,// OMTensor for Bias (can be NULL)
+    OMTensor* Y_omTensor,  // OMTensor for Output Y
+    int64_t M,             // Dimension M (passed directly)
+    int64_t N,             // Dimension N (passed directly)
+    int64_t K,             // Dimension K (passed directly)
+    int64_t transA,        // Transpose A flag (passed directly)
+    int64_t transB         // Transpose B flag (passed directly)
+) {
+
+    // Use fprintf to stderr to ensure it prints immediately before a crash
+    fprintf(stderr, ">>> C++ ort_cpu_ep_fused_gemm (OMTensor version) called:\n");
+    fprintf(stderr, "    M=%lld, N=%lld, K=%lld, transA=%lld, transB=%lld\n",
+            (long long)M, (long long)N, (long long)K, (long long)transA, (long long)transB);
+    fprintf(stderr, "    A OMTensor*: %p, B OMTensor*: %p, Bias OMTensor*: %p, Y OMTensor*: %p\n",
+            (void*)A_omTensor, (void*)B_omTensor, (void*)Bias_omTensor, (void*)Y_omTensor);
+    fflush(stderr);
+
+    // Check for NULL OMTensor pointers for required inputs/outputs
+    if (!A_omTensor || !B_omTensor || !Y_omTensor) {
+         fprintf(stderr, "    ERROR: Received NULL OMTensor pointer for A, B, or Y!\n");
+         fflush(stderr);
+         // Consider returning or aborting if essential tensors are missing
+         return; // Or handle error appropriately
+    }
+
+    // Extract raw data pointers from OMTensors
+    const float* A = static_cast<const float*>(omTensorGetDataPtr(A_omTensor));
+    const float* B = static_cast<const float*>(omTensorGetDataPtr(B_omTensor));
+    float* Y       = static_cast<float*>(omTensorGetDataPtr(Y_omTensor));
+    const float* Bias = nullptr; // Initialize Bias pointer to null
+
+    // Bias is optional, only extract if the OMTensor is not NULL
+    if (Bias_omTensor) {
+        Bias = static_cast<const float*>(omTensorGetDataPtr(Bias_omTensor));
+    }
+
+    fprintf(stderr, "    Extracted A ptr: %p, B ptr: %p, Bias ptr: %p, Y ptr: %p\n",
+            (void*)A, (void*)B, (void*)Bias, (void*)Y);
+    fflush(stderr);
+
+    // Check for NULL pointers *after* extraction (omTensorGetDataPtr might return null)
+    if (!A || !B || !Y) {
+         fprintf(stderr, "    ERROR: Extracted data pointer for A, B, or Y is NULL!\n");
+         fflush(stderr);
+         return; // Or handle error appropriately
+    }
+
+    // --- Core Logic (remains the same, using extracted pointers A, B, Bias, Y) ---
+    std::cout << ">>> Running C++ Placeholder: ort_cpu_ep_fused_gemm <<<" << std::endl;
+    std::cout << "    M=" << M << ", N=" << N << ", K=" << K
+              << ", transA=" << transA << ", transB=" << transB << std::endl;
+
+    for (int64_t m = 0; m < M; ++m) {
+        for (int64_t n = 0; n < N; ++n) {
+            float sum = 0.0f;
+            for (int64_t k = 0; k < K; ++k) {
+                int64_t a_idx = transA ? (k * M + m) : (m * K + k);
+                int64_t b_idx = transB ? (n * K + k) : (k * N + n);
+
+                // Basic bounds check (more important now with potentially complex layouts)
+                // TODO: Consider using omTensorGetStride(A_omTensor, dim) if layout isn't guaranteed row-major
+                // For now, assume dense row-major based on M, N, K for simplicity
+                int64_t max_a_idx = transA ? (K * M) : (M * K);
+                int64_t max_b_idx = transB ? (N * K) : (K * N);
+                if (a_idx < 0 || a_idx >= max_a_idx || b_idx < 0 || b_idx >= max_b_idx) {
+                     fprintf(stderr, "    ERROR: Calculated index out of bounds! a_idx=%lld (max %lld), b_idx=%lld (max %lld)\n",
+                             (long long)a_idx, (long long)max_a_idx, (long long)b_idx, (long long)max_b_idx);
+                     // Handle error: skip, return, abort?
+                     continue;
+                }
+
+                sum += A[a_idx] * B[b_idx];
+
+                // Debug print for first element
+                if (m == 0 && n == 0 && k == 0) {
+                     fprintf(stderr, "    Loop(0,0,0): a_idx=%lld, b_idx=%lld\n", (long long)a_idx, (long long)b_idx);
+                     fprintf(stderr, "    Loop(0,0,0): A[a_idx]=%f, B[b_idx]=%f\n", A[a_idx], B[b_idx]);
+                     fflush(stderr);
+                }
+            }
+
+            // Add Bias (check Bias pointer is valid)
+            float biased_sum = sum + (Bias ? Bias[n] : 0.0f);
+
+            // Apply ReLU activation
+            // TODO: Check bounds for Y write: m * N + n < M * N
+            if ((m * N + n) >= (M * N)) {
+                 fprintf(stderr, "    ERROR: Output index out of bounds! Y_idx=%lld (max %lld)\n",
+                         (long long)(m * N + n), (long long)(M * N));
+                 continue; // Skip write
+            }
+            Y[m * N + n] = relu(biased_sum);
+        }
+    }
+    // --- End Core Logic ---
+
+    std::cout << ">>> Finished C++ Placeholder: ort_cpu_ep_fused_gemm <<<" << std::endl;
+    fprintf(stderr, ">>> Finished C++ ort_cpu_ep_fused_gemm (OMTensor version) <<<\n");
+    fflush(stderr);
+}
\ No newline at end of file
diff --git a/src/matmul_relu_matmul_fashion_mnist/FusedGemmRuntime.cpp b/src/matmul_relu_matmul_fashion_mnist/FusedGemmRuntime.cpp
new file mode 100644
index 0000000..06cf348
--- /dev/null
+++ b/src/matmul_relu_matmul_fashion_mnist/FusedGemmRuntime.cpp
@@ -0,0 +1,106 @@
+#include <cstdint> // For int64_t
+#include <vector>
+#include <cmath>   // For std::max
+#include <iostream> // For debug prints
+#include <cstdio> // Include for fprintf
+
+// build command: cmake --build . --target OMTensorUtils
+
+// Basic ReLU activation
+float relu(float x) {
+    return std::max(0.0f, x);
+}
+
+// Placeholder implementation for FusedGemm: Y = ReLU(alpha * (A @ B) + beta * Bias)
+// We simplify to: Y = ReLU((A @ B) + Bias) by assuming alpha=1.0, beta=1.0 for this placeholder.
+extern "C" void ort_cpu_ep_fused_gemm(
+    const float* A,     // Pointer to Matrix A data
+    const float* B,     // Pointer to Matrix B data
+    const float* Bias,  // Pointer to Bias data
+    float* Y,           // Pointer to Output data Y
+    int64_t M,          // Dimension M of Output (Rows of A or B')
+    int64_t N,          // Dimension N of Output (Cols of B or A')
+    int64_t K,          // Dimension K (Cols of A or Rows of A', Rows of B or Cols of B')
+    int64_t transA,     // Transpose A flag (0 or 1)
+    int64_t transB      // Transpose B flag (0 or 1)
+    // Note: alpha, beta, and activation type are ignored in this simple placeholder
+) {
+
+    // Use fprintf to stderr to ensure it prints immediately before a crash
+    fprintf(stderr, ">>> C++ ort_cpu_ep_fused_gemm called:\n");
+    fprintf(stderr, "    M=%lld, N=%lld, K=%lld, transA=%lld, transB=%lld\n",
+            (long long)M, (long long)N, (long long)K, (long long)transA, (long long)transB);
+    fprintf(stderr, "    A ptr: %p, B ptr: %p, Bias ptr: %p, Y ptr: %p\n",
+            (void*)A, (void*)B, (void*)Bias, (void*)Y);
+    fflush(stderr); // Ensure output is flushed
+
+    // Check for NULL pointers immediately
+    if (!A || !B || !Y) { // Bias might be optional
+         fprintf(stderr, "    ERROR: Received NULL pointer for A, B, or Y!\n");
+         fflush(stderr);
+         // Decide how to handle: maybe return early or abort?
+         // For debugging, maybe just continue to see where it crashes.
+    }
+
+    std::cout << ">>> Running C++ Placeholder: ort_cpu_ep_fused_gemm <<<" << std::endl;
+    std::cout << "    M=" << M << ", N=" << N << ", K=" << K
+              << ", transA=" << transA << ", transB=" << transB << std::endl;
+
+    // Simple Row-Major Matrix Multiplication Logic
+    for (int64_t m = 0; m < M; ++m) {
+        for (int64_t n = 0; n < N; ++n) {
+            float sum = 0.0f;
+            for (int64_t k = 0; k < K; ++k) {
+                // Calculate indices based on transpose flags
+                // Assuming Row-Major layout for both A and B
+                int64_t a_idx = transA ? (k * M + m) : (m * K + k); // Index for A[m, k] or A[k, m]
+                int64_t b_idx = transB ? (n * K + k) : (k * N + n); // Index for B[k, n] or B[n, k]
+
+                // Basic bounds check (optional but good practice)
+                // These checks depend heavily on how strides/leading dimensions would be handled
+                // For placeholder, assume dense packing based on M, N, K
+                // if (a_idx >= (transA ? K*M : M*K) || b_idx >= (transB ? N*K : K*N)) {
+                //     std::cerr << "Error: Index out of bounds!" << std::endl;
+                //     continue; // Skip this element
+                // }
+
+                sum += A[a_idx] * B[b_idx];
+
+                // Add print inside the loop (maybe only first iteration)
+                if (m == 0 && n == 0 && k == 0) {
+                     fprintf(stderr, "    Loop(0,0,0): a_idx=%lld, b_idx=%lld\n", (long long)a_idx, (long long)b_idx);
+                     // Maybe print A[a_idx] and B[b_idx] if pointers are valid
+                     if (A && B) {
+                         // Add bounds checks before accessing!
+                         // Example check (adjust based on actual expected sizes):
+                         bool a_ok = transA ? (k < K && m < M) : (m < M && k < K);
+                         bool b_ok = transB ? (n < N && k < K) : (k < K && n < N);
+                         if (a_ok && b_ok) {
+                            // Calculate actual max index based on dimensions
+                            int64_t max_a_idx = transA ? (K * M) : (M * K);
+                            int64_t max_b_idx = transB ? (N * K) : (K * N);
+                            if (a_idx >= 0 && a_idx < max_a_idx && b_idx >= 0 && b_idx < max_b_idx) {
+                               fprintf(stderr, "    Loop(0,0,0): A[a_idx]=%f, B[b_idx]=%f\n", A[a_idx], B[b_idx]);
+                            } else {
+                               fprintf(stderr, "    Loop(0,0,0): Calculated indices a_idx=%lld or b_idx=%lld seem out of bounds (max_a=%lld, max_b=%lld)!\n",
+                                       (long long)a_idx, (long long)b_idx, (long long)max_a_idx, (long long)max_b_idx);
+                            }
+                         } else {
+                             fprintf(stderr, "    Loop(0,0,0): m, n, or k seem out of bounds for dimensions M, N, K!\n");
+                         }
+                     }
+                     fflush(stderr);
+                }
+            }
+
+            // Add Bias (assuming Bias has size N)
+            float biased_sum = sum + (Bias ? Bias[n] : 0.0f);
+
+            // Apply ReLU activation
+            Y[m * N + n] = relu(biased_sum);
+        }
+    }
+     std::cout << ">>> Finished C++ Placeholder: ort_cpu_ep_fused_gemm <<<" << std::endl;
+     fprintf(stderr, ">>> Finished C++ ort_cpu_ep_fused_gemm <<<\n");
+     fflush(stderr);
+}
\ No newline at end of file
diff --git a/src/matmul_relu_matmul_fashion_mnist/FusedGemmRuntime_omtensor.cpp b/src/matmul_relu_matmul_fashion_mnist/FusedGemmRuntime_omtensor.cpp
new file mode 100644
index 0000000..6bc6f0c
--- /dev/null
+++ b/src/matmul_relu_matmul_fashion_mnist/FusedGemmRuntime_omtensor.cpp
@@ -0,0 +1,252 @@
+/**********************************************
+ * IMPORT LIBRARIES
+ **********************************************/
+#include <cstdint>  // For int64_t
+#include <vector>   // Not strictly used here, but common
+#include <cmath>    // For std::max
+#include <iostream> // For std::cout, std::endl (placeholder logging)
+#include <cstdio>   // For fprintf, stderr, fflush (debug logging)
+
+// ONNX-MLIR Runtime API
+#include "OnnxMlirRuntime.h"
+
+/**********************************************
+ * CONSTANTS & PARAMETERS
+ **********************************************/
+// None defined for this specific file.
+
+/**********************************************
+ * HELPER FUNCTION DEFINITIONS
+ **********************************************/
+
+/*
+ * Purpose: Basic ReLU activation function.
+ * Parameters:
+ *    - x (float): Input value.
+ * Returns:
+ *    - float: max(0.0f, x).
+ */
+inline float relu(float x) {
+    return std::max(0.0f, x);
+}
+
+/*
+ * Purpose: Compute the linear offset into a flat buffer for a 2D tensor
+ *          given its strides and logical indices.
+ * Parameters:
+ *    - strides (const int64_t*): Pointer to the strides array for the tensor.
+ *                                 Assumes strides[0] is stride for dim 0, strides[1] for dim 1.
+ *    - i (int64_t): Logical index for the first dimension.
+ *    - j (int64_t): Logical index for the second dimension.
+ * Returns:
+ *    - int64_t: The calculated offset.
+ */
+inline int64_t offset2d(const int64_t* strides, int64_t i, int64_t j) {
+    // Handle potential null strides defensively, although unlikely for valid tensors
+    if (!strides) return 0; // Or handle error appropriately
+    return i * strides[0] + j * strides[1];
+}
+
+/*
+ * Purpose: Compute the linear offset into a flat buffer for a 1D tensor
+ *          given its stride and logical index.
+ * Parameters:
+ *    - strides (const int64_t*): Pointer to the strides array (only strides[0] is used).
+ *    - i (int64_t): Logical index for the dimension.
+ * Returns:
+ *    - int64_t: The calculated offset.
+ */
+inline int64_t offset1d(const int64_t* strides, int64_t i) {
+    if (!strides) return 0;
+    return i * strides[0];
+}
+
+
+/**********************************************
+ * MAIN RUNTIME FUNCTION DEFINITION
+ **********************************************/
+
+/*
+ * Purpose: Implements the FusedGemm operation (Gemm + Bias + ReLU) using OMTensor inputs.
+ *          Mimics ONNX Gemm (alpha=1, beta=1) followed by ONNX ReLU.
+ *          Handles tensor strides and bias broadcasting.
+ * Parameters:
+ *    - A_omTensor (OMTensor*): Input tensor A (MxK or KxM).
+ *    - B_omTensor (OMTensor*): Input tensor B (KxN or NxK).
+ *    - Bias_omTensor (OMTensor*): Optional input tensor C/Bias, broadcastable to (MxN).
+ *    - Y_omTensor (OMTensor*): Output tensor Y (MxN).
+ *    - M (int64_t): Dimension M of the output.
+ *    - N (int64_t): Dimension N of the output.
+ *    - K (int64_t): Dimension K (shared dimension).
+ *    - transA (int64_t): Flag indicating if A should be transposed (0=No, Non-zero=Yes).
+ *    - transB (int64_t): Flag indicating if B should be transposed (0=No, Non-zero=Yes).
+ * Returns:
+ *    - void: Output Y is modified in place.
+ */
+extern "C" void ort_cpu_ep_fused_gemm(
+    OMTensor* A_omTensor,
+    OMTensor* B_omTensor,
+    OMTensor* Bias_omTensor, // Corresponds to Gemm's 'C' input
+    OMTensor* Y_omTensor,
+    int64_t M,
+    int64_t N,
+    int64_t K,
+    int64_t transA,
+    int64_t transB
+) {
+
+    /******************************************
+     * INITIAL LOGGING & VALIDATION
+     ******************************************/
+    // Use fprintf for immediate output, helpful before potential crashes
+    fprintf(stderr, ">>> C++ ort_cpu_ep_fused_gemm (OMTensor version) called:\n");
+    fprintf(stderr, "    M=%lld, N=%lld, K=%lld, transA=%lld, transB=%lld\n",
+            (long long)M, (long long)N, (long long)K, (long long)transA, (long long)transB);
+    fprintf(stderr, "    A OMTensor*: %p, B OMTensor*: %p, Bias OMTensor*: %p, Y OMTensor*: %p\n",
+            (void*)A_omTensor, (void*)B_omTensor, (void*)Bias_omTensor, (void*)Y_omTensor);
+    fflush(stderr);
+
+    // Check for NULL OMTensor pointers for required inputs/outputs
+    if (!A_omTensor || !B_omTensor || !Y_omTensor) {
+         fprintf(stderr, "    ERROR: Received NULL OMTensor pointer for A, B, or Y!\n");
+         fflush(stderr);
+         return; // Cannot proceed
+    }
+
+    /******************************************
+     * EXTRACT DATA POINTERS & METADATA
+     ******************************************/
+    // Extract raw data pointers (assuming float32 based on typical usage)
+    // TODO: Add type checking if supporting other data types is needed.
+    const float* A_data = static_cast<const float*>(omTensorGetDataPtr(A_omTensor));
+    const float* B_data = static_cast<const float*>(omTensorGetDataPtr(B_omTensor));
+    float* Y_data       = static_cast<float*>(omTensorGetDataPtr(Y_omTensor));
+
+    // Get strides for A, B, Y (crucial for correct indexing)
+    const int64_t* strideA = omTensorGetStrides(A_omTensor);
+    const int64_t* strideB = omTensorGetStrides(B_omTensor);
+    const int64_t* strideY = omTensorGetStrides(Y_omTensor);
+
+    // Check for NULL pointers *after* extraction
+    if (!A_data || !B_data || !Y_data || !strideA || !strideB || !strideY) {
+         fprintf(stderr, "    ERROR: Extracted data pointer or strides for A, B, or Y is NULL!\n");
+         fprintf(stderr, "    A_data=%p, B_data=%p, Y_data=%p, strideA=%p, strideB=%p, strideY=%p\n",
+                 (void*)A_data, (void*)B_data, (void*)Y_data, (void*)strideA, (void*)strideB, (void*)strideY);
+         fflush(stderr);
+         return; // Cannot proceed
+    }
+
+    // Extract Bias data (if present)
+    const float* Bias_data = nullptr;
+    const int64_t* strideBias = nullptr;
+    const int64_t* dimsBias = nullptr;
+    int biasRank = 0;
+    if (Bias_omTensor) {
+        Bias_data = static_cast<const float*>(omTensorGetDataPtr(Bias_omTensor));
+        strideBias = omTensorGetStrides(Bias_omTensor);
+        /*(doesn't work)*/ //dimsBias = omTensorGetDimensions(Bias_omTensor); // Needed for broadcasting rules
+        dimsBias = omTensorGetShape(Bias_omTensor); // Assuming this function gives the shape/dims
+        biasRank = omTensorGetRank(Bias_omTensor);
+
+        if (!Bias_data || !strideBias || !dimsBias) {
+             fprintf(stderr, "    WARNING: Bias OMTensor exists but extracted data pointer, strides, or dims is NULL! Treating as no bias.\n");
+             fprintf(stderr, "    Bias_data=%p, strideBias=%p, dimsBias=%p\n", (void*)Bias_data, (void*)strideBias, (void*)dimsBias);
+             fflush(stderr);
+             Bias_data = nullptr; // Treat as if no bias was provided
+        } else {
+             fprintf(stderr, "    Bias Info: Rank=%d, Data=%p, Strides=%p, Dims=%p\n", biasRank, (void*)Bias_data, (void*)strideBias, (void*)dimsBias);
+             // Optional: Print actual dims/strides
+             // for(int i=0; i<biasRank; ++i) fprintf(stderr, " Bias dim[%d]=%lld, stride[%d]=%lld\n", i, (long long)dimsBias[i], i, (long long)strideBias[i]);
+        }
+    } else {
+        fprintf(stderr, "    Bias OMTensor is NULL.\n");
+    }
+    fflush(stderr);
+
+
+    /******************************************
+     * CORE GEMM + BIAS + RELU LOGIC
+     ******************************************/
+    // Note: This implementation assumes alpha=1.0 and beta=1.0 as per Gemm defaults,
+    // because these attributes are not passed to this custom function.
+
+    for (int64_t m = 0; m < M; ++m) {
+        for (int64_t n = 0; n < N; ++n) {
+
+            // --- GEMM Calculation (A' * B') ---
+            float gemm_sum = 0.0f;
+            for (int64_t k = 0; k < K; ++k) {
+                // Determine logical indices based on transpose flags
+                int64_t a_idx_dim0 = transA ? k : m;
+                int64_t a_idx_dim1 = transA ? m : k;
+                int64_t b_idx_dim0 = transB ? n : k;
+                int64_t b_idx_dim1 = transB ? k : n;
+
+                // Calculate physical offsets using strides
+                int64_t offset_a = offset2d(strideA, a_idx_dim0, a_idx_dim1);
+                int64_t offset_b = offset2d(strideB, b_idx_dim0, b_idx_dim1);
+
+                // Accumulate product
+                gemm_sum += A_data[offset_a] * B_data[offset_b];
+            } // End K loop
+
+            // --- Bias Addition (gemm_sum + C/Bias) ---
+            // Implements unidirectional broadcasting for C/Bias to shape (M, N)
+            bool biasIsValid = (Bias_data && strideBias && dimsBias);
+            float bias_val = 0.0f;
+            if (biasIsValid) {
+              switch (biasRank) {
+                case 0:
+                  bias_val = Bias_data[0];
+                  break;
+                case 1:
+                  // length == N? broadcast across rows
+                  if (dimsBias[0] == N)
+                    bias_val = Bias_data[offset1d(strideBias, n)];
+                  // length == M? broadcast across cols
+                  else if (dimsBias[0] == M)
+                    bias_val = Bias_data[offset1d(strideBias, m)];
+                  // length == 1? scalar
+                  else if (dimsBias[0] == 1)
+                    bias_val = Bias_data[0];
+                  else
+                    fprintf(stderr, "[FusedGemm] Bad 1D bias length %lld\n", (long long)dimsBias[0]);
+                  break;
+                case 2:
+                  if (dimsBias[0] == M && dimsBias[1] == N)
+                    bias_val = Bias_data[offset2d(strideBias, m, n)];
+                  else if (dimsBias[0] == 1 && dimsBias[1] == N)
+                    bias_val = Bias_data[offset2d(strideBias, 0, n)];
+                  else if (dimsBias[0] == M && dimsBias[1] == 1)
+                    bias_val = Bias_data[offset2d(strideBias, m, 0)];
+                  else if (dimsBias[0] == 1 && dimsBias[1] == 1)
+                    bias_val = Bias_data[0];
+                  else
+                    fprintf(stderr, "[FusedGemm] Bad 2D bias shape %lldx%lld\n",
+                            (long long)dimsBias[0], (long long)dimsBias[1]);
+                  break;
+                default:
+                  fprintf(stderr, "[FusedGemm] Unsupported bias rank %d\n", biasRank);
+              }
+            }
+            // Assuming beta = 1.0 for Bias/C
+            float biased_sum = gemm_sum + bias_val;
+
+            // --- ReLU Activation ---
+            float final_result = relu(biased_sum);
+
+            // --- Store Result in Output Tensor Y ---
+            // Calculate output offset using strides
+            int64_t offset_y = offset2d(strideY, m, n);
+            Y_data[offset_y] = final_result;
+
+        } // End N loop
+    } // End M loop
+
+    /******************************************
+     * FINAL LOGGING
+     ******************************************/
+    // std::cout << ">>> Finished C++ Placeholder: ort_cpu_ep_fused_gemm <<<" << std::endl; // Less critical logging
+    fprintf(stderr, ">>> Finished C++ ort_cpu_ep_fused_gemm (OMTensor version) <<<\n");
+    fflush(stderr);
+}
\ No newline at end of file
diff --git a/src/matmul_relu_matmul_fashion_mnist/RunONNXModel.py b/src/matmul_relu_matmul_fashion_mnist/RunONNXModel.py
new file mode 100755
index 0000000..2c4c284
--- /dev/null
+++ b/src/matmul_relu_matmul_fashion_mnist/RunONNXModel.py
@@ -0,0 +1,1110 @@
+#!/usr/bin/env python3
+# SPDX-License-Identifier: Apache-2.0
+
+##################### RunONNXModel.py #########################################
+#
+# Copyright 2019-2025 The IBM Research Authors.
+#
+################################################################################
+#
+# This script is to run and debug an onnx model.
+
+################################################################################
+
+import os
+import sys
+import argparse
+import onnx
+import time
+import signal
+import subprocess
+import numpy as np
+import tempfile
+import json
+import importlib.util
+import shlex
+import shutil
+
+from onnx import numpy_helper
+from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
+from collections import OrderedDict
+
+################################################################################
+# Test environment and set global environment variables.
+
+if not os.environ.get("ONNX_MLIR_HOME", None):
+    raise RuntimeError(
+        "Environment variable ONNX_MLIR_HOME is not set, please set it to the"
+        " path to the HOME directory for onnx-mlir. The HOME directory for"
+        " onnx-mlir refers to the parent folder containing the bin, lib, etc"
+        " sub-folders in which ONNX-MLIR executables and libraries can be found,"
+        " typically `onnx-mlir/build/Debug`."
+    )
+ONNX_MLIR_EXENAME = "onnx-mlir.exe" if sys.platform == "win32" else "onnx-mlir"
+ONNX_MLIR = os.path.join(os.environ["ONNX_MLIR_HOME"], "bin", ONNX_MLIR_EXENAME)
+# Include runtime directory in python paths, so PyRuntime can be imported.
+RUNTIME_DIR = os.path.join(os.environ["ONNX_MLIR_HOME"], "lib")
+sys.path.append(RUNTIME_DIR)
+
+VERBOSE = os.environ.get("VERBOSE", False)
+
+################################################################################
+# Check and import Onnx Mlir Execution session / python interface.
+
+try:
+    from PyRuntime import OMExecutionSession
+except ImportError:
+    raise ImportError(
+        "Looks like you did not build the PyRuntime target, build it by running"
+        " `make PyRuntime`. You may need to set ONNX_MLIR_HOME to"
+        " `onnx-mlir/build/Debug` since `make PyRuntime` outputs to"
+        " `build/Debug` by default."
+    )
+
+################################################################################
+# Support functions for parsing environment.
+
+
+def valid_onnx_input(fname):
+    valid_exts = ["onnx", "mlir", "onnxtext"]
+    ext = os.path.splitext(fname)[1][1:]
+
+    if ext not in valid_exts:
+        parser.error(
+            "Only accept an input model with one of extensions {}".format(valid_exts)
+        )
+    return fname
+
+
+def check_positive(argname, value):
+    value = int(value)
+    if value <= 0:
+        parser.error("Value passed to {} must be positive".format(argname))
+    return value
+
+
+def check_non_negative(argname, value):
+    value = int(value)
+    if value < 0:
+        parser.error("Value passed to {} must be non-negative".format(argname))
+    return value
+
+
+################################################################################
+# Command arguments.
+
+parser = argparse.ArgumentParser()
+parser.add_argument(
+    "--log-to-file",
+    action="store",
+    nargs="?",
+    const="compilation.log",
+    default=None,
+    help="Output compilation messages to file, default compilation.log.",
+)
+parser.add_argument(
+    "-m",
+    "--model",
+    type=lambda s: valid_onnx_input(s),
+    help="Path to an ONNX model (.onnx or .mlir).",
+)
+parser.add_argument(
+    "-c",
+    "--compile-args",
+    type=str,
+    default="",
+    help="Arguments passed directly to onnx-mlir command." " See bin/onnx-mlir --help.",
+)
+parser.add_argument(
+    "-C", "--compile-only", action="store_true", help="Only compile the input model."
+)
+parser.add_argument("--print-input", action="store_true", help="Print out inputs.")
+parser.add_argument(
+    "--print-output",
+    action="store_true",
+    help="Print out inference outputs produced by onnx-mlir.",
+)
+parser.add_argument(
+    "--print-signatures",
+    action="store_true",
+    help="Print out the input and output signatures of the model.",
+)
+parser.add_argument(
+    "--save-onnx",
+    metavar="PATH",
+    type=str,
+    help="File path to save the onnx model. Only effective if --verify=onnxruntime.",
+)
+parser.add_argument(
+    "--verify",
+    choices=["onnxruntime", "ref"],
+    help="Verify the output by using onnxruntime or reference"
+    " inputs/outputs. By default, no verification. When being"
+    " enabled, --verify-with-softmax or --verify-every-value"
+    " must be used to specify verification mode.",
+)
+parser.add_argument(
+    "--verify-all-ops",
+    action="store_true",
+    help="Verify all operation outputs when using onnxruntime.",
+)
+parser.add_argument(
+    "--verify-with-softmax",
+    metavar="AXIS_INDEX",
+    type=str,
+    default=None,
+    help="Verify the result obtained by applying softmax along with"
+    " specific axis. The axis can be specified"
+    " by --verify-with-softmax=<axis>.",
+)
+parser.add_argument(
+    "--verify-every-value",
+    action="store_true",
+    help="Verify every value of the output using atol and rtol.",
+)
+parser.add_argument(
+    "--rtol", type=str, default="0.05", help="Relative tolerance for verification."
+)
+parser.add_argument(
+    "--atol", type=str, default="0.01", help="Absolute tolerance for verification."
+)
+
+lib_group = parser.add_mutually_exclusive_group()
+lib_group.add_argument(
+    "--save-model",
+    metavar="PATH",
+    type=str,
+    help="Path to a folder to save the compiled model.",
+)
+lib_group.add_argument(
+    "--load-model",
+    metavar="PATH",
+    type=str,
+    help="Path to a folder to load a compiled model for "
+    "inference, and the ONNX model will not be re-compiled.",
+)
+lib_group.add_argument(
+    "--cache-model",
+    metavar="PATH",
+    type=str,
+    help="When finding a compiled model in given path, reuse it. "
+    "Otherwise, compile model and save it into the given path.",
+)
+
+parser.add_argument(
+    "-o",
+    "--default-model-name",
+    metavar="MODEL_NAME",
+    type=str,
+    default="model",
+    help="Change the default model name that is used for two generated files: "
+    " .so and .constants.bin. Default is model.",
+)
+
+parser.add_argument(
+    "--save-ref",
+    metavar="PATH",
+    type=str,
+    help="Path to a folder to save the inputs and outputs in protobuf.",
+)
+data_group = parser.add_mutually_exclusive_group()
+data_group.add_argument(
+    "--load-ref",
+    metavar="PATH",
+    type=str,
+    help="Path to a folder containing reference inputs and outputs stored in protobuf."
+    " If --verify=ref, inputs and outputs are reference data for verification.",
+)
+data_group.add_argument(
+    "--inputs-from-arrays", help="List of numpy arrays used as inputs for inference."
+)
+data_group.add_argument(
+    "--load-ref-from-numpy",
+    metavar="PATH",
+    type=str,
+    help="Path to a python script that defines variables inputs and outputs that are"
+    " a list of numpy arrays. "
+    " For example, inputs = [np.array([1], dtype=np.int64), np.array([2], dtype=np.float32]."
+    " Variable outputs can be omitted if --verify is not used.",
+)
+data_group.add_argument(
+    "--shape-info",
+    type=str,
+    help="Shape for each dynamic input of the model, e.g. 0:1x10x20,1:7x5x3. "
+    "Used to generate random inputs for the model if --load-ref is not set.",
+)
+
+parser.add_argument(
+    "--lower-bound",
+    type=str,
+    help="Lower bound values for each data type. Used inputs."
+    " E.g. --lower-bound=int64:-10,float32:-0.2,uint8:1."
+    " Supported types are bool, uint8, int8, uint16, int16, uint32, int32,"
+    " uint64, int64,float16, float32, float64.",
+)
+parser.add_argument(
+    "--upper-bound",
+    type=str,
+    help="Upper bound values for each data type. Used to generate random inputs."
+    " E.g. --upper-bound=int64:10,float32:0.2,uint8:9."
+    " Supported types are bool, uint8, int8, uint16, int16, uint32, int32,"
+    " uint64, int64, float16, float32, float64.",
+)
+parser.add_argument(
+    "-w",
+    "--warmup",
+    type=lambda s: check_non_negative("--warmup", s),
+    default=0,
+    help="The number of warmup inference runs.",
+)
+parser.add_argument(
+    "-n",
+    "--n-iteration",
+    type=lambda s: check_positive("--n-iteration", s),
+    default=1,
+    help="The number of inference runs excluding warmup.",
+)
+parser.add_argument(
+    "--seed",
+    type=str,
+    default="42",
+    help="seed to initialize the random num generator for inputs.",
+)
+
+
+def verify_arg():
+    if (
+        args.verify
+        and (args.verify_with_softmax is None)
+        and (not args.verify_every_value)
+    ):
+        raise RuntimeError(
+            "Choose verification mode: --verify-with-softmax or "
+            "--verify-every-value or both"
+        )
+    if args.verify_with_softmax is not None and (not args.verify):
+        raise RuntimeError("Must specify --verify to use --verify-with-softmax")
+    if args.verify_every_value and (not args.verify):
+        raise RuntimeError("Must specify --verify to use --verify-every-value")
+
+    if args.verify and args.verify.lower() == "onnxruntime":
+        if not args.model or (args.model and not args.model.endswith(".onnx")):
+            raise RuntimeError(
+                "Set input onnx model using argument --model when verifying"
+                " using onnxruntime."
+            )
+
+
+################################################################################
+# Support functions for RunONNXModel functionality.
+# Functions are free of args (all needed parameters are passed to the function).
+
+
+# A type mapping from MLIR to Numpy.
+MLIR_TYPE_TO_NP_TYPE = {
+    "f64": np.dtype("float64"),
+    "f32": np.dtype("float32"),
+    "f16": np.dtype("float16"),
+    "i64": np.dtype("int64"),
+    "i32": np.dtype("int32"),
+    "i16": np.dtype("int16"),
+    "i8": np.dtype("int8"),
+    "ui64": np.dtype("uint64"),
+    "ui32": np.dtype("uint32"),
+    "ui16": np.dtype("uint16"),
+    "ui8": np.dtype("uint8"),
+    "i1": np.dtype("bool"),
+    "string": np.dtype("str_"),
+}
+
+# Default lower bound for generating random inputs.
+DEFAULT_LB = {
+    "float64": -0.1,
+    "float32": -0.1,
+    "float16": -0.1,
+    "int64": -10,
+    "int32": -10,
+    "int16": -10,
+    "int8": -10,
+    "uint64": 0,
+    "uint32": 0,
+    "uint16": 0,
+    "uint8": 0,
+    # For some reason, random.uniform with lb/ub to 0/1 resulted in 1 only.
+    "bool": -10,  # treated as int32
+}
+
+# Default upper bound for generating random inputs.
+DEFAULT_UB = {
+    "float64": 0.1,
+    "float32": 0.1,
+    "float16": 0.1,
+    "int64": 10,
+    "int32": 10,
+    "int16": 10,
+    "int8": 10,
+    "uint64": 10,
+    "uint32": 10,
+    "uint16": 10,
+    "uint8": 10,
+    # For some reason, random.uniform with lb/ub to 0/1 resulted in 1 only.
+    "bool": 9,  # treated as int32
+}
+
+
+def ordinal(n):
+    suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
+    if 11 <= (n % 100) <= 13:
+        suffix = "th"
+    return str(n) + suffix
+
+
+def softmax(x, axis_value):
+    return np.exp(x) / np.sum(np.exp(x), axis=axis_value, keepdims=True)
+
+
+def execute_commands(cmds):
+    if VERBOSE:
+        print(cmds)
+    out = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
+    stdout, stderr = out.communicate()
+    msg = stderr.decode("utf-8") + stdout.decode("utf-8")
+    if out.returncode == -signal.SIGSEGV:
+        return (False, "Segfault")
+    if out.returncode != 0:
+        return (False, msg)
+    return (True, msg)
+
+
+def extend_model_output(model, intermediate_outputs):
+    # Run shape inference to make sure we have valid tensor value infos for all
+    # intermediate tensors available
+    model = onnx.shape_inference.infer_shapes(model)
+    value_infos = {vi.name: vi for vi in model.graph.value_info}
+    graph_inputs = {vi.name: vi for vi in model.graph.input}
+    graph_outputs = {vi.name: vi for vi in model.graph.output}
+
+    # Retrieve tensor value info for each intermediate output
+    new_outputs = []
+    for name in intermediate_outputs:
+        if name in value_infos:
+            new_outputs.append(value_infos[name])
+        elif name in graph_inputs:
+            new_outputs.append(graph_inputs[name])
+        elif name in graph_outputs:
+            new_outputs.append(graph_outputs[name])
+        else:
+            raise RuntimeError(f"Unable to find value infos for {name}")
+
+    # Clear old graph outputs and replace by new set of intermediate outputs
+    while len(model.graph.output):
+        model.graph.output.pop()
+
+    model.graph.output.extend(new_outputs)
+    return model
+
+
+def get_names_in_signature(signature):
+    names = []
+    # Load the input signature.
+    signature_dict = json.loads(signature)
+    for sig in signature_dict:
+        names.append(sig["name"])
+    return names
+
+
+def read_input_from_refs(num_inputs, load_ref_filename, is_load_ref):
+    print("Reading inputs from {} ...".format(load_ref_filename))
+    inputs = []
+
+    if is_load_ref:
+        for i in range(num_inputs):
+            input_file = load_ref_filename + "/input_{}.pb".format(i)
+            input_ts = onnx.TensorProto()
+            with open(input_file, "rb") as f:
+                input_ts.ParseFromString(f.read())
+            input_np = numpy_helper.to_array(input_ts)
+            inputs += [input_np]
+    else:
+        spec = importlib.util.spec_from_file_location("om_load_ref", load_ref_filename)
+        module = importlib.util.module_from_spec(spec)
+        spec.loader.exec_module(module)
+        inputs = module.inputs
+
+    for i in range(len(inputs)):
+        input_np = inputs[i]
+        print(
+            "  - {} input: [{}x{}]".format(
+                ordinal(i + 1),
+                "x".join([str(i) for i in input_np.shape]),
+                input_np.dtype,
+            )
+        )
+
+    print("  done.\n")
+    return inputs
+
+
+def read_output_from_refs(num_outputs, load_ref_filename, is_load_ref):
+    print("Reading reference outputs from {} ...".format(load_ref_filename))
+    reference_output = []
+
+    if is_load_ref:
+        for i in range(num_outputs):
+            output_file = load_ref_filename + "/output_{}.pb".format(i)
+            output_ts = onnx.TensorProto()
+            with open(output_file, "rb") as f:
+                output_ts.ParseFromString(f.read())
+            output_np = numpy_helper.to_array(output_ts)
+            reference_output += [output_np]
+    else:
+        spec = importlib.util.spec_from_file_location(
+            "om_load_ref_output", load_ref_filename
+        )
+        module = importlib.util.module_from_spec(spec)
+        spec.loader.exec_module(module)
+        reference_output = module.outputs
+
+    for i in range(len(reference_output)):
+        output_np = reference_output[i]
+        print(
+            "  - {} output: [{}x{}]".format(
+                ordinal(i + 1),
+                "x".join([str(i) for i in output_np.shape]),
+                output_np.dtype,
+            )
+        )
+    print("  done.\n")
+    return reference_output
+
+
+def generate_random_input(input_signature, shape_info, seed, lower_bound, upper_bound):
+    # Load random values: first get shape info, where shape_info in the form of
+    # 'input_index:d1xd2, input_index:d1xd2'
+    input_shapes = {}
+    if shape_info:
+        for input_shape in shape_info.strip().split(","):
+            input_index_shape = input_shape.split(":")
+            input_index = input_index_shape[0]
+            assert not (input_index in input_shapes), "Duplicate input indices"
+            dims = [int(d) for d in input_index_shape[1].split("x")]
+            input_shapes[int(input_index)] = dims
+
+    # Then fill shapes with random numbers.
+    # Numpy expect an int, tolerate int/float strings.
+    curr_seed = int(float(seed))
+    print("Generating random inputs using seed", curr_seed, "...")
+    # Generate random data as input.
+    inputs = []
+
+    # Load the input signature.
+    signature = json.loads(input_signature)
+
+    np.random.seed(curr_seed)
+    for i, sig in enumerate(signature):
+        # Get shape.
+        explicit_shape = []
+        for d, dim in enumerate(sig["dims"]):
+            if dim >= 0:
+                explicit_shape.append(dim)
+                continue
+            if i in input_shapes:
+                if d < len(input_shapes[i]):
+                    explicit_shape.append(input_shapes[i][d])
+                else:
+                    print(
+                        "The {} dim".format(ordinal(d + 1)),
+                        "of the {} input is unknown.".format(ordinal(i + 1)),
+                        "Use --shape-info to set.",
+                    )
+                    print(" - The input signature: ", sig)
+                    exit(1)
+            else:
+                print(
+                    "The shape of the {} input".format(ordinal(i + 1)),
+                    "is unknown. Use --shape-info to set.",
+                )
+                print(" - The input signature: ", sig)
+                exit(1)
+        # Get element type.
+        elem_type = sig["type"]
+        np_elem_type = MLIR_TYPE_TO_NP_TYPE[elem_type]
+
+        # Set a range for random values.
+        custom_lb = {}
+        custom_ub = {}
+        # Get user's range if any.
+        if lower_bound:
+            for type_lbs in lower_bound.strip().split(","):
+                type_lb = type_lbs.split(":")
+                assert not (type_lb[0] in custom_lb), "Duplicate types"
+                custom_lb[type_lb[0]] = type_lb[1]
+        if upper_bound:
+            for type_ubs in upper_bound.strip().split(","):
+                type_ub = type_ubs.split(":")
+                assert not (type_ub[0] in custom_ub), "Duplicate types"
+                custom_ub[type_ub[0]] = type_ub[1]
+        DEFAULT_LB.update(custom_lb)
+        DEFAULT_UB.update(custom_ub)
+
+        lb = ub = 0
+        random_element_type = np_elem_type
+        if np.issubdtype(np_elem_type, np.dtype(bool).type):
+            # For some reason, random.uniform with lb/ub to 0/1 resulted in 1 only.
+            lb = int(DEFAULT_LB["bool"])
+            ub = int(DEFAULT_UB["bool"])
+            random_element_type = np.dtype("int32")
+        elif np.issubdtype(np_elem_type, np.uint8):
+            lb = int(DEFAULT_LB["uint8"])
+            ub = int(DEFAULT_UB["uint8"])
+        elif np.issubdtype(np_elem_type, np.uint16):
+            lb = int(DEFAULT_LB["uint16"])
+            ub = int(DEFAULT_UB["uint16"])
+        elif np.issubdtype(np_elem_type, np.uint32):
+            lb = int(DEFAULT_LB["uint32"])
+            ub = int(DEFAULT_UB["uint32"])
+        elif np.issubdtype(np_elem_type, np.uint64):
+            lb = int(DEFAULT_LB["uint64"])
+            ub = int(DEFAULT_UB["uint64"])
+        elif np.issubdtype(np_elem_type, np.int8):
+            lb = int(DEFAULT_LB["int8"])
+            ub = int(DEFAULT_UB["int8"])
+        elif np.issubdtype(np_elem_type, np.int16):
+            lb = int(DEFAULT_LB["int16"])
+            ub = int(DEFAULT_UB["int16"])
+        elif np.issubdtype(np_elem_type, np.int32):
+            lb = int(DEFAULT_LB["int32"])
+            ub = int(DEFAULT_UB["int32"])
+        elif np.issubdtype(np_elem_type, np.int64):
+            lb = int(DEFAULT_LB["int64"])
+            ub = int(DEFAULT_UB["int64"])
+        elif np.issubdtype(np_elem_type, np.float64):
+            lb = float(DEFAULT_LB["float64"])
+            ub = float(DEFAULT_UB["float64"])
+        elif np.issubdtype(np_elem_type, np.float32):
+            lb = float(DEFAULT_LB["float32"])
+            ub = float(DEFAULT_UB["float32"])
+        elif np.issubdtype(np_elem_type, np.float16):
+            lb = float(DEFAULT_LB["float16"])
+            ub = float(DEFAULT_UB["float16"])
+        elif np.issubdtype(np_elem_type, np.str_):
+            lb = 0
+            ub = 64
+            random_element_type = np.dtype("int32")
+        else:
+            raise AssertionError("Unsupported element type")
+        rinput = np.random.uniform(lb, ub, explicit_shape).astype(random_element_type)
+        # For boolean, transform range into True/False using greater_equal
+        if np.issubdtype(np_elem_type, np.dtype(bool).type):
+            rinput = np.greater_equal(rinput, [0])
+        elif np.issubdtype(np_elem_type, np.str_):
+            rinput = np.array(rinput, dtype=np.str_)
+            # rinput = np.array(["ab", "defg"], dtype=np.str_)
+            rinput = np.array(rinput, dtype=object)
+        print(
+            "  - {} input's shape {}, element type {}.".format(
+                ordinal(i + 1), rinput.shape, np_elem_type
+            ),
+            "Value ranges [{}, {}]".format(lb, ub),
+        )
+        inputs.append(rinput)
+    print("  done.\n")
+    return inputs
+
+
+def verify_outs(actual_outs, ref_outs, atol, rtol):
+    total_elements = 0
+    mismatched_elements = 0
+    for index, actual_val in np.ndenumerate(actual_outs):
+        total_elements += 1
+        ref_val = ref_outs[index]
+        if np.issubdtype(actual_outs.dtype, np.dtype(bool).type):
+            if ref_val == actual_val:
+                continue
+        else:
+            # Use equation atol + rtol * abs(desired), that is used in assert_allclose.
+            diff = float(atol) + float(rtol) * abs(ref_val)
+            if abs(actual_val - ref_val) <= diff:
+                continue
+        mismatched_elements += 1
+        print(
+            "  at {}".format(index),
+            "mismatch {} (actual)".format(actual_val),
+            "vs {} (reference)".format(ref_val),
+        )
+    if mismatched_elements == 0:
+        print("  correct.\n")
+    else:
+        raise AssertionError(
+            "  got mismatched elements {}/{}, abort.\n".format(
+                mismatched_elements, total_elements
+            )
+        )
+
+
+def data_without_top_bottom_quartile(data, percent):
+    data = np.array(sorted(data))
+    trim = int(percent * data.size / 100.0)
+    if trim == 0 or data.size - 2 * trim < 1:
+        # Want at least one element, return as is.
+        return data
+    return data[trim:-trim]
+
+
+################################################################################
+# Inference Session implementing RunONNXModel.
+#
+# Constructor: fetch the model and compile if needed, save model if requested.
+# process_inputs: initialize the inputs, which can come from various sources.
+# run_inference: run one inference using the inputs set in process_inputs.
+# process_output: verify values generated in run, save outputs,...
+# process_perf_results: compute and print performance data.
+#
+# run_performance_test: process inputs, perform several inferences (warmup and perf),
+#   process performance results and validate outputs,
+
+
+class InferenceSession:
+    """
+    Init the class by loading / compiling and build an execution session.
+    model_file: the file name of the model, possibly needing compilation.
+    options: parsed and added into args.
+    """
+
+    # Init load the model or compile it, and build an execution session.
+    # For init, either options have been parsed because this file is executed
+    # as a main, or a model_file is expected as parameter to init.
+    # In either case, args will be parsed and thus be available.
+    #
+    # Object variables are:
+    #  default_model_name
+    #  model_dir
+    #  session
+    #  inputs (definition of inputs delayed to process_inputs).
+    #  input_names, output_names
+    #  temp_dir
+
+    def __init__(self, model_file=None, **kwargs):
+        global args
+
+        # Get options passes, if any.
+        options = kwargs["options"] if "options" in kwargs.keys() else ""
+        # Add model file to options, if given.
+        if model_file:
+            if model_file.endswith(".onnx") or model_file.endswith(".mlir"):
+                options += " --model=" + model_file
+            else:
+                options += " --load-model=" + model_file
+        # Parse options
+        if options:
+            args = parser.parse_args(shlex.split(options))
+        # Default model name that will be used for the compiled model.
+        # e.g. model.so, model.constants.bin, ...
+        self.default_model_name = args.default_model_name
+
+        # Handle cache_model.
+        if args.cache_model:
+            shared_lib_path = args.cache_model + f"/{self.default_model_name}.so"
+            if not os.path.exists(shared_lib_path):
+                print(
+                    'Cached compiled model not found in "'
+                    + args.cache_model
+                    + '": save model this run.'
+                )
+                args.save_model = args.cache_model
+            else:
+                print(
+                    'Cached compiled model found in "'
+                    + args.cache_model
+                    + '": load model this run.'
+                )
+                args.load_model = args.cache_model
+            args.cache_model = None
+
+        # Load the onnx model.
+        if args.model and args.model.endswith(".onnx"):
+            model = onnx.load(args.model)
+            # Get names of all intermediate tensors and modify model such that each of
+            # them will be an output of the model. If using onnxruntime for
+            # verification, we can then verify every operation output.
+            output_names = [o.name for o in model.graph.output]
+            output_names = list(OrderedDict.fromkeys(output_names))
+            if args.verify and args.verify == "onnxruntime" and args.verify_all_ops:
+                print("Extending the onnx model to check every node output ...\n")
+                output_names = sum(
+                    [[n for n in node.output if n != ""] for node in model.graph.node],
+                    [],
+                )
+                output_names = list(OrderedDict.fromkeys(output_names))
+                model = extend_model_output(model, output_names)
+
+                # Save the modified onnx file of the model if required.
+                if args.save_onnx:
+                    print("Saving modified onnx model to ", args.save_onnx, "\n")
+                    onnx.save(model, args.save_onnx)
+
+        # If a shared library is given, use it without compiling the ONNX model.
+        # Otherwise, compile the ONNX model.
+        if args.load_model:
+            self.model_dir = args.load_model
+        else:
+            # Compile the ONNX model.
+            self.temp_dir = tempfile.TemporaryDirectory()
+            print("Temporary directory has been created at {}\n".format(self.temp_dir))
+            print("Compiling the model ...")
+            self.model_dir = self.temp_dir.name
+            # Prepare input and output paths.
+            output_path = os.path.join(self.model_dir, self.default_model_name)
+            if args.model.endswith(".onnx"):
+                if args.verify and args.verify == "onnxruntime" and args.verify_all_ops:
+                    input_model_path = os.path.join(
+                        self.model_dir, f"{self.default_model_name}.onnx"
+                    )
+                    onnx.save(model, input_model_path)
+                else:
+                    input_model_path = args.model
+            elif args.model.endswith(".mlir") or args.model.endswith(".onnxtext"):
+                input_model_path = args.model
+            else:
+                print(
+                    "Invalid input model path. Must end with .onnx or .mlir or .onnxtext"
+                )
+                exit(1)
+
+            # Prepare compiler arguments.
+            command_str = [ONNX_MLIR]
+            if args.compile_args:
+                command_str += args.compile_args.split()
+            command_str += [input_model_path]
+            command_str += ["-o", output_path]
+
+            # Compile the model.
+            start = time.perf_counter()
+            ok, msg = execute_commands(command_str)
+            # Dump the compilation log into a file.
+            if args.log_to_file:
+                log_file = (
+                    args.log_to_file
+                    if args.log_to_file.startswith("/")
+                    else os.path.join(os.getcwd(), args.log_to_file)
+                )
+                print("  Compilation log is dumped into {}".format(log_file))
+                with open(log_file, "w") as f:
+                    f.write(msg)
+            if not ok:
+                print(msg)
+                exit(1)
+            end = time.perf_counter()
+            print("  took ", end - start, " seconds.\n")
+
+            # Save the following information:
+            # - .so file,
+            # - .constants.bin file, and
+            # - compilation.log containing the compilation output.
+            if args.save_model:
+                if not os.path.exists(args.save_model):
+                    os.makedirs(args.save_model)
+                if not os.path.isdir(args.save_model):
+                    print("Path to --save-model is not a folder")
+                    exit(0)
+                # .so file.
+                shared_lib_path = self.model_dir + f"/{self.default_model_name}.so"
+                if os.path.exists(shared_lib_path):
+                    print("Saving the shared library to", args.save_model)
+                    shutil.copy2(shared_lib_path, args.save_model)
+                # .constants.bin file.
+                constants_file_path = os.path.join(
+                    self.model_dir, f"{self.default_model_name}.constants.bin"
+                )
+                if os.path.exists(constants_file_path):
+                    print("Saving the constants file to", args.save_model, "\n")
+                    shutil.copy2(constants_file_path, args.save_model)
+                # Compilation log.
+                log_file_path = os.path.join(args.save_model, "compile.log")
+                with open(log_file_path, "w") as f:
+                    print("Saving the compilation log to", args.save_model, "\n")
+                    f.write(msg)
+
+            # Exit if only compiling the model.
+            if args.compile_only:
+                exit(0)
+
+        # Use the generated shared library to create an execution session.
+        start = time.perf_counter()
+        shared_lib_path = self.model_dir + f"/{self.default_model_name}.so"
+        if not os.path.exists(shared_lib_path):
+            print(f"Input model {shared_lib_path} does not exist")
+            exit(0)
+        print("Loading the compiled model ...")
+        if args.load_model:
+            session = OMExecutionSession(shared_lib_path, tag="None")
+        else:
+            session = OMExecutionSession(shared_lib_path)
+        end = time.perf_counter()
+        print("  took ", end - start, " seconds.\n")
+        self.session = session
+
+        # Additional model info.
+        self.inputs = []
+        input_signature = self.session.input_signature()
+        output_signature = self.session.output_signature()
+        self.input_names = get_names_in_signature(input_signature)
+        self.output_names = get_names_in_signature(output_signature)
+        if args.print_signatures:
+            print("Model's input signature: ", input_signature.strip())
+            print("Model's output signature: ", output_signature.strip())
+
+        # Let onnx-mlir know where to find the constants file.
+        os.environ["OM_CONSTANT_PATH"] = self.model_dir
+
+    """
+    process_inputs: define the model inputs for the model and store them in self.inputs.
+    Print input if requested.
+    """
+
+    def process_inputs(self, input_feed=None):
+        # Define inputs.
+        self.inputs = []
+        if input_feed:
+            # Get input from input_feed.
+            if isinstance(input_feed, dict):
+                for name in self.input_names:
+                    if name in input_feed:
+                        self.inputs.append(input_feed[name])
+                    else:
+                        print("input name given: ", input_feed.keys())
+                        print("input name expected by model: ", self.input_names)
+                        print("do not match")
+                        exit(1)
+                # Since Python guarantees the order of values in a dictionary,
+                # the name check could be ignored as follows:
+                # inputs = list(input_feed.values())
+            else:
+                self.inputs = input_feed
+        elif args.load_ref:
+            # Get input from reference file.
+            self.inputs = read_input_from_refs(
+                len(self.input_names), args.load_ref, is_load_ref=True
+            )
+        elif args.load_ref_from_numpy:
+            # Get input from numpy.
+            self.inputs = read_input_from_refs(
+                len(self.input_names), args.load_ref_from_numpy, is_load_ref=False
+            )
+        elif args.inputs_from_arrays:
+            # Get input from array.
+            self.inputs = args.inputs_from_arrays
+        else:
+            self.inputs = generate_random_input(
+                self.session.input_signature(),
+                args.shape_info,
+                args.seed,
+                args.lower_bound,
+                args.upper_bound,
+            )
+
+        # Print the input if required.
+        if args.print_input:
+            for i, inp in enumerate(self.inputs):
+                print(
+                    "The {} input {}:[{}x{}] is: \n {} \n".format(
+                        ordinal(i + 1),
+                        self.input_names[i],
+                        "x".join([str(i) for i in inp.shape]),
+                        inp.dtype,
+                        inp,
+                    )
+                )
+
+    """
+    Perform one inference without any timing.
+    """
+
+    def run_inference(self):
+        return self.session.run(self.inputs)
+
+    """
+    When requested outputs are printed, verified, and/or saved.
+    """
+
+    def process_outputs(self, outs):
+        # Print the output if required.
+        if args.print_output:
+            for i, out in enumerate(outs):
+                print(
+                    "The {} output {}:[{}x{}] is: \n {} \n".format(
+                        ordinal(i + 1),
+                        self.output_names[i],
+                        "x".join([str(i) for i in out.shape]),
+                        out.dtype,
+                        out,
+                    )
+                )
+
+        # Store the input and output if required.
+        if args.save_ref:
+            load_ref = args.save_ref
+            if not os.path.exists(load_ref):
+                os.mkdir(load_ref)
+            for i in range(len(self.inputs)):
+                tensor = numpy_helper.from_array(self.inputs[i])
+                tensor_path = os.path.join(load_ref, "input_{}.pb".format(i))
+                with open(tensor_path, "wb") as f:
+                    f.write(tensor.SerializeToString())
+            for i in range(len(outs)):
+                tensor = numpy_helper.from_array(outs[i])
+                tensor_path = os.path.join(load_ref, "output_{}.pb".format(i))
+                with open(tensor_path, "wb") as f:
+                    f.write(tensor.SerializeToString())
+
+        # Verify the output if required.
+        if args.verify:
+            ref_outs = []
+            if args.verify.lower() == "onnxruntime":
+                input_model_path = args.model
+                # Reference backend by using onnxruntime.
+                import onnxruntime
+
+                input_feed = dict(zip(self.input_names, self.inputs))
+                print("Running inference using onnxruntime ...")
+                start = time.perf_counter()
+                ref_session = onnxruntime.InferenceSession(input_model_path)
+                ref_outs = ref_session.run(self.output_names, input_feed)
+                end = time.perf_counter()
+                print("  took ", end - start, " seconds.\n")
+            elif args.verify.lower() == "ref":
+                # Reference output available in protobuf.
+                if args.load_ref:
+                    ref_outs = read_output_from_refs(
+                        len(self.output_names), args.load_ref, is_load_ref=True
+                    )
+                elif args.load_ref_from_numpy:
+                    ref_outs = read_output_from_refs(
+                        len(self.output_names),
+                        args.load_ref_from_numpy,
+                        is_load_ref=False,
+                    )
+            else:
+                print("Invalid verify option")
+                exit(1)
+
+            # Verify using softmax first.
+            if args.verify_with_softmax is not None:
+                axis = int(args.verify_with_softmax)
+                for i, name in enumerate(self.output_names):
+                    print(
+                        "Verifying using softmax along with "
+                        "axis {}".format(args.verify_with_softmax),
+                        "for output {}:{}".format(name, list(outs[i].shape)),
+                        "using atol={}, rtol={} ...".format(args.atol, args.rtol),
+                    )
+                    softmax_outs = softmax(outs[i], axis)
+                    softmax_ref_outs = softmax(ref_outs[i], axis)
+                    verify_outs(softmax_outs, softmax_ref_outs, args.atol, args.rtol)
+
+            # For each output tensor, compare every value.
+            if args.verify_every_value:
+                for i, name in enumerate(self.output_names):
+                    print(
+                        "Verifying value of {}:{}".format(name, list(outs[i].shape)),
+                        "using atol={}, rtol={} ...".format(args.atol, args.rtol),
+                    )
+                    verify_outs(outs[i], ref_outs[i], args.atol, args.rtol)
+
+    """
+    Perform a short analysis of time spent in the model.
+    """
+
+    def process_perf_results(self, perf_results):
+        # Print statistics info, e.g., min/max/stddev inference time.
+        if args.n_iteration > 1:
+            print(
+                "  Statistics 1 (excluding warmup),"
+                " min, {:.6e}, max, {:.6e}, mean, {:.6e}, stdev, {:.6e}".format(
+                    np.min(perf_results),
+                    np.max(perf_results),
+                    np.mean(perf_results),
+                    np.std(perf_results, dtype=np.float64),
+                )
+            )
+            t_perf_results = data_without_top_bottom_quartile(perf_results, 25)
+            print(
+                "  Statistics 2 (no warmup/quart.),"
+                " min, {:.6e}, max, {:.6e}, mean, {:.6e}, stdev, {:.6e}".format(
+                    np.min(t_perf_results),
+                    np.max(t_perf_results),
+                    np.mean(t_perf_results),
+                    np.std(t_perf_results, dtype=np.float64),
+                )
+            )
+
+    """
+    From onnxruntime API:
+
+    run_performance_test(output_names, input_feed)
+    Compute the predictions.
+
+    PARAMETERS:
+    output_names – name of the outputs (optional)
+    input_feed – dictionary { input_name: input_value }
+    RETURNS:
+    list of results, every result is either a numpy array, a sparse tensor, or
+    a list or a dictionary.
+    
+    For onnxmlir, the run_options is ignored. If 'input_feed' is None, the
+    input could be randomly generated or read from file, as args specified.
+    In future, add '--shape-info' here. Better than in InferenceSession to
+    allow different shape from run to run. 
+    """
+
+    def run_performance_test(self, output_name=None, input_feed=None, **kwargs):
+        # Process inputs, saved in self.inputs.
+        self.process_inputs(input_feed)
+        # Running inference.
+        print("Running inference ...")
+        for i in range(args.warmup):
+            start = time.perf_counter()
+            outs = self.run_inference()  # Using inputs from self.inputs.
+            end = time.perf_counter()
+            print("  {} warmup: {} seconds".format(ordinal(i + 1), end - start))
+
+        perf_results = []
+        for i in range(args.n_iteration):
+            start = time.perf_counter()
+            outs = self.run_inference()  # Using inputs from self.inputs.
+            end = time.perf_counter()
+            elapsed = end - start
+            perf_results += [elapsed]
+            print("  {} iteration, {}, seconds".format(ordinal(i + 1), elapsed))
+
+        # Print performance results and verify output.
+        self.process_perf_results(perf_results)
+        self.process_outputs(outs)
+        if output_name:
+            res = {output_name[i]: outs[i] for i in range(len(outs))}
+            return res
+        else:
+            return outs
+
+
+################################################################################
+# Standalone driver
+
+
+def main():
+    # In main mode, parse the args here.
+    global args
+    args = parser.parse_args()
+    if not (args.model or args.load_model):
+        print("error: no input model, use argument --model and/or --load-model.")
+        print(parser.format_usage())
+        exit(1)
+
+    # Create inference session and perform a performance run test, which load,
+    # compute, and possibly verify data.
+    session = InferenceSession()
+    return session.run_performance_test()
+
+
+if __name__ == "__main__":
+    main()
diff --git a/src/matmul_relu_matmul_fashion_mnist/fasionmnist_mlp_params.pkl b/src/matmul_relu_matmul_fashion_mnist/fasionmnist_mlp_params.pkl
new file mode 100644
index 0000000..eefcfe8
Binary files /dev/null and b/src/matmul_relu_matmul_fashion_mnist/fasionmnist_mlp_params.pkl differ
diff --git a/src/matmul_relu_matmul_fashion_mnist/onnx-mlir_test.py b/src/matmul_relu_matmul_fashion_mnist/onnx-mlir_test.py
new file mode 100644
index 0000000..7c9eece
--- /dev/null
+++ b/src/matmul_relu_matmul_fashion_mnist/onnx-mlir_test.py
@@ -0,0 +1,544 @@
+##############################################
+# IMPORT LIBRARIES ###########################
+##############################################
+
+"""
+Libraries and packages used in this script and Versions Info for tools, libraries, and packages used in this script.
+"""
+import numpy as np          # Version: 1.26.4 (latest as of 2025-04-14)
+import time
+# import pickle as pkl # Not used directly if loading ONNX
+import torch                # Version: 2.2.2 (latest as of 2025-04-14)
+import torchvision          # Version: 0.17.2 (latest as of 2025-04-14)
+# import torch.nn as nn # Not needed if only loading ONNX
+import os
+import subprocess           # Version: Python Standard Library
+import sys                  # Version: Python Standard Library
+# import tempfile # Not used if INFERENCE_TEMP_DIR_BASE is used
+import shutil               # Version: Python Standard Library
+import importlib.util       # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import json                 # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import argparse             # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import signal               # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import shlex                # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import tempfile             # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import onnx                 # Version: 1.16.0 (latest as of 2025-04-14, required by RunONNXModel.py)
+
+
+###############################################
+# CONSTANTS & PARAMETERS ######################
+###############################################
+
+"""
+Constants and parameters used in this script.
+"""
+# Paths
+MODEL_PATH = "mnist_model_cpu_optimized.onnx" # Path to the ONNX model to compile
+# COMPILED_MODEL_BASE = "mnist_model_cpu_initial" # No longer needed, will use "model" inside a dir
+COMPILED_MODEL_DIR = "compiled_model_onnx_mlir" # Directory to store the compiled model.so and model.constants.bin
+DATA_ROOT = "data"
+ONNX_MLIR_PATH = "onnx-mlir" # Assumes onnx-mlir executable is in PATH or provide full path
+RUN_ONNX_MODEL_SCRIPT_PATH = "/workdir/onnx-mlir/utils/RunONNXModel.py" # Absolute path to the script
+# Directory to store temporary files for each inference run, created in the script's CWD
+INFERENCE_TEMP_DIR_BASE = "onnx_mlir_inference_runs"
+
+# Dataset configuration
+BATCH_SIZE = 1
+FEATURE_DIM = 784  # 28x28 flattened
+IMG_SHAPE = (1, 28, 28)
+NUM_CLASSES = 10
+
+# Benchmark configuration
+NUM_ITERATIONS = 100 # Reduced for potentially slower subprocess calls
+WARMUP_ITERATIONS = 20  # Number of warmup iterations
+
+# Class names for interpretation
+CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
+               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
+
+print("Starting ONNX-MLIR benchmark script (using RunONNXModel.py via subprocess)")
+
+##############################################
+# FUNCTION DEFINITIONS #######################
+##############################################
+
+def find_executable(name, default_path):
+    """
+    Finds an executable by checking the default path and then the system PATH.
+
+    Args:
+        name (str): The name of the executable.
+        default_path (str): The default path to check first.
+
+    Returns:
+        str: The full path to the executable if found. Exits if not found.
+    """
+    if os.path.exists(default_path) and os.access(default_path, os.X_OK):
+        print(f"✅ Found '{name}' at specified path: {default_path}")
+        return os.path.abspath(default_path)
+
+    found_path = shutil.which(name)
+    if found_path:
+        print(f"✅ Found '{name}' in system PATH: {found_path}")
+        return os.path.abspath(found_path)
+
+    print(f"❌ Error: '{name}' not found at '{default_path}' or in system PATH.")
+    sys.exit(1) # Exit if executable not found
+
+def find_script(name, default_path):
+    """
+    Finds a script file.
+
+    Args:
+        name (str): The name of the script.
+        default_path (str): The path to check for the script.
+
+    Returns:
+        str: The full path to the script if found. Exits if not found.
+    """
+    absolute_path = os.path.abspath(default_path)
+    if os.path.exists(absolute_path) and os.path.isfile(absolute_path):
+         print(f"✅ Found script '{name}' at specified path: {absolute_path}")
+         return absolute_path
+    print(f"❌ Error: Script '{name}' not found at '{default_path}' (abs: {absolute_path}). Please provide the correct path.")
+    sys.exit(1) # Exit if script not found
+
+def compile_onnx_model(onnx_model_path, output_dir, onnx_mlir_exec_path):
+    """
+    Compiles an ONNX model using onnx-mlir, saving the output library
+    as 'model.so' inside the specified output directory.
+
+    Args:
+        onnx_model_path (str): Path to the input ONNX model file.
+        output_dir (str): Path to the directory where 'model.so' and
+                          'model.constants.bin' should be saved.
+        onnx_mlir_exec_path (str): Path to the onnx-mlir executable.
+
+    Returns:
+        tuple(bool, str or None): A tuple containing:
+            - bool: True if compilation was successful, False otherwise.
+            - str or None: The absolute path to the output directory if successful,
+                           otherwise None.
+    """
+    print(f"Compiling ONNX model '{onnx_model_path}' with onnx-mlir...")
+    absolute_onnx_model_path = os.path.abspath(onnx_model_path)
+    absolute_output_dir = os.path.abspath(output_dir)
+
+    if not os.path.exists(absolute_onnx_model_path):
+        print(f"❌ Error: Input ONNX model not found at '{absolute_onnx_model_path}'")
+        return False, None
+
+    # Ensure output directory exists and is clean
+    if os.path.exists(absolute_output_dir):
+        print(f"Removing existing compilation output directory: {absolute_output_dir}")
+        shutil.rmtree(absolute_output_dir)
+    os.makedirs(absolute_output_dir, exist_ok=True)
+    print(f"Created compilation output directory: {absolute_output_dir}")
+
+    # Define the base name for output files *inside* the output directory
+    output_base_name = os.path.join(absolute_output_dir, "model")
+    expected_lib_path = output_base_name + ".so"
+
+    compile_command = [
+        onnx_mlir_exec_path,
+        "--EmitLib",
+        absolute_onnx_model_path, # Use absolute path for input model
+        "-o", output_base_name    # Use absolute path/basename for output
+    ]
+    print(f"Running command: {' '.join(compile_command)}")
+
+    try:
+        # Run the compilation
+        result = subprocess.run(compile_command, check=True, capture_output=True, text=True, timeout=300) # Use default CWD
+        print("✅ ONNX-MLIR compilation successful.")
+        # print("Compiler Output:\n", result.stdout) # Optional: print compiler output
+        # print("Compiler Stderr:\n", result.stderr) # Optional: print compiler stderr
+
+        # Verify the expected output library exists
+        if not os.path.exists(expected_lib_path):
+             print(f"❌ Error: Compiled library '{expected_lib_path}' not found after compilation.")
+             print("Compiler Stdout:\n", result.stdout)
+             print("Compiler Stderr:\n", result.stderr)
+             return False, None
+
+        # Return the absolute path to the output DIRECTORY
+        return True, absolute_output_dir
+    except FileNotFoundError:
+        print(f"❌ Error: '{onnx_mlir_exec_path}' command not found.")
+        return False, None
+    except subprocess.TimeoutExpired:
+        print(f"❌ Error: ONNX-MLIR compilation timed out.")
+        return False, None
+    except subprocess.CalledProcessError as e:
+        print(f"❌ Error: ONNX-MLIR compilation failed with exit code {e.returncode}.")
+        print("Compiler Stderr:\n", e.stderr)
+        print("Compiler Stdout:\n", e.stdout)
+        return False, None
+
+def run_inference_with_script(run_script_path, compiled_model_dir_path, input_data_list, output_dir):
+    """
+    Runs inference by calling the RunONNXModel.py script via subprocess.
+    Assumes compiled model is 'model.so' inside compiled_model_dir_path.
+    Saves input numpy arrays and a loader script to output_dir, then executes RunONNXModel.py.
+
+    Args:
+        run_script_path (str): Absolute path to the RunONNXModel.py script.
+        compiled_model_dir_path (str): Absolute path to the directory containing the
+                                       compiled 'model.so'.
+        input_data_list (list): A list of NumPy arrays, one for each model input.
+        output_dir (str): The directory (relative to CWD) to use for this specific
+                          run's input/output files.
+
+    Returns:
+        tuple(list or None, float): A tuple containing:
+            - list or None: A list of NumPy arrays representing the model outputs if successful, otherwise None.
+            - float: The time taken for the subprocess call (includes overhead).
+    """
+    loaded_outputs = None
+
+    # Ensure paths are absolute
+    absolute_run_script_path = os.path.abspath(run_script_path)
+    absolute_compiled_model_dir = os.path.abspath(compiled_model_dir_path)
+    absolute_output_dir = os.path.abspath(output_dir) # Dir for this run's I/O
+
+    # --- Basic Checks ---
+    if not os.path.exists(absolute_run_script_path):
+        print(f"❌ Error: Run script not found at: {absolute_run_script_path}")
+        return None, 0
+    if not os.path.isdir(absolute_compiled_model_dir):
+        print(f"❌ Error: Compiled model directory not found or not a directory: {absolute_compiled_model_dir}")
+        return None, 0
+    expected_model_so_path = os.path.join(absolute_compiled_model_dir, "model.so")
+    if not os.path.exists(expected_model_so_path):
+        print(f"❌ Error: Expected compiled model 'model.so' not found in: {absolute_compiled_model_dir}")
+        return None, 0
+    # --- End Basic Checks ---
+
+    # Create the output directory for this specific run
+    os.makedirs(absolute_output_dir, exist_ok=True)
+
+    # --- Prepare Input Files and Loader Script ---
+    loader_script_path = os.path.join(absolute_output_dir, "_loader.py")
+    loader_script_content = """
+# Generated loader script for RunONNXModel.py --load-ref-from-numpy
+import numpy as np
+import os
+
+inputs = []
+i = 0
+while True:
+    input_npy_path = os.path.join(os.path.dirname(__file__), f"input_{i}.npy")
+    if os.path.exists(input_npy_path):
+        try:
+            inputs.append(np.load(input_npy_path))
+            i += 1
+        except Exception as e:
+            print(f"Error loading {input_npy_path}: {e}")
+            # Decide how to handle load errors, e.g., raise or break
+            break
+    else:
+        break
+
+# Optional: Define outputs = [] if needed for verification later
+# outputs = []
+"""
+    try:
+        # Save the input numpy arrays
+        for i, data in enumerate(input_data_list):
+            input_path_abs = os.path.join(absolute_output_dir, f"input_{i}.npy")
+            np.save(input_path_abs, data)
+            # print(f"   Saved input {i} to {input_path_abs}") # Optional debug print
+
+        # Save the loader script
+        with open(loader_script_path, "w") as f:
+            f.write(loader_script_content)
+        # print(f"   Saved loader script to {loader_script_path}") # Optional debug print
+
+    except Exception as e:
+        print(f"❌ Error preparing input files/loader script in {absolute_output_dir}: {e}")
+        return None, 0
+    # --- End Input Preparation ---
+
+
+    # Construct the command for RunONNXModel.py
+    run_command = [
+        sys.executable,
+        absolute_run_script_path,
+        "--load-model", absolute_compiled_model_dir, # Pass the DIRECTORY containing model.so
+        "--save-ref", absolute_output_dir,           # Save outputs to this run's dir
+        "--load-ref-from-numpy", loader_script_path  # Use the loader script
+    ]
+
+    print(f"Running inference command: {' '.join(run_command)}")
+    # Set the working directory for the subprocess to the run's output directory.
+    # This is where --save-ref will save files and where _loader.py will look for input_*.npy
+    print(f"  Working directory for subprocess: {absolute_output_dir}")
+    # Run the script with cwd set to the specified output directory
+    env = os.environ.copy()
+    if "ONNX_MLIR_HOME" not in env:
+            print("⚠️ Warning: ONNX_MLIR_HOME environment variable not found. RunONNXModel.py might fail.")
+            # Consider adding ONNX_MLIR_HOME if known and missing? For now, just warn.
+    start_time = time.time()
+    result = subprocess.run(run_command, check=True, capture_output=True, text=True, cwd=absolute_output_dir, timeout=60, env=env)
+    end_time = time.time()
+    # print("RunONNXModel Output:\n", result.stdout) # Often empty on success
+    # print("RunONNXModel Stderr:\n", result.stderr) # Check stderr for potential info
+
+    # --- Load output files from the specified output directory ---
+    loaded_outputs = []
+    i = 0
+    while True:
+        # Look for output files in the absolute_output_dir (where --save-ref saved them)
+        # RunONNXModel.py saves outputs as output_0.pb, output_1.pb etc. with --save-ref
+        output_path = os.path.join(absolute_output_dir, f"output_{i}.pb") # <-- Changed extension to .pb
+
+        if os.path.exists(output_path):
+            try:
+                # Load the protobuf tensor
+                output_ts = onnx.TensorProto()
+                with open(output_path, "rb") as f:
+                    output_ts.ParseFromString(f.read())
+                # Convert to numpy array
+                output_np = onnx.numpy_helper.to_array(output_ts)
+                loaded_outputs.append(output_np)
+                print(f"   ✅ Loaded output file: {output_path}")
+                i += 1
+            except Exception as load_err: # Use more specific exception if possible
+                print(f"   ⚠️ Error loading output file {output_path}: {load_err}")
+                loaded_outputs = None # Mark as failed if loading fails
+                break
+        else:
+            # Stop if output_i.pb is not found.
+            # If i is 0, it means no output files (output_0.pb) were found at all.
+            break # Exit the while loop
+
+    if not loaded_outputs:
+            # This warning now means the script ran successfully but didn't produce
+            # output_0.pb in the specified directory.
+            print(f"⚠️ Warning: No valid output files (e.g., output_0.pb) found or loaded from directory: {absolute_output_dir}")
+            # Print stdout/stderr from the script to help debug why it didn't save output
+            print("RunONNXModel stdout:\n", result.stdout)
+            print("RunONNXModel stderr:\n", result.stderr)
+            # exit message and exit program
+            print("❌ Exiting due to missing output files.")
+            exit(1)
+
+    
+    elapsed_time = end_time - start_time
+    return loaded_outputs, elapsed_time
+
+##################################################################################
+# MAIN PROGRAM ###################################################################
+##################################################################################
+
+"""
+Main execution block for the ONNX-MLIR benchmark script.
+"""
+
+if __name__ == "__main__":
+
+    #########################################
+    # SETUP OUTPUT DIRECTORY ################
+    #########################################
+    # Create a base directory for all inference runs in the current working directory
+    abs_inference_base_dir = os.path.abspath(INFERENCE_TEMP_DIR_BASE)
+    if os.path.exists(abs_inference_base_dir):
+        print(f"Removing existing base inference output directory: {abs_inference_base_dir}")
+        shutil.rmtree(abs_inference_base_dir)
+    os.makedirs(abs_inference_base_dir, exist_ok=True)
+    print(f"Created base directory for inference outputs: {abs_inference_base_dir}")
+
+
+    #########################################
+    # CHECK PREREQUISITES ###################
+    #########################################
+    print("Checking prerequisites...")
+    onnx_mlir_executable = find_executable("onnx-mlir", ONNX_MLIR_PATH)
+    absolute_run_script = find_script("RunONNXModel.py", RUN_ONNX_MODEL_SCRIPT_PATH)
+
+    # No need to check absolute_run_script again, find_script exits if not found
+    if not os.path.exists(MODEL_PATH):
+        print(f"❌ Error: Input ONNX model not found at '{MODEL_PATH}'. Please set the MODEL_PATH constant.")
+        sys.exit(1)
+    print("✅ Prerequisites check passed.")
+
+    #########################################
+    # COMPILE MODEL #########################
+    #########################################
+    # compile_onnx_model now returns the absolute path to the *directory* containing model.so
+    compilation_success, absolute_compiled_model_dir = compile_onnx_model(
+        MODEL_PATH,
+        COMPILED_MODEL_DIR, # Pass the desired output directory name
+        onnx_mlir_executable
+    )
+    if not compilation_success:
+        print("❌ Exiting due to compilation failure.")
+        sys.exit(1)
+    print(f"✅ Model compiled successfully. Library 'model.so' is in: {absolute_compiled_model_dir}")
+
+
+    #########################################
+    # DATASET LOADING #######################
+    #########################################
+    print("Loading FashionMNIST dataset...")
+    test_data = torchvision.datasets.FashionMNIST(
+        root=DATA_ROOT,
+        train=False,
+        download=True,
+        transform=torchvision.transforms.ToTensor()
+    )
+    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False) # Use shuffle=False for consistency
+    print("✅ Dataset loaded.")
+
+    #########################################
+    # BENCHMARK PREPARATION #################
+    #########################################
+    num_samples_to_load = min(len(test_data), NUM_ITERATIONS + WARMUP_ITERATIONS)
+    print(f"Preloading {num_samples_to_load} test samples for benchmarking...")
+    test_samples = []
+    test_loader_iter = iter(test_loader)
+
+    for i in range(num_samples_to_load):
+        try:
+            img, label = next(test_loader_iter)
+        except StopIteration:
+            print("Warning: Reached end of dataset during preloading.")
+            break # Stop preloading if dataset ends early
+
+        # Prepare input as expected by the model (flattened float32 numpy array)
+        img_np = img.reshape(BATCH_SIZE, FEATURE_DIM).numpy().astype(np.float32)
+        label_np = label.numpy() # Keep label as numpy scalar/array
+        test_samples.append((img_np, label_np))
+
+        if i == 0:
+            print(f"Sample input numpy shape: {img_np.shape}, dtype: {img_np.dtype}")
+
+    actual_warmup_iters = min(WARMUP_ITERATIONS, len(test_samples))
+    actual_bench_iters = min(NUM_ITERATIONS, len(test_samples) - actual_warmup_iters)
+
+    if len(test_samples) < WARMUP_ITERATIONS + NUM_ITERATIONS:
+         print(f"⚠️ Warning: Loaded only {len(test_samples)} samples, less than requested {WARMUP_ITERATIONS + NUM_ITERATIONS}.")
+         print(f"Adjusting: Warmup={actual_warmup_iters}, Benchmark={actual_bench_iters}")
+
+    if actual_bench_iters <= 0 and actual_warmup_iters <=0:
+        print("❌ Error: No samples loaded for benchmarking or warmup. Exiting.")
+        sys.exit(1)
+
+    print(f"✅ Preloaded {len(test_samples)} samples.")
+
+    #########################################
+    # BENCHMARKING ##########################
+    #########################################
+    print(f"\nRunning benchmark using '{absolute_run_script}'...")
+    total_time = 0
+    correct_predictions = 0
+    inference_failed = False
+
+    # --- Warmup Phase ---
+    if actual_warmup_iters > 0:
+        print(f"Starting {actual_warmup_iters} warmup iterations...")
+        for i in range(actual_warmup_iters):
+            img_np, _ = test_samples[i]
+            # Create a unique subdirectory for this warmup run
+            run_output_dir = os.path.join(abs_inference_base_dir, f"warmup_{i}")
+            # Pass the absolute path to the directory containing model.so
+            outputs, _ = run_inference_with_script( # Ignore time for warmup
+                absolute_run_script, absolute_compiled_model_dir, [img_np], run_output_dir
+            )
+            # Check for failure even in warmup
+            if outputs is None:
+                 print(f"❌ Warmup inference failed for sample {i}. Stopping.")
+                 inference_failed = True
+                 break
+        if not inference_failed:
+            print(f"✅ Completed {actual_warmup_iters} warmup iterations.")
+    else:
+        print("Skipping warmup phase (0 iterations).")
+
+
+    # --- Benchmarking Phase ---
+    if not inference_failed and actual_bench_iters > 0:
+        print(f"Starting {actual_bench_iters} benchmarking iterations...")
+        start_index = actual_warmup_iters
+        for i in range(actual_bench_iters):
+            sample_index = start_index + i
+            img_np, label_np = test_samples[sample_index]
+
+            # Create a unique subdirectory for this benchmark run
+            run_output_dir = os.path.join(abs_inference_base_dir, f"run_{i}")
+
+            # Time the inference script call, passing the absolute path to the model directory
+            outputs, elapsed_time = run_inference_with_script(
+                absolute_run_script, absolute_compiled_model_dir, [img_np], run_output_dir
+            )
+
+            if outputs is None:
+                print(f"❌ Inference failed for sample {sample_index}. Stopping benchmark.")
+                inference_failed = True
+                break # Stop benchmarking loop on failure
+
+            total_time += elapsed_time
+
+            # Process prediction (outside timing loop)
+            # Check if outputs is a non-empty list containing at least one numpy array
+            if outputs and isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], np.ndarray):
+                # Assuming the first output contains the logits
+                try:
+                    pred = np.argmax(outputs[0], axis=1) # Get predicted class index
+                    # Compare prediction with the ground truth label
+                    if pred == label_np: # Assumes label_np is a scalar or 1-element array
+                        correct_predictions += 1
+                except IndexError:
+                     # Handle cases where argmax might fail (e.g., unexpected output shape)
+                     print(f"⚠️ Error processing output for sample {sample_index}. Output shape: {outputs[0].shape}")
+            else:
+                 # This case is hit if run_inference_with_script returned an empty list or None
+                 # (though the None case should have been caught earlier)
+                 print(f"⚠️ No valid output data loaded for sample {sample_index}, cannot check accuracy.")
+
+        if not inference_failed:
+            print(f"✅ Completed {actual_bench_iters} benchmarking iterations.")
+    elif not inference_failed:
+        print("Skipping benchmarking phase (0 iterations).")
+
+
+    #########################################
+    # RESULTS REPORTING #####################
+    #########################################
+    print("\n======= ONNX-MLIR BENCHMARK RESULTS (via RunONNXModel.py) =======")
+    print(f"NOTE: Timing includes subprocess overhead for each inference call.")
+    if inference_failed:
+        print("Benchmark stopped early due to inference failure.")
+    elif actual_bench_iters > 0:
+        avg_time_ms = (total_time / actual_bench_iters) * 1000
+        # Ensure accuracy calculation avoids division by zero if correct_predictions is somehow > 0 but actual_bench_iters is 0
+        accuracy = (correct_predictions / actual_bench_iters) * 100 if actual_bench_iters > 0 else 0
+        # Ensure throughput calculation avoids division by zero
+        throughput = actual_bench_iters / total_time if total_time > 0 else 0
+
+        print(f"Compiled Model Directory: {absolute_compiled_model_dir}")
+        print(f"Inference Script: {absolute_run_script}")
+        print(f"Total Samples Benchmarked: {actual_bench_iters}")
+        print(f"Correct Predictions: {correct_predictions}")
+        print(f"Accuracy: {accuracy:.2f}%")
+        print(f"Total Inference Script Time: {total_time:.3f} s")
+        print(f"Avg. Inference Script Time: {avg_time_ms:.3f} ms/inference")
+        print(f"Throughput: {throughput:.2f} inferences/second (including overhead of running external file)")
+        print(f"Input/Output files stored under: {abs_inference_base_dir}")
+    else:
+        print("No benchmark iterations were successfully run.")
+
+    #########################################
+    # CLEANUP (Optional) ####################
+    #########################################
+    # Keep the INFERENCE_TEMP_DIR_BASE and COMPILED_MODEL_DIR for inspection by default.
+    # Uncomment below to clean up.
+    # print(f"\nCleaning up inference run directory: {abs_inference_base_dir}")
+    # if os.path.exists(abs_inference_base_dir):
+    #     shutil.rmtree(abs_inference_base_dir)
+
+    # print(f"Cleaning up compiled model directory: {absolute_compiled_model_dir}")
+    # if os.path.exists(absolute_compiled_model_dir):
+    #     shutil.rmtree(absolute_compiled_model_dir)
+
+    print("\nScript finished.")
\ No newline at end of file
diff --git a/src/matmul_relu_matmul_fashion_mnist/onnx-mlir_test_v2.py b/src/matmul_relu_matmul_fashion_mnist/onnx-mlir_test_v2.py
new file mode 100644
index 0000000..422b00b
--- /dev/null
+++ b/src/matmul_relu_matmul_fashion_mnist/onnx-mlir_test_v2.py
@@ -0,0 +1,628 @@
+##############################################
+# IMPORT LIBRARIES ###########################
+##############################################
+
+"""
+Libraries and packages used in this script and Versions Info for tools, libraries, and packages used in this script.
+"""
+import numpy as np          # Version: 1.26.4 (latest as of 2025-04-14)
+import time
+# import pickle as pkl # Not used directly if loading ONNX
+import torch                # Version: 2.2.2 (latest as of 2025-04-14)
+import torchvision          # Version: 0.17.2 (latest as of 2025-04-14)
+# import torch.nn as nn # Not needed if only loading ONNX
+import os
+import subprocess           # Version: Python Standard Library
+import sys                  # Version: Python Standard Library
+# import tempfile # Not used if INFERENCE_TEMP_DIR_BASE is used
+import shutil               # Version: Python Standard Library
+import importlib.util       # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import json                 # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import argparse             # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import signal               # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import shlex                # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import tempfile             # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import onnx                 # Version: 1.16.0 (latest as of 2025-04-14, required by RunONNXModel.py)
+
+
+###############################################
+# CONSTANTS & PARAMETERS ######################
+###############################################
+
+"""
+Constants and parameters used in this script.
+"""
+# Paths
+MODEL_PATH = "mnist_model_cpu_optimized.onnx" # Path to the ONNX model to compile
+# COMPILED_MODEL_BASE = "mnist_model_cpu_initial" # No longer needed, will use "model" inside a dir
+COMPILED_MODEL_DIR = "compiled_model_onnx_mlir" # Directory to store the compiled model.so and model.constants.bin
+DATA_ROOT = "data"
+ONNX_MLIR_PATH = "onnx-mlir" # Assumes onnx-mlir executable is in PATH or provide full path
+RUN_ONNX_MODEL_SCRIPT_PATH = "/workdir/onnx-mlir/utils/RunONNXModel.py" # Absolute path to the script
+# Directory to store temporary files for each inference run, created in the script's CWD
+INFERENCE_TEMP_DIR_BASE = "onnx_mlir_inference_runs"
+
+# Dataset configuration
+BATCH_SIZE = 1
+FEATURE_DIM = 784  # 28x28 flattened
+IMG_SHAPE = (1, 28, 28)
+NUM_CLASSES = 10
+
+# Benchmark configuration
+NUM_ITERATIONS = 100 # Reduced for potentially slower subprocess calls
+WARMUP_ITERATIONS = 20  # Number of warmup iterations
+
+# Class names for interpretation
+CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
+               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
+
+print("Starting ONNX-MLIR benchmark script (using RunONNXModel.py via subprocess)")
+
+##############################################
+# FUNCTION DEFINITIONS #######################
+##############################################
+
+def find_executable(name, default_path):
+    """
+    Finds an executable by checking the default path and then the system PATH.
+
+    Args:
+        name (str): The name of the executable.
+        default_path (str): The default path to check first.
+
+    Returns:
+        str: The full path to the executable if found. Exits if not found.
+    """
+    if os.path.exists(default_path) and os.access(default_path, os.X_OK):
+        print(f"✅ Found '{name}' at specified path: {default_path}")
+        return os.path.abspath(default_path)
+
+    found_path = shutil.which(name)
+    if found_path:
+        print(f"✅ Found '{name}' in system PATH: {found_path}")
+        return os.path.abspath(found_path)
+
+    print(f"❌ Error: '{name}' not found at '{default_path}' or in system PATH.")
+    sys.exit(1) # Exit if executable not found
+
+def find_script(name, default_path):
+    """
+    Finds a script file.
+
+    Args:
+        name (str): The name of the script.
+        default_path (str): The path to check for the script.
+
+    Returns:
+        str: The full path to the script if found. Exits if not found.
+    """
+    absolute_path = os.path.abspath(default_path)
+    if os.path.exists(absolute_path) and os.path.isfile(absolute_path):
+         print(f"✅ Found script '{name}' at specified path: {absolute_path}")
+         return absolute_path
+    print(f"❌ Error: Script '{name}' not found at '{default_path}' (abs: {absolute_path}). Please provide the correct path.")
+    sys.exit(1) # Exit if script not found
+
+# ...existing code...
+
+def compile_onnx_model(onnx_model_path, output_dir, onnx_mlir_exec_path):
+    """
+    Compiles FusedGemmRuntime.cpp, then compiles an ONNX model using onnx-mlir,
+    and finally links them together into a shared library.
+    Assumes FusedGemmRuntime.cpp is in the current working directory.
+    """
+    print(f"Starting compilation process for '{onnx_model_path}'...")
+    absolute_onnx_model_path = os.path.abspath(onnx_model_path)
+    absolute_output_dir = os.path.abspath(output_dir)
+    fused_gemm_cpp_path = os.path.abspath("FusedGemmRuntime.cpp") # Assumed in CWD
+
+    # --- Basic Checks ---
+    if not os.path.exists(absolute_onnx_model_path):
+        print(f"❌ Error: Input ONNX model not found at '{absolute_onnx_model_path}'")
+        return False, None
+    if not os.path.exists(fused_gemm_cpp_path):
+        print(f"❌ Error: FusedGemmRuntime.cpp not found at '{fused_gemm_cpp_path}' (expected in current directory)")
+        return False, None
+    # --- End Basic Checks ---
+
+    # Ensure output directory exists and is clean
+    if os.path.exists(absolute_output_dir):
+        print(f"Removing existing compilation output directory: {absolute_output_dir}")
+        shutil.rmtree(absolute_output_dir)
+    os.makedirs(absolute_output_dir, exist_ok=True)
+    print(f"Created compilation output directory: {absolute_output_dir}")
+
+    # Define paths for intermediate and final files *inside* the output directory
+    fused_gemm_obj_path = os.path.join(absolute_output_dir, "FusedGemmRuntime.o")
+    model_base_name = os.path.join(absolute_output_dir, "model") # Base for onnx-mlir output
+    model_obj_path = model_base_name + ".o"
+    model_lib_path = model_base_name + ".so"
+
+    # Step 1: Compile FusedGemmRuntime.cpp to FusedGemmRuntime.o
+    print(f"Compiling custom runtime '{fused_gemm_cpp_path}'...")
+    # Add -I include paths if necessary for FusedGemmRuntime.cpp
+    compile_custom_cmd = [
+        "clang++",
+        "-c", fused_gemm_cpp_path,
+        "-o", fused_gemm_obj_path,
+        "-fPIC",
+        # "-I/path/to/onnx-mlir/include" # Example: Add if needed
+    ]
+    print(f"Running command: {' '.join(compile_custom_cmd)}")
+    try:
+        result = subprocess.run(compile_custom_cmd, check=True, capture_output=True, text=True, timeout=120)
+        print("✅ Custom runtime compilation successful.")
+        if not os.path.exists(fused_gemm_obj_path):
+            print(f"❌ Error: Custom object '{fused_gemm_obj_path}' not found after compilation.")
+            print("Compiler Stdout:\n", result.stdout)
+            print("Compiler Stderr:\n", result.stderr)
+            return False, None
+    except subprocess.CalledProcessError as e:
+        print(f"❌ Error: Custom runtime compilation failed.")
+        print("Command:", ' '.join(e.cmd))
+        print("Return Code:", e.returncode)
+        print("Stdout:\n", e.stdout)
+        print("Stderr:\n", e.stderr)
+        return False, None
+    except Exception as e:
+        print(f"❌ Error: Custom runtime compilation failed with unexpected error: {e}")
+        return False, None
+
+
+    # Step 2: Compile ONNX model to model.o using onnx-mlir
+    print(f"Compiling ONNX model '{absolute_onnx_model_path}' to object file...")
+    compile_onnx_cmd = [
+        onnx_mlir_exec_path,
+        "--EmitObj",
+        absolute_onnx_model_path,
+        "-o", model_base_name # onnx-mlir appends .o automatically
+    ]
+    print(f"Running command: {' '.join(compile_onnx_cmd)}")
+    try:
+        result = subprocess.run(compile_onnx_cmd, check=True, capture_output=True, text=True, timeout=300)
+        print("✅ ONNX-MLIR object emission successful.")
+        if not os.path.exists(model_obj_path):
+            print(f"❌ Error: Compiled model object '{model_obj_path}' not found after compilation.")
+            print("Compiler Stdout:\n", result.stdout)
+            print("Compiler Stderr:\n", result.stderr)
+            return False, None
+    except subprocess.CalledProcessError as e:
+        print(f"❌ Error: ONNX-MLIR object emission failed.")
+        print("Command:", ' '.join(e.cmd))
+        print("Return Code:", e.returncode)
+        print("Stdout:\n", e.stdout)
+        print("Stderr:\n", e.stderr)
+        return False, None
+    except Exception as e:
+        print(f"❌ Error: ONNX-MLIR object emission failed with unexpected error: {e}")
+        return False, None
+
+    # Step 3: Link model.o and FusedGemmRuntime.o into model.so
+    print(f"Linking '{model_obj_path}' and '{fused_gemm_obj_path}' into '{model_lib_path}'...")
+    # Determine ONNX_MLIR_HOME to find the runtime library path
+    # You might need a more robust way to find this, e.g., environment variable or config
+    onnx_mlir_home = os.environ.get("ONNX_MLIR_HOME", "/workdir/onnx-mlir/build/Debug") # Default guess
+    lib_dir = os.path.join(onnx_mlir_home, "lib")
+    if not os.path.isdir(lib_dir):
+         print(f"⚠️ Warning: ONNX-MLIR library directory '{lib_dir}' not found. Linking might fail.")
+         # Attempt default system link paths? For now, proceed with the potentially incorrect path.
+
+    link_cmd = [
+        "clang++",
+        model_obj_path,
+        fused_gemm_obj_path,
+        "-o", model_lib_path,
+        "-shared", "-fPIC",
+        f"-L{lib_dir}", "-lcruntime" # Link against onnx-mlir C runtime
+    ]
+    print(f"Running command: {' '.join(link_cmd)}")
+    try:
+        result = subprocess.run(link_cmd, check=True, capture_output=True, text=True, timeout=120)
+        print("✅ Linking successful.")
+        if not os.path.exists(model_lib_path):
+            print(f"❌ Error: Linked library '{model_lib_path}' not found after linking.")
+            print("Linker Stdout:\n", result.stdout)
+            print("Linker Stderr:\n", result.stderr)
+            return False, None
+        # Return the directory containing the final .so file
+        return True, absolute_output_dir
+    except subprocess.CalledProcessError as e:
+        print(f"❌ Error: Linking failed.")
+        print("Command:", ' '.join(e.cmd))
+        print("Return Code:", e.returncode)
+        print("Stdout:\n", e.stdout)
+        print("Stderr:\n", e.stderr)
+        return False, None
+    except Exception as e:
+        print(f"❌ Error: Linking failed with unexpected error: {e}")
+        return False, None
+
+# ...existing code...
+
+def run_inference_with_script(run_script_path, compiled_model_dir_path, input_data_list, output_dir):
+    """
+    Runs inference by calling the RunONNXModel.py script via subprocess.
+    Assumes compiled model is 'model.so' inside compiled_model_dir_path.
+    Saves input numpy arrays and a loader script to output_dir, then executes RunONNXModel.py.
+
+    Args:
+        run_script_path (str): Absolute path to the RunONNXModel.py script.
+        compiled_model_dir_path (str): Absolute path to the directory containing the
+                                       compiled 'model.so'.
+        input_data_list (list): A list of NumPy arrays, one for each model input.
+        output_dir (str): The directory (relative to CWD) to use for this specific
+                          run's input/output files.
+
+    Returns:
+        tuple(list or None, float): A tuple containing:
+            - list or None: A list of NumPy arrays representing the model outputs if successful, otherwise None.
+            - float: The time taken for the subprocess call (includes overhead).
+    """
+    loaded_outputs = None
+
+    # Ensure paths are absolute
+    absolute_run_script_path = os.path.abspath(run_script_path)
+    absolute_compiled_model_dir = os.path.abspath(compiled_model_dir_path)
+    absolute_output_dir = os.path.abspath(output_dir) # Dir for this run's I/O
+
+    # --- Basic Checks ---
+    if not os.path.exists(absolute_run_script_path):
+        print(f"❌ Error: Run script not found at: {absolute_run_script_path}")
+        return None, 0
+    if not os.path.isdir(absolute_compiled_model_dir):
+        print(f"❌ Error: Compiled model directory not found or not a directory: {absolute_compiled_model_dir}")
+        return None, 0
+    expected_model_so_path = os.path.join(absolute_compiled_model_dir, "model.so")
+    if not os.path.exists(expected_model_so_path):
+        print(f"❌ Error: Expected compiled model 'model.so' not found in: {absolute_compiled_model_dir}")
+        return None, 0
+    # --- End Basic Checks ---
+
+    # Create the output directory for this specific run
+    os.makedirs(absolute_output_dir, exist_ok=True)
+
+    # --- Prepare Input Files and Loader Script ---
+    loader_script_path = os.path.join(absolute_output_dir, "_loader.py")
+    loader_script_content = """
+# Generated loader script for RunONNXModel.py --load-ref-from-numpy
+import numpy as np
+import os
+
+inputs = []
+i = 0
+while True:
+    input_npy_path = os.path.join(os.path.dirname(__file__), f"input_{i}.npy")
+    if os.path.exists(input_npy_path):
+        try:
+            inputs.append(np.load(input_npy_path))
+            i += 1
+        except Exception as e:
+            print(f"Error loading {input_npy_path}: {e}")
+            # Decide how to handle load errors, e.g., raise or break
+            break
+    else:
+        break
+
+# Optional: Define outputs = [] if needed for verification later
+# outputs = []
+"""
+    try:
+        # Save the input numpy arrays
+        for i, data in enumerate(input_data_list):
+            input_path_abs = os.path.join(absolute_output_dir, f"input_{i}.npy")
+            np.save(input_path_abs, data)
+            # print(f"   Saved input {i} to {input_path_abs}") # Optional debug print
+
+        # Save the loader script
+        with open(loader_script_path, "w") as f:
+            f.write(loader_script_content)
+        # print(f"   Saved loader script to {loader_script_path}") # Optional debug print
+
+    except Exception as e:
+        print(f"❌ Error preparing input files/loader script in {absolute_output_dir}: {e}")
+        return None, 0
+    # --- End Input Preparation ---
+
+
+    # Construct the command for RunONNXModel.py
+    run_command = [
+        sys.executable,
+        absolute_run_script_path,
+        "--load-model", absolute_compiled_model_dir, # Pass the DIRECTORY containing model.so
+        "--save-ref", absolute_output_dir,           # Save outputs to this run's dir
+        "--load-ref-from-numpy", loader_script_path  # Use the loader script
+    ]
+
+    print(f"Running inference command: {' '.join(run_command)}")
+    # Set the working directory for the subprocess to the run's output directory.
+    # This is where --save-ref will save files and where _loader.py will look for input_*.npy
+    print(f"  Working directory for subprocess: {absolute_output_dir}")
+    # Run the script with cwd set to the specified output directory
+    env = os.environ.copy()
+    onnx_mlir_home = os.environ.get("ONNX_MLIR_HOME", "/workdir/onnx-mlir/build/Debug") # Make sure this is correct
+    lib_dir = os.path.join(onnx_mlir_home, "lib")
+    if os.path.isdir(lib_dir):
+         env['LD_LIBRARY_PATH'] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
+         print(f"  Setting LD_LIBRARY_PATH for subprocess: {env['LD_LIBRARY_PATH']}")
+    else:
+         print(f"⚠️ Warning: ONNX-MLIR lib dir '{lib_dir}' not found. LD_LIBRARY_PATH not set.")
+
+    if "ONNX_MLIR_HOME" not in env:
+         print("⚠️ Warning: ONNX_MLIR_HOME environment variable not found. RunONNXModel.py might fail.")
+
+    start_time = time.time()
+    try: # Add try/except around subprocess.run
+        result = subprocess.run(run_command, check=True, capture_output=True, text=True, cwd=absolute_output_dir, timeout=60, env=env)
+        end_time = time.time()
+        # print("RunONNXModel Output:\n", result.stdout)
+        # print("RunONNXModel Stderr:\n", result.stderr)
+    except subprocess.CalledProcessError as e:
+         print(f"❌ Subprocess failed!")
+         print("Command:", ' '.join(e.cmd))
+         print("Return Code:", e.returncode)
+         print("Signal:", e.signal if hasattr(e, 'signal') else 'N/A') # Check if signal attribute exists
+         print("Stdout:\n", e.stdout)
+         print("Stderr:\n", e.stderr)
+         return None, 0 # Indicate failure
+    except Exception as e:
+         print(f"❌ Subprocess failed with unexpected error: {e}")
+         return None, 0 # Indicate failure
+
+    # --- Load output files from the specified output directory ---
+    loaded_outputs = []
+    i = 0
+    while True:
+        # Look for output files in the absolute_output_dir (where --save-ref saved them)
+        # RunONNXModel.py saves outputs as output_0.pb, output_1.pb etc. with --save-ref
+        output_path = os.path.join(absolute_output_dir, f"output_{i}.pb") # <-- Changed extension to .pb
+
+        if os.path.exists(output_path):
+            try:
+                # Load the protobuf tensor
+                output_ts = onnx.TensorProto()
+                with open(output_path, "rb") as f:
+                    output_ts.ParseFromString(f.read())
+                # Convert to numpy array
+                output_np = onnx.numpy_helper.to_array(output_ts)
+                loaded_outputs.append(output_np)
+                print(f"   ✅ Loaded output file: {output_path}")
+                i += 1
+            except Exception as load_err: # Use more specific exception if possible
+                print(f"   ⚠️ Error loading output file {output_path}: {load_err}")
+                loaded_outputs = None # Mark as failed if loading fails
+                break
+        else:
+            # Stop if output_i.pb is not found.
+            # If i is 0, it means no output files (output_0.pb) were found at all.
+            break # Exit the while loop
+
+    if not loaded_outputs:
+            # This warning now means the script ran successfully but didn't produce
+            # output_0.pb in the specified directory.
+            print(f"⚠️ Warning: No valid output files (e.g., output_0.pb) found or loaded from directory: {absolute_output_dir}")
+            # Print stdout/stderr from the script to help debug why it didn't save output
+            print("RunONNXModel stdout:\n", result.stdout)
+            print("RunONNXModel stderr:\n", result.stderr)
+            # exit message and exit program
+            print("❌ Exiting due to missing output files.")
+            exit(1)
+
+    
+    elapsed_time = end_time - start_time
+    return loaded_outputs, elapsed_time
+
+##################################################################################
+# MAIN PROGRAM ###################################################################
+##################################################################################
+
+"""
+Main execution block for the ONNX-MLIR benchmark script.
+"""
+
+if __name__ == "__main__":
+
+    #########################################
+    # SETUP OUTPUT DIRECTORY ################
+    #########################################
+    # Create a base directory for all inference runs in the current working directory
+    abs_inference_base_dir = os.path.abspath(INFERENCE_TEMP_DIR_BASE)
+    if os.path.exists(abs_inference_base_dir):
+        print(f"Removing existing base inference output directory: {abs_inference_base_dir}")
+        shutil.rmtree(abs_inference_base_dir)
+    os.makedirs(abs_inference_base_dir, exist_ok=True)
+    print(f"Created base directory for inference outputs: {abs_inference_base_dir}")
+
+
+    #########################################
+    # CHECK PREREQUISITES ###################
+    #########################################
+    print("Checking prerequisites...")
+    onnx_mlir_executable = find_executable("onnx-mlir", ONNX_MLIR_PATH)
+    absolute_run_script = find_script("RunONNXModel.py", RUN_ONNX_MODEL_SCRIPT_PATH)
+
+    # No need to check absolute_run_script again, find_script exits if not found
+    if not os.path.exists(MODEL_PATH):
+        print(f"❌ Error: Input ONNX model not found at '{MODEL_PATH}'. Please set the MODEL_PATH constant.")
+        sys.exit(1)
+    print("✅ Prerequisites check passed.")
+
+    #########################################
+    # COMPILE MODEL #########################
+    #########################################
+    # compile_onnx_model now returns the absolute path to the *directory* containing model.so
+    compilation_success, absolute_compiled_model_dir = compile_onnx_model(
+        MODEL_PATH,
+        COMPILED_MODEL_DIR, # Pass the desired output directory name
+        onnx_mlir_executable
+    )
+    if not compilation_success:
+        print("❌ Exiting due to compilation failure.")
+        sys.exit(1)
+    print(f"✅ Model compiled successfully. Library 'model.so' is in: {absolute_compiled_model_dir}")
+
+
+    #########################################
+    # DATASET LOADING #######################
+    #########################################
+    print("Loading FashionMNIST dataset...")
+    test_data = torchvision.datasets.FashionMNIST(
+        root=DATA_ROOT,
+        train=False,
+        download=True,
+        transform=torchvision.transforms.ToTensor()
+    )
+    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False) # Use shuffle=False for consistency
+    print("✅ Dataset loaded.")
+
+    #########################################
+    # BENCHMARK PREPARATION #################
+    #########################################
+    num_samples_to_load = min(len(test_data), NUM_ITERATIONS + WARMUP_ITERATIONS)
+    print(f"Preloading {num_samples_to_load} test samples for benchmarking...")
+    test_samples = []
+    test_loader_iter = iter(test_loader)
+
+    for i in range(num_samples_to_load):
+        try:
+            img, label = next(test_loader_iter)
+        except StopIteration:
+            print("Warning: Reached end of dataset during preloading.")
+            break # Stop preloading if dataset ends early
+
+        # Prepare input as expected by the model (flattened float32 numpy array)
+        img_np = img.reshape(BATCH_SIZE, FEATURE_DIM).numpy().astype(np.float32)
+        label_np = label.numpy() # Keep label as numpy scalar/array
+        test_samples.append((img_np, label_np))
+
+        if i == 0:
+            print(f"Sample input numpy shape: {img_np.shape}, dtype: {img_np.dtype}")
+
+    actual_warmup_iters = min(WARMUP_ITERATIONS, len(test_samples))
+    actual_bench_iters = min(NUM_ITERATIONS, len(test_samples) - actual_warmup_iters)
+
+    if len(test_samples) < WARMUP_ITERATIONS + NUM_ITERATIONS:
+         print(f"⚠️ Warning: Loaded only {len(test_samples)} samples, less than requested {WARMUP_ITERATIONS + NUM_ITERATIONS}.")
+         print(f"Adjusting: Warmup={actual_warmup_iters}, Benchmark={actual_bench_iters}")
+
+    if actual_bench_iters <= 0 and actual_warmup_iters <=0:
+        print("❌ Error: No samples loaded for benchmarking or warmup. Exiting.")
+        sys.exit(1)
+
+    print(f"✅ Preloaded {len(test_samples)} samples.")
+
+    #########################################
+    # BENCHMARKING ##########################
+    #########################################
+    print(f"\nRunning benchmark using '{absolute_run_script}'...")
+    total_time = 0
+    correct_predictions = 0
+    inference_failed = False
+
+    # --- Warmup Phase ---
+    if actual_warmup_iters > 0:
+        print(f"Starting {actual_warmup_iters} warmup iterations...")
+        for i in range(actual_warmup_iters):
+            img_np, _ = test_samples[i]
+            # Create a unique subdirectory for this warmup run
+            run_output_dir = os.path.join(abs_inference_base_dir, f"warmup_{i}")
+            # Pass the absolute path to the directory containing model.so
+            outputs, _ = run_inference_with_script( # Ignore time for warmup
+                absolute_run_script, absolute_compiled_model_dir, [img_np], run_output_dir
+            )
+            # Check for failure even in warmup
+            if outputs is None:
+                 print(f"❌ Warmup inference failed for sample {i}. Stopping.")
+                 inference_failed = True
+                 break
+        if not inference_failed:
+            print(f"✅ Completed {actual_warmup_iters} warmup iterations.")
+    else:
+        print("Skipping warmup phase (0 iterations).")
+
+
+    # --- Benchmarking Phase ---
+    if not inference_failed and actual_bench_iters > 0:
+        print(f"Starting {actual_bench_iters} benchmarking iterations...")
+        start_index = actual_warmup_iters
+        for i in range(actual_bench_iters):
+            sample_index = start_index + i
+            img_np, label_np = test_samples[sample_index]
+
+            # Create a unique subdirectory for this benchmark run
+            run_output_dir = os.path.join(abs_inference_base_dir, f"run_{i}")
+
+            # Time the inference script call, passing the absolute path to the model directory
+            outputs, elapsed_time = run_inference_with_script(
+                absolute_run_script, absolute_compiled_model_dir, [img_np], run_output_dir
+            )
+
+            if outputs is None:
+                print(f"❌ Inference failed for sample {sample_index}. Stopping benchmark.")
+                inference_failed = True
+                break # Stop benchmarking loop on failure
+
+            total_time += elapsed_time
+
+            # Process prediction (outside timing loop)
+            # Check if outputs is a non-empty list containing at least one numpy array
+            if outputs and isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], np.ndarray):
+                # Assuming the first output contains the logits
+                try:
+                    pred = np.argmax(outputs[0], axis=1) # Get predicted class index
+                    # Compare prediction with the ground truth label
+                    if pred == label_np: # Assumes label_np is a scalar or 1-element array
+                        correct_predictions += 1
+                except IndexError:
+                     # Handle cases where argmax might fail (e.g., unexpected output shape)
+                     print(f"⚠️ Error processing output for sample {sample_index}. Output shape: {outputs[0].shape}")
+            else:
+                 # This case is hit if run_inference_with_script returned an empty list or None
+                 # (though the None case should have been caught earlier)
+                 print(f"⚠️ No valid output data loaded for sample {sample_index}, cannot check accuracy.")
+
+        if not inference_failed:
+            print(f"✅ Completed {actual_bench_iters} benchmarking iterations.")
+    elif not inference_failed:
+        print("Skipping benchmarking phase (0 iterations).")
+
+
+    #########################################
+    # RESULTS REPORTING #####################
+    #########################################
+    print("\n======= ONNX-MLIR BENCHMARK RESULTS (via RunONNXModel.py) =======")
+    print(f"NOTE: Timing includes subprocess overhead for each inference call.")
+    if inference_failed:
+        print("Benchmark stopped early due to inference failure.")
+    elif actual_bench_iters > 0:
+        avg_time_ms = (total_time / actual_bench_iters) * 1000
+        # Ensure accuracy calculation avoids division by zero if correct_predictions is somehow > 0 but actual_bench_iters is 0
+        accuracy = (correct_predictions / actual_bench_iters) * 100 if actual_bench_iters > 0 else 0
+        # Ensure throughput calculation avoids division by zero
+        throughput = actual_bench_iters / total_time if total_time > 0 else 0
+
+        print(f"Compiled Model Directory: {absolute_compiled_model_dir}")
+        print(f"Inference Script: {absolute_run_script}")
+        print(f"Total Samples Benchmarked: {actual_bench_iters}")
+        print(f"Correct Predictions: {correct_predictions}")
+        print(f"Accuracy: {accuracy:.2f}%")
+        print(f"Total Inference Script Time: {total_time:.3f} s")
+        print(f"Avg. Inference Script Time: {avg_time_ms:.3f} ms/inference")
+        print(f"Throughput: {throughput:.2f} inferences/second (including overhead of running external file)")
+        print(f"Input/Output files stored under: {abs_inference_base_dir}")
+    else:
+        print("No benchmark iterations were successfully run.")
+
+    #########################################
+    # CLEANUP (Optional) ####################
+    #########################################
+    # Keep the INFERENCE_TEMP_DIR_BASE and COMPILED_MODEL_DIR for inspection by default.
+    # Uncomment below to clean up.
+    # print(f"\nCleaning up inference run directory: {abs_inference_base_dir}")
+    # if os.path.exists(abs_inference_base_dir):
+    #     shutil.rmtree(abs_inference_base_dir)
+
+    # print(f"Cleaning up compiled model directory: {absolute_compiled_model_dir}")
+    # if os.path.exists(absolute_compiled_model_dir):
+    #     shutil.rmtree(absolute_compiled_model_dir)
+
+    print("\nScript finished.")
\ No newline at end of file
diff --git a/src/matmul_relu_matmul_fashion_mnist/onnx-mlir_test_v3.py b/src/matmul_relu_matmul_fashion_mnist/onnx-mlir_test_v3.py
new file mode 100644
index 0000000..38498bd
--- /dev/null
+++ b/src/matmul_relu_matmul_fashion_mnist/onnx-mlir_test_v3.py
@@ -0,0 +1,570 @@
+##############################################
+# IMPORT LIBRARIES ###########################
+##############################################
+
+"""
+Libraries and packages used in this script and Versions Info for tools, libraries, and packages used in this script.
+"""
+import numpy as np          # Version: 1.26.4 (latest as of 2025-04-14)
+import time
+# import pickle as pkl # Not used directly if loading ONNX
+import torch                # Version: 2.2.2 (latest as of 2025-04-14)
+import torchvision          # Version: 0.17.2 (latest as of 2025-04-14)
+# import torch.nn as nn # Not needed if only loading ONNX
+import os
+import subprocess           # Version: Python Standard Library
+import sys                  # Version: Python Standard Library
+# import tempfile # Not used if INFERENCE_TEMP_DIR_BASE is used
+import shutil               # Version: Python Standard Library
+import importlib.util       # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import json                 # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import argparse             # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import signal               # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import shlex                # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import tempfile             # Version: Python Standard Library (Used indirectly via RunONNXModel.py)
+import onnx                 # Version: 1.16.0 (latest as of 2025-04-14, required by RunONNXModel.py)
+
+
+###############################################
+# CONSTANTS & PARAMETERS ######################
+###############################################
+
+"""
+Constants and parameters used in this script.
+"""
+# Paths
+MODEL_PATH = "mnist_model_cpu_optimized.onnx" # Path to the ONNX model to compile
+# COMPILED_MODEL_BASE = "mnist_model_cpu_initial" # No longer needed, will use "model" inside a dir
+COMPILED_MODEL_DIR = "compiled_model_onnx_mlir" # Directory to store the compiled model.so and model.constants.bin
+DATA_ROOT = "data"
+ONNX_MLIR_PATH = "onnx-mlir" # Assumes onnx-mlir executable is in PATH or provide full path
+RUN_ONNX_MODEL_SCRIPT_PATH = "/workdir/onnx-mlir/utils/RunONNXModel.py" # Absolute path to the script
+# Directory to store temporary files for each inference run, created in the script's CWD
+INFERENCE_TEMP_DIR_BASE = "onnx_mlir_inference_runs"
+
+# Dataset configuration
+BATCH_SIZE = 1
+FEATURE_DIM = 784  # 28x28 flattened
+IMG_SHAPE = (1, 28, 28)
+NUM_CLASSES = 10
+
+# Benchmark configuration
+NUM_ITERATIONS = 100 # Reduced for potentially slower subprocess calls
+WARMUP_ITERATIONS = 20  # Number of warmup iterations
+
+# Class names for interpretation
+CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
+               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
+
+print("Starting ONNX-MLIR benchmark script (using RunONNXModel.py via subprocess)")
+
+##############################################
+# FUNCTION DEFINITIONS #######################
+##############################################
+
+def find_executable(name, default_path):
+    """
+    Finds an executable by checking the default path and then the system PATH.
+
+    Args:
+        name (str): The name of the executable.
+        default_path (str): The default path to check first.
+
+    Returns:
+        str: The full path to the executable if found. Exits if not found.
+    """
+    if os.path.exists(default_path) and os.access(default_path, os.X_OK):
+        print(f"✅ Found '{name}' at specified path: {default_path}")
+        return os.path.abspath(default_path)
+
+    found_path = shutil.which(name)
+    if found_path:
+        print(f"✅ Found '{name}' in system PATH: {found_path}")
+        return os.path.abspath(found_path)
+
+    print(f"❌ Error: '{name}' not found at '{default_path}' or in system PATH.")
+    sys.exit(1) # Exit if executable not found
+
+def find_script(name, default_path):
+    """
+    Finds a script file.
+
+    Args:
+        name (str): The name of the script.
+        default_path (str): The path to check for the script.
+
+    Returns:
+        str: The full path to the script if found. Exits if not found.
+    """
+    absolute_path = os.path.abspath(default_path)
+    if os.path.exists(absolute_path) and os.path.isfile(absolute_path):
+         print(f"✅ Found script '{name}' at specified path: {absolute_path}")
+         return absolute_path
+    print(f"❌ Error: Script '{name}' not found at '{default_path}' (abs: {absolute_path}). Please provide the correct path.")
+    sys.exit(1) # Exit if script not found
+
+# ...existing code...
+
+def compile_onnx_model(onnx_model_path, output_dir, onnx_mlir_exec_path):
+    """
+    Compiles an ONNX model using onnx-mlir, then compiles and links with FusedGemmRuntime_omtensor.cpp.
+    """
+    print(f"Compiling ONNX model '{onnx_model_path}' with onnx-mlir...")
+    absolute_onnx_model_path = os.path.abspath(onnx_model_path)
+    absolute_output_dir = os.path.abspath(output_dir)
+
+    if not os.path.exists(absolute_onnx_model_path):
+        print(f"❌ Error: Input ONNX model not found at '{absolute_onnx_model_path}'")
+        return False, None
+
+    # Ensure output directory exists and is clean
+    if os.path.exists(absolute_output_dir):
+        print(f"Removing existing compilation output directory: {absolute_output_dir}")
+        shutil.rmtree(absolute_output_dir)
+    os.makedirs(absolute_output_dir, exist_ok=True)
+    print(f"Created compilation output directory: {absolute_output_dir}")
+
+    # Define the base name for output files *inside* the output directory
+    output_base_name = os.path.join(absolute_output_dir, "model")
+    expected_lib_path = output_base_name + ".so"
+    expected_obj_path = output_base_name + ".o"
+
+    # Step 1: Emit object file from ONNX-MLIR
+    compile_command = [
+        onnx_mlir_exec_path,
+        "--EmitObj",
+        absolute_onnx_model_path,
+        "-o", output_base_name
+    ]
+    print(f"Running command: {' '.join(compile_command)}")
+    try:
+        result = subprocess.run(compile_command, check=True, capture_output=True, text=True, timeout=300)
+        print("✅ ONNX-MLIR object emission successful.")
+        if not os.path.exists(expected_obj_path):
+            print(f"❌ Error: Compiled object '{expected_obj_path}' not found after compilation.")
+            print("Compiler Stdout:\n", result.stdout)
+            print("Compiler Stderr:\n", result.stderr)
+            return False, None
+    except Exception as e:
+        print(f"❌ Error: ONNX-MLIR object emission failed: {e}")
+        return False, None
+
+    # Step 2: Compile FusedGemmRuntime_omtensor.cpp to shared library
+    fused_cpp = os.path.abspath("FusedGemmRuntime_omtensor.cpp")
+    fused_so = os.path.abspath("libFusedGemmRuntime_omtensor.so")
+    onnx_mlir_include = "/workdir/onnx-mlir/include"
+    if not os.path.exists(fused_cpp):
+        print(f"❌ Error: FusedGemmRuntime_omtensor.cpp not found at {fused_cpp}")
+        return False, None
+    clang_compile_cmd = [
+        "clang++",
+        "-fPIC", "-shared",
+        f"-I{onnx_mlir_include}",
+        "-o", fused_so,
+        fused_cpp
+    ]
+    print(f"Compiling FusedGemmRuntime_omtensor.cpp: {' '.join(clang_compile_cmd)}")
+    try:
+        result = subprocess.run(clang_compile_cmd, check=True, capture_output=True, text=True, timeout=120)
+        print("✅ FusedGemmRuntime_omtensor.cpp compilation successful.")
+        if not os.path.exists(fused_so):
+            print(f"❌ Error: Compiled shared library '{fused_so}' not found after compilation.")
+            print("Compiler Stdout:\n", result.stdout)
+            print("Compiler Stderr:\n", result.stderr)
+            return False, None
+    except Exception as e:
+        print(f"❌ Error: FusedGemmRuntime_omtensor.cpp compilation failed: {e}")
+        return False, None
+
+    # Step 3: Link ONNX-MLIR object and FusedGemmRuntime_omtensor.so into final model.so
+    clang_link_cmd = [
+        "clang++",
+        expected_obj_path,
+        fused_so,
+        "-o", expected_lib_path,
+        "-shared", "-fPIC",
+        "-L/workdir/onnx-mlir/build/Debug/lib", "-lcruntime"
+    ]
+    print(f"Linking model and FusedGemmRuntime_omtensor.so: {' '.join(clang_link_cmd)}")
+    try:
+        result = subprocess.run(clang_link_cmd, check=True, capture_output=True, text=True, timeout=120)
+        print("✅ Linking successful.")
+        if not os.path.exists(expected_lib_path):
+            print(f"❌ Error: Linked library '{expected_lib_path}' not found after linking.")
+            print("Linker Stdout:\n", result.stdout)
+            print("Linker Stderr:\n", result.stderr)
+            return False, None
+        return True, absolute_output_dir
+    except Exception as e:
+        print(f"❌ Error: Linking failed: {e}")
+        return False, None
+
+# ...existing code...
+
+def run_inference_with_script(run_script_path, compiled_model_dir_path, input_data_list, output_dir):
+    """
+    Runs inference by calling the RunONNXModel.py script via subprocess.
+    Assumes compiled model is 'model.so' inside compiled_model_dir_path.
+    Saves input numpy arrays and a loader script to output_dir, then executes RunONNXModel.py.
+
+    Args:
+        run_script_path (str): Absolute path to the RunONNXModel.py script.
+        compiled_model_dir_path (str): Absolute path to the directory containing the
+                                       compiled 'model.so'.
+        input_data_list (list): A list of NumPy arrays, one for each model input.
+        output_dir (str): The directory (relative to CWD) to use for this specific
+                          run's input/output files.
+
+    Returns:
+        tuple(list or None, float): A tuple containing:
+            - list or None: A list of NumPy arrays representing the model outputs if successful, otherwise None.
+            - float: The time taken for the subprocess call (includes overhead).
+    """
+    loaded_outputs = None
+
+    # Ensure paths are absolute
+    absolute_run_script_path = os.path.abspath(run_script_path)
+    absolute_compiled_model_dir = os.path.abspath(compiled_model_dir_path)
+    absolute_output_dir = os.path.abspath(output_dir) # Dir for this run's I/O
+
+    # --- Basic Checks ---
+    if not os.path.exists(absolute_run_script_path):
+        print(f"❌ Error: Run script not found at: {absolute_run_script_path}")
+        return None, 0
+    if not os.path.isdir(absolute_compiled_model_dir):
+        print(f"❌ Error: Compiled model directory not found or not a directory: {absolute_compiled_model_dir}")
+        return None, 0
+    expected_model_so_path = os.path.join(absolute_compiled_model_dir, "model.so")
+    if not os.path.exists(expected_model_so_path):
+        print(f"❌ Error: Expected compiled model 'model.so' not found in: {absolute_compiled_model_dir}")
+        return None, 0
+    # --- End Basic Checks ---
+
+    # Create the output directory for this specific run
+    os.makedirs(absolute_output_dir, exist_ok=True)
+
+    # --- Prepare Input Files and Loader Script ---
+    loader_script_path = os.path.join(absolute_output_dir, "_loader.py")
+    loader_script_content = """
+# Generated loader script for RunONNXModel.py --load-ref-from-numpy
+import numpy as np
+import os
+
+inputs = []
+i = 0
+while True:
+    input_npy_path = os.path.join(os.path.dirname(__file__), f"input_{i}.npy")
+    if os.path.exists(input_npy_path):
+        try:
+            inputs.append(np.load(input_npy_path))
+            i += 1
+        except Exception as e:
+            print(f"Error loading {input_npy_path}: {e}")
+            # Decide how to handle load errors, e.g., raise or break
+            break
+    else:
+        break
+
+# Optional: Define outputs = [] if needed for verification later
+# outputs = []
+"""
+    try:
+        # Save the input numpy arrays
+        for i, data in enumerate(input_data_list):
+            input_path_abs = os.path.join(absolute_output_dir, f"input_{i}.npy")
+            np.save(input_path_abs, data)
+            # print(f"   Saved input {i} to {input_path_abs}") # Optional debug print
+
+        # Save the loader script
+        with open(loader_script_path, "w") as f:
+            f.write(loader_script_content)
+        # print(f"   Saved loader script to {loader_script_path}") # Optional debug print
+
+    except Exception as e:
+        print(f"❌ Error preparing input files/loader script in {absolute_output_dir}: {e}")
+        return None, 0
+    # --- End Input Preparation ---
+
+
+    # Construct the command for RunONNXModel.py
+    run_command = [
+        sys.executable,
+        absolute_run_script_path,
+        "--load-model", absolute_compiled_model_dir, # Pass the DIRECTORY containing model.so
+        "--save-ref", absolute_output_dir,           # Save outputs to this run's dir
+        "--load-ref-from-numpy", loader_script_path  # Use the loader script
+    ]
+
+    print(f"Running inference command: {' '.join(run_command)}")
+    # Set the working directory for the subprocess to the run's output directory.
+    # This is where --save-ref will save files and where _loader.py will look for input_*.npy
+    print(f"  Working directory for subprocess: {absolute_output_dir}")
+    # Run the script with cwd set to the specified output directory
+    env = os.environ.copy()
+    if "ONNX_MLIR_HOME" not in env:
+            print("⚠️ Warning: ONNX_MLIR_HOME environment variable not found. RunONNXModel.py might fail.")
+            # Consider adding ONNX_MLIR_HOME if known and missing? For now, just warn.
+    start_time = time.time()
+    result = subprocess.run(run_command, check=True, capture_output=True, text=True, cwd=absolute_output_dir, timeout=60, env=env)
+    end_time = time.time()
+    # print("RunONNXModel Output:\n", result.stdout) # Often empty on success
+    # print("RunONNXModel Stderr:\n", result.stderr) # Check stderr for potential info
+
+    # --- Load output files from the specified output directory ---
+    loaded_outputs = []
+    i = 0
+    while True:
+        # Look for output files in the absolute_output_dir (where --save-ref saved them)
+        # RunONNXModel.py saves outputs as output_0.pb, output_1.pb etc. with --save-ref
+        output_path = os.path.join(absolute_output_dir, f"output_{i}.pb") # <-- Changed extension to .pb
+
+        if os.path.exists(output_path):
+            try:
+                # Load the protobuf tensor
+                output_ts = onnx.TensorProto()
+                with open(output_path, "rb") as f:
+                    output_ts.ParseFromString(f.read())
+                # Convert to numpy array
+                output_np = onnx.numpy_helper.to_array(output_ts)
+                loaded_outputs.append(output_np)
+                print(f"   ✅ Loaded output file: {output_path}")
+                i += 1
+            except Exception as load_err: # Use more specific exception if possible
+                print(f"   ⚠️ Error loading output file {output_path}: {load_err}")
+                loaded_outputs = None # Mark as failed if loading fails
+                break
+        else:
+            # Stop if output_i.pb is not found.
+            # If i is 0, it means no output files (output_0.pb) were found at all.
+            break # Exit the while loop
+
+    if not loaded_outputs:
+            # This warning now means the script ran successfully but didn't produce
+            # output_0.pb in the specified directory.
+            print(f"⚠️ Warning: No valid output files (e.g., output_0.pb) found or loaded from directory: {absolute_output_dir}")
+            # Print stdout/stderr from the script to help debug why it didn't save output
+            print("RunONNXModel stdout:\n", result.stdout)
+            print("RunONNXModel stderr:\n", result.stderr)
+            # exit message and exit program
+            print("❌ Exiting due to missing output files.")
+            exit(1)
+
+    
+    elapsed_time = end_time - start_time
+    return loaded_outputs, elapsed_time
+
+##################################################################################
+# MAIN PROGRAM ###################################################################
+##################################################################################
+
+"""
+Main execution block for the ONNX-MLIR benchmark script.
+"""
+
+if __name__ == "__main__":
+
+    #########################################
+    # SETUP OUTPUT DIRECTORY ################
+    #########################################
+    # Create a base directory for all inference runs in the current working directory
+    abs_inference_base_dir = os.path.abspath(INFERENCE_TEMP_DIR_BASE)
+    if os.path.exists(abs_inference_base_dir):
+        print(f"Removing existing base inference output directory: {abs_inference_base_dir}")
+        shutil.rmtree(abs_inference_base_dir)
+    os.makedirs(abs_inference_base_dir, exist_ok=True)
+    print(f"Created base directory for inference outputs: {abs_inference_base_dir}")
+
+
+    #########################################
+    # CHECK PREREQUISITES ###################
+    #########################################
+    print("Checking prerequisites...")
+    onnx_mlir_executable = find_executable("onnx-mlir", ONNX_MLIR_PATH)
+    absolute_run_script = find_script("RunONNXModel.py", RUN_ONNX_MODEL_SCRIPT_PATH)
+
+    # No need to check absolute_run_script again, find_script exits if not found
+    if not os.path.exists(MODEL_PATH):
+        print(f"❌ Error: Input ONNX model not found at '{MODEL_PATH}'. Please set the MODEL_PATH constant.")
+        sys.exit(1)
+    print("✅ Prerequisites check passed.")
+
+    #########################################
+    # COMPILE MODEL #########################
+    #########################################
+    # compile_onnx_model now returns the absolute path to the *directory* containing model.so
+    compilation_success, absolute_compiled_model_dir = compile_onnx_model(
+        MODEL_PATH,
+        COMPILED_MODEL_DIR, # Pass the desired output directory name
+        onnx_mlir_executable
+    )
+    if not compilation_success:
+        print("❌ Exiting due to compilation failure.")
+        sys.exit(1)
+    print(f"✅ Model compiled successfully. Library 'model.so' is in: {absolute_compiled_model_dir}")
+
+
+    #########################################
+    # DATASET LOADING #######################
+    #########################################
+    print("Loading FashionMNIST dataset...")
+    test_data = torchvision.datasets.FashionMNIST(
+        root=DATA_ROOT,
+        train=False,
+        download=True,
+        transform=torchvision.transforms.ToTensor()
+    )
+    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False) # Use shuffle=False for consistency
+    print("✅ Dataset loaded.")
+
+    #########################################
+    # BENCHMARK PREPARATION #################
+    #########################################
+    num_samples_to_load = min(len(test_data), NUM_ITERATIONS + WARMUP_ITERATIONS)
+    print(f"Preloading {num_samples_to_load} test samples for benchmarking...")
+    test_samples = []
+    test_loader_iter = iter(test_loader)
+
+    for i in range(num_samples_to_load):
+        try:
+            img, label = next(test_loader_iter)
+        except StopIteration:
+            print("Warning: Reached end of dataset during preloading.")
+            break # Stop preloading if dataset ends early
+
+        # Prepare input as expected by the model (flattened float32 numpy array)
+        img_np = img.reshape(BATCH_SIZE, FEATURE_DIM).numpy().astype(np.float32)
+        label_np = label.numpy() # Keep label as numpy scalar/array
+        test_samples.append((img_np, label_np))
+
+        if i == 0:
+            print(f"Sample input numpy shape: {img_np.shape}, dtype: {img_np.dtype}")
+
+    actual_warmup_iters = min(WARMUP_ITERATIONS, len(test_samples))
+    actual_bench_iters = min(NUM_ITERATIONS, len(test_samples) - actual_warmup_iters)
+
+    if len(test_samples) < WARMUP_ITERATIONS + NUM_ITERATIONS:
+         print(f"⚠️ Warning: Loaded only {len(test_samples)} samples, less than requested {WARMUP_ITERATIONS + NUM_ITERATIONS}.")
+         print(f"Adjusting: Warmup={actual_warmup_iters}, Benchmark={actual_bench_iters}")
+
+    if actual_bench_iters <= 0 and actual_warmup_iters <=0:
+        print("❌ Error: No samples loaded for benchmarking or warmup. Exiting.")
+        sys.exit(1)
+
+    print(f"✅ Preloaded {len(test_samples)} samples.")
+
+    #########################################
+    # BENCHMARKING ##########################
+    #########################################
+    print(f"\nRunning benchmark using '{absolute_run_script}'...")
+    total_time = 0
+    correct_predictions = 0
+    inference_failed = False
+
+    # --- Warmup Phase ---
+    if actual_warmup_iters > 0:
+        print(f"Starting {actual_warmup_iters} warmup iterations...")
+        for i in range(actual_warmup_iters):
+            img_np, _ = test_samples[i]
+            # Create a unique subdirectory for this warmup run
+            run_output_dir = os.path.join(abs_inference_base_dir, f"warmup_{i}")
+            # Pass the absolute path to the directory containing model.so
+            outputs, _ = run_inference_with_script( # Ignore time for warmup
+                absolute_run_script, absolute_compiled_model_dir, [img_np], run_output_dir
+            )
+            # Check for failure even in warmup
+            if outputs is None:
+                 print(f"❌ Warmup inference failed for sample {i}. Stopping.")
+                 inference_failed = True
+                 break
+        if not inference_failed:
+            print(f"✅ Completed {actual_warmup_iters} warmup iterations.")
+    else:
+        print("Skipping warmup phase (0 iterations).")
+
+
+    # --- Benchmarking Phase ---
+    if not inference_failed and actual_bench_iters > 0:
+        print(f"Starting {actual_bench_iters} benchmarking iterations...")
+        start_index = actual_warmup_iters
+        for i in range(actual_bench_iters):
+            sample_index = start_index + i
+            img_np, label_np = test_samples[sample_index]
+
+            # Create a unique subdirectory for this benchmark run
+            run_output_dir = os.path.join(abs_inference_base_dir, f"run_{i}")
+
+            # Time the inference script call, passing the absolute path to the model directory
+            outputs, elapsed_time = run_inference_with_script(
+                absolute_run_script, absolute_compiled_model_dir, [img_np], run_output_dir
+            )
+
+            if outputs is None:
+                print(f"❌ Inference failed for sample {sample_index}. Stopping benchmark.")
+                inference_failed = True
+                break # Stop benchmarking loop on failure
+
+            total_time += elapsed_time
+
+            # Process prediction (outside timing loop)
+            # Check if outputs is a non-empty list containing at least one numpy array
+            if outputs and isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], np.ndarray):
+                # Assuming the first output contains the logits
+                try:
+                    pred = np.argmax(outputs[0], axis=1) # Get predicted class index
+                    # Compare prediction with the ground truth label
+                    if pred == label_np: # Assumes label_np is a scalar or 1-element array
+                        correct_predictions += 1
+                except IndexError:
+                     # Handle cases where argmax might fail (e.g., unexpected output shape)
+                     print(f"⚠️ Error processing output for sample {sample_index}. Output shape: {outputs[0].shape}")
+            else:
+                 # This case is hit if run_inference_with_script returned an empty list or None
+                 # (though the None case should have been caught earlier)
+                 print(f"⚠️ No valid output data loaded for sample {sample_index}, cannot check accuracy.")
+
+        if not inference_failed:
+            print(f"✅ Completed {actual_bench_iters} benchmarking iterations.")
+    elif not inference_failed:
+        print("Skipping benchmarking phase (0 iterations).")
+
+
+    #########################################
+    # RESULTS REPORTING #####################
+    #########################################
+    print("\n======= ONNX-MLIR BENCHMARK RESULTS (via RunONNXModel.py) =======")
+    print(f"NOTE: Timing includes subprocess overhead for each inference call.")
+    if inference_failed:
+        print("Benchmark stopped early due to inference failure.")
+    elif actual_bench_iters > 0:
+        avg_time_ms = (total_time / actual_bench_iters) * 1000
+        # Ensure accuracy calculation avoids division by zero if correct_predictions is somehow > 0 but actual_bench_iters is 0
+        accuracy = (correct_predictions / actual_bench_iters) * 100 if actual_bench_iters > 0 else 0
+        # Ensure throughput calculation avoids division by zero
+        throughput = actual_bench_iters / total_time if total_time > 0 else 0
+
+        print(f"Compiled Model Directory: {absolute_compiled_model_dir}")
+        print(f"Inference Script: {absolute_run_script}")
+        print(f"Total Samples Benchmarked: {actual_bench_iters}")
+        print(f"Correct Predictions: {correct_predictions}")
+        print(f"Accuracy: {accuracy:.2f}%")
+        print(f"Total Inference Script Time: {total_time:.3f} s")
+        print(f"Avg. Inference Script Time: {avg_time_ms:.3f} ms/inference")
+        print(f"Throughput: {throughput:.2f} inferences/second (including overhead of running external file)")
+        print(f"Input/Output files stored under: {abs_inference_base_dir}")
+    else:
+        print("No benchmark iterations were successfully run.")
+
+    #########################################
+    # CLEANUP (Optional) ####################
+    #########################################
+    # Keep the INFERENCE_TEMP_DIR_BASE and COMPILED_MODEL_DIR for inspection by default.
+    # Uncomment below to clean up.
+    # print(f"\nCleaning up inference run directory: {abs_inference_base_dir}")
+    # if os.path.exists(abs_inference_base_dir):
+    #     shutil.rmtree(abs_inference_base_dir)
+
+    # print(f"Cleaning up compiled model directory: {absolute_compiled_model_dir}")
+    # if os.path.exists(absolute_compiled_model_dir):
+    #     shutil.rmtree(absolute_compiled_model_dir)
+
+    print("\nScript finished.")
\ No newline at end of file
diff --git a/src/matmul_relu_matmul_fashion_mnist/onnx_cpu_ep_export.py b/src/matmul_relu_matmul_fashion_mnist/onnx_cpu_ep_export.py
new file mode 100644
index 0000000..33a7717
--- /dev/null
+++ b/src/matmul_relu_matmul_fashion_mnist/onnx_cpu_ep_export.py
@@ -0,0 +1,280 @@
+##############################################
+# IMPORT LIBRARIES ###########################
+##############################################
+
+"""
+Libraries and packages used in this script and Versions Info for tools, libraries, and packages used in this script.
+"""
+import numpy as np      # Version: 1.26.4 (latest as of cutoff)
+import time
+import pickle as pkl
+import torch            # Version: 2.2.2 (latest as of cutoff)
+import torchvision      # Version: 0.17.2 (latest as of cutoff)
+import torch.nn as nn
+import onnxruntime as ort # Version: 1.17.1 (latest as of cutoff)
+import os
+
+###############################################
+# CONSTANTS & PARAMETERS ######################
+###############################################
+
+"""
+Constants and parameters used in this script.
+"""
+# Paths
+DATA_ROOT = "data"
+MODEL_PARAMS_PATH = "fasionmnist_mlp_params.pkl"
+ONNX_FP32_PATH = "mnist_model_cpu_initial.onnx"
+OPTIMIZED_ONNX_FP32_PATH = "mnist_model_cpu_optimized.onnx" # Path for the optimized model
+
+# Device settings
+DEVICE = "cpu" # Force CPU usage
+
+# Dataset configuration
+BATCH_SIZE = 1
+FEATURE_DIM = 784  # 28x28 flattened
+IMG_SHAPE = (1, 28, 28)
+NUM_CLASSES = 10
+
+# Benchmark configuration
+NUM_ITERATIONS = 500
+WARMUP_ITERATIONS = 100  # Number of warmup iterations
+
+# Class names for interpretation
+CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
+               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
+
+print("Starting ONNX Runtime with CPU EP benchmark and optimization script")
+
+##############################################
+# FUNCTION DEFINITIONS #######################
+##############################################
+
+class MLPModel(nn.Module):
+    """
+    Defines a simple Multi-Layer Perceptron model.
+
+    Args:
+        w0 (np.ndarray): Weight matrix for the first linear layer.
+        b0 (np.ndarray): Bias vector for the first linear layer.
+        w1 (np.ndarray): Weight matrix for the second linear layer.
+        b1 (np.ndarray): Bias vector for the second linear layer.
+        dtype (torch.dtype): Data type for the model parameters (default: torch.float32).
+    """
+    def __init__(self, w0, b0, w1, b1, dtype=torch.float32):
+        super(MLPModel, self).__init__()
+        # Ensure dimensions match expected PyTorch linear layer (out_features, in_features)
+        self.fc1 = nn.Linear(w0.shape[1], w0.shape[0])
+        self.fc2 = nn.Linear(w1.shape[1], w1.shape[0])
+        self.relu = nn.ReLU()
+
+        # Load weights and biases, ensuring correct data type
+        self.fc1.weight = nn.Parameter(torch.tensor(w0, dtype=dtype))
+        self.fc1.bias = nn.Parameter(torch.tensor(b0, dtype=dtype))
+        self.fc2.weight = nn.Parameter(torch.tensor(w1, dtype=dtype))
+        self.fc2.bias = nn.Parameter(torch.tensor(b1, dtype=dtype))
+
+    def forward(self, x):
+        """
+        Forward pass through the MLP.
+
+        Args:
+            x (torch.Tensor): Input tensor.
+
+        Returns:
+            torch.Tensor: Output tensor (logits).
+        """
+        x = self.fc1(x)
+        x = self.relu(x)
+        x = self.fc2(x)
+        return x
+
+##################################################################################
+# MAIN PROGRAM ###################################################################
+##################################################################################
+
+if __name__ == "__main__":
+    #########################################
+    # DATASET LOADING #######################
+    #########################################
+    print("Loading FashionMNIST dataset...")
+    test_data = torchvision.datasets.FashionMNIST(
+        root=DATA_ROOT,
+        train=False,
+        download=True,
+        transform=torchvision.transforms.ToTensor()
+    )
+    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
+
+    # Get a sample input for tracing/export
+    img_sample, _ = next(iter(test_loader))
+    img_sample = img_sample.reshape(-1, FEATURE_DIM) # Flatten
+    print(f"Sample input image shape: {img_sample.shape}, dtype: {img_sample.dtype}")
+
+    #########################################
+    # LOAD MODEL & WEIGHTS ##################
+    #########################################
+    print(f"Loading pre-trained weights from {MODEL_PARAMS_PATH}...")
+    try:
+        mlp_params = pkl.load(open(MODEL_PARAMS_PATH, "rb"))
+    except FileNotFoundError:
+        print(f"❌ Error: Model parameters file not found at {MODEL_PARAMS_PATH}")
+        exit(1)
+    except Exception as e:
+        print(f"❌ Error loading model parameters: {e}")
+        exit(1)
+
+    # Create FP32 model instance on CPU
+    model_fp32 = MLPModel(
+        mlp_params["w0"],
+        mlp_params["b0"],
+        mlp_params["w1"],
+        mlp_params["b1"],
+        dtype=torch.float32
+    ).to(DEVICE) # Ensure model is on CPU
+    model_fp32.eval() # Set to evaluation mode
+    print("✅ Model created and weights loaded onto CPU.")
+
+    #########################################
+    # INITIAL ONNX EXPORT (FP32) ############
+    #########################################
+    print(f"Exporting initial FP32 model to {ONNX_FP32_PATH}...")
+    try:
+        # Prepare input tensor on the correct device (CPU)
+        tracing_input_fp32 = img_sample.to(DEVICE, dtype=torch.float32)
+
+        torch.onnx.export(
+            model_fp32,                   # model being run
+            tracing_input_fp32,           # model input (or a tuple for multiple inputs)
+            ONNX_FP32_PATH,               # where to save the model
+            export_params=True,           # store the trained parameter weights inside the model file
+            opset_version=11,             # the ONNX version to export the model to
+            do_constant_folding=True,     # whether to execute constant folding for optimization
+            input_names=['input'],        # the model's input names
+            output_names=['output'],      # the model's output names
+            dynamic_axes={'input': {0: 'batch_size'}, # variable length axes
+                          'output': {0: 'batch_size'}}
+        )
+        print(f"✅ Successfully exported initial FP32 ONNX model to {ONNX_FP32_PATH}")
+    except Exception as e:
+        print(f"❌ Error exporting initial ONNX model: {e}")
+        exit(1)
+
+    ####################################################
+    # ONNX RUNTIME CPU SESSION & OPTIMIZATION ##########
+    ####################################################
+    print("Initializing ONNX Runtime with CPU Execution Provider and enabling optimization...")
+
+    # Check available providers
+    print(f"Available ONNX Runtime providers: {ort.get_available_providers()}")
+    if 'CPUExecutionProvider' not in ort.get_available_providers():
+        print("❌ Critical Error: CPUExecutionProvider is not available in this ONNX Runtime build!")
+        exit(1)
+
+    try:
+        # Configure session options for optimization
+        sess_options = ort.SessionOptions()
+
+        # Set graph optimization level (ORT_ENABLE_ALL includes layout optimizations etc.)
+        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
+
+        # Specify the path to save the optimized model
+        sess_options.optimized_model_filepath = OPTIMIZED_ONNX_FP32_PATH
+
+        # Create the inference session with only the CPU provider
+        # ONNX Runtime will apply optimizations and save the result to optimized_model_filepath
+        onnx_cpu_session = ort.InferenceSession(
+            ONNX_FP32_PATH,
+            sess_options=sess_options,
+            providers=['CPUExecutionProvider'] # Explicitly use only CPU EP
+        )
+        print(f"✅ Successfully created ONNX Runtime CPU session.")
+        print(f"✅ ONNX Runtime optimizations applied. Optimized model saved to: {OPTIMIZED_ONNX_FP32_PATH}")
+
+        # Verify the optimized file exists
+        if not os.path.exists(OPTIMIZED_ONNX_FP32_PATH):
+             print(f"⚠️ Warning: Optimized model file was expected but not found at {OPTIMIZED_ONNX_FP32_PATH}")
+
+    except Exception as e:
+        print(f"❌ Failed to create ONNX CPU session or apply optimizations: {e}")
+        exit(1)
+
+    #########################################
+    # BENCHMARK PREPARATION #################
+    #########################################
+    print(f"Preloading {NUM_ITERATIONS} test samples for benchmarking...")
+    test_samples = []
+    test_loader_iter = iter(test_loader)
+
+    # Preload data samples as NumPy arrays for direct ONNX Runtime inference
+    for _ in range(NUM_ITERATIONS + WARMUP_ITERATIONS): # Preload enough for warmup + benchmark
+        try:
+            img, label = next(test_loader_iter)
+        except StopIteration:
+            test_loader_iter = iter(test_loader)
+            img, label = next(test_loader_iter)
+
+        img_np = img.reshape(BATCH_SIZE, FEATURE_DIM).numpy().astype(np.float32) # Flatten and convert
+        label_np = label.numpy()
+        test_samples.append((img_np, label_np))
+
+    print(f"Preloaded {len(test_samples)} samples.")
+
+    # Get input name from the session
+    input_name = onnx_cpu_session.get_inputs()[0].name
+
+    #########################################
+    # BENCHMARKING ##########################
+    #########################################
+    print(f"\nRunning benchmark with CPU Execution Provider...")
+    total_time = 0
+    correct_predictions = 0
+
+    # --- Warmup Phase ---
+    print(f"Starting {WARMUP_ITERATIONS} warmup iterations...")
+    for i in range(WARMUP_ITERATIONS):
+        img_np, _ = test_samples[i]
+        _ = onnx_cpu_session.run(None, {input_name: img_np})
+    print(f"Completed {WARMUP_ITERATIONS} warmup iterations.")
+
+    # --- Benchmarking Phase ---
+    print(f"Starting {NUM_ITERATIONS} benchmarking iterations...")
+    start_index = WARMUP_ITERATIONS # Start after warmup samples
+    for i in range(NUM_ITERATIONS):
+        sample_index = start_index + i
+        img_np, label_np = test_samples[sample_index]
+
+        # Time ONLY the inference call
+        time_start = time.time()
+        outputs = onnx_cpu_session.run(None, {input_name: img_np})
+        time_end = time.time()
+
+        # Accumulate time
+        total_time += (time_end - time_start)
+
+        # Process prediction (outside timing)
+        pred = np.argmax(outputs[0], axis=1)
+        if pred == label_np:
+            correct_predictions += 1
+
+    print(f"Completed {NUM_ITERATIONS} benchmarking iterations.")
+
+    #########################################
+    # RESULTS REPORTING #####################
+    #########################################
+    print("\n======= CPU BENCHMARK RESULTS =======")
+    if NUM_ITERATIONS > 0:
+        avg_time_ms = (total_time / NUM_ITERATIONS) * 1000
+        accuracy = (correct_predictions / NUM_ITERATIONS) * 100
+        throughput = 1.0 / (total_time / NUM_ITERATIONS) if total_time > 0 else 0
+
+        print(f"Execution Provider: CPU")
+        print(f"Total Samples: {NUM_ITERATIONS}")
+        print(f"Accuracy: {accuracy:.2f}%")
+        print(f"Avg. Inference Time: {avg_time_ms:.3f} ms")
+        print(f"Throughput: {throughput:.2f} inferences/second")
+    else:
+        print("No benchmark iterations run.")
+
+    print(f"\nOptimized ONNX model saved at: {OPTIMIZED_ONNX_FP32_PATH}")
+    print("Script finished.")
\ No newline at end of file
diff --git a/src/matmul_relu_matmul_fashion_mnist/test_gemm.c b/src/matmul_relu_matmul_fashion_mnist/test_gemm.c
new file mode 100644
index 0000000..a4887b4
--- /dev/null
+++ b/src/matmul_relu_matmul_fashion_mnist/test_gemm.c
@@ -0,0 +1,108 @@
+#include <stdio.h>
+#include <stdint.h> // For int64_t
+#include <stdlib.h> // For malloc/free (optional, could use stack arrays)
+
+// Declare the external function from FusedGemmRuntime.o
+extern void ort_cpu_ep_fused_gemm(
+    const float* A,
+    const float* B,
+    const float* Bias,
+    float* Y,
+    int64_t M,
+    int64_t N,
+    int64_t K,
+    int64_t transA,
+    int64_t transB
+);
+
+// Helper function to print a matrix
+void print_matrix(const char* name, const float* matrix, int64_t rows, int64_t cols) {
+    printf("%s (%lld x %lld):\n", name, (long long)rows, (long long)cols);
+    for (int64_t i = 0; i < rows; ++i) {
+        printf("  [");
+        for (int64_t j = 0; j < cols; ++j) {
+            printf("%8.3f", matrix[i * cols + j]);
+            if (j < cols - 1) printf(", ");
+        }
+        printf("]\n");
+    }
+    printf("\n");
+}
+
+int main() {
+    // --- Define Sample Data ---
+    // Example: A (2x3) @ B (3x2) + Bias (2) -> Y (2x2)
+    // No transpose for simplicity first (transA=0, transB=0)
+
+    const int64_t M = 2;
+    const int64_t N = 2;
+    const int64_t K = 3;
+    const int64_t transA = 0;
+    const int64_t transB = 0;
+
+    // Matrix A (M x K) = (2 x 3)
+    float A[] = {
+        1.0f, 2.0f, 3.0f,  // Row 0
+        4.0f, 5.0f, 6.0f   // Row 1
+    };
+
+    // Matrix B (K x N) = (3 x 2)
+    float B[] = {
+        7.0f,  8.0f,   // Row 0
+        9.0f, 10.0f,   // Row 1
+       11.0f, 12.0f    // Row 2
+    };
+
+    // Bias (N) = (2)
+    float Bias[] = { 0.1f, -0.2f };
+
+    // Output Matrix Y (M x N) = (2 x 2) - Allocate space
+    float Y[M * N]; // Use stack allocation for small example
+
+    printf("--- Input Data ---\n");
+    print_matrix("Matrix A", A, M, K);
+    print_matrix("Matrix B", B, K, N);
+    print_matrix("Bias", Bias, 1, N); // Print bias as a row vector
+
+    // --- Call the Fused GEMM function ---
+    printf("--- Calling ort_cpu_ep_fused_gemm ---\n");
+    ort_cpu_ep_fused_gemm(A, B, Bias, Y, M, N, K, transA, transB);
+    printf("--- Returned from ort_cpu_ep_fused_gemm ---\n\n");
+
+    // --- Print the Result ---
+    print_matrix("Result Y", Y, M, N);
+
+    // --- Expected Result Calculation (Manual for verification) ---
+    // Y[0,0] = relu((1*7 + 2*9 + 3*11) + 0.1) = relu(7 + 18 + 33 + 0.1) = relu(58.1) = 58.1
+    // Y[0,1] = relu((1*8 + 2*10 + 3*12) - 0.2) = relu(8 + 20 + 36 - 0.2) = relu(63.8) = 63.8
+    // Y[1,0] = relu((4*7 + 5*9 + 6*11) + 0.1) = relu(28 + 45 + 66 + 0.1) = relu(139.1) = 139.1
+    // Y[1,1] = relu((4*8 + 5*10 + 6*12) - 0.2) = relu(32 + 50 + 72 - 0.2) = relu(153.8) = 153.8
+    printf("--- Expected Result (Manual Calculation) ---\n");
+    printf("  [  58.100,   63.800]\n");
+    printf("  [ 139.100,  153.800]\n\n");
+
+
+    // --- Test with Transpose B ---
+    // A (2x3) @ B' (2x3) -> Y (2x2) ? This doesn't match dimensions.
+    // Let's redefine B to be (N x K) = (2 x 3) so B' is (K x N) = (3 x 2)
+    printf("--- Testing Transpose B ---\n");
+    const int64_t transB_test = 1;
+    float B_t[] = { // B is now (N x K) = (2 x 3)
+        7.0f, 9.0f, 11.0f, // Represents column 0 of original B
+        8.0f, 10.0f, 12.0f // Represents column 1 of original B
+    };
+    print_matrix("Matrix A", A, M, K);
+    print_matrix("Matrix B (Layout for Transpose)", B_t, N, K); // Note dimensions N, K
+    print_matrix("Bias", Bias, 1, N);
+
+    printf("--- Calling ort_cpu_ep_fused_gemm (transB=1) ---\n");
+    ort_cpu_ep_fused_gemm(A, B_t, Bias, Y, M, N, K, transA, transB_test);
+    printf("--- Returned from ort_cpu_ep_fused_gemm ---\n\n");
+    print_matrix("Result Y (transB=1)", Y, M, N);
+    printf("--- Expected Result (Should be same as before) ---\n");
+    printf("  [  58.100,   63.800]\n");
+    printf("  [ 139.100,  153.800]\n\n");
+
+
+    return 0;
+}
\ No newline at end of file
