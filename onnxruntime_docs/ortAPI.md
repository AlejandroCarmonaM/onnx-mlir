OrtApi Struct Reference

#include <onnxruntime_c_api.h>

Public Member Functions
OrtStatus * 	SynchronizeBoundInputs (OrtIoBinding *binding_ptr)
 	Synchronize bound inputs. The call may be necessary for some providers, such as cuda, in case the system that allocated bound memory operated on a different stream. However, the operation is provider specific and could be a no-op.
 
OrtStatus * 	SynchronizeBoundOutputs (OrtIoBinding *binding_ptr)
 	Synchronize bound outputs. The call may be necessary for some providers, such as cuda, in case the system that allocated bound memory operated on a different stream. However, the operation is provider specific and could be a no-op.
 
OrtStatus * 	SessionOptionsAppendExecutionProvider_MIGraphX (OrtSessionOptions *options, const OrtMIGraphXProviderOptions *migraphx_options)
 	Append MIGraphX provider to session options.
 
OrtStatus * 	AddExternalInitializers (OrtSessionOptions *options, const char *const *initializer_names, const OrtValue *const *initializers, size_t num_initializers)
 	Replace initialized Tensors with external data with the data provided in initializers.
 
OrtStatus * 	CreateOpAttr (const char *name, const void *data, int len, OrtOpAttrType type, OrtOpAttr **op_attr)
 	: Create attribute of onnxruntime operator
 
void 	ReleaseOpAttr (OrtOpAttr *input)
 
OrtStatus * 	CreateOp (const OrtKernelInfo *info, const char *op_name, const char *domain, int version, const char **type_constraint_names, const ONNXTensorElementDataType *type_constraint_values, int type_constraint_count, const OrtOpAttr *const *attr_values, int attr_count, int input_count, int output_count, OrtOp **ort_op)
 	: Create onnxruntime native operator
 
OrtStatus * 	InvokeOp (const OrtKernelContext *context, const OrtOp *ort_op, const OrtValue *const *input_values, int input_count, OrtValue *const *output_values, int output_count)
 	: Invoke the operator created by OrtApi::CreateOp The inputs must follow the order as specified in onnx specification
 
void 	ReleaseOp (OrtOp *input)
 
OrtStatus * 	SessionOptionsAppendExecutionProvider (OrtSessionOptions *options, const char *provider_name, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
 	: Append execution provider to the session options.
 
OrtStatus * 	CopyKernelInfo (const OrtKernelInfo *info, OrtKernelInfo **info_copy)
 
void 	ReleaseKernelInfo (OrtKernelInfo *input)
 
OrtStatus * 	SessionOptionsAppendExecutionProvider_CANN (OrtSessionOptions *options, const OrtCANNProviderOptions *cann_options)
 	Append CANN provider to session options.
 
OrtStatus * 	CreateCANNProviderOptions (OrtCANNProviderOptions **out)
 	Create an OrtCANNProviderOptions.
 
OrtStatus * 	UpdateCANNProviderOptions (OrtCANNProviderOptions *cann_options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
 	Set options in a CANN Execution Provider.
 
OrtStatus * 	GetCANNProviderOptionsAsString (const OrtCANNProviderOptions *cann_options, OrtAllocator *allocator, char **ptr)
 	Get serialized CANN provider options string.
 
OrtStatus * 	UpdateEnvWithCustomLogLevel (OrtEnv *ort_env, OrtLoggingLevel log_severity_level)
 
OrtStatus * 	SetGlobalIntraOpThreadAffinity (OrtThreadingOptions *tp_options, const char *affinity_string)
 
OrtStatus * 	RegisterCustomOpsLibrary_V2 (OrtSessionOptions *options, const char *library_name)
 	Register custom ops from a shared library.
 
OrtStatus * 	RegisterCustomOpsUsingFunction (OrtSessionOptions *options, const char *registration_func_name)
 	Register custom ops by calling a RegisterCustomOpsFn function.
 
OrtStatus * 	SessionOptionsAppendExecutionProvider_Dnnl (OrtSessionOptions *options, const OrtDnnlProviderOptions *dnnl_options)
 	Append dnnl provider to session options.
 
OrtStatus * 	CreateDnnlProviderOptions (OrtDnnlProviderOptions **out)
 	Create an OrtDnnlProviderOptions.
 
OrtStatus * 	UpdateDnnlProviderOptions (OrtDnnlProviderOptions *dnnl_options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
 	Set options in a oneDNN Execution Provider.
 
OrtStatus * 	GetDnnlProviderOptionsAsString (const OrtDnnlProviderOptions *dnnl_options, OrtAllocator *allocator, char **ptr)
 
OrtStatus * 	KernelInfoGetConstantInput_tensor (const OrtKernelInfo *info, size_t index, int *is_constant, const OrtValue **out)
 	Get a OrtValue tensor stored as a constant initializer in the graph node.
 
OrtStatus * 	CastTypeInfoToOptionalTypeInfo (const OrtTypeInfo *type_info, const OrtOptionalTypeInfo **out)
 	Get Optional Type information from an OrtTypeInfo.
 
OrtStatus * 	GetOptionalContainedTypeInfo (const OrtOptionalTypeInfo *optional_type_info, OrtTypeInfo **out)
 	Get OrtTypeInfo for the allowed contained type from an OrtOptionalTypeInfo.
 
OrtStatus * 	GetResizedStringTensorElementBuffer (OrtValue *value, size_t index, size_t length_in_bytes, char **buffer)
 	Set a single string in a string tensor Do not zero terminate the string data.
 
OrtStatus * 	KernelContext_GetAllocator (const OrtKernelContext *context, const OrtMemoryInfo *mem_info, OrtAllocator **out)
 	Get Allocator from KernelContext for a specific memoryInfo. Please use C API ReleaseAllocator to release out object.
 
Public Attributes
void(* 	ReleaseCANNProviderOptions )(OrtCANNProviderOptions *input)
 	Release an OrtCANNProviderOptions.
 
void(* 	MemoryInfoGetDeviceType )(const OrtMemoryInfo *ptr, OrtMemoryInfoDeviceType *out)
 
void(* 	ReleaseDnnlProviderOptions )(OrtDnnlProviderOptions *input)
 	Release an OrtDnnlProviderOptions.
 
const char *(* 	GetBuildInfoString )(void)
 	Returns a null terminated string of the build info including git info and cxx flags.
 
OrtStatus
OrtStatus *(* 	CreateStatus )(OrtErrorCode code, const char *msg) __attribute__((nonnull))
 	Create an OrtStatus from a null terminated string.
 
OrtErrorCode(* 	GetErrorCode )(const OrtStatus *status) __attribute__((nonnull))
 	Get OrtErrorCode from OrtStatus.
 
const char *(* 	GetErrorMessage )(const OrtStatus *status) __attribute__((nonnull))
 	Get error string from OrtStatus.
 
void 	ReleaseStatus (OrtStatus *input)
 
OrtIoBinding
void(* 	ClearBoundInputs )(OrtIoBinding *binding_ptr) __attribute__((nonnull))
 	Clears any previously set Inputs for an OrtIoBinding.
 
void(* 	ClearBoundOutputs )(OrtIoBinding *binding_ptr) __attribute__((nonnull))
 	Clears any previously set Outputs for an OrtIoBinding.
 
void 	ReleaseIoBinding (OrtIoBinding *input)
 	Release an OrtIoBinding obtained from OrtApi::CreateIoBinding.
 
OrtStatus * 	BindInput (OrtIoBinding *binding_ptr, const char *name, const OrtValue *val_ptr)
 	Bind an OrtValue to an OrtIoBinding input.
 
OrtStatus * 	BindOutput (OrtIoBinding *binding_ptr, const char *name, const OrtValue *val_ptr)
 	Bind an OrtValue to an OrtIoBinding output.
 
OrtStatus * 	BindOutputToDevice (OrtIoBinding *binding_ptr, const char *name, const OrtMemoryInfo *mem_info_ptr)
 	Bind an OrtIoBinding output to a device.
 
OrtStatus * 	GetBoundOutputNames (const OrtIoBinding *binding_ptr, OrtAllocator *allocator, char **buffer, size_t **lengths, size_t *count)
 	Get the names of an OrtIoBinding's outputs.
 
OrtStatus * 	GetBoundOutputValues (const OrtIoBinding *binding_ptr, OrtAllocator *allocator, OrtValue ***output, size_t *output_count)
 	Get the output OrtValue objects from an OrtIoBinding.
 
OrtTensorRTProviderOptionsV2
void(* 	ReleaseTensorRTProviderOptions )(OrtTensorRTProviderOptionsV2 *input)
 	Release an OrtTensorRTProviderOptionsV2.
 
OrtStatus * 	CreateTensorRTProviderOptions (OrtTensorRTProviderOptionsV2 **out)
 	Create an OrtTensorRTProviderOptionsV2.
 
OrtStatus * 	UpdateTensorRTProviderOptions (OrtTensorRTProviderOptionsV2 *tensorrt_options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
 	Set options in a TensorRT Execution Provider.
 
OrtStatus * 	GetTensorRTProviderOptionsAsString (const OrtTensorRTProviderOptionsV2 *tensorrt_options, OrtAllocator *allocator, char **ptr)
 	Get serialized TensorRT provider options string.
 
OrtCUDAProviderOptionsV2
void(* 	ReleaseCUDAProviderOptions )(OrtCUDAProviderOptionsV2 *input)
 	Release an OrtCUDAProviderOptionsV2.
 
OrtStatus * 	CreateCUDAProviderOptions (OrtCUDAProviderOptionsV2 **out)
 	Create an OrtCUDAProviderOptionsV2.
 
OrtStatus * 	UpdateCUDAProviderOptions (OrtCUDAProviderOptionsV2 *cuda_options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
 	Set options in a CUDA Execution Provider.
 
OrtStatus * 	GetCUDAProviderOptionsAsString (const OrtCUDAProviderOptionsV2 *cuda_options, OrtAllocator *allocator, char **ptr)
 
Ort Training
const OrtTrainingApi *(* 	GetTrainingApi )(uint32_t version)
 	Gets the Training C Api struct.
 
OrtROCMProviderOptions
void(* 	ReleaseROCMProviderOptions )(OrtROCMProviderOptions *input)
 	Release an OrtROCMProviderOptions.
 
OrtStatus * 	CreateROCMProviderOptions (OrtROCMProviderOptions **out)
 	Create an OrtROCMProviderOptions.
 
OrtStatus * 	UpdateROCMProviderOptions (OrtROCMProviderOptions *rocm_options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
 	Set options in a ROCm Execution Provider.
 
OrtStatus * 	GetROCMProviderOptionsAsString (const OrtROCMProviderOptions *rocm_options, OrtAllocator *allocator, char **ptr)
 
OrtStatus * 	CreateAndRegisterAllocatorV2 (OrtEnv *env, const char *provider_type, const OrtMemoryInfo *mem_info, const OrtArenaCfg *arena_cfg, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
 	Create an allocator with specific type and register it with the OrtEnv This API enhance CreateAndRegisterAllocator that it can create an allocator with specific type, not just CPU allocator Enables sharing the allocator between multiple sessions that use the same env instance. Lifetime of the created allocator will be valid for the duration of the environment. Returns an error if an allocator with the same OrtMemoryInfo is already registered.
 
OrtStatus * 	RunAsync (OrtSession *session, const OrtRunOptions *run_options, const char *const *input_names, const OrtValue *const *input, size_t input_len, const char *const *output_names, size_t output_names_len, OrtValue **output, RunAsyncCallbackFn run_async_callback, void *user_data)
 	Run the model asynchronously in a thread owned by intra op thread pool.
 
OrtStatus * 	UpdateTensorRTProviderOptionsWithValue (OrtTensorRTProviderOptionsV2 *tensorrt_options, const char *key, void *value)
 
OrtStatus * 	GetTensorRTProviderOptionsByName (const OrtTensorRTProviderOptionsV2 *tensorrt_options, const char *key, void **ptr)
 
OrtStatus * 	UpdateCUDAProviderOptionsWithValue (OrtCUDAProviderOptionsV2 *cuda_options, const char *key, void *value)
 
OrtStatus * 	GetCUDAProviderOptionsByName (const OrtCUDAProviderOptionsV2 *cuda_options, const char *key, void **ptr)
 
OrtStatus * 	KernelContext_GetResource (const OrtKernelContext *context, int resource_version, int resource_id, void **resource)
 
OrtStatus * 	SetUserLoggingFunction (OrtSessionOptions *options, OrtLoggingFunction user_logging_function, void *user_logging_param)
 	Set user logging function.
 
OrtStatus * 	ShapeInferContext_GetInputCount (const OrtShapeInferContext *context, size_t *out)
 
OrtStatus * 	ShapeInferContext_GetInputTypeShape (const OrtShapeInferContext *context, size_t index, OrtTensorTypeAndShapeInfo **info)
 
OrtStatus * 	ShapeInferContext_GetAttribute (const OrtShapeInferContext *context, const char *attr_name, const OrtOpAttr **attr)
 
OrtStatus * 	ShapeInferContext_SetOutputTypeShape (const OrtShapeInferContext *context, size_t index, const OrtTensorTypeAndShapeInfo *info)
 
OrtStatus * 	SetSymbolicDimensions (OrtTensorTypeAndShapeInfo *info, const char *dim_params[], size_t dim_params_length)
 
OrtStatus * 	ReadOpAttr (const OrtOpAttr *op_attr, OrtOpAttrType type, void *data, size_t len, size_t *out)
 
OrtStatus * 	SetDeterministicCompute (OrtSessionOptions *options, bool value)
 	Set whether to use deterministic compute.
 
OrtStatus * 	KernelContext_ParallelFor (const OrtKernelContext *context, void(*fn)(void *, size_t), size_t total, size_t num_batch, void *usr_data)
 
OrtStatus * 	SessionOptionsAppendExecutionProvider_OpenVINO_V2 (OrtSessionOptions *options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
 	Append OpenVINO execution provider to the session options.
 
OrtStatus * 	SessionOptionsAppendExecutionProvider_VitisAI (OrtSessionOptions *options, const char *const *provider_options_keys, const char *const *provider_options_values, size_t num_keys)
 	Append VitisAI provider to session options.
 
OrtStatus * 	KernelContext_GetScratchBuffer (const OrtKernelContext *context, const OrtMemoryInfo *mem_info, size_t count_or_bytes, void **out)
 	Get scratch buffer from the corresponding allocator under the sepcific OrtMemoryInfo object. NOTE: callers are responsible to release this scratch buffer from the corresponding allocator.
 
OrtStatus * 	KernelInfoGetAllocator (const OrtKernelInfo *info, OrtMemType mem_type, OrtAllocator **out)
 	Get allocator from KernelInfo for a specific memory type. Please use C API ReleaseAllocator to release out object.
 
OrtStatus * 	AddExternalInitializersFromFilesInMemory (OrtSessionOptions *options, const char *const *external_initializer_file_names, char *const *external_initializer_file_buffer_array, const size_t *external_initializer_file_lengths, size_t num_external_initializer_files)
 	Replace initialized Tensors with external data with the provided files in memory.
 
OrtStatus * 	CreateLoraAdapter (const char *adapter_file_path, OrtAllocator *allocator, OrtLoraAdapter **out)
 	Create an OrtLoraAdapter.
 
OrtStatus * 	CreateLoraAdapterFromArray (const void *bytes, size_t num_bytes, OrtAllocator *allocator, OrtLoraAdapter **out)
 	Create an OrtLoraAdapter.
 
void 	ReleaseLoraAdapter (OrtLoraAdapter *input)
 	Release an OrtLoraAdapter obtained from OrtApi::CreateLoraAdapter.
 
OrtStatus * 	RunOptionsAddActiveLoraAdapter (OrtRunOptions *options, const OrtLoraAdapter *adapter)
 	Add the Lora Adapter to the list of active adapters.
 
OrtEnv
OrtStatus * 	CreateEnv (OrtLoggingLevel log_severity_level, const char *logid, OrtEnv **out)
 	Create an OrtEnv.
 
OrtStatus * 	CreateEnvWithCustomLogger (OrtLoggingFunction logging_function, void *logger_param, OrtLoggingLevel log_severity_level, const char *logid, OrtEnv **out)
 	Create an OrtEnv.
 
OrtStatus * 	EnableTelemetryEvents (const OrtEnv *env)
 	Enable Telemetry.
 
OrtStatus * 	DisableTelemetryEvents (const OrtEnv *env)
 	Disable Telemetry.
 
void 	ReleaseEnv (OrtEnv *input)
 
OrtStatus * 	CreateEnvWithGlobalThreadPools (OrtLoggingLevel log_severity_level, const char *logid, const OrtThreadingOptions *tp_options, OrtEnv **out)
 	Create an OrtEnv.
 
OrtStatus * 	CreateAndRegisterAllocator (OrtEnv *env, const OrtMemoryInfo *mem_info, const OrtArenaCfg *arena_cfg)
 	Create an allocator and register it with the OrtEnv.
 
OrtStatus * 	SetLanguageProjection (const OrtEnv *ort_env, OrtLanguageProjection projection)
 	Set language projection.
 
OrtStatus * 	CreateEnvWithCustomLoggerAndGlobalThreadPools (OrtLoggingFunction logging_function, void *logger_param, OrtLoggingLevel log_severity_level, const char *logid, const struct OrtThreadingOptions *tp_options, OrtEnv **out)
 
OrtSession
OrtStatus * 	CreateSession (const OrtEnv *env, const char *model_path, const OrtSessionOptions *options, OrtSession **out)
 	Create an OrtSession from a model file.
 
OrtStatus * 	CreateSessionFromArray (const OrtEnv *env, const void *model_data, size_t model_data_length, const OrtSessionOptions *options, OrtSession **out)
 	Create an OrtSession from memory.
 
OrtStatus * 	Run (OrtSession *session, const OrtRunOptions *run_options, const char *const *input_names, const OrtValue *const *inputs, size_t input_len, const char *const *output_names, size_t output_names_len, OrtValue **outputs)
 	Run the model in an OrtSession.
 
OrtStatus * 	SessionGetInputCount (const OrtSession *session, size_t *out)
 	Get input count for a session.
 
OrtStatus * 	SessionGetOutputCount (const OrtSession *session, size_t *out)
 	Get output count for a session.
 
OrtStatus * 	SessionGetOverridableInitializerCount (const OrtSession *session, size_t *out)
 	Get overridable initializer count.
 
OrtStatus * 	SessionGetInputTypeInfo (const OrtSession *session, size_t index, OrtTypeInfo **type_info)
 	Get input type information.
 
OrtStatus * 	SessionGetOutputTypeInfo (const OrtSession *session, size_t index, OrtTypeInfo **type_info)
 	Get output type information.
 
OrtStatus * 	SessionGetOverridableInitializerTypeInfo (const OrtSession *session, size_t index, OrtTypeInfo **type_info)
 	Get overridable initializer type information.
 
OrtStatus * 	SessionGetInputName (const OrtSession *session, size_t index, OrtAllocator *allocator, char **value)
 	Get input name.
 
OrtStatus * 	SessionGetOutputName (const OrtSession *session, size_t index, OrtAllocator *allocator, char **value)
 	Get output name.
 
OrtStatus * 	SessionGetOverridableInitializerName (const OrtSession *session, size_t index, OrtAllocator *allocator, char **value)
 	Get overridable initializer name.
 
void 	ReleaseSession (OrtSession *input)
 
OrtStatus * 	SessionEndProfiling (OrtSession *session, OrtAllocator *allocator, char **out)
 	End profiling and return filename of the profile data.
 
OrtStatus * 	SessionGetModelMetadata (const OrtSession *session, OrtModelMetadata **out)
 	Get OrtModelMetadata from an OrtSession.
 
OrtStatus * 	RunWithBinding (OrtSession *session, const OrtRunOptions *run_options, const OrtIoBinding *binding_ptr)
 	Run a model using Io Bindings for the inputs & outputs.
 
OrtStatus * 	CreateIoBinding (OrtSession *session, OrtIoBinding **out)
 	Create an OrtIoBinding instance.
 
OrtStatus * 	SessionGetProfilingStartTimeNs (const OrtSession *session, uint64_t *out)
 	Return the time that profiling was started.
 
OrtStatus * 	CreateSessionWithPrepackedWeightsContainer (const OrtEnv *env, const char *model_path, const OrtSessionOptions *options, OrtPrepackedWeightsContainer *prepacked_weights_container, OrtSession **out)
 	Create session with prepacked weights container.
 
OrtStatus * 	CreateSessionFromArrayWithPrepackedWeightsContainer (const OrtEnv *env, const void *model_data, size_t model_data_length, const OrtSessionOptions *options, OrtPrepackedWeightsContainer *prepacked_weights_container, OrtSession **out)
 	Create session from memory with prepacked weights container.
 
OrtSessionOptions
Custom operator APIs

OrtStatus * 	CreateSessionOptions (OrtSessionOptions **options)
 	Create an OrtSessionOptions object.
 
OrtStatus * 	SetOptimizedModelFilePath (OrtSessionOptions *options, const char *optimized_model_filepath)
 	Set filepath to save optimized model after graph level transformations.
 
OrtStatus * 	CloneSessionOptions (const OrtSessionOptions *in_options, OrtSessionOptions **out_options)
 	Create a copy of an existing OrtSessionOptions.
 
OrtStatus * 	SetSessionExecutionMode (OrtSessionOptions *options, ExecutionMode execution_mode)
 	Set execution mode.
 
OrtStatus * 	EnableProfiling (OrtSessionOptions *options, const char *profile_file_prefix)
 	Enable profiling for a session.
 
OrtStatus * 	DisableProfiling (OrtSessionOptions *options)
 	Disable profiling for a session.
 
OrtStatus * 	EnableMemPattern (OrtSessionOptions *options)
 	Enable the memory pattern optimization.
 
OrtStatus * 	DisableMemPattern (OrtSessionOptions *options)
 	Disable the memory pattern optimization.
 
OrtStatus * 	EnableCpuMemArena (OrtSessionOptions *options)
 	Enable the memory arena on CPU.
 
OrtStatus * 	DisableCpuMemArena (OrtSessionOptions *options)
 	Disable the memory arena on CPU.
 
OrtStatus * 	SetSessionLogId (OrtSessionOptions *options, const char *logid)
 	Set session log id.
 
OrtStatus * 	SetSessionLogVerbosityLevel (OrtSessionOptions *options, int session_log_verbosity_level)
 	Set session log verbosity level.
 
OrtStatus * 	SetSessionLogSeverityLevel (OrtSessionOptions *options, int session_log_severity_level)
 	Set session log severity level.
 
OrtStatus * 	SetSessionGraphOptimizationLevel (OrtSessionOptions *options, GraphOptimizationLevel graph_optimization_level)
 	Set the optimization level to apply when loading a graph.
 
OrtStatus * 	SetIntraOpNumThreads (OrtSessionOptions *options, int intra_op_num_threads)
 	Sets the number of threads used to parallelize the execution within nodes.
 
OrtStatus * 	SetInterOpNumThreads (OrtSessionOptions *options, int inter_op_num_threads)
 	Sets the number of threads used to parallelize the execution of the graph.
 
OrtStatus * 	AddCustomOpDomain (OrtSessionOptions *options, OrtCustomOpDomain *custom_op_domain)
 	Add custom op domain to a session options.
 
OrtStatus * 	RegisterCustomOpsLibrary (OrtSessionOptions *options, const char *library_path, void **library_handle)
 
OrtStatus * 	AddFreeDimensionOverride (OrtSessionOptions *options, const char *dim_denotation, int64_t dim_value)
 	Override session symbolic dimensions.
 
void 	ReleaseSessionOptions (OrtSessionOptions *input)
 
OrtStatus * 	DisablePerSessionThreads (OrtSessionOptions *options)
 	Use global thread pool on a session.
 
OrtStatus * 	AddFreeDimensionOverrideByName (OrtSessionOptions *options, const char *dim_name, int64_t dim_value)
 
OrtStatus * 	AddSessionConfigEntry (OrtSessionOptions *options, const char *config_key, const char *config_value)
 	Set a session configuration entry as a pair of strings.
 
OrtStatus * 	AddInitializer (OrtSessionOptions *options, const char *name, const OrtValue *val)
 	Add a pre-allocated initializer to a session.
 
OrtStatus * 	SessionOptionsAppendExecutionProvider_CUDA (OrtSessionOptions *options, const OrtCUDAProviderOptions *cuda_options)
 	Append CUDA provider to session options.
 
OrtStatus * 	SessionOptionsAppendExecutionProvider_ROCM (OrtSessionOptions *options, const OrtROCMProviderOptions *rocm_options)
 	Append ROCM execution provider to the session options.
 
OrtStatus * 	SessionOptionsAppendExecutionProvider_OpenVINO (OrtSessionOptions *options, const OrtOpenVINOProviderOptions *provider_options)
 	Append OpenVINO execution provider to the session options.
 
OrtStatus * 	SessionOptionsAppendExecutionProvider_TensorRT (OrtSessionOptions *options, const OrtTensorRTProviderOptions *tensorrt_options)
 	Append TensorRT provider to session options.
 
OrtStatus * 	SessionOptionsAppendExecutionProvider_TensorRT_V2 (OrtSessionOptions *options, const OrtTensorRTProviderOptionsV2 *tensorrt_options)
 	Append TensorRT execution provider to the session options.
 
OrtStatus * 	EnableOrtCustomOps (OrtSessionOptions *options)
 	Enable custom operators.
 
OrtStatus * 	HasValue (const OrtValue *value, int *out)
 	Sets out to 1 iff an optional type OrtValue has an element, 0 otherwise (OrtValue is None) Use this API to find if the optional type OrtValue is None or not. If the optional type OrtValue is not None, use the OrtValue just like any other OrtValue. For example, if you get an OrtValue that corresponds to Optional(tensor) and if HasValue() returns true, use it as tensor and so on.
 
OrtStatus * 	SessionOptionsAppendExecutionProvider_CUDA_V2 (OrtSessionOptions *options, const OrtCUDAProviderOptionsV2 *cuda_options)
 	Append CUDA execution provider to the session options.
 
OrtStatus * 	HasSessionConfigEntry (const OrtSessionOptions *options, const char *config_key, int *out)
 	Checks if the given session configuration entry exists.
 
OrtStatus * 	GetSessionConfigEntry (const OrtSessionOptions *options, const char *config_key, char *config_value, size_t *size)
 	Get a session configuration value.
 
OrtCustomOpDomain
OrtStatus * 	CreateCustomOpDomain (const char *domain, OrtCustomOpDomain **out)
 	Create a custom op domain.
 
OrtStatus * 	CustomOpDomain_Add (OrtCustomOpDomain *custom_op_domain, const OrtCustomOp *op)
 	Add a custom op to a custom op domain.
 
void 	ReleaseCustomOpDomain (OrtCustomOpDomain *input)
 
OrtRunOptions
OrtStatus * 	CreateRunOptions (OrtRunOptions **out)
 	Create an OrtRunOptions.
 
OrtStatus * 	RunOptionsSetRunLogVerbosityLevel (OrtRunOptions *options, int log_verbosity_level)
 	Set per-run log verbosity level.
 
OrtStatus * 	RunOptionsSetRunLogSeverityLevel (OrtRunOptions *options, int log_severity_level)
 	Set per-run log severity level.
 
OrtStatus * 	RunOptionsSetRunTag (OrtRunOptions *options, const char *run_tag)
 	Set per-run tag.
 
OrtStatus * 	RunOptionsGetRunLogVerbosityLevel (const OrtRunOptions *options, int *log_verbosity_level)
 	Get per-run log verbosity level.
 
OrtStatus * 	RunOptionsGetRunLogSeverityLevel (const OrtRunOptions *options, int *log_severity_level)
 	Get per-run log severity level.
 
OrtStatus * 	RunOptionsGetRunTag (const OrtRunOptions *options, const char **run_tag)
 	Get per-run tag.
 
OrtStatus * 	RunOptionsSetTerminate (OrtRunOptions *options)
 	Set terminate flag.
 
OrtStatus * 	RunOptionsUnsetTerminate (OrtRunOptions *options)
 	Clears the terminate flag.
 
void 	ReleaseRunOptions (OrtRunOptions *input)
 
OrtStatus * 	AddRunConfigEntry (OrtRunOptions *options, const char *config_key, const char *config_value)
 	Set a single run configuration entry as a pair of strings.
 
OrtValue
OrtStatus * 	CreateTensorAsOrtValue (OrtAllocator *allocator, const int64_t *shape, size_t shape_len, ONNXTensorElementDataType type, OrtValue **out)
 	Create a tensor.
 
OrtStatus * 	CreateTensorWithDataAsOrtValue (const OrtMemoryInfo *info, void *p_data, size_t p_data_len, const int64_t *shape, size_t shape_len, ONNXTensorElementDataType type, OrtValue **out)
 	Create a tensor backed by a user supplied buffer.
 
OrtStatus * 	IsTensor (const OrtValue *value, int *out)
 	Return if an OrtValue is a tensor type.
 
OrtStatus * 	GetTensorMutableData (OrtValue *value, void **out)
 	Get a pointer to the raw data inside a tensor.
 
OrtStatus * 	FillStringTensor (OrtValue *value, const char *const *s, size_t s_len)
 	Set all strings at once in a string tensor.
 
OrtStatus * 	GetStringTensorDataLength (const OrtValue *value, size_t *len)
 	Get total byte length for all strings in a string tensor.
 
OrtStatus * 	GetStringTensorContent (const OrtValue *value, void *s, size_t s_len, size_t *offsets, size_t offsets_len)
 	Get all strings from a string tensor.
 
OrtStatus * 	GetTensorTypeAndShape (const OrtValue *value, OrtTensorTypeAndShapeInfo **out)
 	Get type and shape information from a tensor OrtValue.
 
OrtStatus * 	GetTypeInfo (const OrtValue *value, OrtTypeInfo **out)
 	Get type information of an OrtValue.
 
OrtStatus * 	GetValueType (const OrtValue *value, enum ONNXType *out)
 	Get ONNXType of an OrtValue.
 
OrtStatus * 	GetValue (const OrtValue *value, int index, OrtAllocator *allocator, OrtValue **out)
 	Get non tensor data from an OrtValue.
 
OrtStatus * 	GetValueCount (const OrtValue *value, size_t *out)
 	Get non tensor value count from an OrtValue.
 
OrtStatus * 	CreateValue (const OrtValue *const *in, size_t num_values, enum ONNXType value_type, OrtValue **out)
 	Create a map or sequence OrtValue.
 
OrtStatus * 	CreateOpaqueValue (const char *domain_name, const char *type_name, const void *data_container, size_t data_container_size, OrtValue **out)
 	Create an opaque (custom user defined type) OrtValue.
 
OrtStatus * 	GetOpaqueValue (const char *domain_name, const char *type_name, const OrtValue *in, void *data_container, size_t data_container_size)
 	Get internal data from an opaque (custom user defined type) OrtValue.
 
void 	ReleaseValue (OrtValue *input)
 
OrtStatus * 	GetStringTensorElementLength (const OrtValue *value, size_t index, size_t *out)
 	Get the length of a single string in a string tensor.
 
OrtStatus * 	GetStringTensorElement (const OrtValue *value, size_t s_len, size_t index, void *s)
 	Get a single string from a string tensor.
 
OrtStatus * 	FillStringTensorElement (OrtValue *value, const char *s, size_t index)
 	Set a single string in a string tensor.
 
OrtStatus * 	TensorAt (OrtValue *value, const int64_t *location_values, size_t location_values_count, void **out)
 	Direct memory access to a specified tensor element.
 
OrtStatus * 	IsSparseTensor (const OrtValue *value, int *out)
 	Sets *out to 1 iff an OrtValue is a SparseTensor, and 0 otherwise.
 
OrtStatus * 	CreateSparseTensorAsOrtValue (OrtAllocator *allocator, const int64_t *dense_shape, size_t dense_shape_len, ONNXTensorElementDataType type, OrtValue **out)
 	Create an OrtValue with a sparse tensor that is empty.
 
OrtStatus * 	FillSparseTensorCoo (OrtValue *ort_value, const OrtMemoryInfo *data_mem_info, const int64_t *values_shape, size_t values_shape_len, const void *values, const int64_t *indices_data, size_t indices_num)
 
OrtStatus * 	FillSparseTensorCsr (OrtValue *ort_value, const OrtMemoryInfo *data_mem_info, const int64_t *values_shape, size_t values_shape_len, const void *values, const int64_t *inner_indices_data, size_t inner_indices_num, const int64_t *outer_indices_data, size_t outer_indices_num)
 
OrtStatus * 	FillSparseTensorBlockSparse (OrtValue *ort_value, const OrtMemoryInfo *data_mem_info, const int64_t *values_shape, size_t values_shape_len, const void *values, const int64_t *indices_shape_data, size_t indices_shape_len, const int32_t *indices_data)
 
OrtStatus * 	CreateSparseTensorWithValuesAsOrtValue (const OrtMemoryInfo *info, void *p_data, const int64_t *dense_shape, size_t dense_shape_len, const int64_t *values_shape, size_t values_shape_len, ONNXTensorElementDataType type, OrtValue **out)
 
OrtStatus * 	UseCooIndices (OrtValue *ort_value, int64_t *indices_data, size_t indices_num)
 
OrtStatus * 	UseCsrIndices (OrtValue *ort_value, int64_t *inner_data, size_t inner_num, int64_t *outer_data, size_t outer_num)
 
OrtStatus * 	UseBlockSparseIndices (OrtValue *ort_value, const int64_t *indices_shape, size_t indices_shape_len, int32_t *indices_data)
 
OrtStatus * 	GetSparseTensorFormat (const OrtValue *ort_value, enum OrtSparseFormat *out)
 	Returns sparse tensor format enum iff a given ort value contains an instance of sparse tensor.
 
OrtStatus * 	GetSparseTensorValuesTypeAndShape (const OrtValue *ort_value, OrtTensorTypeAndShapeInfo **out)
 	Returns data type and shape of sparse tensor values (nnz) iff OrtValue contains a SparseTensor.
 
OrtStatus * 	GetSparseTensorValues (const OrtValue *ort_value, const void **out)
 	Returns numeric data for sparse tensor values (nnz). For string values use GetStringTensor*().
 
OrtStatus * 	GetSparseTensorIndicesTypeShape (const OrtValue *ort_value, enum OrtSparseIndicesFormat indices_format, OrtTensorTypeAndShapeInfo **out)
 	Returns data type, shape for the type of indices specified by indices_format.
 
OrtStatus * 	GetSparseTensorIndices (const OrtValue *ort_value, enum OrtSparseIndicesFormat indices_format, size_t *num_indices, const void **indices)
 	Returns indices data for the type of the indices specified by indices_format.
 
OrtTypeInfo
OrtStatus * 	CastTypeInfoToTensorInfo (const OrtTypeInfo *type_info, const OrtTensorTypeAndShapeInfo **out)
 	Get OrtTensorTypeAndShapeInfo from an OrtTypeInfo.
 
OrtStatus * 	GetOnnxTypeFromTypeInfo (const OrtTypeInfo *type_info, enum ONNXType *out)
 	Get ONNXType from OrtTypeInfo.
 
void 	ReleaseTypeInfo (OrtTypeInfo *input)
 
OrtStatus * 	GetDenotationFromTypeInfo (const OrtTypeInfo *type_info, const char **const denotation, size_t *len)
 	Get denotation from type information.
 
OrtStatus * 	CastTypeInfoToMapTypeInfo (const OrtTypeInfo *type_info, const OrtMapTypeInfo **out)
 	Get detailed map information from an OrtTypeInfo.
 
OrtStatus * 	CastTypeInfoToSequenceTypeInfo (const OrtTypeInfo *type_info, const OrtSequenceTypeInfo **out)
 	Cast OrtTypeInfo to an OrtSequenceTypeInfo.
 
OrtTensorTypeAndShapeInfo
OrtStatus * 	CreateTensorTypeAndShapeInfo (OrtTensorTypeAndShapeInfo **out)
 	Create an OrtTensorTypeAndShapeInfo object.
 
OrtStatus * 	SetTensorElementType (OrtTensorTypeAndShapeInfo *info, enum ONNXTensorElementDataType type)
 	Set element type in OrtTensorTypeAndShapeInfo.
 
OrtStatus * 	SetDimensions (OrtTensorTypeAndShapeInfo *info, const int64_t *dim_values, size_t dim_count)
 	Set shape information in OrtTensorTypeAndShapeInfo.
 
OrtStatus * 	GetTensorElementType (const OrtTensorTypeAndShapeInfo *info, enum ONNXTensorElementDataType *out)
 	Get element type in OrtTensorTypeAndShapeInfo.
 
OrtStatus * 	GetDimensionsCount (const OrtTensorTypeAndShapeInfo *info, size_t *out)
 	Get dimension count in OrtTensorTypeAndShapeInfo.
 
OrtStatus * 	GetDimensions (const OrtTensorTypeAndShapeInfo *info, int64_t *dim_values, size_t dim_values_length)
 	Get dimensions in OrtTensorTypeAndShapeInfo.
 
OrtStatus * 	GetSymbolicDimensions (const OrtTensorTypeAndShapeInfo *info, const char *dim_params[], size_t dim_params_length)
 	Get symbolic dimension names in OrtTensorTypeAndShapeInfo.
 
OrtStatus * 	GetTensorShapeElementCount (const OrtTensorTypeAndShapeInfo *info, size_t *out)
 	Get total number of elements in a tensor shape from an OrtTensorTypeAndShapeInfo.
 
void 	ReleaseTensorTypeAndShapeInfo (OrtTensorTypeAndShapeInfo *input)
 
OrtMemoryInfo
OrtStatus * 	CreateMemoryInfo (const char *name, enum OrtAllocatorType type, int id, enum OrtMemType mem_type, OrtMemoryInfo **out)
 	Create an OrtMemoryInfo.
 
OrtStatus * 	CreateCpuMemoryInfo (enum OrtAllocatorType type, enum OrtMemType mem_type, OrtMemoryInfo **out)
 	Create an OrtMemoryInfo for CPU memory.
 
OrtStatus * 	CompareMemoryInfo (const OrtMemoryInfo *info1, const OrtMemoryInfo *info2, int *out)
 	Compare OrtMemoryInfo objects for equality.
 
OrtStatus * 	MemoryInfoGetName (const OrtMemoryInfo *ptr, const char **out)
 	Get name from OrtMemoryInfo.
 
OrtStatus * 	MemoryInfoGetId (const OrtMemoryInfo *ptr, int *out)
 	Get the id from OrtMemoryInfo.
 
OrtStatus * 	MemoryInfoGetMemType (const OrtMemoryInfo *ptr, OrtMemType *out)
 	Get the OrtMemType from OrtMemoryInfo.
 
OrtStatus * 	MemoryInfoGetType (const OrtMemoryInfo *ptr, OrtAllocatorType *out)
 	Get the OrtAllocatorType from OrtMemoryInfo.
 
void 	ReleaseMemoryInfo (OrtMemoryInfo *input)
 
OrtAllocator
OrtStatus * 	AllocatorAlloc (OrtAllocator *ort_allocator, size_t size, void **out)
 	Calls OrtAllocator::Alloc function.
 
OrtStatus * 	AllocatorFree (OrtAllocator *ort_allocator, void *p)
 	Calls OrtAllocator::Free function.
 
OrtStatus * 	AllocatorGetInfo (const OrtAllocator *ort_allocator, const struct OrtMemoryInfo **out)
 	Calls OrtAllocator::Info function.
 
OrtStatus * 	GetAllocatorWithDefaultOptions (OrtAllocator **out)
 	Get the default allocator.
 
OrtStatus * 	CreateAllocator (const OrtSession *session, const OrtMemoryInfo *mem_info, OrtAllocator **out)
 	Create an allocator for an OrtSession following an OrtMemoryInfo.
 
void 	ReleaseAllocator (OrtAllocator *input)
 	Release an OrtAllocator obtained from OrtApi::CreateAllocator.
 
OrtStatus * 	RegisterAllocator (OrtEnv *env, OrtAllocator *allocator)
 	Register a custom allocator.
 
OrtStatus * 	UnregisterAllocator (OrtEnv *env, const OrtMemoryInfo *mem_info)
 	Unregister a custom allocator.
 
OrtKernelInfo
Custom operator APIs.

OrtStatus * 	KernelInfoGetAttribute_float (const OrtKernelInfo *info, const char *name, float *out)
 	Get a float stored as an attribute in the graph node.
 
OrtStatus * 	KernelInfoGetAttribute_int64 (const OrtKernelInfo *info, const char *name, int64_t *out)
 	Fetch a 64-bit int stored as an attribute in the graph node.
 
OrtStatus * 	KernelInfoGetAttribute_string (const OrtKernelInfo *info, const char *name, char *out, size_t *size)
 	Fetch a string stored as an attribute in the graph node.
 
OrtStatus * 	KernelInfoGetAttributeArray_float (const OrtKernelInfo *info, const char *name, float *out, size_t *size)
 	Fetch an array of int64_t values stored as an attribute in the graph node.
 
OrtStatus * 	KernelInfoGetAttributeArray_int64 (const OrtKernelInfo *info, const char *name, int64_t *out, size_t *size)
 	Fetch an array of int64_t values stored as an attribute in the graph node.
 
OrtStatus * 	KernelInfo_GetInputCount (const OrtKernelInfo *info, size_t *out)
 	Get the number of inputs from OrtKernelInfo.
 
OrtStatus * 	KernelInfo_GetOutputCount (const OrtKernelInfo *info, size_t *out)
 	Get the number of outputs from OrtKernelInfo.
 
OrtStatus * 	KernelInfo_GetInputName (const OrtKernelInfo *info, size_t index, char *out, size_t *size)
 	Get the name of a OrtKernelInfo's input.
 
OrtStatus * 	KernelInfo_GetOutputName (const OrtKernelInfo *info, size_t index, char *out, size_t *size)
 	Get the name of a OrtKernelInfo's output.
 
OrtStatus * 	KernelInfo_GetInputTypeInfo (const OrtKernelInfo *info, size_t index, OrtTypeInfo **type_info)
 	Get the type information for a OrtKernelInfo's input.
 
OrtStatus * 	KernelInfo_GetOutputTypeInfo (const OrtKernelInfo *info, size_t index, OrtTypeInfo **type_info)
 	Get the type information for a OrtKernelInfo's output.
 
OrtStatus * 	KernelInfoGetAttribute_tensor (const OrtKernelInfo *info, const char *name, OrtAllocator *allocator, OrtValue **out)
 	Get a OrtValue tensor stored as an attribute in the graph node.
 
OrtStatus * 	KernelInfo_GetNodeName (const OrtKernelInfo *info, char *out, size_t *size)
 	Get the graph node name from OrtKernelInfo.
 
OrtStatus * 	KernelInfo_GetLogger (const OrtKernelInfo *info, const OrtLogger **logger)
 	Get the session logger from OrtKernelInfo.
 
OrtKernelContext
Custom operator APIs.

OrtStatus * 	KernelContext_GetInputCount (const OrtKernelContext *context, size_t *out)
 	Used for custom operators, get the input count of a kernel.
 
OrtStatus * 	KernelContext_GetOutputCount (const OrtKernelContext *context, size_t *out)
 	Used for custom operators, get the output count of a kernel.
 
OrtStatus * 	KernelContext_GetInput (const OrtKernelContext *context, size_t index, const OrtValue **out)
 	Used for custom operators, get an input of a kernel.
 
OrtStatus * 	KernelContext_GetOutput (OrtKernelContext *context, size_t index, const int64_t *dim_values, size_t dim_count, OrtValue **out)
 	Used for custom operators, get an output of a kernel.
 
OrtStatus * 	KernelContext_GetGPUComputeStream (const OrtKernelContext *context, void **out)
 	Used for custom operators, gets the GPU compute stream to use to launch the custom a GPU kernel.
 
OrtStatus * 	KernelContext_GetLogger (const OrtKernelContext *context, const OrtLogger **logger)
 	Get the runtime logger from OrtKernelContext.
 
OrtMapTypeInfo
OrtStatus * 	GetMapKeyType (const OrtMapTypeInfo *map_type_info, enum ONNXTensorElementDataType *out)
 	Get key type from an OrtMapTypeInfo.
 
OrtStatus * 	GetMapValueType (const OrtMapTypeInfo *map_type_info, OrtTypeInfo **type_info)
 	Get the value type from an OrtMapTypeInfo.
 
void 	ReleaseMapTypeInfo (OrtMapTypeInfo *input)
 
OrtSequenceTypeInfo
OrtStatus * 	GetSequenceElementType (const OrtSequenceTypeInfo *sequence_type_info, OrtTypeInfo **type_info)
 	Get element type from an OrtSequenceTypeInfo.
 
void 	ReleaseSequenceTypeInfo (OrtSequenceTypeInfo *input)
 
OrtModelMetadata
OrtStatus * 	ModelMetadataGetProducerName (const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char **value)
 	Get producer name from an OrtModelMetadata.
 
OrtStatus * 	ModelMetadataGetGraphName (const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char **value)
 	Get graph name from an OrtModelMetadata.
 
OrtStatus * 	ModelMetadataGetDomain (const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char **value)
 	Get domain from an OrtModelMetadata.
 
OrtStatus * 	ModelMetadataGetDescription (const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char **value)
 	Get description from an OrtModelMetadata.
 
OrtStatus * 	ModelMetadataLookupCustomMetadataMap (const OrtModelMetadata *model_metadata, OrtAllocator *allocator, const char *key, char **value)
 	Return data for a key in the custom metadata map in an OrtModelMetadata.
 
OrtStatus * 	ModelMetadataGetVersion (const OrtModelMetadata *model_metadata, int64_t *value)
 	Get version number from an OrtModelMetadata.
 
void 	ReleaseModelMetadata (OrtModelMetadata *input)
 
OrtStatus * 	ModelMetadataGetCustomMetadataMapKeys (const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char ***keys, int64_t *num_keys)
 
OrtStatus * 	ModelMetadataGetGraphDescription (const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char **value)
 
OrtThreadingOptions
OrtStatus * 	CreateThreadingOptions (OrtThreadingOptions **out)
 	Create an OrtThreadingOptions.
 
void 	ReleaseThreadingOptions (OrtThreadingOptions *input)
 
OrtStatus * 	SetGlobalIntraOpNumThreads (OrtThreadingOptions *tp_options, int intra_op_num_threads)
 	Set global intra-op thread count.
 
OrtStatus * 	SetGlobalInterOpNumThreads (OrtThreadingOptions *tp_options, int inter_op_num_threads)
 	Set global inter-op thread count.
 
OrtStatus * 	SetGlobalSpinControl (OrtThreadingOptions *tp_options, int allow_spinning)
 	Set global spin control options.
 
OrtStatus * 	SetGlobalDenormalAsZero (OrtThreadingOptions *tp_options)
 	Set threading flush-to-zero and denormal-as-zero.
 
OrtStatus * 	SetGlobalCustomCreateThreadFn (OrtThreadingOptions *tp_options, OrtCustomCreateThreadFn ort_custom_create_thread_fn)
 	Set custom thread creation function for global thread pools.
 
OrtStatus * 	SetGlobalCustomThreadCreationOptions (OrtThreadingOptions *tp_options, void *ort_custom_thread_creation_options)
 	Set custom thread creation options for global thread pools.
 
OrtStatus * 	SetGlobalCustomJoinThreadFn (OrtThreadingOptions *tp_options, OrtCustomJoinThreadFn ort_custom_join_thread_fn)
 	Set custom thread join function for global thread pools.
 
Misc
OrtStatus * 	GetAvailableProviders (char ***out_ptr, int *provider_length)
 	Get the names of all available providers.
 
OrtStatus * 	ReleaseAvailableProviders (char **ptr, int providers_length)
 	Release data from OrtApi::GetAvailableProviders. This API will never fail so you can rely on it in a noexcept code.
 
OrtStatus * 	SetCurrentGpuDeviceId (int device_id)
 	Set current GPU device ID.
 
OrtStatus * 	GetCurrentGpuDeviceId (int *device_id)
 	Get current GPU device ID.
 
OrtArenaCfg
OrtStatus * 	CreateArenaCfg (size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes, int max_dead_bytes_per_chunk, OrtArenaCfg **out)
 
void 	ReleaseArenaCfg (OrtArenaCfg *input)
 
OrtStatus * 	CreateArenaCfgV2 (const char *const *arena_config_keys, const size_t *arena_config_values, size_t num_keys, OrtArenaCfg **out)
 	Create an OrtArenaCfg.
 
OrtPrepackedWeightsContainer
OrtStatus * 	CreatePrepackedWeightsContainer (OrtPrepackedWeightsContainer **out)
 	Create an OrtPrepackedWeightsContainer.
 
void 	ReleasePrepackedWeightsContainer (OrtPrepackedWeightsContainer *input)
 	Release OrtPrepackedWeightsContainer instance.
 
GetTensorMemoryInfo
OrtStatus * 	GetTensorMemoryInfo (const OrtValue *value, const OrtMemoryInfo **mem_info)
 	Returns a pointer to the OrtMemoryInfo of a Tensor.
 
GetExecutionProviderApi
OrtStatus * 	GetExecutionProviderApi (const char *provider_name, uint32_t version, const void **provider_api)
 	Get a pointer to the requested version of the Execution Provider specific API extensions to the OrtApi.
 
SessionOptions
OrtStatus * 	SessionOptionsSetCustomCreateThreadFn (OrtSessionOptions *options, OrtCustomCreateThreadFn ort_custom_create_thread_fn)
 	Set custom thread creation function.
 
OrtStatus * 	SessionOptionsSetCustomThreadCreationOptions (OrtSessionOptions *options, void *ort_custom_thread_creation_options)
 	Set creation options for custom thread.
 
OrtStatus * 	SessionOptionsSetCustomJoinThreadFn (OrtSessionOptions *options, OrtCustomJoinThreadFn ort_custom_join_thread_fn)
 	Set custom thread join function.
 
OrtLogger
Custom operator APIs.

OrtStatus * 	Logger_LogMessage (const OrtLogger *logger, OrtLoggingLevel log_severity_level, const char *message, const char *file_path, int line_number, const char *func_name)
 	Logs a message at the given severity level using the provided OrtLogger.
 
OrtStatus * 	Logger_GetLoggingSeverityLevel (const OrtLogger *logger, OrtLoggingLevel *out)
 	Get the logging severity level of the OrtLogger.
 
OrtEpDynamicOptions
OrtStatus * 	SetEpDynamicOptions (OrtSession *sess, const char *const *keys, const char *const *values, size_t kv_len)
 	Set DynamicOptions for EPs (Execution Providers)
 
