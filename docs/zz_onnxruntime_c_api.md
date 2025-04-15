# Get started with ORT for C / C++

## Contents
- [Builds](#builds)
- [API Reference](#api-reference)
- [Features](#features)
- [Deployment](#deployment)
- [Telemetry](#telemetry)
- [Samples](#samples)

---

## Builds

| Artifact | Description | Supported Platforms |
|---------|-------------|---------------------|
| `Microsoft.ML.OnnxRuntime` | CPU (Release) | Windows, Linux, Mac, X64, X86 (Windows-only), ARM64 (Windows-only)… [more details: compatibility](https://onnxruntime.ai/docs/build/eps.html) |
| `Microsoft.ML.OnnxRuntime.Gpu` | GPU - CUDA (Release) | Windows, Linux, Mac, X64… [more details: compatibility](https://onnxruntime.ai/docs/build/eps.html) |
| `Microsoft.ML.OnnxRuntime.DirectML` | GPU - DirectML (Release) | Windows 10 1709+ |
| `onnxruntime` | CPU, GPU (Dev), CPU (On-Device Training) | Same as Release versions |
| `Microsoft.ML.OnnxRuntime.Training` | CPU On-Device Training (Release) | Windows, Linux, Mac, X64, X86 (Windows-only), ARM64 (Windows-only)… [more details: compatibility](https://onnxruntime.ai/docs/build/eps.html) |

`.zip` and `.tgz` files are also included as assets in each GitHub release.

---

## API Reference

Refer to `onnxruntime_c_api.h`

### Usage:
1. Include `onnxruntime_c_api.h`.
2. Call `OrtCreateEnv`.
3. Create Session: `OrtCreateSession(env, model_uri, nullptr,…)`.
4. (Optional) Add more execution providers (e.g., `OrtSessionOptionsAppendExecutionProvider_CUDA`).
5. Create Tensor:
    - `OrtCreateMemoryInfo`
    - `OrtCreateTensorWithDataAsOrtValue`
6. Run: `OrtRun`

---

## Features

- Create `InferenceSession` from on-disk model with `SessionOptions`.
- Register custom loggers and allocators.
- Register predefined execution providers (e.g. CUDA, DNNL) with priority.
- Inputs must be in **CPU memory**.
- Convert in-memory ONNX Tensor (protobuf) to pointer.
- Set thread pool size per session.
- Configure graph optimization level.
- Load custom ops dynamically.
- Load model from byte array: `OrtCreateSessionFromArray`.

### Global/Shared Threadpools

#### Usage:
- Create `ThreadingOptions`, set values to `0` for defaults.
- Create env with `CreateEnvWithGlobalThreadPools()`.
- Disable per-session threads: `DisablePerSessionThreads()`.
- Call `Run()` as usual.

### Shared Allocator

#### Usage:
- Create & register shared allocator: `CreateAndRegisterAllocator`.
- Set `session.use_env_allocators` to `"1"` for shared usage.

> **Example:** See `TestSharedAllocatorUsingCreateAndRegisterAllocator`  
> [test_inference.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/shared_lib/test_inference.cc)

---

### Configuring `OrtArenaCfg`

Use `CreateArenaCfgV2` (from ORT 1.8+) with:

- `max_mem`: Max arena memory.
- `arena_extend_strategy`: `kSameAsRequested` or `kNextPowerOfTwo`.
- `initial_chunk_size_bytes`: First chunk size for arena.
- `initial_growth_chunk_size_bytes`: First allocation size post shrinkage.
- `max_dead_bytes_per_chunk`: Controls splitting for allocation requests.

> **Example:** See `ConfigureCudaArenaAndDemonstrateMemoryArenaShrinkage`  
> [test_inference.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/shared_lib/test_inference.cc)

---

### Memory Arena Shrinkage

By default, memory isn’t returned to the system. Enable shrinkage **at the end of each `Run()`**.

#### Scenario:
Large memory request for dynamic shape → held forever. Shrinkage reclaims unused memory.

> **Applicable only to arena allocators.**  
> See: `ConfigureCudaArenaAndDemonstrateMemoryArenaShrinkage`

---

### Allocate Initializers from Non-Arena Memory

Prevents arena growth from storing large initializers.

#### Scenario:
Many initializers → arena allocates too much memory.

> See: `AllocateInitializersFromNonArenaMemory`  
> [test_inference.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/shared_lib/test_inference.cc)

---

### Share Initializers and Pre-Packed Versions

Avoid duplicating same initializers across sessions.

#### Usage:
- Use `AddInitializer` in `SessionOptions`.
- Reuse same `SessionOptions` for multiple sessions.
- Use `CreatePrepackedWeightsContainer` to share pre-packed weights.

> See:  
> - C API: `TestSharingOfInitializerAndItsPrepackedVersion`  
> - C# API: `TestSharingOfInitializerAndItsPrepackedVersion`  
> - Kernel test: `SharedPrepackedWeights`

---

## Deployment

### Windows 10

- Place `onnxruntime.dll` in the same folder as your app.
- Use **load-time** or **run-time dynamic linking**.

### DLL Search Order

Refer to [Dynamic Link Library Search Order](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order)

> **Important:**  
> Don't modify the system `%PATH%`. Instead, keep your DLL and `onnxruntime.dll` in the same folder. Use `GetModulePath()` to get the DLL folder.

---

## Telemetry

To enable/disable telemetry in official Windows builds, use:

- `EnableTelemetryEvents()`
- `DisableTelemetryEvents()`

See [Privacy](https://privacy.microsoft.com/en-us/privacystatement) for details.

---

## Samples

See: **Candy Style Transfer**
