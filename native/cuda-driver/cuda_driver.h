#ifndef G_NATIVE_CUDA_DRIVER_CUDA_DRIVER_H_
#define G_NATIVE_CUDA_DRIVER_CUDA_DRIVER_H_

#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace g::cuda_native {

// Repository-private CUDA Driver API support shared by native FFI translation
// units. ErrorFactory converts neutral failure kinds into each crate's existing
// exception and status domains, so this header does not own a public ABI.

using CudaResult = std::int32_t;
using CudaDevice = std::int32_t;
using CudaDeviceAttribute = std::int32_t;
using CudaJitOption = std::int32_t;

struct CudaContextOpaque;
struct CudaModuleOpaque;
struct CudaFunctionOpaque;
struct CUstream_st;

using CudaContext = CudaContextOpaque*;
using CudaModule = CudaModuleOpaque*;
using CudaFunction = CudaFunctionOpaque*;
using CudaStream = CUstream_st*;

inline constexpr CudaResult kCudaSuccess = 0;
inline constexpr CudaResult kCudaErrorInvalidDevice = 101;
inline constexpr CudaDeviceAttribute kComputeCapabilityMajor = 75;
inline constexpr CudaDeviceAttribute kComputeCapabilityMinor = 76;

using CudaInit = CudaResult (*)(unsigned int flags);
using CudaDriverGetVersion = CudaResult (*)(std::int32_t* version);
using CudaDeviceGet = CudaResult (*)(CudaDevice* device, std::int32_t ordinal);
using CudaDeviceGetAttribute = CudaResult (*)(std::int32_t* value, CudaDeviceAttribute attribute, CudaDevice device);
using CudaContextGetCurrent = CudaResult (*)(CudaContext* context);
using CudaContextGetDevice = CudaResult (*)(CudaDevice* device);
using CudaModuleLoadDataEx = CudaResult (*)(CudaModule* module,
                                            const void* image,
                                            unsigned int option_count,
                                            CudaJitOption* options,
                                            void** option_values);
using CudaModuleUnload = CudaResult (*)(CudaModule module);
using CudaModuleGetFunction = CudaResult (*)(CudaFunction* function, CudaModule module, const char* name);
using CudaLaunchKernel = CudaResult (*)(CudaFunction function,
                                        unsigned int grid_width,
                                        unsigned int grid_height,
                                        unsigned int grid_depth,
                                        unsigned int block_width,
                                        unsigned int block_height,
                                        unsigned int block_depth,
                                        unsigned int shared_memory_bytes,
                                        CudaStream stream,
                                        void** kernel_parameters,
                                        void** extra);

static_assert(sizeof(CudaResult) == 4);
static_assert(sizeof(CudaDevice) == 4);
static_assert(sizeof(CudaContext) == sizeof(void*));
static_assert(sizeof(CudaStream) == sizeof(void*));

enum class CudaInitializationFailureKind {
  kCudaDriverUnavailable,
  kRequiredSymbolUnavailable,
  kCudaDriverFailure,
  kCudaDeviceUnavailable,
};

enum class CudaRuntimeFailureKind {
  kInternal,
  kFailedPrecondition,
};

[[nodiscard]] constexpr CudaInitializationFailureKind cuda_device_lookup_failure_kind(CudaResult status) {
  return status == kCudaErrorInvalidDevice ? CudaInitializationFailureKind::kCudaDeviceUnavailable
                                           : CudaInitializationFailureKind::kCudaDriverFailure;
}

static_assert(cuda_device_lookup_failure_kind(kCudaErrorInvalidDevice) ==
              CudaInitializationFailureKind::kCudaDeviceUnavailable);
static_assert(cuda_device_lookup_failure_kind(1) == CudaInitializationFailureKind::kCudaDriverFailure);
static_assert(cuda_device_lookup_failure_kind(3) == CudaInitializationFailureKind::kCudaDriverFailure);
static_assert(cuda_device_lookup_failure_kind(4) == CudaInitializationFailureKind::kCudaDriverFailure);
static_assert(cuda_device_lookup_failure_kind(201) == CudaInitializationFailureKind::kCudaDriverFailure);
static_assert(cuda_device_lookup_failure_kind(999) == CudaInitializationFailureKind::kCudaDriverFailure);

[[nodiscard]] static std::string dynamic_loader_error(std::string_view operation) {
  const char* detail = ::dlerror();
  if (detail == nullptr) {
    return std::string(operation) + ": dynamic loader returned no diagnostic";
  }
  return std::string(operation) + ": " + detail;
}

struct DynamicLibraryCloser {
  void operator()(void* library) const noexcept {
    if (library != nullptr) {
      static_cast<void>(::dlclose(library));
    }
  }
};

using DynamicLibrary = std::unique_ptr<void, DynamicLibraryCloser>;

template <typename ErrorFactory, typename InitializationFailureKind>
DynamicLibrary open_dynamic_library(const char* soname, InitializationFailureKind failure_kind) {
  ::dlerror();
  void* handle = ::dlopen(soname, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    throw ErrorFactory::initialization_failure(failure_kind, dynamic_loader_error(std::string("load ") + soname));
  }
  return DynamicLibrary(handle);
}

template <typename Function, typename ErrorFactory, typename InitializationFailureKind>
Function load_dynamic_library_symbol(void* library,
                                     const char* library_name,
                                     const char* symbol_name,
                                     InitializationFailureKind failure_kind) {
  static_assert(std::is_pointer_v<Function>);
  static_assert(sizeof(Function) == sizeof(void*));

  ::dlerror();
  void* symbol = ::dlsym(library, symbol_name);
  if (const char* detail = ::dlerror(); detail != nullptr) {
    throw ErrorFactory::initialization_failure(
        failure_kind,
        std::string("load ") + library_name + " symbol " + symbol_name + ": " + detail);
  }
  if (symbol == nullptr) {
    throw ErrorFactory::initialization_failure(
        failure_kind,
        std::string("load ") + library_name + " symbol " + symbol_name + ": dynamic loader returned a null symbol");
  }
  Function function = nullptr;
  std::memcpy(&function, &symbol, sizeof(function));
  return function;
}

template <typename ErrorFactory>
class CudaDriverApi {
 public:
  CudaDriverApi()
      : library_(
            open_dynamic_library<ErrorFactory>("libcuda.so.1", CudaInitializationFailureKind::kCudaDriverUnavailable)),
        initialize_(load_driver_symbol<CudaInit>(library_.get(), "cuInit")),
        driver_get_version_(load_driver_symbol<CudaDriverGetVersion>(library_.get(), "cuDriverGetVersion")),
        device_get_(load_driver_symbol<CudaDeviceGet>(library_.get(), "cuDeviceGet")),
        device_get_attribute_(load_driver_symbol<CudaDeviceGetAttribute>(library_.get(), "cuDeviceGetAttribute")),
        context_get_current_(load_driver_symbol<CudaContextGetCurrent>(library_.get(), "cuCtxGetCurrent")),
        context_get_device_(load_driver_symbol<CudaContextGetDevice>(library_.get(), "cuCtxGetDevice")),
        module_load_data_(load_driver_symbol<CudaModuleLoadDataEx>(library_.get(), "cuModuleLoadDataEx")),
        module_unload_(load_driver_symbol<CudaModuleUnload>(library_.get(), "cuModuleUnload")),
        module_get_function_(load_driver_symbol<CudaModuleGetFunction>(library_.get(), "cuModuleGetFunction")),
        launch_kernel_(load_driver_symbol<CudaLaunchKernel>(library_.get(), "cuLaunchKernel")) {
    check_initialization(initialize_(0), "initialize CUDA driver");
    check_initialization(driver_get_version_(&driver_version_), "query CUDA driver version");
  }

  CudaDriverApi(const CudaDriverApi&) = delete;
  CudaDriverApi& operator=(const CudaDriverApi&) = delete;

  [[nodiscard]] std::int32_t driver_version() const noexcept { return driver_version_; }

  template <typename Capability>
  [[nodiscard]] CudaDevice inspect_device(std::int32_t device_ordinal, Capability& capability) const {
    capability.device_ordinal = device_ordinal;
    CudaDevice device = 0;
    const CudaResult device_status = device_get_(&device, device_ordinal);
    if (device_status != kCudaSuccess) {
      throw ErrorFactory::initialization_failure(cuda_device_lookup_failure_kind(device_status),
                                                 "query CUDA device ordinal " + std::to_string(device_ordinal) +
                                                     " failed with status " + std::to_string(device_status));
    }
    check_initialization(device_get_attribute_(&capability.compute_capability_major, kComputeCapabilityMajor, device),
                         "query CUDA compute-capability major version");
    check_initialization(device_get_attribute_(&capability.compute_capability_minor, kComputeCapabilityMinor, device),
                         "query CUDA compute-capability minor version");
    return device;
  }

  [[nodiscard]] CudaContext current_context() const {
    CudaContext context = nullptr;
    check_runtime(context_get_current_(&context), "query current CUDA context");
    if (context == nullptr) {
      throw ErrorFactory::runtime_failure(CudaRuntimeFailureKind::kFailedPrecondition,
                                          "the XLA FFI execution thread has no current CUDA context");
    }
    return context;
  }

  void validate_current_context_device(CudaDevice qualified_device,
                                       std::int32_t minimum_compute_capability_major,
                                       std::int32_t minimum_compute_capability_minor,
                                       std::string_view unsupported_device_detail) const {
    CudaDevice device = 0;
    check_runtime(context_get_device_(&device), "query current XLA CUDA context device");
    if (device != qualified_device) {
      throw ErrorFactory::runtime_failure(CudaRuntimeFailureKind::kFailedPrecondition,
                                          "current XLA CUDA context device " + std::to_string(device) +
                                              " does not match qualified CUDA device " +
                                              std::to_string(qualified_device));
    }
    std::int32_t compute_capability_major = 0;
    std::int32_t compute_capability_minor = 0;
    check_runtime(device_get_attribute_(&compute_capability_major, kComputeCapabilityMajor, device),
                  "query current XLA CUDA context compute-capability major version");
    check_runtime(device_get_attribute_(&compute_capability_minor, kComputeCapabilityMinor, device),
                  "query current XLA CUDA context compute-capability minor version");
    if (compute_capability_major < minimum_compute_capability_major ||
        (compute_capability_major == minimum_compute_capability_major &&
         compute_capability_minor < minimum_compute_capability_minor)) {
      throw ErrorFactory::runtime_failure(
          CudaRuntimeFailureKind::kFailedPrecondition,
          "current XLA CUDA context uses device " + std::to_string(device) + " with unsupported compute capability " +
              std::to_string(compute_capability_major) + "." + std::to_string(compute_capability_minor) + "; " +
              std::string(unsupported_device_detail));
    }
  }

  [[nodiscard]] CudaModule load_module(const void* image, std::string_view operation) const {
    CudaModule module = nullptr;
    check_runtime(module_load_data_(&module, image, 0, nullptr, nullptr), operation);
    if (module == nullptr) {
      throw ErrorFactory::runtime_failure(CudaRuntimeFailureKind::kInternal,
                                          std::string(operation) + " returned a null CUDA module");
    }
    return module;
  }

  void unload_module(CudaModule module) const noexcept {
    if (module != nullptr && module_unload_ != nullptr) {
      static_cast<void>(module_unload_(module));
    }
  }

  [[nodiscard]] CudaFunction get_function(CudaModule module, const char* name) const {
    CudaFunction function = nullptr;
    check_runtime(module_get_function_(&function, module, name), std::string("find CUDA kernel ") + name);
    if (function == nullptr) {
      throw ErrorFactory::runtime_failure(CudaRuntimeFailureKind::kInternal,
                                          std::string("find CUDA kernel ") + name + " returned a null symbol");
    }
    return function;
  }

  void launch_kernel(CudaFunction function,
                     unsigned int grid_width,
                     unsigned int block_width,
                     CudaStream stream,
                     void** arguments,
                     std::string_view operation) const {
    check_runtime(launch_kernel_(function, grid_width, 1, 1, block_width, 1, 1, 0, stream, arguments, nullptr),
                  operation);
  }

 private:
  template <typename Function>
  static Function load_driver_symbol(void* library, const char* symbol_name) {
    return load_dynamic_library_symbol<Function, ErrorFactory>(
        library,
        "CUDA driver",
        symbol_name,
        CudaInitializationFailureKind::kRequiredSymbolUnavailable);
  }

  static void check_initialization(CudaResult status, std::string_view operation) {
    if (status != kCudaSuccess) {
      throw ErrorFactory::initialization_failure(
          CudaInitializationFailureKind::kCudaDriverFailure,
          std::string(operation) + " failed with CUDA driver status " + std::to_string(status));
    }
  }

  static void check_runtime(CudaResult status, std::string_view operation) {
    if (status != kCudaSuccess) {
      throw ErrorFactory::runtime_failure(
          CudaRuntimeFailureKind::kInternal,
          std::string(operation) + " failed with CUDA driver status " + std::to_string(status));
    }
  }

  DynamicLibrary library_;
  CudaInit initialize_;
  CudaDriverGetVersion driver_get_version_;
  CudaDeviceGet device_get_;
  CudaDeviceGetAttribute device_get_attribute_;
  CudaContextGetCurrent context_get_current_;
  CudaContextGetDevice context_get_device_;
  CudaModuleLoadDataEx module_load_data_;
  CudaModuleUnload module_unload_;
  CudaModuleGetFunction module_get_function_;
  CudaLaunchKernel launch_kernel_;
  std::int32_t driver_version_ = 0;
};

template <typename Driver>
class CudaModuleOwner {
 public:
  CudaModuleOwner(const Driver& driver, CudaModule module) noexcept : driver_(&driver), module_(module) {}

  ~CudaModuleOwner() { driver_->unload_module(module_); }

  CudaModuleOwner(const CudaModuleOwner&) = delete;
  CudaModuleOwner& operator=(const CudaModuleOwner&) = delete;

  CudaModuleOwner(CudaModuleOwner&& other) noexcept
      : driver_(other.driver_), module_(std::exchange(other.module_, nullptr)) {}

  CudaModuleOwner& operator=(CudaModuleOwner&& other) noexcept {
    if (this != &other) {
      driver_->unload_module(module_);
      driver_ = other.driver_;
      module_ = std::exchange(other.module_, nullptr);
    }
    return *this;
  }

  [[nodiscard]] CudaModule get() const noexcept { return module_; }

 private:
  const Driver* driver_;
  CudaModule module_;
};

}  // namespace g::cuda_native

#endif  // G_NATIVE_CUDA_DRIVER_CUDA_DRIVER_H_
