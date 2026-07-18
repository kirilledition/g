#include "cuda_driver_abi.h"

#include <xla/ffi/api/ffi.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

static_assert(XLA_FFI_API_MAJOR == 0);
static_assert(XLA_FFI_API_MINOR == 3);
static_assert(sizeof(std::size_t) == sizeof(std::uint64_t));

struct GComputeCudaCapability {
  std::int32_t cuda_driver_version;
  std::int32_t device_ordinal;
  std::int32_t compute_capability_major;
  std::int32_t compute_capability_minor;
};

static_assert(sizeof(GComputeCudaCapability) == 16);
static_assert(std::is_standard_layout_v<GComputeCudaCapability>);

namespace {

using g::compute_cuda::abi::CudaContext;
using g::compute_cuda::abi::CudaContextGetCurrent;
using g::compute_cuda::abi::CudaContextGetDevice;
using g::compute_cuda::abi::CudaDevice;
using g::compute_cuda::abi::CudaDeviceGet;
using g::compute_cuda::abi::CudaDeviceGetAttribute;
using g::compute_cuda::abi::CudaDriverGetVersion;
using g::compute_cuda::abi::CudaFunction;
using g::compute_cuda::abi::CudaInit;
using g::compute_cuda::abi::CudaLaunchKernel;
using g::compute_cuda::abi::CudaModule;
using g::compute_cuda::abi::CudaModuleGetFunction;
using g::compute_cuda::abi::CudaModuleLoadDataEx;
using g::compute_cuda::abi::CudaModuleUnload;
using g::compute_cuda::abi::CudaResult;
using g::compute_cuda::abi::CudaStream;
using xla::ffi::AnyBuffer;
using xla::ffi::DataType;
using xla::ffi::Error;
using xla::ffi::ErrorCode;

constexpr std::string_view kKernelName = "compute_firth_components";
constexpr std::int32_t kMinimumCudaDriverVersion = 12020;
constexpr std::int32_t kMinimumComputeCapabilityMajor = 7;
constexpr unsigned int kKernelBlockSize = 256;

constexpr char kFirthComponentsKernelPtx[] =
#include "firth_components_kernel_ptx.inc"
    ;

enum class InitializationStatus : std::int32_t {
  kSuccess = 0,
  kCudaDriverUnavailable = 1,
  kRequiredSymbolUnavailable = 2,
  kCudaDriverFailure = 3,
  kCudaDriverTooOld = 4,
  kCudaDeviceUnavailable = 5,
  kComputeCapabilityUnsupported = 6,
  kInternal = 7,
};

class InitializationFailure final : public std::runtime_error {
 public:
  InitializationFailure(InitializationStatus status, std::string detail)
      : std::runtime_error(std::move(detail)), status_(status) {}

  [[nodiscard]] InitializationStatus status() const noexcept { return status_; }

 private:
  InitializationStatus status_;
};

class HandlerFailure final : public std::runtime_error {
 public:
  HandlerFailure(ErrorCode code, std::string detail)
      : std::runtime_error(std::move(detail)), code_(code) {}

  [[nodiscard]] ErrorCode code() const noexcept { return code_; }

 private:
  ErrorCode code_;
};

[[noreturn]] void fail_initialization(
    InitializationStatus status,
    std::string detail) {
  throw InitializationFailure(status, std::move(detail));
}

[[noreturn]] void fail_handler(ErrorCode code, std::string detail) {
  throw HandlerFailure(code, std::move(detail));
}

[[noreturn]] void fail_runtime(std::string detail) {
  fail_handler(ErrorCode::kInternal, std::move(detail));
}

std::string dynamic_loader_error(std::string_view operation) {
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

DynamicLibrary open_cuda_driver() {
  ::dlerror();
  void* handle = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    fail_initialization(
        InitializationStatus::kCudaDriverUnavailable,
        dynamic_loader_error("load libcuda.so.1"));
  }
  return DynamicLibrary(handle);
}

template <typename Function>
Function load_symbol(void* library, const char* symbol_name) {
  static_assert(std::is_pointer_v<Function>);
  static_assert(sizeof(Function) == sizeof(void*));
  ::dlerror();
  void* symbol = ::dlsym(library, symbol_name);
  if (const char* detail = ::dlerror(); detail != nullptr) {
    fail_initialization(
        InitializationStatus::kRequiredSymbolUnavailable,
        std::string("load CUDA driver symbol ") + symbol_name + ": " + detail);
  }
  Function function = nullptr;
  std::memcpy(&function, &symbol, sizeof(function));
  return function;
}

void check_cuda_initialization(CudaResult status, std::string_view operation) {
  if (status != g::compute_cuda::abi::kCudaSuccess) {
    fail_initialization(
        InitializationStatus::kCudaDriverFailure,
        std::string(operation) + " failed with CUDA driver status " +
            std::to_string(status));
  }
}

void check_cuda_runtime(CudaResult status, std::string_view operation) {
  if (status != g::compute_cuda::abi::kCudaSuccess) {
    fail_runtime(
        std::string(operation) + " failed with CUDA driver status " +
        std::to_string(status));
  }
}

class CudaDriverApi {
 public:
  CudaDriverApi()
      : library_(open_cuda_driver()),
        initialize_(load_symbol<CudaInit>(library_.get(), "cuInit")),
        driver_get_version_(
            load_symbol<CudaDriverGetVersion>(library_.get(), "cuDriverGetVersion")),
        device_get_(load_symbol<CudaDeviceGet>(library_.get(), "cuDeviceGet")),
        device_get_attribute_(load_symbol<CudaDeviceGetAttribute>(
            library_.get(), "cuDeviceGetAttribute")),
        context_get_current_(load_symbol<CudaContextGetCurrent>(
            library_.get(), "cuCtxGetCurrent")),
        context_get_device_(load_symbol<CudaContextGetDevice>(
            library_.get(), "cuCtxGetDevice")),
        module_load_data_(load_symbol<CudaModuleLoadDataEx>(
            library_.get(), "cuModuleLoadDataEx")),
        module_unload_(load_symbol<CudaModuleUnload>(
            library_.get(), "cuModuleUnload")),
        module_get_function_(load_symbol<CudaModuleGetFunction>(
            library_.get(), "cuModuleGetFunction")),
        launch_kernel_(load_symbol<CudaLaunchKernel>(
            library_.get(), "cuLaunchKernel")) {
    check_cuda_initialization(initialize_(0), "initialize CUDA driver");
    check_cuda_initialization(
        driver_get_version_(&driver_version_), "query CUDA driver version");
  }

  CudaDriverApi(const CudaDriverApi&) = delete;
  CudaDriverApi& operator=(const CudaDriverApi&) = delete;

  [[nodiscard]] std::int32_t driver_version() const noexcept {
    return driver_version_;
  }

  void inspect_device(
      std::int32_t device_ordinal,
      GComputeCudaCapability& capability) const {
    capability.device_ordinal = device_ordinal;
    CudaDevice device = 0;
    const CudaResult device_status = device_get_(&device, device_ordinal);
    if (device_status != g::compute_cuda::abi::kCudaSuccess) {
      fail_initialization(
          InitializationStatus::kCudaDeviceUnavailable,
          "query CUDA device ordinal " + std::to_string(device_ordinal) +
              " failed with status " + std::to_string(device_status));
    }
    check_cuda_initialization(
        device_get_attribute_(
            &capability.compute_capability_major,
            g::compute_cuda::abi::kComputeCapabilityMajor,
            device),
        "query CUDA compute-capability major version");
    check_cuda_initialization(
        device_get_attribute_(
            &capability.compute_capability_minor,
            g::compute_cuda::abi::kComputeCapabilityMinor,
            device),
        "query CUDA compute-capability minor version");
  }

  [[nodiscard]] CudaContext current_context() const {
    CudaContext context = nullptr;
    check_cuda_runtime(context_get_current_(&context), "query current CUDA context");
    if (context == nullptr) {
      fail_handler(
          ErrorCode::kFailedPrecondition,
          "the XLA FFI execution thread has no current CUDA context");
    }
    return context;
  }

  void validate_current_context_device() const {
    CudaDevice device = 0;
    check_cuda_runtime(
        context_get_device_(&device), "query current XLA CUDA context device");
    std::int32_t compute_capability_major = 0;
    std::int32_t compute_capability_minor = 0;
    check_cuda_runtime(
        device_get_attribute_(
            &compute_capability_major,
            g::compute_cuda::abi::kComputeCapabilityMajor,
            device),
        "query current XLA CUDA context compute-capability major version");
    check_cuda_runtime(
        device_get_attribute_(
            &compute_capability_minor,
            g::compute_cuda::abi::kComputeCapabilityMinor,
            device),
        "query current XLA CUDA context compute-capability minor version");
    if (compute_capability_major < kMinimumComputeCapabilityMajor) {
      fail_handler(
          ErrorCode::kFailedPrecondition,
          "current XLA CUDA context uses device " + std::to_string(device) +
              " with unsupported compute capability " +
              std::to_string(compute_capability_major) + "." +
              std::to_string(compute_capability_minor) +
              "; CUDA Firth components require 7.0 or newer");
    }
  }

  [[nodiscard]] CudaModule load_module() const {
    CudaModule module = nullptr;
    check_cuda_runtime(
        module_load_data_(
            &module, kFirthComponentsKernelPtx, 0, nullptr, nullptr),
        "load Firth component compute_70 PTX");
    return module;
  }

  void unload_module(CudaModule module) const noexcept {
    if (module != nullptr) {
      static_cast<void>(module_unload_(module));
    }
  }

  [[nodiscard]] CudaFunction get_function(
      CudaModule module,
      const char* name) const {
    CudaFunction function = nullptr;
    check_cuda_runtime(
        module_get_function_(&function, module, name),
        std::string("find CUDA kernel ") + name);
    return function;
  }

  void launch(
      CudaFunction function,
      unsigned int grid_width,
      CudaStream stream,
      void** arguments) const {
    check_cuda_runtime(
        launch_kernel_(
            function,
            grid_width,
            1,
            1,
            kKernelBlockSize,
            1,
            1,
            0,
            stream,
            arguments,
            nullptr),
        "launch Firth component kernel");
  }

 private:
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

class CudaModuleOwner {
 public:
  CudaModuleOwner(const CudaDriverApi& driver, CudaModule module)
      : driver_(&driver), module_(module) {}

  ~CudaModuleOwner() { driver_->unload_module(module_); }

  CudaModuleOwner(const CudaModuleOwner&) = delete;
  CudaModuleOwner& operator=(const CudaModuleOwner&) = delete;
  CudaModuleOwner(CudaModuleOwner&& other) noexcept
      : driver_(other.driver_), module_(std::exchange(other.module_, nullptr)) {}

  CudaModuleOwner& operator=(CudaModuleOwner&&) = delete;

  [[nodiscard]] CudaModule get() const noexcept { return module_; }

 private:
  const CudaDriverApi* driver_;
  CudaModule module_;
};

class FirthComponentsKernel {
 public:
  explicit FirthComponentsKernel(const CudaDriverApi& driver)
      : driver_(driver), module_(driver, driver.load_module()) {
    function_ = driver_.get_function(module_.get(), kKernelName.data());
  }

  FirthComponentsKernel(const FirthComponentsKernel&) = delete;
  FirthComponentsKernel& operator=(const FirthComponentsKernel&) = delete;

  void launch(
      const double* phenotype,
      const double* genotype,
      const double* offset,
      const bool* active_sample_mask,
      const double* non_active_deviance,
      const double* beta,
      const double* minimum_variance,
      std::size_t lane_count,
      std::size_t sample_count,
      double* genotype_information,
      double* score_adjustment,
      double* penalized_deviance,
      double* score,
      bool* valid,
      CudaStream stream) const {
    if (lane_count > std::numeric_limits<unsigned int>::max()) {
      fail_handler(
          ErrorCode::kInvalidArgument,
          "Firth component kernel lane count exceeds the CUDA grid domain");
    }
    std::uint64_t lane_count_argument = lane_count;
    std::uint64_t sample_count_argument = sample_count;
    void* arguments[] = {
        &phenotype,
        &genotype,
        &offset,
        &active_sample_mask,
        &non_active_deviance,
        &beta,
        &minimum_variance,
        &lane_count_argument,
        &sample_count_argument,
        &genotype_information,
        &score_adjustment,
        &penalized_deviance,
        &score,
        &valid,
    };
    driver_.launch(
        function_, static_cast<unsigned int>(lane_count), stream, arguments);
  }

 private:
  const CudaDriverApi& driver_;
  CudaModuleOwner module_;
  CudaFunction function_ = nullptr;
};

class FirthComponentsKernelCache {
 public:
  explicit FirthComponentsKernelCache(const CudaDriverApi& driver)
      : driver_(driver) {}

  [[nodiscard]] const FirthComponentsKernel& for_current_context() {
    const CudaContext context = driver_.current_context();
    struct ThreadCache {
      const FirthComponentsKernelCache* owner = nullptr;
      CudaContext context = nullptr;
      const FirthComponentsKernel* kernel = nullptr;
    };
    thread_local ThreadCache thread_cache;
    if (thread_cache.owner == this && thread_cache.context == context &&
        thread_cache.kernel != nullptr) {
      return *thread_cache.kernel;
    }

    std::scoped_lock lock(mutex_);
    auto iterator = kernels_.find(context);
    if (iterator == kernels_.end()) {
      driver_.validate_current_context_device();
      iterator = kernels_
                     .emplace(
                         context,
                         std::make_unique<FirthComponentsKernel>(driver_))
                     .first;
    }
    thread_cache = ThreadCache{this, context, iterator->second.get()};
    return *thread_cache.kernel;
  }

 private:
  const CudaDriverApi& driver_;
  std::mutex mutex_;
  std::unordered_map<CudaContext, std::unique_ptr<FirthComponentsKernel>> kernels_;
};

class RuntimeState {
 public:
  RuntimeState() : kernels_(driver_) {}

  RuntimeState(const RuntimeState&) = delete;
  RuntimeState& operator=(const RuntimeState&) = delete;

  void validate(
      std::int32_t device_ordinal,
      GComputeCudaCapability& capability) const {
    capability.cuda_driver_version = driver_.driver_version();
    capability.device_ordinal = device_ordinal;
    if (capability.cuda_driver_version < kMinimumCudaDriverVersion) {
      fail_initialization(
          InitializationStatus::kCudaDriverTooOld,
          "embedded PTX ISA 8.2 requires CUDA driver API version 12020 or newer");
    }
    driver_.inspect_device(device_ordinal, capability);
    if (capability.compute_capability_major < kMinimumComputeCapabilityMajor) {
      fail_initialization(
          InitializationStatus::kComputeCapabilityUnsupported,
          "CUDA Firth components require compute capability 7.0 or newer");
    }
  }

  [[nodiscard]] const FirthComponentsKernel& kernel() {
    return kernels_.for_current_context();
  }

 private:
  CudaDriverApi driver_;
  FirthComponentsKernelCache kernels_;
};

RuntimeState& runtime_state() {
  static RuntimeState state;
  return state;
}

bool dimensions_equal(
    const AnyBuffer& left,
    const AnyBuffer& right) {
  return left.dimensions() == right.dimensions();
}

bool has_lane_dimensions(
    const AnyBuffer& buffer,
    DataType data_type,
    const AnyBuffer& output) {
  return buffer.element_type() == data_type &&
      dimensions_equal(buffer, output);
}

bool has_sample_dimensions(
    const AnyBuffer& buffer,
    DataType data_type,
    const AnyBuffer& output,
    std::size_t sample_count) {
  if (buffer.element_type() != data_type ||
      buffer.dimensions().size() != output.dimensions().size() + 1 ||
      buffer.dimensions().back() < 0 ||
      static_cast<std::size_t>(buffer.dimensions().back()) != sample_count) {
    return false;
  }
  for (std::size_t dimension_index = 0;
       dimension_index < output.dimensions().size();
       ++dimension_index) {
    if (buffer.dimensions()[dimension_index] !=
        output.dimensions()[dimension_index]) {
      return false;
    }
  }
  return true;
}

std::size_t checked_lane_count(const AnyBuffer& output) {
  std::size_t lane_count = 1;
  for (const std::int64_t dimension : output.dimensions()) {
    if (dimension <= 0 ||
        static_cast<std::uint64_t>(dimension) >
            std::numeric_limits<std::size_t>::max() / lane_count) {
      fail_handler(
          ErrorCode::kInvalidArgument,
          "Firth component output dimensions are empty or overflow size_t");
    }
    lane_count *= static_cast<std::size_t>(dimension);
  }
  return lane_count;
}

Error compute_firth_components(
    AnyBuffer phenotype,
    AnyBuffer genotype,
    AnyBuffer offset,
    AnyBuffer active_sample_mask,
    AnyBuffer non_active_deviance,
    AnyBuffer beta,
    AnyBuffer minimum_variance,
    xla::ffi::Result<AnyBuffer> genotype_information,
    xla::ffi::Result<AnyBuffer> score_adjustment,
    xla::ffi::Result<AnyBuffer> penalized_deviance,
    xla::ffi::Result<AnyBuffer> score,
    xla::ffi::Result<AnyBuffer> valid,
    CudaStream stream) {
  try {
    if (phenotype.dimensions().size() == 0 || phenotype.dimensions().back() <= 0) {
      return Error::InvalidArgument(
          "Firth component sample inputs must have a nonempty final dimension");
    }
    const std::size_t sample_count =
        static_cast<std::size_t>(phenotype.dimensions().back());
    const bool outputs_match =
        has_lane_dimensions(*genotype_information, DataType::F64, beta) &&
        has_lane_dimensions(*score_adjustment, DataType::F64, beta) &&
        has_lane_dimensions(*penalized_deviance, DataType::F64, beta) &&
        has_lane_dimensions(*score, DataType::F64, beta) &&
        has_lane_dimensions(*valid, DataType::PRED, beta);
    const bool sample_inputs_match =
        has_sample_dimensions(phenotype, DataType::F64, beta, sample_count) &&
        has_sample_dimensions(genotype, DataType::F64, beta, sample_count) &&
        has_sample_dimensions(offset, DataType::F64, beta, sample_count) &&
        has_sample_dimensions(
            active_sample_mask, DataType::PRED, beta, sample_count);
    const bool lane_inputs_match =
        has_lane_dimensions(non_active_deviance, DataType::F64, beta) &&
        has_lane_dimensions(beta, DataType::F64, beta) &&
        has_lane_dimensions(minimum_variance, DataType::F64, beta);
    if (!outputs_match || !sample_inputs_match || !lane_inputs_match) {
      return Error::InvalidArgument(
          "Firth component operands and results must use matching batch prefixes, "
          "f64 values, and boolean masks");
    }
    const std::size_t lane_count = checked_lane_count(beta);
    runtime_state().kernel().launch(
        phenotype.typed_data<double>(),
        genotype.typed_data<double>(),
        offset.typed_data<double>(),
        active_sample_mask.typed_data<bool>(),
        non_active_deviance.typed_data<double>(),
        beta.typed_data<double>(),
        minimum_variance.typed_data<double>(),
        lane_count,
        sample_count,
        genotype_information->typed_data<double>(),
        score_adjustment->typed_data<double>(),
        penalized_deviance->typed_data<double>(),
        score->typed_data<double>(),
        valid->typed_data<bool>(),
        stream);
    return Error::Success();
  } catch (const HandlerFailure& failure) {
    return Error(failure.code(), failure.what());
  } catch (const std::exception& exception) {
    return Error(ErrorCode::kInternal, exception.what());
  } catch (...) {
    return Error(
        ErrorCode::kInternal,
        "unknown native Firth component handler failure");
  }
}

}  // namespace

extern "C" std::int32_t g_compute_cuda_initialize_firth_components_runtime(
    std::int32_t device_ordinal,
    GComputeCudaCapability* capability,
    const char** detail) noexcept {
  thread_local std::string detail_storage;
  if (capability == nullptr || detail == nullptr) {
    return static_cast<std::int32_t>(InitializationStatus::kInternal);
  }
  *capability = {};
  *detail = nullptr;
  try {
    runtime_state().validate(device_ordinal, *capability);
    return static_cast<std::int32_t>(InitializationStatus::kSuccess);
  } catch (const InitializationFailure& failure) {
    detail_storage = failure.what();
    *detail = detail_storage.c_str();
    return static_cast<std::int32_t>(failure.status());
  } catch (const std::exception& exception) {
    detail_storage = exception.what();
    *detail = detail_storage.c_str();
    return static_cast<std::int32_t>(InitializationStatus::kInternal);
  } catch (...) {
    detail_storage = "unknown CUDA compute initialization failure";
    *detail = detail_storage.c_str();
    return static_cast<std::int32_t>(InitializationStatus::kInternal);
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    g_firth_components_ffi,
    compute_firth_components,
    xla::ffi::Ffi::Bind()
        .Arg<AnyBuffer>()
        .Arg<AnyBuffer>()
        .Arg<AnyBuffer>()
        .Arg<AnyBuffer>()
        .Arg<AnyBuffer>()
        .Arg<AnyBuffer>()
        .Arg<AnyBuffer>()
        .Ret<AnyBuffer>()
        .Ret<AnyBuffer>()
        .Ret<AnyBuffer>()
        .Ret<AnyBuffer>()
        .Ret<AnyBuffer>()
        .Ctx<xla::ffi::PlatformStream<CudaStream>>());
