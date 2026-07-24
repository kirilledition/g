#include <xla/ffi/api/ffi.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "cuda_driver.h"

static_assert(XLA_FFI_API_MAJOR == 0);
static_assert(XLA_FFI_API_MINOR == 3);
static_assert(sizeof(std::size_t) == sizeof(std::uint64_t));
static_assert(sizeof(bool) == 1);
static_assert(alignof(bool) == 1);

struct GComputeCudaCapability {
  std::int32_t cuda_driver_version;
  std::int32_t device_ordinal;
  std::int32_t compute_capability_major;
  std::int32_t compute_capability_minor;
};

static_assert(sizeof(GComputeCudaCapability) == 16);
static_assert(std::is_standard_layout_v<GComputeCudaCapability>);

namespace {

using g::cuda_native::CudaContext;
using g::cuda_native::CudaFunction;
using g::cuda_native::CudaStream;
using xla::ffi::AnyBuffer;
using xla::ffi::DataType;
using xla::ffi::Error;
using xla::ffi::ErrorCode;

#include "firth_components_artifact_identity.inc"

constexpr char kKernelName[] = "compute_firth_components";
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
  InitializationFailure(InitializationStatus status, const std::string& detail)
      : std::runtime_error(detail), status_(status) {}

  [[nodiscard]] InitializationStatus status() const noexcept { return status_; }

 private:
  InitializationStatus status_;
};

class HandlerFailure final : public std::runtime_error {
 public:
  HandlerFailure(ErrorCode code, const std::string& detail) : std::runtime_error(detail), code_(code) {}

  [[nodiscard]] ErrorCode code() const noexcept { return code_; }

 private:
  ErrorCode code_;
};

[[noreturn]] void fail_initialization(InitializationStatus status, const std::string& detail) {
  throw InitializationFailure(status, detail);
}

[[noreturn]] void fail_handler(ErrorCode code, const std::string& detail) { throw HandlerFailure(code, detail); }

struct CudaErrorFactory {
  static InitializationFailure initialization_failure(g::cuda_native::CudaInitializationFailureKind failure_kind,
                                                      const std::string& detail) {
    InitializationStatus status = InitializationStatus::kInternal;
    switch (failure_kind) {
      case g::cuda_native::CudaInitializationFailureKind::kCudaDriverUnavailable:
        status = InitializationStatus::kCudaDriverUnavailable;
        break;
      case g::cuda_native::CudaInitializationFailureKind::kRequiredSymbolUnavailable:
        status = InitializationStatus::kRequiredSymbolUnavailable;
        break;
      case g::cuda_native::CudaInitializationFailureKind::kCudaDriverFailure:
        status = InitializationStatus::kCudaDriverFailure;
        break;
      case g::cuda_native::CudaInitializationFailureKind::kCudaDeviceUnavailable:
        status = InitializationStatus::kCudaDeviceUnavailable;
        break;
    }
    return InitializationFailure(status, detail);
  }

  static HandlerFailure runtime_failure(g::cuda_native::CudaRuntimeFailureKind failure_kind,
                                        const std::string& detail) {
    const ErrorCode code = failure_kind == g::cuda_native::CudaRuntimeFailureKind::kFailedPrecondition
                               ? ErrorCode::kFailedPrecondition
                               : ErrorCode::kInternal;
    return HandlerFailure(code, detail);
  }
};

using CudaDriverApi = g::cuda_native::CudaDriverApi<CudaErrorFactory>;
using CudaDevice = g::cuda_native::CudaDevice;
using CudaModuleOwner = g::cuda_native::CudaModuleOwner<CudaDriverApi>;

class FirthComponentsKernel {
 public:
  explicit FirthComponentsKernel(const CudaDriverApi& driver)
      : driver_(driver),
        module_(driver,
                driver.load_module(kFirthComponentsKernelPtx,
                                   std::string("load Firth component ") + kFirthComponentsPtxTarget + " PTX")) {
    function_ = driver_.get_function(module_.get(), kKernelName);
  }

  FirthComponentsKernel(const FirthComponentsKernel&) = delete;
  FirthComponentsKernel& operator=(const FirthComponentsKernel&) = delete;

  void launch(const double* phenotype,
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
      fail_handler(ErrorCode::kInvalidArgument, "Firth component kernel lane count exceeds the CUDA grid domain");
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
    driver_.launch_kernel(function_,
                          static_cast<unsigned int>(lane_count),
                          kKernelBlockSize,
                          stream,
                          arguments,
                          "launch Firth component kernel");
  }

 private:
  const CudaDriverApi& driver_;
  CudaModuleOwner module_;
  CudaFunction function_ = nullptr;
};

class FirthComponentsKernelCache {
 public:
  explicit FirthComponentsKernelCache(const CudaDriverApi& driver) : driver_(driver) {}

  [[nodiscard]] const FirthComponentsKernel& for_current_context(CudaDevice qualified_device) {
    const CudaContext context = driver_.current_context();
    struct ThreadCache {
      const FirthComponentsKernelCache* owner = nullptr;
      CudaContext context = nullptr;
      const FirthComponentsKernel* kernel = nullptr;
    };
    thread_local ThreadCache thread_cache;
    if (thread_cache.owner == this && thread_cache.context == context && thread_cache.kernel != nullptr) {
      return *thread_cache.kernel;
    }

    std::scoped_lock lock(mutex_);
    auto iterator = kernels_.find(context);
    if (iterator == kernels_.end()) {
      driver_.validate_current_context_device(qualified_device,
                                              kMinimumComputeCapabilityMajor,
                                              kMinimumComputeCapabilityMinor,
                                              "CUDA Firth components require compute capability " +
                                                  std::to_string(kMinimumComputeCapabilityMajor) + "." +
                                                  std::to_string(kMinimumComputeCapabilityMinor) + " or newer");
      iterator = kernels_.emplace(context, std::make_unique<FirthComponentsKernel>(driver_)).first;
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

  void validate(std::int32_t device_ordinal, GComputeCudaCapability& capability) {
    capability.cuda_driver_version = driver_.driver_version();
    capability.device_ordinal = device_ordinal;
    if (capability.cuda_driver_version < kMinimumCudaDriverVersion) {
      fail_initialization(InitializationStatus::kCudaDriverTooOld,
                          std::string("embedded PTX ISA ") + kFirthComponentsPtxIsa +
                              " requires CUDA driver API version " + std::to_string(kMinimumCudaDriverVersion) +
                              " or newer");
    }
    const CudaDevice inspected_device = driver_.inspect_device(device_ordinal, capability);
    if (capability.compute_capability_major < kMinimumComputeCapabilityMajor ||
        (capability.compute_capability_major == kMinimumComputeCapabilityMajor &&
         capability.compute_capability_minor < kMinimumComputeCapabilityMinor)) {
      fail_initialization(InitializationStatus::kComputeCapabilityUnsupported,
                          "CUDA Firth components require compute capability " +
                              std::to_string(kMinimumComputeCapabilityMajor) + "." +
                              std::to_string(kMinimumComputeCapabilityMinor) + " or newer");
    }
    CudaDevice unqualified_device = kUnqualifiedDevice;
    if (!qualified_device_.compare_exchange_strong(unqualified_device, inspected_device) &&
        unqualified_device != inspected_device) {
      fail_initialization(InitializationStatus::kInternal,
                          "CUDA Firth runtime was already qualified for a different visible device");
    }
  }

  [[nodiscard]] const FirthComponentsKernel& kernel() {
    const CudaDevice qualified_device = qualified_device_.load();
    if (qualified_device == kUnqualifiedDevice) {
      fail_handler(ErrorCode::kFailedPrecondition,
                   "CUDA Firth handler was invoked before its execution device was qualified");
    }
    return kernels_.for_current_context(qualified_device);
  }

 private:
  static constexpr CudaDevice kUnqualifiedDevice = -1;
  CudaDriverApi driver_;
  std::atomic<CudaDevice> qualified_device_{kUnqualifiedDevice};
  FirthComponentsKernelCache kernels_;
};

RuntimeState& runtime_state() {
  // JAX owns the CUDA contexts and can destroy them before C++ static
  // destructors run. Keep the driver library and context-bound modules alive
  // until process exit instead of attempting context-unsafe teardown.
  static RuntimeState* const state = new RuntimeState();
  return *state;
}

bool dimensions_equal(const AnyBuffer& left, const AnyBuffer& right) { return left.dimensions() == right.dimensions(); }

bool has_lane_dimensions(const AnyBuffer& buffer, DataType data_type, const AnyBuffer& output) {
  return buffer.element_type() == data_type && dimensions_equal(buffer, output);
}

bool has_sample_dimensions(const AnyBuffer& buffer,
                           DataType data_type,
                           const AnyBuffer& output,
                           std::size_t sample_count) {
  if (buffer.element_type() != data_type || buffer.dimensions().size() != output.dimensions().size() + 1 ||
      buffer.dimensions().back() < 0 || static_cast<std::size_t>(buffer.dimensions().back()) != sample_count) {
    return false;
  }
  for (std::size_t dimension_index = 0; dimension_index < output.dimensions().size(); ++dimension_index) {
    if (buffer.dimensions()[dimension_index] != output.dimensions()[dimension_index]) {
      return false;
    }
  }
  return true;
}

std::size_t checked_lane_count(const AnyBuffer& output) {
  std::size_t lane_count = 1;
  for (const std::int64_t dimension : output.dimensions()) {
    if (dimension <= 0 ||
        static_cast<std::uint64_t>(dimension) > std::numeric_limits<std::size_t>::max() / lane_count) {
      fail_handler(ErrorCode::kInvalidArgument, "Firth component output dimensions are empty or overflow size_t");
    }
    lane_count *= static_cast<std::size_t>(dimension);
  }
  return lane_count;
}

Error compute_firth_components(AnyBuffer phenotype,
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
      return Error::InvalidArgument("Firth component sample inputs must have a nonempty final dimension");
    }
    const std::size_t sample_count = static_cast<std::size_t>(phenotype.dimensions().back());
    const bool outputs_match = has_lane_dimensions(*genotype_information, DataType::F64, beta) &&
                               has_lane_dimensions(*score_adjustment, DataType::F64, beta) &&
                               has_lane_dimensions(*penalized_deviance, DataType::F64, beta) &&
                               has_lane_dimensions(*score, DataType::F64, beta) &&
                               has_lane_dimensions(*valid, DataType::PRED, beta);
    const bool sample_inputs_match = has_sample_dimensions(phenotype, DataType::F64, beta, sample_count) &&
                                     has_sample_dimensions(genotype, DataType::F64, beta, sample_count) &&
                                     has_sample_dimensions(offset, DataType::F64, beta, sample_count) &&
                                     has_sample_dimensions(active_sample_mask, DataType::PRED, beta, sample_count);
    const bool lane_inputs_match = has_lane_dimensions(non_active_deviance, DataType::F64, beta) &&
                                   has_lane_dimensions(beta, DataType::F64, beta) &&
                                   has_lane_dimensions(minimum_variance, DataType::F64, beta);
    if (!outputs_match || !sample_inputs_match || !lane_inputs_match) {
      return Error::InvalidArgument(
          "Firth component operands and results must use matching batch prefixes, "
          "f64 values, and boolean masks");
    }
    const std::size_t lane_count = checked_lane_count(beta);
    runtime_state().kernel().launch(phenotype.typed_data<double>(),
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
    return Error(ErrorCode::kInternal, "unknown native Firth component handler failure");
  }
}

}  // namespace

extern "C" std::int32_t g_compute_cuda_initialize_firth_components_runtime(std::int32_t device_ordinal,
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(g_firth_components_ffi,
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
