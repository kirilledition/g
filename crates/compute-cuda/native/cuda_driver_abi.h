#ifndef G_COMPUTE_CUDA_DRIVER_ABI_H_
#define G_COMPUTE_CUDA_DRIVER_ABI_H_

#include <cstdint>

namespace g::compute_cuda::abi {

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

}  // namespace g::compute_cuda::abi

#endif  // G_COMPUTE_CUDA_DRIVER_ABI_H_
