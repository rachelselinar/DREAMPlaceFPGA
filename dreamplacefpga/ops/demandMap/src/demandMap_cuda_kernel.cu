#include "utility/src/utils.cuh"
#include "utility/src/limits.h"
// local dependency
#include "demandMap/src/demand_function.h"

DREAMPLACE_BEGIN_NAMESPACE

/// define compute_demand_function 
template <typename T>
inline __device__ DEFINE_COMPUTE_DEMAND_FUNCTION(T);

template <typename T, typename AtomicOp>
__global__ void __launch_bounds__(1024, 8) computeDemandMap(
        const int *site_type_map, const T *node_size_x, const T *node_size_y,
        const int num_bins_x, const int num_bins_y, 
        const int width, const int height,
        AtomicOp atomicAddOp,
        typename AtomicOp::type *binCapMap0,
        typename AtomicOp::type *binCapMap2,
        typename AtomicOp::type *binCapMap3)
{
    T binW = T(width)/T(num_bins_x);
    T binH = T(height)/T(num_bins_y);

    int idx = blockIdx.x * blockDim.z + threadIdx.z;
    if (idx < width*height)
    {
            int rw = int(idx/height);
            int cl = int(idx%height);

        if (site_type_map[idx] == 1)
        {
            T nodeX = node_size_x[1];
            T nodeY = node_size_y[1];
            T col = DREAMPLACE_STD_NAMESPACE::round(cl/nodeY)*nodeY;
            int iLo = int(rw/binW);
            int jLo = int(col/binH);
            int iHi = DREAMPLACE_STD_NAMESPACE::min(int((rw + nodeX)/binW), num_bins_x-1);
            int jHi = DREAMPLACE_STD_NAMESPACE::min(int((col + nodeY)/binH), num_bins_y-1);

            for (int i = iLo + threadIdx.y; i <= iHi; i += blockDim.y)
            {
                T w = compute_demand_function(i, binW, T(rw), nodeX);
                for (int j = jLo + threadIdx.x; j <= jHi; j += blockDim.x)
                {
                    T h = compute_demand_function(j, binH, col, nodeY);
                    T area = w * h;
                    atomicAddOp(&binCapMap0[i*num_bins_y + j], area);
                }
            }
        } else if (site_type_map[idx] == 2)
        {
            T nodeX = node_size_x[2];
            T nodeY = node_size_y[2];
            T col = DREAMPLACE_STD_NAMESPACE::round(cl/nodeY)*nodeY;
            int iLo = int(rw/binW);
            int jLo = int(col/binH);
            int iHi = DREAMPLACE_STD_NAMESPACE::min(int((rw + nodeX)/binW), num_bins_x-1);
            int jHi = DREAMPLACE_STD_NAMESPACE::min(int((col + nodeY)/binH), num_bins_y-1);

            for (int i = iLo + threadIdx.y; i <= iHi; i += blockDim.y)
            {
                T w = compute_demand_function(i, binW, T(rw), nodeX);
                for (int j = jLo + threadIdx.x; j <= jHi; j += blockDim.x)
                {
                    T h = compute_demand_function(j, binH, col, nodeY);
                    T area = w * h;
                    atomicAddOp(&binCapMap2[i*num_bins_y + j], area);
                }
            }
        } else if (site_type_map[idx] == 3)
        {
            T nodeX = node_size_x[3];
            T nodeY = node_size_y[3];
            T col = DREAMPLACE_STD_NAMESPACE::round(cl/nodeY)*nodeY;
            int iLo = int(rw/binW);
            int jLo = int(col/binH);
            int iHi = DREAMPLACE_STD_NAMESPACE::min(int((rw + nodeX)/binW), num_bins_x-1);
            int jHi = DREAMPLACE_STD_NAMESPACE::min(int((col + nodeY)/binH), num_bins_y-1);
            for (int i = iLo + threadIdx.y; i <= iHi; i += blockDim.y)
            {
                T w = compute_demand_function(i, binW, T(rw), nodeX);
                for (int j = jLo + threadIdx.x; j <= jHi; j += blockDim.x)
                {
                    T h = compute_demand_function(j, binH, col, nodeY);
                    T area = w * h;
                    atomicAddOp(&binCapMap3[i*num_bins_y + j], area);
                }
            }
        }
    }
}



template <typename T, typename AtomicOp>
int computeDemandMapCallKernel(
        const int *site_type_map, const T *node_size_x,
        const T *node_size_y, const int num_bins_x,
        const int num_bins_y, const int width, const int height,
        AtomicOp atomicAddOp,
        typename AtomicOp::type *binCapMap0,
        typename AtomicOp::type *binCapMap2,
        typename AtomicOp::type *binCapMap3
        )
{
  int thread_count = 64;
  dim3 blockSize(2, 2, thread_count);

  int block_count = (width*height - 1 + thread_count) / thread_count;

    computeDemandMap<<<block_count, blockSize>>>(
            site_type_map, node_size_x, node_size_y,
            num_bins_x, num_bins_y, width, height,
            atomicAddOp, binCapMap0,
            binCapMap2, binCapMap3);

    return 0;
}


template <typename T>
int computeDemandMapCudaLauncher(
        const int *site_type_map,
        const T *node_size_x, 
        const T *node_size_y, 
        const int num_bins_x, 
        const int num_bins_y, 
        const int width, 
        const int height, 
        const int deterministic_flag,
        T *binCapMap0,
        T *binCapMap2,
        T *binCapMap3
        )
{
    if (deterministic_flag == 1)
    {
    // total die area
    double diearea = width * height;
    int integer_bits = max((int)ceil(log2(diearea)) + 1, 32);
    int fraction_bits = max(64 - integer_bits, 0);
    unsigned long long int scale_factor = (1UL << fraction_bits);
    int num_bins = num_bins_x * num_bins_y;

    unsigned long long int *bin_cap_map_0 = NULL;
    allocateCUDA(bin_cap_map_0, num_bins, unsigned long long int);
    unsigned long long int *bin_cap_map_2 = NULL;
    allocateCUDA(bin_cap_map_2, num_bins, unsigned long long int);
    unsigned long long int *bin_cap_map_3 = NULL;
    allocateCUDA(bin_cap_map_3, num_bins, unsigned long long int);

    AtomicAddCUDA<unsigned long long int> atomicAddOp(scale_factor);
    int thread_count = 512;

    copyScaleArray<<<(num_bins + thread_count - 1) / thread_count,
                     thread_count>>>(
        bin_cap_map_0, binCapMap0, scale_factor, num_bins);
    copyScaleArray<<<(num_bins + thread_count - 1) / thread_count,
                     thread_count>>>(
        bin_cap_map_2, binCapMap2, scale_factor, num_bins);
    copyScaleArray<<<(num_bins + thread_count - 1) / thread_count,
                     thread_count>>>(
        bin_cap_map_3, binCapMap3, scale_factor, num_bins);

    computeDemandMapCallKernel<T, decltype(atomicAddOp)>(
                site_type_map, node_size_x, node_size_y,
                num_bins_x, num_bins_y, width, height,
                atomicAddOp, bin_cap_map_0, bin_cap_map_2, bin_cap_map_3);

    copyScaleArray<<<(num_bins + thread_count - 1) / thread_count,
                     thread_count>>>(binCapMap0,
                     bin_cap_map_0, T(1.0 / scale_factor), num_bins);
    copyScaleArray<<<(num_bins + thread_count - 1) / thread_count,
                     thread_count>>>(binCapMap2,
                     bin_cap_map_2, T(1.0 / scale_factor), num_bins);
    copyScaleArray<<<(num_bins + thread_count - 1) / thread_count,
                     thread_count>>>(binCapMap3,
                     bin_cap_map_3, T(1.0 / scale_factor), num_bins);

    destroyCUDA(bin_cap_map_0);
    destroyCUDA(bin_cap_map_2);
    destroyCUDA(bin_cap_map_3);
  } else
    {
        AtomicAddCUDA<T> atomicAddOp;

        computeDemandMapCallKernel<T, decltype(atomicAddOp)>(
                site_type_map, node_size_x, node_size_y,
                num_bins_x, num_bins_y, width, height,
                atomicAddOp, binCapMap0, binCapMap2, binCapMap3);
    }
    return 0;
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                         \
    template int computeDemandMapCudaLauncher<T>(           \
        const int *site_type_map, const T *node_size_x,     \
        const T *node_size_y, const int num_bins_x,         \
        const int num_bins_y, const int width,              \
        const int height, const int deterministic_flag,     \
        T *binCapMap0, T *binCapMap2, T *binCapMap3);

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
