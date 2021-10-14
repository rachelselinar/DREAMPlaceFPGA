#include "utility/src/utils.cuh"
#include "utility/src/limits.cuh"
// local dependency
#include "demandMap/src/atomic_ops.cuh"
#include "demandMap/src/demand_function.h"

DREAMPLACE_BEGIN_NAMESPACE

/// define compute_demand_function 
template <typename T>
inline __device__ DEFINE_COMPUTE_DEMAND_FUNCTION(T);

template <typename T, typename AtomicOp>
__global__ void __launch_bounds__(1024, 8) computeDemandMap(
        const int *site_type_map, const int num_bins_x, const int num_bins_y, 
        const int width, const int height, const T *node_size_x, 
        const T *node_size_y, AtomicOp atomicAddOp,
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
        const int *site_type_map, const int num_bins_x, const int num_bins_y, 
        const int width, const int height, const T *node_size_x, 
        const T *node_size_y, AtomicOp atomicAddOp,
        typename AtomicOp::type *binCapMap0,
        typename AtomicOp::type *binCapMap2,
        typename AtomicOp::type *binCapMap3
        )
{
  int thread_count = 64;
  dim3 blockSize(2, 2, thread_count);

  int block_count = (width*height - 1 + thread_count) / thread_count;

    computeDemandMap<<<block_count, blockSize>>>(
            site_type_map, num_bins_x, num_bins_y, width, height,
            node_size_x, node_size_y, atomicAddOp, binCapMap0,
            binCapMap2, binCapMap3);

    return 0;
}


template <typename T>
int computeDemandMapCudaLauncher(
        const int *site_type_map,
        const int num_bins_x, 
        const int num_bins_y, 
        const int width, 
        const int height, 
        const T *node_size_x, 
        const T *node_size_y, 
        T *binCapMap0,
        T *binCapMap2,
        T *binCapMap3
        )
{
    AtomicAdd<T> atomicAddOp;

    computeDemandMapCallKernel<T, decltype(atomicAddOp)>(
            site_type_map, num_bins_x, num_bins_y,
            width, height, node_size_x, node_size_y,
            atomicAddOp, binCapMap0, binCapMap2, binCapMap3);

    return 0;
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T)                         \
    int instantiatecomputeDemandMapLauncher(                \
        const int *site_type_map, const int num_bins_x,     \
        const int num_bins_y, const int width,              \
        const int height, const T *node_size_x,             \
        const T *node_size_y, T *binCapMap0, T *binCapMap2, \
        T *binCapMap3) {                                    \
        return computeDemandMapCudaLauncher(                \
                site_type_map, num_bins_x, num_bins_y,      \
                width, height, node_size_x, node_size_y,    \
                binCapMap0, binCapMap2, binCapMap3);        \
    }

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
