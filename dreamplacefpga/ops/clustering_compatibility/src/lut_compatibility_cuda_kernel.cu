/**
 * @file   lut_compatibility_cuda_kernel.cu
 * @author Rachel Selina Rajarathnam (DREAMPlaceFPGA)
 * @date   Oct 2020
 * @brief  Compute the Clustering compatibility map for LUT based on elfPlace.
 */

#include "utility/src/utils.cuh"
#include "utility/src/limits.h"
// local dependency
#include "clustering_compatibility/src/functions.h"

DREAMPLACE_BEGIN_NAMESPACE

/// define gaussian_auc_function 
template <typename T>
inline __device__ DEFINE_GAUSSIAN_AUC_FUNCTION(T);
/// define lut_compute_areas_function
template <typename T>
inline __device__ DEFINE_LUT_COMPUTE_AREAS_FUNCTION(T);

template <typename T>
__global__ void fillDemandMapLUT(const T *pos_x,
                                 const T *pos_y,
                                 const int *indices,
                                 const int *type,
                                 const T *node_size_x,
                                 const T *node_size_y,
                                 const int num_bins_x, 
                                 const int num_bins_y,
                                 const int num_bins_l,
                                 int num_nodes, T stddev_x, T stddev_y,
                                 T inv_stddev_x, T inv_stddev_y, 
                                 int ext_bin, T inv_sqrt,
                                 T *demMap)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num_nodes)
    {
        const int idx = indices[i];
        T node_x = pos_x[idx] + 0.5 * node_size_x[idx];
        T node_y = pos_y[idx] + 0.5 * node_size_y[idx];
        int lutType = type[idx];

        int binX = int(node_x * inv_stddev_x);
        int binY = int(node_y * inv_stddev_y);

        // compute the bin box that this net will affect
        int bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(binX - ext_bin, 0);
        int bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(binX + ext_bin + 1, num_bins_x);

        int bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(binY - ext_bin, 0);
        int bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(binY + ext_bin + 1, num_bins_y);

        T gaussianX = gaussian_auc_function(node_x, stddev_x, bin_index_xl*stddev_x, bin_index_xh* stddev_x, inv_sqrt); 
        T gaussianY = gaussian_auc_function(node_y, stddev_y, bin_index_yl*stddev_y, bin_index_yh* stddev_y, inv_sqrt); 

        T sf = 1.0 / (gaussianX * gaussianY);

        for (int x = bin_index_xl; x < bin_index_xh; ++x)
        {
            T dem_xmbin_index_xl = gaussian_auc_function(node_x, stddev_x, x*stddev_x, (x+1)*stddev_x, inv_sqrt);
            for (int y = bin_index_yl; y < bin_index_yh; ++y)
            {
                //#pragma omp atomic update
                int index = x * (num_bins_y * num_bins_l) + y * num_bins_l + lutType;
                T dem_ymbin_index_yl = gaussian_auc_function(node_y, stddev_y, y*stddev_y, (y+1)*stddev_y, inv_sqrt);
                //T dem = sf * demandX[x - bin_index_xl] * demandY[y - bin_index_yl];
                T dem = sf * dem_xmbin_index_xl * dem_ymbin_index_yl;
                //demMap[index] += dem;
                atomicAdd(demMap + index, dem);
            }
        }
    }
}

// Given a Gaussian demand map, compute demand of each instance type based on local window demand distribution
template <typename T>
__global__ void computeInstanceAreaLUT(const T *demMap,
                                       const int num_bins_x, 
                                       const int num_bins_y,
                                       const int num_bins_l,
                                       T stddev_x, T stddev_y,
                                       int ext_bin, T bin_area, 
                                       T *areaMap)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    int total_bins = num_bins_x*num_bins_y;
    if (i < total_bins)
    {
        int binX = int(i/num_bins_y);
        int binY = int(i%num_bins_y);

        // compute the bin box
        int bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(binX - ext_bin, 0);
        int bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(binX + ext_bin + 1, num_bins_x);

        int bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(binY - ext_bin, 0);
        int bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(binY + ext_bin + 1, num_bins_y);

        int idx = binX * (num_bins_y * num_bins_l) + binY * num_bins_l;

        for (int x = bin_index_xl; x < bin_index_xh; ++x)
        {
            for (int y = bin_index_yl; y < bin_index_yh; ++y)
            {
                int index = x * (num_bins_y * num_bins_l) + y * num_bins_l;
                if (x != binX && y != binY)
                {
                    for (int l = 0; l < num_bins_l; ++l)
                    {
                        areaMap[idx + l] += demMap[index + l];
                    }
                }
            }
        }
        
        // Compute instance areas based on the window demand distribution
        T winArea = (bin_index_xh - bin_index_xl) * (bin_index_yh - bin_index_yl) * bin_area;
        lut_compute_areas_function(winArea, areaMap, idx, num_bins_l);

    }
}


// Set a set of instance area in a area vector based on the given area map
template <typename T>
__global__ void collectInstanceAreasLUT(const T *pos_x,
                                        const T *pos_y,
                                        const int *indices,
                                        const int *type,
                                        const T *node_size_x,
                                        const T *node_size_y,
                                        const int num_bins_y,
                                        const int num_bins_l,
                                        const T *areaMap,
                                        int num_nodes,
                                        T inv_stddev_x, T inv_stddev_y,
                                        T *instAreas)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num_nodes)
    {
        const int idx = indices[i];
        T node_x = pos_x[idx] + 0.5 * node_size_x[idx];
        T node_y = pos_y[idx] + 0.5 * node_size_y[idx];

        int binX = int(node_x * inv_stddev_x);
        int binY = int(node_y * inv_stddev_y);

        int index = binX * (num_bins_y * num_bins_l) + binY * num_bins_l + type[idx];
        instAreas[idx] = areaMap[index];
    }
}

// fill the demand map net by net
template <typename T>
int fillDemandMapLUTCuda(const T *pos_x, const T *pos_y,
                         const int *indices, const int *type,
                         const T *node_size_x, const T *node_size_y,
                         const int num_bins_x, const int num_bins_y, 
                         const int num_bins_l, int num_nodes,
                         T stddev_x, T stddev_y,
                         T inv_stddev_x, T inv_stddev_y, 
                         int ext_bin, T inv_sqrt,
                         T *demMap)
{
    int thread_count = 512;
    int block_count = ceilDiv(num_nodes, thread_count);
    fillDemandMapLUT<<<block_count, thread_count>>>(
                                                    pos_x,
                                                    pos_y,
                                                    indices,
                                                    type,
                                                    node_size_x,
                                                    node_size_y,
                                                    num_bins_x, num_bins_y,
                                                    num_bins_l,
                                                    num_nodes, stddev_x, stddev_y,
                                                    inv_stddev_x, inv_stddev_y, 
                                                    ext_bin, inv_sqrt,
                                                    demMap
                                                    );
    return 0;
}

// Given a Gaussian demand map, compute demand of each instance type based on local window demand distribution
template <typename T>
int computeInstanceAreaLUTCuda(const T *demMap,
                               const int num_bins_x, 
                               const int num_bins_y,
                               const int num_bins_l,
                               T stddev_x, T stddev_y,
                               int ext_bin, T bin_area, 
                               T *areaMap)
{
    int thread_count = 512;
    int block_count = ceilDiv(num_bins_x*num_bins_y, thread_count);
    computeInstanceAreaLUT<<<block_count, thread_count>>>(
                                                          demMap,
                                                          num_bins_x, num_bins_y,
                                                          num_bins_l,
                                                          stddev_x, stddev_y,
                                                          ext_bin, bin_area, areaMap
                                                          );
    return 0;
}


// Set a set of instance area in a area vector based on the given area map
template <typename T>
int collectInstanceAreasLUTCuda(const T *pos_x,
                                const T *pos_y,
                                const int *indices,
                                const int *type,
                                const T *node_size_x,
                                const T *node_size_y,
                                const int num_bins_y,
                                const int num_bins_l,
                                const T *areaMap,
                                int num_nodes,
                                T inv_stddev_x, T inv_stddev_y,
                                T *instAreas)
{
    int thread_count = 512;
    int block_count = ceilDiv(num_nodes, thread_count);
    collectInstanceAreasLUT<<<block_count, thread_count>>>(
                                                           pos_x, pos_y,
                                                           indices, type, 
                                                           node_size_x, node_size_y,
                                                           num_bins_y, num_bins_l,
                                                           areaMap, num_nodes,
                                                           inv_stddev_x, inv_stddev_y,
                                                           instAreas
                                                           );
    return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                                            \
    template int fillDemandMapLUTCuda<T>(                                      \
        const T *pos_x, const T *pos_y, const int *indices, const int *type,   \
        const T *node_size_x, const T *node_size_y, int num_bins_x,            \
        int num_bins_y, int num_bins_l, int num_nodes, T stddev_x, T stddev_y, \
        T inv_stddev_x, T inv_stddev_y, int ext_bin, T inv_sqrt,               \
        T *demMap);                                                            \
                                                                               \
    template int computeInstanceAreaLUTCuda<T>(                                \
        const T *demMap, const int num_bins_x, const int num_bins_y,           \
        const int num_bins_l, T stddev_x, T stddev_y,                          \
        int ext_bin, T bin_area, T *areaMap);                                  \
                                                                               \
    template int collectInstanceAreasLUTCuda<T>(                               \
        const T *pos_x, const T *pos_y, const int *indices, const int *type,   \
        const T *node_size_x, const T *node_size_y, int num_bins_y,            \
        int num_bins_l, const T *areaMap, int num_nodes, T inv_stddev_x,       \
        T inv_stddev_y, T *instAreas);

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
