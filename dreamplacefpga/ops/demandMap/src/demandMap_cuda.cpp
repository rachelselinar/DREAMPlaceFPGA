/**
 * @file   demandMap_cuda.cpp
 * @author Rachel Selina Rajarathnam (DREAMPlaceFPGA) 
 * @date   Nov 2020
 * @brief  Compute binCapMap
 * 
 * Modifications Copyright(C) 2024 Advanced Micro Devices, Inc. All rights reserved
 *
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"
using namespace torch::indexing;

DREAMPLACE_BEGIN_NAMESPACE

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
        T *binCapMap1,
        T *binCapMap4,
        T *binCapMap5
        );

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")


/// @brief Compute bin capacity map
void forward(
        at::Tensor site_type_map, 
        at::Tensor node_size_x, 
        at::Tensor node_size_y, 
        int num_bins_x,
        int num_bins_y,
        int width, 
        int height, 
        int deterministic_flag,
        at::Tensor binCapMap0,
        at::Tensor binCapMap1,
        at::Tensor binCapMap4,
        at::Tensor binCapMap5)
{
    CHECK_FLAT(site_type_map); 
    CHECK_CONTIGUOUS(site_type_map);
    CHECK_FLAT(node_size_x);
    CHECK_CONTIGUOUS(node_size_x);
    CHECK_FLAT(node_size_y);
    CHECK_CONTIGUOUS(node_size_y);

    DREAMPLACE_DISPATCH_FLOATING_TYPES(node_size_x, "computeDemandMapCudaLauncher", [&] {
            computeDemandMapCudaLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(site_type_map, int),
                    DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t), 
                    num_bins_x, num_bins_y,
                    width, height,
                    deterministic_flag,
                    DREAMPLACE_TENSOR_DATA_PTR(binCapMap0, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(binCapMap1, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(binCapMap4, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(binCapMap5, scalar_t)
                    );
            });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::forward, "DemandMap forward (CUDA)");
}
