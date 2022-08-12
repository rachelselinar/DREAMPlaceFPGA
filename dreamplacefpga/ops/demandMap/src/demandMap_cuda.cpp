/**
 * @file   demandMap_cuda.cpp
 * @author Rachel Selina Rajarathnam (DREAMPlaceFPGA) 
 * @date   Nov 2020
 * @brief  Compute binCapMap
 */
#include "utility/src/torch.h"
#include "utility/src/Msg.h"
using namespace torch::indexing;

DREAMPLACE_BEGIN_NAMESPACE

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
        );

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")


/// @brief Compute bin capacity map
void forward(
        at::Tensor site_type_map, 
        int num_bins_x,
        int num_bins_y,
        int width, 
        int height, 
        at::Tensor node_size_x, 
        at::Tensor node_size_y, 
        at::Tensor binCapMap0,
        at::Tensor binCapMap2,
        at::Tensor binCapMap3)
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
                    num_bins_x, num_bins_y,
                    width, height,
                    DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(binCapMap0, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(binCapMap2, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(binCapMap3, scalar_t)
                    );
            });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::forward, "DemandMap forward (CUDA)");
}
