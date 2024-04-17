/**
 * @file   precondTiming_cuda.cpp
 * @author Zhili Xiong (DREAMPlaceFPGA)
 * @date   Aug 2023
 * @brief  Compute precond Timing
 */
#include "utility/src/torch.h"
#include "utility/src/Msg.h"
using namespace torch::indexing;

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computePrecondTimingCudaLauncher(
        const int *flat_tnet2pin,
        const int *pin2node_map, 
        const T *tnet_weights, 
        int num_tnets, 
        int num_nodes,
        const T xl, const T yl,
        const T xh, const T yh,
        bool deterministic_flag,
        T *out 
        );

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")


/// @brief Compute half-perimeter wirelength along with net bbox 
/// @param pos cell locations, array of x locations and then y locations 
/// @param flat_netpin similar to the JA array in CSR format, which is flattened from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is the number of pins in each net, the length of IA is number of nets + 1
/// @param net_mask an array to record whether compute the where for a net or not 
void forward(
        at::Tensor flat_tnet2pin,
        at::Tensor pin2node_map,
        at::Tensor tnet_weights,
        int num_tnets,
        double xl, double yl,
        double xh, double yh, 
        int deterministic_flag,
        at::Tensor out
        ) 
{
    CHECK_FLAT(flat_tnet2pin); 
    CHECK_CONTIGUOUS(flat_tnet2pin);
    CHECK_FLAT(pin2node_map);
    CHECK_CONTIGUOUS(pin2node_map);

    CHECK_FLAT(tnet_weights);
    CHECK_CONTIGUOUS(tnet_weights);

    int num_nodes = out.numel();

    DREAMPLACE_DISPATCH_FLOATING_TYPES(out, "computePrecondTimingCudaLauncher", [&] {
            computePrecondTimingCudaLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(flat_tnet2pin, int),
                    DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int), 
                    DREAMPLACE_TENSOR_DATA_PTR(tnet_weights, scalar_t), 
                    num_tnets, num_nodes, xl, yl, xh, yh,
                    (bool)deterministic_flag,
                    DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t)
                    );
            });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::forward, "PrecondTiming forward (CUDA)");
}
