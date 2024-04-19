/**
 * @file   timing_net_wirelength_cuda_merged.cpp
 * @author Zhili Xiong (DREAMPlaceFPGA)
 * @date   Aug 2023
 * @brief  Compute timing-net wirelength and gradient according to e-place
 */
#include "utility/src/torch.h"
#include "utility/src/Msg.h"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief Compute weighted average wirelength and gradient.
/// WL = \sum_i x_i*exp(x_i/gamma) / \sum_i exp(x_i/gamma) - \sum_i x_i*exp(-x_i/gamma) / \sum_i x_i*exp(-x_i/gamma),
/// where x_i is pin location.
///
/// @param x x location of pins.
/// @param y y location of pins.
/// @param flat_tnetpin consists pins of each net, pins belonging to the same net are abutting to each other.
/// @param tnet_weights weight of nets
/// @param num_tnets number of nets.
/// @param inv_gamma the inverse number of gamma coefficient in weighted average wirelength.
/// @param partial_wl wirelength in x and y directions of each net. The first half is the wirelength in x direction, and the second half is the wirelength in y direction.
/// @param grad_tensor back-propagated gradient from previous stage.
/// @param grad_x_tensor gradient in x direction.
/// @param grad_y_tensor gradient in y direction.
/// @return 0 if successfully done.
template <typename T>
int computeTimingNetWirelengthCudaMergedLauncher(
        const T* x, const T* y, 
        const int* flat_tnetpin, 
        const T* tnet_weights,
        int num_tnets,
        int num_pins,
        const T* inv_gamma, 
        T* partial_wl,
        const T xl, const T yl, 
        const T xh, const T yh,
        bool deterministic_flag,
        T* grad_intermediate_x, T* grad_intermediate_y
    );


/// @brief Compute weighted average wirelength and gradient.
/// WL = \sum_i x_i*exp(x_i/gamma) / \sum_i exp(x_i/gamma) - \sum_i x_i*exp(-x_i/gamma) / \sum_i x_i*exp(-x_i/gamma),
/// where x_i is pin location.
///
/// @param x x location of pins.
/// @param y y location of pins.
/// @param flat_tnetpin consists pins of each net, pins belonging to the same net are abutting to each other.
/// @param tnet_weights weight of nets
/// @param num_tnets number of nets.
/// @param inv_gamma the inverse number of gamma coefficient in weighted average wirelength.
/// @param partial_wl wirelength in x and y directions of each net. The first half is the wirelength in x direction, and the second half is the wirelength in y direction.
/// @param grad_tensor back-propagated gradient from previous stage.
/// @param grad_x_tensor gradient in x direction.
/// @param grad_y_tensor gradient in y direction.
/// @return 0 if successfully done.
template <typename T>
int computeTimingNetWirelengthCudaMergedLauncherFPGA(
        const T* x, const T* y, 
        const int* flat_tnetpin, 
        const T* tnet_weights,
        int num_tnets,
        int num_pins,
        const T* inv_gamma, 
        const T* bbox_min_x, const T* bbox_min_y,
        const T* bbox_max_x, const T* bbox_max_y,
        T* partial_wl,
        const T xl, const T yl, 
        const T xh, const T yh,
        bool deterministic_flag,
        T* grad_intermediate_x, T* grad_intermediate_y
    );

// /// @brief add net weights to gradient
// template <typename T>
// void integrateTimingNetWeightsCudaLauncher(
//     const T *tnet_weights,
//     T *grad_x_tensor, T *grad_y_tensor,
//     int num_pins);

#define CHECK_FLAT(x) AT_ASSERTM(x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on GPU")
#define CHECK_EVEN(x) AT_ASSERTM((x.numel() & 1) == 0, #x "must have even number of elements")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// @brief Compute weighted average wirelength and gradient.
/// WL = \sum_i x_i*exp(x_i/gamma) / \sum_i exp(x_i/gamma) - \sum_i x_i*exp(-x_i/gamma) / \sum_i x_i*exp(-x_i/gamma),
/// where x_i is pin location.
///
/// @param pos location of pins, x array followed by y array.
/// @param flat_tnetpin consists pins of each net, pins belonging to the same net are abutting to each other.
/// @param tnet_weights weight of nets
/// @param inv_gamma the inverse number of gamma coefficient in weighted average wirelength.
/// @return total wirelength cost.
std::vector<at::Tensor> timing_net_wirelength_forward(
    at::Tensor pos,
    at::Tensor flat_tnetpin,
    at::Tensor tnet_weights,
    at::Tensor inv_gamma,
    double xl, double yl,
    double xh, double yh, 
    int deterministic_flag)
{
    CHECK_FLAT(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(flat_tnetpin);
    CHECK_CONTIGUOUS(flat_tnetpin);
    CHECK_FLAT(tnet_weights);
    CHECK_CONTIGUOUS(tnet_weights);

    int num_tnets = flat_tnetpin.numel() / 2;
    int num_pins = pos.numel() / 2;
    
    // x, y interleave 
    at::Tensor partial_wl = at::zeros({num_tnets, 2}, pos.options());
    // timed with grad_in yet 
    at::Tensor grad_intermediate = at::zeros_like(pos);

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeTimingNetWirelengthCudaMergedLauncher", [&] {
        computeTimingNetWirelengthCudaMergedLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
            DREAMPLACE_TENSOR_DATA_PTR(flat_tnetpin, int),
            DREAMPLACE_TENSOR_DATA_PTR(tnet_weights, scalar_t),
            num_tnets,
            num_pins,
            DREAMPLACE_TENSOR_DATA_PTR(inv_gamma, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(partial_wl, scalar_t),
            xl, yl, 
            xh, yh,
            (bool)deterministic_flag,
            DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t) + num_pins
            );
        if (tnet_weights.numel())
        {
            partial_wl.mul_(tnet_weights.view({num_tnets, 1}));
        }
    });

    auto wl = partial_wl.sum();
    //at::Tensor wl = at::zeros(1, pos.options());
    return {wl, grad_intermediate};
}

/// @brief Compute weighted average wirelength and gradient.
/// WL = \sum_i x_i*exp(x_i/gamma) / \sum_i exp(x_i/gamma) - \sum_i x_i*exp(-x_i/gamma) / \sum_i x_i*exp(-x_i/gamma),
/// where x_i is pin location.
///
/// @param pos location of pins, x array followed by y array.
/// @param flat_tnetpin consists pins of each net, pins belonging to the same net are abutting to each other.
/// @param tnet_weights weight of nets
/// @param inv_gamma the inverse number of gamma coefficient in weighted average wirelength.
/// @return total wirelength cost.
std::vector<at::Tensor> timing_net_wirelength_forward_fpga(
    at::Tensor pos,
    at::Tensor flat_tnetpin,
    at::Tensor tnet_weights,
    at::Tensor inv_gamma,
    at::Tensor net_bounding_box_min,
    at::Tensor net_bounding_box_max,
    double xl, double yl,
    double xh, double yh, 
    int deterministic_flag)
{
    CHECK_FLAT(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(flat_tnetpin);
    CHECK_CONTIGUOUS(flat_tnetpin);
    CHECK_FLAT(tnet_weights);
    CHECK_CONTIGUOUS(tnet_weights);
    CHECK_FLAT(net_bounding_box_min);
    CHECK_EVEN(net_bounding_box_min);
    CHECK_CONTIGUOUS(net_bounding_box_min);
    CHECK_FLAT(net_bounding_box_max);
    CHECK_EVEN(net_bounding_box_max);
    CHECK_CONTIGUOUS(net_bounding_box_max);

    int num_tnets = flat_tnetpin.numel() / 2;
    int num_pins = pos.numel() / 2;
    
    // x, y interleave 
    at::Tensor partial_wl = at::zeros({num_tnets, 2}, pos.options());
    // timed with grad_in yet 
    at::Tensor grad_intermediate = at::zeros_like(pos);

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeTimingNetWirelengthCudaMergedLauncherFPGA", [&] {
        computeTimingNetWirelengthCudaMergedLauncherFPGA<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
            DREAMPLACE_TENSOR_DATA_PTR(flat_tnetpin, int),
            DREAMPLACE_TENSOR_DATA_PTR(tnet_weights, scalar_t),
            num_tnets,
            num_pins,
            DREAMPLACE_TENSOR_DATA_PTR(inv_gamma, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(net_bounding_box_min, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(net_bounding_box_min, scalar_t) + num_tnets,
            DREAMPLACE_TENSOR_DATA_PTR(net_bounding_box_max, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(net_bounding_box_max, scalar_t) + num_tnets,
            DREAMPLACE_TENSOR_DATA_PTR(partial_wl, scalar_t),
            xl, yl,
            xh, yh, 
            (bool)deterministic_flag,
            DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t), DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t) + num_pins
            );
        if (tnet_weights.numel())
        {
            partial_wl.mul_(tnet_weights.view({num_tnets, 1}));
        }
    });

    auto wl = partial_wl.sum();
    //at::Tensor wl = at::zeros(1, pos.options());
    return {wl, grad_intermediate};
}

/// @brief Compute gradient
/// @param grad_pos input gradient from backward propagation
/// @param pos locations of pins
/// @param flat_tnetpin similar to the JA array in CSR format, which is flattened from the net2pin map (array of array)
/// @param tnet_weights weight of timing nets
/// @param inv_gamma a scalar tensor for the parameter in the equation
at::Tensor timing_net_wirelength_backward(
    at::Tensor grad_pos,
    at::Tensor pos,
    at::Tensor grad_intermediate, 
    at::Tensor flat_tnetpin,
    at::Tensor tnet_weights,
    at::Tensor inv_gamma)
{
    CHECK_FLAT(pos);
    CHECK_EVEN(pos);
    CHECK_CONTIGUOUS(pos);
    CHECK_FLAT(flat_tnetpin);
    CHECK_CONTIGUOUS(flat_tnetpin);
    CHECK_FLAT(tnet_weights);
    CHECK_CONTIGUOUS(tnet_weights);
    CHECK_FLAT(grad_intermediate);
    CHECK_EVEN(grad_intermediate);
    CHECK_CONTIGUOUS(grad_intermediate);

    at::Tensor grad_out = grad_intermediate.mul_(grad_pos);
    //int num_nets = netpin_start.numel() - 1;
    int num_pins = pos.numel() / 2;

    return grad_out;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &DREAMPLACE_NAMESPACE::timing_net_wirelength_forward, "TimingNetWirelength forward (CUDA)");
    m.def("backward", &DREAMPLACE_NAMESPACE::timing_net_wirelength_backward, "TimingNetWirelength backward (CUDA)");
    m.def("forward_fpga", &DREAMPLACE_NAMESPACE::timing_net_wirelength_forward_fpga, "TimingNetWirelength forward reuse net bbox(CUDA)");
}
