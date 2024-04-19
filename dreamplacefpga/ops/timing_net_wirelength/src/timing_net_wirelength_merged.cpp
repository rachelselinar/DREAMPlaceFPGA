#include "utility/src/torch.h"
#include "utility/src/utils.h"

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
template <typename T, typename AtomicOp>
int computeTimingNetWirelengthMergedLauncher(
    const T *x, const T *y,
    const int *flat_tnetpin,
    const T *tnet_weights,
    int num_tnets,
    const T *inv_gamma,
    T *partial_wl,
    int num_threads, 
    AtomicOp atomic_add_op,
    typename AtomicOp::type* buf_map0,
    typename AtomicOp::type* buf_map1
    )
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_tnets / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int i = 0; i < num_tnets; ++i)
    {
        // int degree = netpin_start[ii+1]-netpin_start[ii];
        T x_max = -std::numeric_limits<T>::max();
        T x_min = std::numeric_limits<T>::max();
        T y_max = -std::numeric_limits<T>::max();
        T y_min = std::numeric_limits<T>::max();
        int tnetpin_start = i*2;
        for (int j = tnetpin_start; j < tnetpin_start + 2; ++j)
        {
            T xx = x[flat_tnetpin[j]];
            x_max = DREAMPLACE_STD_NAMESPACE::max(xx, x_max);
            x_min = DREAMPLACE_STD_NAMESPACE::min(xx, x_min);
            T yy = y[flat_tnetpin[j]];
            y_max = DREAMPLACE_STD_NAMESPACE::max(yy, y_max);
            y_min = DREAMPLACE_STD_NAMESPACE::min(yy, y_min);
        }

        T xexp_x_sum = 0;
        T xexp_nx_sum = 0;
        T exp_x_sum = 0;
        T exp_nx_sum = 0;

        T yexp_y_sum = 0;
        T yexp_ny_sum = 0;
        T exp_y_sum = 0;
        T exp_ny_sum = 0;

        for (int j = tnetpin_start; j < tnetpin_start + 2; ++j)
        {
            T xx = x[flat_tnetpin[j]];
            T exp_x = exp((xx - x_max) * (*inv_gamma));
            T exp_nx = exp((x_min - xx) * (*inv_gamma));

            xexp_x_sum += xx * exp_x;
            xexp_nx_sum += xx * exp_nx;
            exp_x_sum += exp_x;
            exp_nx_sum += exp_nx;

            T yy = y[flat_tnetpin[j]];
            T exp_y = exp((yy - y_max) * (*inv_gamma));
            T exp_ny = exp((y_min - yy) * (*inv_gamma));

            yexp_y_sum += yy * exp_y;
            yexp_ny_sum += yy * exp_ny;
            exp_y_sum += exp_y;
            exp_ny_sum += exp_ny;
        }

        partial_wl[i] = xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum +
                        yexp_y_sum / exp_y_sum - yexp_ny_sum / exp_ny_sum;

        T b_x = (*inv_gamma) / (exp_x_sum);
        T a_x = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
        T b_nx = -(*inv_gamma) / (exp_nx_sum);
        T a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;

        T b_y = (*inv_gamma) / (exp_y_sum);
        T a_y = (1.0 - b_y * yexp_y_sum) / exp_y_sum;
        T b_ny = -(*inv_gamma) / (exp_ny_sum);
        T a_ny = (1.0 - b_ny * yexp_ny_sum) / exp_ny_sum;

        T wt = tnet_weights[i];

        for (int j = tnetpin_start; j < tnetpin_start + 2; ++j)
        {
            T xx = x[flat_tnetpin[j]];
            T exp_x = exp((xx - x_max) * (*inv_gamma));
            T exp_nx = exp((x_min - xx) * (*inv_gamma));

            T grad_x_tmp = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;
            atomic_add_op(&buf_map0[flat_tnetpin[j]], wt * grad_x_tmp);  
                
            T yy = y[flat_tnetpin[j]];
            T exp_y = exp((yy - y_max) * (*inv_gamma));
            T exp_ny = exp((y_min - yy) * (*inv_gamma));

            T grad_y_tmp = (a_y + b_y * yy) * exp_y - (a_ny + b_ny * yy) * exp_ny;
            atomic_add_op(&buf_map1[flat_tnetpin[j]], wt * grad_y_tmp);   
        }
        
    }

    return 0; 
}

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
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
    int num_threads,
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
    at::Tensor partial_wl = at::zeros({num_tnets}, pos.options());
    // timed with grad_in yet
    at::Tensor grad_intermediate = at::zeros_like(pos);

    DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeTimingNetWirelengthMergedLauncher", [&] {
            if (deterministic_flag == 1){
            double diearea =  (xh - xl) * (yh - yl);
            int integer_bits = DREAMPLACE_STD_NAMESPACE::max((int)ceil(log2(diearea)) + 1, 32);
            int fraction_bits = DREAMPLACE_STD_NAMESPACE::max(64 - integer_bits, 0);
            long scale_factor = (1L << fraction_bits);

            std::vector<long> buf0(num_pins, 0);
            std::vector<long> buf1(num_pins, 0);
            AtomicAdd<long> atomic_add_op(scale_factor);
            computeTimingNetWirelengthMergedLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
                    DREAMPLACE_TENSOR_DATA_PTR(flat_tnetpin, int),
                    DREAMPLACE_TENSOR_DATA_PTR(tnet_weights, scalar_t),
                    num_tnets,
                    DREAMPLACE_TENSOR_DATA_PTR(inv_gamma, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(partial_wl, scalar_t),
                    num_threads, atomic_add_op,
                    buf0.data(), buf1.data());
            scaleAdd(DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t),
                    buf0.data(), 1.0 / scale_factor, num_pins,
                    num_threads);
            scaleAdd(DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t) + num_pins,
                    buf1.data(), 1.0 / scale_factor, num_pins,
                    num_threads);
            } else {
                auto buf0 = DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t); 
                auto buf1 = DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t) + num_pins;
                AtomicAdd<scalar_t> atomic_add_op;
                computeTimingNetWirelengthMergedLauncher<scalar_t>(
                        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
                        DREAMPLACE_TENSOR_DATA_PTR(flat_tnetpin, int),
                        DREAMPLACE_TENSOR_DATA_PTR(tnet_weights, scalar_t),
                        num_tnets,
                        DREAMPLACE_TENSOR_DATA_PTR(inv_gamma, scalar_t),
                        DREAMPLACE_TENSOR_DATA_PTR(partial_wl, scalar_t),
                        num_threads, atomic_add_op,
                        buf0, buf1);
            }
            if (tnet_weights.numel())
            {
                partial_wl.mul_(tnet_weights);
            }
    });

    auto wl = partial_wl.sum();
    //at::Tensor wl = at::zeros(1, pos.options());
    return {wl, grad_intermediate};
}

/// @brief Compute gradient
/// @param grad_pos input gradient from backward propagation
/// @param pos locations of pins
/// @param flat_netpin similar to the JA array in CSR format, which is flattened from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is the number of pins in each net, the length of IA is number of nets + 1
/// @param net_weights weight of nets
/// @param net_mask an array to record whether compute the where for a net or not
/// @param inv_gamma a scalar tensor for the parameter in the equation
at::Tensor timing_net_wirelength_backward(
    at::Tensor grad_pos,
    at::Tensor pos,
    at::Tensor grad_intermediate,
    at::Tensor flat_tnetpin,
    at::Tensor tnet_weights,
    at::Tensor inv_gamma,
    int num_threads)
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

    return grad_out;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &DREAMPLACE_NAMESPACE::timing_net_wirelength_forward, "TimingNetWirelength forward");
    m.def("backward", &DREAMPLACE_NAMESPACE::timing_net_wirelength_backward, "TimingNetWirelength backward");
}
