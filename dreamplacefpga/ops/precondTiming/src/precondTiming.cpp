/**
 * @file   precondTiming.cpp
 * @author Zhili Xiong(DREAMPlaceFPGA)
 * @date   Aug 2023
 * @brief  Compute precond Timing
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T, typename AtomicOp>
int computePrecondTimingLauncher(
        const T *tnet_weights, 
        const int *flat_tnet2pin,
        const int *pin2node_map, 
        int num_tnets, 
        int num_threads, 
        AtomicOp atomic_add_op,
        typename AtomicOp::type *buf_out
        );


#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

/// @brief Compute wirelength preconditioner
void forward(
        at::Tensor tnet_weights,
        at::Tensor flat_tnet2pin,
        at::Tensor pin2node_map,
        int num_tnets,
        int num_threads,
        double xl, double yl,
        double xh, double yh, 
        int deterministic_flag,
        at::Tensor out)
{
    CHECK_FLAT(tnet_weights); 
    CHECK_CONTIGUOUS(tnet_weights);

    CHECK_FLAT(flat_tnet2pin); 
    CHECK_CONTIGUOUS(flat_tnet2pin);
    CHECK_FLAT(pin2node_map);
    CHECK_CONTIGUOUS(pin2node_map);

    int num_nodes = out.numel();

    DREAMPLACE_DISPATCH_FLOATING_TYPES(out, "computePrecondTimingLauncher", [&] {
            if(deterministic_flag == 1){
                double diearea = (xh - xl) * (yh - yl);
                int integer_bits = DREAMPLACE_STD_NAMESPACE::max((int)ceil(log2(diearea)) + 1, 32);
                int fraction_bits = DREAMPLACE_STD_NAMESPACE::max(64 - integer_bits, 0);
                long scale_factor = (1L << fraction_bits);

                std::vector<long> buf_out(num_nodes, 0);
                AtomicAdd<long> atomic_add_op(scale_factor);

                computePrecondTimingLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(tnet_weights, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(flat_tnet2pin, int), 
                    DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int),
                    num_tnets, num_threads,
                    atomic_add_op, buf_out.data()
                    );
                
                scaleAdd(DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t), buf_out.data(), 1.0 / scale_factor, num_nodes, num_threads);
            }else {
                auto buf_out = DREAMPLACE_TENSOR_DATA_PTR(out, scalar_t);
                AtomicAdd<scalar_t> atomic_add_op;

                computePrecondTimingLauncher<scalar_t>(
                    DREAMPLACE_TENSOR_DATA_PTR(tnet_weights, scalar_t), 
                    DREAMPLACE_TENSOR_DATA_PTR(flat_tnet2pin, int), 
                    DREAMPLACE_TENSOR_DATA_PTR(pin2node_map, int),
                    num_tnets, num_threads,
                    atomic_add_op, buf_out
                    );
            }
            });
}

template <typename T, typename AtomicOp>
int computePrecondTimingLauncher(
        const T *tnet_weights, 
        const int *flat_tnet2pin,
        const int *pin2node_map, 
        int num_tnets, 
        int num_threads, 
        AtomicOp atomic_add_op,
        typename AtomicOp::type *buf_out
        )
{
    int chunk_size = DREAMPLACE_STD_NAMESPACE::max(int(num_tnets/ num_threads / 16), 1);
    //#pragma omp parallel for schedule(static)
    //#pragma omp parallel for num_threads(num_threads)
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)

    for (int i = 0; i < num_tnets; ++i)
    {
        int src = flat_tnet2pin[2*i];
        int dst = flat_tnet2pin[2*i+1];
        
        int src_node_id = pin2node_map[src];
        int dst_node_id = pin2node_map[dst];
 
        atomic_add_op(&buf_out[src_node_id], tnet_weights[i]); 
        atomic_add_op(&buf_out[dst_node_id], tnet_weights[i]); 
    }

    return 0; 
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::forward, "PrecondTiming forward");
}
