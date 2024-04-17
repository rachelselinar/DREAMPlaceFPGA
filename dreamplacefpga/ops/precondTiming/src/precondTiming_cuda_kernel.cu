#include "utility/src/utils.cuh"
#include "utility/src/limits.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T, typename AtomicOp>
__global__ void computePrecondTiming(
        const int *flat_tnet2pin,
        const int *pin2node_map, 
        const T *tnet_weights, 
        int num_tnets, 
        AtomicOp atomicAddOp,
        typename AtomicOp::type *out
        )
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num_tnets)
    {
        int src = flat_tnet2pin[2*i];
        int dst = flat_tnet2pin[2*i+1];

        int src_node_id = pin2node_map[src];
        int dst_node_id = pin2node_map[dst];

        atomicAddOp(&out[src_node_id], tnet_weights[i]);
        atomicAddOp(&out[dst_node_id], tnet_weights[i]);

    }
}

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
        )
{
    int thread_count = 512;
    int block_count = ceilDiv(num_tnets, thread_count);

    if (deterministic_flag)
    {
        // total die area
        double diearea = (xh - xl) * (yh - yl);
        int integer_bits = max((int)ceil(log2(diearea)) + 1, 32);
        int fraction_bits = max(64 - integer_bits, 0);
        unsigned long long int scale_factor = (1UL << fraction_bits);
        unsigned long long int *scaled_out = NULL;
        allocateCUDA(scaled_out, num_nodes, unsigned long long int);

        AtomicAddCUDA<unsigned long long int> atomicAddOp(scale_factor);
        int copy_thread = 512;

        copyScaleArray<<<num_nodes+copy_thread-1, copy_thread>>>(
                scaled_out, out, scale_factor, num_nodes);

        computePrecondTiming<<<block_count, thread_count>>>(
                flat_tnet2pin,
                pin2node_map,
                tnet_weights,
                num_tnets,
                atomicAddOp,
                scaled_out 
                );
        
        copyScaleArray<<<num_nodes+copy_thread-1, copy_thread>>>(out, scaled_out, T(1.0 / scale_factor), num_nodes);
        destroyCUDA(scaled_out);
    }
    else
    {   
        AtomicAddCUDA<T> atomicAddOp;
        computePrecondTiming<<<block_count, thread_count>>>(
                flat_tnet2pin,
                pin2node_map,
                tnet_weights,
                num_tnets,
                atomicAddOp,
                out 
                );
    }
    return 0;
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T) \
    template int computePrecondTimingCudaLauncher<T>( \
        const int *flat_tnet2pin, \
        const int *pin2node_map, \
        const T *tnet_weights, \
        int num_tnets, \
        int num_nodes, \
        const T xl, const T yl, \
        const T xh, const T yh, \
        bool deterministic_flag, \
        T *out \
        );

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
