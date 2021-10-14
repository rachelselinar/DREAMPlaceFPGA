#include "utility/src/utils.cuh"
#include "utility/src/limits.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void computePrecondWL(
        const int *flat_node2pin_start_map,
        const int *flat_node2pin_map, 
        const int *pin2net_map, 
        const int *flat_net2pin, 
        int num_nodes, 
        T *out 
        )
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < num_nodes)
    {
        int beg = flat_node2pin_start_map[i];
        int end = flat_node2pin_start_map[i+1];

        for (int p = beg; p < end; ++p)
        {
            int netId = pin2net_map[flat_node2pin_map[p]];
            int numPins = flat_net2pin[netId+1] - flat_net2pin[netId];
            out[i] += T(1.0/(numPins-1));
        }
    }
}

template <typename T>
int computePrecondWLCudaLauncher(
        const int *flat_node2pin_start_map,
        const int *flat_node2pin_map, 
        const int *pin2net_map, 
        const int *flat_net2pin, 
        int num_nodes, 
        T *out 
        )
{
    int thread_count = 512;
    int block_count = CPUCeilDiv(num_nodes, thread_count);

    computePrecondWL<<<block_count, thread_count>>>(
            flat_node2pin_start_map,
            flat_node2pin_map,
            pin2net_map,
            flat_net2pin,
            num_nodes,
            out 
            );

    return 0;
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(T) \
    int instantiatecomputePrecondWLCudaLauncher( \
        const int *flat_node2pin_start_map, \
        const int *flat_node2pin_map, \
        const int *pin2net_map, \
        const int *flat_net2pin, \
        int num_nodes, \
        T *out \
        ) \
    { \
        return computePrecondWLCudaLauncher( \
                flat_node2pin_start_map, \
                flat_node2pin_map, \
                pin2net_map, \
                flat_net2pin, \
                num_nodes, \
                out \
                ); \
    } \

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
