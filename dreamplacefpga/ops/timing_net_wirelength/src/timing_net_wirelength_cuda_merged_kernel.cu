/**
 * @file   timing_net_wirelength_cuda_merged_kernel.cu
 * @author Zhili Xiong (DREAMPlaceFPGA)
 * @date   Aug 2023
 */

#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T, typename AtomicOp>
__global__ void computeTimingNetWirelength(
        const T* x, const T* y, 
        const int* flat_tnetpin, 
        const T* tnet_weights,
        int num_tnets,
        const T* inv_gamma, 
        T* partial_wl,
        AtomicOp atomicAddOp,
        typename AtomicOp::type *grad_intermediate_x, 
        typename AtomicOp::type *grad_intermediate_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ii = i >> 1;
    if (ii < num_tnets)
    {
        const T *values;
        // typename AtomicOp::type *grads;
        if (i & 1)
        {
            values = y;
            // grads = grad_intermediate_y;
        }
        else
        {
            values = x;
            // grads = grad_intermediate_x;
        }

        // int degree = netpin_start[ii+1]-netpin_start[ii];
        T x_max = -FLT_MAX;
        T x_min = FLT_MAX;

        int tnetpin_start = ii*2;

        for (int j = tnetpin_start; j < tnetpin_start + 2; ++j)
        {
            T xx = values[flat_tnetpin[j]];
            x_max = max(xx, x_max);
            x_min = min(xx, x_min);
        }

        T xexp_x_sum = 0;
        T xexp_nx_sum = 0;
        T exp_x_sum = 0;
        T exp_nx_sum = 0;
        for (int j = tnetpin_start; j < tnetpin_start + 2; ++j)
        {
            T xx = values[flat_tnetpin[j]];
            T exp_x = exp((xx - x_max) * (*inv_gamma));
            T exp_nx = exp((x_min - xx) * (*inv_gamma));

            xexp_x_sum += xx * exp_x;
            xexp_nx_sum += xx * exp_nx;
            exp_x_sum += exp_x;
            exp_nx_sum += exp_nx;
        }

        partial_wl[i] = xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum;

        T b_x = (*inv_gamma) / (exp_x_sum);
        T a_x = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
        T b_nx = -(*inv_gamma) / (exp_nx_sum);
        T a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;

        T wt = tnet_weights[ii];

        for (int j = tnetpin_start; j < tnetpin_start + 2; ++j)
        {
            T xx = values[flat_tnetpin[j]];
            T exp_x = exp((xx - x_max) * (*inv_gamma));
            T exp_nx = exp((x_min - xx) * (*inv_gamma)); 

            T grad_tmp = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;

            if (i & 1){
                atomicAddOp(&grad_intermediate_y[flat_tnetpin[j]], wt * grad_tmp);
            } else {
                atomicAddOp(&grad_intermediate_x[flat_tnetpin[j]], wt * grad_tmp);
            }

        }
    }
}

template <typename T,  typename AtomicOp>
__global__ void computeTimingNetWirelengthFPGA(
        const T* x, const T* y, 
        const int* flat_tnetpin,
        const T* tnet_weights,
        int num_tnets,
        const T* inv_gamma, 
        const T* bbox_min_x, const T* bbox_min_y,
        const T* bbox_max_x, const T* bbox_max_y,
        T* partial_wl,
        AtomicOp atomicAddOp,
        typename AtomicOp::type *grad_intermediate_x, 
        typename AtomicOp::type *grad_intermediate_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ii = i >> 1;
    if (ii < num_tnets)
    {
        const T *values;
        const T *bbox_min;
        const T *bbox_max;
        // typename AtomicOp::type *grads;
        if (i & 1)
        {
            values = y;
            // grads = grad_intermediate_y;
            bbox_min = bbox_min_y;
            bbox_max = bbox_max_y;
        }
        else
        {
            values = x;
            // grads = grad_intermediate_x;
            bbox_min = bbox_min_x;
            bbox_max = bbox_max_x;
        }

        T xexp_x_sum = 0;
        T xexp_nx_sum = 0;
        T exp_x_sum = 0;
        T exp_nx_sum = 0;
        T wt = tnet_weights[ii];

        int tnetpin_start = ii*2;
        
        for (int j = tnetpin_start; j < tnetpin_start + 2; ++j)
        {
            T xx = values[flat_tnetpin[j]];
            //T exp_x = exp((xx - x_max) * (*inv_gamma));
            //T exp_nx = exp((x_min - xx) * (*inv_gamma));
            T exp_x = exp((xx - bbox_max[ii]) * (*inv_gamma));
            T exp_nx = exp((bbox_min[ii] - xx) * (*inv_gamma));

            xexp_x_sum += xx * exp_x;
            xexp_nx_sum += xx * exp_nx;
            exp_x_sum += exp_x;
            exp_nx_sum += exp_nx;
        }

        // partial_wl[i] = xexp_x_sum / exp_x_sum - xexp_nx_sum / exp_nx_sum;

        T b_x = (*inv_gamma) / (exp_x_sum);
        T a_x = (1.0 - b_x * xexp_x_sum) / exp_x_sum;
        T b_nx = -(*inv_gamma) / (exp_nx_sum);
        T a_nx = (1.0 - b_nx * xexp_nx_sum) / exp_nx_sum;

        for (int j = tnetpin_start; j < tnetpin_start + 2; ++j)
        {   
            T xx = values[flat_tnetpin[j]];
            //T exp_x = exp((xx - x_max) * (*inv_gamma));
            //T exp_nx = exp((x_min - xx) * (*inv_gamma));
            T exp_x = exp((xx - bbox_max[ii]) * (*inv_gamma));
            T exp_nx = exp((bbox_min[ii] - xx) * (*inv_gamma));

            T grad_tmp = (a_x + b_x * xx) * exp_x - (a_nx + b_nx * xx) * exp_nx;
            
            if (i & 1){
                atomicAddOp(&grad_intermediate_y[flat_tnetpin[j]], wt * grad_tmp);
            } else {
                atomicAddOp(&grad_intermediate_x[flat_tnetpin[j]], wt * grad_tmp);
            }
        }
    }
}

template <typename T>
int computeTimingNetWirelengthCudaMergedLauncher(
    const T *x, const T *y,
    const int *flat_tnetpin,
    const T *tnet_weights,
    int num_tnets,
    int num_pins,
    const T *inv_gamma,
    T *partial_wl,
    const T xl, const T yl, 
    const T xh, const T yh,
    bool deterministic_flag,
    T *grad_intermediate_x, T *grad_intermediate_y)
{
    int thread_count = 64;
    int block_count = (num_tnets * 2 + thread_count - 1) / thread_count; // separate x and y

    if (deterministic_flag){
    //     // total die area
    //     double diearea = (xh - xl) * (yh - yl);
    //     int integer_bits = max((int)ceil(log2(diearea)) + 1, 8);
    //     int fraction_bits = max(16 - integer_bits, 0);
        int scale_factor = (1L << 8);
        
        int *scaled_grad_intermediate_x = NULL;
        allocateCUDA(scaled_grad_intermediate_x, num_pins, int);
        int *scaled_grad_intermediate_y = NULL;
        allocateCUDA(scaled_grad_intermediate_y, num_pins, int);
        
        AtomicAddCUDA<int> atomicAddOp(scale_factor);
        int copy_thread = 512;

        copyScaleArray<<<(num_pins + copy_thread - 1) / copy_thread,
                        copy_thread>>>(scaled_grad_intermediate_x, grad_intermediate_x, scale_factor, num_pins);
        copyScaleArray<<<(num_pins + copy_thread - 1) / copy_thread,
                        copy_thread>>>(scaled_grad_intermediate_y, grad_intermediate_y, scale_factor, num_pins);

        computeTimingNetWirelength<<<block_count, thread_count>>>(
            x, y,
            flat_tnetpin,
            tnet_weights,
            num_tnets,
            inv_gamma,
            partial_wl,
            atomicAddOp,
            scaled_grad_intermediate_x, scaled_grad_intermediate_y);

        copyScaleArray<<<(num_pins + copy_thread - 1) / copy_thread,
                        copy_thread>>>(grad_intermediate_x, scaled_grad_intermediate_x, T(1.0 / scale_factor), num_pins);
        copyScaleArray<<<(num_pins + copy_thread - 1) / copy_thread,
                        copy_thread>>>(grad_intermediate_y, scaled_grad_intermediate_y, T(1.0 / scale_factor), num_pins);
        destroyCUDA(scaled_grad_intermediate_x);
        destroyCUDA(scaled_grad_intermediate_y);

    } else {
        AtomicAddCUDA<T> atomicAddOp;
        computeTimingNetWirelength<<<block_count, thread_count>>>(
            x, y,
            flat_tnetpin,
            tnet_weights,
            num_tnets,
            inv_gamma,
            partial_wl,
            atomicAddOp,
            grad_intermediate_x, grad_intermediate_y);
    }

    return 0;
}

template <typename T>
int computeTimingNetWirelengthCudaMergedLauncherFPGA(
    const T *x, const T *y,
    const int *flat_tnetpin,
    const T *tnet_weights,
    int num_tnets,
    int num_pins,
    const T *inv_gamma,
    const T* bbox_min_x, const T* bbox_min_y,
    const T* bbox_max_x, const T* bbox_max_y,
    T *partial_wl,
    const T xl, const T yl,
    const T xh, const T yh,
    bool deterministic_flag,
    T *grad_intermediate_x, T *grad_intermediate_y)
{
    int thread_count = 64;
    int block_count = (num_tnets * 2 + thread_count - 1) / thread_count; // separate x and y

    if (deterministic_flag){
        // total die area
        // double diearea = (xh - xl) * (yh - yl);
        // int integer_bits = max((int)ceil(log2(diearea)) + 1, 32);
        // int fraction_bits = max(64 - integer_bits, 0);
        int scale_factor = (1L << 8);

        int *scaled_grad_intermediate_x = NULL;
        allocateCUDA(scaled_grad_intermediate_x, num_pins, int);
        int *scaled_grad_intermediate_y = NULL;
        allocateCUDA(scaled_grad_intermediate_y, num_pins, int);

        AtomicAddCUDA<int> atomicAddOp(scale_factor);
        int copy_thread = 512;

        copyScaleArray<<<(num_pins + copy_thread - 1) / copy_thread,
                        copy_thread>>>(scaled_grad_intermediate_x, grad_intermediate_x, scale_factor, num_pins);
        copyScaleArray<<<(num_pins + copy_thread - 1) / copy_thread,
                        copy_thread>>>(scaled_grad_intermediate_y, grad_intermediate_y, scale_factor, num_pins);

        computeTimingNetWirelengthFPGA<<<block_count, thread_count>>>(
              x, y,
            flat_tnetpin,
            tnet_weights,
            num_tnets,
            inv_gamma,
            bbox_min_x, bbox_min_y,
            bbox_max_x, bbox_max_y,
            partial_wl,
            atomicAddOp,
            scaled_grad_intermediate_x, scaled_grad_intermediate_y);

        copyScaleArray<<<(num_pins + copy_thread - 1) / copy_thread,
                        copy_thread>>>(grad_intermediate_x, scaled_grad_intermediate_x, T(1.0 / scale_factor), num_pins);
        copyScaleArray<<<(num_pins + copy_thread - 1) / copy_thread,
                        copy_thread>>>(grad_intermediate_y, scaled_grad_intermediate_y, T(1.0 / scale_factor), num_pins);
        destroyCUDA(scaled_grad_intermediate_x);
        destroyCUDA(scaled_grad_intermediate_y);
    } else {
        AtomicAddCUDA<T> atomicAddOp;
        computeTimingNetWirelengthFPGA<<<block_count, thread_count>>>(
            x, y,
            flat_tnetpin,
            tnet_weights,
            num_tnets,
            inv_gamma,
            bbox_min_x, bbox_min_y,
            bbox_max_x, bbox_max_y,
            partial_wl,
            atomicAddOp,
            grad_intermediate_x, grad_intermediate_y);
    }

    return 0;
    
}

#define REGISTER_KERNEL_LAUNCHER(T)                                         \
    template int computeTimingNetWirelengthCudaMergedLauncher<T>(     \
        const T *x, const T *y,                                             \
        const int *flat_tnetpin,                                             \
        const T *tnet_weights,                                               \
        int num_tnets,                                                       \
        int num_pins,                                                       \
        const T *inv_gamma,                                                 \
        T *partial_wl,                                                      \
        const T xl, const T yl,                                             \
        const T xh, const T yh,                                             \
        bool deterministic_flag,                                            \
        T *grad_intermediate_x, T *grad_intermediate_y);                    \
                                                                            \
    template int computeTimingNetWirelengthCudaMergedLauncherFPGA<T>( \
        const T *x, const T *y,                                             \
        const int *flat_tnetpin,                                             \
        const T *tnet_weights,                                               \
        int num_tnets,                                                       \
        int num_pins,                                                       \
        const T *inv_gamma,                                                 \
        const T *bbox_min_x, const T *bbox_min_y,                           \
        const T *bbox_max_x, const T *bbox_max_y,                           \
        T *partial_wl,                                                      \
        const T xl, const T yl,                                             \
        const T xh, const T yh,                                             \
        bool deterministic_flag,                                            \
        T *grad_intermediate_x, T *grad_intermediate_y);

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
