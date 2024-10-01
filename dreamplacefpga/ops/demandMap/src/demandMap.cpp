/**
 * @file   demandMap.cpp
 * @author Rachel Selina Rajarathnam (DREAMPlaceFPGA) 
 * @date   Nov 2020
 * @brief  Compute binCapMap
 *
 * Modifications Copyright(C) 2024 Advanced Micro Devices, Inc. All rights reserved
 *
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"
// local dependency
#include "demandMap/src/demand_function.h"

DREAMPLACE_BEGIN_NAMESPACE

/// define compute_demand_function 
template <typename T>
DEFINE_COMPUTE_DEMAND_FUNCTION(T);

template <typename T, typename AtomicOp>
int computeDemandMapLauncher(
        const int *site_type_map, 
        const T *node_size_x, 
        const T *node_size_y, 
        const int num_bins_x, 
        const int num_bins_y, 
        const int width, 
        const int height, 
        const int num_threads,
        AtomicOp atomic_add_op,
        typename AtomicOp::type* buf_map0,
        typename AtomicOp::type* buf_map1,
        typename AtomicOp::type* buf_map4,
        typename AtomicOp::type* buf_map5
        );

#define CHECK_FLAT(x) AT_ASSERTM(!x.is_cuda() && x.ndimension() == 1, #x "must be a flat tensor on CPU")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")

#define CALL_FPGA_LAUNCHER(atomic_add_op, map_ptr0, map_ptr1, map_ptr4, map_ptr5) \
  computeDemandMapLauncher<scalar_t, decltype(atomic_add_op)>(           \
      DREAMPLACE_TENSOR_DATA_PTR(site_type_map, int),                    \
      DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),                 \
      DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),                 \
      num_bins_x, num_bins_y, width, height,                             \
      num_threads, atomic_add_op, map_ptr0, map_ptr1, map_ptr4, map_ptr5)

/// @brief Compute wirelength preconditioner
int forward(
        at::Tensor site_type_map,
        at::Tensor node_size_x,
        at::Tensor node_size_y,
        int num_bins_x,
        int num_bins_y,
        int width,
        int height,
        at::Tensor binCapMap0,
        at::Tensor binCapMap1,
        at::Tensor binCapMap4,
        at::Tensor binCapMap5,
        int num_threads,
        int deterministic_flag)
{
    CHECK_FLAT(site_type_map); 
    CHECK_CONTIGUOUS(site_type_map);
    CHECK_FLAT(node_size_x);
    CHECK_CONTIGUOUS(node_size_x);
    CHECK_FLAT(node_size_y);
    CHECK_CONTIGUOUS(node_size_y);

    DREAMPLACE_DISPATCH_FLOATING_TYPES(node_size_x, "computeDemandMapLauncher", [&] {
            if (deterministic_flag == 1)
            {
                double diearea = width * height;
                int integer_bits = DREAMPLACE_STD_NAMESPACE::max((int)ceil(log2(diearea)) + 1, 32);
                int fraction_bits = DREAMPLACE_STD_NAMESPACE::max(64 - integer_bits, 0);
                long scale_factor = (1L << fraction_bits);
                int num_bins = num_bins_x * num_bins_y;

                std::vector<long> buf0(num_bins, 0);
                std::vector<long> buf1(num_bins, 0);
                std::vector<long> buf4(num_bins, 0);
                std::vector<long> buf5(num_bins, 0);
                AtomicAdd<long> atomic_add_op(scale_factor);

                CALL_FPGA_LAUNCHER(atomic_add_op, buf0.data(), buf1.data(), buf4.data(), buf5.data());

                scaleAdd(DREAMPLACE_TENSOR_DATA_PTR(binCapMap0, scalar_t),
                        buf0.data(), 1.0 / scale_factor, num_bins, num_threads);
                scaleAdd(DREAMPLACE_TENSOR_DATA_PTR(binCapMap1, scalar_t),
                        buf1.data(), 1.0 / scale_factor, num_bins, num_threads);
                scaleAdd(DREAMPLACE_TENSOR_DATA_PTR(binCapMap4, scalar_t),
                        buf4.data(), 1.0 / scale_factor, num_bins, num_threads);
                scaleAdd(DREAMPLACE_TENSOR_DATA_PTR(binCapMap5, scalar_t),
                        buf5.data(), 1.0 / scale_factor, num_bins, num_threads);
            } else
            {
                auto buf0 = DREAMPLACE_TENSOR_DATA_PTR(binCapMap0, scalar_t);
                auto buf1 = DREAMPLACE_TENSOR_DATA_PTR(binCapMap1, scalar_t);
                auto buf4 = DREAMPLACE_TENSOR_DATA_PTR(binCapMap4, scalar_t);
                auto buf5 = DREAMPLACE_TENSOR_DATA_PTR(binCapMap5, scalar_t);
                AtomicAdd<scalar_t> atomic_add_op;
                CALL_FPGA_LAUNCHER(atomic_add_op, buf0, buf1, buf4, buf5);
            }

    });
    return 0; 
}

template <typename T, typename AtomicOp>
int computeDemandMapLauncher(
        const int *site_type_map, 
        const T *node_size_x, 
        const T *node_size_y, 
        const int num_bins_x, 
        const int num_bins_y, 
        const int width, 
        const int height, 
        const int num_threads,
        AtomicOp atomic_add_op,
        typename AtomicOp::type* buf_map0,
        typename AtomicOp::type* buf_map1,
        typename AtomicOp::type* buf_map4,
        typename AtomicOp::type* buf_map5
        )
{
        T binW = T(width)/T(num_bins_x);
        T binH = T(height)/T(num_bins_y);
#pragma omp parallel for num_threads(num_threads)
    for (int s = 0; s < width*height; ++s)
    {
        int rw = int(s/height);
        int cl = int(s%height);

        if (site_type_map[s] == 1 || site_type_map[s] == 2)
        {
            T nodeX = node_size_x[1];
            T nodeY = node_size_y[1];
            T col = DREAMPLACE_STD_NAMESPACE::round(cl/nodeY)*nodeY;
            int iLo = int(rw/binW);
            int jLo = int(col/binH);
            int iHi = DREAMPLACE_STD_NAMESPACE::min(int((rw + nodeX)/binW), num_bins_x-1);
            int jHi = DREAMPLACE_STD_NAMESPACE::min(int((col + nodeY)/binH), num_bins_y-1);
            for (int i = iLo; i <= iHi; ++i)
            {
                T w = compute_demand_function(i, binW, T(rw), nodeX);
                for (int j = jLo; j <= jHi; ++j)
                {
                    T h = compute_demand_function(j, binH, col, nodeY);
                    T area = w * h;
                    atomic_add_op(&buf_map0[i*num_bins_y + j], area);
                    if (site_type_map[s] == 2)
                    {
                        atomic_add_op(&buf_map1[i*num_bins_y + j], area);
                    }
                }
            }
        } else if (site_type_map[s] == 3)
        {
            T nodeX = node_size_x[4];
            T nodeY = node_size_y[4];
            T col = DREAMPLACE_STD_NAMESPACE::round(cl/nodeY)*nodeY;
            int iLo = int(rw/binW);
            int jLo = int(col/binH);
            int iHi = DREAMPLACE_STD_NAMESPACE::min(int((rw + nodeX)/binW), num_bins_x-1);
            int jHi = DREAMPLACE_STD_NAMESPACE::min(int((col + nodeY)/binH), num_bins_y-1);
            for (int i = iLo; i <= iHi; ++i)
            {
                T w = compute_demand_function(i, binW, T(rw), nodeX);
                for (int j = jLo; j <= jHi; ++j)
                {
                    T h = compute_demand_function(j, binH, col, nodeY);
                    T area = w * h;
                    atomic_add_op(&buf_map4[i*num_bins_y + j], area);
                }
            }
        } else if (site_type_map[s] == 4)
        {
            T nodeX = node_size_x[5];
            T nodeY = node_size_y[5];
            T col = DREAMPLACE_STD_NAMESPACE::round(cl/nodeY)*nodeY;
            int iLo = int(rw/binW);
            int jLo = int(col/binH);
            int iHi = DREAMPLACE_STD_NAMESPACE::min(int((rw + nodeX)/binW), num_bins_x-1);
            int jHi = DREAMPLACE_STD_NAMESPACE::min(int((col + nodeY)/binH), num_bins_y-1);
            for (int i = iLo; i <= iHi; ++i)
            {
                T w = compute_demand_function(i, binW, T(rw), nodeX);
                for (int j = jLo; j <= jHi; ++j)
                {
                    T h = compute_demand_function(j, binH, col, nodeY);
                    T area = w * h;
                    atomic_add_op(&buf_map5[i*num_bins_y + j], area);
                }
            }
        }
    }
    return 0; 
}

#undef CALL_FPGA_LAUNCHER

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::forward, "DemandMap forward");
}
