'''
@File: clustering_compatibility.py
@Author: Rachel Selina Rajarathnam (DREAMPlaceFPGA) 
@Date: Oct 2020
'''
import math
import torch
from torch import nn
from torch.autograd import Function
import matplotlib.pyplot as plt
import pdb 

import dreamplacefpga.ops.clustering_compatibility.clustering_compatibility_cpp as clustering_compatibility_cpp
try:
    import dreamplacefpga.ops.clustering_compatibility.clustering_compatibility_cuda as clustering_compatibility_cuda
except:
    pass

class LUTCompatibility(nn.Module):
    def __init__(self,
                 lut_indices, lut_type, node_size_x, node_size_y,
                 num_bins_x, num_bins_y, num_bins_l,
                 xl, yl, xh, yh, inst_stddev_x, inst_stddev_y,
                 inst_stddev_trunc, deterministic_flag,
                 num_threads
                 ):
        super(LUTCompatibility, self).__init__()
        self.lut_indices = lut_indices
        self.lut_type = lut_type
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.num_threads = num_threads
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.num_bins_l = num_bins_l
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.inst_stddev_x = inst_stddev_x
        self.inst_stddev_y = inst_stddev_y
        self.inst_stddev_trunc = inst_stddev_trunc
        self.deterministic_flag = deterministic_flag

    def forward(self, pos):
        lutType_DemMap = torch.zeros((self.num_bins_x, self.num_bins_y, self.num_bins_l), dtype=pos.dtype, device=pos.device)
        resource_areas = torch.zeros(len(self.node_size_x), dtype=pos.dtype, device=pos.device)

        ext_bin = max(round(self.inst_stddev_trunc - 0.5), 0);
        demandX = torch.zeros((2 * ext_bin + 1), dtype=pos.dtype, device=pos.device)
        demandY = torch.zeros_like(demandX)

        if pos.is_cuda:
            areaMap = clustering_compatibility_cuda.lut_compatibility(
                    pos,
                    self.lut_indices,
                    self.lut_type,
                    self.node_size_x,
                    self.node_size_y,
                    self.num_bins_x,
                    self.num_bins_y, 
                    self.num_bins_l, 
                    self.xl,
                    self.yl,
                    self.xh,
                    self.yh,
                    self.inst_stddev_x, 
                    self.inst_stddev_y, 
                    1.0/self.inst_stddev_x,
                    1.0/self.inst_stddev_y,
                    ext_bin,
                    self.inst_stddev_x * self.inst_stddev_y, 
                    1/math.sqrt(2.0),
                    self.deterministic_flag,
                    lutType_DemMap, 
                    resource_areas
                    )
        else:
            areaMap = clustering_compatibility_cpp.lut_compatibility(
                    pos,
                    self.lut_indices,
                    self.lut_type,
                    self.node_size_x,
                    self.node_size_y,
                    self.num_bins_x,
                    self.num_bins_y, 
                    self.num_bins_l, 
                    self.num_threads, 
                    self.inst_stddev_x, 
                    self.inst_stddev_y, 
                    1.0/self.inst_stddev_x,
                    1.0/self.inst_stddev_y,
                    ext_bin,
                    self.inst_stddev_x * self.inst_stddev_y, 
                    1/math.sqrt(2.0),
                    demandX,
                    demandY,
                    lutType_DemMap, 
                    resource_areas
                    )

        # Include post-processing if any here
        resource_areas /= 16.0

        return resource_areas


class FFCompatibility(nn.Module):
    def __init__(self,
                 flop_indices, flop_ctrlSets, node_size_x, node_size_y,
                 num_bins_x, num_bins_y, num_bins_ck, num_bins_ce,
                 xl, yl, xh, yh,
                 inst_stddev_x, inst_stddev_y,
                 inst_stddev_trunc,
                 deterministic_flag,
                 num_threads
                 ):
        super(FFCompatibility, self).__init__()
        self.flop_indices = flop_indices
        self.flop_ctrlSets = flop_ctrlSets
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.num_threads = num_threads
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.num_bins_ck = num_bins_ck
        self.num_bins_ce = num_bins_ce
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.inst_stddev_x = inst_stddev_x
        self.inst_stddev_y = inst_stddev_y
        self.inst_stddev_trunc = inst_stddev_trunc
        self.deterministic_flag = deterministic_flag

    def forward(self, pos):
        flopType_DemMap = torch.zeros((self.num_bins_x, self.num_bins_y, self.num_bins_ck, self.num_bins_ce), dtype=pos.dtype, device=pos.device)
        resource_areas = torch.zeros(len(self.node_size_x), dtype=pos.dtype, device=pos.device)

        ext_bin = max(round(self.inst_stddev_trunc - 0.5), 0);
        demandX = torch.zeros((2 * ext_bin + 1), dtype=pos.dtype, device=pos.device)
        demandY = torch.zeros_like(demandX)

        if pos.is_cuda:
            areaMap = clustering_compatibility_cuda.flop_compatibility(
                    pos,
                    self.flop_indices,
                    self.flop_ctrlSets,
                    self.node_size_x,
                    self.node_size_y,
                    self.num_bins_x,
                    self.num_bins_y, 
                    self.num_bins_ck, 
                    self.num_bins_ce, 
                    self.xl,
                    self.yl,
                    self.xh,
                    self.yh,
                    self.inst_stddev_x, 
                    self.inst_stddev_y, 
                    1.0/self.inst_stddev_x,
                    1.0/self.inst_stddev_y,
                    ext_bin,
                    self.inst_stddev_x * self.inst_stddev_y, 
                    1/math.sqrt(2.0),
                    16, #SLICE_CAPACITY
                    self.deterministic_flag,
                    flopType_DemMap, 
                    resource_areas
                    )
        else:
            areaMap = clustering_compatibility_cpp.flop_compatibility(
                    pos,
                    self.flop_indices,
                    self.flop_ctrlSets,
                    self.node_size_x,
                    self.node_size_y,
                    self.num_bins_x,
                    self.num_bins_y, 
                    self.num_bins_ck, 
                    self.num_bins_ce, 
                    self.num_threads, 
                    self.inst_stddev_x, 
                    self.inst_stddev_y, 
                    1.0/self.inst_stddev_x,
                    1.0/self.inst_stddev_y,
                    ext_bin,
                    self.inst_stddev_x * self.inst_stddev_y, 
                    1/math.sqrt(2.0),
                    16, #SLICE_CAPACITY
                    demandX,
                    demandY,
                    flopType_DemMap, 
                    resource_areas
                    )

        # Include post-processing if any here
        resource_areas /= 16.0

        return resource_areas
 
