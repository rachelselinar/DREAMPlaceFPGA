##
# @file   demandMap.py
# @author Rachel Selina Rajarathnam (DREAMPlaceFPGA)
# @date   Nov 2020
#

import torch
from torch.autograd import Function
from torch import nn
import numpy as np 
import pdb 
import time

import dreamplacefpga.ops.demandMap.demandMap_cpp as demandMap_cpp
import dreamplacefpga.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE": 
    import dreamplacefpga.ops.demandMap.demandMap_cuda as demandMap_cuda

class DemandMap(nn.Module):
    """ 
    @brief Build binCapMap and fixedDemandMap
    """
    def __init__(self, site_type_map, num_bins_x, num_bins_y, width, height, node_size_x, node_size_y,
                 xh, xl, yh, yl, device, num_threads):
        """
        @brief initialization 
        @param site_type_map
        @param num_bins_x
        @param num_bins_y
        @param node_size_x
        @param node_size_y
        @param xh
        @param xl
        @param yh
        @param yl
        @param device 
        @param num_threads
        """
        super(DemandMap, self).__init__()
        self.site_type_map=site_type_map
        self.num_bins_x=num_bins_x
        self.num_bins_y=num_bins_y
        self.width=width
        self.height=height
        self.node_size_x=node_size_x
        self.node_size_y=node_size_y
        self.xh=xh
        self.xl=xl
        self.yh=yh
        self.yl=yl
        self.device=device
        self.num_threads = num_threads

    def forward(self): 
        binCapMap0 = torch.zeros((self.num_bins_x, self.num_bins_y), dtype=torch.float32, device=self.device)
        binCapMap1 = torch.zeros_like(binCapMap0)
        binCapMap2 = torch.zeros_like(binCapMap0)
        binCapMap3 = torch.zeros_like(binCapMap0)
        
        if binCapMap0.is_cuda:
            demandMap_cuda.forward(
                                   self.site_type_map.flatten(), 
                                   self.num_bins_x,
                                   self.num_bins_y,
                                   self.width, 
                                   self.height, 
                                   self.node_size_x, 
                                   self.node_size_y, 
                                   binCapMap0,
                                   binCapMap2,
                                   binCapMap3)
        else:
            demandMap_cpp.forward(
                                   self.site_type_map.flatten(), 
                                   self.num_bins_x,
                                   self.num_bins_y,
                                   self.width, 
                                   self.height, 
                                   self.node_size_x, 
                                   self.node_size_y, 
                                   binCapMap0,
                                   binCapMap2,
                                   binCapMap3,
                                   self.num_threads)

        binCapMap1 = binCapMap0
        # Generate fixed demand maps from the bin capacity maps
        fixedDemMap0 = torch.zeros_like(binCapMap0)
        fixedDemMap1 = torch.zeros_like(binCapMap0)
        fixedDemMap2 = torch.zeros_like(binCapMap0)
        fixedDemMap3 = torch.zeros_like(binCapMap0)

        binX = (self.xh - self.xl)/self.num_bins_x
        binY = (self.yh - self.yl)/self.num_bins_y
        binArea = binX * binY
        fixedDemMap0 = binArea - binCapMap0
        fixedDemMap1 = binArea - binCapMap1
        fixedDemMap2 = binArea - binCapMap2
        fixedDemMap3 = binArea - binCapMap3

        return [fixedDemMap0, fixedDemMap1, fixedDemMap2, fixedDemMap3]

