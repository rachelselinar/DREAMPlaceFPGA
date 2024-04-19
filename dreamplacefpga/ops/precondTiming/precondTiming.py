##
# @file   precondTiming.py
# @author Zhili Xiong(DREAMPlaceFPGA)
# @date   June 2023
# @brief  Preconditioner for timing term
##

import torch
from torch.autograd import Function
from torch import nn
import numpy as np 
import pdb 
import time

import dreamplacefpga.ops.precondTiming.precondTiming_cpp as precondTiming_cpp
import dreamplacefpga.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE": 
    import dreamplacefpga.ops.precondTiming.precondTiming_cuda as precondTiming_cuda

class PrecondTiming(nn.Module):
    """ 
    @brief Compute timing arc wirelength preconditioner. 
    """
    def __init__(self, flat_tnet2pin, pin2node_map, num_tnets, num_nodes, num_movable_nodes, device, num_threads, xl, yl, xh, yh, deterministic_flag):
        """
        @brief initialization 
        @param flat_tnet2pin_start flat tnet2pin map
        @param flat_tnet2pin flat tnet2pin map
        @param pin2node_map pin2node map
        @param tnet_weights weight of timing nets 
        @param tnet_criticality criticality of timing nets
        @param num_nodes 
        @param num_threads
        """
        super(PrecondTiming, self).__init__()
        self.flat_tnet2pin=flat_tnet2pin
        self.pin2node_map=pin2node_map
        self.num_tnets=num_tnets
        self.num_nodes=num_nodes
        self.num_movable_nodes=num_movable_nodes
        self.num_threads=num_threads
        self.device=device
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.deterministic_flag = deterministic_flag

    def forward(self, beta, tnet_weights): 
        """
        @brief Compute timing arc wirelength preconditioner.
        @param beta lagrangian multiplier
        @param tnet_weights weight of timing srcs
        """
        out = torch.zeros(self.num_nodes, dtype=torch.float32, device=self.device)

        if out.is_cuda:
            precondTiming_cuda.forward(
                                   self.flat_tnet2pin, 
                                   self.pin2node_map, 
                                   tnet_weights,
                                   self.num_tnets,
                                   self.xl,
                                   self.yl,
                                   self.xh,
                                   self.yh,
                                   self.deterministic_flag,
                                   out)       
        else:
            precondTiming_cpp.forward(
                                  tnet_weights,
                                  self.flat_tnet2pin, 
                                  self.pin2node_map, 
                                  self.num_tnets, 
                                  self.num_threads, 
                                  self.xl,
                                  self.yl,
                                  self.xh,
                                  self.yh,
                                  self.deterministic_flag,
                                  out)
        out = out * beta
        return out
