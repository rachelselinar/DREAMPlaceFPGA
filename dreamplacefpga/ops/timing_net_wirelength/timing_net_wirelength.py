##
# @file   timing_net_wirelength.py
# @author Zhili Xiong (DREAMPlaceFPGA)
# @date   Jun 2018
# @brief  Compute timing-net wirelength
#

import time
import torch
from torch import nn
from torch.autograd import Function
import logging

import dreamplacefpga.ops.timing_net_wirelength.timing_net_wirelength_cpp_merged as timing_net_wirelength_cpp_merged
import dreamplacefpga.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE": 
    import dreamplacefpga.ops.timing_net_wirelength.timing_net_wirelength_cuda_merged as timing_net_wirelength_cuda_merged
import pdb

logger = logging.getLogger(__name__)

class TimingNetWirelengthMergedFunction(Function):
    """
    @brief compute weighted average wirelength for timing nets.
    """
    @staticmethod
    def forward(ctx, pos, flat_tnetpin, tnet_weights, pin_mask, inv_gamma, net_bounding_box_min, net_bounding_box_max, num_threads, xl, yl, xh, yh, deterministic_flag):
        """
        @param pos pin location (x array, y array), not cell location
        @param flat_tnetpin flat tnet2pin map
        @param flat_pin2tnet flat pin2tnet map
        @param pin2tnet_start starting index in pin2tnet map for each pin, length of #pins+1
        @param tnet_weights weight of timing nets
        @param pin_mask whether compute gradient for a pin, 1 means to fill with zero, 0 means to compute
        @param inv_gamma 1/gamma, the larger, the closer to HPWL
        """
        tt = time.time()
        if pos.is_cuda:
            #pdb.set_trace()
            output = timing_net_wirelength_cuda_merged.forward(pos.view(pos.numel()), flat_tnetpin, tnet_weights, inv_gamma, xl, yl, xh, yh, deterministic_flag)
            #output = timing_net_wirelength_cuda_merged.forward_fpga(pos.view(pos.numel()), flat_tnetpin, pin2net_map, net_weights, net_mask, inv_gamma, net_bounding_box_min, net_bounding_box_max)
        else:
            output = timing_net_wirelength_cpp_merged.forward(pos.view(pos.numel()), flat_tnetpin, tnet_weights, inv_gamma, num_threads, xl, yl, xh, yh, deterministic_flag)
            ctx.num_threads = num_threads
        ctx.flat_tnetpin = flat_tnetpin
        ctx.tnet_weights = tnet_weights
        ctx.pin_mask = pin_mask
        ctx.inv_gamma = inv_gamma
        ctx.grad_intermediate = output[1]
        # print("maximum intermediate gradient: %g, minimum intermediate gradient: %g" % (ctx.grad_intermediate.max(), ctx.grad_intermediate.min()))
        ctx.pos = pos
        if pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("timing wirelength forward %.3f ms" % ((time.time()-tt)*1000))

        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        tt = time.time()
        if grad_pos.is_cuda:
            output = timing_net_wirelength_cuda_merged.backward(
                    grad_pos,
                    ctx.pos,
                    ctx.grad_intermediate,
                    ctx.flat_tnetpin,
                    ctx.tnet_weights,
                    ctx.inv_gamma
                    )
        else:
            output = timing_net_wirelength_cpp_merged.backward(
                    grad_pos,
                    ctx.pos,
                    ctx.grad_intermediate,
                    ctx.flat_tnetpin,
                    ctx.tnet_weights,
                    ctx.inv_gamma,
                    ctx.num_threads
                    )            
        output[:int(output.numel()//2)].masked_fill_(ctx.pin_mask, 0.0)
        output[int(output.numel()//2):].masked_fill_(ctx.pin_mask, 0.0)
        if grad_pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("wirelength backward %.3f ms" % ((time.time()-tt)*1000))
        return output, None, None, None, None, None, None, None, None, None, None, None, None

class TimingNetWirelength(nn.Module):
    """
    @brief Compute weighted average wirelength.
    CPU only supports net-by-net algorithm.
    GPU supports three algorithms: net-by-net, atomic, merged.
    Different parameters are required for different algorithms.
    """
    def __init__(self, flat_tnetpin=None, tnet_weights=None, pin_mask=None, gamma=None, net_bounding_box_min=None, net_bounding_box_max=None, num_threads=None, xl=None, yl=None, xh=None, yh=None, deterministic_flag=None, algorithm='merged'):
        """
        @brief initialization
        @param flat_tnetpin flat netpin map, length of #pins
        @param tnet_weights weight of timing nets
        @param pin_mask whether compute gradient for a pin, 1 means to fill with zero, 0 means to compute
        @param gamma the smaller, the closer to HPWL
        @param algorithm must be net-by-net | atomic | merged
        """
        super(TimingNetWirelength, self).__init__()
        assert tnet_weights is not None \
                and pin_mask is not None \
                and gamma is not None, "tnet_weights, pin_mask, gamma are requried parameters"
        if algorithm in ['merged']:
            assert flat_tnetpin is not None , "flat_tnetpin is requried parameters for algorithm %s" % (algorithm)
        else:
            assert False, "unsupported algorithm %s" % (algorithm)

        self.flat_tnetpin = flat_tnetpin
        self.netpin_values = None
        self.tnet_weights = tnet_weights
        self.pin_mask = pin_mask
        self.gamma = gamma
        self.net_bounding_box_min = net_bounding_box_min
        self.net_bounding_box_max = net_bounding_box_max
        self.algorithm = algorithm
        self.num_threads = num_threads
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.deterministic_flag = deterministic_flag

        # pdb.set_trace()
    def forward(self, pos):
        if self.algorithm == 'merged':
            return TimingNetWirelengthMergedFunction.apply(pos,
                    self.flat_tnetpin,
                    self.tnet_weights,
                    self.pin_mask,
                    1.0/self.gamma, # do not store inv_gamma as gamma is changing
                    self.net_bounding_box_min,
                    self.net_bounding_box_max,
                    self.num_threads,
                    self.xl,
                    self.yl,
                    self.xh,
                    self.yh,
                    self.deterministic_flag
                    )
        else:
            assert False, "unsupported algorithm for timing net wirelength %s" % (self.algorithm)

         

