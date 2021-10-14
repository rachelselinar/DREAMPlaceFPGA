'''
@File: dsp_ram_legalization.py
@Author: Rachel Selina Rajarathnam (DREAMPlaceFPGA) 
@Date: Oct 2020
'''
import math
import torch
from torch import nn
from torch.autograd import Function
import matplotlib.pyplot as plt
import pdb 
import numpy as np

import dreamplacefpga.ops.dsp_ram_legalization.legalize_cpp as legalize_cpp
import dreamplacefpga.configure as configure

import logging
logger = logging.getLogger(__name__)

class LegalizeDSPRAMFunction(Function):
    @staticmethod
    def legalize(pos, placedb, region_id, model):
        """
        @brief legalize DSP/RAM at the end of Global Placement
        @param locX Instance locX ndarray
        @param locY Instance locY ndarray
        @param num_nodes Instance count
        @param num_sites Instance site count
        @param sites Instance site ndarray 
        @param precondWL Instance wirelength preconditioner ndarray 
        @param dInit lg_max_dist_init
        @param dIncr lg_max_dist_incr
        @param fScale lg_flow_cost_scale
        @param movVal Maximum & Average Instance movement (list)
        @param outLoc Legalized Instance locations list - {x0, x1, ... xn, y0, y1, ... yn} 
        """
        lg_max_dist_init=10.0
        lg_max_dist_incr=10.0
        lg_flow_cost_scale=100.0
        numNodes = int(pos.numel()/2)
        num_inst = int(placedb.num_movable_nodes_fence_region[region_id])
        outLoc = np.zeros(2*num_inst, dtype=np.float32).tolist()

        if region_id == 2:
            mask = model.data_collections.dsp_mask
            sites = placedb.dspSiteXYs
        elif region_id == 3:
            mask = model.data_collections.ram_mask
            sites = placedb.ramSiteXYs

        locX = pos[:placedb.num_physical_nodes][mask].cpu().detach().numpy()
        locY = pos[numNodes:numNodes+placedb.num_physical_nodes][mask].cpu().detach().numpy()

        num_sites = len(sites)
        precondWL = model.precondWL[:placedb.num_physical_nodes][mask].cpu().detach().numpy()
        movVal = np.zeros(2, dtype=np.float32).tolist()

        legalize_cpp.legalize(locX, locY, num_inst, num_sites, sites.flatten(), precondWL, lg_max_dist_init, lg_max_dist_incr, lg_flow_cost_scale, movVal, outLoc)

        outLoc = np.array(outLoc)
        updLoc = torch.from_numpy(outLoc).to(dtype=pos.dtype, device=pos.device)
        pos.data[:placedb.num_physical_nodes].masked_scatter_(mask, updLoc[:num_inst])
        pos.data[numNodes:numNodes+placedb.num_physical_nodes].masked_scatter_(mask, updLoc[num_inst:])

        return movVal 

