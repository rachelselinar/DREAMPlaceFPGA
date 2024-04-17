##
# @file   timing.py 
# @author Zhili Xiong
# @date   Mar 2023
# @brief  Main file implementing the net-weighting for timing-driven placement.
#

import os
import math
import sys
import torch
from torch.autograd import Function
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import logging
import time
import pdb

class TimingFeedback(nn.Module):
    def __init__(self, timer, tnet2net, tnet_criticality, tnet_weights, criticality_exp, device):
        """
        @brief Initialize the feedback module that inherits from the
         base neural network module in torch framework.
        @param timer the Timer python object, incuding the timing model and lookups
        @param tnet2net the mapping from timing net to net
        @param tnet_criticality the criticality of each timing net
        @param tnet_weights the weights of each timing net
        @param criticality_exp the criticality exponent
        @param device the device gpu or cpu
        """
        super(TimingFeedback, self).__init__()
        self.timer = timer
        self.tnet2net = tnet2net
        self.tnet_criticality = tnet_criticality
        self.tnet_weights = tnet_weights
        self.criticality_exp = criticality_exp
        self.num_tnets=len(self.tnet_weights)
        self.device=device

    def update_timing(self, pos, route_utilization_map, pin_utilization_map, route_utilization_thresh_5, pin_utilization_thresh_5):
        self.timer.tgraph.pin_pos = pos.data.clone().cpu().numpy()
        self.timer.tgraph.route_utilization_map = route_utilization_map
        self.timer.tgraph.pin_utilization_map = pin_utilization_map
        self.timer.tgraph.route_utilization_thresh_5 = route_utilization_thresh_5
        self.timer.tgraph.pin_utilization_thresh_5 = pin_utilization_thresh_5

        self.timer.tgraph.reset()

        tt=time.time()
        self.timer.tgraph.compute_arrival_time()
        logging.info("compute arrival time takes %.6f (s)" % (time.time()-tt))

        tt=time.time()
        self.timer.tgraph.compute_required_time()
        logging.info("compute required time takes %.6f (s)" % (time.time()-tt))

        tt=time.time()
        self.timer.tgraph.compute_slack()
        logging.info("compute slack takes %.6f (s)" % (time.time()-tt))
        
        upd_tnet_wts_criticality = apply_net_weighting(
            timer=self.timer,
            num_tnets=self.num_tnets,
            tnet2net=self.tnet2net,
            tnet_weights=self.tnet_weights, 
            tnet_criticality=self.tnet_criticality, 
            criticality_exp=self.criticality_exp,
            device=self.device)

        tt=time.time()
        wns, tns = self.timer.tgraph.report_wns_tns()
        logging.info("timing wns %.6f (ps)" % (wns))
        logging.info("timing tns %.6f (ps)" % (tns))

        # report critical path
        tt=time.time()
        self.timer.tgraph.report_critical_path(upd_tnet_wts_criticality[:self.num_tnets], self.device)
        logging.info("report critical path takes %.6f (s)" % (time.time()-tt))

        return upd_tnet_wts_criticality

    def report_timing_paths(self, path_file, out_file):
        """
        @brief report timing paths
        """
        test_path_delays = []
        with open(path_file, "r") as f:
            paths = f.readlines()
            for path in paths:
                start, end, _ = path.split()
                path_delay = self.timer.tgraph.report_path(start, end)
                test_path_delays.append(path_delay)

        with open(out_file, "w") as f:
            for path_delay in test_path_delays:
                f.write("%s\n" % (path_delay))


def apply_net_weighting(timer, num_tnets, tnet2net, tnet_weights, tnet_criticality, criticality_exp, device):
    """
    @brief apply different net_weighting scheme
    @param timer the Timer python object, incuding the timing model and lookups
    @param num_tnets the number of timing nets
    @param tnet2net the mapping from timing net to net
    @param tnet_weights the net weights in placedb
    @prams tnet_criticality the cricality of a net based on slack
    @param criticality_exp the criticality exponent
    @param device the device cpu or gpu
    """

    upd_tnet_wts_criticality = torch.zeros(2*num_tnets, dtype=tnet_weights.dtype, device=device)
    upd_tnet_wts_criticality = vpr_net_weighting(timer=timer, num_tnets=num_tnets, tnet2net=tnet2net, tnet_weights=tnet_weights, tnet_criticality=tnet_criticality, criticality_exp=criticality_exp, device=device)
    
    return upd_tnet_wts_criticality

def vpr_net_weighting(timer, num_tnets, tnet2net, tnet_weights, tnet_criticality, criticality_exp, device):
    """
    @brief implement the vpr net-weighting, and explore the best timing-driven interval and criticality exponent
    """
    
    Dmax = timer.tgraph.timing_constraint - timer.tgraph.report_wns_tns()[0]
    upd_tnet_wts_criticality = torch.zeros(2*num_tnets, dtype=tnet_weights.dtype, device=device)

    for tnet_id in range(num_tnets):
        if tnet_id in timer.tgraph.tnet2edge:
            slack = timer.tgraph.tnet2edge[tnet_id].slack
        else:
            continue

        # update net_criticality
        upd_tnet_wts_criticality[num_tnets + tnet_id] = 1 - slack/Dmax

        if slack < 0:
            # update net_weights, set the upper bound of wts to 55
            # upd_tnet_wts_criticality[tnet_id] = min(55, pow(upd_tnet_wts_criticality[num_tnets + tnet_id], criticality_exp))
            upd_tnet_wts_criticality[tnet_id] = pow(upd_tnet_wts_criticality[num_tnets + tnet_id], criticality_exp)

    return upd_tnet_wts_criticality



