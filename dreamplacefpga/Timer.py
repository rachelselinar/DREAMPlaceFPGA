##
# @file   Timer.py
# @author Zhili Xiong
# @date   JAN 2023
# @brief  Main file implementing the timing model in python.
#

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import os
import sys 
import time 
import numpy as np 
import scipy.io
import random
import logging
# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
	sys.path.append(root_dir)

import dreamplacefpga.configure as configure 
import dreamplacefpga.ops.timing.timing_graph as timing_graph
from Params import *
from PlaceDB import *
import pdb 
import scipy.optimize

class Timer:
    """
    @brief the Timer class for static timing analysis
    """

    def __init__(self, params, placedb):
        # get device information through RapidWright API
        self.part = params.part_name
        self.timing_file_dir = params.timing_file_dir
        self.loc2site_map = placedb.loc2site_map

        self.num_nodes = placedb.num_nodes
        self.num_sites_x = placedb.num_sites_x
        self.num_sites_y = placedb.num_sites_y
        self.site_type_map = placedb.site_type_map

        # build timing model and timing graph
        self.tmodel = TimingModel(self.part, self.timing_file_dir, self.loc2site_map, self.num_sites_x, self.num_sites_y)
        tt = time.time()
        self.tgraph = timing_graph.TimingGraph(self.tmodel, placedb, params.timing_constraint)   
        logging.info("Build timing graph and logic delays takes %.2f seconds" % (time.time() - tt))


class TimingModel():
    """ 
    @brief build timing model and lookups
    """
    def __init__(self, part, timing_file_dir, loc2site_map, num_sites_x, num_sites_y):
        """
        @brief initialization
        @param device device extracted by Rapidwright
        @param loc2site_map map x,y,z to site_name
        """

        self.part = part
        self.timing_file_dir = timing_file_dir
        self.loc2site_map = loc2site_map
        self.num_sites_x = num_sites_x
        self.num_sites_y = num_sites_y
        self.delay_logic = {}
        self.build_logic_delay_lookup()
        self.a0, self.a1, self.bias = self.curve_fit_linear() # curve fit parameters
        self.d_p = 1000 # delay penalty per unit route utilization
        self.d_r = 50 # delay penalty per unit pin utilization

    def get_net_delay(self, source_x, source_y, sink_x, sink_y):
        """
        @brief get net delay between two cells
        @param source_x source cell x coordinate
        @param sink_x sink cell x coordinate
        @param source_y source cell y coordinate
        @param sink_y sink cell y coordinate
        """

        dist_x = sink_x - source_x
        dist_y = sink_y - source_y

        net_delay = round(self.a0 * abs(dist_x) + self.a1 * abs(dist_y)) + self.bias

        return net_delay


    def get_congestion_delay(self, source_x, source_y, sink_x, sink_y, route_utilization_map, pin_utilization_map, route_utilization_thresh_5, pin_utilization_thresh_5):
        """
        @brief get congestion delay between two cells
        @param source_cell source cell object
        @param sink_cell sink cell object
        """
        xl = int(min(source_x, sink_x))
        xh = int(max(source_x, sink_x))
        yl = int(min(source_y, sink_y))
        yh = int(max(source_y, sink_y))

        grid_area = (xh-xl+1) * (yh-yl+1)

        # compute the average route utilization in the region
        route_utilization_avg = route_utilization_map[xl: xh+1, yl: yh+1].sum()/grid_area
        # compute the average pin utilization in the region
        pin_utilization_avg = pin_utilization_map[xl: xh+1, yl: yh+1].sum()/grid_area

        congestion_delay = 0
        if route_utilization_avg > route_utilization_thresh_5 and pin_utilization_avg > pin_utilization_thresh_5:
            congestion_delay = self.d_p * route_utilization_avg + self.d_r * pin_utilization_avg

        return congestion_delay


    def get_logic_delay(self, key):
        """ 
        @brief return logic delay by it's cell type
        @param key either a port name or node type as a key
        """
        return self.delay_logic[key]

    def build_logic_delay_lookup(self):
        """
        @brief build logic delays for different cell types
        """

        with open("%s/%s" % (self.timing_file_dir, "logic_delays.txt"), 'r') as fin:
            for line in fin:
                key, value = line.split()
                self.delay_logic[key] = int(value)

    def curve_fit_linear(self):
        """
        @brief curve fit using linear function
        """

        delay_vert = {} # vertical delay from a x coordinate to num_sites_x/2 -1 
        delay_vert_max = 0 
        delay_vert_min = 1000000
        delay_hort = {} # horizontal delay from a y coordinate to num_sites_y/2 -1
        delay_hort_max = 0
        delay_hort_min = 1000000

        abs_dist_x = []
        abs_dist_y = []
        delays = []

        horizontal_delays_dir = "%s/%s" % (self.timing_file_dir, "net_delays_x.txt")
        vertical_delays_dir = "%s/%s" % (self.timing_file_dir, "net_delays_y.txt")

        with open(vertical_delays_dir, 'r') as fin:
            for line in fin:
                y, delay = line.split()
                delay_vert[int(y)] = float(delay)
                if float(delay) > delay_vert_max:
                    delay_vert_max = float(delay)
                if float(delay) < delay_vert_min:
                    delay_vert_min = float(delay)
                abs_dist_x.append(float(0))
                abs_dist_y.append(float(abs(int(y) - (self.num_sites_y/2 - 1))))
                delays.append(float(delay))

        with open(horizontal_delays_dir, 'r') as fin:
            for line in fin:
                x, delay = line.split()
                delay_hort[int(x)] = float(delay)
                if float(delay) > delay_hort_max:
                    delay_hort_max = float(delay)
                if float(delay) < delay_hort_min:
                    delay_hort_min = float(delay)
                abs_dist_x.append(float(abs(int(x) - (self.num_sites_x/2 - 1))))
                abs_dist_y.append(float(0))
                delays.append(float(delay))
        
        a0 = round((delay_hort_max - delay_hort_min)/(self.num_sites_x/2))
        a1 = round((delay_vert_max - delay_vert_min)/(self.num_sites_y/2))

        def delay_func(X, a2):
            delta_x, delta_y = X
            return a0 * delta_x + a1 * delta_y + a2

        # curve fit for net delay
        param, _ = scipy.optimize.curve_fit(delay_func, (abs_dist_x, abs_dist_y), delays)

        bias = round(param[0])
        
        return a0, a1, bias

        

