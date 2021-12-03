##
# @file   PlaceDB.py
# @author Rachel Selina Rajarathnam (DREAMPlaceFPGA)
# @date   Oct 2020
# @brief  FPGA placement database 
#

import sys
import os
import re
import math
import time 
import numpy as np 
import logging
import Params
import dreamplacefpga 
import dreamplacefpga.ops.place_io.place_io as place_io 
import pdb 
from enum import IntEnum 

datatypes = {
        'float32' : np.float32, 
        'float64' : np.float64
        }

class PlaceDBFPGA (object):
    """
    initialization
    To avoid the usage of list, flatten everything.  
    """
    def __init__(self):
        self.rawdb = None # raw placement database, a C++ object
        self.num_physical_nodes = 0 # number of real nodes, including movable nodes, terminals, and terminal_NIs
        self.node_names = [] # name of instances 
        self.node_name2id_map = {} # map instance name to instance id 
        self.node_types = [] # instance types 
        self.node_x = [] # site location
        self.node_y = [] # site location 
        self.node_z = [] # site specific location
        self.ctrlSets = [] #Used for Flops
        self.flat_ctrlSets = [] #Used for Flops
        self.flop2ctrlSetId_map = [] #Used for Flop to ctrlset Id map
        self.node_size_x = []# 1D array, cell width  
        self.node_size_y = []# 1D array, cell height
        self.resource_size_x = None# 1D array, resource type-based cell width  
        self.resource_size_y = None# 1D array, resource type-based cell height

        self.pin_names = [] # pin names 
        self.pin_types = [] # pin types 
        self.pin_offset_x = []# 1D array, pin offset x to its node 
        self.pin_offset_y = []# 1D array, pin offset y to its node 
        self.pin2nodeType_map = [] # 1D array, pin to node type map
        self.node2pin_map = [] # nested array of array to record pins in each instance 
        self.flat_node2pin_map = [] #Flattened array of node2pin_map
        self.flat_node2pin_start_map = [] #Contains start index for flat_node2pin_map
        self.pin2node_map = [] # map pin to node 

        self.net_names = [] # net names 
        self.net2pin_map = [] # nested array of array to record pins in each net 
        self.flat_net2pin_map = [] # flattend version of net2pin_map
        self.flat_net2pin_start_map = [] # starting point for flat_net2pin_map
        self.pin2net_map = None # map pin to net 

        self.num_bins_x = None# number of bins in horizontal direction 
        self.num_bins_y = None# number of bins in vertical direction 
        self.bin_size_x = None# bin width, currently 1 site  
        self.bin_size_y = None# bin height, currently 1 site  

        self.num_sites_x = None # number of sites in horizontal direction
        self.num_sites_y = None # number of sites in vertical direction 
        self.site_type_map = None # site type of each site 
        self.lg_siteXYs = None # site type of each site 
        self.dspSiteXYs = [] #Sites for DSP instances
        self.ramSiteXYs = [] #Sites for RAM instances

        self.xWirelenWt = None #X-directed wirelength weight
        self.yWirelenWt = None #Y-directed wirelength weight
        self.baseWirelenGammaBinRatio = None # The base wirelenGamma is <this value> * average bin size
        self.instDemStddevTrunc = None # We truncate Gaussian distribution outside the instDemStddevTrunc * instDemStddev
        # Resource Area Parameters
        self.gpInstStddev = None 
        self.gpInstStddevTrunc = None 
        self.instDemStddevX = None
        self.instDemStddevY = None
        # Routability and pin density optimization parameters
        self.unitHoriRouteCap = 0
        self.unitVertRouteCap = 0
        self.unitPinCap = 0

        #Area type parameters
        self.filler_size_x = [] #Filler size X for each resourceType
        self.filler_size_y = [] #Filler size Y for each resourceType
        self.targetOverflow = [] #Target overflow
        self.overflowInstDensityStretchRatio = [] #OVFL density stretch ratio

        self.rawdb = None # raw placement database, a C++ object 

        self.num_movable_nodes = 0# number of movable nodes
        self.num_terminals = 0# number of IOs, essentially fixed instances
        self.net_weights = None # weights for each net

        self.xl = None 
        self.yl = None 
        self.xh = None 
        self.yh = None 

        self.num_movable_pins = None 

        self.total_movable_node_area = None # total movable cell area 
        self.total_fixed_node_area = None # total fixed cell area 
        self.total_space_area = None # total placeable space area excluding fixed cells 

        # enable filler cells 
        # the Idea from e-place and RePlace 
        self.total_filler_node_area = None 
        self.num_filler_nodes = 0 

        self.routing_grid_xl = None 
        self.routing_grid_yl = None 
        self.routing_grid_xh = None 
        self.routing_grid_yh = None 
        self.num_routing_grids_x = None
        self.num_routing_grids_y = None
        self.num_routing_layers = None
        self.unit_horizontal_capacity = None # per unit distance, projected to one layer 
        self.unit_vertical_capacity = None # per unit distance, projected to one layer 
        self.unit_horizontal_capacities = None # per unit distance, layer by layer 
        self.unit_vertical_capacities = None # per unit distance, layer by layer 
        self.initial_horizontal_demand_map = None # routing demand map from fixed cells, indexed by (grid x, grid y), projected to one layer  
        self.initial_vertical_demand_map = None # routing demand map from fixed cells, indexed by (grid x, grid y), projected to one layer  
        self.dtype = None
        #Use Fence region structure for different resource type placement
        self.regions = 5 #FF, LUT, DSP, RAM & IO
        #self.regionsLimits = []# array of 1D array with column min/max of x & y locations
        self.flat_region_boxes = []# flat version of regionsLimits
        self.flat_region_boxes_start = []# start indices of regionsLimits, length of num regions + 1
        self.node2fence_region_map = []# map cell to a region, maximum integer if no fence region
        self.node_count = [] #Count of nodes based on resource type
        #Introduce masks
        self.flop_mask = None
        self.lut_mask = None
        self.lut_type = None
        self.ram_mask = None
        self.dsp_mask = None


    """
    @return number of nodes
    """
    @property
    def num_nodes_nofiller(self):
        return self.num_physical_nodes
    """
    @return number of nodes
    """
    @property
    def num_nodes(self):
        return self.num_physical_nodes + self.num_filler_nodes
    """
    @return number of nets
    """
    @property
    def num_nets(self):
        return len(self.net2pin_map)
    """
    @return number of pins 
    """
    @property 
    def num_pins(self):
        return len(self.pin2node_map)

    @property
    def width(self):
        """
        @return width of layout
        """
        return self.num_sites_x

    @property
    def height(self):
        """
        @return height of layout
        """
        return self.num_sites_y

    @property
    def area(self):
        """
        @return area of layout
        """
        return self.width*self.height

    @property
    def routing_grid_size_x(self):
        return (self.routing_grid_xh - self.routing_grid_xl) / self.num_routing_grids_x 

    @property 
    def routing_grid_size_y(self):
        return (self.routing_grid_yh - self.routing_grid_yl) / self.num_routing_grids_y 

    def num_bins(self, l, h, bin_size):
        """
        @brief compute number of bins
        @param l lower bound
        @param h upper bound
        @param bin_size bin size
        @return number of bins
        """
        return int(np.ceil((h-l)/bin_size))

    """
    read all files including .inst, .pin, .net, .routingUtil files 
    """
    def read(self, params):
        self.dtype = datatypes[params.dtype]

        self.rawdb = place_io.PlaceIOFunction.read(params)

        self.initialize_from_rawdb(params)
        self.lut_mask = self.node2fence_region_map == 0
        self.flop_mask = self.node2fence_region_map == 1
        self.dsp_mask = self.node2fence_region_map == 2
        self.ram_mask = self.node2fence_region_map == 3

    def initialize_from_rawdb(self, params):
        """
        @brief initialize data members from raw database
        @param params parameters
        """
        pydb = place_io.PlaceIOFunction.pydb(self.rawdb)

        self.node_names = np.array(pydb.node_names, dtype=np.str_)
        self.node_size_x = np.array(pydb.node_size_x, dtype=self.dtype)
        self.node_size_y = np.array(pydb.node_size_y, dtype=self.dtype)
        self.node_types = np.array(pydb.node_types, dtype=np.str_)
        self.flop_indices = np.array(pydb.flop_indices)
        self.node2fence_region_map = np.array(pydb.node2fence_region_map, dtype=np.int32)
        self.node_x = np.array(pydb.node_x, dtype=self.dtype)
        self.node_y = np.array(pydb.node_y, dtype=self.dtype)
        self.node_z = np.array(pydb.node_z, dtype=np.int32)

        self.node2pin_map = pydb.node2pin_map

        self.flat_node2pin_map = np.array(pydb.flat_node2pin_map, dtype=np.int32)
        self.flat_node2pin_start_map = np.array(pydb.flat_node2pin_start_map, dtype=np.int32)
        self.node2pincount_map = np.array(pydb.node2pincount_map, dtype=np.int32)
        self.net2pincount_map = np.array(pydb.net2pincount_map, dtype=np.int32)
        self.node2outpinIdx_map = np.array(pydb.node2outpinIdx_map, dtype=np.int32)
        self.lut_type = np.array(pydb.lut_type, dtype=np.int32)
        self.node_name2id_map = pydb.node_name2id_map

        self.num_terminals = pydb.num_terminals
        self.num_movable_nodes = pydb.num_movable_nodes
        self.num_physical_nodes = pydb.num_physical_nodes
        self.node_count = np.array(pydb.node_count, dtype=np.int32)

        self.pin_offset_x = np.array(pydb.pin_offset_x, dtype=self.dtype)
        self.pin_offset_y = np.array(pydb.pin_offset_y, dtype=self.dtype)
        self.pin2nodeType_map = np.array(pydb.pin2nodeType_map, dtype=np.int32)

        self.pin_names = np.array(pydb.pin_names, dtype=np.str_)
        self.pin_types = np.array(pydb.pin_types, dtype=np.str_)
        self.pin_typeIds = np.array(pydb.pin_typeIds, dtype=np.int32)
        self.pin2net_map = np.array(pydb.pin2net_map, dtype=np.int32)
        self.pin2node_map = np.array(pydb.pin2node_map, dtype=np.int32)

        self.net_names = np.array(pydb.net_names, dtype=np.str_)
        self.net2pin_map = pydb.net2pin_map

        self.flat_net2pin_map = np.array(pydb.flat_net2pin_map, dtype=np.int32)
        self.flat_net2pin_start_map = np.array(pydb.flat_net2pin_start_map, dtype=np.int32)
        self.net_name2id_map = pydb.net_name2id_map
        self.net_weights = np.array(np.ones(len(self.net_names)), dtype=self.dtype)

        self.num_sites_x = pydb.num_sites_x
        self.num_sites_y = pydb.num_sites_y
        self.site_type_map = pydb.site_type_map
        self.site_type_map = np.array(self.site_type_map)
        self.lg_siteXYs = pydb.lg_siteXYs
        self.lg_siteXYs = np.array(self.lg_siteXYs)

        self.dspSiteXYs = np.array(pydb.dspSiteXYs, dtype=self.dtype)
        self.ramSiteXYs = np.array(pydb.ramSiteXYs, dtype=self.dtype)

        self.flat_region_boxes = np.array(pydb.flat_region_boxes, dtype=self.dtype)
        self.flat_region_boxes_start = np.array(pydb.flat_region_boxes_start, dtype=np.int32)
        self.ctrlSets = np.array(pydb.ctrlSets, dtype=np.int32)
        self.flat_ctrlSets = np.array(pydb.flat_ctrlSets, dtype=np.int32)
        self.flop2ctrlSetId_map = np.zeros(self.num_physical_nodes, dtype=np.int32)
        self.flop2ctrlSetId_map[self.node2fence_region_map == 1] = np.arange(len(self.flop_indices))

        self.num_routing_grids_x = pydb.num_routing_grids_x
        self.num_routing_grids_y = pydb.num_routing_grids_y
        self.routing_grid_xl = float(pydb.routing_grid_xl)
        self.routing_grid_yl = float(pydb.routing_grid_yl)
        self.routing_grid_xh = float(pydb.routing_grid_xh)
        self.routing_grid_yh = float(pydb.routing_grid_yh)

        self.xl = float(pydb.xl)
        self.yl = float(pydb.yl)
        self.xh = float(pydb.xh)
        self.yh = float(pydb.yh)
        self.row_height = float(pydb.row_height)
        self.site_width = float(pydb.site_width)

        self.num_routing_layers = 1
        self.unit_horizontal_capacity = 0.95 * params.unit_horizontal_capacity
        self.unit_vertical_capacity = 0.95 * params.unit_vertical_capacity

    def print_node(self, node_id):
        """
        @brief print node information
        @param node_id cell index
        """
        logging.debug("node %s(%d), size (%g, %g), pos (%g, %g)" % (self.node_names[node_id], node_id, self.node_size_x[node_id], self.node_size_y[node_id], self.node_x[node_id], self.node_y[node_id]))
        pins = "pins "
        for pin_id in self.node2pin_map[node_id]:
            pins += "%s(%s, %d) " % (self.node_names[self.pin2node_map[pin_id]], self.net_names[self.pin2net_map[pin_id]], pin_id)
        logging.debug(pins)

    def print_net(self, net_id):
        """
        @brief print net information
        @param net_id net index
        """
        logging.debug("net %s(%d)" % (self.net_names[net_id], net_id))
        pins = "pins "
        for pin_id in self.net2pin_map[net_id]:
            pins += "%s(%s, %d) " % (self.node_names[self.pin2node_map[pin_id]], self.net_names[self.pin2net_map[pin_id]], pin_id)
        logging.debug(pins)

    def flatten_nested_map(self, net2pin_map):
        """
        @brief flatten an array of array to two arrays like CSV format
        @param net2pin_map array of array
        @return a pair of (elements, cumulative column indices of the beginning element of each row)
        """
        # flat netpin map, length of #pins
        flat_net2pin_map = np.zeros(len(pin2net_map), dtype=np.int32)
        # starting index in netpin map for each net, length of #nets+1, the last entry is #pins
        flat_net2pin_start_map = np.zeros(len(net2pin_map)+1, dtype=np.int32)
        count = 0
        for i in range(len(net2pin_map)):
            flat_net2pin_map[count:count+len(net2pin_map[i])] = net2pin_map[i]
            flat_net2pin_start_map[i] = count
            count += len(net2pin_map[i])
        assert flat_net2pin_map[-1] != 0
        flat_net2pin_start_map[len(net2pin_map)] = len(pin2net_map)

        return flat_net2pin_map, flat_net2pin_start_map

    def __call__(self, params):
        """
        @brief top API to read placement files 
        @param params parameters 
        """
        tt = time.time()

        self.read(params)
        self.initialize(params)

        logging.info("reading benchmark takes %g seconds" % (time.time()-tt))

    def calc_num_filler_for_fence_region(self, region_id, node2fence_region_map, filler_size_x, filler_size_y):
        '''
        @description: calculate number of fillers for each fence region
        @param fence_regions{type}
        @return:
        '''
        num_regions = self.regions-1
        node2fence_region_map = node2fence_region_map[:self.num_movable_nodes]
        
        if(region_id < self.regions-1):
            fence_region_mask = (node2fence_region_map == region_id)
        else:
            fence_region_mask = (node2fence_region_map >= self.regions-1)

        num_movable_nodes = self.num_movable_nodes
        movable_node_size_x = self.node_size_x[:num_movable_nodes][fence_region_mask]
        movable_node_size_y = self.node_size_y[:num_movable_nodes][fence_region_mask]

        #Updated by Rachel - Calcuation based on region size 
        if (region_id < self.regions-1):
            region = self.flat_region_boxes[self.flat_region_boxes_start[region_id]:self.flat_region_boxes_start[region_id+1]]
            placeable_area = np.sum((region[:, 2]-region[:, 0])*(region[:, 3]-region[:, 1]))
        total_movable_node_area = np.sum(self.node_size_x[:num_movable_nodes][fence_region_mask]*self.node_size_y[:num_movable_nodes][fence_region_mask])

        if (region_id >= self.regions-1):
            return 0, 0, self.num_terminals

        #If no cells of particular resourceType
        if np.sum(fence_region_mask) == 0:
            return 0, 0, 0.0

        total_filler_node_area = max(placeable_area-total_movable_node_area, 0.0)

        num_filler = int(math.floor(total_filler_node_area/(filler_size_x*filler_size_y)))
        logging.info("Region:%2d #movable_nodes = %8d movable_node_area =%10.1f, placeable_area =%10.1f, filler_node_area =%10.1f, #fillers =%8d, filler sizes =%2.4gx%g\n" % (region_id, fence_region_mask.sum(), total_movable_node_area, placeable_area, total_filler_node_area, num_filler, filler_size_x, filler_size_y))

        return num_filler, total_movable_node_area, np.sum(fence_region_mask).astype(np.float32)


    def initialize(self, params):
        """
        @brief initialize data members after reading 
        @param params parameters 
        """
        self.resource_size_x = np.ones(4, dtype=params.dtype)
        self.resource_size_y = np.ones(4, dtype=params.dtype)
        self.resource_size_y[2] = 2.5
        self.resource_size_y[3] = 5.0

        #Parameter initialization - Can be changed later through params
        self.xWirelenWt = 0.7
        self.yWirelenWt = 1.2
        self.instDemStddevTrunc = 2.5
        
        #Resource area parameter
        self.gpInstStddev = math.sqrt(2.5e-4 * self.num_nodes) / (2.0 * self.instDemStddevTrunc)
        self.gpInstStddevTrunc = self.instDemStddevTrunc
        
        self.instDemStddevX = self.gpInstStddev
        self.instDemStddevY = self.gpInstStddev

        #Parameter for Direct Legalization
        self.nbrDistEnd = 1.2 * self.gpInstStddev * self.gpInstStddevTrunc
        
        # Routability and pin density optimization parameters
        self.unitPinCap = 0

        #Area type parameters - Consider default fillerstrategy of FIXED_SHAPE
        #   0 - LUT
        #   1 - FF
        #   2 - DSP
        #   3 - RAM

        self.filler_size_x = np.zeros(4)
        self.filler_size_y = np.zeros(4)
        self.targetOverflow = np.zeros(4)
        self.overflowInstDensityStretchRatio = np.zeros(4)

        # 0 - LUT
        self.filler_size_x[0] = math.sqrt(0.125)
        self.filler_size_y[0] = math.sqrt(0.125)
        self.targetOverflow[0] = 0.1
        self.overflowInstDensityStretchRatio[0] = math.sqrt(2.0)

        # 1 - FF
        self.filler_size_x[1] = math.sqrt(0.125)
        self.filler_size_y[1] = math.sqrt(0.125)
        self.targetOverflow[1] = 0.1
        self.overflowInstDensityStretchRatio[1] = math.sqrt(2.0)

        # 2 - DSP
        self.filler_size_x[2] = 1.0
        self.filler_size_y[2] = 2.5
        self.targetOverflow[2] = 0.2
        self.overflowInstDensityStretchRatio[2] = 0

        # 3 - RAM
        self.filler_size_x[3] = 1.0
        self.filler_size_y[3] = 5.0
        self.targetOverflow[3] = 0.2
        self.overflowInstDensityStretchRatio[3] = 0

        #set number of bins
        self.num_bins_x = 512
        self.num_bins_y = 512
        self.bin_size_x = self.width/self.num_bins_x
        self.bin_size_y = self.height/self.num_bins_y

        # set total cell area
        movable_cell_region_mask01 = (self.node2fence_region_map[:self.num_movable_nodes] < 2)
        movable_cell_region_mask23 = (self.node2fence_region_map[:self.num_movable_nodes] == 2) | (self.node2fence_region_map[:self.num_movable_nodes] == 3)
        self.total_movable_node_area = float(np.sum(movable_cell_region_mask01)*self.filler_size_x[0]*self.filler_size_y[0])
        if movable_cell_region_mask23.sum() > 0:
            self.total_movable_node_area += float(np.sum(self.node_size_x[:self.num_movable_nodes][movable_cell_region_mask23]*self.node_size_y[:self.num_movable_nodes][movable_cell_region_mask23]))
        # total fixed node area should exclude the area outside the layout and the area of terminal_NIs
        self.total_fixed_node_area = float(self.num_terminals)
        self.total_space_area = self.width * self.height

        self.region_boxes = []

        ## calculate fence region virtual macro
        ##Rachel: For FPGA, the regions are fixed for each resourceType
        #virtual_macro_for_fence_region = []
        for region_id in range(self.regions-1):
            #if region_id >= 4:
            #    continue
            region = self.flat_region_boxes[self.flat_region_boxes_start[region_id]:self.flat_region_boxes_start[region_id+1]] 
            self.region_boxes.append(region)

        # insert filler nodes
        ### calculate fillers for different resourceTypes
        self.filler_size_x_fence_region = []
        self.filler_size_y_fence_region = []
        self.num_filler_nodes = 0
        self.num_filler_nodes_fence_region = []
        self.num_movable_nodes_fence_region = []
        self.total_movable_node_area_fence_region = []
        self.target_density_fence_region = []
        self.filler_start_map = None
        filler_node_size_x_list = []
        filler_node_size_y_list = []
        self.total_filler_node_area = 0

        for i in range(len(self.region_boxes)):
            num_filler_i, total_movable_node_area_i, num_movable_nodes_i = self.calc_num_filler_for_fence_region(i, self.node2fence_region_map, self.filler_size_x[i], self.filler_size_y[i])
            self.num_movable_nodes_fence_region.append(num_movable_nodes_i)
            self.num_filler_nodes_fence_region.append(num_filler_i)
            self.total_movable_node_area_fence_region.append(total_movable_node_area_i)
            self.target_density_fence_region.append(self.targetOverflow[i])
            self.filler_size_x_fence_region.append(self.filler_size_x[i])
            self.filler_size_y_fence_region.append(self.filler_size_y[i])
            self.num_filler_nodes += num_filler_i
            filler_node_size_x_list.append(np.full(num_filler_i, fill_value=self.filler_size_x[i], dtype=self.node_size_x.dtype))
            filler_node_size_y_list.append(np.full(num_filler_i, fill_value=self.filler_size_y[i], dtype=self.node_size_y.dtype))
            filler_node_area_i = num_filler_i * (self.filler_size_x[i]*self.filler_size_y[i])
            self.total_filler_node_area += filler_node_area_i

        self.total_movable_node_area_fence_region = np.array(self.total_movable_node_area_fence_region)
        self.num_movable_nodes_fence_region = np.array(self.num_movable_nodes_fence_region)

        if params.enable_fillers:
            # the way to compute this is still tricky; we need to consider place_io together on how to
            # summarize the area of fixed cells, which may overlap with each other.
            self.filler_start_map = np.cumsum([0]+self.num_filler_nodes_fence_region)
            self.num_filler_nodes_fence_region = np.array(self.num_filler_nodes_fence_region)
            self.node_size_x = np.concatenate([self.node_size_x] + filler_node_size_x_list)
            self.node_size_y = np.concatenate([self.node_size_y] + filler_node_size_y_list)
        else:
            self.total_filler_node_area = 0
            self.num_filler_nodes = 0
            filler_size_x, filler_size_y = 0, 0
            if(len(self.region_boxes) > 0):
                self.filler_start_map = np.zeros(len(self.region_boxes)+1, dtype=np.int32)
                self.num_filler_nodes_fence_region = np.zeros(len(self.num_filler_nodes_fence_region))

    def write(self, params, pl_file):
        """
        @brief write placement solution as .pl file
        @param pl_file .pl file 
        """
        tt = time.time()
        #logging.info("writing to %s" % (pl_file))

        node_x = self.node_x
        node_y = self.node_y
        node_z = self.node_z

        content = ""
        str_node_names = self.node_names
        node_area = self.node_size_x*self.node_size_y
        for i in range(self.num_physical_nodes):
            content += "\n%s %.6E %.6E %g %.6E" % (
                    str_node_names[i],
                    node_x[i], 
                    node_y[i], 
                    node_z[i],
                    node_area[i]
                    )
        with open(pl_file, "w") as f:
            f.write(content)
        logging.info("write placement solution to %s took %.3f seconds" % (pl_file, time.time()-tt))

    def read_pl(self, params, pl_file):
        """
        @brief read .pl file
        @param pl_file .pl file
        """
        tt = time.time()
        logging.info("reading %s" % (pl_file))
        count = 0
        with open(pl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("UCLA"):
                    continue
                # node positions
                pos = re.search(r"(\w+)\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s*:\s*(\w+)", line)
                if pos:
                    node_id = self.node_name2id_map[pos.group(1)]
                    self.node_x[node_id] = float(pos.group(2))
                    self.node_y[node_id] = float(pos.group(6))
                    self.node_orient[node_id] = pos.group(10)
                    orient = pos.group(4)
        #if params.scale_factor != 1.0:
        #    self.scale_pl(params.scale_factor)
        logging.info("read_pl takes %.3f seconds" % (time.time()-tt))

    def apply(self, params, node_x, node_y, node_z):
        """
        @brief apply placement solution and update database 
        """

        # assign solution
        self.node_x[:self.num_movable_nodes] = node_x[:self.num_movable_nodes]
        self.node_y[:self.num_movable_nodes] = node_y[:self.num_movable_nodes]
        self.node_z[:self.num_movable_nodes] = node_z[:self.num_movable_nodes]

        node_x = self.node_x
        node_y = self.node_y
        node_z = self.node_z

        # update raw database 
        place_io.PlaceIOFunction.apply(self.rawdb, node_x.astype(params.dtype), node_y.astype(params.dtype), node_z.astype(np.int32))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("One input parameters in json format in required")

    params = Params.Params()
    params.load(sys.argv[sys.argv[1]])
    logging.info("parameters = %s" % (params))

    db = PlaceDB()
    db(params)

    db.print_node(1)
    db.print_net(1)
    db.print_row(1)


