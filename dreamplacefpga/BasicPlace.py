##
# @file   BasicPlace.py
# @author Yibo Lin (DREAMPlace), Rachel Selina Rajarathnam (DREAMPlaceFPGA)
# @date   Sep 2020
# @brief  Base placement class
#

import os
import sys
import time
import gzip
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import re
import numpy as np
import logging
import torch
import torch.nn as nn
import dreamplacefpga.ops.move_boundary.move_boundary as move_boundary
import dreamplacefpga.ops.hpwl.hpwl as hpwl
import dreamplacefpga.ops.electric_potential.electric_overflow as electric_overflow
import dreamplacefpga.ops.draw_place.draw_place as draw_place
import dreamplacefpga.ops.pin_pos.pin_pos as pin_pos
import dreamplacefpga.ops.precondWL.precondWL as precondWL
import dreamplacefpga.ops.demandMap.demandMap as demandMap
import dreamplacefpga.ops.sortNode2Pin.sortNode2Pin as sortNode2Pin
import dreamplacefpga.ops.lut_ff_legalization.lut_ff_legalization as lut_ff_legalization
import pdb

datatypes = {
        'float32' : torch.float32, 
        'float64' : torch.float64
        }

class PlaceDataCollectionFPGA(object):
    """
    @brief A wraper for all data tensors on device for building ops 
    """
    def __init__(self, pos, params, placedb, device):
        """
        @brief initialization 
        @param pos locations of cells 
        @param params parameters 
        @param placedb placement database 
        @param device cpu or cuda 
        """
        self.device = device
        torch.set_num_threads(params.num_threads)
        # position should be parameter
        self.pos = pos

        with torch.no_grad():
            # other tensors required to build ops

            self.node_size_x = torch.from_numpy(placedb.node_size_x).to(device)
            self.node_size_y = torch.from_numpy(placedb.node_size_y).to(device)
            self.resource_size_x = torch.from_numpy(placedb.resource_size_x).to(device)
            self.resource_size_y = torch.from_numpy(placedb.resource_size_y).to(device)
            self.node_x = torch.from_numpy(placedb.node_x).to(device)
            self.node_y = torch.from_numpy(placedb.node_y).to(device)
            self.node_z = torch.from_numpy(placedb.node_z.astype(np.int32)).to(device)
            self.site_type_map = torch.from_numpy(placedb.site_type_map.astype(np.int32)).to(device)
            self.lg_siteXYs = torch.from_numpy(placedb.lg_siteXYs).to(device)

            if params.routability_opt_flag:
                self.original_node_size_x = self.node_size_x.clone()
                self.original_node_size_y = self.node_size_y.clone()

            self.pin_offset_x = torch.from_numpy(placedb.pin_offset_x).to(device)
            self.pin_offset_y = torch.from_numpy(placedb.pin_offset_y).to(device)
            self.lg_pin_offset_x = torch.from_numpy(placedb.lg_pin_offset_x).to(device)
            self.lg_pin_offset_y = torch.from_numpy(placedb.lg_pin_offset_y).to(device)

            # original pin offset for legalization, since they will be adjusted in global placement
            if params.routability_opt_flag:
                self.original_pin_offset_x = self.pin_offset_x.clone()
                self.original_pin_offset_y = self.pin_offset_y.clone()

            self.node_areas = self.node_size_x * self.node_size_y
            self.movable_macro_mask = None

            self.pin2node_map = torch.from_numpy(placedb.pin2node_map).to(device)
            self.flat_node2pin_map = torch.from_numpy(placedb.flat_node2pin_map).to(device)
            self.flat_node2pin_start_map = torch.from_numpy(placedb.flat_node2pin_start_map).to(device)
            self.node2outpinIdx_map = torch.from_numpy(placedb.node2outpinIdx_map).to(device)
            self.node2pincount_map = torch.from_numpy(placedb.node2pincount_map).to(device)
            self.net2pincount_map = torch.from_numpy(placedb.net2pincount_map).to(device)

            self.dspSiteXYs = torch.from_numpy(placedb.dspSiteXYs).to(dtype=datatypes[params.dtype],device=device)
            self.ramSiteXYs = torch.from_numpy(placedb.ramSiteXYs).to(dtype=datatypes[params.dtype],device=device)

            # number of pins for each cell
            self.pin_weights = (self.flat_node2pin_start_map[1:] -
                                self.flat_node2pin_start_map[:-1]).to(
                                    self.node_size_x.dtype)
            ## Resource type masks
            self.flop_mask = torch.from_numpy(placedb.flop_mask).to(device)
            self.lut_mask = torch.from_numpy(placedb.lut_mask).to(device)
            self.ram_mask = torch.from_numpy(placedb.ram_mask).to(device)
            self.dsp_mask = torch.from_numpy(placedb.dsp_mask).to(device)
            self.flop_lut_mask = self.flop_mask | self.lut_mask
            self.dsp_ram_mask = self.dsp_mask | self.ram_mask

            #LUT type list
            self.lut_type = torch.from_numpy(placedb.lut_type).to(dtype=torch.int32,device=device)
            self.cluster_lut_type = torch.from_numpy(placedb.cluster_lut_type).to(dtype=torch.int32,device=device)
            self.pin_typeIds = torch.from_numpy(placedb.pin_typeIds).to(dtype=torch.int32,device=device)

            #FF control sets
            self.flop_ctrlSets = torch.from_numpy(placedb.flat_ctrlSets).to(dtype=torch.int32,device=device)
            #FF to ctrlset ID
            self.flop2ctrlSetId_map = torch.from_numpy(placedb.flop2ctrlSetId_map).to(dtype=torch.int32,device=device)
            #Spiral accessor for legalization
            self.spiral_accessor = torch.from_numpy(placedb.spiral_accessor).to(dtype=torch.int32,device=device)

            #Resource type indexing
            self.flop_indices = torch.from_numpy(placedb.flop_indices).to(dtype=torch.int32,device=device)
            self.lut_indices = torch.nonzero(self.lut_mask, as_tuple=True)[0].to(dtype=torch.int32)
            self.flop_lut_indices = torch.nonzero(self.flop_lut_mask, as_tuple=True)[0].to(dtype=torch.int32)
            self.pin_weights[self.flop_mask] = params.ffPinWeight
            self.unit_pin_capacity = torch.empty(1, dtype=self.pos[0].dtype, device=device)
            self.unit_pin_capacity.data.fill_(params.unit_pin_capacity)

            # routing information
            # project initial routing utilization map to one layer
            self.initial_horizontal_utilization_map = None
            self.initial_vertical_utilization_map = None
            if params.routability_opt_flag and placedb.initial_horizontal_demand_map is not None:
                self.initial_horizontal_utilization_map = torch.from_numpy(
                    placedb.initial_horizontal_demand_map).to(device).div_(
                        placedb.routing_grid_size_y *
                        placedb.unit_horizontal_capacity)
                self.initial_vertical_utilization_map = torch.from_numpy(
                    placedb.initial_vertical_demand_map).to(device).div_(
                        placedb.routing_grid_size_x *
                        placedb.unit_vertical_capacity)

            self.pin2net_map = torch.from_numpy(placedb.pin2net_map.astype(np.int32)).to(device)
            self.flat_net2pin_map = torch.from_numpy(placedb.flat_net2pin_map).to(device)
            self.flat_net2pin_start_map = torch.from_numpy(placedb.flat_net2pin_start_map).to(device)
            if np.amin(placedb.net_weights) == np.amax(placedb.net_weights):  # empty tensor
                logging.warning("net weights are all the same, ignored")
                #self.net_weights = torch.Tensor().to(device)
            self.net_weights = torch.from_numpy(placedb.net_weights).to(device)

            # regions
            self.region_boxes = [torch.tensor(region).to(device) for region in placedb.region_boxes]
            self.flat_region_boxes = torch.from_numpy(
                placedb.flat_region_boxes).to(device)
            self.flat_region_boxes_start = torch.from_numpy(
                placedb.flat_region_boxes_start).to(device)
            self.node2fence_region_map = torch.from_numpy(
                placedb.node2fence_region_map).to(device)

            self.num_nodes = torch.tensor(placedb.num_nodes, dtype=torch.int32, device=device)
            self.num_movable_nodes = torch.tensor(placedb.num_movable_nodes, dtype=torch.int32, device=device)
            self.num_filler_nodes = torch.tensor(placedb.num_filler_nodes, dtype=torch.int32, device=device)
            self.num_physical_nodes = torch.tensor(placedb.num_physical_nodes, dtype=torch.int32, device=device)
            self.filler_start_map = torch.from_numpy(placedb.filler_start_map).to(device)

            ## this is for overflow op
            self.total_movable_node_area_fence_region = torch.from_numpy(placedb.total_movable_node_area_fence_region).to(device)
            ## this is for gamma update
            self.num_movable_nodes_fence_region = torch.from_numpy(placedb.num_movable_nodes_fence_region).to(device)
            ## this is not used yet
            self.num_filler_nodes_fence_region = torch.from_numpy(placedb.num_filler_nodes_fence_region).to(device)

            self.net_mask_all = torch.from_numpy(np.ones(placedb.num_nets,dtype=np.uint8)).to(device)  # all nets included
            net_degrees = np.array([len(net2pin) for net2pin in placedb.net2pin_map])
            net_mask = np.logical_and(2 <= net_degrees,
                net_degrees < params.ignore_net_degree).astype(np.uint8)
            self.net_mask_ignore_large_degrees = torch.from_numpy(net_mask).to(device)  # nets with large degrees are ignored

            # For WL computation
            self.net_bounding_box_min = torch.zeros(placedb.num_nets * 2, dtype=datatypes[params.dtype], device=self.device)
            self.net_bounding_box_max = torch.zeros_like(self.net_bounding_box_min)

            # avoid computing gradient for fixed macros
            # 1 is for fixed macros - IOs
            self.pin_mask_ignore_fixed_macros = (self.pin2node_map >= placedb.num_movable_nodes)

            # sort nodes by size, return their sorted indices, designed for memory coalesce in electrical force
            movable_size_x = self.node_size_x[:placedb.num_movable_nodes]
            _, self.sorted_node_map = torch.sort(movable_size_x)
            self.sorted_node_map = self.sorted_node_map.to(torch.int32)

class PlaceOpCollectionFPGA(object):
    """
    @brief A wrapper for all ops
    """
    def __init__(self):
        """
        @brief initialization
        """
        self.demandMap_op = None
        self.pin_pos_op = None
        self.move_boundary_op = None
        self.hpwl_op = None
        self.precondwl_op = None
        self.wirelength_op = None
        self.update_gamma_op = None
        self.density_op = None
        self.update_density_weight_op = None
        self.precondition_op = None
        self.noise_op = None
        self.draw_place_op = None
        self.route_utilization_map_op = None
        self.pin_utilization_map_op = None
        self.clustering_compatibility_lut_area_op= None
        self.clustering_compatibility_ff_area_op= None
        self.adjust_node_area_op = None
        self.sort_node2pin_op = None
        self.lut_ff_legalization_op = None

class BasicPlaceFPGA(nn.Module):
    """
    @brief Base placement class. 
    All placement engines should be derived from this class. 
    """
    def __init__(self, params, placedb):
        """
        @brief initialization
        @param params parameter 
        @param placedb placement database 
        """
        torch.manual_seed(params.random_seed)
        super(BasicPlaceFPGA, self).__init__()

        self.init_pos = np.zeros(placedb.num_nodes * 2, dtype=placedb.dtype)

        ##Settings to ensure reproduciblity
        manualSeed = 0
        np.random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        if params.gpu:
            torch.cuda.manual_seed(manualSeed)
            torch.cuda.manual_seed_all(manualSeed)

        numPins = 0
        initLocX = 0
        initLocY = 0

        if placedb.num_terminals > 0:
            numPins = 0
            ##Use the average fixed pin location (weighted by pin count) as the initial location
            for nodeID in range(placedb.num_movable_nodes,placedb.num_physical_nodes):
                for pID in placedb.node2pin_map[nodeID]:
                    initLocX += placedb.node_x[nodeID] + placedb.pin_offset_x[pID]
                    initLocY += placedb.node_y[nodeID] + placedb.pin_offset_y[pID]
                numPins += len(placedb.node2pin_map[nodeID])
            initLocX /= numPins
            initLocY /= numPins
        else: ##Design does not have IO pins - place in center
            initLocX = 0.5 * (placedb.xh - placedb.xl)
            initLocY = 0.5 * (placedb.yh - placedb.yl)

        # x position
        self.init_pos[0:placedb.num_physical_nodes] = placedb.node_x
        if params.global_place_flag and params.random_center_init_flag:  # move to centroid of layout
            #logging.info("Move cells to the centroid of fixed IOs with random noise")
            self.init_pos[0:placedb.num_movable_nodes] = np.random.normal(
                loc = initLocX,
                scale = min(placedb.xh - placedb.xl, placedb.yh - placedb.yl) * 0.001,
                size = placedb.num_movable_nodes)
        self.init_pos[0:placedb.num_movable_nodes] -= (0.5 * placedb.node_size_x[0:placedb.num_movable_nodes])

        # y position
        self.init_pos[placedb.num_nodes:placedb.num_nodes+placedb.num_physical_nodes] = placedb.node_y
        if params.global_place_flag and params.random_center_init_flag:  # move to center of layout
            self.init_pos[placedb.num_nodes:placedb.num_nodes+placedb.num_movable_nodes] = np.random.normal(
                loc = initLocY,
                scale = min(placedb.xh - placedb.xl, placedb.yh - placedb.yl) * 0.001,
                size = placedb.num_movable_nodes)
        self.init_pos[placedb.num_nodes:placedb.num_nodes+placedb.num_movable_nodes] -= (0.5 * placedb.node_size_y[0:placedb.num_movable_nodes])
        #logging.info("Random Init Place in python takes %.2f seconds" % (time.time() - tt))

        if placedb.num_filler_nodes:  # uniformly distribute filler cells in the layout
            ### uniformly spread fillers in fence region
            ### for cells in the fence region
            for i, region in enumerate(placedb.region_boxes):
                if i < 4:
                    #Construct Nx4 np array for region using placedb.flat_region_boxes
                    filler_beg, filler_end = placedb.filler_start_map[i:i+2]
                    if filler_end-filler_beg > 0:
                        subregion_areas = (region[:,2]-region[:,0])*(region[:,3]-region[:,1])
                        total_area = np.sum(subregion_areas)
                        subregion_area_ratio = subregion_areas / total_area
                        subregion_num_filler = np.round((filler_end - filler_beg) * subregion_area_ratio)
                        subregion_num_filler[-1] = (filler_end - filler_beg) - np.sum(subregion_num_filler[:-1])
                        subregion_num_filler_start_map = np.concatenate([np.zeros([1]),np.cumsum(subregion_num_filler)],0).astype(np.int32)
                        for j, subregion in enumerate(region):
                            sub_filler_beg, sub_filler_end = subregion_num_filler_start_map[j:j+2]
                            self.init_pos[placedb.num_physical_nodes+filler_beg+sub_filler_beg:placedb.num_physical_nodes+filler_beg+sub_filler_end]=np.random.uniform(
                                    low=subregion[0],
                                    high=subregion[2] -
                                    placedb.filler_size_x_fence_region[i],
                                    size=sub_filler_end-sub_filler_beg)
                            self.init_pos[placedb.num_nodes+placedb.num_physical_nodes+filler_beg+sub_filler_beg:placedb.num_nodes+placedb.num_physical_nodes+filler_beg+sub_filler_end]=np.random.uniform(
                                    low=subregion[1],
                                    high=subregion[3] -
                                    placedb.filler_size_y_fence_region[i],
                                    size=sub_filler_end-sub_filler_beg)
                #Skip for IOs - regions[4]
                else:
                    continue

            #logging.info("Random Init Place in Python takes %.2f seconds" % (time.time() - t2))
        
        self.device = torch.device("cuda" if params.gpu else "cpu")

        # position should be parameter
        # must be defined in BasicPlace
        #tbp = time.time()
        self.pos = nn.ParameterList(
            [nn.Parameter(torch.from_numpy(self.init_pos).to(self.device))])
        #logging.info("build pos takes %.2f seconds" % (time.time() - tbp))
        # shared data on device for building ops
        # I do not want to construct the data from placedb again and again for each op
        #tt = time.time()
        self.data_collections = PlaceDataCollectionFPGA(self.pos, params, placedb, self.device)
        #logging.info("build data_collections takes %.2f seconds" %
        #              (time.time() - tt))

        # similarly I wrap all ops
        #tt = time.time()
        self.op_collections = PlaceOpCollectionFPGA()
        #logging.info("build op_collections takes %.2f seconds" %
        #              (time.time() - tt))

        tt = time.time()
        # Demand Map computation
        self.op_collections.demandMap_op = self.build_demandMap(params, placedb, self.data_collections, self.device)
        # position to pin position
        self.op_collections.pin_pos_op = self.build_pin_pos(params, placedb, self.data_collections, self.device)
        # bound nodes to layout region
        self.op_collections.move_boundary_op = self.build_move_boundary(params, placedb, self.data_collections, self.device)
        # hpwl and density overflow ops for evaluation
        self.op_collections.hpwl_op = self.build_hpwl(params, placedb, self.data_collections, self.op_collections.pin_pos_op, self.device)
        # WL preconditioner
        self.op_collections.precondwl_op = self.build_precondwl(params, placedb, self.data_collections, self.device)
        # Sorting node2pin map
        self.op_collections.sort_node2pin_op = self.build_sortNode2Pin(params, placedb, self.data_collections, self.device)
        # rectilinear minimum steiner tree wirelength from flute
        # can only be called once
        self.op_collections.density_overflow_op = self.build_electric_overflow(params, placedb, self.data_collections, self.device)

        #Legalization
        self.op_collections.lut_ff_legalization_op = self.build_lut_ff_legalization(params, placedb, self.data_collections, self.device)
 
        # draw placement
        self.op_collections.draw_place_op = self.build_draw_placement(params, placedb)

        # flag for rmst_wl_op
        # can only read once
        self.read_lut_flag = True

        #logging.info("build BasicPlace ops takes %.2f seconds" %
        #              (time.time() - tt))

    def __call__(self, params, placedb):
        """
        @brief Solve placement.
        placeholder for derived classes.
        @param params parameters
        @param placedb placement database
        """
        pass

    def build_pin_pos(self, params, placedb, data_collections, device):
        """
        @brief sum up the pins for each cell
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        # Yibo: I found CPU version of this is super slow, more than 2s for ISPD2005 bigblue4 with 10 threads.
        # So I implemented a custom CPU version, which is around 20ms
        #pin2node_map = data_collections.pin2node_map.long()
        #def build_pin_pos_op(pos):
        #    pin_x = data_collections.pin_offset_x.add(torch.index_select(pos[0:placedb.num_physical_nodes], dim=0, index=pin2node_map))
        #    pin_y = data_collections.pin_offset_y.add(torch.index_select(pos[placedb.num_nodes:placedb.num_nodes+placedb.num_physical_nodes], dim=0, index=pin2node_map))
        #    pin_pos = torch.cat([pin_x, pin_y], dim=0)

        #    return pin_pos
        #return build_pin_pos_op

        return pin_pos.PinPos(
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            pin2node_map=data_collections.pin2node_map,
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            num_physical_nodes=placedb.num_physical_nodes,
            num_threads=params.num_threads,
            algorithm="node-by-node")

    def build_move_boundary(self, params, placedb, data_collections, device):
        """
        @brief bound nodes into layout region
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        return move_boundary.MoveBoundary(
            data_collections.node_size_x,
            data_collections.node_size_y,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
            num_threads=params.num_threads)

    def build_hpwl(self, params, placedb, data_collections, pin_pos_op, device):
        """
        @brief compute half-perimeter wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        @param device cpu or cuda
        """
        wirelength_for_pin_op = hpwl.HPWL(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            #net_mask=data_collections.net_mask_all,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            net_bounding_box_min=data_collections.net_bounding_box_min,
            net_bounding_box_max=data_collections.net_bounding_box_max,
            num_threads=params.num_threads,
            algorithm='net-by-net')

        # wirelength for position
        def build_wirelength_op(pos):
            return wirelength_for_pin_op(pin_pos_op(pos))

        return build_wirelength_op

    def build_demandMap(self, params, placedb, data_collections, device):
        """
        @brief Build binCapMap and fixedDemandMap
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        return demandMap.DemandMap(
            site_type_map=data_collections.site_type_map,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            width=placedb.width,
            height=placedb.height,
            node_size_x=data_collections.resource_size_x,
            node_size_y=data_collections.resource_size_y,
            xh=placedb.xh,
            xl=placedb.xl,
            yh=placedb.yh,
            yl=placedb.yl,
            deterministic_flag=params.deterministic_flag,
            device=device,
            num_threads=params.num_threads)

    def build_precondwl(self, params, placedb, data_collections, device):
        """
        @brief compute wirelength precondtioner
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        return precondWL.PrecondWL(
            flat_node2pin_start=data_collections.flat_node2pin_start_map,
            flat_node2pin=data_collections.flat_node2pin_map,
            pin2net_map=data_collections.pin2net_map,
            flat_net2pin=data_collections.flat_net2pin_start_map,
            net_weights=data_collections.net_weights,
            num_nodes=placedb.num_nodes,
            num_movable_nodes=placedb.num_physical_nodes,#Compute for fixed nodes as well for Legalization
            device=device,
            num_threads=params.num_threads)

    def build_sortNode2Pin(self, params, placedb, data_collections, device):
        """
        @brief sort instance node2pin mapping
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        return sortNode2Pin.SortNode2Pin(
            flat_node2pin_start=data_collections.flat_node2pin_start_map,
            flat_node2pin=data_collections.flat_node2pin_map,
            num_nodes=placedb.num_physical_nodes,
            device=device,
            num_threads=params.num_threads)

    def build_electric_overflow(self, params, placedb, data_collections, device):
        """
        @brief compute electric density overflow
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        return electric_overflow.ElectricOverflow(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=placedb.bin_size_x,
            bin_size_y=placedb.bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=0,
            deterministic_flag=params.deterministic_flag,
            sorted_node_map=data_collections.sorted_node_map)


    def build_lut_ff_legalization(self, params, placedb, data_collections, device):
        """
        @brief legalization of LUT/FF Instances
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        # legalize LUT/FF
        #Avg areas
        avgLUTArea = data_collections.node_areas[:placedb.num_physical_nodes][data_collections.node2fence_region_map == 0].sum()
        avgLUTArea /= placedb.node_count[0]
        avgFFArea = data_collections.node_areas[:placedb.num_physical_nodes][data_collections.node2fence_region_map == 1].sum()
        avgFFArea /= placedb.node_count[1]
        #Inst Areas
        inst_areas = data_collections.node_areas[:placedb.num_physical_nodes].detach().clone()
        inst_areas[data_collections.node2fence_region_map > 1] = 0.0 #Area of non CLB nodes set to 0.0
        inst_areas[data_collections.node2fence_region_map == 0] /= avgLUTArea
        inst_areas[data_collections.node2fence_region_map == 1] /= avgFFArea
        #Site types
        site_types = data_collections.site_type_map.detach().clone()
        site_types[site_types > 1] = 0 #Set non CLB to 0

        if (len(data_collections.net_weights)):
            net_wts = data_collections.net_weights
        else:
            net_wts = torch.ones(placedb.num_nets, dtype=self.pos[0].dtype, device=device)

        return lut_ff_legalization.LegalizeCLB(
            lutFlopIndices=data_collections.flop_lut_indices,
            nodeNames=placedb.node_names,
            flop2ctrlSet=data_collections.flop2ctrlSetId_map,
            flop_ctrlSet=data_collections.flop_ctrlSets,
            pin2node=data_collections.pin2node_map,
            pin2net=data_collections.pin2net_map,
            flat_net2pin=data_collections.flat_net2pin_map,
            flat_net2pin_start=data_collections.flat_net2pin_start_map,
            flat_node2pin=data_collections.flat_node2pin_map,
            flat_node2pin_start=data_collections.flat_node2pin_start_map,
            node2fence=data_collections.node2fence_region_map,
            pin_types=data_collections.pin_typeIds,
            lut_type=data_collections.lut_type,
            net_wts=net_wts,
            avg_lut_area=avgLUTArea,
            avg_ff_area=avgFFArea,
            inst_areas=inst_areas,
            pin_offset_x=data_collections.lg_pin_offset_x,
            pin_offset_y=data_collections.lg_pin_offset_y,
            site_types=site_types,
            site_xy=data_collections.lg_siteXYs,
            node_size_x=data_collections.node_size_x[:placedb.num_physical_nodes],
            node_size_y=data_collections.node_size_y[:placedb.num_physical_nodes],
            node2outpin=data_collections.node2outpinIdx_map[:placedb.num_physical_nodes],
            net2pincount=data_collections.net2pincount_map,
            node2pincount=data_collections.node2pincount_map,
            spiral_accessor=data_collections.spiral_accessor,
            num_nets=placedb.num_nets,
            num_movable_nodes=placedb.num_movable_nodes,
            num_nodes=placedb.num_physical_nodes,
            num_sites_x=placedb.num_sites_x,
            num_sites_y=placedb.num_sites_y,
            xWirelenWt=placedb.xWirelenWt,
            yWirelenWt=placedb.yWirelenWt,
            nbrDistEnd=placedb.nbrDistEnd,
            num_threads=params.num_threads,
            device=device)

    def build_draw_placement(self, params, placedb):
        """
        @brief plot placement
        @param params parameters
        @param placedb placement database
        """
        return draw_place.DrawPlaceFPGA(placedb)

    def validate(self, placedb, pos, iteration):
        """
        @brief validate placement
        @param placedb placement database
        @param pos locations of cells
        @param iteration optimization step
        """
        pos = torch.from_numpy(pos).to(self.device)
        hpwl = self.op_collections.hpwl_op(pos)
        overflow, max_density = self.op_collections.density_overflow_op(pos)

        return hpwl, overflow, max_density

    def plot(self, params, placedb, iteration, pos):
        """
        @brief plot layout
        @param params parameters
        @param placedb placement database
        @param iteration optimization step
        @param pos locations of cells
        """
        tt = time.time()
        path = "%s/%s" % (params.result_dir, params.design_name())
        figname = "%s/plot/iter%s.png" % (path, '{:04}'.format(iteration))
        os.system("mkdir -p %s" % (os.path.dirname(figname)))
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)
        self.op_collections.draw_place_op(pos, figname)
        logging.info("plotting to %s takes %.3f seconds" %
                     (figname, time.time() - tt))

