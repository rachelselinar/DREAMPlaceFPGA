/**
 * @file   PyPlaceDB.h
 * @author Yibo Lin (DREAMPlace), Rachel Selina Rajarathnam (DREAMPlaceFPGA)
 * @date   Mar 2021
 * @brief  Placement database for python 
 */

#ifndef _DREAMPLACE_PLACE_IO_PYPLACEDB_H
#define _DREAMPLACE_PLACE_IO_PYPLACEDB_H

//#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <sstream>
//#include <boost/timer/timer.hpp>
#include "PlaceDB.h"
#include "Iterators.h"
#include "utility/src/torch.h"

DREAMPLACE_BEGIN_NAMESPACE

bool readBookshelf(PlaceDB& db, std::string const& auxPath);

/// database for python 
struct PyPlaceDB
{
    pybind11::list node_names; ///< 1D array, cell name 
    pybind11::list node_size_x; ///< 1D array, cell width  
    pybind11::list node_size_y; ///< 1D array, cell height
    pybind11::list node_types; ///< 1D array, nodeTypes(FPGA)
    pybind11::list flop_indices; ///< 1D array, nodeTypes(FPGA)
    //pybind11::list lut_indices; ///< 1D array, nodeTypes(FPGA)
    //pybind11::list flop_lut_indices; ///< 1D array, nodeTypes(FPGA)
    //pybind11::list dsp_indices; ///< 1D array, nodeTypes(FPGA)
    //pybind11::list ram_indices; ///< 1D array, nodeTypes(FPGA)
    //pybind11::list dsp_ram_indices; ///< 1D array, nodeTypes(FPGA)
    pybind11::list node2fence_region_map; ///< only record fence regions for each cell 

    pybind11::list node_x; ///< 1D array, cell position x 
    pybind11::list node_y; ///< 1D array, cell position y 
    pybind11::list node_z; ///< 1D array, cell position z  (FPGA)
    pybind11::list node2pin_map; ///< array of 1D array, contains pin id of each node 
    pybind11::list flat_node2pin_map; ///< flatten version of node2pin_map 
    pybind11::list flat_node2pin_start_map; ///< starting index of each node in flat_node2pin_map
    pybind11::list node2pincount_map; ///< array of 1D array, number of pins in node
    pybind11::list net2pincount_map; ///< array of 1D array, number of pins in net
    pybind11::list node2outpinIdx_map; ///< array of 1D array, output pin idx of each node
    pybind11::list lut_type; ///< 1D array, nodeTypes(FPGA)
    pybind11::list cluster_lut_type; ///< 1D array, LUT types for clustering
    pybind11::dict node_name2id_map; ///< node name to id map, cell name 
    //pybind11::dict movable_node_name2id_map; ///< node name to id map, cell name 
    //pybind11::dict fixed_node_name2id_map; ///< node name to id map, cell name 
    //pybind11::list fixedNodes; ///< 1D array, nodeTypes(FPGA)
    unsigned int num_terminals; ///< number of terminals, essentially IOs
    unsigned int num_movable_nodes; ///< number of movable nodes
    unsigned int num_physical_nodes; ///< number of movable nodes + terminals (FPGA)
    pybind11::list node_count; ///< 1D array, count of each cell type 

    pybind11::list pin_offset_x; ///< 1D array, pin offset x to its node 
    pybind11::list pin_offset_y; ///< 1D array, pin offset y to its node 
    pybind11::list pin_names; ///< 1D array, pin names (FPGA)
    pybind11::list pin_types; ///< 1D array, pin types (FPGA)
    pybind11::list pin_typeIds; ///< 1D array, pin types (FPGA)
    pybind11::list pin2node_map; ///< 1D array, contain parent node id of each pin 
    pybind11::list pin2net_map; ///< 1D array, contain parent net id of each pin 
    pybind11::list pin2nodeType_map; ///< 1D array, pin to node type

    pybind11::list net_names; ///< net name 
    pybind11::list net2pin_map; ///< array of 1D array, each row stores pin id
    pybind11::list flat_net2pin_map; ///< flatten version of net2pin_map 
    pybind11::list flat_net2pin_start_map; ///< starting index of each net in flat_net2pin_map
    pybind11::dict net_name2id_map; ///< net name to id map
    //pybind11::list net_weights; ///< net weight 

    int num_sites_x; ///< number of sites in horizontal direction (FPGA)
    int num_sites_y; ///< number of sites in vertical direction (FPGA)
    pybind11::list site_type_map; ///< 2D array, site type of each site (FPGA)
    pybind11::list lg_siteXYs; ///< 2D array, site XYs for CLB at center (FPGA)
    //pybind11::list regions; ///< array of 1D array, each region contains rectangles 
    pybind11::list dspSiteXYs; ///< 1D array of DSP sites (FPGA)
    pybind11::list ramSiteXYs; ///< 1D array of RAM sites (FPGA)
    //pybind11::list regionsLimits; ///< array of 1D array, each region contains rectangles 
    pybind11::list flat_region_boxes; ///< flatten version of regions 
    pybind11::list flat_region_boxes_start; ///< starting index of each region in flat_region_boxes
    pybind11::list spiral_accessor; ///< spiral accessor

    pybind11::list ctrlSets; ///< 1D array, FF ctrl set (FPGA)
    pybind11::list flat_ctrlSets; ///< 1D array, FF ctrl set (FPGA)
    //unsigned int num_nodes; ///< number of nodes, including terminals and terminal_NIs 
    //unsigned int width; ///< number of nodes, including terminals and terminal_NIs 
    //unsigned int height; ///< number of nodes, including terminals and terminal_NIs 
    unsigned int spiral_maxVal; ///< maxVal in spiral_accessor
    unsigned int num_routing_grids_x; ///< number of routing grids in x 
    unsigned int num_routing_grids_y; ///< number of routing grids in y 
    int routing_grid_xl; ///< routing grid region may be different from placement region 
    int routing_grid_yl; 
    int routing_grid_xh; 
    int routing_grid_yh;
    pybind11::list tnet2net_map; ///< array of 1D array, each row stores net id
    pybind11::list net2tnet_start_map; ///< starting index of each net in tnet2net_map
    pybind11::list flat_tnet2pin_map; ///< flatten version of tnet2pin_map
    pybind11::list snkpin2tnet_map; ///< array of 1D array, each row stores tnet id
    int xl; 
    int yl; 
    int xh; 
    int yh; 
    int row_height;
    int site_width;

    //pybind11::list node2orig_node_map; ///< due to some fixed nodes may have non-rectangular shapes, we flat the node list; 
    //                                    ///< this map maps the new indices back to the original ones 
    //pybind11::list pin_direct; ///< 1D array, pin direction IO 
    //pybind11::list rows; ///< NumRows x 4 array, stores xl, yl, xh, yh of each row 
    //pybind11::list node_count; ///< Node count based on resource type (FPGA)
    //pybind11::list unit_horizontal_capacities; ///< number of horizontal tracks of layers per unit distance 
    //pybind11::list unit_vertical_capacities; /// number of vertical tracks of layers per unit distance 
    //pybind11::list initial_horizontal_demand_map; ///< initial routing demand from fixed cells, indexed by (layer, grid x, grid y) 
    //pybind11::list initial_vertical_demand_map; ///< initial routing demand from fixed cells, indexed by (layer, grid x, grid y)   
    //pybind11::list binCapMaps; ///< array of 2D array, Bin Capacity map for all resource types (FPGA)
    //pybind11::list fixedDemandMaps; ///< array of 2D array, Bin Capacity map for all resource types (FPGA)
    //double total_space_area; ///< total placeable space area excluding fixed cells. 
    //                        ///< This is not the exact area, because we cannot exclude the overlapping fixed cells within a bin. 
    //int num_movable_pins; 

    PyPlaceDB()
    {
    }

    PyPlaceDB(PlaceDB const& db)
    {
        set(db); 
    }

    void set(PlaceDB const& db);
};

DREAMPLACE_END_NAMESPACE

#endif

