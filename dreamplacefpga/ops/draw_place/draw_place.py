##
# @file   draw_place.py
# @author Yibo Lin (DREAMPlace), Rachel Selina Rajarathnam (DREAMPlaceFPGA)
# @date   Jan 2021
# @brief  Plot placement to an image 
#

import os 
import sys 
import torch 
from torch.autograd import Function

import dreamplacefpga.ops.draw_place.draw_place_cpp as draw_place_cpp
import dreamplacefpga.ops.draw_place.PlaceDrawer as PlaceDrawer 
import pdb
import numpy as np

class DrawPlaceFunction(Function):
    @staticmethod
    def forward(
            pos, 
            node_size_x, node_size_y, 
            pin_offset_x, pin_offset_y, 
            pin2node_map, 
            xl, yl, xh, yh, 
            site_width, row_height, 
            bin_size_x, bin_size_y, 
            num_movable_nodes, num_filler_nodes, 
            filename
            ):
        ret = draw_place_cpp.forward(
                pos, 
                node_size_x, node_size_y, 
                pin_offset_x, pin_offset_y, 
                pin2node_map, 
                xl, yl, xh, yh, 
                site_width, row_height, 
                bin_size_x, bin_size_y, 
                num_movable_nodes, num_filler_nodes, 
                filename
                )
        # if C/C++ API failed, try with python implementation 
        if not filename.endswith(".gds") and not ret:
            ret = PlaceDrawer.PlaceDrawer.forward(
                    pos, 
                    node_size_x, node_size_y, 
                    pin_offset_x, pin_offset_y, 
                    pin2node_map, 
                    xl, yl, xh, yh, 
                    site_width, row_height, 
                    bin_size_x, bin_size_y, 
                    num_movable_nodes, num_filler_nodes, 
                    filename
                    )
        return ret 

class DrawPlace(object):
    """ 
    @brief Draw placement
    """
    def __init__(self, placedb):
        """
        @brief initialization 
        """
        self.node_size_x = torch.from_numpy(placedb.node_size_x).float()
        self.node_size_y = torch.from_numpy(placedb.node_size_y).float()
        self.pin_offset_x = torch.FloatTensor(placedb.pin_offset_x).float()
        self.pin_offset_y = torch.FloatTensor(placedb.pin_offset_y).float()
        self.pin2node_map = torch.from_numpy(placedb.pin2node_map)
        self.xl = placedb.xl 
        self.yl = placedb.yl 
        self.xh = placedb.xh 
        self.yh = placedb.yh 
        self.site_width = placedb.width
        self.row_height = placedb.height 
        self.bin_size_x = placedb.bin_size_x 
        self.bin_size_y = placedb.bin_size_y
        self.num_movable_nodes = placedb.num_movable_nodes
        self.num_filler_nodes = placedb.num_filler_nodes

    def forward(self, pos, filename): 
        """ 
        @param pos cell locations, array of x locations and then y locations 
        @param filename suffix specifies the format 
        """
        return DrawPlaceFunction.forward(
                pos, 
                self.node_size_x, 
                self.node_size_y, 
                self.pin_offset_x, 
                self.pin_offset_y, 
                self.pin2node_map, 
                self.xl, 
                self.yl, 
                self.xh, 
                self.yh, 
                self.site_width, 
                self.row_height, 
                self.bin_size_x, 
                self.bin_size_y, 
                self.num_movable_nodes, 
                self.num_filler_nodes, 
                filename
                )

    def __call__(self, pos, filename):
        """
        @brief top API 
        @param pos cell locations, array of x locations and then y locations 
        @param filename suffix specifies the format 
        """
        return self.forward(pos, filename)

# FPGA version - Added by Rachel
class DrawPlaceFunctionFPGA(Function):
    @staticmethod
    def forward(
            pos, 
            node_size_x, node_size_y, 
            pin_offset_x, pin_offset_y, 
            pin2node_map, 
            xl, yl, xh, yh, 
            site_width, row_height, 
            bin_size_x, bin_size_y, 
            num_movable_nodes, num_filler_nodes, 
            node2fence_region_map,
            filename
            ):
        ret = draw_place_cpp.fpga(
                pos, 
                node_size_x, node_size_y, 
                pin_offset_x, pin_offset_y, 
                pin2node_map, 
                xl, yl, xh, yh, 
                site_width, row_height, 
                bin_size_x, bin_size_y, 
                num_movable_nodes, num_filler_nodes, 
                node2fence_region_map,
                filename
                )
        # if C/C++ API failed, try with python implementation 
        if not filename.endswith(".gds") and not ret:
            ret = PlaceDrawer.PlaceDrawer.forward(
                    pos, 
                    node_size_x, node_size_y, 
                    pin_offset_x, pin_offset_y, 
                    pin2node_map, 
                    xl, yl, xh, yh, 
                    site_width, row_height, 
                    bin_size_x, bin_size_y, 
                    num_movable_nodes, num_filler_nodes, 
                    filename
                    )
        return ret 

class DrawPlaceFPGA(object):
    """ 
    @brief Draw placement
    """
    def __init__(self, placedb):
        """
        @brief initialization 
        """
        self.node_size_x = torch.from_numpy(placedb.node_size_x).float()
        self.node_size_y = torch.from_numpy(placedb.node_size_y).float()
        self.pin_offset_x = torch.FloatTensor(placedb.pin_offset_x).float()
        self.pin_offset_y = torch.FloatTensor(placedb.pin_offset_y).float()
        self.pin2node_map = torch.from_numpy(placedb.pin2node_map)
        self.xl = placedb.xl 
        self.yl = placedb.yl 
        self.xh = placedb.xh 
        self.yh = placedb.yh 
        self.site_width = placedb.width
        self.row_height = placedb.height 
        self.bin_size_x = placedb.bin_size_x 
        self.bin_size_y = placedb.bin_size_y
        self.num_movable_nodes = placedb.num_movable_nodes
        self.num_filler_nodes = placedb.num_filler_nodes
        self.node2fence_region_map = torch.from_numpy(placedb.node2fence_region_map)
        self.fmask = self.node2fence_region_map == 4
        self.node_x = torch.from_numpy(placedb.node_x).float()
        self.node_y = torch.from_numpy(placedb.node_y).float()

    def forward(self, pos, filename): 
        """ 
        @param pos cell locations, array of x locations and then y locations 
        @param filename suffix specifies the format 
        """
        fillers = torch.tensor(np.zeros(self.num_filler_nodes)).bool()
        fmask = torch.cat((self.fmask,fillers,self.fmask, fillers),0)
        allLoc = torch.cat((self.node_x, fillers.float(), self.node_y, fillers.float()),0)
        omask = ~fmask
        newpos = pos*omask.float() + allLoc*fmask.float() 

        return DrawPlaceFunctionFPGA.forward(
                newpos, 
                self.node_size_x, 
                self.node_size_y, 
                self.pin_offset_x, 
                self.pin_offset_y, 
                self.pin2node_map, 
                self.xl, 
                self.yl, 
                self.xh, 
                self.yh, 
                self.site_width, 
                self.row_height, 
                self.bin_size_x, 
                self.bin_size_y, 
                self.num_movable_nodes, 
                self.num_filler_nodes, 
                self.node2fence_region_map,
                filename
                )

    def __call__(self, pos, filename):
        """
        @brief top API 
        @param pos cell locations, array of x locations and then y locations 
        @param filename suffix specifies the format 
        """
        return self.forward(pos, filename)
