##
# @file   timing_graph.py
# @author Zhili Xiong
# @date   Mar 2023
# @brief  Main file implementing the timing graph in python.
#

import os
import math
import sys
import torch
from torch.autograd import Function
from torch import nn
import numpy as np
import logging
import time
import pdb
import igraph as ig
import itertools
from matplotlib import pyplot as plt

class TimingVertex():
    """ 
    @brief create a Timing vertex object
    A timing vertex is always a cell pin 
    """
    
    def __init__(self, pin_id):
        """
        @brief initialization
        @param pin_id the index of a cell pin
        """
        self.pin_id = pin_id
        self.arrival_time = 0 # max
        self.required_time = 0 # min
        self.slack = 0
        self.prev = None
        self.topo_level = 0

        self.is_flop_in = False
        self.is_dsp_ram_in = False
        self.is_flop_out = False

    def set_flop_in(self):
        """ 
        @brief set if this pin is the input of a FF 
        """
        self.is_flop_in = True
        self.is_flop_out = False
    
    def set_flop_out(self):
        """ 
        @brief set if this pin is the output of a FF 
        """
        self.is_flop_in = False
        self.is_flop_out = True

    def set_dsp_ram_in(self):
        """ 
        @brief set if this pin is the input of a DSP or BRAM 
        """
        self.is_dsp_ram_in = True

    def compute_slack(self):
        """
        @brief compute the slack of this timing vertex.
        """
        self.slack = self.required_time - self.arrival_time

    def reset(self):
        """
        @brief reset the arrival time, required_time and slack of a timing vertex
        """
        self.arrival_time = 0
        self.required_time = 0
        self.slack = 0

class TimingEdge():
    """ 
    @brief create a timing edge object
    A timing edge is either a logic element or a connection
    """
    
    def __init__(self, src_node, dst_node, tnet_id):
        """
        @brief initialization.
        @param src_node the souce node timing vertex object
        @param dst_node the destination node timing vertex object
        @param net_id the id of the net that the timing edge is related to
        """
        self.src_node = src_node
        self.dst_node = dst_node
        self.tnet = tnet_id
        self.logic_delay = 0
        self.net_delay = 0
        self.slack = 0
        self.is_high_fanout = False

    def compute_slack(self):
        """
        @brief compute the slack of this timing edge
        """
        self.slack = self.dst_node.required_time - self.src_node.arrival_time - self.logic_delay - self.net_delay


class TimingGraph():
    """ 
    @brief build timing graph for static timing analysis
    """

    def __init__(self, tmodel, placedb, timing_constraint):
        """
        @brief initialization
        @param tmodel the timing model object
        @param placedb the placement database object
        @param timing_constraint the timing constraint
        """
        self.node_names = placedb.node_names 
        self.node_types = placedb.node_types
        self.node2fence_region_map = placedb.node2fence_region_map
        self.node_name2id_map = placedb.node_name2id_map 
        self.node2outpinIdx = placedb.node2outpinIdx_map
        self.flat_node2pin = placedb.flat_node2pin_map
        self.flat_node2pin_start = placedb.flat_node2pin_start_map
        self.pin_names = placedb.pin_names
        self.pin_types = placedb.pin_typeIds
        self.pin2node_map = placedb.pin2node_map
        self.net_names = placedb.net_names
        self.flat_net2pin = placedb.flat_net2pin_map
        self.flat_net2pin_start = placedb.flat_net2pin_start_map
        self.net2pincount_map = placedb.net2pincount_map
        self.tnet2net = placedb.tnet2net_map
        self.flat_tnet2pin = placedb.flat_tnet2pin_map
        self.timing_constraint = timing_constraint
        
        self.num_nets = len(self.net_names)
        self.num_tnets = len(self.tnet2net)
        self.num_pins = len(self.pin2node_map)
        self.num_tnets = len(self.tnet2net)
        self.num_physical_nodes = placedb.num_physical_nodes
        self.num_nodes = placedb.num_nodes

        self.pin_pos = None
        self.route_utilization_map = None
        self.pin_utilization_map = None
        self.route_utilization_thresh_5 = 0
        self.pin_utilization_thresh_5 = 0
        
        # Creating a empty graph using the igraph tool
        self.tgraph = ig.Graph(directed=True)
        self.tmodel = tmodel

        # Safely add timing vertices for pins
        self.pin2vertex = {} # stores {pin_id: (vertex_id, timing_vertex_obj)}
        self.num_vertices = 0

        # Safely add timing edges for timing nets
        self.tnet2edge = {} # stores {tnet_id: timing_edge_obj}
        tt = time.time()
        self.build_logic_edges()
        # logging.info("Adding timing edges for logics takes %.2f seconds" % (time.time() - tt))

        tt = time.time()
        self.build_net_edges()
        # logging.info("Adding timing edges for nets takes %.2f seconds" % (time.time() - tt))

        self.super_Source = None
        self.super_Sink = None
        tt = time.time()
        self.build_super_paths()

        # ordering vertices based on topological sort
        tt = time.time()
        self.levelized_vertices = {} # stores {level: [vertex, ...]} 
        self.max_level = 0
        
        self.ordered_timing_vertices = list(self.tgraph.topological_sorting(mode="OUT"))
        self.reversed_timing_vertices = list(reversed(self.ordered_timing_vertices))

    def build_logic_edges(self):
        """
        @brief build logic delays for combinational elements(LUTs).
        @NOTE: Using the python-igraph, it's better to use the add_vertices()/add_edges() functions instead of the add_vertex()/add_edge() functions 
        to avoid runtime issue.
        """
        logic_edge_list = []
        edge_obj_list = {'obj': []}
        vertex_count = 0
        vertex_obj_list = {'obj': []}

        for node_id in range(self.num_physical_nodes):
            # LUT0 are made-up power nodes for VCC and GND
            if self.node2fence_region_map[node_id] == 0 and self.node_types[node_id] != 'LUT0' and self.node_types[node_id] != 'LUT6_2':
                # Add timing vertices for sink pins 
                sink_pin = self.node2outpinIdx[node_id]
                if sink_pin not in self.pin2vertex:
                    vD = TimingVertex(sink_pin)
                    self.pin2vertex[sink_pin] = (vertex_count, vD)
                    vertex_obj_list['obj'].append(vD)
                    vertex_count += 1
                
                sink_vertex_id, vD = self.pin2vertex[sink_pin]
           
                for i in range(self.flat_node2pin_start[node_id], self.flat_node2pin_start[node_id+1]):
                    pin_id = self.flat_node2pin[i]
                    if pin_id != sink_pin:
                        source_pin = pin_id

                        # Add timing vertices for source pins
                        if source_pin not in self.pin2vertex:
                            vS = TimingVertex(source_pin)
                            self.pin2vertex[source_pin] = (vertex_count, vS)
                            vertex_obj_list['obj'].append(vS)
                            vertex_count += 1

                        source_vertex_id, vS = self.pin2vertex[source_pin]

                        # Add timing edges
                        e = TimingEdge(src_node=vS, dst_node=vD, tnet_id=self.num_tnets)
                        e.logic_delay = self.tmodel.get_logic_delay(key=self.node_types[node_id])
                        logic_edge_list.append((source_vertex_id, sink_vertex_id))
                        edge_obj_list['obj'].append(e)
            
            elif self.node_types[node_id] == 'LUT6_2':
                source_pins = []
                sink_pins = []
                A6_pin_flag = False
                for i in range(self.flat_node2pin_start[node_id], self.flat_node2pin_start[node_id+1]):
                    pin_id = self.flat_node2pin[i]
                    if self.pin_types[pin_id] == 0:
                        sink_pins.append(pin_id)
                    else:
                        source_pins.append(pin_id)

                for source_pin in source_pins:
                    if source_pin not in self.pin2vertex:
                        vS = TimingVertex(source_pin)
                        self.pin2vertex[source_pin] = (vertex_count, vS)
                        vertex_obj_list['obj'].append(vS)
                        vertex_count += 1

                    source_vertex_id, vS = self.pin2vertex[source_pin]
                    if self.pin_names[source_pin] == 'I5':
                        A6_pin_flag = True

                    for sink_pin in sink_pins:
                        if sink_pin not in self.pin2vertex:
                            vD = TimingVertex(sink_pin)
                            self.pin2vertex[sink_pin] = (vertex_count, vD)
                            vertex_obj_list['obj'].append(vD)
                            vertex_count += 1

                        sink_vertex_id, vD = self.pin2vertex[sink_pin]

                        if A6_pin_flag == True and self.pin_names[sink_pin] == 'O5':
                            continue

                        e = TimingEdge(src_node=vS, dst_node=vD, tnet_id=self.num_tnets)
                        e.logic_delay = self.tmodel.get_logic_delay(key=self.node_types[node_id])
                        logic_edge_list.append((source_vertex_id, sink_vertex_id))
                        edge_obj_list['obj'].append(e)

        self.num_vertices += vertex_count
        self.tgraph.add_vertices(vertex_count, vertex_obj_list)
        self.tgraph.add_edges(logic_edge_list, edge_obj_list)

    def build_net_edges(self):
        """
        @brief build edges on the timing graph for timing nets.
        For edges that are purely combinational, the tnet_id is set to be num_tnets.
        """
        net_edge_list = []
        edge_obj_list = {'obj': []}
        vertex_count = 0
        vertex_obj_list = {'obj': []}

        for i in range(self.num_tnets):
            source_pin_id = self.flat_tnet2pin[i*2]
            source_node_id = self.pin2node_map[source_pin_id]
            source_node_type = self.node_types[source_node_id]
            
            sink_pin_id = self.flat_tnet2pin[i*2+1]
            sink_node_id = self.pin2node_map[sink_pin_id]
            sink_node_type = self.node_types[sink_node_id]

            # LUT0 are made-up power nodes for VCC and GND
            if source_node_type != 'LUT0' and sink_node_type != 'LUT0':
                if source_pin_id not in self.pin2vertex:
                    vS = TimingVertex(source_pin_id)
                    self.pin2vertex[source_pin_id] = (self.num_vertices+vertex_count, vS)
                    vertex_obj_list['obj'].append(vS)
                    vertex_count += 1
                
                source_vertex_id, vS = self.pin2vertex[source_pin_id]
                
                if sink_pin_id not in self.pin2vertex:
                    vD = TimingVertex(sink_pin_id)
                    self.pin2vertex[sink_pin_id] = (self.num_vertices+vertex_count, vD)
                    vertex_obj_list['obj'].append(vD)
                    vertex_count += 1
                
                sink_vertex_id, vD = self.pin2vertex[sink_pin_id]

                # Set logic delay values at src pins
                if self.node2fence_region_map[source_node_id] >= 1 and self.node2fence_region_map[source_node_id] < 4:
                    logic_delay = self.tmodel.get_logic_delay(key=source_node_type)
                else:
                    logic_delay = 0
                
                # Set the is_flop_in and is_flop_out for FF pins    
                if self.node2fence_region_map[source_node_id] == 1:
                    vS.set_flop_out()
                if self.node2fence_region_map[sink_node_id] == 1:
                    vD.set_flop_in()
                elif self.node2fence_region_map[sink_node_id] == 2 or self.node2fence_region_map[sink_node_id] == 3:
                    vD.set_dsp_ram_in()
              
                e = TimingEdge(src_node=vS, dst_node=vD, tnet_id=i)
                if self.net2pincount_map[self.tnet2net[i]] > 1000:
                    e.is_high_fanout = True
                self.tnet2edge[i] = e
                e.logic_delay = logic_delay
                net_edge_list.append((source_vertex_id, sink_vertex_id))
                edge_obj_list['obj'].append(e)

        self.num_vertices += vertex_count
        self.tgraph.add_vertices(vertex_count, vertex_obj_list)
        self.tgraph.add_edges(net_edge_list, edge_obj_list)
        
    def build_super_paths(self):
        """
        @brief Connects all the sources and sinks of timing paths to a super_Source and a super_Sink
        """
        sources = []
        sinks = []
        for v in self.tgraph.vs:
            if v.indegree() == 0 and v.outdegree() > 0:
                sources.append(v['obj'])
            elif v['obj'].is_flop_in == True and v.indegree() > 0 and v.outdegree() == 0:
                sinks.append(v['obj'])
            # For BRAM/DSP pins 
            elif v['obj'].is_dsp_ram_in == True and v.indegree() > 0 and v.outdegree() == 0:
                sinks.append(v['obj'])
                
        self.super_Source = TimingVertex(self.num_pins)
        self.super_Sink = TimingVertex(self.num_pins+1)
        self.tgraph.add_vertices(2, {'obj': [self.super_Source, self.super_Sink]})

        edge_list_super_source = []
        edge_obj_list_super_source = {'obj': []}
        edge_list_super_sink = []
        edge_obj_list_super_sink = {'obj': []}

        for v in sources:
            e = TimingEdge(self.super_Source, v, self.num_tnets)
            vertex_id = self.pin2vertex[v.pin_id][0]
            edge_list_super_source.append((self.num_vertices, vertex_id))
            edge_obj_list_super_source['obj'].append(e)

        for v in sinks:
            e = TimingEdge(v, self.super_Sink, self.num_tnets+1)
            vertex_id = self.pin2vertex[v.pin_id][0]
            edge_list_super_sink.append((vertex_id, self.num_vertices+1))
            edge_obj_list_super_sink['obj'].append(e)

        self.tgraph.add_edges(edge_list_super_source, edge_obj_list_super_source)
        self.tgraph.add_edges(edge_list_super_sink, edge_obj_list_super_sink)
    
    def update_net_delays(self, e):
        """
        @brief update net delays of a timing edge
        @param e the timing edge object
        """
        # Exlude the connections to super_Source and super_Sink
        if e.src_node == self.super_Source or e.dst_node == self.super_Sink:
            net_delay = 0
        elif e.tnet >= self.num_tnets:
            net_delay = 0
        
        else:
            src_node = self.pin2node_map[e.src_node.pin_id]
            dst_node = self.pin2node_map[e.dst_node.pin_id]
            src_pin_x = self.pin_pos[e.src_node.pin_id]
            src_pin_y = self.pin_pos[e.src_node.pin_id + self.num_pins]
            dst_pin_x = self.pin_pos[e.dst_node.pin_id]
            dst_pin_y = self.pin_pos[e.dst_node.pin_id + self.num_pins]

            net_delay = self.tmodel.get_net_delay(src_pin_x, src_pin_y, dst_pin_x, dst_pin_y) + self.tmodel.get_congestion_delay(src_pin_x, src_pin_y, dst_pin_x, dst_pin_y, self.route_utilization_map, self.pin_utilization_map, self.route_utilization_thresh_5, self.pin_utilization_thresh_5)
            if e.is_high_fanout == True:
                net_delay = net_delay * 1.5
            e.net_delay = net_delay

    def reset(self):
        """
        @brief reset the timing graph
        """
        for v in self.tgraph.vs:
            v['obj'].reset()

    def compute_arrival_time(self):
        """
        @brief compute arrival time through topological order
        """
        for i in self.ordered_timing_vertices:
            v = self.tgraph.vs[i]
            outEdges = v.out_edges()
            if v.indegree() == 0: 
                v['obj'].arrival_time = 0
                
            for e in outEdges:
                timing_edge = e['obj']
                self.update_net_delays(timing_edge)

                arrival = timing_edge.src_node.arrival_time + timing_edge.logic_delay + timing_edge.net_delay
                if timing_edge.dst_node.arrival_time == 0:
                    timing_edge.dst_node.arrival_time = arrival
                else: 
                    timing_edge.dst_node.arrival_time = max(arrival, timing_edge.dst_node.arrival_time)
                timing_edge.dst_node.prev = timing_edge.src_node

    def compute_required_time(self):
        """
        @brief compute required arrival time through reversed topological order
        """
        inEdges =[]
        for i in self.reversed_timing_vertices:
            v = self.tgraph.vs[i]
            inEdges = v.in_edges()

            if v.outdegree() == 0:
                if v['obj'] == self.super_Sink:
                    v['obj'].required_time = self.timing_constraint

            for e in inEdges:
                timing_edge = e['obj']
                remainingRequiredTime = timing_edge.dst_node.required_time - timing_edge.logic_delay - timing_edge.net_delay
                if timing_edge.src_node.required_time == 0:
                    timing_edge.src_node.required_time = remainingRequiredTime
                else:
                    timing_edge.src_node.required_time = min(remainingRequiredTime, timing_edge.src_node.required_time)

    def compute_slack(self):
        """
        @brief compute the slack for timing vertices and edges
        """
        for v in self.tgraph.vs:
            v['obj'].compute_slack()

        for i in range(self.num_tnets):
            if i in self.tnet2edge:
                self.tnet2edge[i].compute_slack()

    def report_wns_tns(self):
        """
        @brief report the wosrt negative slack value.
        """
        # find the longest path
        worst_slack = 0
        tns = 0
        for e in self.tgraph.vs[self.num_vertices+1].in_edges():
            endpoint = e['obj'].src_node 
            worst_slack = min(worst_slack, endpoint.slack)
            tns += min(0, endpoint.slack)

        wns = min(0, worst_slack)

        return wns, tns
    
    def report_path(self, start, end):
        """
        @brief report delay from start node to end node
        @param start_node the full name of the cellpin, eg. "inst_xxx/x"
        @param end_node the full name of the cellpin
        """

        start_node_name, start_pin = get_node_pin(start)
        end_node_name, end_pin = get_node_pin(end)
        start_node_id = self.node_name2id_map[start_node_name]
        end_node_id = self.node_name2id_map[end_node_name]
        p_start = self.flat_node2pin_start[start_node_id]
        p_end = self.flat_node2pin_start[start_node_id+1]

        for i in range(p_start, p_end):
            pin_id = self.flat_node2pin[i]
            if self.pin_names[pin_id] == start_pin:
                start_pin_id = pin_id
                break

        p_start = self.flat_node2pin_start[end_node_id]
        p_end = self.flat_node2pin_start[end_node_id+1]

        for i in range(p_start, p_end):
            pin_id = self.flat_node2pin[i]
            if self.pin_names[pin_id] == end_pin:
                end_pin_id = pin_id
                break
          
        start_v = self.pin2vertex[start_pin_id][0]
        end_v = self.pin2vertex[end_pin_id][0]

        path = self.tgraph.get_all_simple_paths(start_v, end_v)[0]
        path_delay = 0
        # add up the path delays
        for i in range(len(path)-1):
            edge_id = self.tgraph.get_eid(path[i], path[i+1])
            logic_delay = self.tgraph.es[edge_id]['obj'].logic_delay
            net_delay = self.tgraph.es[edge_id]['obj'].net_delay
            path_delay += logic_delay + net_delay
            print("From %s to %s: logic delay = %d, net delay = %d" % (self.tgraph.vs[path[i]]['obj'].pin_id, self.tgraph.vs[path[i+1]]['obj'].pin_id, logic_delay, net_delay))
        
        return path_delay

    def report_critical_path(self, tnet_weights, device):
        """
        @brief report the critical path with detailed information
        @param tnet_weights the updated tnet_weights
        @param device gpu or cpu
        """
        
        critical_path = self.find_critical_path()
        a, b = itertools.tee(critical_path)
        next(b, None)
        critical_path_pairs = zip(a, b)
        tnet_wts = torch.zeros(self.num_tnets, dtype=tnet_weights.dtype, device=device)
        tnet_wts[:self.num_tnets].data.copy_(tnet_weights)

        combi_delay = 0
        for (i, j) in critical_path_pairs:
            if i>=self.num_vertices or j>=self.num_vertices:
                continue
            
            src_v = self.tgraph.vs[i]
            dst_v = self.tgraph.vs[j]
            src_str = self.node_names[self.pin2node_map[src_v['obj'].pin_id]] + '/' + self.pin_names[src_v['obj'].pin_id]
            dst_str = self.node_names[self.pin2node_map[dst_v['obj'].pin_id]] + '/' + self.pin_names[dst_v['obj'].pin_id]
            edge_id = self.tgraph.get_eid(i, j)
            tnet_id = self.tgraph.es[edge_id]['obj'].tnet
            logic_delay = self.tgraph.es[edge_id]['obj'].logic_delay
            net_delay = self.tgraph.es[edge_id]['obj'].net_delay
            src_pin_pos = (self.pin_pos[src_v['obj'].pin_id], self.pin_pos[src_v['obj'].pin_id+self.num_pins])
            dst_pin_pos = (self.pin_pos[dst_v['obj'].pin_id], self.pin_pos[dst_v['obj'].pin_id+self.num_pins])

            if tnet_id < self.num_tnets:
                net_id = self.tnet2net[tnet_id]
                logging.info("--------  From %s to %s:  --------" % (src_str, dst_str))
                logging.info("net: %s, net-weight: %s" % (self.net_names[net_id], tnet_wts[tnet_id]))
                logging.info("Source pin: (%.3f, %.3f), Sink pin: (%.3f, %.3f)" % (src_pin_pos[0], src_pin_pos[1], dst_pin_pos[0], dst_pin_pos[1]))
                logging.info("Logic delay = %d, Net delay = %d, Total delay = %d" % (logic_delay+combi_delay, net_delay, logic_delay+net_delay+combi_delay))
                combi_delay = 0
            else:
                combi_delay = logic_delay
                continue

    def find_critical_path(self):
        """
        Returns the longest path in a directed acyclic graph (DAG).
        """

        # update edge weights based on delay values
        for e in self.tgraph.es:
            e['weight'] = e['obj'].logic_delay + e['obj'].net_delay

        dist = {}  # stores {v : (length, u)}
        for v in self.ordered_timing_vertices:
            us = []
            for e in self.tgraph.vs[v].in_edges():
                u = e.source
                us.append((dist[u][0] + e['weight'], u))

            # Use the best predecessor if there is one and its distance is
            # non-negative, otherwise terminate.
            maxu = max(us, key=lambda x: x[0]) if us else (0, v)
            dist[v] = maxu if maxu[0] >= 0 else (0, v)

        u = None
        v = max(dist, key=lambda x: dist[x][0])
        path = []
        while u != v:
            path.append(v)
            u = v
            v = dist[v][1]

        path.reverse()
        return path


    def vertex_levelization(self):
        """
        @brief levelize the timing vertices and build a bucket list for each level
        """

        self.levelized_vertices = {} # stores {level: [vertex, ...]}
        self.levelized_vertices[0] = [self.tgraph.vs[self.num_vertices]]
        l = 0

        while l <= self.max_level:
            self.levelized_vertices[l+1] = []

            for v in self.levelized_vertices[l]:
                for u in v.predecessors():
                    v['obj'].topo_level = max(v['obj'].topo_level, u['obj'].topo_level + 1)
                
                for s in v.successors():
                    if v['obj'].topo_level + 1 > s['obj'].topo_level:
                        s['obj'].topo_level = v['obj'].topo_level + 1
                        self.levelized_vertices[l+1].append(s)
                        self.max_level = max(self.max_level, l+1)

            l += 1

def get_node_pin(full_name):
    """ 
    @brief Debugging function given a full name of a tming vertex and return its node_name, pin_name.
    """
    node_name = full_name.split('/')[0]
    pin_name = full_name.split('/')[-1]

    return node_name, pin_name