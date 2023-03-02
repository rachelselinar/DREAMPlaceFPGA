import capnp
import capnp.lib.capnp
capnp.remove_import_hook()

import os
import sys 
import time
import gzip 
import enum
import numpy as np 
import logging
import argparse
# import Params
# import dreamplacefpga 
# import dreamplacefpga.ops.place_io.place_io as place_io 
import pdb 
from collections import namedtuple
# from Params import *
# from PlaceDB import *

NO_TRAVERSAL_LIMIT = 2**63 - 1

NESTING_LIMIT = 1024


class IF2bookshelf():
    """ IFParser calss to write out bookshelf files from logic_netlist and device_srouces. 

    logic_netlist - object input, contains all the data of each field in LogicalNetlist.capnp

    bookshelf: design.nodes design.nets design.wts design.pl design.scl design.lib

    """
    def __init__(self, bookshelf_dir, netlist_file):
        """ LogicalNetlist and  DeviceResources"""
        self.bookshelf_dir = bookshelf_dir
        schema_dir = os.path.join(os.path.dirname(__file__), '../thirdparty/RapidWright/interchange/fpga-interchange-schema/interchange')
        self.netlist_obj = LogicalNetlist(schema_dir, netlist_file)
        self.device_obj = DeviceResources(schema_dir, 'xcvu3p-ffvc1517-2-e')

        # for design.lib
        self.lib_cells = ['FDRE', 'LUT6_2', 'LUT6', 'LUT5', 'LUT4', 'LUT3', 'LUT2', 'LUT1', 'CARRY8', 'DSP48E2', 'RAMB36E2', 'BUFGCE', 'IBUF', 'OBUF']
        self.ctrl_pins = ['R', 'CE']
        self.clock_pins = ['C', 'CLK']

        # for design.nodes
        self.const_nodes = ['VCC', 'GND']

        # for design.nets
        # self.const_nets = ['<const1>', '<const0>', 'clk1', 'GND_1', 'GND_2']
        self.const_nets = ['<const1>', '<const0>']
        # self.port_nets = ['ip_', 'op_', 'VCC', 'GND']
        self.port_nets = ['ip_', 'op_']

        # for design.scl
        self.site_types = {
            'SLICEM' : 'SLICE', 
            'SLICEL' : 'SLICE',
            'DSP48E2' : 'DSP', 
            'RAMB36E2' : 'BRAM',
            'RAMBFIFO36' : 'BRAM',
            'HPIOB' : 'IO', 
            'HRIO' : 'IO', 
            'HPIOB_M' : 'IO',
            'HPIOB_S' : 'IO',
            'HPIOB_SNGL' : 'IO',
            'BUFGCE' : 'IO' 
        }

        # for golden reference 
        self.golden_dir = '/home/local/eda15/zhilix/projects/ispd2016_fpga_placement/FPGA02/'
        

    def build_lib(self):
        """ write out design.lib """
        cell_ports = self.netlist_obj.cell_ports

        file_name = os.path.join(self.bookshelf_dir, 'design.lib')

        with open(file_name, 'w') as lib_file:
            lib_file.write(os.linesep)
            for cell_type, port_list in cell_ports.items():
                if cell_type in self.lib_cells:
                    head = 'CELL' + ' ' + cell_type
                    lib_file.write(head + os.linesep)
                    for port_pin in port_list:
                        port_name, port_dir = port_pin
                        line = '  ' + 'PIN' + ' ' + port_name + ' ' + port_dir.name.upper()
                        if port_name in self.ctrl_pins:
                            line = line + ' ' + 'CTRL' 
                        elif port_name in self.clock_pins:
                            line = line + ' ' + 'CLOCK'
                        lib_file.write(line + os.linesep)
                    end = 'END CELL\n'
                    lib_file.write(end + os.linesep)
                else:
                    continue

            # create LUT0 cell for VCC and GND
            head = 'CELL LUT0'
            lib_file.write(head + os.linesep)
            line = '  PIN O OUTPUT'
            lib_file.write(line + os.linesep)
            end = 'END CELL\n'
            lib_file.write(end + os.linesep)

    
    def build_nodes(self):
        """ write out design.nodes """
        nodes = self.netlist_obj.nodes
        file_name = os.path.join(self.bookshelf_dir, 'design.nodes')


        with open(file_name, 'w') as nodes_file:
            for node_name, node_type in nodes.items():
                if node_name[:3] not in self.const_nodes:
                    line = node_name + ' ' + node_type
                    nodes_file.write(line + os.linesep)

            for node_name in self.const_nodes:
                line = node_name + ' LUT0'
                nodes_file.write(line + os.linesep)
        

    def build_nets(self):
        """ write out design.nets """
        nets = self.netlist_obj.nets
        file_name = os.path.join(self.bookshelf_dir, 'design.nets')

        with open(file_name, 'w') as nets_file:
            for net_name, pin_list in nets.items():
                if net_name not in self.const_nets and net_name[:3] not in self.port_nets:
                    len_pin = str(len(pin_list))
                    head = 'net' + ' ' + net_name + ' ' + len_pin
                    nets_file.write(head + os.linesep)
                    for pin in pin_list:
                        cell, port_name = pin
                        if cell not in self.const_nodes:
                            line = '\t' + cell + ' ' + port_name
                        else:
                            line = '\t' + cell + ' O'
                        nets_file.write(line + os.linesep)
                    end = 'endnet'
                    nets_file.write(end + os.linesep)
                else:
                    continue


    def build_aux(self):
        """ write out design.aux """

        file_name = os.path.join(self.bookshelf_dir, 'design.aux')

        with open(file_name, 'w') as aux_file:
            line = 'design : design.nodes design.nets design.wts design.pl design.scl design.lib'
            aux_file.write(line + os.linesep)

    def build_wts(self):
        file_name = os.path.join(self.bookshelf_dir, 'design.wts')

        with open(file_name, 'w') as aux_file:
            line = '#empty'
            aux_file.write(line + os.linesep)

    def build_pl(self):
        file_name = os.path.join(self.bookshelf_dir, 'design.pl')

        with open(file_name, 'w') as aux_file:
            aux_file.write(os.linesep)


    def build_scl(self):
        """ write out design.scl 
        
        TODO: did not differentiate SLICEM and SLICEL.
        
        """
        tile_loc_map = self.device_obj.tile_loc_map
        site_type_map = self.device_obj.site_type_map
        site_loc_map = {}
        site_map = {}

        col_num = {}
        max_col, max_row = (0, 0)
        site_x, site_y = (0, 0)
        slice_map = {}
        for loc, site_list in tile_loc_map.items():
            col, row = loc
            if col > max_col:
                max_col = col
            if row > max_row:
                max_row = row

            for site_name in site_list:
                site_type = site_type_map[site_name]


                if site_type in self.site_types:
                    site_scl = self.site_types[site_type]
                    
                    # for the not continuous slice rows
                    if site_scl == 'SLICE':
                        index_x = site_name.index('X')
                        index_y = site_name.index('Y')
                        site_x = int(site_name[index_x+1: index_y])
                        site_y = int(site_name[index_y+1:])
                        slice_map[col, row] = site_y
                    
                    site_loc_map[col, row] = site_scl
                


        loc_X, loc_Y = (0, 0)
        max_X, max_Y = (0, 0)
        for i in range(0, max_col):
            col_found = False
            loc_Y = 0
            for j in range(max_row, 0, -1):
                if (i, j) in site_loc_map:

                    if site_loc_map[i,j] == 'SLICE':
                        loc_Y = slice_map[i,j] 

                    site_map[loc_X, loc_Y] = site_loc_map[i,j]
                    loc_Y += 1

                    col_found = True
                    if loc_Y > max_Y:
                        max_Y = loc_Y
        
            if col_found == True:
                loc_X += 1
            
            if loc_X > max_X:
                max_X = loc_X
            
        max_X = max_X + 2

        site_dict = {}
        io_col = []
        dsp_col = []
        bram_col = []
        new_io_col =[]

        for i in range(0, max_X):
            site_dict[i] = []


        IO_60_num = int(max_Y/60)
        IO_30_num = int(max_Y/30)

        for j in range(0, IO_60_num):
            site_dict[0].append((j*60, 'IO'))
            site_dict[max_X - 1].append((j*60, 'IO'))
        
        col_offset = 1
        row = 0
        is_io = False
        for loc, site_type in site_map.items():
            X, Y = loc
            # construct SLICE cols
            if site_type == 'SLICE':
                row = Y
                site_dict[X + col_offset].append((row, site_type))
            elif site_type == 'DSP':
                if X in dsp_col:
                    continue
                else:
                    dsp_col.append(X)
            elif site_type == 'BRAM':
                if X in bram_col:
                    continue
                else:
                    bram_col.append(X)
            elif site_type == 'IO':
                if X in io_col:
                    continue
                else:
                    io_col.append(X)

        # construct IO cols
        for col in io_col:
            if col+1 in io_col:
                for i in range(0, IO_30_num):
                    site_dict[col + col_offset].append((i*30, 'IO'))
                for j in range(0, IO_60_num):
                    site_dict[col + 1 + col_offset].append((j*60, 'IO'))
            else:
                continue

        dsp_num = int(max_Y/2.5)
        # construct DSP cols
        for col in dsp_col:
            for i in range(0, dsp_num):
                site_dict[col + col_offset].append((int(i*2.5), 'DSP'))

        bram_num = int(max_Y/5)
        # construct BRAM cols
        for col in bram_col:
            for i in range(0, bram_num):
                site_dict[col + col_offset].append((int(i*5), 'BRAM'))


        file_name = os.path.join(self.bookshelf_dir, 'design.scl')

        # TODO: How to get this part of resources?
        site_resource = {
            'SLICE' : [('LUT', '16'), ('FF', '16'), ('CARRY8', '1')],
            'DSP' : [('DSP48E2', '1')],
            'BRAM' : [('RAMB36E2', '1')],
            'IO' : [('IO', '64')]
        }

        cell_resource = {
            'LUT' : ['LUT1', 'LUT2', 'LUT3', 'LUT4', 'LUT5', 'LUT6', 'LUT6_2', 'LUT0'],
            'FF' : ['FDRE'],
            'CARRY8' : ['CARRY8'],
            'DSP48E2' : ['DSP48E2'],
            'RAMB36E2': ['RAMB36E2'] ,
            'IO' : ['IBUF', 'OBUF', 'BUFGCE']
        }

        # write out to files
        with open(file_name, 'w') as scl_file:
            # First to write 4 types of sites
            for site, resources in site_resource.items():
                site_head = 'SITE ' + site
                scl_file.write(site_head + os.linesep)
                for bel in resources:
                    line = '  ' + bel[0] + ' ' + bel[1] 
                    scl_file.write(line + os.linesep)
                site_end = 'END SITE\n'
                scl_file.write(site_end + os.linesep)

            # Then write cell resources
            resources_head = 'RESOURCES'
            scl_file.write(resources_head + os.linesep)
            for bel, cell_list  in cell_resource.items():
                line = '  ' + bel
                for cell in cell_list:
                    line += ' ' + cell
                scl_file.write(line + os.linesep)
            
            resources_end = 'END RESOURCES\n'
            scl_file.write(resources_end + os.linesep)

            # Finally sitemap
            sitemap_head = 'SITEMAP' + ' ' + str(max_X) + ' ' + str(max_Y)
            scl_file.write(sitemap_head + os.linesep)
            for col_num, site_list in site_dict.items():
                for site in site_list:
                    row_num, sitetype = site
                    line = str(col_num) + ' ' + str(row_num) + ' ' + sitetype
                    scl_file.write(line + os.linesep)
                
            sitemap_end = 'END SITEMAP\n'
            scl_file.write(sitemap_end + os.linesep)   

    

    def verify_bookshelf(self):
        """ verify generated bookshelf files"""

        # verify design.nodes
        golden_nodes_file = os.path.join(self.bookshelf_dir, 'design.nodes')

        golden_nodes = {}
        with open(golden_nodes_file, 'r') as golden:
            for line in golden:
                node_name, node_type = line.split()
                if node_name not in golden_nodes:
                    golden_nodes[node_name] = node_type

        nodes_file = os.path.join(self.bookshelf_dir, 'design.nodes')
        with open(nodes_file, 'r') as nodes:
            for line in nodes:
                node_name, node_type = line.split()
                assert node_name in golden_nodes, "Missing node!" 
                assert golden_nodes[node_name] == node_type, "Node type mismatch!"


        # verify design.nets
        golden_nets_file = os.path.join(self.bookshelf_dir, 'design.nets')

        golden_nets = {}
        with open(golden_nets_file, 'r') as golden:
            for line in golden:
                if line.split()[0] == 'net':
                    golden_nets[line.split()[1]] = line.split()[2]


        nets_file = os.path.join(self.bookshelf_dir, 'design.nets')
        with open(nets_file, 'r') as nets:
            for line in nets:
                if line.split()[0] == 'net':
                    net_name = line.split()[1] 
                    pin_num = line.split()[2]
                    print(net_name)
                    assert net_name in golden_nets, "Misssing net!" 
                    assert golden_nets[net_name] == pin_num, "Pin num mismatch!"

    

class Direction(enum.Enum):
    input = 0
    output = 1
    inout = 2

class LogicalNetlist:
    """
    Parse Logical Netlist file. 
    This is for parameter pin mapping in BRAMs.
    However, all the parameter pins have the same properties and value in ISPD16 benchmarks.
    So the parameter pins have been hard-coded.

    For example,
    para_map['DOA_REG'] = '1'
    para_map['WRITE_WIDTH_A'] = '1'
    para_map['WRITE_WIDTH_B'] = '72'
    para_map['DOB_REG'] = '1'

    """
    def __init__(self, schema_dir, netlist_file):
        """ Read and compile logical netlist for FPGA02-12 benchmarks 
        
        """
        import_path = [os.path.dirname(os.path.dirname(capnp.__file__))]
        # import_path.append(os.path.join(schema_dir, '../../schema'))
        import_path.append('IFsupport')
        self.logical_netlist_capnp = capnp.load(os.path.join(schema_dir, 'LogicalNetlist.capnp'), imports=import_path)

        with open(netlist_file, 'rb') as in_f:
            f_comp = gzip.GzipFile(fileobj = in_f, mode='rb')

            with self.logical_netlist_capnp.Netlist.from_bytes(f_comp.read(), traversal_limit_in_words=NO_TRAVERSAL_LIMIT, nesting_limit=NESTING_LIMIT) as message:
                self.logical_netlist = message 

        self.strs = [s for s in self.logical_netlist.strList]

        self.string_index = {}
        for idx, s in enumerate(self.strs):
            self.string_index[s] = idx

        self.inst_list = self.logical_netlist.instList
        self.cell_decls = self.logical_netlist.cellDecls
        self.port_list = self.logical_netlist.portList
        self.cell_list = self.logical_netlist.cellList

        # database for design.lib
        self.cell_ports = {}
        port_bus2idx = {}
        for cell in self.cell_decls:
            cell_type = self.strs[cell.name]
            if cell_type not in self.cell_ports:
                cell_ports = cell.ports
                self.cell_ports[cell_type] = []
                for port_idx in cell_ports:
                    port_name = self.strs[self.port_list[port_idx].name]
                    port_dir = Direction[self.port_list[port_idx].dir]
                    if self.port_list[port_idx].which() == 'bus':
                        bus_start = self.port_list[port_idx].bus.busStart
                        bus_end = self.port_list[port_idx].bus.busEnd
                        port_bus2idx[port_name] = []
                        for idx in range(bus_start, bus_end - 1, -1):
                            port_bus2idx[port_name].append(idx)
                            port_name_bus = port_name + '[' + str(idx) + ']'
                            self.cell_ports[cell_type].append((port_name_bus, port_dir))
           
                    else:
                        self.cell_ports[cell_type].append((port_name, port_dir))


        # database for design.nodes
        self.nodes = {}
        for cell_inst in self.inst_list:
            cell_name = self.strs[cell_inst.name]
            cell_type = self.strs[self.cell_decls[cell_inst.cell].name]
            if cell_name not in self.nodes:
                self.nodes[cell_name] = cell_type

                
        # database for design.nets
        self.nets = {}
        for cell in self.cell_list:
            cell_type = self.strs[self.cell_decls[cell_inst.cell].name] 
            for inst_idx in cell.insts:
                cell_name = self.strs[self.inst_list[inst_idx].name]

            for net in cell.nets:   
                net_name = self.strs[net.name]
                if net_name not in self.nets:
                    self.nets[net_name] = []
                for port_inst in net.portInsts:
                    port_name = self.strs[self.port_list[port_inst.port].name]
            
                    if port_inst.which() == 'inst':
                        cell = self.strs[self.inst_list[port_inst.inst].name]
                        
                    if port_inst.busIdx.which() == 'idx':
                        port_idx = port_bus2idx[port_name].index(port_inst.busIdx.idx)
                        port_name_idx = port_name + '[' + str(port_idx) + ']'
                        self.nets[net_name].append((cell, port_name_idx))
                    else:
                        self.nets[net_name].append((cell, port_name))

                        
class DeviceResources:
    """DeviceResources class to parse the part's placement resources.

    """
    def __init__(self, schema_dir, part_name):
        """ Read and compile device resources for part assigned by part_name
        
        """
        import_path = [os.path.dirname(os.path.dirname(capnp.__file__))]
        # import_path.append(os.path.join(schema_dir, '../../schema'))
        import_path.append('IFsupport')
        self.device_resources_capnp = capnp.load(os.path.join(schema_dir, 'DeviceResources.capnp'), imports=import_path)

        device_file = os.path.join('IFsupport', part_name)
        
        with open(device_file + '.device', 'rb') as in_f:
            f_comp = gzip.GzipFile(fileobj = in_f, mode='rb')

            with self.device_resources_capnp.Device.from_bytes(f_comp.read(), traversal_limit_in_words=NO_TRAVERSAL_LIMIT, nesting_limit=NESTING_LIMIT) as message:
                self.device_resources = message 
        
        self.strs = [s for s in self.device_resources.strList]

        self.string_index = {}
        for idx, s in enumerate(self.strs):
            self.string_index[s] = idx

        self.site_type_names = []
        self.site_type_name_to_index = {}

                # generate site_type object
        for site_type_index, site_type in enumerate(self.device_resources.siteTypeList):
            site_type_name = self.strs[site_type.name]
            self.site_type_names.append(site_type_name)
            self.site_type_name_to_index[site_type_name] = site_type_index

        tiletype_list = self.device_resources.tileTypeList
        sitetype_list = self.device_resources.siteTypeList
        
     
        self.site_type_map = {}
        self.tile_loc_map = {}

        for tile_idx, tile in enumerate(self.device_resources.tileList):
            tile_name = self.strs[tile.name]
            tile_name_index = self.string_index[tile_name]
            tile_row = tile.row
            tile_col = tile.col
            self.tile_loc_map[tile_col, tile_row] = []

            for site_idx, site in enumerate(tile.sites):
                site_in_tile = self.strs[site.name]
                self.tile_loc_map[tile_col, tile_row].append(site_in_tile)

                tile_type_site_type_index = site.type
                site_types = tiletype_list[tile.type].siteTypes  
                site_type_index = site_types[site.type].primaryType
                site_type_name = self.strs[sitetype_list[site_type_index].name]

                self.site_type_map[site_in_tile] = site_type_name


if __name__ == "__main__":
    """
    brief main function to test the IFParser: from logicalnetlist to bookshelf files.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--netlist")
    args = parser.parse_args()

    design_name = os.path.basename(args.netlist).replace(".netlist", "").replace(".NETLIST", "")

    dump_bookshelf = 'benchmarks/IF2bookshelf'
    bookshelf_dir = os.path.join(dump_bookshelf, design_name)
    
    if os.path.exists(bookshelf_dir) == False:
        os.mkdir(bookshelf_dir)
        
    if_parser = IF2bookshelf(bookshelf_dir, args.netlist)
    if_parser.build_lib()
    if_parser.build_nodes()
    if_parser.build_nets()
    if_parser.build_aux()
    if_parser.build_scl()
    if_parser.build_wts()
    if_parser.build_pl()





