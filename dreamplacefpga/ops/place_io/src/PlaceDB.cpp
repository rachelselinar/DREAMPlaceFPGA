/*************************************************************************
    > File Name: PlaceDB.cpp
    > Author: Yibo Lin (DREAMPlace), Rachel Selina Rajarathnam (DREAMPlaceFPGA)
    > Mail: yibolin@utexas.edu
    > Created Time: Mon 14 Mar 2016 09:22:46 PM CDT
    > Updated: Mar 2021
 ************************************************************************/
/**
 * Modifications Copyright(C) 2024 Advanced Micro Devices, Inc. All rights reserved
 */

#include "PlaceDB.h"
#include <numeric>
#include <algorithm>
#include <cmath>
#include "BookshelfWriter.h"
#include "Iterators.h"
#include "utility/src/Msg.h"
//#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

/// default constructor
PlaceDB::PlaceDB() {
  num_movable_nodes = 0;
  original_num_movable_nodes = 0;
  num_fixed_nodes = 0;
  m_numLibCell = 0;
  m_numLUT = 0;
  m_numLUTRAM = 0;
  m_numFF = 0;
  m_numMUX = 0;
  m_numCARRY = 0;
  m_numDSP = 0;
  m_numRAM = 0;
  m_numShape = 0;
}

void PlaceDB::add_bookshelf_node(std::string& name, std::string& type) 
{
    double sqrt0p0625(std::sqrt(0.0625)), sqrt0p125(std::sqrt(0.125)), sqrt0p5(std::sqrt(0.5)), sqrt0p25(std::sqrt(0.25));

    //Updated approach
    if (limbo::iequals(type, "FDRE") || limbo::iequals(type, "FDSE"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      flop_indices.emplace_back(mov_node_names.size());
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(rFFIdx);
      mov_node_size_x.push_back(sqrt0p0625);
      mov_node_size_y.push_back(sqrt0p0625);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(0);
      cluster_lut_type.emplace_back(0);
      m_numFF += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "LUT0") || limbo::iequals(type, "GND") || limbo::iequals(type, "VCC"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(rLutIdx);
      mov_node_size_x.push_back(sqrt0p0625);
      mov_node_size_y.push_back(sqrt0p0625);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(0);
      cluster_lut_type.emplace_back(0);
      m_numLUT += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "LUT1"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(rLutIdx);
      mov_node_size_x.push_back(sqrt0p0625);
      mov_node_size_y.push_back(sqrt0p0625);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(1);
      cluster_lut_type.emplace_back(0);
      m_numLUT += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "LUT2"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(rLutIdx);
      mov_node_size_x.push_back(sqrt0p0625);
      mov_node_size_y.push_back(sqrt0p0625);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(2);
      cluster_lut_type.emplace_back(1);
      m_numLUT += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "LUT3"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(rLutIdx);
      mov_node_size_x.push_back(sqrt0p0625);
      mov_node_size_y.push_back(sqrt0p0625);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(3);
      cluster_lut_type.emplace_back(2);
      m_numLUT += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "LUT4"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(rLutIdx);
      mov_node_size_x.push_back(sqrt0p125);
      mov_node_size_y.push_back(sqrt0p125);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(4);
      cluster_lut_type.emplace_back(3);
      m_numLUT += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "LUT5"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(rLutIdx);
      mov_node_size_x.push_back(sqrt0p125);
      mov_node_size_y.push_back(sqrt0p125);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(5);
      cluster_lut_type.emplace_back(4);
      m_numLUT += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "LUT6"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(rLutIdx);
      mov_node_size_x.push_back(sqrt0p125);
      mov_node_size_y.push_back(sqrt0p125);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(6);
      cluster_lut_type.emplace_back(5);
      m_numLUT += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "LUT6_2"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(rLutIdx);
      mov_node_size_x.push_back(sqrt0p125);
      mov_node_size_y.push_back(sqrt0p125);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(6); //Treating same as LUT6
      cluster_lut_type.emplace_back(5); //Treating same as LUT6
      m_numLUT += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "RAM32M") || limbo::iequals(type, "RAM64M"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(rLutramIdx);
      mov_node_size_x.push_back(sqrt0p5);
      mov_node_size_y.push_back(sqrt0p5);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(0);
      cluster_lut_type.emplace_back(0); 
      m_numLUTRAM += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "CARRY8"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(rCarryIdx);
      mov_node_size_x.push_back(1);
      mov_node_size_y.push_back(1);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(0); 
      cluster_lut_type.emplace_back(0); 
      m_numCARRY += 1;
      ++num_movable_nodes;
    }
    else if (type.find("DSP") != std::string::npos)
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(rDspIdx);
      mov_node_size_x.push_back(1.0);
      mov_node_size_y.push_back(2.5);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(0);
      cluster_lut_type.emplace_back(0);
      m_numDSP += 1;
      ++num_movable_nodes;
    }
    else if (type.find("RAMB") != std::string::npos)
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(rBramIdx);
      mov_node_size_x.push_back(1.0);
      mov_node_size_y.push_back(5.0);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(0);
      cluster_lut_type.emplace_back(0);
      m_numRAM += 1;
      ++num_movable_nodes;
    }
    else if (type.find("BUF") != std::string::npos)
    {
      fixed_node_name2id_map.insert(std::make_pair(name, fixed_node_names.size()));
      fixed_node_names.emplace_back(name);
      fixed_node_types.emplace_back(type);
      fixed_node_x.emplace_back(0.0);
      fixed_node_y.emplace_back(0.0);
      fixed_node_z.emplace_back(0);
      ++num_fixed_nodes;
    }
    else
    {
        dreamplacePrint(kWARN, "Unknown type component found in .nodes file: %s, %s\n",
                name.c_str(), type.c_str());
    }
    std::vector<index_type> temp;
    node2pin_map.emplace_back(temp);
    node2outpinIdx_map.emplace_back(0);
    node2pincount_map.emplace_back(0);
}

void PlaceDB::add_bookshelf_net(BookshelfParser::Net const& n) {
    ////DBG
    //std::cout << "Add net: " << n.net_name << " with " << n.vNetPin.size() << " pins" << std::endl;
    ////DBG
    // check the validity of nets
    // if a node has multiple pins in the net, only one is kept
    std::vector<BookshelfParser::NetPin> vNetPin = n.vNetPin;

    index_type netId(net_names.size());
    net2pincount_map.emplace_back(vNetPin.size());
    net_name2id_map.insert(std::make_pair(n.net_name, netId));
    net_names.emplace_back(n.net_name);

    std::vector<index_type> netPins;
    std::vector<index_type> SourcePins;
    std::vector<index_type> SinkPins;

    if (flat_net2pin_start_map.size() == 0)
    {
        flat_net2pin_start_map.emplace_back(0);
        net2tnet_start_map.emplace_back(0);
    }

    for (unsigned i = 0, ie = vNetPin.size(); i < ie; ++i) 
    {
        BookshelfParser::NetPin const& netPin = vNetPin[i];
        index_type nodeId, pinId(pin_names.size());

        pin_names.emplace_back(netPin.pin_name);
        pin2net_map.emplace_back(netId);
        snkpin2tnet_map.emplace_back(-1);

        string2index_map_type::iterator found = node_name2id_map.find(netPin.node_name);
        std::string nodeType;

        if (found != node_name2id_map.end())
        {
            nodeId = node_name2id_map.at(netPin.node_name);
            pin2nodeType_map.emplace_back(node2fence_region_map[nodeId]);
            pin_offset_x.emplace_back(0.5*mov_node_size_x[nodeId]);
            pin_offset_y.emplace_back(0.5*mov_node_size_y[nodeId]);
            nodeType = mov_node_types[nodeId];
        } else
        {
            string2index_map_type::iterator fnd = fixed_node_name2id_map.find(netPin.node_name);
            if (fnd != fixed_node_name2id_map.end())
            {
                nodeId = fixed_node_name2id_map.at(netPin.node_name);
                pin2nodeType_map.emplace_back(4);
                pin_offset_x.emplace_back(0.5);
                pin_offset_y.emplace_back(0.5);
                nodeType = fixed_node_types[nodeId];
                nodeId += num_movable_nodes;
            } else
            {
                dreamplacePrint(kERROR, "Net %s connects to instance %s pin %s. However instance %s is not specified in .nodes file. FIX\n",
                        n.net_name.c_str(), netPin.node_name.c_str(), netPin.pin_name.c_str(), netPin.node_name.c_str());
            }
        }

        std::string pType("");
        LibCell const& lCell = m_vLibCell.at(m_LibCellName2Index.at(nodeType));
        int pinTypeId(lCell.pinType(netPin.pin_name));

        if (pinTypeId == -1)
        {
            dreamplacePrint(kWARN, "Net %s connects to instance %s pin %s. However pin %s is not listed in .lib as a valid pin for instance type %s. FIX\n",
                    n.net_name.c_str(), netPin.node_name.c_str(), netPin.pin_name.c_str(), netPin.pin_name.c_str(), nodeType.c_str());
        }

        switch(pinTypeId)
        {   
            case 0: //Output
                {
                    // Skip IO pins for timing nets
                    if (pin2nodeType_map[pinId] != 4)
                    {
                        SourcePins.emplace_back(pinId);
                    }
                    break;
                }
            case 1: //Input
                {   
                    // Skip IO pins for timing nets
                    if (pin2nodeType_map[pinId] != 4)
                    {
                        SinkPins.emplace_back(pinId);
                    }
                    break;
                }
            case 2: //CLK
                {   
                    if (pin2nodeType_map[pinId] != 4)
                    {
                        SinkPins.emplace_back(pinId);
                    }
                    pType = "CK";
                    break;
                }
            case 3: //CTRL
                {   
                    if (pin2nodeType_map[pinId] != 4)
                    {
                        SinkPins.emplace_back(pinId);
                    }
                    
                    if (netPin.pin_name.find("CE") != std::string::npos)
                    {
                        pType = "CE";
                    } else
                    {
                        pType = "SR";
                        pinTypeId = 4;
                    } 
                    break;
                }
            default:
                {
                    break;
                }
        }
        pin_types.emplace_back(pType);
        pin_typeIds.emplace_back(pinTypeId);

        ++node2pincount_map[nodeId];
        pin2node_map.emplace_back(nodeId);
        node2pin_map[nodeId].emplace_back(pinId);
        if (pinTypeId == 0) //Output pin
        {
            node2outpinIdx_map[nodeId] = pinId;
        }

        netPins.emplace_back(pinId);
        flat_net2pin_map.emplace_back(pinId);
    }
    flat_net2pin_start_map.emplace_back(flat_net2pin_map.size());
    net2pin_map.emplace_back(netPins);
    ////DBG
    //std::cout << "Successfully added net: " << n.net_name << " with " << n.vNetPin.size() << " pins" << std::endl;
    ////DBG

    for ( unsigned i = 0, ie = SourcePins.size(); i < ie; ++i)
    {
        for ( unsigned j = 0, je = SinkPins.size(); j < je; ++j)
        {   
            if(netPins.size() > 3000){
                break;
            }
            flat_tnet2pin_map.emplace_back(SourcePins[i]);
            flat_tnet2pin_map.emplace_back(SinkPins[j]);
            snkpin2tnet_map[SinkPins[j]] = tnet2net_map.size();
            tnet2net_map.emplace_back(netId);
        }
    }
    net2tnet_start_map.emplace_back(tnet2net_map.size());

}
void PlaceDB::add_interchange_node(std::string& name, std::string& type)
{   
    if (type.find("BUF") != std::string::npos)
    {
        fixed_node_name2id_map.insert(std::make_pair(name, fixed_node_names.size()));
        fixed_node_names.emplace_back(name);
        fixed_node_types.emplace_back(type);
        fixed_node_x.emplace_back(0.0);
        fixed_node_y.emplace_back(0.0);
        fixed_node_z.emplace_back(0);
        ++num_fixed_nodes;
    } else 
    {
        original_node_name2id_map.insert(std::make_pair(name, original_mov_node_names.size()));
        original_mov_node_names.emplace_back(name);
        original_mov_node_types.emplace_back(type);
        original_mov_node_z.emplace_back(0);
        original_node_is_shape_inst.emplace_back(0);
        original_node_cluster_flag.emplace_back(0);
        original_node2node_map.emplace_back(0);
        org_node_x_offset.emplace_back(0.0);
        org_node_y_offset.emplace_back(0.0);
        org_node_z_offset.emplace_back(0.0);
        org_node_pin_offset_x.emplace_back(0.0);
        org_node_pin_offset_y.emplace_back(0.0);
        ++original_num_movable_nodes;
        // std::cout << "Added node: " << name << " with type: " << type << std::endl;
    }  
}

void PlaceDB::update_interchange_nodes(){
    // before parsing .nets file, update placeable nodes through clustering shape nodes

    // Add clustered nodes first
    for (int i = 0; i < m_numShape; i++)
    {
        if (shape2cluster_node_start[i] == -1)
        {
            continue;
        }

        index_type cluster_start_nodeId = shape2cluster_node_start[i];
        std::string name = original_mov_node_names[cluster_start_nodeId];
        std::string type = original_mov_node_types[cluster_start_nodeId];

        std::vector<index_type> shapeNodes = shape2org_node_map[i];
        for (int j = 0; j < shapeNodes.size(); j++)
        {   
            original_node2node_map[shapeNodes[j]] = mov_node_names.size();
        }
        
        if (limbo::iequals(type, "CARRY8"))
        {
            node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
            mov_node_names.emplace_back(name);
            mov_node_types.emplace_back(type);
            node2fence_region_map.emplace_back(rCarryIdx);
            mov_node_size_x.push_back(shape_widths[i]);
            mov_node_size_y.push_back(shape_heights[i]);
            mov_node_x.emplace_back(0.0);
            mov_node_y.emplace_back(0.0);
            mov_node_z.emplace_back(original_mov_node_z[cluster_start_nodeId]);
            lut_type.emplace_back(0);
            cluster_lut_type.emplace_back(0);
            m_numCARRY += 1;
            ++num_movable_nodes;
        } else if (limbo::iequals(type, "RAM32M") || limbo::iequals(type, "RAM64M"))
        {
            node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
            mov_node_names.emplace_back(name);
            mov_node_types.emplace_back(type);
            node2fence_region_map.emplace_back(rLutramIdx);
            mov_node_size_x.push_back(shape_widths[i]);
            mov_node_size_y.push_back(shape_heights[i]);
            mov_node_x.emplace_back(0.0);
            mov_node_y.emplace_back(0.0);
            mov_node_z.emplace_back(original_mov_node_z[cluster_start_nodeId]);
            lut_type.emplace_back(0);
            cluster_lut_type.emplace_back(0);
            m_numLUTRAM += 1;
            ++num_movable_nodes;
        } else
        {
            dreamplacePrint(kWARN, "Unknown type component found in the clustered nodes: %s, %s\n",
                    name.c_str(), type.c_str());
        }
        std::vector<index_type> temp;
        node2pin_map.emplace_back(temp);
        node2outpinIdx_map.emplace_back(0);
        node2pincount_map.emplace_back(0);

    }  
    
    // Add not clustered nodes
    double sqrt0p0625(std::sqrt(0.0625)), sqrt0p125(std::sqrt(0.125)), sqrt0p5(std::sqrt(0.5)), sqrt0p25(std::sqrt(0.25));

    // std::cout << "Original movable nodes num" << original_num_movable_nodes << std::endl;
    // std::cout << "size of original node_names map " << original_mov_node_names.size() << std::endl;

    for (int i = 0; i < original_num_movable_nodes; ++i)
    {   
        std::string name = original_mov_node_names[i];
        std::string type = original_mov_node_types[i];
        
        if (original_node_cluster_flag[i] == 0)
        {   
            //Updated approach
            if (limbo::iequals(type, "FDRE") || limbo::iequals(type, "FDSE"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                flop_indices.emplace_back(mov_node_names.size());
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rFFIdx);
                mov_node_size_x.push_back(sqrt0p0625);
                mov_node_size_y.push_back(sqrt0p0625);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(0);
                cluster_lut_type.emplace_back(0);
                m_numFF += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT0") || limbo::iequals(type, "GND") || limbo::iequals(type, "VCC"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rLutIdx);
                mov_node_size_x.push_back(sqrt0p0625);
                mov_node_size_y.push_back(sqrt0p0625);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(0);
                cluster_lut_type.emplace_back(0);
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "MUXF7"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rMuxIdx);
                mov_node_size_x.push_back(sqrt0p25);
                mov_node_size_y.push_back(sqrt0p25);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(0);
                cluster_lut_type.emplace_back(0);
                m_numMUX += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "MUXF8"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rMuxIdx);
                mov_node_size_x.push_back(sqrt0p5);
                mov_node_size_y.push_back(sqrt0p5);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(0);
                cluster_lut_type.emplace_back(0);
                m_numMUX += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT1"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rLutIdx);
                mov_node_size_x.push_back(sqrt0p0625);
                mov_node_size_y.push_back(sqrt0p0625);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(1);
                cluster_lut_type.emplace_back(0);
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT2"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rLutIdx);
                mov_node_size_x.push_back(sqrt0p0625);
                mov_node_size_y.push_back(sqrt0p0625);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(2);
                cluster_lut_type.emplace_back(1);
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT3"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rLutIdx);
                mov_node_size_x.push_back(sqrt0p0625);
                mov_node_size_y.push_back(sqrt0p0625);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(3);
                cluster_lut_type.emplace_back(2);
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT4"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rLutIdx);
                mov_node_size_x.push_back(sqrt0p125);
                mov_node_size_y.push_back(sqrt0p125);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(4);
                cluster_lut_type.emplace_back(3);
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT5"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rLutIdx);
                mov_node_size_x.push_back(sqrt0p125);
                mov_node_size_y.push_back(sqrt0p125);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(5);
                cluster_lut_type.emplace_back(4);
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT6"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rLutIdx);
                mov_node_size_x.push_back(sqrt0p125);
                mov_node_size_y.push_back(sqrt0p125);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(6);
                cluster_lut_type.emplace_back(5);
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "LUT6_2"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rLutIdx);
                mov_node_size_x.push_back(sqrt0p125);
                mov_node_size_y.push_back(sqrt0p125);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(6); //Treating same as LUT6
                cluster_lut_type.emplace_back(5); 
                m_numLUT += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "RAM32M") || limbo::iequals(type, "RAM64M"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rLutramIdx);
                mov_node_size_x.push_back(sqrt0p5);
                mov_node_size_y.push_back(sqrt0p5);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(0);
                cluster_lut_type.emplace_back(0); 
                m_numLUTRAM += 1;
                ++num_movable_nodes;
            }
            else if (limbo::iequals(type, "CARRY8"))
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rCarryIdx);
                mov_node_size_x.push_back(1);
                mov_node_size_y.push_back(1);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(0); 
                cluster_lut_type.emplace_back(0); 
                m_numCARRY += 1;
                ++num_movable_nodes;
            }
            else if (type.find("DSP") != std::string::npos)
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rDspIdx);
                mov_node_size_x.push_back(1.0);
                mov_node_size_y.push_back(2.5);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(0);
                cluster_lut_type.emplace_back(0);
                m_numDSP += 1;
                ++num_movable_nodes;
            }
            else if (type.find("RAMB") != std::string::npos)
            {
                node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
                original_node2node_map[i] = mov_node_names.size();
                mov_node_names.emplace_back(name);
                mov_node_types.emplace_back(type);
                node2fence_region_map.emplace_back(rBramIdx);
                mov_node_size_x.push_back(1.0);
                mov_node_size_y.push_back(5.0);
                mov_node_x.emplace_back(0.0);
                mov_node_y.emplace_back(0.0);
                mov_node_z.emplace_back(original_mov_node_z[i]);
                lut_type.emplace_back(0);
                cluster_lut_type.emplace_back(0);
                m_numRAM += 1;
                ++num_movable_nodes;
            }
            else
            {
                dreamplacePrint(kWARN, "Unknown type component found in the movable nodes: %s, %s\n",
                        name.c_str(), type.c_str());
            }

            std::vector<index_type> temp;
            node2pin_map.emplace_back(temp);
            node2outpinIdx_map.emplace_back(0);
            node2pincount_map.emplace_back(0);
        } 
    }

    // Add fixed IOs
    for (int i = 0; i < num_fixed_nodes; ++i)
    {
        original_node2node_map.emplace_back(i+num_movable_nodes);
        std::vector<index_type> temp;
        node2pin_map.emplace_back(temp);
        node2outpinIdx_map.emplace_back(0);
        node2pincount_map.emplace_back(0);
    }
}
void PlaceDB::add_interchange_net(BookshelfParser::Net const& n) {
    //// add interchange net with shape support

    std::vector<BookshelfParser::NetPin> vNetPin = n.vNetPin;

    index_type netId(net_names.size());
    net2pincount_map.emplace_back(vNetPin.size());
    net_name2id_map.insert(std::make_pair(n.net_name, netId));
    net_names.emplace_back(n.net_name);

    std::vector<index_type> netPins;
    std::vector<index_type> SourcePins;
    std::vector<index_type> SinkPins;

    if (flat_net2pin_start_map.size() == 0)
    {
        flat_net2pin_start_map.emplace_back(0);
        net2tnet_start_map.emplace_back(0);
    }

    for (unsigned i = 0, ie = vNetPin.size(); i < ie; ++i) 
    {
        BookshelfParser::NetPin const& netPin = vNetPin[i];
        index_type nodeId, pinId(pin_names.size()), org_nodeId;

        pin_names.emplace_back(netPin.pin_name);
        pin2net_map.emplace_back(netId);
        snkpin2tnet_map.emplace_back(-1);

        string2index_map_type::iterator found = node_name2id_map.find(netPin.node_name);
        string2index_map_type::iterator found_org = original_node_name2id_map.find(netPin.node_name);
        std::string nodeType;

        if (found != node_name2id_map.end())
        {
            org_nodeId = original_node_name2id_map.at(netPin.node_name);
            nodeId = node_name2id_map.at(netPin.node_name);
            pin2nodeType_map.emplace_back(node2fence_region_map[nodeId]);
            if (original_node_cluster_flag[org_nodeId] == 1)
            {
                pin_offset_x.emplace_back(org_node_pin_offset_x[org_nodeId]);
                pin_offset_y.emplace_back(org_node_pin_offset_y[org_nodeId]);
            } else
            {
                pin_offset_x.emplace_back(0.5*mov_node_size_x[nodeId]);
                pin_offset_y.emplace_back(0.5*mov_node_size_y[nodeId]);
            }
            nodeType = mov_node_types[nodeId];
        } else if (found_org != original_node_name2id_map.end())
        {
            org_nodeId = original_node_name2id_map.at(netPin.node_name);
            nodeId = original_node2node_map[org_nodeId];
            pin2nodeType_map.emplace_back(node2fence_region_map[nodeId]);
            pin_offset_x.emplace_back(org_node_pin_offset_x[org_nodeId]);
            pin_offset_y.emplace_back(org_node_pin_offset_y[org_nodeId]);
            nodeType = mov_node_types[nodeId];
        } else
        {
            string2index_map_type::iterator fnd = fixed_node_name2id_map.find(netPin.node_name);
            if (fnd != fixed_node_name2id_map.end())
            {
                nodeId = fixed_node_name2id_map.at(netPin.node_name);
                org_nodeId = -1;
                pin2nodeType_map.emplace_back(4);
                pin_offset_x.emplace_back(0.5);
                pin_offset_y.emplace_back(0.5);
                nodeType = fixed_node_types[nodeId];
                nodeId += num_movable_nodes;
            } else
            {
                dreamplacePrint(kERROR, "Net %s connects to instance %s pin %s. However instance %s is not specified in .nodes file. FIX\n",
                        n.net_name.c_str(), netPin.node_name.c_str(), netPin.pin_name.c_str(), netPin.node_name.c_str());
            }
        }

        std::string pType("");
        LibCell const& lCell = m_vLibCell.at(m_LibCellName2Index.at(nodeType));
        int pinTypeId(lCell.pinType(netPin.pin_name));

        if (pinTypeId == -1)
        {
            dreamplacePrint(kWARN, "Net %s connects to instance %s pin %s. However pin %s is not listed in .lib as a valid pin for instance type %s. FIX\n",
                    n.net_name.c_str(), netPin.node_name.c_str(), netPin.pin_name.c_str(), netPin.pin_name.c_str(), nodeType.c_str());
        }

        switch(pinTypeId)
        {   
            case 0: //Output
                {
                    // Skip IO pins for timing nets
                    if (pin2nodeType_map[pinId] != 4)
                    {
                        SourcePins.emplace_back(pinId);
                    }
                    break;
                }
            case 1: //Input
                {   
                    // Skip IO pins for timing nets
                    if (pin2nodeType_map[pinId] != 4)
                    {
                        SinkPins.emplace_back(pinId);
                    }
                    break;
                }
            case 2: //CLK
                {   
                    if (pin2nodeType_map[pinId] != 4)
                    {
                        SinkPins.emplace_back(pinId);
                    }
                    pType = "CK";
                    break;
                }
            case 3: //CTRL
                {   
                    if (pin2nodeType_map[pinId] != 4)
                    {
                        SinkPins.emplace_back(pinId);
                    }
                    
                    if (netPin.pin_name.find("CE") != std::string::npos)
                    {
                        pType = "CE";
                    } else
                    {
                        pType = "SR";
                        pinTypeId = 4;
                    } 
                    break;
                }
            default:
                {
                    break;
                }
        }
        pin_types.emplace_back(pType);
        pin_typeIds.emplace_back(pinTypeId);

        ++node2pincount_map[nodeId];
        pin2node_map.emplace_back(nodeId);
        pin2org_node_map.emplace_back(org_nodeId);
        node2pin_map[nodeId].emplace_back(pinId);
        if (pinTypeId == 0) //Output pin
        {
            node2outpinIdx_map[nodeId] = pinId;
        }

        netPins.emplace_back(pinId);
        flat_net2pin_map.emplace_back(pinId);
    }
    flat_net2pin_start_map.emplace_back(flat_net2pin_map.size());
    net2pin_map.emplace_back(netPins);
    ////DBG
    //std::cout << "Successfully added net: " << n.net_name << " with " << n.vNetPin.size() << " pins" << std::endl;
    ////DBG

    for ( unsigned i = 0, ie = SourcePins.size(); i < ie; ++i)
    {
        for ( unsigned j = 0, je = SinkPins.size(); j < je; ++j)
        {   
            if(netPins.size() > 3000){
                break;
            }
            flat_tnet2pin_map.emplace_back(SourcePins[i]);
            flat_tnet2pin_map.emplace_back(SinkPins[j]);
            snkpin2tnet_map[SinkPins[j]] = tnet2net_map.size();
            tnet2net_map.emplace_back(netId);
        }
    }
    net2tnet_start_map.emplace_back(tnet2net_map.size());

}
void PlaceDB::add_interchange_shape(double height, double width)
{
    shape_heights.emplace_back(height);
    shape_widths.emplace_back(width);
    shape_types.emplace_back(0);
    std::vector<index_type> tempShapeNodes;
    shape2org_node_map.emplace_back(tempShapeNodes);
    shape2cluster_node_start.emplace_back(-1);
    m_numShape += 1;
    numShapeClusterNodesTemp = 0;
}
void PlaceDB::add_org_node_to_shape(std::string const& cellName, std::string const& belName, int dx, int dy)
{   
    // Split the instance name to get the parent name
    std::string name = cellName;
    std::string parent_name = name.substr(0, name.find_last_of("/"));
    string2index_map_type::iterator found = original_node_name2id_map.find(name);
    string2index_map_type::iterator found_parent = original_node_name2id_map.find(parent_name);
    
    index_type nodeId;
    if (found != original_node_name2id_map.end())
    {   
        nodeId = original_node_name2id_map.at(name);
        org_node_x_offset[nodeId] = dx;
        org_node_y_offset[nodeId] = dy;

        hashspace::unordered_map<std::string, int>::iterator found_bel = bel2ZLocation.find(belName);
        if (found_bel != bel2ZLocation.end())
        {
            original_mov_node_z[nodeId] = bel2ZLocation.at(belName);
        }
        shape2org_node_map[m_numShape-1].emplace_back(nodeId);
        original_node_is_shape_inst[nodeId] = 1;
        if (limbo::iequals(original_mov_node_types[nodeId], "CARRY8"))
        {
            shape_types[m_numShape-1] = 1;
            original_node_cluster_flag[nodeId] = 1;
            org_node_pin_offset_x[nodeId] = dx + 0.5 *1;
            org_node_pin_offset_y[nodeId] = dy + 0.5 *1;
            if (numShapeClusterNodesTemp == 0)
            {
                shape2cluster_node_start[m_numShape-1] = nodeId;
            }
            numShapeClusterNodesTemp += 1;
        } else if (limbo::iequals(original_mov_node_types[nodeId], "MUXF7") || limbo::iequals(original_mov_node_types[nodeId], "MUXF8"))
        {
            shape_types[m_numShape-1] = 3;
        } 
    } else if (found_parent != original_node_name2id_map.end())
    {
        nodeId = original_node_name2id_map.at(parent_name);
        // for macro instances, check if it is already added to the shape
        if (original_node_is_shape_inst[nodeId] == 0)
        {
            org_node_x_offset[nodeId] = dx;
            org_node_y_offset[nodeId] = dy;
            hashspace::unordered_map<std::string, int>::iterator found_bel = bel2ZLocation.find(belName);
            if (found_bel != bel2ZLocation.end())
            {
                original_mov_node_z[nodeId] = bel2ZLocation.at(belName);
            }
            shape2org_node_map[m_numShape-1].emplace_back(nodeId);
            original_node_is_shape_inst[nodeId] = 1;
            if (limbo::iequals(original_mov_node_types[nodeId], "RAM32M") || limbo::iequals(original_mov_node_types[nodeId], "RAM64M"))
            {
                shape_types[m_numShape-1] = 2;
                original_node_cluster_flag[nodeId] = 1;
                org_node_pin_offset_x[nodeId] = dx + 0.5 * 0.5;
                org_node_pin_offset_y[nodeId] = dy + 0.5 * 0.5;
                org_node_z_offset[nodeId] = bel2ZLocation.at(belName);
                if (numShapeClusterNodesTemp == 0)
                {
                    shape2cluster_node_start[m_numShape-1] = nodeId;
                }
                numShapeClusterNodesTemp += 1;
            }
        }
    } else
    {
        dreamplacePrint(kWARN, "Instance %s or %s not found in the original nodes list\n",
                name.c_str(), parent_name.c_str());
    }
}
void PlaceDB::resize_sites(int xSize, int ySize)
{
    m_dieArea.set(0, 0, xSize, ySize);
    m_siteDB.resize(xSize, std::vector<index_type>(ySize, 0));
    m_siteNameDB.resize(xSize, std::vector<std::string>(ySize, ""));
}
void PlaceDB::site_info_update(int x, int y, int val)
{
    m_siteDB[x][y] = val;
}
void PlaceDB::add_site_name(int x, int y, std::string const& name)
{
    m_siteNameDB[x][y] = name;
}
void PlaceDB::add_bel_map(std::string const& name, int z)
{
    bel2ZLocation.insert(std::make_pair(name, z));
}
void PlaceDB::resize_clk_regions(int xReg, int yReg)
{
    m_clkRegX = xReg;
    m_clkRegY = yReg;
}
void PlaceDB::add_clk_region(std::string const& name, int xl, int yl, int xh, int yh, int xm, int ym)
{
    clk_region temp;
    temp.xl = xl;
    temp.yl = yl;
    temp.xh = xh;
    temp.yh = yh;
    temp.xm = xm;
    temp.ym = ym;
    m_clkRegionDB.emplace_back(temp);
    m_clkRegions.emplace_back(name);
}
void PlaceDB::add_lib_cell(std::string const& name)
{
  string2index_map_type::iterator found = m_LibCellName2Index.find(name);
  if (found == m_LibCellName2Index.end())  // Ignore if already exists
  {
    m_vLibCell.push_back(LibCell(name));
    LibCell& lCell = m_vLibCell.back();
    //lCell.setName(name);
    lCell.setId(m_vLibCell.size() - 1);
    std::pair<string2index_map_type::iterator, bool> insertRet =
        m_LibCellName2Index.insert(std::make_pair(lCell.name(), lCell.id()));
    dreamplaceAssertMsg(insertRet.second, "failed to insert libCell (%s, %d)",
                        lCell.name().c_str(), lCell.id());

    m_numLibCell = m_vLibCell.size();  // update number of libCells 
  }
  m_libCellTemp = name;
}
void PlaceDB::add_input_pin(std::string& pName)
{
    string2index_map_type::iterator found = m_LibCellName2Index.find(m_libCellTemp);

    if (found != m_LibCellName2Index.end())  
    {
        LibCell& lCell = m_vLibCell.at(m_LibCellName2Index.at(m_libCellTemp));
        lCell.addInputPin(pName);
    } else
    {
        dreamplacePrint(kWARN, "libCell not found in .lib file: %s\n",
                m_libCellTemp.c_str());
    }
}
void PlaceDB::add_output_pin(std::string& pName)
{
    string2index_map_type::iterator found = m_LibCellName2Index.find(m_libCellTemp);

    if (found != m_LibCellName2Index.end())  
    {
        LibCell& lCell = m_vLibCell.at(m_LibCellName2Index.at(m_libCellTemp));
        lCell.addOutputPin(pName);
    } else
    {
        dreamplacePrint(kWARN, "libCell not found in .lib file: %s\n",
                m_libCellTemp.c_str());
    }
}
void PlaceDB::add_clk_pin(std::string& pName)
{
    string2index_map_type::iterator found = m_LibCellName2Index.find(m_libCellTemp);

    if (found != m_LibCellName2Index.end())  
    {
        LibCell& lCell = m_vLibCell.at(m_LibCellName2Index.at(m_libCellTemp));
        lCell.addClkPin(pName);
    } else
    {
        dreamplacePrint(kWARN, "libCell not found in .lib file: %s\n",
                m_libCellTemp.c_str());
    }
}
void PlaceDB::add_ctrl_pin(std::string& pName)
{
    string2index_map_type::iterator found = m_LibCellName2Index.find(m_libCellTemp);

    if (found != m_LibCellName2Index.end())  
    {
        LibCell& lCell = m_vLibCell.at(m_LibCellName2Index.at(m_libCellTemp));
        lCell.addCtrlPin(pName);
    } else
    {
        dreamplacePrint(kWARN, "libCell not found in .lib file: %s\n",
                m_libCellTemp.c_str());
    }
}

void PlaceDB::set_bookshelf_node_pos(std::string const& name, double x, double y, int z)
{
    string2index_map_type::iterator found = fixed_node_name2id_map.find(name);
    //bool fixed(true);

    if (found != fixed_node_name2id_map.end())
    {
        fixed_node_x.at(fixed_node_name2id_map.at(name)) = x;
        fixed_node_y.at(fixed_node_name2id_map.at(name)) = y;
        fixed_node_z.at(fixed_node_name2id_map.at(name)) = z;
    } else
    {
        //string2index_map_type::iterator fnd = mov_node_name2id_map.find(name);
        mov_node_x.at(node_name2id_map.at(name)) = x;
        mov_node_y.at(node_name2id_map.at(name)) = y;
        mov_node_z.at(node_name2id_map.at(name)) = z;
    }

}

void PlaceDB::set_bookshelf_design(std::string& name) {
  m_designName.swap(name);
}
void PlaceDB::bookshelf_end() {
    //  // parsing bookshelf format finishes
    //  // now it is necessary to init data that is not set in bookshelf
    //Flatten node2pin
    flat_node2pin_map.reserve(pin_names.size());
    flat_node2pin_start_map.emplace_back(0);
    for (const auto& sub : node2pin_map)
    {
        flat_node2pin_map.insert(flat_node2pin_map.end(), sub.begin(), sub.end());
        flat_node2pin_start_map.emplace_back(flat_node2pin_map.size());
    }

    for (auto& el : fixed_node_name2id_map)
    {
        el.second += num_movable_nodes;
    }

    node_name2id_map.insert(fixed_node_name2id_map.begin(), fixed_node_name2id_map.end());
}

void PlaceDB::interchange_end(){
    //Flatten shape2org_node_map
    flat_shape2org_node_start_map.emplace_back(0);
    for (const auto& sub : shape2org_node_map)
    {   
        flat_shape2org_node_map.insert(flat_shape2org_node_map.end(), sub.begin(), sub.end());
        flat_shape2org_node_start_map.emplace_back(flat_shape2org_node_map.size());
    }

    //Flatten node2pin
    flat_node2pin_map.reserve(pin_names.size());
    flat_node2pin_start_map.emplace_back(0);
    for (const auto& sub : node2pin_map)
    {
        flat_node2pin_map.insert(flat_node2pin_map.end(), sub.begin(), sub.end());
        flat_node2pin_start_map.emplace_back(flat_node2pin_map.size());
    }

    for (auto& el : fixed_node_name2id_map)
    {
        el.second += num_movable_nodes;
    }

    node_name2id_map.insert(fixed_node_name2id_map.begin(), fixed_node_name2id_map.end());
}

bool PlaceDB::write(std::string const& filename) const {

  return write(filename, NULL, NULL);
}

bool PlaceDB::write(std::string const& filename,
                    float const* x,
                    float const* y,
                    PlaceDB::index_type const* z) const {
  return BookShelfWriter(*this).write(filename, x, y, z);
}

DREAMPLACE_END_NAMESPACE

