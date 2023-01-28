/*************************************************************************
    > File Name: PlaceDB.cpp
    > Author: Yibo Lin (DREAMPlace), Rachel Selina Rajarathnam (DREAMPlaceFPGA)
    > Mail: yibolin@utexas.edu
    > Created Time: Mon 14 Mar 2016 09:22:46 PM CDT
    > Updated: Mar 2021
 ************************************************************************/

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
  num_fixed_nodes = 0;
  m_numLibCell = 0;
  m_numLUT = 0;
  m_numFF = 0;
  m_numDSP = 0;
  m_numRAM = 0;
}

void PlaceDB::add_bookshelf_node(std::string& name, std::string& type) 
{
    double sqrt0p0625(std::sqrt(0.0625)), sqrt0p125(std::sqrt(0.125));

    //Updated approach
    if (limbo::iequals(type, "FDRE"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      flop_indices.emplace_back(mov_node_names.size());
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(1);
      mov_node_size_x.push_back(sqrt0p0625);
      mov_node_size_y.push_back(sqrt0p0625);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(0);
      m_numFF += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "LUT2"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(0);
      mov_node_size_x.push_back(sqrt0p0625);
      mov_node_size_y.push_back(sqrt0p0625);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(1);
      m_numLUT += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "LUT3"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(0);
      mov_node_size_x.push_back(sqrt0p0625);
      mov_node_size_y.push_back(sqrt0p0625);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(2);
      m_numLUT += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "LUT4"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(0);
      mov_node_size_x.push_back(sqrt0p125);
      mov_node_size_y.push_back(sqrt0p125);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(3);
      m_numLUT += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "LUT5"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(0);
      mov_node_size_x.push_back(sqrt0p125);
      mov_node_size_y.push_back(sqrt0p125);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(4);
      m_numLUT += 1;
      ++num_movable_nodes;
    }
    else if (limbo::iequals(type, "LUT6"))
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(0);
      mov_node_size_x.push_back(sqrt0p125);
      mov_node_size_y.push_back(sqrt0p125);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(5);
      m_numLUT += 1;
      ++num_movable_nodes;
    }
    else if (type.find("DSP") != std::string::npos)
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(2);
      mov_node_size_x.push_back(1.0);
      mov_node_size_y.push_back(2.5);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(0);
      m_numDSP += 1;
      ++num_movable_nodes;
    }
    else if (type.find("RAM") != std::string::npos)
    {
      node_name2id_map.insert(std::make_pair(name, mov_node_names.size()));
      mov_node_names.emplace_back(name);
      mov_node_types.emplace_back(type);
      node2fence_region_map.emplace_back(3);
      mov_node_size_x.push_back(1.0);
      mov_node_size_y.push_back(5.0);
      mov_node_x.emplace_back(0.0);
      mov_node_y.emplace_back(0.0);
      mov_node_z.emplace_back(0);
      lut_type.emplace_back(0);
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
    // check the validity of nets
    // if a node has multiple pins in the net, only one is kept
    std::vector<BookshelfParser::NetPin> vNetPin = n.vNetPin;

    index_type netId(net_names.size());
    net2pincount_map.emplace_back(vNetPin.size());
    net_name2id_map.insert(std::make_pair(n.net_name, netId));
    net_names.emplace_back(n.net_name);

    std::vector<index_type> netPins;
    if (flat_net2pin_start_map.size() == 0)
    {
        flat_net2pin_start_map.emplace_back(0);
    }

    for (unsigned i = 0, ie = vNetPin.size(); i < ie; ++i) 
    {
        BookshelfParser::NetPin const& netPin = vNetPin[i];
        index_type nodeId, pinId(pin_names.size());

        pin_names.emplace_back(netPin.pin_name);
        pin2net_map.emplace_back(netId);

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
            }
        }

        std::string pType("");
        LibCell const& lCell = m_vLibCell.at(m_LibCellName2Index.at(nodeType));
        index_type pinTypeId(lCell.pinType(netPin.pin_name));

        switch(pinTypeId)
        {
            case 2: //CLK
                {
                    pType = "CK";
                    break;
                }
            case 3: //CTRL
                {
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
}
void PlaceDB::resize_sites(int xSize, int ySize)
{
    m_dieArea.set(0, 0, xSize, ySize);
    m_siteDB.resize(xSize, std::vector<index_type>(ySize, 0));
}
void PlaceDB::site_info_update(int x, int y, int val)
{
    m_siteDB[x][y] = val;
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

