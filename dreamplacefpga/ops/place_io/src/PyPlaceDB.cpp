/**
 * @file   PyPlaceDB.cpp
 * @author Yibo Lin (DREAMPlace), Rachel Selina Rajarathnam (DREAMPlaceFPGA)
 * @date   Mar 2021
 * @brief  Placement database for python 
 */
/**
 * Modifications Copyright(C) 2024 Advanced Micro Devices, Inc. All rights reserved
 */

#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include "PyPlaceDB.h"
#include <boost/polygon/polygon.hpp>

DREAMPLACE_BEGIN_NAMESPACE

bool readBookshelf(PlaceDB& db, std::string const& auxPath)
{
    // read bookshelf 
    if (!auxPath.empty())
    {
        std::string const& filename = auxPath;
        dreamplacePrint(kINFO, "reading %s\n", filename.c_str());
        bool flag = BookshelfParser::read(db, filename);
        if (!flag)
        {
            dreamplacePrint(kERROR, "Bookshelf file parsing failed: %s\n", filename.c_str());
            return false;
        }
    }
    else dreamplacePrint(kWARN, "no Bookshelf file specified\n");

    return true;
}

bool readInterchange(PlaceDB& db, std::string const& deviceFile, std::string const& netlistFile)
{   
    if (!deviceFile.empty() & !netlistFile.empty())
    {
        bool flag = DREAMPLACE_NAMESPACE::readDeviceNetlist(db, deviceFile, netlistFile);

        if (!flag)
        {
            dreamplacePrint(kERROR, "Interchange file parsing failed\n");
            return false;
        }
    }
    else dreamplacePrint(kWARN, "Missing Interchange file\n");

    return true;
}

void PyPlaceDB::set(PlaceDB const& db) 
{

    num_terminals = db.numFixed(); //IOs
    num_movable_nodes = db.numMovable();  // Movable cells
    num_physical_nodes = num_terminals + num_movable_nodes;

    node_count.append(db.numLUT());
    node_count.append(db.numLUTRAM());
    node_count.append(db.numFF());
    node_count.append(db.numMUX());
    node_count.append(db.numCARRY());
    node_count.append(db.numDSP());
    node_count.append(db.numRAM());
    node_count.append(num_terminals);

    std::vector<double> fixed_node_size(num_terminals, 1.0); 
    std::vector<PlaceDB::index_type> fixed_node2fence_region_map(num_terminals, 7); 
    std::vector<PlaceDB::index_type> fixed_lut_type(num_terminals, 0); 
    //Node Info
    node_names = pybind11::cast(std::move(db.movNodeNames())) + pybind11::cast(std::move(db.fixedNodeNames()));
    node_types = pybind11::cast(std::move(db.movNodeTypes())) + pybind11::cast(std::move(db.fixedNodeTypes()));
    original_node_names = pybind11::cast(std::move(db.originalMovNodeNames())) + pybind11::cast(std::move(db.fixedNodeNames()));
    original_node_types = pybind11::cast(std::move(db.originalMovNodeTypes())) + pybind11::cast(std::move(db.fixedNodeTypes()));
    node_size_x = pybind11::cast(std::move(db.movNodeXSizes())) + pybind11::cast(std::move(fixed_node_size));
    node_size_y = pybind11::cast(std::move(db.movNodeYSizes())) + pybind11::cast(std::move(fixed_node_size));
    node2fence_region_map = pybind11::cast(std::move(db.node2FenceRegionMap())) + pybind11::cast(std::move(fixed_node2fence_region_map));
    node_x = pybind11::cast(std::move(db.movNodeXLocs())) + pybind11::cast(std::move(db.fixedNodeXLocs()));
    node_y = pybind11::cast(std::move(db.movNodeYLocs())) + pybind11::cast(std::move(db.fixedNodeYLocs()));
    node_z = pybind11::cast(std::move(db.movNodeZLocs())) + pybind11::cast(std::move(db.fixedNodeZLocs()));
    flop_indices = pybind11::cast(std::move(db.flopIndices()));
    lut_type = pybind11::cast(std::move(db.lutTypes())) + pybind11::cast(std::move(fixed_lut_type));
    cluster_lut_type = pybind11::cast(std::move(db.clusterlutTypes())) + pybind11::cast(std::move(fixed_lut_type));
    node2outpinIdx_map = pybind11::cast(std::move(db.node2OutPinId()));
    node2pincount_map = pybind11::cast(std::move(db.node2PinCount()));
    node2pin_map = pybind11::cast(std::move(db.node2PinMap()));
    node_name2id_map = pybind11::cast(std::move(db.nodeName2Index()));
    original_node_name2id_map = pybind11::cast(std::move(db.originalNodeName2Index()));
    original_node2node_map = pybind11::cast(std::move(db.originalNode2NodeMap()));
    org_node_x_offset = pybind11::cast(std::move(db.orgNodeXOffset()));
    org_node_y_offset = pybind11::cast(std::move(db.orgNodeYOffset()));
    org_node_z_offset = pybind11::cast(std::move(db.orgNodeZOffset()));
    //movable_node_name2id_map = pybind11::cast(std::move(db.movNodeName2Index()));
    //fixed_node_name2id_map = pybind11::cast(std::move(db.fixedNodeName2Index()));
    flat_node2pin_map = pybind11::cast(std::move(db.flatNode2PinMap()));
    flat_node2pin_start_map = pybind11::cast(std::move(db.flatNode2PinStartMap()));
    net_names = pybind11::cast(std::move(db.netNames()));
    net2pincount_map = pybind11::cast(std::move(db.net2PinCount()));
    net2pin_map = pybind11::cast(std::move(db.net2PinMap()));
    flat_net2pin_map = pybind11::cast(std::move(db.flatNet2PinMap()));
    flat_net2pin_start_map = pybind11::cast(std::move(db.flatNet2PinStartMap()));
    net_name2id_map = pybind11::cast(std::move(db.netName2Index()));

    pin_names = pybind11::cast(std::move(db.pinNames()));
    pin_types = pybind11::cast(std::move(db.pinTypes()));
    pin_typeIds = pybind11::cast(std::move(db.pinTypeIds()));
    pin_offset_x = pybind11::cast(std::move(db.pinOffsetX()));
    pin_offset_y = pybind11::cast(std::move(db.pinOffsetY()));
    pin2net_map = pybind11::cast(std::move(db.pin2NetMap()));
    pin2node_map = pybind11::cast(std::move(db.pin2NodeMap()));
    pin2org_node_map = pybind11::cast(std::move(db.pin2OrgNodeMap()));
    pin2nodeType_map = pybind11::cast(std::move(db.pin2NodeTypeMap()));

    tnet2net_map = pybind11::cast(std::move(db.tnet2NetMap()));
    net2tnet_start_map = pybind11::cast(std::move(db.net2TNetStartMap()));
    flat_tnet2pin_map = pybind11::cast(std::move(db.flatTNet2PinMap()));
    snkpin2tnet_map = pybind11::cast(std::move(db.snkPin2TNetMap()));

    shape_heights = pybind11::cast(std::move(db.shapeHeights()));
    shape_widths = pybind11::cast(std::move(db.shapeWidths()));
    shape_types = pybind11::cast(std::move(db.shapeTypes()));
    shape2org_node_map = pybind11::cast(std::move(db.shape2OrgNodeMap()));
    flat_shape2org_node_map = pybind11::cast(std::move(db.flatShape2OrgNodeMap()));
    flat_shape2org_node_start_map = pybind11::cast(std::move(db.flatShape2OrgNodeStartMap()));
    shape2cluster_node_start = pybind11::cast(std::move(db.shape2ClusterNodeStart()));
    original_node_is_shape_inst = pybind11::cast(std::move(db.originalNodeIsShapeInst()));

    //num_terminals = db.numFixed(); //IOs
    //num_movable_nodes = db.numMovable();  // Movable cells
    //num_physical_nodes = num_terminals + num_movable_nodes;

    //node_count.append(db.numLUT());
    //node_count.append(db.numFF());
    //node_count.append(db.numDSP());
    //node_count.append(db.numRAM());
    //node_count.append(db.numFixed());

    //CtrlSets
    std::unordered_map<PlaceDB::index_type, PlaceDB::index_type> ceMapping;
    std::unordered_map<PlaceDB::index_type, std::unordered_map<PlaceDB::index_type, PlaceDB::index_type> > cksrMapping;
    PlaceDB::index_type numCKSR(0), numCE(0);

    for (unsigned int idx = 0; idx < flop_indices.size(); ++idx)
    {
        PlaceDB::index_type fIdx = flop_indices[idx].cast<PlaceDB::index_type>();
        //Node const& node = db.node(fIdx); 

        int ck(-1), sr(-1), ce(-1), cksrId(-1), ceId(-1);

        //for (auto pin_id : node.pinIdArray())
        for (unsigned int pIdx = 0; pIdx < db.node2PinCnt(fIdx); ++pIdx)
        {
            PlaceDB::index_type pin_id = db.node2PinIdx(fIdx, pIdx);

            switch(pin_typeIds[pin_id].cast<PlaceDB::index_type>())
            {
                case 2:
                {
                    ck = pin2net_map[pin_id].cast<PlaceDB::index_type>();
                    break;
                }
                case 3:
                {
                    ce = pin2net_map[pin_id].cast<PlaceDB::index_type>();
                    break;
                }
                case 4:
                {
                    sr = pin2net_map[pin_id].cast<PlaceDB::index_type>();
                    break;
                }
                default:
                {
                    break;
                }
            }
        }

        auto ckIt = cksrMapping.find(ck);

        if (ckIt == cksrMapping.end())
        {
            cksrId = numCKSR;
            cksrMapping[ck][sr] = numCKSR++;
        } else
        {
            auto &srMap = ckIt->second;
            auto srIt = srMap.find(sr);
            if (srIt == srMap.end())
            {
                cksrId = numCKSR;
                srMap[sr] = numCKSR++;
            } else
            {
                cksrId = srIt->second;
            }
        }

        auto ceIt = ceMapping.find(ce);
        if (ceIt == ceMapping.end())
        {
            ceId = numCE;
            ceMapping[ce] = numCE++;
        } else
        {
            ceId = ceIt->second;
        }
        ctrlSets.append(std::make_tuple(fIdx, cksrId, ceId));
        flat_ctrlSets.append(fIdx);
        flat_ctrlSets.append(cksrId);
        flat_ctrlSets.append(ceId);
    }

    //SiteInfo
    // pybind11::list region0, region2, region3, region4, region5, region6;
    std::vector<std::unordered_set<PlaceDB::index_type> > updCols;
    std::unordered_set<PlaceDB::index_type> colSet0; //LUTL
    std::unordered_set<PlaceDB::index_type> colSet1; //LUTM
    std::unordered_set<PlaceDB::index_type> colSet2; //FF   
    std::unordered_set<PlaceDB::index_type> colSet3; //MUX
    std::unordered_set<PlaceDB::index_type> colSet4; //CARRY
    std::unordered_set<PlaceDB::index_type> colSet5; //DSP
    std::unordered_set<PlaceDB::index_type> colSet6; //BRAM
    std::unordered_set<PlaceDB::index_type> colSet7; //IO

    for (unsigned int i = 0, ie = db.siteRows(); i < ie; ++i)
    {
        pybind11::list rowVals, rowNames, lg_rowXY;
        for (unsigned int j = 0, je = db.siteCols(); j < je; ++j)
        {
            pybind11::list siteXY, lg_Site;

            if (db.siteVal(i,j) == 1 || db.siteVal(i,j) == 2)
            {
                lg_Site.append(i+0.5);
                lg_Site.append(j+0.5);
            } else
            {
                lg_Site.append(i);
                lg_Site.append(j);
            }
            lg_rowXY.append(lg_Site);
            //std::cout << "Site value at (" << i << ", " << j << ") is " << db.siteVal(i,j) << std::endl;
            switch(db.siteVal(i,j))
            {
                case 1: //SLICEL: LUTL/FF/CARRY/MUX
                    {   
                        siteXY.append(i);
                        siteXY.append(std::round(j));
                        sliceSiteXYs.append(siteXY);
                        colSet0.insert(i);
                        colSet2.insert(i);
                        colSet3.insert(i);
                        colSet4.insert(i);
                        break;
                    }
                case 2: //SLICEM: LUTL/LUTM/FF/CARRY
                    {
                        siteXY.append(i);
                        siteXY.append(std::round(j));
                        sliceSiteXYs.append(siteXY);
                        colSet0.insert(i);
                        colSet1.insert(i);
                        colSet2.insert(i);
                        colSet3.insert(i);
                        colSet4.insert(i);
                        break;
                    }
                case 3: //DSP
                    {
                        siteXY.append(i);
                        siteXY.append(std::round(j/2.5)*2.5);
                        dspSiteXYs.append(siteXY);
                        colSet5.insert(i);
                        break;
                    }
                case 4: //RAM
                    {
                        siteXY.append(i);
                        siteXY.append(std::round(j/5.0)*5.0);
                        ramSiteXYs.append(siteXY);
                        colSet6.insert(i);
                        break;
                    }
                case 5: //IO
                    {    
                        colSet7.insert((PlaceDB::index_type)i);
                        break;
                    }
                default: //Empty
                    {
                        break;
                    }
            }
            rowVals.append(db.siteVal(i,j));
            rowNames.append(db.siteName(i,j));
        }
        site_type_map.append(rowVals);
        site_name_map.append(rowNames);
        lg_siteXYs.append(lg_rowXY);
    }

    updCols.emplace_back(colSet0);
    updCols.emplace_back(colSet1);
    updCols.emplace_back(colSet2);
    updCols.emplace_back(colSet3);
    updCols.emplace_back(colSet4);
    updCols.emplace_back(colSet5);
    updCols.emplace_back(colSet6);
    updCols.emplace_back(colSet7);

    unsigned int flat_len = 0;
    flat_region_boxes_start.append(flat_len);
    for (unsigned int rgn = 0; rgn < 8; ++rgn)
    {
        for (unsigned int cEl = 0; cEl < updCols[rgn].size(); ++cEl)
        {
            auto elm = updCols[rgn].begin();
            std::advance(elm, cEl);
            pybind11::list flat_region;
            flat_region.append(*elm);
            flat_region.append(0);
            flat_region.append(*elm + 1);
            flat_region.append(db.height());

            flat_region_boxes.append(flat_region);
            flat_len += 1;
        }
        flat_region_boxes_start.append(flat_len);
    }

    xl = db.xl(); 
    yl = db.yl(); 
    xh = db.xh(); 
    yh = db.yh(); 

    row_height = db.height();
    site_width = db.width(); 

    num_sites_x = db.siteRows();
    num_sites_y = db.siteCols();

    // routing information initialized 
    num_routing_grids_x = db.width(); 
    num_routing_grids_y = db.height(); 
    routing_grid_xl = xl; 
    routing_grid_yl = yl; 
    routing_grid_xh = xh; 
    routing_grid_yh = yh; 

    ////Spiral Accessor
    unsigned int rad = std::max(num_sites_x, num_sites_y);
    spiral_maxVal = (2 * rad * (1+rad)) +1; 
    spiral_accessor.append(std::make_tuple(0, 0));

    for(int r = 1; r <= rad; ++r)
    {
        // The 1st quadrant
        for (int x = r, y = 0; y < r; --x, ++y)
        {
            spiral_accessor.append(std::make_tuple(x, y));
        }
        // The 2nd quadrant
        for (int x = 0, y = r; y > 0; --x, --y)
        {
            spiral_accessor.append(std::make_tuple(x, y));
        }
        // The 3rd quadrant
        for (int x = -r, y = 0; y > -r; ++x, --y)
        {
            spiral_accessor.append(std::make_tuple(x, y));
        }
        // The 4th quadrant
        for (int x = 0, y = -r; y < 0; ++x, ++y)
        {
            spiral_accessor.append(std::make_tuple(x, y));
        }
    }
    
}

DREAMPLACE_END_NAMESPACE

