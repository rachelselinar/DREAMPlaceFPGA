/**
 @file   InterchangeDriver.h
 @author Zhili Xiong (interchange format)
 @date   May 2024
 @brief  Implementation of FPGA interchange format parser
 */

#include "InterchangeDriver.h"
#include <map>
#include <fstream>
#include <fcntl.h>
#include <kj/compat/gzip.h>
#include <kj/std/iostream.h>
#include <capnp/serialize.h>
#include <capnp/message.h>
#include <interchange/DeviceResources.capnp.h>
#include <interchange/LogicalNetlist.capnp.h>

DREAMPLACE_BEGIN_NAMESPACE

InterchangeDriver::InterchangeDriver(InterchangeDataBase& db)
            : m_db(db)
{
}

bool InterchangeDriver::parse_device(std::string const& filename)
{   
    int fd = open(filename.c_str(), O_RDONLY);
    kj::FdInputStream fdInput(fd);
    kj::GzipInputStream gzipInput(fdInput);
    capnp::InputStreamMessageReader messageReader(gzipInput, {1024L*1024L*1024L*64L, 64});
    DeviceResources::Device::Reader deviceRoot = messageReader.getRoot<DeviceResources::Device>();
    
    // Test print out the device name
    std::cout << "Parsing interchange for device : " << deviceRoot.getName().cStr() << std::endl;

    auto strings = deviceRoot.getStrList();
    auto tileTypes = deviceRoot.getTileTypeList();
    auto siteTypes = deviceRoot.getSiteTypeList();
    
    std::map<std::pair<int, int>, int> tile2SiteTypeId; //Map a tile (col, row) location to a site type index
    std::map<std::pair<int, int>, int> sliceTile2Y; //Record a tile (col, row) location that has slice site to siteY

    auto tiles = deviceRoot.getTileList();
    int maxY = 0;
    for (int i=0; i < tiles.size(); i++){
        auto tile = tiles[i];
        auto tileType = tileTypes[tile.getType()];
        int row = tile.getRow(); // the row order is reversed in tile location
        int col = tile.getCol();
        std::pair<int, int> tileLoc = std::make_pair(col, row);

        auto sites = tile.getSites();
        for (int j=0; j < sites.size(); j++){
            auto site = sites[j];
            auto tileSiteTypes = tileType.getSiteTypes();
            int siteTypeIdx = tileSiteTypes[site.getType()].getPrimaryType();
            auto siteType = siteTypes[siteTypeIdx];

            std::string siteTypeName = strings[siteType.getName()].cStr();
            std::string siteName = strings[site.getName()].cStr();

            if ((siteTypeName.find("SLICEL") != std::string::npos) || (siteTypeName.find("SLICEM") != std::string::npos))
            {
                int siteY = std::stoi(siteName.substr(siteName.find("Y")+1));
                maxY = (siteY > maxY)? siteY : maxY;
                sliceTile2Y.insert(std::make_pair(tileLoc, siteY));
                tile2SiteTypeId.insert(std::make_pair(tileLoc, 1));
            } else if (siteTypeName.find("DSP48") != std::string::npos)
            {
                tile2SiteTypeId.insert(std::make_pair(tileLoc, 2)); 
            } else if (siteTypeName.find("RAMBFIFO36") != std::string::npos)
            {
                tile2SiteTypeId.insert(std::make_pair(tileLoc, 3));
            // "HPIOB" and "HRIO" are from Ultrascale, 
            // "HPIOB_M", "HPIOB_S" and "HPIOB_SNGL" are from Ultrascale+.
            } else if (limbo::iequals(siteTypeName, "HPIOB") || limbo::iequals(siteTypeName, "HRIO") || \
                limbo::iequals(siteTypeName, "HPIOB_M") || limbo::iequals(siteTypeName, "HPIOB_S") || \
                limbo::iequals(siteTypeName, "HPIOB_SNGL"))
            {
                tile2SiteTypeId.insert(std::make_pair(tileLoc, 4));
            // Make BUFGCE a different type due to different site size, merge to IO type in m_db
            } else if ( limbo::iequals(siteTypeName, "BUFGCE"))
            {
                tile2SiteTypeId.insert(std::make_pair(tileLoc, 5));
            }
        }
    }
    
    int xIdx = 0, yIdx = 0, numGridX = 0, numGridY = 0;
    int numSitesCol = 0;
    std::map<std::pair<int, int>, int> siteMap;
    // Add IO padding for the first and last column, this is ONLY for the ISPD'16 benchmarks
    bool ioPadding = true; 
    if (ioPadding)
    {
        // Add IO padding for the first column
        for (int i=0; i < int((maxY+1)/60); i++)
        {
            yIdx = int(60.0*i);
            siteMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), 4));
        }
    }
    xIdx++;

    // Add sites based on cols
    int lastCol = (tile2SiteTypeId.begin()->first).first;
    for (auto it = tile2SiteTypeId.begin(); it != tile2SiteTypeId.end(); it++)
    {
        std::pair<int, int> loc = it->first;
        int col = loc.first;
        int row = loc.second;
        int siteTypeId = it->second;

        if (col!= lastCol)
        {
            xIdx++;
            numSitesCol = 0;
        }
      
        switch(siteTypeId)
        {
            case 1: //SLICEL/SLICEM
                {   
                    // For Ultrascale+, slices sometimes don't occupy a full column
                    yIdx = sliceTile2Y.at(loc);
                    siteMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), 1));
                    numSitesCol++;
                    break;
                }
            case 2: //DSP
                {   
                    for (int k=0; k < 2; k++)
                    {
                        yIdx = int(2.5*numSitesCol);
                        siteMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), 2));
                        numSitesCol++;
                    }
                    break;
                }
            case 3: //BRAM
                {
                    yIdx = int(5.0*numSitesCol);
                    siteMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), 3));
                    numSitesCol++;
                    break;
                }
            case 4: //IOB
                {   
                    yIdx = int(30.0*numSitesCol);
                    siteMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), 4));
                    numSitesCol++;
                    break;
                }
            case 5: //BUFGCE
                {
                    yIdx = int(60.0*numSitesCol);
                    siteMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), 4));
                    numSitesCol++;
                    break;
                }
            default:
                {
                    break;
                }
        }

        // Update last col number
        lastCol = col;
        if (numSitesCol > numGridY) {numGridY = numSitesCol;}
    }

    // Add IO padding for the first and last column, this is ONLY for the ISPD'16 benchmarks
    xIdx++;
    if (ioPadding)
    {
        // Add IO padding for the last column
        for (int i=0; i < int((maxY+1)/60); i++)
        {
            yIdx = int(60.0*i);
            siteMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), 4));
        }
    }
    numGridX = xIdx+1;
    
    ////////////////////////// call functions for building database /////////////////////////////////
    m_db.resize_sites(numGridX, numGridY);
    // std::cout << "numGridX: " << numGridX << ", numGridY: "  << numGridY << std::endl; 
    for (auto it = siteMap.begin(); it != siteMap.end(); it++){
        std::pair<int, int> loc = it->first;
        m_db.site_info_update(loc.first, loc.second, it->second);
        // std::cout << "x: " << loc.first << ", y: "  << loc.second << ", type:"<< it->second << std::endl;
    }

    return true;
}

bool InterchangeDriver::parse_logical_netlist(std::string const& filename)
{
    int fd = open(filename.c_str(), O_RDONLY);
    kj::FdInputStream fdInput(fd);
    kj::GzipInputStream gzipInput(fdInput);
    capnp::InputStreamMessageReader messageReader(gzipInput, {1024L*1024L*1024L*64L, 64});
    LogicalNetlist::Netlist::Reader NetlistRoot = messageReader.getRoot<LogicalNetlist::Netlist>();

    std::cout << "Logical Netlist Name: " << NetlistRoot.getName().cStr() << std::endl;
    auto strings = NetlistRoot.getStrList();
    auto cells = NetlistRoot.getCellDecls();

    return true;
}


DREAMPLACE_END_NAMESPACE