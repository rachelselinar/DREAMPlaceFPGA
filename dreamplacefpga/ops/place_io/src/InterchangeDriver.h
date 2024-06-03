/**
 @file   InterchangeDriver.h
 @author Zhili Xiong (interchange format)
 @date   May 2024
 @brief  Implementation of FPGA interchange format parser
 */

#ifndef INTERCHANGE_DRIVER_H
#define INTERCHANGE_DRIVER_H

#include "InterchangeDataBase.h" 
#include "boost/unordered_map.hpp"

#include <map>
#include <fstream>
#include <fcntl.h>
#include <numeric>
#include <kj/compat/gzip.h>
#include <kj/std/iostream.h>
#include <capnp/serialize.h>
#include <capnp/message.h>
#include <interchange/DeviceResources.capnp.h>
#include <interchange/LogicalNetlist.capnp.h>


DREAMPLACE_BEGIN_NAMESPACE

/// @brief parse FPGA interchange format
class InterchangeDriver
{   
public:
    /// construct a new parser 
    /// @param db reference to database 
    InterchangeDriver(InterchangeDataBase& db);

    InterchangeDataBase& m_db;  

    bool parse_device(DeviceResources::Device::Reader const& deviceRoot);
    bool parse_netlist(LogicalNetlist::Netlist::Reader const& netlistRoot);

    /// @brief from .device file, build a map of tile2SiteTypeId and sliceTile2Y
    void setTileToSiteType();
    /// @brief from .device file, build a site map from (xIdx, yIdx) to SiteType
    void setSiteMap();
    /// @brief from .device file, write information from siteMap to Database
    void addSiteMapToDateBase();

    /// @brief from .netlist file, record libcells
    void addLibCellsToDataBase();
    /// @brief from .netlist file, record nodes
    void addNodesToDataBase();
    /// @brief from .netlist file, record nets
    void addNetsToDataBase();

protected: 
    BookshelfParser::Net m_net; ///< temporary storage of net    

    DeviceResources::Device::Reader interchangeDeviceRoot;
    LogicalNetlist::Netlist::Reader interchangeNetlistRoot;

    boost::unordered_map<std::pair<int, int>, int> siteMap;
    // std::map<std::pair<int, int>, int> siteMap;
    boost::unordered_map<std::pair<int, int>, std::string> siteNameMap;

    std::vector<std::vector<std::string>> tile2SiteNames; /// Tile to site names
    std::vector<std::pair<int, int>> tile2ColRow; /// Tile to col and row
    std::vector<int> tile2SiteTypeId; /// Tile to site type id

    int maxSiteY; /// Max Y location for SLICE sites 
    int maxTileRow; /// Max num of tile rows
    int maxTileCol; /// Max num of tile cols

    int numGridX; /// Max num of grids at X direction
    int numGridY; /// Max num of grids at Y direction

    std::vector<std::vector<std::string>> cellType2BusNames;
    hashspace::unordered_map<std::string, int> cellType2Index;
    hashspace::unordered_map<std::string, int> netName2Index;
    std::vector<std::string> netNames;

};

/// @brief read logical netlist file and device file for interchange format
/// Read .device file for device information 
/// Read .netlist file for logical netlist information
/// @param db database which is derived from @ref DREAMPLACE_NAMESPACE::InterchangeDataBase
/// @param netlistFile .netlist file
bool readDeviceNetlist(InterchangeDataBase& db, const std::string& deviceFile, const std::string& netlistFile);


DREAMPLACE_END_NAMESPACE

#endif