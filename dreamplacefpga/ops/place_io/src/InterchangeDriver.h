/**
 @file   InterchangeDriver.h
 @author Zhili Xiong (interchange format)
 @date   May 2024
 @brief  Implementation of FPGA interchange format parser
 */

#ifndef INTERCHANGE_DRIVER_H
#define INTERCHANGE_DRIVER_H

#include "InterchangeDataBase.h" 

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
    /**
    * @param filename	input file name
    * @return		true if successfully parsed
    */
    // bool parse_device(std::string const& filename); 
    // bool parse_logical_netlist(std::string const& filename);

    InterchangeDataBase& m_db;  

    /// @brief from .device file, build a map of tile2SiteTypeId and sliceTile2Y
    void setTileToSiteType(DeviceResources::Device::Reader const& deviceRoot);
    /// @brief from .device file, build a site map from (xIdx, yIdx) to SiteType
    void setSiteMap();
    /// @brief from .device file, write information from siteMap to Database
    void addSiteMapToDateBase();

    /// @brief from .netlist file, record libcells
    void addLibCellsToDataBase(LogicalNetlist::Netlist::Reader const& netlistRoot);
    ///
    void addNodesToDataBase(LogicalNetlist::Netlist::Reader const& netlistRoot);
    ///
    void addNetsToDataBase(LogicalNetlist::Netlist::Reader const& netlistRoot);

protected: 
    BookshelfParser::Net m_net; ///< temporary storage of net    
    std::vector<std::string> m_vInterchangeFiles; ///< store FPGA interchange files
    std::string m_deviceFile; 
    std::string m_netlistFile;

    /// Map a tile (col, row) location to a site type index
    std::map<std::pair<int, int>, int> tile2SiteTypeId; 
    /// Record a tile (col, row) location that has slice site to siteY
    std::map<std::pair<int, int>, int> sliceTile2Y; 
    /// Map a site (x, y) location to a site type
    std::map<std::pair<int, int>, int> siteMap;
    
    int maxY; /// Max Y location for SLICE sites 
    int numGridX; /// Max num of grids at X direction 
    int numGridY; /// Max num of grids at Y direction
    
    // std::vector<std::string> supported_libCells = {''};
    std::vector<std::vector<std::string>> cellType2BusNames;
    hashspace::unordered_map<std::string, int> cellType2Index;

    hashspace::unordered_map<std::string, int> netName2Index;
    std::vector<std::string> netNames;

};

/// @brief read device file for interchange format
/// Read .device file for device information 
/// @param db database which is derived from @ref DREAMPLACE_NAMESPACE::InterchangeDataBase
/// @param deviceFile .device file 
bool readDevice(InterchangeDataBase& db, const std::string& deviceFile);
/// @brief read logical netlist file for interchange format
/// Read .net file for logical netlist information
/// @param db database which is derived from @ref DREAMPLACE_NAMESPACE::InterchangeDataBase
/// @param netlistFile .netlist file
bool readNetlist(InterchangeDataBase& db, const std::string& netlistFile);


DREAMPLACE_END_NAMESPACE

#endif