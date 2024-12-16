/**
 @file   InterchangeDriver.h
 @author Zhili Xiong (interchange format)
 @date   May 2024
 @brief  Implementation of FPGA interchange format parser
 */

#include "InterchangeDriver.h"
#include "utility/src/Msg.h"
#include <chrono>

DREAMPLACE_BEGIN_NAMESPACE

InterchangeDriver::InterchangeDriver(InterchangeDataBase& db)
            : m_db(db)
{
    m_net.reset();
    maxSiteY = 0;
    numGridX = 0;
    numGridY = 0;
}

void InterchangeDriver::setTileToSiteType()
{
    auto strings = interchangeDeviceRoot.getStrList();
    auto tileTypes = interchangeDeviceRoot.getTileTypeList();
    auto siteTypes = interchangeDeviceRoot.getSiteTypeList();
    auto tiles = interchangeDeviceRoot.getTileList();

    for (int i = 0; i < tiles.size(); i++)
    {
        auto tile = tiles[i];
        auto tileType = tileTypes[tile.getType()];
        int row = tile.getRow(); // the row order is reversed in tile location
        int col = tile.getCol();
        std::pair<int, int> tileLoc = std::make_pair(col, row);
        tile2ColRow.emplace_back(tileLoc);
        std::vector<std::string> siteNames;
        tile2SiteTypeId.emplace_back(0);
 
        auto sites = tile.getSites();
        for (int j = 0; j < sites.size(); j++)
        {
            auto site = sites[j];
            auto tileSiteTypes = tileType.getSiteTypes();
            int siteTypeIdx = tileSiteTypes[site.getType()].getPrimaryType();
            auto siteType = siteTypes[siteTypeIdx];

            std::string siteTypeName = strings[siteType.getName()].cStr();
            std::string siteName = strings[site.getName()].cStr();

            if ((siteTypeName.find("SLICEL") != std::string::npos) || (siteTypeName.find("SLICEM") != std::string::npos))
            {
                int siteY = std::stoi(siteName.substr(siteName.find("Y")+1));
                maxSiteY = (siteY > maxSiteY)? siteY : maxSiteY;
                tile2SiteTypeId[i] = 1;
                siteNames.emplace_back(siteName);

            } else if (siteTypeName.find("DSP48") != std::string::npos)
            {
                tile2SiteTypeId[i] = 2;
                siteNames.emplace_back(siteName);
            } else if (siteTypeName.find("RAMBFIFO36") != std::string::npos)
            {
                tile2SiteTypeId[i] = 3;
                siteNames.emplace_back(siteName);
            // "HPIOB" and "HRIO" are from Ultrascale, 
            // "HPIOB_M", "HPIOB_S" and "HPIOB_SNGL" are from Ultrascale+.
            } else if (limbo::iequals(siteTypeName, "HPIOB") || limbo::iequals(siteTypeName, "HRIO") || \
                limbo::iequals(siteTypeName, "HPIOB_M") || limbo::iequals(siteTypeName, "HPIOB_S") || \
                limbo::iequals(siteTypeName, "HPIOB_SNGL"))
            {
                tile2SiteTypeId[i] = 4;
                siteNames.emplace_back(siteName);
            // Make BUFGCE a different type due to different site size, merge to IO type in m_db
            } else if ( limbo::iequals(siteTypeName, "BUFGCE"))
            {
                tile2SiteTypeId[i] = 5;
                siteNames.emplace_back(siteName);
            } 
        }
        tile2SiteNames.emplace_back(siteNames);
    }

}

void InterchangeDriver::setSiteMap()
{
    int xIdx = 0, yIdx = 0, xIdx_temp = 0, numTilePerCol = 0;

    /// sort the tiles based on the col order and reversed row order
    std::vector<int> SortedTileIndices(tile2ColRow.size());
    std::iota(SortedTileIndices.begin(), SortedTileIndices.end(), 0);
    std::sort(SortedTileIndices.begin(), SortedTileIndices.end(), 
        [&](int i, int j) {return tile2ColRow[i].first == tile2ColRow[j].first? tile2ColRow[i].second > tile2ColRow[j].second : tile2ColRow[i].first < tile2ColRow[j].first;});

    // Add IO padding for the first and last column, this is ONLY for the ISPD'16 benchmarks
    bool ioPadding = true; 
    if (ioPadding)
    {
        // Add IO padding for the first column
        for (int i = 0; i < int((maxSiteY+1)/60); i++)
        {
            yIdx = int(60.0*i);
            siteMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), 4));
            siteNameMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), "padding"));
        }
    }

    int lastCol = -1;
    // Add sites based on cols
    for (int i = 0; i < SortedTileIndices.size(); i++)
    {
        int tileId = SortedTileIndices[i];
        int col = tile2ColRow[tileId].first;
        int row = tile2ColRow[tileId].second;
        int siteTypeId = tile2SiteTypeId[tileId];
        std::vector<std::string> siteNames = tile2SiteNames[tileId];

        if (siteTypeId == 0) {continue;}

        if(col != lastCol)
        {
            xIdx = (xIdx_temp > xIdx)? xIdx_temp+1 : xIdx+1;
            numTilePerCol = 0;
        }

        xIdx_temp = xIdx;
        switch(siteTypeId)
        {
            case 1: //SLICEL/SLICEM
                {   
                    int lastSLICEX = -1;
                    // sort siteName base on SITEX
                    std::sort(siteNames.begin(), siteNames.end(), 
                        [&](std::string a, std::string b) {return std::stoi(a.substr(a.find("X")+1)) < std::stoi(b.substr(b.find("X")+1));});

                    for (int k = 0; k < siteNames.size(); k++)
                    {   
                        std::string siteName = siteNames[k];
                        int SLICEY = std::stoi(siteName.substr(siteName.find("Y")+1));
                        int SLICEX = std::stoi(siteName.substr(siteName.find("X")+1, siteName.find("Y")-siteName.find("X")-1));
                        yIdx = SLICEY;
                        if (SLICEX != lastSLICEX && lastSLICEX != -1) {xIdx_temp++;}
                        siteMap.insert(std::make_pair(std::make_pair(xIdx_temp, yIdx), 1));
                        siteNameMap.insert(std::make_pair(std::make_pair(xIdx_temp, yIdx), siteName));
                        lastSLICEX = SLICEX;
                    }
                    break;
                }
            case 2: //DSP
                {   
                    for (int k = 0; k < siteNames.size(); k++)
                    {
                        std::string siteName = siteNames[k];
                        int DSPY = std::stoi(siteName.substr(siteName.find("Y")+1));
                        yIdx = int(2.5*DSPY);
                        siteMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), 2));
                        siteNameMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), siteName));
                    }
                    break;
                }
            case 3: //BRAM
                {
                    for (int k = 0; k < siteNames.size(); k++)
                    {
                        std::string siteName = siteNames[k];
                        int BRAMY = std::stoi(siteName.substr(siteName.find("Y")+1));
                        yIdx = int(5.0*BRAMY);
                        siteMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), 3));
                        siteNameMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), siteName));
                    }
                    break;
                }
            case 4: //IOB
                {   
                    yIdx = int(30.0*numTilePerCol);
                    siteMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), 4));
                    siteNameMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), "IOB"));
                    numTilePerCol++;
                    break;
                }
            case 5: //BUFGCE
                {   
                    yIdx = int(60.0*numTilePerCol);
                    siteMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), 4));
                    siteNameMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), "BUFGCE"));
                    numTilePerCol++;
                    break;
                }
            default:
                {
                    break;
                }
        }

        lastCol = col;
    }

    xIdx = (xIdx_temp > xIdx)? xIdx_temp+1 : xIdx+1;
    // Add IO padding for the first and last column, this is ONLY for the ISPD'16 benchmarks
    if (ioPadding)
    {
        // Add IO padding for the last column
        for (int i = 0; i < int((maxSiteY+1)/60); i++)
        {
            yIdx = int(60.0*i);
            siteMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), 4));
            siteNameMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), "padding"));
        }
    }
    numGridX = xIdx+1;
    numGridY = maxSiteY+1;
}

void InterchangeDriver::addSiteMapToDateBase()
{   
    m_db.resize_sites(numGridX, numGridY);
    for (auto it = siteMap.begin(); it != siteMap.end(); it++)
    {
        std::pair<int, int> loc = it->first;
        m_db.site_info_update(loc.first, loc.second, it->second);
        m_db.add_site_name(loc.first, loc.second, siteNameMap.at(loc));
    }
}

void InterchangeDriver::addLibCellsToDataBase()
{   
    //// prepare data from logical netlist root for lib cells
    auto strings = interchangeNetlistRoot.getStrList();
    auto libCells = interchangeNetlistRoot.getCellDecls();
    auto cellPorts = interchangeNetlistRoot.getPortList();

    for (int i = 0; i < libCells.size(); i++)
    {
        auto libCell = libCells[i];
        auto Ports = libCell.getPorts();

        std::string cellTypeName = strings[libCell.getName()].cStr();
        
        m_db.add_lib_cell(cellTypeName);

        for (int j = 0; j < Ports.size(); j++)
        {   
            auto portIdx = Ports[j];
            std::string portName = strings[cellPorts[portIdx].getName()].cStr();
            auto portDir = cellPorts[portIdx].getDir();
            std::vector<std::string> busNames;

            //Address the bus ports
            if (cellPorts[portIdx].isBus())
            {
                auto busStart = cellPorts[portIdx].getBus().getBusStart();
                auto busEnd = cellPorts[portIdx].getBus().getBusEnd();
                std::vector<int> busIndices(std::abs(int(busStart-busEnd))+1);
                int busPortId = port2BusNames.size();
                
                if (busStart > busEnd) 
                {   
                    std::iota(busIndices.rbegin(), busIndices.rend(), busEnd);
                } else {
                    std::iota(busIndices.begin(), busIndices.end(), busStart);
                }

                for (int k = 0; k < busIndices.size(); k++)
                {
                    std::string busName = portName + '[' + std::to_string(busIndices[k]) + ']';
                    busNames.emplace_back(busName);
                    
                    switch (portDir) 
                    {
                        case LogicalNetlist::Netlist::Direction::INPUT:
                        {
                            m_db.add_input_pin(busName);
                            break;
                        }
                        case LogicalNetlist::Netlist::Direction::OUTPUT:
                        {
                            m_db.add_output_pin(busName);
                            break;
                        }
                        case LogicalNetlist::Netlist::Direction::INOUT:
                        {
                            m_db.add_input_pin(busName);
                            m_db.add_output_pin(busName);
                            break;
                        }
                    }
                }
                port2BusNames.emplace_back(busNames);
                busPort2Index.insert(std::make_pair(portIdx, busPortId));

            // for ports that are not bus
            } else {
                switch (portDir) 
                    {
                        case LogicalNetlist::Netlist::Direction::INPUT:
                        {   
                            if (limbo::iequals(portName, "CLK") || limbo::iequals(portName, "C"))
                            {
                                m_db.add_clk_pin(portName);
                            } else if (limbo::iequals(portName, "R") || limbo::iequals(portName, "CE")){
                                m_db.add_ctrl_pin(portName);
                            } else {
                                m_db.add_input_pin(portName);
                            }
                            break;
                        }
                        case LogicalNetlist::Netlist::Direction::OUTPUT:
                        {
                            m_db.add_output_pin(portName);
                            break;
                        }
                        case LogicalNetlist::Netlist::Direction::INOUT:
                        {
                            m_db.add_input_pin(portName);
                            m_db.add_output_pin(portName);
                            break;
                        }
                    }
            }
        } 
    }
}

void InterchangeDriver::addNodesToDataBase()
{   
    //// prepare data from logical netlist root for nodes
    auto strings = interchangeNetlistRoot.getStrList();
    auto instList = interchangeNetlistRoot.getInstList();
    auto libCells = interchangeNetlistRoot.getCellDecls();

    for (int i = 0; i < instList.size(); i++)
    {
        auto inst = instList[i];
        std::string instName = strings[inst.getName()].cStr();
        std::string instTypeName = strings[libCells[inst.getCell()].getName()].cStr();

        m_db.add_bookshelf_node(instName, instTypeName);
    }
    
}

void InterchangeDriver::addNetsToDataBase()
{
    //// prepare data from logical netlist root for nets
    auto strings = interchangeNetlistRoot.getStrList();
    auto cellList = interchangeNetlistRoot.getCellList();
    auto instList = interchangeNetlistRoot.getInstList();
    auto libCells = interchangeNetlistRoot.getCellDecls();
    auto cellPorts = interchangeNetlistRoot.getPortList();

    for (int i = 0; i < cellList.size(); i++)
    {
        auto cell = cellList[i];
        auto cellInsts = cell.getInsts();
        auto cellNets = cell.getNets();

        for (int j = 0; j < cellNets.size(); j++)
        {
            auto net = cellNets[j];
            std::string netName = strings[net.getName()].cStr();

            hashspace::unordered_map<std::string, int>::iterator fnd = netName2Index.find(netName);
            if (fnd == netName2Index.end()) // SKIP if the net is already in the database
            {
                int netId(netNames.size());
                netName2Index.insert(std::make_pair(netName, netId));
                netNames.emplace_back(netName);
                bool isExtPortNet = false;
                bool isVCCGND = false;

                m_net.net_name = netName;
                m_net.vNetPin.clear();

                auto netPorts = net.getPortInsts();
                for (int k = 0; k < netPorts.size(); k++)
                { 
                    auto port = netPorts[k];
                    std::string portName = strings[cellPorts[port.getPort()].getName()].cStr();

                    if (port.getBusIdx().isIdx())
                    {
                        int busPortId = busPort2Index.at(port.getPort());
                        std::string busName = port2BusNames.at(busPortId).at(port.getBusIdx().getIdx());
                        portName = busName;
                    }

                    if (port.isInst())
                    {
                        auto inst = instList[port.getInst()];
                        std::string cellName = strings[inst.getName()].cStr();
                        std::string cellTypeName = strings[libCells[inst.getCell()].getName()].cStr();
                        if (limbo::iequals(cellTypeName, "VCC")|| limbo::iequals(cellTypeName, "GND"))
                        {
                            isVCCGND = true;
                        }
                        m_net.vNetPin.push_back(BookshelfParser::NetPin(cellName, portName));

                    } else if (port.isExtPort())
                    {
                        isExtPortNet = true;
                    }   
                    
                }

                m_db.add_bookshelf_net(m_net);
                m_net.reset();
            }
               
        }
    }
}

bool InterchangeDriver::parse_device(DeviceResources::Device::Reader const& deviceRoot)
{
    interchangeDeviceRoot = deviceRoot;

    this->setTileToSiteType();
    this->setSiteMap();
    this->addSiteMapToDateBase();

    return true;
}

bool InterchangeDriver::parse_netlist(LogicalNetlist::Netlist::Reader const& netlistRoot)
{
    interchangeNetlistRoot = netlistRoot;

    this->addLibCellsToDataBase();
    this->addNodesToDataBase();
    this->addNetsToDataBase();

    return true;
}

bool readDeviceNetlist(InterchangeDataBase& db, std::string const& deviceFile, std::string const& netlistFile)
{
    InterchangeDriver driver(db);

    if (deviceFile.empty() || netlistFile.empty())
    {   
        std::cerr << "Error: device file or netlist file is not specified." << std::endl;
        return false;
    }

    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    int fd_device = open(deviceFile.c_str(), O_RDONLY);
    kj::FdInputStream fdInputDevice(fd_device);
    kj::GzipInputStream gzipInputDevice(fdInputDevice);
    capnp::InputStreamMessageReader deviceReader(gzipInputDevice, {1024L*1024L*1024L*64L, 64});
    DeviceResources::Device::Reader deviceRoot = deviceReader.getRoot<DeviceResources::Device>();

    std::cout << "Parsing device file " << deviceFile << std::endl;
    driver.parse_device(deviceRoot);

    // std::cout << "Total time for parsing device file: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count()/1000000.0 << " seconds" << std::endl;
    // std::chrono::steady_clock::time_point begin_netlist = std::chrono::steady_clock::now();

    int fd_netlist = open(netlistFile.c_str(), O_RDONLY);
    kj::FdInputStream fdInputNetlist(fd_netlist);
    kj::GzipInputStream gzipInputNetlist(fdInputNetlist);
    capnp::InputStreamMessageReader netlistReader(gzipInputNetlist, {1024L*1024L*1024L*64L, 64});
    LogicalNetlist::Netlist::Reader netlistRoot = netlistReader.getRoot<LogicalNetlist::Netlist>();

    std::cout << "Parsing netlist file " << netlistFile << std::endl;
    driver.parse_netlist(netlistRoot);
    
    // std::cout << "Total time for parsing netlist file: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin_netlist).count()/1000000.0 << " seconds" << std::endl;
    // std::chrono::steady_clock::time_point begin_db = std::chrono::steady_clock::now();

    db.bookshelf_end(); // Finalize the database

    // std::cout << "Total time for finalizing database: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin_db).count()/1000000.0 << " seconds" << std::endl;

    return true;
}


DREAMPLACE_END_NAMESPACE
