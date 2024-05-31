/**
 @file   InterchangeDriver.h
 @author Zhili Xiong (interchange format)
 @date   May 2024
 @brief  Implementation of FPGA interchange format parser
 */

#include "InterchangeDriver.h"

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
}

void InterchangeDriver::setSiteMap()
{
    int xIdx = 0, yIdx = 0;
    int numSitesCol = 0;

    // Add IO padding for the first and last column, this is ONLY for the ISPD'16 benchmarks
    bool ioPadding = true; 
    if (ioPadding)
    {
        // Add IO padding for the first column
        for (int i = 0; i < int((maxSiteY+1)/60); i++)
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
        for (int i = 0; i < int((maxSiteY+1)/60); i++)
        {
            yIdx = int(60.0*i);
            siteMap.insert(std::make_pair(std::make_pair(xIdx, yIdx), 4));
        }
    }
    numGridX = xIdx+1;
}

void InterchangeDriver::addSiteMapToDateBase()
{
    m_db.resize_sites(numGridX, numGridY);
    for (auto it = siteMap.begin(); it != siteMap.end(); it++)
    {
        std::pair<int, int> loc = it->first;
        m_db.site_info_update(loc.first, loc.second, it->second);
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
        std::vector<std::string> busNames;
        int cellTypeId = cellType2BusNames.size();

        for (int j = 0; j < Ports.size(); j++)
        {   
            auto portIdx = Ports[j];
            std::string portName = strings[cellPorts[portIdx].getName()].cStr();
            auto portDir = cellPorts[portIdx].getDir();

            //Address the bus ports
            if (cellPorts[portIdx].isBus())
            {
                auto busStart = cellPorts[portIdx].getBus().getBusStart();
                auto busEnd = cellPorts[portIdx].getBus().getBusEnd();
                std::vector<int> busIndices(std::abs(int(busStart-busEnd))+1);
                
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

            // for ports that are not bus
            } else {
                busNames.emplace_back("");
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
        cellType2BusNames.emplace_back(busNames);
        cellType2Index.insert(make_pair(cellTypeName, cellTypeId));
        
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

                m_net.net_name = netName;
                m_net.vNetPin.clear();

                auto netPorts = net.getPortInsts();
                for (int k = 0; k < netPorts.size(); k++)
                { 
                    auto port = netPorts[k];
                    std::string portName = strings[cellPorts[port.getPort()].getName()].cStr();
                    
                    if (port.isInst())
                    {
                        auto inst = instList[port.getInst()];
                        std::string cellName = strings[inst.getName()].cStr();
                        std::string cellTypeName = strings[libCells[inst.getCell()].getName()].cStr();

                        if (port.getBusIdx().isIdx())
                        {
                            if (port.isExtPort())
                            {   
                                m_net.vNetPin.push_back(BookshelfParser::NetPin(cellName, netName));
                               
                            } else {
                                int cellTypeId = cellType2Index.at(cellTypeName);
                                std::string busName = cellType2BusNames.at(cellTypeId).at(port.getBusIdx().getIdx());
                                m_net.vNetPin.push_back(BookshelfParser::NetPin(cellName, busName));
                                
                            }

                        } else {
                            m_net.vNetPin.push_back(BookshelfParser::NetPin(cellName, portName));
                        }
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

    int fd_device = open(deviceFile.c_str(), O_RDONLY);
    kj::FdInputStream fdInputDevice(fd_device);
    kj::GzipInputStream gzipInputDevice(fdInputDevice);
    capnp::InputStreamMessageReader deviceReader(gzipInputDevice, {1024L*1024L*1024L*64L, 64});
    DeviceResources::Device::Reader deviceRoot = deviceReader.getRoot<DeviceResources::Device>();

    std::cout << "Parsing device file " << deviceFile << std::endl;
    driver.parse_device(deviceRoot);

    int fd_netlist = open(netlistFile.c_str(), O_RDONLY);
    kj::FdInputStream fdInputNetlist(fd_netlist);
    kj::GzipInputStream gzipInputNetlist(fdInputNetlist);
    capnp::InputStreamMessageReader netlistReader(gzipInputNetlist, {1024L*1024L*1024L*64L, 64});
    LogicalNetlist::Netlist::Reader netlistRoot = netlistReader.getRoot<LogicalNetlist::Netlist>();

    std::cout << "Parsing netlist file " << netlistFile << std::endl;
    driver.parse_netlist(netlistRoot);

    db.bookshelf_end(); // Finalize the database

    return true;
}


DREAMPLACE_END_NAMESPACE