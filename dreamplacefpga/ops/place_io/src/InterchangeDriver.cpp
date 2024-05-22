/**
 @file   InterchangeDriver.h
 @author Zhili Xiong (interchange format)
 @date   May 2024
 @brief  Implementation of FPGA interchange format parser
 */

#include "InterchangeDriver.h"
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

    std::cout << "Device Name: " << deviceRoot.getName().cStr() << std::endl;

    auto strings = deviceRoot.getStrList();

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