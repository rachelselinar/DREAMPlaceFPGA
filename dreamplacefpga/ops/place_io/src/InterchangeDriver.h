/**
 @file   InterchangeDriver.h
 @author Zhili Xiong (interchange format)
 @date   May 2024
 @brief  Implementation of FPGA interchange format parser
 */

#ifndef INTERCHANGE_DRIVER_H
#define INTERCHANGE_DRIVER_H

#include "InterchangeDataBase.h" 

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
        bool parse_device(std::string const& filename); 
        bool parse_logical_netlist(std::string const& filename);

        InterchangeDataBase& m_db;   
};

DREAMPLACE_END_NAMESPACE

#endif