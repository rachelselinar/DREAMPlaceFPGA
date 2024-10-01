/**
 @file   InterchangeDataBase.h
 @author Zhili Xiong (interchange format)
 @date   May 2024
 @brief  Implementation of FPGA interchange format database
 */
/**
 * Modifications Copyright(C) 2024 Advanced Micro Devices, Inc. All rights reserved
 */

#ifndef INTERCHANGE_DATABASE_H
#define INTERCHANGE_DATABASE_H

#include <limits>
#include "Util.h"
#include <limbo/parsers/bookshelf/bison/BookshelfDriver.h> // bookshelf parser 

DREAMPLACE_BEGIN_NAMESPACE

class InterchangeDataBase
{
    public:
        ///@brief update site size 
        virtual void resize_sites(int, int);
        ///@brief update site information
        virtual void site_info_update(int, int, int);
        ///@brief update site name 
        virtual void add_site_name(int, int, std::string const&);
        /// @brief add library cell by name
        virtual void add_bel_map(std::string const& name, int z);
        /// @brief add library cell by name
        virtual void add_lib_cell(std::string const& name);
        /// @brief add cell input pin
        virtual void add_input_pin(std::string& pName);
        /// @brief add cell output pin
        virtual void add_output_pin(std::string& pName);
        /// @brief add cell clock pin
        virtual void add_clk_pin(std::string& pName);
        /// @brief add cell control pin
        virtual void add_ctrl_pin(std::string& pName);
        /// @brief add node from interchange .netlist
        virtual void add_interchange_node(std::string& name, std::string& type); 
        /// @brief update placement nodes for shape support
        virtual void update_interchange_nodes();
        /// @brief add shape from interchange .netlist
        virtual void add_interchange_shape(double height, double width);
        /// @brief add nodes to shape from interchange .netlist
        virtual void add_org_node_to_shape(std::string const& cellName, std::string const& belName, int dx, int dy);
        /// @brief add net as bookshelf format 
        virtual void add_interchange_net(BookshelfParser::Net const& n); 
        /// @brief a callback when a bookshelf file reaches to the end 
        virtual void interchange_end();

    private:
        /// @brief remind users to define some optional callback functions at runtime 
        /// @param str message including the information to the callback function in the reminder 
        void interchange_user_cbk_reminder(const char* str) const;
};


DREAMPLACE_END_NAMESPACE

#endif
