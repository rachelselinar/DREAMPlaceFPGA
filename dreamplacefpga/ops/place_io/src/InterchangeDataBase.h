/**
 @file   InterchangeDataBase.h
 @author Zhili Xiong (interchange format)
 @date   May 2024
 @brief  Implementation of FPGA interchange format database
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
        /// @brief add node as bookshelf format 
        /// @Zhili TODO: Is it necessary to modify these 3 function for interchange?
        virtual void add_bookshelf_node(std::string& name, std::string& type); 
        /// @brief add net as bookshelf format 
        virtual void add_bookshelf_net(BookshelfParser::Net const& n); 
        /// @brief a callback when a bookshelf file reaches to the end 
        virtual void bookshelf_end(); 

    private:
        /// @brief remind users to define some optional callback functions at runtime 
        /// @param str message including the information to the callback function in the reminder 
        void interchange_user_cbk_reminder(const char* str) const;
};


DREAMPLACE_END_NAMESPACE

#endif