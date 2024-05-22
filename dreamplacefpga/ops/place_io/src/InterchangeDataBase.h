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

DREAMPLACE_BEGIN_NAMESPACE

class InterchangeDataBase
{
    public:
        ///@brief update site size 
        virtual void resize_sites(int, int);
        ///@brief update site information
        virtual void site_info_update(int, int, int);

        // virtual void add_lib_cell(std::string const&) = 0; 

        private:
        /// @brief remind users to define some optional callback functions at runtime 
        /// @param str message including the information to the callback function in the reminder 
        void interchange_user_cbk_reminder(const char* str) const;
};


DREAMPLACE_END_NAMESPACE

#endif