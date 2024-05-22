/**
 @file   InterchangeDataBase.cpp
 @author Zhili Xiong (interchange format)
 @date   May 2024
 @brief  Implementation of FPGA interchange format database
 */

#include "InterchangeDataBase.h"

DREAMPLACE_BEGIN_NAMESPACE

void InterchangeDataBase::resize_sites(int xSize, int ySize)
{
    std::cerr << "Bookshelf has " << xSize << " x " << ySize << " sites" << std::endl; 
    interchange_user_cbk_reminder(__func__); 
}

void InterchangeDataBase::site_info_update(int xloc, int yloc, int val)
{
    std::cerr << "Bookshelf has site (" << xloc << ", " << yloc << ") of type " << val << std::endl; 
    interchange_user_cbk_reminder(__func__); 
}

void InterchangeDataBase::interchange_user_cbk_reminder(const char* str) const 
{
    std::cerr << "A corresponding user-defined callback is necessary: " << str << std::endl;
    exit(0);
}

DREAMPLACE_END_NAMESPACE