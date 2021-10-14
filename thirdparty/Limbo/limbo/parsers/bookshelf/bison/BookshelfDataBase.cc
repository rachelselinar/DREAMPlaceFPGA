/**
 @file   BookshelfDataBase.cc
 @author Rachel Selina
 @date   Jan 2021
 @brief  Implementation of @ref BookshelfParser::BookshelfDataBase
 */

#include <limbo/parsers/bookshelf/bison/BookshelfDataBase.h>
#include <cstring>
#include <cstdlib>

namespace BookshelfParser {

void BookshelfDataBase::resize_clk_regions(int xReg, int yReg)
{
    cerr << "Bookshelf has " << xReg << " x " << yReg << " clock regions" << endl; 
    bookshelf_user_cbk_reminder(__func__); 
}

void BookshelfDataBase::resize_sites(int xSize, int ySize)
{
    cerr << "Bookshelf has " << xSize << " x " << ySize << " sites" << endl; 
    bookshelf_user_cbk_reminder(__func__); 
}

void BookshelfDataBase::site_info_update(int xloc, int yloc, int val)
{
    cerr << "Bookshelf has site (" << xloc << ", " << yloc << ") of type " << val << endl; 
    bookshelf_user_cbk_reminder(__func__); 
}

void BookshelfDataBase::add_clk_region(string const& name, int xh, int yh, int xl, int yl, int xm, int ym)
{
    cerr << "Bookshelf has clk region " << name << " at (" << xh << "," << yh << "," << xl << "," << yl << "," << xm << "," << ym << ")" << endl; 
    bookshelf_user_cbk_reminder(__func__); 
}

void BookshelfDataBase::add_lib_cell(string const& name)
{
    cerr << "Bookshelf has macro " << name << endl;
    bookshelf_user_cbk_reminder(__func__); 
}

//void BookshelfDataBase::resize_bookshelf_niterminal_layers(int n)
//{
//    cerr << "Bookshelf route has " << n << " NI terminals with layers" << endl; 
//    bookshelf_user_cbk_reminder(__func__); 
//}
//
//void BookshelfDataBase::resize_bookshelf_blockage_layers(int n)
//{
//    cerr << "Bookshelf route has " << n << " blockages with layers" << endl; 
//    bookshelf_user_cbk_reminder(__func__); 
//}
//
//void BookshelfDataBase::add_bookshelf_terminal_NI(string& n, int, int)
//{
//    cerr << "Bookshelf has terminal_NI " << n << endl; 
//    bookshelf_user_cbk_reminder(__func__);
//}
//
//void BookshelfDataBase::set_bookshelf_net_weight(string const& name, double w)
//{
//    cerr << "Bookshelf net weight: " << name << " " << w << endl;
//    bookshelf_user_cbk_reminder(__func__);
//}
//
//void BookshelfDataBase::set_bookshelf_route_info(RouteInfo const&)
//{
//    cerr << "Bookshelf route: RouteInfo" << endl; 
//    bookshelf_user_cbk_reminder(__func__); 
//}
//
//void BookshelfDataBase::add_bookshelf_niterminal_layer(string const& name, string const& layer)
//{
//    cerr << "Bookshelf route: " << name << ", " << layer << endl; 
//    bookshelf_user_cbk_reminder(__func__); 
//}
//
//void BookshelfDataBase::add_bookshelf_blockage_layers(string const& name, vector<string> const&)
//{
//    cerr << "Bookshelf route: " << name << endl; 
//    bookshelf_user_cbk_reminder(__func__); 
//}

void BookshelfDataBase::bookshelf_user_cbk_reminder(const char* str) const 
{
    cerr << "A corresponding user-defined callback is necessary: " << str << endl;
    exit(0);
}

} // namespace BookshelfParser
