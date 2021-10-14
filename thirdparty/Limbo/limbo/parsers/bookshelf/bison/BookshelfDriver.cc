/**
 * @file   BookshelfDriver.cc
 * @author Rachel Selina
 * @date   Jan 2021
 * @brief  Implementation of @ref BookshelfParser::Driver
 */

#include "BookshelfDriver.h"
#include "BookshelfScanner.h"
#include <limbo/string/String.h>
#include <algorithm>
#include <fstream>
#include <string>
#if ZLIB == 1 
#include <limbo/thirdparty/gzstream/gzstream.h>
#endif

namespace BookshelfParser {

Driver::Driver(BookshelfDataBase& db)
    : trace_scanning(false),
      trace_parsing(false),
      m_db(db) 
      //m_plFlag(false)
{
    m_net.reset();
    //m_row.reset();
    //m_routeInfo.reset();
    m_vBookshelfFiles.clear(); ///< store bookshelf files except .aux 
    m_libFile.clear();
    m_sclFile.clear();
    m_nodeFile.clear();
    m_netFile.clear();
    m_plFile.clear();
    m_wtFile.clear();

}

bool Driver::parse_stream(std::istream& in, const std::string& sname)
{
    streamname = sname;

    Scanner scanner(&in);
    scanner.set_debug(trace_scanning);
    this->lexer = &scanner;

    Parser parser(*this);
    parser.set_debug_level(trace_parsing);
    return (parser.parse() == 0);
}

bool Driver::parse_file(const std::string &filename)
{
#if ZLIB == 1
    if (limbo::get_file_suffix(filename) == "gz") // detect .gz file 
    {
        igzstream in (filename.c_str());

        if (!in.good()) return false;
        return parse_stream(in, filename);
    }
#endif
    std::ifstream in(filename.c_str());
    if (!in.good()) return false;
    return parse_stream(in, filename);
}

bool Driver::parse_string(const std::string &input, const std::string& sname)
{
    std::istringstream iss(input);
    return parse_stream(iss, sname);
}

void Driver::error(const class location& l,
		   const std::string& m)
{
    std::cerr << l << ": " << m << std::endl;
}

void Driver::error(const std::string& m)
{
    std::cerr << m << std::endl;
}

//// control m_plFlag
//void Driver::setPlFlag(bool flag) 
//{
//    m_plFlag = flag;
//}

// .nodes file 
void Driver::nodeEntryCbk(std::string& name, std::string& type)
{
    m_db.add_bookshelf_node(name, type);
}
//void Driver::numNodeTerminalsCbk(int nn, int nt)
//{
//    m_db.resize_bookshelf_node_terminals(nn, nt);
//}
//void Driver::terminalEntryCbk(string& name, int w, int h)
//{
//    m_db.add_bookshelf_terminal(name, w, h);
//}
//void Driver::terminalNIEntryCbk(string& name, int w, int h)
//{
//    m_db.add_bookshelf_terminal_NI(name, w, h);
//}
//void Driver::nodeEntryCbk(string& name, int w, int h, string&)
//{
//    m_db.add_bookshelf_node(name, w, h);
//}
//void Driver::nodeEntryCbk(string& name, int w, int h)
//{
//    m_db.add_bookshelf_node(name, w, h);
//}

// .nets file 
void Driver::addNetCbk(const std::string& name, int n)
{
    m_net.net_name = name;
    m_net.vNetPin.clear();
    //m_net.vNetPin.reserve(n);
}
void Driver::addPinCbk(std::string& nodeName, std::string& pinName)
{
    m_net.vNetPin.push_back(NetPin(nodeName, pinName));
}
//void Driver::numNetCbk(int n)
//{
//    m_db.resize_bookshelf_net(n);
//}
//void Driver::numPinCbk(int n)
//{
//    m_db.resize_bookshelf_pin(n);
//}
//void Driver::netPinEntryCbk(string& node_name, char direct, double offsetX, double offsetY, double w, double h, string& pin_name)
//{
//    // not sure whether w or h has the correct meaning 
//    m_net.vNetPin.push_back(NetPin(node_name, direct, offsetX, offsetY, w, h, pin_name));
//}
//void Driver::netPinEntryCbk(string& node_name, char direct, double offsetX, double offsetY, double w, double h)
//{
//    // not sure whether w or h has the correct meaning 
//    m_net.vNetPin.push_back(NetPin(node_name, direct, offsetX, offsetY, w, h));
//}
//void Driver::netNameAndDegreeCbk(string& net_name, int degree)
//{
//    m_net.net_name.swap(net_name);
//    m_net.vNetPin.clear();
//    m_net.vNetPin.reserve(degree);
//}
void Driver::netEntryCbk()
{
    m_db.add_bookshelf_net(m_net);
    m_net.reset();
}

// .pl file 
void Driver::plNodeEntryCbk(const std::string& node_name, double x, double y, int z)
{
    m_db.set_bookshelf_node_pos(node_name, x, y, z);
}
//void Driver::plNodeEntryCbk(string& node_name, double x, double y, string& orient, string& status)
//{
//    m_db.set_bookshelf_node_position(node_name, x, y, orient, status, m_plFlag);
//}
//void Driver::plNodeEntryCbk(string& node_name, double x, double y, string& orient)
//{
//    m_db.set_bookshelf_node_position(node_name, x, y, orient, "", m_plFlag);
//}
// .scl file 
/// @brief from .scl file, xh and yh
void Driver::routeGridCbk(int numGridX, int numGridY)
{
    m_db.resize_sites(numGridX, numGridY);
} 
void Driver::setSiteTypeToSliceLCbk(int xIdx, int yIdx)
{
    m_db.site_info_update(xIdx, yIdx, 1);
}
void Driver::setSiteTypeToSliceMCbk(int xIdx, int yIdx)
{
    m_db.site_info_update(xIdx, yIdx, 1);
}
void Driver::setSiteTypeToDspCbk(int xIdx, int yIdx)
{
    m_db.site_info_update(xIdx, yIdx, 2);
}
void Driver::setSiteTypeToRamCbk(int xIdx, int yIdx)
{
    m_db.site_info_update(xIdx, yIdx, 3);
}
void Driver::setSiteTypeToIoCbk(int xIdx, int yIdx)
{
    m_db.site_info_update(xIdx, yIdx, 4);
}
void Driver::initClockRegionsCbk(int xReg, int yReg)
{
    m_db.resize_clk_regions(xReg, yReg);
}
void Driver::addClockRegionCbk(const std::string& name, int xl, int yl, int xh, int yh, int xm, int ym)
{
    m_db.add_clk_region(name, xl, yl, xh, yh, xm, ym);
}
//void Driver::sclNumRows(int n)
//{
//    m_db.resize_bookshelf_row(n);
//}
//void Driver::sclCoreRowStart(string const& orient)
//{
//    m_row.orient = orient;
//}
//void Driver::sclCoreRowCoordinate(int y)
//{
//    m_row.origin[1] = y;
//}
//void Driver::sclCoreRowHeight(int h)
//{
//    m_row.height = h;
//}
//void Driver::sclCoreRowSitewidth(int site_width)
//{
//    m_row.site_width = site_width;
//}
//void Driver::sclCoreRowSitespacing(int site_spacing)
//{
//    m_row.site_spacing = site_spacing;
//}
//void Driver::sclCoreRowSiteorient(int site_orient)
//{
//    m_row.site_orient = site_orient;
//}
//void Driver::sclCoreRowSiteorient(string& site_orient)
//{
//    m_row.site_orient_str.swap(site_orient);
//}
//void Driver::sclCoreRowSitesymmetry(int site_symmetry)
//{
//    m_row.site_symmetry = site_symmetry;
//}
//void Driver::sclCoreRowSitesymmetry(string& site_symmetry)
//{
//    m_row.site_symmetry_str.swap(site_symmetry);
//}
//void Driver::sclCoreRowSubRowOrigin(int x)
//{
//    m_row.origin[0] = x;
//}
//void Driver::sclCoreRowNumSites(int site_num)
//{
//    m_row.site_num = site_num;
//}
//void Driver::sclCoreRowEnd()
//{
//    m_db.add_bookshelf_row(m_row);
//    m_row.reset();
//}
// .wts file 
//void Driver::wtsNetWeightEntry(string& net_name, double weight)
//{
//    m_db.set_bookshelf_net_weight(net_name, weight);
//}
///// @brief from .route file, Vertical capacity per tile edge on each layer 
//void Driver::routeVerticalCapacityCbk(IntegerArray& vVerticalCapacity)
//{
//    m_routeInfo.vVerticalCapacity.swap(vVerticalCapacity);
//}
///// @brief from .route file, Horizontal capacity per tile edge on each layer 
//void Driver::routeHorizontalCapacityCbk(IntegerArray& vHorizontalCapacity)
//{
//    m_routeInfo.vHorizontalCapacity.swap(vHorizontalCapacity);
//} 
///// @brief from .route file, Minimum metal width on each layer 
//void Driver::routeMinWireWidthCbk(IntegerArray& vMinWireWidth)
//{
//    m_routeInfo.vMinWireWidth.swap(vMinWireWidth);
//}
///// @brief from .route file, Minimum spacing on each layer 
//void Driver::routeMinWireSpacingCbk(IntegerArray& vMinWireSpacing)
//{
//    m_routeInfo.vMinWireSpacing.swap(vMinWireSpacing);
//}
///// @brief from .route file, Via spacing per layer 
//void Driver::routeViaSpacingCbk(IntegerArray& vViaSpacing)
//{
//    m_routeInfo.vViaSpacing.swap(vViaSpacing);
//} 
///// @brief from .route file, Absolute coordinates of the origin of the grid (grid_lowerleft_X grid_lowerleft_Y)
//void Driver::routeGridOriginCbk(double gridOriginX, double gridOriginY)
//{
//    m_routeInfo.gridOrigin[0] = gridOriginX;
//    m_routeInfo.gridOrigin[1] = gridOriginY;
//}
///// @brief from .route file, tile_width tile_height 
//void Driver::routeTileSizeCbk(double tileSizeX, double tileSizeY)
//{
//    m_routeInfo.tileSize[0] = tileSizeX; 
//    m_routeInfo.tileSize[1] = tileSizeY; 
//} 
///// @brief from .route file, Porosity for routing blockages
///// (Zero implies the blockage completely blocks overlapping routing tracks. Default = 0)
//void Driver::routeBlockagePorosityCbk(int bp)
//{
//    m_routeInfo.blockagePorosity = bp;
//} 
///// @brief from .route file, number of IO pins  
//void Driver::routeNumNiTerminalsCbk(int n)
//{
//    if (!m_routeInfo.vVerticalCapacity.empty() || !m_routeInfo.vHorizontalCapacity.empty())
//    {
//        m_db.set_bookshelf_route_info(m_routeInfo); 
//        m_routeInfo.reset(); 
//    }
//    m_db.resize_bookshelf_niterminal_layers(n); 
//} 
///// @brief from .route file, for IO pins, (node_name layer_id_for_all_node_pins) 
//void Driver::routePinLayerCbk(string& name, int layer)
//{
//    // convert to name 
//    char buf[64]; 
//    sprintf(buf, "%d", layer);
//    string layerName (buf); 
//    m_db.add_bookshelf_niterminal_layer(name, layerName);
//} 
//void Driver::routePinLayerCbk(string& name, string& layer)
//{
//    m_db.add_bookshelf_niterminal_layer(name, layer);
//} 
///// @brief from .route file, number of blockage nodes
//void Driver::routeNumBlockageNodes(int n)
//{
//    if (!m_routeInfo.vVerticalCapacity.empty() || !m_routeInfo.vHorizontalCapacity.empty())
//    {
//        m_db.set_bookshelf_route_info(m_routeInfo); 
//        m_routeInfo.reset(); 
//    }
//    m_db.resize_bookshelf_blockage_layers(n);
//} 
///// @brief from .route file, for blockages, (node_name num_blocked_layers list_of_blocked_layers) 
//void Driver::routeBlockageNodeLayerCbk(string& name, int, IntegerArray& vLayer)
//{
//    // convert to name 
//    std::vector<std::string> vLayerName (vLayer.size()); 
//    for (unsigned int i = 0; i < vLayer.size(); ++i)
//    {
//        char buf[64]; 
//        sprintf(buf, "%d", vLayer[i]); 
//        vLayerName[i] = buf; 
//    }
//    m_db.add_bookshelf_blockage_layers(name, vLayerName);
//} 
//void Driver::routeBlockageNodeLayerCbk(string& name, int, StringArray& vLayer)
//{
//    m_db.add_bookshelf_blockage_layers(name, vLayer);
//} 
// .lib file 
void Driver::addCellCbk(const std::string& name)
{
    m_db.add_lib_cell(name);
}
void Driver::addCellInputPinCbk(std::string& pName)
{
    m_db.add_input_pin(pName);
}
void Driver::addCellOutputPinCbk(std::string& pName)
{
    m_db.add_output_pin(pName);
}
void Driver::addCellClockPinCbk(std::string& pName)
{
    m_db.add_clk_pin(pName);
}
void Driver::addCellCtrlPinCbk(std::string& pName)
{
    m_db.add_ctrl_pin(pName);
}
// .aux file 
void Driver::auxCbk(std::string& design_name, vector<std::string>& vBookshelfFiles)
{
    m_db.set_bookshelf_design(design_name);
    m_vBookshelfFiles.swap(vBookshelfFiles);
    //std::cout << "Design name : " << design_name << std::endl;
    //std::cout << "Total bookshelf files: " << vBookshelfFiles.size() << std::endl;
}
void Driver::setLibFileCbk(const std::string &str)
{
    //std::cout << "Lib File: " << str << std::endl; 
    m_libFile = str;
    m_vBookshelfFiles.push_back(str);
}

void Driver::setSclFileCbk(const std::string &str)
{
    //std::cout << "Scl File: " << str << std::endl; 
    m_sclFile = str;
    m_vBookshelfFiles.push_back(str);
}

void Driver::setNodeFileCbk(const std::string &str)
{
    //std::cout << "Node File: " << str << std::endl; 
    m_nodeFile = str;
    m_vBookshelfFiles.push_back(str);
}

void Driver::setNetFileCbk(const std::string &str)
{
    //std::cout << "Net File: " << str << std::endl; 
    m_netFile = str;
    m_vBookshelfFiles.push_back(str);
}

void Driver::setPlFileCbk(const std::string &str)
{
    //std::cout << "Pl File: " << str << std::endl; 
    m_plFile = str;
    m_vBookshelfFiles.push_back(str);
}

void Driver::setWtFileCbk(const std::string &str)
{
    //std::cout << "Wt File: " << str << std::endl; 
    m_wtFile = str;
    m_vBookshelfFiles.push_back(str);
}

/// a local helper function 
struct SortByPairFirst
{
    inline bool operator()(std::pair<int, int> const& p1, std::pair<int, int> const& p2) const 
    {
        return p1.first < p2.first; 
    }
};

bool read(BookshelfDataBase& db, const std::string& auxFile)
{
    // first read .aux 
	BookshelfParser::Driver driverAux (db);
	//driver.trace_scanning = true;
	//driver.trace_parsing = true;

    bool gzFlag = limbo::iequals(limbo::get_file_suffix(auxFile), "gz"); // compressed or not 
#if ZLIB == 0
    if (gzFlag)
    {
        std::cerr << "compile with ZLIB_DIR defined to read .gz files\n";
        return false; 
    }
#endif

    std::ifstream ifs;
    ifs.open(auxFile);
    if (!ifs.good())
    {
        std::cerr << "Cannot open file " << auxFile << std::endl;
        return false;
    }
    //std::cout << "Parsing File " << auxFile << std::endl;
    driverAux.parse_stream(ifs);
    ifs.close();

	//bool flagAux = driverAux.parse_file(auxFile);
    //if (!flagAux)
    //    return false;

    std::string auxPath = limbo::get_file_path(auxFile);
    //std::cout << "There are " << std::to_string(driverAux.bookshelfFiles().size()) << " files in bookshelf" << std::endl;

    // (visit_order, index)
    //vector<std::pair<int, int> > vOrder (driverAux.bookshelfFiles().size());
    //for (unsigned i = 0; i < vOrder.size(); ++i)
    //{
    //    std::string const& filename = driverAux.bookshelfFiles().at(i);
    //    std::string suffix = limbo::get_file_suffix(filename);
    //    if (limbo::iequals(suffix, "scl"))
    //        vOrder[i].first = 0;
    //    else if (limbo::iequals(suffix, "nodes"))
    //        vOrder[i].first = 1;
    //    else if (limbo::iequals(suffix, "nets"))
    //        vOrder[i].first = 2;
    //    else if (limbo::iequals(suffix, "wts"))
    //        vOrder[i].first = 3;
    //    else if (limbo::iequals(suffix, "pl"))
    //        vOrder[i].first = 4;
    //    else if (limbo::iequals(suffix, "lib"))
    //        vOrder[i].first = 5;
    //    else 
    //        vOrder[i].first = vOrder.size();
    //    vOrder[i].second = i;
    //}
    //// order by visit_order 
    //std::sort(vOrder.begin(), vOrder.end(), SortByPairFirst());

    // start parsing 
    //for (vector<std::pair<int, int> >::const_iterator it = vOrder.begin(); it != vOrder.end(); ++it)
    //{
    //    std::pair<int, int> const& order = *it;
    //    std::string filename = driverAux.bookshelfFiles().at(order.second);
    //    if (gzFlag && !limbo::iequals(limbo::get_file_suffix(filename), "gz"))
    //    {
    //        filename += ".gz";
    //    }

    //    Driver driver (db);
    //    bool flag = driver.parse_file(auxPath + "/" + filename);
    //    std::cout << "Reading File " << auxPath << "/" << filename << " " << flag << std::endl;
    //    if (!flag)
    //        return false;
    //}
    for (auto file : {driverAux.libFile(), driverAux.sclFile(), driverAux.nodeFile(), driverAux.plFile(), driverAux.netFile()})
    {
        std::string path = auxPath + "/" + file;
        std::cout << "Parsing File " << path << std::endl;
        ifs.open(path);
        if (!ifs.good())
        {
            std::cerr << "Cannot open file " << path << std::endl;
            return false;
        }
        driverAux.parse_stream(ifs);
        ifs.close();
    }

    //Parse bookshelf files in order

    // inform database that parsing is completed 
    db.bookshelf_end(); 

    return true;
}

///// read .pl file only, the callback only provide positions and orientation 
//bool readPl(BookshelfDataBase& db, const string& plFile)
//{
//    // start parsing 
//    string const& filename = plFile;
//
//    Driver driver (db);
//    driver.setPlFlag(true);
//    bool flag = driver.parse_file(filename);
//    if (!flag)
//        return false;
//    // do not inform the ending 
//    // inform database that parsing is completed 
//    //db.bookshelf_end(); 
//
//    return true;
//}


} // namespace example
