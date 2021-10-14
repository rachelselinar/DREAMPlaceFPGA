/**
 * @file   PybindPlaceDB.cpp
 * @author Yibo Lin (DREAMPlace), Rachel Selina Rajarathnam (DREAMPlaceFPGA)
 * @date   Mar 2021
 * @brief  Python binding for PlaceDB 
 */

#include "PyPlaceDB.h"

PYBIND11_MAKE_OPAQUE(std::vector<bool>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> >);
//PYBIND11_MAKE_OPAQUE(std::vector<long>);
//PYBIND11_MAKE_OPAQUE(std::vector<unsigned long>);
//PYBIND11_MAKE_OPAQUE(std::vector<float>);
//PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
//PYBIND11_MAKE_OPAQUE(std::vector<std::pair<int, int>>);

PYBIND11_MAKE_OPAQUE(DREAMPLACE_NAMESPACE::PlaceDB::string2index_map_type);

PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>>);
//PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::Pin>);
//PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::Node>);
//PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::Net>);
PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::LibCell>);
//PYBIND11_MAKE_OPAQUE(std::vector<DREAMPLACE_NAMESPACE::clk_region>);

void bind_PlaceDB(pybind11::module& m) 
{
    pybind11::bind_vector<std::vector<bool> >(m, "VectorBool");
    pybind11::bind_vector<std::vector<double> >(m, "VectorCoordinate", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> >(m, "VectorIndex", pybind11::buffer_protocol());
    //pybind11::bind_vector<std::vector<std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> > >(m, "2DVectorIndex", pybind11::buffer_protocol());
    //pybind11::bind_vector<std::vector<long> >(m, "VectorLong", pybind11::buffer_protocol());
    //pybind11::bind_vector<std::vector<unsigned long> >(m, "VectorULong", pybind11::buffer_protocol());
    //pybind11::bind_vector<std::vector<float> >(m, "VectorFloat", pybind11::buffer_protocol());
    //pybind11::bind_vector<std::vector<double> >(m, "VectorDouble", pybind11::buffer_protocol());
    pybind11::bind_vector<std::vector<std::string> >(m, "VectorString");

    pybind11::bind_map<DREAMPLACE_NAMESPACE::PlaceDB::string2index_map_type>(m, "MapString2Index");

    // DREAMPLACE_NAMESPACE::Object.h
    pybind11::class_<DREAMPLACE_NAMESPACE::Object> (m, "Object")
        .def(pybind11::init<>())
        .def("id", &DREAMPLACE_NAMESPACE::Object::id)
        .def("__str__", &DREAMPLACE_NAMESPACE::Object::toString)
        ;

    // Box.h
    pybind11::class_<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>> (m, "BoxCoordinate")
        .def(pybind11::init<>())
        .def(pybind11::init<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::coordinate_type>())
        .def("xl", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::xl)
        .def("yl", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::yl)
        .def("xh", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::xh)
        .def("yh", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::yh)
        .def("width", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::width)
        .def("height", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::height)
        .def("area", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::area)
        .def("__str__", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>::toString)
        ;
    pybind11::class_<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>> (m, "BoxIndex")
        .def(pybind11::init<>())
        .def(pybind11::init<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::coordinate_type, DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::coordinate_type>())
        .def("xl", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::xl)
        .def("yl", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::yl)
        .def("xh", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::xh)
        .def("yh", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::yh)
        .def("width", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::width)
        .def("height", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::height)
        .def("area", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::area)
        .def("__str__", &DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>::toString)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::coordinate_type>> >(m, "VectorBoxCoordinate");
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::Box<DREAMPLACE_NAMESPACE::Object::index_type>> >(m, "VectorBoxIndex");

    // DREAMPLACE_NAMESPACE::LibCell.h
    pybind11::class_<DREAMPLACE_NAMESPACE::LibCell, DREAMPLACE_NAMESPACE::Object> (m, "LibCell")
        .def(pybind11::init<>())
        .def("name", &DREAMPLACE_NAMESPACE::LibCell::name)
        .def("id", &DREAMPLACE_NAMESPACE::LibCell::id)
        .def("inputPinArray", (std::vector<std::string> const& (DREAMPLACE_NAMESPACE::LibCell::*)() const) &DREAMPLACE_NAMESPACE::LibCell::inputPinArray)
        .def("outputPinArray", (std::vector<std::string> const& (DREAMPLACE_NAMESPACE::LibCell::*)() const) &DREAMPLACE_NAMESPACE::LibCell::outputPinArray)
        .def("clkPinArray", (std::vector<std::string> const& (DREAMPLACE_NAMESPACE::LibCell::*)() const) &DREAMPLACE_NAMESPACE::LibCell::clkPinArray)
        .def("ctrlPinArray", (std::vector<std::string> const& (DREAMPLACE_NAMESPACE::LibCell::*)() const) &DREAMPLACE_NAMESPACE::LibCell::ctrlPinArray)
        .def("libCellPinName2Type", (DREAMPLACE_NAMESPACE::LibCell::string2index_map_type const& (DREAMPLACE_NAMESPACE::LibCell::*)() const) &DREAMPLACE_NAMESPACE::LibCell::libCellPinName2Type)
        ;
    pybind11::bind_vector<std::vector<DREAMPLACE_NAMESPACE::LibCell> >(m, "VectorLibCell");

    // DREAMPLACE_NAMESPACE::PlaceDB.h
    pybind11::class_<DREAMPLACE_NAMESPACE::PlaceDB> (m, "PlaceDB")
        .def(pybind11::init<>())
        .def("movNodeNames", (std::vector<std::string> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::movNodeNames)
        .def("movNodeName", (std::string const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::movNodeName)
        .def("movNodeTypes", (std::vector<std::string> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::movNodeTypes)
        .def("movNodeType", (std::string const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::movNodeType)
        .def("movNodeXLocs", (std::vector<double> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::movNodeXLocs)
        .def("movNodeX", (double const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::movNodeX)
        .def("movNodeYLocs", (std::vector<double> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::movNodeYLocs)
        .def("movNodeY", (double const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::movNodeY)
        .def("movNodeZLocs", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::movNodeZLocs)
        .def("movNodeZ", (DREAMPLACE_NAMESPACE::PlaceDB::index_type const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::movNodeZ)
        .def("movNodeXSizes", (std::vector<double> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::movNodeXSizes)
        .def("movNodeXSize", (double const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::movNodeXSize)
        .def("movNodeYSizes", (std::vector<double> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::movNodeYSizes)
        .def("movNodeYSize", (double const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::movNodeYSize)
        .def("fixedNodeNames", (std::vector<std::string> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::fixedNodeNames)
        .def("fixedNodeName", (std::string const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::fixedNodeName)
        .def("fixedNodeTypes", (std::vector<std::string> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::fixedNodeTypes)
        .def("fixedNodeType", (std::string const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::fixedNodeType)
        .def("fixedNodeXLocs", (std::vector<double> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::fixedNodeXLocs)
        .def("fixedNodeX", (double const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::fixedNodeX)
        .def("fixedNodeYLocs", (std::vector<double> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::fixedNodeYLocs)
        .def("fixedNodeY", (double const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::fixedNodeY)
        .def("fixedNodeZLocs", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::fixedNodeZLocs)
        .def("fixedNodeZ", (DREAMPLACE_NAMESPACE::PlaceDB::index_type const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::fixedNodeZ)
        .def("node2FenceRegionMap", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::node2FenceRegionMap)
        .def("nodeFenceRegion", (DREAMPLACE_NAMESPACE::PlaceDB::index_type const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::nodeFenceRegion)
        .def("node2OutPinId", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::node2OutPinId)
        .def("node2PinCount", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::node2PinCount)
        .def("flopIndices", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::flopIndices)
        .def("lutTypes", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::lutTypes)
        .def("node2PinMap", (std::vector<std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> > const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::node2PinMap)
        .def("netNames", (std::vector<std::string> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::netNames)
        .def("netName", (std::string const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::netName)
        .def("net2PinCount", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::net2PinCount)
        .def("net2PinMap", (std::vector<std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> > const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::net2PinMap)
        .def("flatNet2PinMap", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::flatNet2PinMap)
        .def("flatNet2PinStartMap", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::flatNet2PinStartMap)
        .def("flatNode2PinMap", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::flatNode2PinMap)
        .def("flatNode2PinStartMap", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::flatNode2PinStartMap)
        .def("pinNames", (std::vector<std::string> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::pinNames)
        .def("pinName", (std::string const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::pinName)
        .def("pin2NetMap", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::pin2NetMap)
        .def("pin2NodeMap", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::pin2NodeMap)
        .def("pin2NodeTypeMap", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::pin2NodeTypeMap)
        .def("pinTypes", (std::vector<std::string> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::pinTypes)
        .def("pinTypeIds", (std::vector<DREAMPLACE_NAMESPACE::PlaceDB::index_type> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::pinTypeIds)
        .def("pinOffsetX", (std::vector<double> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::pinOffsetX)
        .def("pinOffsetY", (std::vector<double> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::pinOffsetY)
        .def("libCells", (std::vector<DREAMPLACE_NAMESPACE::LibCell> const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::libCells)
        .def("libCell", (DREAMPLACE_NAMESPACE::LibCell const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::libCell)
        .def("siteRows", &DREAMPLACE_NAMESPACE::PlaceDB::siteRows)
        .def("siteCols", &DREAMPLACE_NAMESPACE::PlaceDB::siteCols)
        .def("siteVal", (DREAMPLACE_NAMESPACE::PlaceDB::index_type const& (DREAMPLACE_NAMESPACE::PlaceDB::*)(DREAMPLACE_NAMESPACE::PlaceDB::index_type, DREAMPLACE_NAMESPACE::PlaceDB::index_type) const) &DREAMPLACE_NAMESPACE::PlaceDB::siteVal)
        .def("dieArea", &DREAMPLACE_NAMESPACE::PlaceDB::dieArea)
        .def("nodeName2Index", (DREAMPLACE_NAMESPACE::PlaceDB::string2index_map_type const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::nodeName2Index)
        .def("libCellName2Index", (DREAMPLACE_NAMESPACE::PlaceDB::string2index_map_type const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::libCellName2Index)
        .def("netName2Index", (DREAMPLACE_NAMESPACE::PlaceDB::string2index_map_type const& (DREAMPLACE_NAMESPACE::PlaceDB::*)() const) &DREAMPLACE_NAMESPACE::PlaceDB::netName2Index)
        .def("numMovable", &DREAMPLACE_NAMESPACE::PlaceDB::numMovable)
        .def("numFixed", &DREAMPLACE_NAMESPACE::PlaceDB::numFixed)
        .def("numLibCell", &DREAMPLACE_NAMESPACE::PlaceDB::numLibCell)
        .def("numLUT", &DREAMPLACE_NAMESPACE::PlaceDB::numLUT)
        .def("numFF", &DREAMPLACE_NAMESPACE::PlaceDB::numFF)
        .def("numDSP", &DREAMPLACE_NAMESPACE::PlaceDB::numDSP)
        .def("numRAM", &DREAMPLACE_NAMESPACE::PlaceDB::numRAM)
        .def("designName", &DREAMPLACE_NAMESPACE::PlaceDB::designName)
        .def("xl", &DREAMPLACE_NAMESPACE::PlaceDB::xl)
        .def("yl", &DREAMPLACE_NAMESPACE::PlaceDB::yl)
        .def("xh", &DREAMPLACE_NAMESPACE::PlaceDB::xh)
        .def("yh", &DREAMPLACE_NAMESPACE::PlaceDB::yh)
        .def("width", &DREAMPLACE_NAMESPACE::PlaceDB::width)
        .def("height", &DREAMPLACE_NAMESPACE::PlaceDB::height)
        ;
}

