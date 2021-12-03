##
# @file   configure.py
# @author Yibo Lin (DREAMPlace)
# @date   Feb 2020
# @brief  Record all configurations including compilation 
#

compile_configurations = {
        "CMAKE_CXX_COMPILER" : "/usr/bin/g++-6", 
        "CMAKE_CC_COMPILER" : "", 
        "CMAKE_BUILD_TYPE" : "Release", 
        "CMAKE_CXX_ABI" : "0", 
        "CMAKE_CXX_STANDARD" : "11", 
        "PYTHON" : "/home/local/eda02/python-3.7.4-venv/bin/python", 
        "Boost_DIR" : "", 
        "Boost_INCLUDE_DIRS" : "/usr/include", 
        "ZLIB_INCLUDE_DIRS" : "/usr/include", 
        "ZLIB_LIBRARIES" : "/usr/lib/x86_64-linux-gnu/libz.so", 
        "CUDA_FOUND" : "TRUE", 
        "CUDA_TOOLKIT_ROOT_DIR" : "/usr/local/cuda", 
        "CMAKE_CUDA_FLAGS" : "-arch=sm_60;-gencode=arch=compute_60,code=sm_60;-gencode=arch=compute_61,code=sm_61;-gencode=arch=compute_70,code=sm_70;-gencode=arch=compute_75,code=sm_75;-gencode=arch=compute_75,code=compute_75", 
        "CAIRO_FOUND" : "TRUE", 
        "CAIRO_INCLUDE_DIRS" : "/home/polaris/wuxili/common/libs/cairo/include/cairo", 
        "CAIRO_LIBRARIES" : "/home/polaris/wuxili/common/libs/cairo/lib/libcairo.so", 
        }

