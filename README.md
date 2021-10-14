# DREAMPlaceFPGA
An Open-Source Analytical Placer for Large Scale Heterogeneous FPGAs using Deep-Learning Toolkit.

This work leverages the open-source ASIC placement framework, [DREAMPlace](https://github.com/limbo018/DREAMPlace), to build an open-source placement framework for FPGAs that is based on the [elfPlace](https://ieeexplore.ieee.org/document/8942075) algorithm. On the [ISPD'2016 benchmark suite](http://www.ispd.cc/contests/16/FAQ.html), *DREAMPlaceFPGA* is `5.4×` faster for global placement and `1.8×` faster for overall placement than [elfPlace (CPU)](https://ieeexplore.ieee.org/document/8942075), with similar quality of results. In addition, *DREAMPlaceFPGA* outperforms [elfPlace (GPU)](https://ieeexplore.ieee.org/document/9330804) by `19%` for global placement. For more details, please refer to the ['paper'](#citation).

Among the various placement stages: global placement (GP), legalization (LG), and detailed placement (DP), only the global placement stage is accelerated using *DREAMPlaceFPGA*. The [elfPlace (CPU)](thirdparty/elfPlace_LG_DP) binary is used to run the legalization and detailed placement stages. Currently, *DREAMPlaceFPGA* only supports the [ISPD'2016 benchmarks](http://www.ispd.cc/contests/16/FAQ.html), which employs the Xilinx Ultrascale architecture. *DREAMPlaceFPGA* runs on both CPU and GPU. If installed on a machine without GPU, multi-threaded CPU support is available.

* *DREAMPlaceFPGA* Placement Flow
<p align="center">
    <img src=/images/FPGA_placement.png width=40% height=40%>
</p>

## Developer(s)

- Rachel Selina Rajarathnam, [UTDA](https://www.cerc.utexas.edu/utda), ECE Department, The University of Texas at Austin

## External Dependencies

- Python 2.7 or Python 3.5/3.6/3.7

- [CMake](https://cmake.org) version 3.8.2 or later

- [Pytorch](https://pytorch.org/) 1.0.0
    - Other version around 1.0.0 may also work, but not tested

- [GCC](https://gcc.gnu.org/)
    - Recommend GCC 5.1 or later. 
    - Other compilers may also work, but not tested. 

- [cmdline](https://github.com/tanakh/cmdline)
    - a command line parser for C++

- [Flex](http://flex.sourceforge.net)
    - lexical analyzer employed in the bookshelf parser

- [Bison](https://www.gnu.org/software/bison)
    - parser generator employed in the bookshelf parser

- [Boost](https://www.boost.org)
    - Need to install and visible for linking

- [Limbo](https://github.com/limbo018/Limbo)
    - Integrated as a git submodule

- [Flute](https://doi.org/10.1109/TCAD.2007.907068)
    - Integrated as a submodule

- [CUB](https://github.com/NVlabs/cub)
    - Integrated as a git submodule

- [munkres-cpp](https://github.com/saebyn/munkres-cpp)
    - Integrated as a git submodule

- [CUDA 9.1 or later](https://developer.nvidia.com/cuda-toolkit) (Optional)
    - If installed and found, GPU acceleration will be enabled. 
    - Otherwise, only CPU implementation is enabled. 

- GPU architecture compatibility 6.0 or later (Optional)
    - Code has been tested on GPUs with compute compatibility 6.0, 7.0, and 7.5. 
    - Please check the [compatibility](https://developer.nvidia.com/cuda-gpus) of the GPU devices. 
    - The default compilation target is compatibility 6.0. This is the minimum requirement and lower compatibility is not supported for the GPU feature. 
    - For compatibility 7.0, it is necessary to set the CMAKE_CUDA_FLAGS to -gencode=arch=compute_70,code=sm_70. 

- [Cairo](https://github.com/freedesktop/cairo) (Optional)
    - If installed and found, the plotting functions will be faster by using C/C++ implementation. 
    - Otherwise, python implementation is used. 


## Cloning the repository

To pull git submodules in the root directory
```
git submodule init
git submodule update
```

Or alternatively, pull all the submodules when cloning the repository. 
```
git clone --recursive https://github.com/rachelselinar/DREAMPlaceFPGA.git
```

## Build Instructions

### To install Python dependency 

At the root directory:
```
pip install -r requirements.txt 
```

### To Build 

At the root directory, 
```
mkdir build 
cd build 
cmake .. -DCMAKE_INSTALL_PREFIX=your_install_path
make 
make install
```
Third party submodules are automatically built except for [Boost](https://www.boost.org).

To clean, go to the root directory. 
```
rm -r build
```
### Cmake Options 

Here are the available options for CMake. 
- CMAKE_INSTALL_PREFIX: installation directory
    - Example ```cmake -DCMAKE_INSTALL_PREFIX=path/to/your/directory```
- CMAKE_CUDA_FLAGS: custom string for NVCC (default -gencode=arch=compute_60,code=sm_60)
    - Example ```cmake -DCMAKE_CUDA_FLAGS=-gencode=arch=compute_60,code=sm_60```
- CMAKE_CXX_ABI: 0|1 for the value of _GLIBCXX_USE_CXX11_ABI for C++ compiler, default is 0. 
    - Example ```cmake -DCMAKE_CXX_ABI=0```
    - It must be consistent with the _GLIBCXX_USE_CXX11_ABI for compling all the C++ dependencies, such as Boost and PyTorch. 
    - PyTorch in default is compiled with _GLIBCXX_USE_CXX11_ABI=0, but in a customized PyTorch environment, it might be compiled with _GLIBCXX_USE_CXX11_ABI=1. 


## Sample Benchmarks

*DREAMPlaceFPGA* only supports designs for *Xilinx Ultrascale Architecture* in bookshelf format with fixed IOs.
Refer to [ISPD'2016 contest](http://www.ispd.cc/contests/16/FAQ.html) for more information.

Four sample designs are included in `benchmarks` directory.

## Running DREAMPlaceFPGA

Before running, ensure that all python dependent packages have been installed. 
Go to the **install directory** and run with JSON configuration file.  
```
python dreamplacefpga/Placer.py test/FPGA-example1.json
```

Unitests for some of the pytorch operators are provided. To run:
```
python unitest/ops/hpwl_unitest.py
```

### JSON Configurations

The most frequently used options in the JSON file are listed below. For the complete list of available options, please refer to [paramsFPGA.json](./dreamplacefpga/paramsFPGA.json). 

| JSON Parameter                   | Default                 | Description                                                                                                                                                       |
| -------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| aux_input                        | required for bookshelf  | input .aux file                                                                                                                                                   |
| gpu                              | 1                       | enable gpu or not                                                                                                                                                 |
| num_threads                      | 8                       | number of CPU threads                                                                                                                                             |
| num_bins_x                       | 512                     | number of bins in horizontal direction                                                                                                                            |
| num_bins_y                       | 512                     | number of bins in vertical direction                                                                                                                              |
| global_place_stages              | required                | global placement configurations of each stage, a dictionary of {"num_bins_x", "num_bins_y", "iteration", "learning_rate"}, learning_rate is relative to bin size  |
| density_weight                   | 1.0                     | initial weight of density cost                                                                                                                                    |
| gamma                            | 0.5                     | initial coefficient for log-sum-exp and weighted-average wirelength                                                                                               |
| random_seed                      | 1000                    | random seed                                                                                                                                                       |
| scale_factor                     | 0.0                     | scale factor to avoid numerical overflow; 0.0 means not set                                                                                                       |
| result_dir                       | results                 | result directory for output                                                                                                                                       |
| global_place_flag                | 1                       | whether use global placement                                                                                                                                      |
| legalize_and_detailed_place_flag | 1                       | whether to run legalization and detailed placement using elfPlace                                                                                                 |
| dtype                            | float32                 | data type, float32 (or) float64                                                                                                                                   |
| plot_flag                        | 0                       | whether plot solution or not                                                                                                                                      |

## Bug Report

Please report bugs to [rachelselina dot r at utexas dot edu](mailto:rachelselina.r@utexas.edu).

## Publication(s) 

* Rachel Selina Rajarathnam, Mohamed Baker Alawieh, Zixuan Jiang, Mahesh A. Iyer, and [David Z. Pan](http://users.ece.utexas.edu/~dpan), 
  "**DREAMPlaceFPGA: An Open-Source Analytical Placer for Large Scale Heterogeneous FPGAs using Deep-Learning Toolkit**", 
  IEEE/ACM Asian and South Pacific Design Automation Conference (ASP-DAC), Jan 17-20, 2022 (accepted)

## Copyright

This software is released under *BSD 3-Clause "New" or "Revised" License*. Please refer to [LICENSE](./LICENSE) for details.

