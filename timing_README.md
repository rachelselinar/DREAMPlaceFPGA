
## Table of Contents

* [Timing-Driven FPGA Placer](#our_placer)
    - [Overview](#Timing-driven_Placement)
    - [Target Architecture](#target_arch)
* [Developers](#developers)
* [Publication](#publications)
* [Sample Benchmarks](#sample)
* [Running the timing-driven placer](#running)
* [JSON Configurations](#json)
	- [Timing Constraints](#timing_constraints)
* [Running Vivado](#vivado)
* [Copyright](#copyright)

# <a name="our_placer"></a>``Timing-Driven FPGA Placer``
The timing-driven placer is built on [``DREAMPlaceFPGA``](https://github.com/rachelselinar/DREAMPlaceFPGA), an Open-Source GPU-Accelerated Placer for Large Scale Heterogeneous FPGAs using a Deep Learning Toolkit.

### <a name="Timing-driven Placement"></a>Overview
The timing-driven placer includes timing-aware global placement (GP) and packing-legalization (LG) stages as illustrated:
<p align="center">
    <img src=/images/timing_driven_flow.png height="500">
</p>

The light-weight data-driven timing model employed in the timing-driven placer consists of logic and net delay values for the considered [architecture](./timing/ultrascale).

From the design, architecture and timing model information, the timing-driven placer constructs a timing graph for all timing paths as source-load pairs. In addition to reducing the overall wirelength and overlaps, the global placer also optimizes the timing paths once the instance overlaps are minimal.
During a timing GP iteration, the slacks for all the timing arcs are computed based on the current instance locations. Timing criticality is also considered during LUT/FF packing and legalization through cluster scoring functions.

The timing-driven placer runs on both CPU and GPU. The timing arc wirelength and timing preconditioner operators are accelerated on GPU. Please refer to [our paper](#publications) for more details.

### <a name="target_arch"></a>Target Architecture
The timing-driven placer primarily targets the [ISPD'2016 benchmarks](http://www.ispd.cc/contests/16/FAQ.html) on the simplified Xilinx Ultrascale architecture.

## <a name="developers"></a>Developers

- Zhili Xiong, [UTDA](https://www.cerc.utexas.edu/utda), ECE Department, The University of Texas at Austin
- Rachel Selina Rajarathnam, [UTDA](https://www.cerc.utexas.edu/utda), ECE Department, The University of Texas at Austin

## <a name="publications"></a>Publication

*	Z. Xiong, R. S. Rajarathnam, and D. Z. Pan, "A Data-Driven, Congestion-Aware and Open-Source  
Timing-Driven FPGA Placer Accelerated by GPUs," _The 32nd IEEE International Symposium on Field-Programmable Custom Computing Machines (FCCM)_, 2024. *(accepted)*

## <a name="sample"></a>Sample Benchmarks

IO instances are fixed and inputs are read in [Bookshelf](./benchmarks/sample_ispd2016_benchmarks/README) format.
[FPGA01](./benchmarks/sample_ispd2016_benchmarks/FPGA01/) design from the [ISPD'2016 benchmarks](http://www.ispd.cc/contests/16/FAQ.html) is included with the below additional files to enable interface with Vivado:

 - *design.edf*: This file contains design information can be generated by loading design.dcp to Vivado using `'write_edif'` command.
 - *constr_FPGA01.xdc*: This file contains the timing constraints for the design. The value is set to CPD + WS obtained by Vivado in Table II of the [paper](#publications).

Other sample Ultrascale benchmarks can be found in the [ispd2016 benchmarks](./benchmarks/sample_ispd2016_benchmarks)  directory.

## <a name="running"></a>Running the timing-driven placer

Before running, ensure that all python dependent packages have been installed. 
Go to the ***root directory*** and run with JSON configuration file.  
```
python dreamplacefpga/Placer.py <benchmark>.json
```
For example:
```
python dreamplacefpga/Placer.py test/timing_FPGA01.json
```
> Note: If your machine does not have an NVIDIA GPU, set the '***gpu***' flag in JSON configuration file to '***0***' to run on CPU.

### <a name="json"></a>JSON Configurations

The options corresponding to timing-driven placement in the JSON file are listed below. For the complete list of available options, please refer to [paramsFPGA.json](./dreamplacefpga/paramsFPGA.json). 

| JSON Parameter                   | Default                 | Description                                                                                                                                                       |
| -------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| timing_driven_flag               | 0                       | enable timing-driven placement                                                                                      |
| inflation_ratio                  | 1.0                     | used for better routability when export to Vivado router                                                                                                                             |
| timing_file_dir                  | required                | timing files for the data-driven timing model                                                                                                                             |
| enableTimingPreclustering        | 0                       | enable timing-driven preclustering in packing-legalization                                                                                                               |
| timing_constraint                | required                | the timing constraint in ps for our placer                                                                                                               |
| write_tcl_flag                   | 0                       | enable writing out placement solution as Vivado Tcl script                                                                                           |
| write_io_placement_flag          | 0                       | enable writing out fixed io placements as Vivado Tcl script       |

### <a name="timing_constraints"></a>Timing Constraints

| Benchmark                        | Timing Constraint (ps) | 
| -------------------------------- | ----------------------- | 
| FPGA01                           | 3600                    |
| FPGA02                           | 4100                    | 
| FPGA03                           | 8200                    | 
| FPGA04                           | 11200                   | 
| FPGA05                           | 19400                   | 
| FPGA06                           | 15300                   | 
| FPGA07                           | 25500                   | 
| FPGA08                           | 8800                    | 
| FPGA09                           | 23200                   | 
| FPGA10                           | 20300                   | 
| FPGA11                           | 18500                   | 
| FPGA12                           | 18700                   | 


## <a name="vivado"></a>Running Vivado
### <a name="Vivado Timing-driven Place-and-route"></a>Vivado Timing-driven Place-and-route
To run ISPD'2016 benchmarks on different Vivado versions, we need to load the fixed IO locations from the bookshelf design.pl before running place-and-route. Here is a example Tcl script for running Vivado timing-driven place-and-route.
The *place_io_cells.tcl* will be automatically generated when the ***write_io_placement_flag*** is set to be 1. 
```
read_edif benchmarks/sample_ispd2016_benchmarks/FPGA01/design.edf
link_design -part xcvu095-ffva2104-2-e
read_xdc benchmarks/sample_ispd2016_benchmarks/FPGA01/constr_FPGA01.xdc
source place_io_cells.tcl
place_design
route_design -directive AggressiveExplore
report_timing_summary
report_route_status
```

### <a name="Vivado Timing-driven Place-and-route"></a> Our  Timing-driven Placer with Vivado Router
We tested our placement solution after routed in Vivado2022.1, and got the timing report from Vivado2022.1 router. Here is a example Tcl script for loading our placment solution to Vivado and finish routing. 
The *place_cells.tcl* will be automatically generated when the ***write_tcl_flag*** is set to be 1.
 ```
read_edif benchmarks/sample_ispd2016_benchmarks/FPGA01/design.edf
link_design -part xcvu095-ffva2104-2-e
read_xdc benchmarks/sample_ispd2016_benchmarks/FPGA01/constr_FPGA01.xdc
source place_cells.tcl
route_design -directive AggressiveExplore
report_timing_summary
report_route_status
```

## <a name="copyright"></a>Copyright

This software is released under *BSD 3-Clause "New" or "Revised" License*. Please refer to [LICENSE](./LICENSE) for details.
