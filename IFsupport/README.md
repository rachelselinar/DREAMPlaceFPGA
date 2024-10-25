## [FPGA Interchange Format](https://fpga-interchange-schema.readthedocs.io)
The Interchange Format (IF) is a common format for describing FPGA architecture that was developed by AntMicro, Google, and Xilinx/AMD. This format allows interoperability between open-source and proprietary FPGA tools.
The IF format uses a set of schema to describe
- Device Resources (*.device): Information about the FPGA architecture including site map and routing resources.
- Logical Netlist (*.netlist): Represents the logic mapped synthesized netlist before placement and routing.
- Physical Netlist (*.phys): Represents both placed and routed designs.


## IF Support in ``DREAMPlaceFPGA``

>  ***Note***: ``DREAMPlaceFPGA`` supports Interchange Format for most Ultrascale and Ultrascale+ parts. The GNL designs used are available at: [https://github.com/Xilinx/RapidWright/releases/download/v2022.2.1-beta/gnl_timing_designs.zip](https://github.com/Xilinx/RapidWright/releases/download/v2022.2.1-beta/gnl_timing_designs.zip)


The figure below shows the steps involved in running a GNL design on ``DREAMPlaceFPGA``.
<p align="center">
    <img src=/images/IF_support_upd.png height="500">
</p>

## <a name="dependencies"></a>External Dependencies
  
- [CapnProto](https://capnproto.org/) 
	- Cap’n Proto C++ tools for interchange format
	- Tested on version 1.0.2

- [RapidWright](https://www.rapidwright.io/) 
	- Generates *.device and *.netlist interchange files, convert interchange output to .*dcp

 ## <a name="steps"></a>Interchange flow

### I: Generate Interchange Format (IF) files

#### Generating the Interchange format ``*.netlist`` and ``*.device`` using Vivado and RapidWright

 -   From Vivado, generate an ``*.edf`` for the design. The ``*.edf`` file can also be generated from a design check point (dcp).
   To generate ``*.edf`` file from Vivado, use the command 
   ``
   write_edif design.edf
   ``

  -   Use RapidWright to generate the Interchange Format ``*.netlist`` file from **design.edf**
``    
java com.xilinx.rapidwright.interchange.LogicalNetlistExample design.edf
``
This command generates the **design.netlist** IF file.


>  ***Note***: The ``*.device`` is a very large file and has not been included. Please use RapidWright to generate the Interchange Format ``*.device`` file from the part name
``java com.xilinx.rapidwright.interchange.DeviceResourcesExample xcvu3p-ffvc1517-2-e``
> This command generates the **xcvu3p-ffvc1517-2-e.device** IF file.

###  (Optional) Generating ``design.pl`` bookshelf file for fixed IO instances in the design:
If the design contains IO instances, their fixed locations are provided using the ``design.pl`` bookshelf input file.
Follow the below steps to generate the ``design.pl`` file from Vivado:
- Run placement in Vivado for the design using the command:
``place_design``

- Use the TCL script to extract the design’s IO locations in Vivado using the command:
``source IFsupport/get_io_sites.tcl`` The IO placement information is written to **io_placement.txt**.

- Generate the ``design.pl`` file using the **io_placement.txt** file obtained from Vivado, using:`` python IFsupport/create_fixed_io.py --io_place io_placement.txt``

### II: Run placement in ``DREAMPlaceFPGA``


``DREAMPlaceFPGA`` uses a ***.json** file to read in  user-defined parameters.

The json files for the sample GNL designs in the ``benchmarks/IF_netlist`` folder are already provided in the ``test_interchange`` directory. For other designs, the user can create a similar json file.

#### IF related json parameters

The options that are relevant to Interchange Format (IF) in the JSON file are listed below:
| JSON Parameter | Default | Description |
| -------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| interchange_netlist | required for interchange | input *.netlist file |
| interchange_device | required for interchange | input *.device file |
| enable_if | 0 | enable Interchange Format (IF) writer to generate ``*.phys`` for the placed design |
| enable_site_routing | 0 | enable Interchange Format (IF) writer to generate ``*.phys`` with intra-site routing |
| io_pl | required for designs with IO | bookshelf input *.pl with fixed IO locations |


#### Run placement:

``python dreamplacefpga/Placer.py test_interchange/gnl_2_4_3_1_xcvu3p.json``
The placement results can be found at **results/**  

#### Run placement (Bash option):

``bash bin/dreamplacefpga`` This bash scripts automatically generates a ***.json** file and launch placement.

**Usage** : bin/dreamplacefpga -DREAMPlaceFPGA_dir <DREAMPlaceFPGA_dir> -result_dir <result_dir> -interchange_netlist <interchange_netlist> -interchange_device <interchange_device> -gpu <gpu_flag>

### III: Use RapidWright to generate the ``design.dcp`` using the IF ``design.phys``

In order to route the design in Vivado, the IF **design.phys** can be used to obtain a design check point or dcp.
In RapidWright, use the following command to generate the dcp with placement solution:

``java com.xilinx.rapidwright.interchange.PhysicalNetlistToDcp design.netlist design.phys constr.xdc design.dcp``

The ***.xdc** file is used to provide constraints, such as timing and placement, to Vivado. We use an empty ***.xdc** file for the dcp generation.
 
> For the GNL designs:
>  - As the GNL designs do not have any IO instances, they result in critical warnings during routing in Vivado. To avoid these critical warnings, use 
>  ``java com.xilinx.rapidwright.interchange.PhysicalNetlistToDcp --out_of_context``


### IV: Running routing in Vivado

- Read in the design to Vivado using ``open_checkpoint design.dcp``

- Route the design in Vivado using ``route_design``


### V: Common Issues

Segmentation fault when reading the ***.device** file, try 

``ulimit -s unlimited``