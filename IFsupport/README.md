## [FPGA Interchange Format](https://fpga-interchange-schema.readthedocs.io)
The Interchange Format (IF) is a common format for describing FPGA architecture that was developed by AntMicro, Google, and Xilinx/AMD. This format allows interoperability between open-source and proprietary FPGA tools.
The IF format uses a set of schema to describe
- Device Resources (*.device): Information about the FPGA architecture including site map and routing resources.
- Logical Netlist (*.netlist): Represents the logic mapped synthesized netlist before placement and routing. 
- Physical Netlist (*.phys): Represents both placed and routed designs. 

## IF Support for Ultrascale+ architecture in ``DREAMPlaceFPGA``
 > ***Note***: ``DREAMPlaceFPGA`` only supports Interchange Format for the Ultrascale+ part ``xcvu3p-ffvc1517-2-e``.
The GNL designs used are available at: [https://github.com/Xilinx/RapidWright/releases/download/v2022.2.1-beta/gnl_timing_designs.zip](https://github.com/Xilinx/RapidWright/releases/download/v2022.2.1-beta/gnl_timing_designs.zip)

The figure below shows the steps involved in running a GNL design on ``DREAMPlaceFPGA``.
<p align="center">
    <img src=/images/IF_support.png height="500">
</p>

### I: Generate the bookshelf files from the Interchange Format (IF) ``*.netlist``
    
Example usage:
``
python IFsupport/IF2bookshelf.py
--netlist benchmarks/IF_netlist/gnl_2_4_3_1.3_gnl_3000_07_3_80_80.netlist
``
  
The bookshelf files are generated in the **benchmarks/IF2bookshelf** directory

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

 > ***Note***: The ``*.device`` is a very large file and has not been included. Please use RapidWright to generate the Interchange Format ``*.device`` file from the part name
`` 
java com.xilinx.rapidwright.interchange.DeviceResourcesExample xcvu3p-ffvc1517-2-e
``
This command generates the **xcvu3p-ffvc1517-2-e.device** IF file. 
**Move the ``*.device`` file to the 'IFsupport' directory.**

####  Generating ``design.pl`` bookshelf file for fixed IO instances in the design:

 If the design contains IO instances, their fixed locations are provided using the ``design.pl`` bookshelf input file.
Follow the below steps to generate the ``design.pl`` file from Vivado:

-  Run placement in Vivado for the design using the command: 
``place_design``

-  Use the TCL script to extract the design’s IO locations in Vivado using the command: 
``source IFsupport/get_io_sites.tcl``
The IO placement information is written to **io_placement.txt**.

- Generate the ``design.pl`` file using the **io_placement.txt** file obtained from Vivado, using:
`` python IFsupport/create_fixed_io.py --io_place io_placement.txt
``
  
### II: Run placement in ``DREAMPlaceFPGA``    

``DREAMPlaceFPGA`` uses a ***.json** file to read in the bookshelf inputs as well as user-defined parameters.

The json files for the sample GNL designs in the ``benchmarks/IF_netlist`` folder are already provided in the ``test`` directory. For other designs, the user can create a similar json file.

 #### IF related json parameters
The options that are relevant to Interchange Format (IF) in the JSON file are listed below:

| JSON Parameter                   | Default                 | Description                                                                                                                                                       |
| -------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| aux_input                        | required for bookshelf  | input .aux file                                                                                                                                                   |
| enable_if                              | 0                       | enable Interchange Format (IF) writer to generate ``*.phys`` for the placed design                                                                                                                             |
| part_name                      | xcvu3p-ffvc1517-2-e                       | Only Ultrascale+ device ``xcvu3p-ffvc1517-2-e`` is supported for Interchange Format writer                                                                                                                                             |

The user can update the aux_input location to point to the generated bookshelf files. For example,``benchmarks/IF2bookshelf/gnl_2_4_3_1.3_gnl_3000_07_3_80_80/design.aux`` 

#### Run placement:
``
python dreamplacefpga/Placer.py test/gnl_2_4_3_1.3_gnl_3000_07_3_80_80.json
``
 
The placement results can be found at **results/design/design.final.pl**

If IF writer is enabled, the placement solution is also written out in Interchange Format (IF) as a physical netlist at **results/design/design.phys**

### III: Use RapidWright to generate the ``design.dcp`` using the IF ``design.phys``
    
In order to route the design in Vivado, the IF **design.phys** can be used to obtain a design check point or dcp.

In RapidWright, use the following command to generate the dcp with placement solution:
``
java com.xilinx.rapidwright.interchange.PhysicalNetlistToDcp design.netlist design.phys constr.xdc design.dcp
``
  
The ***.xdc** file is used to provide constraints, such as timing and placement, to Vivado. We use an empty ***.xdc** file for the dcp generation.

> For the GNL designs:
> -   As the GNL designs do not have any IO instances, they result in critical warnings during routing in Vivado. To avoid these critical warnings, update RapidWright before generating the dcp
> -   In RapidWright, add the below lines to **PhysicalNetlistToDcp.java** and recompile
``
roundtrip.setAutoIOBuffers(false);
roundtrip.setDesignOutofContext(true);
``

### IV:  Running routing in Vivado  

-   Read in the design to Vivado using ``open_checkpoint design.dcp``
    
-   Route the design in Vivado using ``route_design``
    
-   After routing the design, a dcp can be generated using ``write_checkpoint design_routed.dcp``
    
-   If required, the Interchange format ``*.phys`` can be generated using the **design_routed.dcp** in RapidWright using the below command:
  ``  
java com.xilinx.rapidwright.interchange.PhysicalNetlistExample design_routed.dcp design.edf design_new.dcp
``
This command generates the **design_routed.phys** and the **design_routed.netlist** IF files.
