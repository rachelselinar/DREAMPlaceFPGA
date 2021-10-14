##
# @file   Placer.py
# @author Yibo Lin (DREAMPlace), Rachel Selina Rajarathnam (DREAMPlaceFPGA)
# @date   Sep 2020
# @brief  Main file to run the entire placement flow. 
#

import matplotlib 
matplotlib.use('Agg')
import os
import sys 
import time 
import numpy as np 
import logging
# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
	sys.path.append(root_dir)
import dreamplacefpga.configure as configure 
from Params import *
from PlaceDB import *
from NonLinearPlace import * 
import pdb 

def placeFPGA(params):
    """
    @brief Top API to run the entire placement flow. 
    @param params parameters 
    """
    assert (not params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
            "CANNOT enable GPU without CUDA compiled"

    np.random.seed(params.random_seed)
    # Read Database
    tt = time.time()
    placedb = PlaceDBFPGA()
    placedb(params) #Call function
    #logging.info("Reading database takes %.2f seconds" % (time.time()-tt))

    # Random Initial Placement 
    tt = time.time()
    placer = NonLinearPlaceFPGA(params, placedb)
    #logging.info("non-linear placement initialization takes %.2f seconds" % (time.time()-tt))
    metrics = placer(params, placedb)
    logging.info("Global Placement completed in %.2f seconds" % (time.time()-tt))

    # write placement solution 
    path = "%s/%s" % (params.result_dir, params.design_name())
    if not os.path.exists(path):
        os.system("mkdir -p %s" % (path))
    gp_out_file = os.path.join(path, "%s.gp.pl" % (params.design_name()))
    placedb.write(params, gp_out_file)

    # legalization and detailed placement using UTPlaceF 
    if params.legalize_and_detailed_place_flag and os.path.exists("thirdparty/elfPlace_LG_DP"):
        #elfPlace binary picks file named gp.pl in the current directory
        cp_cmd = "cp %s gp.pl" %(gp_out_file)
        os.system(cp_cmd)
        out_file = os.path.join(path, "%s_final.%s" % (params.design_name(), params.solution_file_suffix()))
        cmd = "./thirdparty/elfPlace_LG_DP --aux %s --numThreads %s --pl %s" % (params.aux_input, params.num_threads, out_file)
        logging.info("Legalization and Detailed Placement run using elfPlace (CPU): %s" % (cmd))
        tt = time.time()
        os.system(cmd)
        logging.info("Legalization and detailed placement completed in %.3f seconds" % (time.time()-tt))
    else:
        logging.warning("External legalization & detailed placement engine NOT found at thirdparty/elfPlace_LG_DP")

def place(params):
    """
    @brief Top API to run the entire placement flow. 
    @param params parameters 
    """

    assert (not params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
            "CANNOT enable GPU without CUDA compiled"

    np.random.seed(params.random_seed)
    # read database 
    tt = time.time()
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    #logging.info("reading database takes %.2f seconds" % (time.time()-tt))

    # solve placement 
    tt = time.time()
    placer = NonLinearPlace.NonLinearPlace(params, placedb)
    #logging.info("non-linear placement initialization takes %.2f seconds" % (time.time()-tt))
    metrics = placer(params, placedb)
    logging.info("non-linear placement takes %.2f seconds" % (time.time()-tt))

    # write placement solution 
    path = "%s/%s" % (params.result_dir, params.design_name())
    if not os.path.exists(path):
        os.system("mkdir -p %s" % (path))
    gp_out_file = os.path.join(path, "%s.gp.%s" % (params.design_name(), params.solution_file_suffix()))
    placedb.write(params, gp_out_file)

    # call external detailed placement
    # TODO: support more external placers, currently only support 
    # 1. NTUplace3/NTUplace4h with Bookshelf format 
    # 2. NTUplace_4dr with LEF/DEF format 
#    if params.detailed_place_engine and os.path.exists(params.detailed_place_engine):
#        logging.info("Use external detailed placement engine %s" % (params.detailed_place_engine))
#        if params.solution_file_suffix() == "pl" and any(dp_engine in params.detailed_place_engine for dp_engine in ['ntuplace3', 'ntuplace4h']): 
#            dp_out_file = gp_out_file.replace(".gp.pl", "")
#            # add target density constraint if provided 
#            target_density_cmd = ""
#            if params.target_density < 1.0 and not params.routability_opt_flag:
#                target_density_cmd = " -util %f" % (params.target_density)
#            cmd = "%s -aux %s -loadpl %s %s -out %s -noglobal %s" % (params.detailed_place_engine, params.aux_input, gp_out_file, target_density_cmd, dp_out_file, params.detailed_place_command)
#            logging.info("%s" % (cmd))
#            tt = time.time()
#            os.system(cmd)
#            logging.info("External detailed placement takes %.2f seconds" % (time.time()-tt))
#
#            if params.plot_flag: 
#                # read solution and evaluate 
#                placedb.read_pl(params, dp_out_file+".ntup.pl")
#                iteration = len(metrics)
#                pos = placer.init_pos
#                pos[0:placedb.num_physical_nodes] = placedb.node_x
#                pos[placedb.num_nodes:placedb.num_nodes+placedb.num_physical_nodes] = placedb.node_y
#                hpwl, density_overflow, max_density = placer.validate(placedb, pos, iteration)
#                logging.info("iteration %4d, HPWL %.3E, overflow %.3E, max density %.3E" % (iteration, hpwl, density_overflow, max_density))
#                placer.plot(params, placedb, iteration, pos)
#        elif 'ntuplace_4dr' in params.detailed_place_engine:
#            dp_out_file = gp_out_file.replace(".gp.def", "")
#            cmd = "%s" % (params.detailed_place_engine)
#            for lef in params.lef_input:
#                if "tech.lef" in lef:
#                    cmd += " -tech_lef %s" % (lef)
#                else:
#                    cmd += " -cell_lef %s" % (lef)
#            cmd += " -floorplan_def %s" % (gp_out_file)
#            cmd += " -verilog %s" % (params.verilog_input)
#            cmd += " -out ntuplace_4dr_out"
#            cmd += " -placement_constraints %s/placement.constraints" % (os.path.dirname(params.verilog_input))
#            cmd += " -noglobal %s ; " % (params.detailed_place_command)
#            cmd += "mv ntuplace_4dr_out.fence.plt %s.fense.plt ; " % (dp_out_file)
#            cmd += "mv ntuplace_4dr_out.init.plt %s.init.plt ; " % (dp_out_file)
#            cmd += "mv ntuplace_4dr_out %s.ntup.def ; " % (dp_out_file)
#            cmd += "mv ntuplace_4dr_out.ntup.overflow.plt %s.ntup.overflow.plt ; " % (dp_out_file)
#            cmd += "mv ntuplace_4dr_out.ntup.plt %s.ntup.plt ; " % (dp_out_file)
#            if os.path.exists("%s/dat" % (os.path.dirname(dp_out_file))):
#                cmd += "rm -r %s/dat ; " % (os.path.dirname(dp_out_file))
#            cmd += "mv dat %s/ ; " % (os.path.dirname(dp_out_file))
#            logging.info("%s" % (cmd))
#            tt = time.time()
#            os.system(cmd)
#            logging.info("External detailed placement takes %.2f seconds" % (time.time()-tt))
#        else:
#            logging.warning("External detailed placement only supports NTUplace3/NTUplace4dr API")
#    elif params.detailed_place_engine:
#        logging.warning("External detailed placement engine %s or aux file NOT found" % (params.detailed_place_engine))

    return metrics

if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow. 
    """
    logging.root.name = 'DREAMPlaceFPGA'
    logging.basicConfig(level=logging.INFO, format='[%(levelname)-7s] %(name)s - %(message)s', stream=sys.stdout)

    if len(sys.argv) < 2:
        logging.error("Input parameters required in json format")
    paramsArray = []
    for i in range(1, len(sys.argv)):
        params = ParamsFPGA()
        params.load(sys.argv[i])
        paramsArray.append(params)
    logging.info("Parameters[%d] = %s" % (len(paramsArray), paramsArray))

    #Settings to minimze non-determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False 
    torch.manual_seed(params.random_seed)
    np.random.seed(params.random_seed)
    #random.seed(params.random_seed)
    if params.gpu:
        torch.cuda.manual_seed_all(params.random_seed)
        torch.cuda.manual_seed(params.random_seed)

    tt = time.time()
    for params in paramsArray: 
        placeFPGA(params)
    logging.info("Completed Placement in %.3f seconds" % (time.time()-tt))

