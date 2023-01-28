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
    logging.info("Placement completed in %.2f seconds" % (time.time()-tt))

    # write placement solution 
    path = "%s/%s" % (params.result_dir, params.design_name())
    if not os.path.exists(path):
        os.system("mkdir -p %s" % (path))
    if params.global_place_flag and params.legalize_flag == 0: ##Only global placement is run
        gp_out_file = os.path.join(path, "%s.gp.pl" % (params.design_name()))
        placedb.write(params, gp_out_file)
        
        ##Use elfPlace binary to run legalization and detatiled placement
        #elfPlace binary picks file named gp.pl in the current directory
        if os.path.exists("thirdparty/elfPlace_LG_DP"):
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

    elif params.global_place_flag and params.legalize_flag: ## Run both global placement and detailed placement
        final_out_file = os.path.join(path, "%s.final.%s" % (params.design_name(), params.solution_file_suffix()))
        placedb.writeFinalSolution(params, final_out_file)
        logging.info("Detailed Placement not run")

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

