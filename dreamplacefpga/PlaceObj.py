##
# @file   PlaceObj.py
# @author Yibo Lin (DREAMPlace), Rachel Selina Rajarathnam (DREAMPlaceFPGA)
# @date   Sep 2020
# @brief  Placement model class defining the placement objective.
#
# Modifications Copyright(C) 2024 Advanced Micro Devices, Inc. All rights reserved
#

import os
import sys
import time
import numpy as np
import itertools
import logging
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb
import gzip
import math
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import dreamplacefpga.ops.weighted_average_wirelength.weighted_average_wirelength as weighted_average_wirelength
import dreamplacefpga.ops.timing_net_wirelength.timing_net_wirelength as timing_net_wirelength
#import dreamplacefpga.ops.logsumexp_wirelength.logsumexp_wirelength as logsumexp_wirelength
import dreamplacefpga.ops.electric_potential.electric_potential as electric_potential
import dreamplacefpga.ops.rudy.rudy as rudy
import dreamplacefpga.ops.pin_utilization.pin_utilization as pin_utilization
#FPGA clustering compatibility resource area computation
import dreamplacefpga.ops.clustering_compatibility.clustering_compatibility as clustering_compatibility
import dreamplacefpga.ops.adjust_node_area.adjust_node_area as adjust_node_area

# For FPGA
class PreconditionOpFPGA:
    """Preconditioning engine is critical for convergence.
    Need to be carefully designed.
    """
    def __init__(self, placedb, data_collections, op_collections):
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.iteration = 0
        self.timing_iteration = 0
        self.precondTiming = torch.zeros(placedb.num_nodes, device=data_collections.pos[0].device, dtype=torch.float32)
        self.movablenode2fence_region_map_clamp = data_collections.node2fence_region_map[:placedb.num_movable_nodes].clamp(max=len(placedb.region_boxes)).long()
        self.filler2fence_region_map = torch.zeros(placedb.num_filler_nodes, device=data_collections.pos[0].device, dtype=torch.long)
        for i in range(len(placedb.region_boxes)):
            filler_beg, filler_end = placedb.filler_start_map[i:i+2]
            self.filler2fence_region_map[filler_beg:filler_end] = i

    def __call__(self, grad, density_weight, precondWL, beta, num_timing_iteration, update_mask=None):
        """Introduce alpha parameter to avoid divergence.
        It is tricky for this parameter to increase.
        """
        with torch.no_grad():
            #FPGA Preconditioning 
            node_areas = self.data_collections.node_areas.clone()

            for mk in range(len(self.placedb.region_boxes)):
                mask = self.data_collections.node2fence_region_map[:self.placedb.num_movable_nodes] == mk
                node_areas[:self.placedb.num_movable_nodes].masked_scatter_(mask, node_areas[:self.placedb.num_movable_nodes][mask]*density_weight[mk])
                filler_beg, filler_end = self.placedb.filler_start_map[mk:mk+2]
                node_areas[self.placedb.num_nodes-self.placedb.num_filler_nodes+filler_beg:self.placedb.num_nodes-self.placedb.num_filler_nodes+filler_end] *= density_weight[mk]

            if self.timing_iteration < num_timing_iteration:
                precondTiming = self.op_collections.precond_timing_op(beta, self.data_collections.tnet_weights)
                self.precondTiming = precondTiming
            elif self.timing_iteration == num_timing_iteration:
                precondTiming = self.precondTiming

            #precond = precondWL + node_areas
            precond = precondWL + node_areas + precondTiming
            # logging.info("precondWL: %g, precondTiming: %g, node_areas: %g" %(precondWL.sum(), precondTiming.sum(), node_areas.sum()))

            #Use alpha to avoid divergence
            #precond = precondWL + self.alpha * node_areas

            precond.clamp_(min=1.0)
            grad[0:self.placedb.num_nodes].div_(precond)
            grad[self.placedb.num_nodes:self.placedb.num_nodes *
                 2].div_(precond)

            #print("Overall preconditioned grad norm1: %g" %(grad.norm(p=1)))
            ### stop gradients for terminated electric field
            if(update_mask is not None):
                grad = grad.view(2, -1)
                update_mask = ~update_mask
                if update_mask.sum() < len(update_mask):
                    movable_mask = update_mask[self.movablenode2fence_region_map_clamp]
                    filler_mask = update_mask[self.filler2fence_region_map]
                    grad[0, :self.placedb.num_movable_nodes].masked_fill_(movable_mask, 0)
                    grad[1, :self.placedb.num_movable_nodes].masked_fill_(movable_mask, 0)
                    grad[0, self.placedb.num_nodes-self.placedb.num_filler_nodes:].masked_fill_(filler_mask, 0)
                    grad[1, self.placedb.num_nodes-self.placedb.num_filler_nodes:].masked_fill_(filler_mask, 0)
                    grad = grad.view(-1)
            self.iteration += 1
            self.timing_iteration = num_timing_iteration

        return grad

class PlaceObjFPGA(nn.Module):
    """
    @brief Define placement objective:
        wirelength + density_weight * density penalty
    It includes various ops related to global placement as well.
    """
    def __init__(self, density_weight, params, placedb, data_collections, op_collections, global_place_params):
        """
        @brief initialize ops for placement
        @param density_weight density weight in the objective
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param op_collections a collection of all ops
        @param global_place_params global placement parameters for current global placement stage
        """
        super(PlaceObjFPGA, self).__init__()

        ### quadratic penalty
        self.density_quad_coeff = 1000 #corresponds to beta/2 in obj function
        self.quad_penalty_coeff = None
        self.init_density = None 
        ### increase density penalty if slow convergence
        self.density_factor = 1

        ### fence region will enable quadratic penalty by default
        self.quad_penalty = True

        ### fence region
        ### update mask controls whether stop gradient/updating, 1 represents allow grad/update
        self.update_mask = None
        self.lock_mask = None
        ### for subregion rough legalization, once stop updating, perform immediate greedy legalization once
        ### this is to avoid repeated legalization
        ### 1 represents already legal
        self.legal_mask = torch.zeros(placedb.regions)
        self.legal_mask[7] = 1 #IOs are legal

        self.params = params
        self.placedb = placedb
        self.data_collections = data_collections
        self.op_collections = op_collections
        self.global_place_params = global_place_params

        self.fixedDemMaps = []

        self.gpu = params.gpu
        self.precondWL = self.op_collections.precondwl_op()
        self.fixedDemMaps = self.op_collections.demandMap_op()

        ### different fence region needs different density weights in multi-electric field algorithm
        self.density_weight = torch.tensor(
            [density_weight]*(len(placedb.region_boxes)),
            dtype=self.data_collections.pos[0].dtype,
            device=self.data_collections.pos[0].device)
        ### Note: even for multi-electric fields, they use the same gamma
        self.gamma = torch.tensor(self.base_gamma(params, placedb)[0],
                                  dtype=self.data_collections.pos[0].dtype,
                                  device=self.data_collections.pos[0].device)
        initOverflow = torch.ones(7, dtype=self.gamma.dtype, device=self.gamma.device)
        self.update_gamma(0, initOverflow, self.base_gamma(params, placedb))

        # compute weighted average wirelength from position
        self.num_bins_x = placedb.num_bins_x
        self.num_bins_y = placedb.num_bins_y

        self.name = "%dx%d bins" % (self.num_bins_x, self.num_bins_y)

        self.op_collections.wirelength_op, self.op_collections.update_gamma_op = self.build_weighted_average_wl(
            params, placedb, self.data_collections, self.op_collections.pin_pos_op)

        # build timing net op
        self.op_collections.tnet_wirelength_op = self.build_tnet_wl(params, placedb, self.data_collections, self.op_collections.pin_pos_op)
        self.beta = 0.0
        self.num_timing_iteration = 0

        self.op_collections.density_op = self.build_electric_potential(
            params,
            placedb,
            self.data_collections,
            self.num_bins_x,
            self.num_bins_y,
            name=self.name)

        ### build multiple density op for multi-electric field
        self.op_collections.fence_region_density_ops, self.op_collections.fence_region_density_merged_op, self.op_collections.fence_region_density_overflow_merged_op = self.build_multi_fence_region_density_op()

        self.op_collections.update_density_weight_op = self.build_update_density_weight(params, placedb)

        self.op_collections.precondition_op = self.build_precondition(params, placedb, self.data_collections, self.op_collections)

        self.op_collections.noise_op = self.build_noise(params, placedb, self.data_collections)

        if params.routability_opt_flag:
            # compute congestion map, RISA/RUDY congestion map
            self.op_collections.route_utilization_map_op = self.build_route_utilization_map(params, placedb, self.data_collections)
            self.op_collections.pin_utilization_map_op = self.build_pin_utilization_map(params, placedb, self.data_collections)
            #FPGA clustering compatibility resource area computation
            self.op_collections.clustering_compatibility_lut_area_op = self.build_clustering_compatibility_lut_map(params, placedb, self.data_collections)
            self.op_collections.clustering_compatibility_ff_area_op = self.build_clustering_compatibility_ff_map(params, placedb, self.data_collections)
            # adjust instance area with congestion map
            self.op_collections.adjust_node_area_op = self.build_adjust_node_area(params, placedb, self.data_collections)

        self.Lgamma_iteration = global_place_params["iteration"]
        if 'Llambda_density_weight_iteration' in global_place_params:
            self.Llambda_density_weight_iteration = global_place_params['Llambda_density_weight_iteration']
        else:
            self.Llambda_density_weight_iteration = 1
        if 'Lsub_iteration' in global_place_params:
            self.Lsub_iteration = global_place_params['Lsub_iteration']
        else:
            self.Lsub_iteration = 1
        if 'routability_Lsub_iteration' in global_place_params:
            self.routability_Lsub_iteration = global_place_params['routability_Lsub_iteration']
        else:
            self.routability_Lsub_iteration = self.Lsub_iteration
        self.start_fence_region_density = False

    def obj_fn(self, pos):
        """
        @brief Compute objective.
            wirelength + density_weight * density penalty
        @param pos locations of cells
        @return objective value
        """
        wirelength = self.op_collections.wirelength_op(pos)

        tnet_wirelength = self.op_collections.tnet_wirelength_op(pos)

        density = self.op_collections.fence_region_density_merged_op(pos)

        if self.init_density is None:
            ### record initial density
            self.init_density = density.data.clone()
            ### density weight subgradient preconditioner
            self.density_weight_grad_precond = self.init_density.masked_scatter(self.init_density != 0, 1/self.init_density[self.init_density != 0])
        ### quadratic density penalty
        if self.quad_penalty_coeff is None:
            self.quad_penalty_coeff = self.density_quad_coeff/2 * self.density_weight_grad_precond

        density = density*(1+self.quad_penalty_coeff * density)

        result = wirelength + self.density_weight_u.dot(density) + self.beta * tnet_wirelength
        #logging.info("result: %g" %(result))
        # logging.info("WL cost: %g, density cost: %g, timing cost: %g " %(wirelength, self.density_weight_u.dot(density), self.beta * tnet_wirelength))


        return result

    def obj_and_grad_fn(self, pos):
        """
        @brief compute objective and gradient.
            wirelength + density_weight * density penalty
        @param pos locations of cells
        @return objective value
        """
        #self.check_gradient(pos)
        if pos.grad is not None:
            pos.grad.zero_()

        obj = self.obj_fn(pos)
        obj.backward()

        self.op_collections.precondition_op(pos.grad, self.density_weight, self.precondWL, self.beta, self.num_timing_iteration, self.update_mask)

        return obj, pos.grad

    def forward(self):
        """
        @brief Compute objective with current locations of cells.
        """
        return self.obj_fn(self.data_collections.pos[0])

    def check_gradient(self, pos):
        """
        @brief check gradient for debug
        @param pos locations of cells
        """
        wirelength = self.op_collections.wirelength_op(pos)
        tnet_wirelength = self.beta * self.op_collections.tnet_wirelength_op(pos)

        if pos.grad is not None:
            pos.grad.zero_()
        wirelength.backward()
        wirelength_grad = pos.grad.clone()

        pos.grad.zero_()
        density = self.density_weight * self.op_collections.density_op(pos)
        density.backward()
        density_grad = pos.grad.clone()

        pos.grad.zero_()
        tnet_wirelength.backward()
        tnet_wirelength_grad = pos.grad.clone()

        wirelength_grad_norm = wirelength_grad.norm(p=1)
        density_grad_norm = density_grad.norm(p=1)
        tnet_wirelength_grad_norm = tnet_wirelength_grad.norm(p=1)

        # logging.info("wirelength grad norm: %g, tnet_wirelength grad norm: %g" % (wirelength_grad_norm, tnet_wirelength_grad_norm))

        pos.grad.zero_()

    def estimate_initial_learning_rate(self, x_k):
        """
        @brief Estimate initial learning rate by moving a small step.
        Computed as | x_k - x_k_1 |_2 / | g_k - g_k_1 |_2.
        @param x_k current solution
        """
        obj_k, g_k = self.obj_and_grad_fn(x_k)
        lr = 0.001 * min((self.placedb.xh - self.placedb.xl), (self.placedb.yh-self.placedb.yl)) * (self.placedb.num_nodes - self.placedb.num_terminals)
        lr /= g_k.norm(p=1)
        x_k_1 = torch.autograd.Variable(x_k - lr * g_k, requires_grad=True)
        obj_k_1, g_k_1 = self.obj_and_grad_fn(x_k_1)
        #print("alpha = %g"%(lr))
        #print("learning rate = %g"%((x_k - x_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2)))

        return (x_k - x_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2)
    
    def build_tnet_wl(self, params, placedb, data_collections, pin_pos_op):
        """
        @brief build the op to compute timing net wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        """
        tnet_wirelength_for_pin_op = timing_net_wirelength.TimingNetWirelength(
            flat_tnetpin=data_collections.flat_tnet2pin_map, 
            tnet_weights=data_collections.tnet_weights, 
            pin_mask=data_collections.pin_mask_ignore_fixed_macros,
            gamma=self.gamma, 
            net_bounding_box_min=data_collections.net_bounding_box_min,
            net_bounding_box_max=data_collections.net_bounding_box_max,
            num_threads=params.num_threads,
            xl=placedb.xl, 
            yl=placedb.yl, 
            xh=placedb.xh,  
            yh=placedb.yh,
            deterministic_flag=params.deterministic_flag,
            algorithm='merged')

        # wirelength for position
        def build_tnet_wirelength_op(pos):
            return tnet_wirelength_for_pin_op(pin_pos_op(pos))

        return build_tnet_wirelength_op

    def build_weighted_average_wl(self, params, placedb, data_collections, pin_pos_op):
        """
        @brief build the op to compute weighted average wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        """
        # use WeightedAverageWirelength atomic
        wirelength_for_pin_op = weighted_average_wirelength.WeightedAverageWirelength(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            pin_mask=data_collections.pin_mask_ignore_fixed_macros,
            gamma=self.gamma,
            net_bounding_box_min=data_collections.net_bounding_box_min,
            net_bounding_box_max=data_collections.net_bounding_box_max,
            num_threads=params.num_threads,
            algorithm='merged')
            #algorithm='net-by-net')

        # wirelength for position
        def build_wirelength_op(pos):
            return wirelength_for_pin_op(pin_pos_op(pos))

        # update gamma
        base_gamma = self.base_gamma(params, placedb)

        def build_update_gamma_op(iteration, overflow):
            self.update_gamma(iteration, overflow, base_gamma)
            #logging.debug("update gamma to %g" % (wirelength_for_pin_op.gamma.data))

        return build_wirelength_op, build_update_gamma_op

    def build_electric_potential(self, params, placedb, data_collections,
                                 num_bins_x, num_bins_y, name, region_id=None, fence_regions=None):
        """
        @brief e-place electrostatic potential
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        @param num_bins_x number of bins in horizontal direction
        @param num_bins_y number of bins in vertical direction
        @param padding number of padding bins to left, right, bottom, top of the placement region
        @param name string for printing
        @param fence_regions a [n_subregions, 7] tensor for fence regions potential penalty
        """
        bin_size_x = (placedb.xh - placedb.xl) / num_bins_x
        bin_size_y = (placedb.yh - placedb.yl) / num_bins_y

        return electric_potential.ElectricPotential(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminals=placedb.num_terminals,
            num_filler_nodes=placedb.num_filler_nodes,
            deterministic_flag=params.deterministic_flag,
            sorted_node_map=data_collections.sorted_node_map,
            region_id=region_id,
            fence_regions=fence_regions,
            node2fence_region_map=data_collections.node2fence_region_map,
            placedb=placedb)
    
    def initialize_timing_beta(self, params, placedb, num_timing_iteration):
        """
        @brief initialize timing weight
        @param params parameters
        @param placedb placement database
        """
        self.num_timing_iteration = num_timing_iteration

        #Updated timing betas
        wirelength = self.op_collections.wirelength_op(self.data_collections.pos[0])
        tnet_wirelength = self.op_collections.tnet_wirelength_op(self.data_collections.pos[0])

        if self.data_collections.pos[0].grad is not None:
            self.data_collections.pos[0].grad.zero_()
        
        wirelength.backward()
        wirelength_grad_norm = self.data_collections.pos[0].grad.norm(p=1)
     
        self.data_collections.pos[0].grad.zero_()

        tnet_wirelength.backward()
        tnet_wirelength_grad_norm = self.data_collections.pos[0].grad.norm(p=1)

        # initial_beta_ratio = params.init_beta
        initial_beta_ratio = 0.1
        # beta_ratio = initial_beta_ratio + params.beta_step * (num_timing_iteration - 1)
        beta_ratio = initial_beta_ratio + 0.1 * (num_timing_iteration - 1)

        if beta_ratio > params.beta_ratio:
            beta_ratio = params.beta_ratio

        if tnet_wirelength_grad_norm != 0:
            self.beta = beta_ratio * wirelength_grad_norm / tnet_wirelength_grad_norm
    
        # logging.info("beta = %g, wirelength_grad_norm = %g, tnet_wirelength_grad_norm = %g" % (self.beta, wirelength_grad_norm, tnet_wirelength_grad_norm))

    def initialize_density_weight(self, params, placedb):
        """
        @brief compute initial density weight
        @param params parameters
        @param placedb placement database
        """
        #Updated to elfPlace
        wirelength = self.op_collections.wirelength_op(self.data_collections.pos[0])

        if self.data_collections.pos[0].grad is not None:
            self.data_collections.pos[0].grad.zero_()
        wirelength.backward()
        wirelength_grad_norm = self.data_collections.pos[0].grad.norm(p=1)
        #content = "Initial WL grad norm = %.3E" % (wirelength_grad_norm)

        self.data_collections.pos[0].grad.zero_()
        density_weight = []
        density_list = []
        density_grad_list = []
        for density_op in self.op_collections.fence_region_density_ops:
            density_i = density_op(self.data_collections.pos[0])
            density_list.append(density_i.data.clone())
            density_i.backward()
            density_grad_list.append(self.data_collections.pos[0].grad.data.clone())
            self.data_collections.pos[0].grad.zero_()

        ## density = self.op_collections.fence_region_density_merged_op(self.data_collections.pos[0])
        #### record initial density
        self.init_density = torch.stack(density_list)

        #### density weight subgradient preconditioner
        self.density_weight_grad_precond = self.init_density.masked_scatter(self.init_density != 0, 1/self.init_density[self.init_density != 0])
        #content += ", Density weight gradient preconditioner = [%s]" % ", ".join(["%.3E" % i for i in self.density_weight_grad_precond.cpu().numpy().tolist()])
        #### compute u
        self.density_weight_u = self.init_density * self.density_weight_grad_precond
        self.density_weight_u += 0.5 * self.density_quad_coeff * self.density_weight_u**2
        #### compute s
        density_weight_s = 1 + self.density_quad_coeff * self.init_density * self.density_weight_grad_precond

        #### compute density grad L1 norm
        density_grad_norm = sum(self.density_weight_u[i]*density_weight_s[i]*density_grad_list[i].norm(p=1) for i in range(density_weight_s.size(0)))
        #content += ", Initial Density grad Norm = %.3E" % (density_grad_norm)

        self.density_weight_u *= params.density_weight * wirelength_grad_norm / density_grad_norm

        #### set initial step size for density weight update
        self.density_weight_step_size_inc_low = 1.05
        self.density_weight_step_size_inc_high = 1.06

        self.density_weight_step_size = (self.density_weight_step_size_inc_low - 1) * self.density_weight_u.norm(p=2)
        ### commit initial density weight
        self.density_weight = self.density_weight_u * density_weight_s

        return self.density_weight

    def reset_density_weight(self, params, placedb, ratio):
        """
        @brief compute initial density weight
        @param params parameters
        @param placedb placement database
        @param ratio to weight density weight
        """
        wirelength = self.op_collections.wirelength_op(self.data_collections.pos[0])

        if self.data_collections.pos[0].grad is not None:
            self.data_collections.pos[0].grad.zero_()
        wirelength.backward()
        wirelength_grad_norm = self.data_collections.pos[0].grad.norm(p=1)

        self.data_collections.pos[0].grad.zero_()
        density_list = []
        density_grad_list = []
        for density_op in self.op_collections.fence_region_density_ops:
            density_i = density_op(self.data_collections.pos[0])
            density_list.append(density_i.data.clone())
            # if(self.quad_penalty):
            #     density_i = density_i + self.density_quad_coeff / 2 / density_i * density_i **2
            density_i.backward()
            density_grad_list.append(self.data_collections.pos[0].grad.data.clone())
            self.data_collections.pos[0].grad.zero_()

        ### record updated density
        self.upd_density = torch.stack(density_list)

        #### Reset lambda - density_weight_u
        self.density_weight_u = self.upd_density * self.density_weight_grad_precond
        self.density_weight_u += 0.5 * self.density_quad_coeff * self.density_weight_u**2
        #### compute s
        density_weight_s = 1 + self.density_quad_coeff * self.upd_density * self.density_weight_grad_precond

        #### compute density grad L1 norm
        density_grad_norm = sum(self.density_weight_u[i]*density_weight_s[i]*density_grad_list[i].norm(p=1) for i in range(density_weight_s.size(0)))
       
        self.density_weight_u *= ratio * wirelength_grad_norm / density_grad_norm

        self.density_weight_step_size = (self.density_weight_step_size_inc_low - 1) * self.density_weight_u.norm(p=2)
        # ### commit the density weight
        self.density_weight = self.density_weight_u * density_weight_s

        return self.density_weight


    def build_update_density_weight(self, params, placedb, algo="overflow"):
        """
        @brief update density weight
        @param params parameters
        @param placedb placement database
        """
        #Updated to elfPlace
        ### params for hpwl mode from RePlAce
        ref_hpwl = params.RePlAce_ref_hpwl
        LOWER_PCOF = params.RePlAce_LOWER_PCOF
        UPPER_PCOF = params.RePlAce_UPPER_PCOF
        ### params for overflow mode from elfPlace
        # alpha_h = 1.06
        # alpha_l = 1.05
        # self.density_step_size = alpha_h-1
        assert algo in {"hpwl", "overflow"}, logging.error("density weight update not supports hpwl mode or overflow mode")

        def update_density_weight_op_hpwl(cur_metric, prev_metric, iteration):
            ### based on hpwl
            with torch.no_grad():
                delta_hpwl = cur_metric.hpwl - prev_metric.hpwl
                if delta_hpwl < 0:
                    mu = UPPER_PCOF * np.maximum(
                        np.power(0.9999, float(iteration)), 0.98)
                    #mu = UPPER_PCOF*np.maximum(np.power(0.9999, float(iteration)), 1.03)
                else:
                    mu = UPPER_PCOF * torch.pow(
                        UPPER_PCOF, -delta_hpwl / ref_hpwl).clamp(
                            min=LOWER_PCOF, max=UPPER_PCOF)
                self.density_weight *= mu

        def update_density_weight_op_overflow(cur_metric, prev_metric, iteration):
            assert self.quad_penalty == True, "[Error] density weight update based on overflow only works for quadratic density penalty"
            ### based on overflow
            ### stop updating if a region has lower overflow than stop overflow
            with torch.no_grad():
                density_norm = cur_metric.density * self.density_weight_grad_precond
                density_weight_grad = density_norm + self.density_quad_coeff/2*density_norm**2

                #content = "Density Norm = [%s]" % ", ".join(["%.3E" % i for i in density_norm.cpu().numpy().tolist()])
                ##Rachel: Possibility of zero in density_norm for some resource types could result in INF in density weight grad computation
                if density_weight_grad.isinf().any():
                    density_weight_grad[density_weight_grad == float("Inf")] = 0

                density_weight_grad /= density_weight_grad.norm(p=2)

                #content += ", Density Weight Grad = [%s]" % ", ".join(["%.3E" % i for i in density_weight_grad.cpu().numpy().tolist()])

                ### self.density_weight += self.density_weight_step_size * density_weight_grad# * 1e-7
                self.density_weight_u += self.density_weight_step_size * density_weight_grad
                density_weight_s = 1 + self.density_quad_coeff * density_norm

                #content += ", Density Weight Step Size = %.3E, " % (self.density_weight_step_size)
                #content += "density_weight_s = [%s]" % ", ".join(["%.3E" % i for i in density_weight_s.cpu().numpy().tolist()])
                #content += ", density_weight_u = [%s]" % ", ".join(["%.3E" % i for i in self.density_weight_u.cpu().numpy().tolist()])

                #### update density weight step size
                rate = torch.log(self.density_quad_coeff * density_norm.norm(p=2)).clamp(min=0)
                rate = rate / (1 + rate)
                rate = rate * (self.density_weight_step_size_inc_high - self.density_weight_step_size_inc_low) + self.density_weight_step_size_inc_low
                self.density_weight_step_size *= rate

                #content += ", Rate = %g" % (rate)
                #### conditional update if this region's overflow is higher than stop overflow

                density_weight_new = self.density_weight_u * density_weight_s

                self.targetOverflow = torch.tensor(self.placedb.targetOverflow, dtype=torch.float, device=self.data_collections.pos[0].device)
                if(self.update_mask is None):
                    self.update_mask = cur_metric.overflow >= self.targetOverflow
                    self.lock_mask = cur_metric.overflow < self.targetOverflow

                self.density_weight.masked_scatter_(self.update_mask, density_weight_new[self.update_mask])
                #content += ", density_weight = [%s]" % ", ".join(["%.3E" % i for i in self.density_weight.cpu().numpy().tolist()])
                #logging.info(content)


        if(not self.quad_penalty and algo == "overflow"):
            logging.warn("quadratic density penalty is disabled, density weight update is forced to be based on HPWL")
            algo = "hpwl"
        if(len(self.placedb.region_boxes) == 0 and algo == "overflow"):
            logging.warn("for benchmark without fence region, density weight update is forced to be based on HPWL")
            algo = "hpwl"

        update_density_weight_op = {"hpwl":update_density_weight_op_hpwl, "overflow": update_density_weight_op_overflow}[algo]

        return update_density_weight_op

    def base_gamma(self, params, placedb):
        """
        @brief compute base gamma
        @param params parameters
        @param placedb placement database
        """
        ## Updated to elfPlace
        self.baseWLGamma = []
        self.WLGammaK = []
        self.WLGammaB = []
        self.WLGammaWt = []

        for i in range(len(placedb.region_boxes)):
            self.baseWLGamma.append(0.5 * params.gamma * (placedb.bin_size_x + placedb.bin_size_y))
            # Compute coeffcient for wirelength gamma updating
            # The basic idea is that we want to achieve
            #   gamma =  10 * base_gamma, if overflow = 1.0
            #   gamma = 0.1 * base_gamma, if overflow = target_overflow
            # We use function f(ovfl) = 10^(k * ovfl + b) to achieve the two above two points
            # So we want
            #   k + b = 1
            #   k * target_overflow + b = -1
            # Then we have
            #   k = 2.0 / (1 - target_overflow)
            #   b = 1.0 - k
            self.WLGammaK.append(2.0/(1.0 - placedb.targetOverflow[i]))
            self.WLGammaB.append(1.0 - self.WLGammaK[i])
            # Compare the wirelength gamma weight to balance gamma updating for different area types
            self.WLGammaWt.append(self.precondWL[:placedb.num_physical_nodes][self.data_collections.node2fence_region_map == i].sum())

        return self.baseWLGamma

    def update_gamma(self, iteration, overflow, base_gamma):
        """
        @brief update gamma in wirelength model
        @param iteration optimization step
        @param overflow evaluated in current step
        @param base_gamma base gamma
        """
        ## Updated to elfPlace
        # Compute the gamma for each area type and use the pin count-averaged value as the final gamma
        totalGamma = 0.0
        totalWt = 0.0
        for i in range(len(self.placedb.region_boxes)):
            gma = base_gamma[i] * pow(10.0, overflow[i] * self.WLGammaK[i] + self.WLGammaB[i])
            totalGamma += gma * self.WLGammaWt[i]
            totalWt += self.WLGammaWt[i]

        self.gamma.data.fill_(totalGamma / totalWt)
        return True

    def build_noise(self, params, placedb, data_collections):
        """
        @brief add noise to cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        """
        node_size = torch.cat([data_collections.node_size_x, data_collections.node_size_y],
            dim=0).to(data_collections.pos[0].device)

        def noise_op(pos, noise_ratio):
            with torch.no_grad():
                noise = torch.rand_like(pos)
                noise.sub_(0.5).mul_(node_size).mul_(noise_ratio)
                # no noise to fixed cells
                noise[placedb.num_movable_nodes:placedb.num_nodes -
                      placedb.num_filler_nodes].zero_()
                noise[placedb.num_nodes +
                      placedb.num_movable_nodes:2 * placedb.num_nodes -
                      placedb.num_filler_nodes].zero_()
                return pos.add_(noise)

        return noise_op

    def build_precondition(self, params, placedb, data_collections, op_collections):
        """
        @brief preconditioning to gradient
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of data and variables required for constructing ops
        """

        #def precondition_op(grad):
        #    with torch.no_grad():
        #        # preconditioning
        #        node_areas = data_collections.node_size_x * data_collections.node_size_y
        #        precond = self.density_weight * node_areas
        #        precond[:placedb.num_physical_nodes].add_(data_collections.pin_weights)
        #        precond.clamp_(min=1.0)
        #        grad[0:placedb.num_nodes].div_(precond)
        #        grad[placedb.num_nodes:placedb.num_nodes*2].div_(precond)
        #        #for p in pos:
        #        #    grad_norm = p.grad.norm(p=2)
        #        #    logging.debug("grad_norm = %g" % (grad_norm.data))
        #        #    p.grad.div_(grad_norm.data)
        #        #    logging.debug("grad_norm = %g" % (p.grad.norm(p=2).data))
        #        #grad.data[0:placedb.num_movable_nodes].div_(grad[0:placedb.num_movable_nodes].norm(p=2))
        #        #grad.data[placedb.num_nodes:placedb.num_nodes+placedb.num_movable_nodes].div_(grad[placedb.num_nodes:placedb.num_nodes+placedb.num_movable_nodes].norm(p=2))
        #    return grad

        #return precondition_op

        return PreconditionOpFPGA(placedb, data_collections, op_collections)

    def build_route_utilization_map(self, params, placedb, data_collections):
        """
        @brief routing congestion map based on current cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        congestion_op = rudy.Rudy(
            netpin_start=data_collections.flat_net2pin_start_map,
            flat_netpin=data_collections.flat_net2pin_map,
            net_weights=data_collections.net_weights,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_bins_x=placedb.num_routing_grids_x,
            num_bins_y=placedb.num_routing_grids_y,
            unit_horizontal_capacity=placedb.unit_horizontal_capacity,
            unit_vertical_capacity=placedb.unit_vertical_capacity,
            deterministic_flag=params.deterministic_flag,
            initial_horizontal_utilization_map=data_collections.
            initial_horizontal_utilization_map,
            initial_vertical_utilization_map=data_collections.
            initial_vertical_utilization_map,
            num_threads=params.num_threads)

        def route_utilization_map_op(pos):
            pin_pos = self.op_collections.pin_pos_op(pos)
            return congestion_op(pin_pos)

        return route_utilization_map_op

    def build_pin_utilization_map(self, params, placedb, data_collections):
        """
        @brief pin density map based on current cell locations
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        return pin_utilization.PinUtilization(
            pin_weights=data_collections.pin_weights,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
            num_bins_x=placedb.num_routing_grids_x,
            num_bins_y=placedb.num_routing_grids_y,
            unit_pin_capacity=data_collections.unit_pin_capacity,
            pin_stretch_ratio=params.pin_stretch_ratio,
            deterministic_flag=params.deterministic_flag,
            num_threads=params.num_threads)

    #Update for LUT
    def build_clustering_compatibility_lut_map(self, params, placedb, data_collections):
        """
        @brief clustering compatibility lut map based on current cell locations to ensure maximum input pin constraint is met
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        bins_x = math.ceil((placedb.xh - placedb.xl)/placedb.instDemStddevX) 
        bins_y = math.ceil((placedb.yh - placedb.yl)/placedb.instDemStddevY) 
        return clustering_compatibility.LUTCompatibility(
            lut_indices=data_collections.lut_indices,
            lut_type=data_collections.cluster_lut_type,
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            num_bins_x=bins_x,
            num_bins_y=bins_y,
            num_bins_l=placedb.cluster_lut_type.max()+1,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            inst_stddev_x=placedb.instDemStddevX,
            inst_stddev_y=placedb.instDemStddevY,
            inst_stddev_trunc=placedb.instDemStddevTrunc,
            deterministic_flag=params.deterministic_flag,
            num_threads=params.num_threads)

    #Update for FF
    def build_clustering_compatibility_ff_map(self, params, placedb, data_collections):
        """
        @brief clustering compatibility flop map based on current cell locations to ensure control set constraint is met
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        """
        bins_x = math.ceil((placedb.xh - placedb.xl)/placedb.instDemStddevX) 
        bins_y = math.ceil((placedb.yh - placedb.yl)/placedb.instDemStddevY) 
        return clustering_compatibility.FFCompatibility(
            flop_indices=data_collections.flop_indices,
            flop_ctrlSets=data_collections.flop_ctrlSets,
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            num_bins_x=bins_x,
            num_bins_y=bins_y,
            num_bins_ck=placedb.ctrlSets[:,1].max()+1,
            num_bins_ce=placedb.ctrlSets[:,2].max()+1,
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            inst_stddev_x=placedb.instDemStddevX,
            inst_stddev_y=placedb.instDemStddevY,
            inst_stddev_trunc=placedb.instDemStddevTrunc,
            deterministic_flag=params.deterministic_flag,
            num_threads=params.num_threads)

    def build_adjust_node_area(self, params, placedb, data_collections):
        """
        @brief adjust cell area according to routing congestion and pin utilization map
        """
        #Include total area only for LUT/FF
        total_movable_area = (
            data_collections.node_size_x[:placedb.num_movable_nodes] *
            data_collections.node_size_y[:placedb.num_movable_nodes] * 
            data_collections.flop_lut_mask[:placedb.num_movable_nodes]).sum()
        flop_lut_fillers = placedb.filler_start_map[2]
        total_filler_area = (
            data_collections.node_size_x[placedb.num_physical_nodes:placedb.num_physical_nodes + flop_lut_fillers] *
            data_collections.node_size_y[placedb.num_physical_nodes:placedb.num_physical_nodes + flop_lut_fillers]).sum()
        total_place_area = (total_movable_area + total_filler_area)
        adjust_node_area_op = adjust_node_area.AdjustNodeArea(
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin_weights=data_collections.pin_weights,
            flop_lut_indices=data_collections.flop_lut_indices,
            flop_lut_mask=data_collections.flop_lut_mask,
            flop_mask=data_collections.flop_mask,
            lut_mask=data_collections.lut_mask,
            filler_start_map=placedb.filler_start_map,
            xl=placedb.routing_grid_xl,
            yl=placedb.routing_grid_yl,
            xh=placedb.routing_grid_xh,
            yh=placedb.routing_grid_yh,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
            route_num_bins_x=placedb.num_routing_grids_x,
            route_num_bins_y=placedb.num_routing_grids_y,
            pin_num_bins_x=placedb.num_routing_grids_x,
            pin_num_bins_y=placedb.num_routing_grids_y,
            total_place_area=total_place_area,
            total_whitespace_area=total_place_area - total_movable_area,
            max_route_opt_adjust_rate=params.max_route_opt_adjust_rate,
            route_opt_adjust_exponent=params.route_opt_adjust_exponent,
            max_pin_opt_adjust_rate=params.max_pin_opt_adjust_rate,
            area_adjust_stop_ratio=params.area_adjust_stop_ratio,
            route_area_adjust_stop_ratio=params.route_area_adjust_stop_ratio,
            pin_area_adjust_stop_ratio=params.pin_area_adjust_stop_ratio,
            unit_pin_capacity=data_collections.unit_pin_capacity,
            inflation_ratio=params.inflation_ratio)

        def build_adjust_node_area_op(pos, resource_areas, route_utilization_map, pin_utilization_map):
            return adjust_node_area_op(
                pos, data_collections.node_size_x,
                data_collections.node_size_y, data_collections.pin_offset_x,
                data_collections.pin_offset_y, 1.0,
                resource_areas, route_utilization_map, pin_utilization_map)

        return build_adjust_node_area_op

    def build_multi_fence_region_density_op(self):
        # region 0, ..., region n, non_fence_region
        self.op_collections.fence_region_density_ops = []
        #self.streams = [torch.cuda.Stream() for i in range(2)]

        for i, fence_region_map in enumerate(self.fixedDemMaps):
            #with torch.cuda.stream(self.streams[i%2]):
            self.op_collections.fence_region_density_ops.append(self.build_electric_potential(
                        self.params,
                        self.placedb,
                        self.data_collections,
                        self.num_bins_x,
                        self.num_bins_y,
                        name=self.name,
                        region_id=i,
                        fence_regions=fence_region_map)
            )

        def merged_density_op(pos):
            #### stop mask is to stop forward of density
            #### 1 represents stop flag

            resdb = torch.stack([densityOp(pos, mode="density") for densityOp in self.op_collections.fence_region_density_ops])

            return resdb

        def merged_density_overflow_op(pos):
            ### stop mask is to stop forward of density
            ### 1 represents stop flag
            overflow_list, max_density_list = [], []
            for density_op in self.op_collections.fence_region_density_ops:
                overflow, max_density = density_op(pos, mode="overflow")
                overflow_list.append(overflow)
                max_density_list.append(max_density)
            overflow_list, max_density_list = torch.stack(overflow_list), torch.stack(max_density_list)
            return overflow_list, max_density_list

        self.op_collections.fence_region_density_merged_op = merged_density_op

        self.op_collections.fence_region_density_overflow_merged_op = merged_density_overflow_op
        return self.op_collections.fence_region_density_ops, self.op_collections.fence_region_density_merged_op, self.op_collections.fence_region_density_overflow_merged_op


