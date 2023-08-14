# Testing framework for OpenRSP

# PyOpenRSP imports
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
dir5_path = os.path.join(current_dir, '..', 'KTHveloxchem/PyOpenRSP')
sys.path.append(dir5_path)

import callbacks_testing as cb
import openrsp
import copy

from pert_tuple_cache import rspPert, rspPertTuple, rspCache

# from decoratorsWorkflow import decorate_all_in_module, function_call_counter, decorator_showname
# decorate_all_in_module(openrsp, decorator_showname)
# decorate_all_in_module(cb, decorator_showname)

# Set up callbacks

cb_a = cb.callbacksA(cb.get_unp_d, cb.get_ovl_mat, cb.get_1el_mat, cb.get_2el_mat, cb.get_xc_mat, cb.get_xc_mat_unpert, cb.get_rsp_sol)
cb_b = cb.callbacksB(cb.get_ovl_exp, cb.get_nuc_exp, cb.get_1el_exp, cb.get_2el_exp, cb.get_xc_exp)
cb_m = cb.callbacksGM(cb.GM_zero, cb.GM_kAB, cb.GM_k1APk2B, cb.GM_TrABt)

callbacks = cb.allCallbacks(cb_a, cb_b, cb_m)

import functools


def function_call_counter(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        # print(f"-----> Function '{func.__name__}' called {wrapper.count} times.")
        print(f"-----> #{wrapper.count} - function '{func.__name__}' call")
        print(f"-----> Arguments: {args}, {kwargs}")
        return func(*args, **kwargs)

    wrapper.count = 0
    return wrapper
# openrsp.rsp_energy_recurse = function_call_counter(openrsp.rsp_energy_recurse)
# openrsp.rsp_energy_recurse = function_call_counter(openrsp.rsp_energy_recurse)

# Build perturbation tuples

# Perturbation templates
geo_templ = openrsp.rspPert('GEO', 0.0)
el_0_templ = openrsp.rspPert('EL', 0.0)
el_w_templ = openrsp.rspPert('EL', 0.08)
el_mw_templ = openrsp.rspPert('EL', -0.08)
# --------------------------------------------------------------------------

# dipole moment mu_beta /dQ
perts_mu_Q = [copy.deepcopy(geo_templ), copy.deepcopy(el_0_templ)]

# Make pert tuple instance
p_tuple_mu_Q = [rspPertTuple(perts_mu_Q)]

# k rule choice
k_mu_Q = 0

# Choose which components of Hessian will be calculated
comps_mu_Q = set([((1,1),), ((1,2),), ((2,2),), ((2,3),)])

# Set up cache datatype
mu_Q = rspCache(p_tuple_mu_Q, k=k_mu_Q, comps=comps_mu_Q)
# --------------------------------------------------------------------------

# dipole moment mu_beta /dQ/dQ
perts_mu_QQ = [copy.deepcopy(geo_templ), copy.deepcopy(geo_templ), copy.deepcopy(el_0_templ)]

# Make pert tuple instance
p_tuple_mu_QQ = [rspPertTuple(perts_mu_QQ)]

# k rule choice
k_mu_QQ = 1

# Choose which components of Hessian will be calculated
comps_mu_QQ = set([((1,1,1),), ((1,2,3),), ((2,2,3),), ((2,3,3),)])

# Set up cache datatype
mu_QQ = rspCache(p_tuple_mu_QQ, k=k_mu_QQ, comps=comps_mu_QQ)
# --------------------------------------------------------------------------

# polarizability alpha_Q
perts_alpha_Q = [copy.deepcopy(geo_templ), copy.deepcopy(el_0_templ), copy.deepcopy(el_0_templ)]

p_tuple_alpha_Q = [rspPertTuple(perts_alpha_Q)]
k_alpha_Q = 1
comps_alpha_Q = set([((1,1,1),), ((1,2,3),), ((2,2,3),), ((2,3,3),)])

# Not specifying any components here and that means all components
alpha_Q = rspCache(p_tuple_alpha_Q, k=k_alpha_Q, comps=comps_alpha_Q)
# --------------------------------------------------------------------------

# polarizability alpha_QQ
perts_alpha_QQ = [copy.deepcopy(geo_templ), copy.deepcopy(geo_templ), copy.deepcopy(el_0_templ), copy.deepcopy(el_0_templ)]

p_tuple_alpha_QQ = [rspPertTuple(perts_alpha_QQ)]
k_alpha_QQ = 1
comps_alpha_QQ = set([((1,1,1,1),), ((1,2,1,1),), ((1,2,2,2),), ((2,2,1,1),), ((2,2,2,2),), ((2,3,2,2),)])

# Not specifying any components here and that means all components
alpha_QQ = rspCache(p_tuple_alpha_QQ, k=k_alpha_QQ, comps=comps_alpha_QQ)
# --------------------------------------------------------------------------

# F_abc
perts_F_abc = [copy.deepcopy(geo_templ), copy.deepcopy(geo_templ), copy.deepcopy(geo_templ)]

p_tuple_F_abc = [rspPertTuple(perts_F_abc)]
k_F_abc = 1
comps_F_abc = set([((0,0,0),), ((0,1,2),), ((1,1,2),), ((1,2,2),)])

# Not specifying any components here and that means all components
F_abc = openrsp.rspCache(p_tuple_F_abc, k=k_F_abc, comps=comps_F_abc)

props_list = [mu_Q, mu_QQ, alpha_Q, alpha_QQ, F_abc]
# props_list = [F_abc]

props_list = openrsp.get_rsp_props(props_list, callbacks)

# print(props_list[0])
sys.path.remove(dir5_path)

# print(props_list[0].vals)
# t = props_list[1].vals


