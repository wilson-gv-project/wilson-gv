#!/usr/bin/env python
import numpy as np
from pandas.tests.tseries.frequencies.test_inference import freqs

from wilson.spectrum.term_evaluation import Term_nD, TermsEvaluator
from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices
from wilson.utils import prep_data_load

from CQCParse.parsing import GaussianParser, GaussianOutput
from CQCParse.relay import DataVault

import wilson.debug as debug
import CQCParse.debug as cqc_debug
debug.level = 0
cqc_debug.level = 0

allterms_str = { 0:
                     # {'resonance': (('a+b,a', 'zero,a'), None),
                     {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': None,
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_QQ', ('a', 'b',), ('G',))),
                      'non_averaged_props': None,
                      'vibene_denom': ('a','b',),
                      'termB_pref': 1.,
                      'termA_pref': 1/24},
                 1:
                     {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': None,
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_QQ', ('a', 'b',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                      'non_averaged_props': None,
                      'vibene_denom': ('a','b',),
                      'termB_pref': 1.,
                      'termA_pref': 1/24},
                 2:
                     {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': ('a+b+c,zero', 'c,a+b'),
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('c',), ('G',))),
                      'non_averaged_props': (('F', ('a', 'b', 'c',)),),
                      'vibene_denom': ('a','b','c'),
                      'termB_pref': 1.,
                      'termA_pref': -1/48.},
                 3:
                     {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': ('a+c,b', 'b+c,a'),
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('c',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                      'non_averaged_props': (('F', ('a', 'c', 'b',)),), # non_averaged_props , no empty tuple
                      'vibene_denom': ('a','b','c'),
                      'termB_pref': 1.,
                      'termA_pref': -1/48.},
                 4:
                     {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': ('a,a+b', 'b,zero'),
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('a',), ('G',))),
                      'non_averaged_props': ('F', ('b', 'c', 'c',)),
                      'vibene_denom': ('a','b','c'),
                      'termB_pref': 0.5,
                      'termA_pref': -1/48.},
                 5:
                     {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': ('b,a+b', 'a,zero'),
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                      'non_averaged_props': (('F', ('a', 'c', 'c',)),),
                      'vibene_denom': ('a','b','c'),
                      'termB_pref': 0.5,
                      'termA_pref': -1 / 48.},
                 6:
                     {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': ('a,a+b', 'b,zero'),
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('a',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                      'non_averaged_props': (('F', ('b', 'c', 'c',)),),
                      'vibene_denom': ('a','b','c'),
                      'termB_pref': -0.5,
                      'termA_pref': -1 / 48.},
                 7:
                     {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': ('b,a+b', 'a,zero'),
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                      'CFF': ('F', ('a', 'c', 'c',), tuple()), # fixme later
                      'vibene_denom': ('a','b','c'),
                      'termB_pref': -0.5,
                      'termA_pref': -1 / 48.}
                 }

molecule, method, basis = 'FORM', 'B3LYP', 'cc_pVQZ'

Gamma = 4.7
diag_margin = 5.
start1, end1 = 1000., 3150.
step1 = 79.8
start2, end2 = 1000., 6150.
step2 = 79.8
omega1 = np.arange(start1, end1, step1)
omega2 = np.arange(start2, end2, step2)

old_new_dict = {3:0, 5:1, 2:2, 1:3, 0:4, 4:5}
elevels = 'anharm'
enelvl = True

data_vault = DataVault("/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/files_database.csv")
dataframe_gaussian = data_vault.getting_files_DB("gaussian")

aa = dataframe_gaussian[(dataframe_gaussian['code'] == molecule)
                        & (dataframe_gaussian['method'] == method)
                        & (dataframe_gaussian['basis_set'] == basis)]['g16_3quanta_full']
filename = aa.iloc[0]
gout = GaussianOutput(molecule, method, basis, 'gaussian', filename)

parser = GaussianParser(gout)
parser.load()
parsed_data = parser.parse(linear_molecule=False)

parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
parsed_data.upd_indices_several_parts(old_new_dict)
deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

print('\nderiv_data')
for k in deriv_data:
    print(k, deriv_data[k].shape)

# print('\nallstates')
# print(allstates.keys())

states = {0: 0., 1: np.zeros(6), 2: np.zeros((6,6)), 3: np.zeros((6,6,6))}
for k in allstates:
    if len(k) == 1:
        states[1][int(k[0])] = allstates[k]
    elif len(k) == 2:
        states[2][int(k[0]), int(k[1])] = allstates[k]
        states[2][int(k[1]), int(k[0])] = allstates[k]
    elif len(k) == 3:
        states[3][int(k[0]), int(k[1]), int(k[2])] = allstates[k]
        states[3][int(k[1]), int(k[0]), int(k[2])] = allstates[k]
        states[3][int(k[1]), int(k[2]), int(k[0])] = allstates[k]
        states[3][int(k[0]), int(k[2]), int(k[1])] = allstates[k]
        states[3][int(k[2]), int(k[1]), int(k[0])] = allstates[k]
        states[3][int(k[2]), int(k[0]), int(k[1])] = allstates[k]

# print(states[1])
# print()
# print(states[2])

freqs = states[1]
data = {
    (1, 1): deriv_data['mu_Q'],
    (1, 2): deriv_data['mu_QQ'],
    (2, 1): deriv_data['alpha_Q'],
    (2, 2): deriv_data['alpha_QQ'],
}

# exit()

# set up
t0 = Term_nD(0, allterms_str[0])
t1 = Term_nD(1, allterms_str[1])
t2 = Term_nD(2, allterms_str[2])
t3 = Term_nD(3, allterms_str[3])
terms = [t0, t1, t2, t3]

tts = TermsEvaluator(terms)
tts.identify_to_precalculate()

# data
# states = {0: 0.,
#           1: np.array([1, 100, 1000]),
#           2: np.array([[20, 200, 2000],
#                        [200, 40, 400],
#                        [2000, 400, 80]]),
#           3: np.array([[[1., 2., 3.],
#                         [2., 4., 5.],
#                         [3., 5., 6.]],
#
#                        [[2., 4., 5.],
#                         [4., 7., 8.],
#                         [5., 8., 9.]],
#
#                        [[3., 5., 6.],
#                         [5., 8., 9.],
#                         [6., 9., 10.]]])}
#
# freqs = np.array([2., 4., 8.])

# axes_dict_1d = {1: np.array([2., 4., 8.]), 2: np.array([8., 16., 32.])}
axes_dict_1d = {1: omega1, 2: omega2}
x, y = np.meshgrid(axes_dict_1d[1], axes_dict_1d[2])
axes_dict = {1: x, 2: y}

Nnmodes = 6

# data = {
#     (1, 1): np.arange(Nnmodes * 3).reshape((Nnmodes, 3)),
#     (1, 2): np.arange(Nnmodes * Nnmodes * 3).reshape((Nnmodes, Nnmodes, 3)),
#     (2, 1): np.arange(Nnmodes * 3 * 3).reshape((Nnmodes, 3, 3)),
#     (2, 2): np.arange(Nnmodes * Nnmodes * 3 * 3).reshape((Nnmodes, Nnmodes, 3, 3)),
# }

avrg_terms = get_AlphaBetaGammaDelta_indices(num_f=4)

alldata = [Nnmodes, data, avrg_terms, axes_dict, states, states]

# precalculate stuff
big_dict = tts.precalculate(alldata)

#
# if __name__ == "__main__":
#     main()