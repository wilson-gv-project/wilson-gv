"""
Each test creates own instances of Term2D

Duplicated intro in test_term_evaluation
"""
from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices
from wilson.spectrum.term import Term2D
from wilson.spectrum.term_evaluation import TermsEvaluator
from wilson.utils import Conditions, prep_data_load
from wilson_main import abstractions as abst

from testing_utils import require_asserts

from CQCParse.parsing import GaussianParser, GaussianOutput
from CQCParse.relay import DataVault

import wilson.debug as debug
import CQCParse.debug as cqc_debug
debug.level = 0
cqc_debug.level = 0

print()


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
                      'non_averaged_props': (('F', ('a', 'c', 'b',)),),
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
                      # 'CFF': ('F', ('a', 'c', 'c',), tuple()), # old
                      'non_averaged_props': (('F', ('a', 'c', 'c',)),),
                      'vibene_denom': ('a','b','c'),
                      'termB_pref': -0.5,
                      'termA_pref': -1 / 48.}
            }

gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)

molecule, method, basis = 'FORM', 'B3LYP', 'cc_pVQZ'

Gamma = 4.7
diag_margin = 5.

start1, end1 = 1000., 3150.
step1 = 79.8
start2, end2 = 1000., 6150.
step2 = 79.8

old_new_dict = {3:0, 5:1, 2:2, 1:3, 0:4, 4:5}

elevels = 'anharm'
enelvl = True

#######################################################################
###     CQCParse use - getting calc data
#######################################################################

data_vault = DataVault("/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/files_database.csv")
dataframe_gaussian = data_vault.getting_files_DB("gaussian")

aa = dataframe_gaussian[(dataframe_gaussian['code'] == molecule)
                        & (dataframe_gaussian['method'] == method)
                        & (dataframe_gaussian['basis_set'] == basis)]['g16_3quanta_full']
filename = aa.iloc[0]
gout = GaussianOutput(molecule, method, basis, 'gaussian', filename)

parser = GaussianParser(gout)
parser.load()

#######################################################################


@require_asserts
def test_instance():
    print()

    t0 = Term2D(0, allterms_str[0])

    assert t0.expression == allterms_str[0]
    assert t0.term_label == 'EL'
    assert t0.resonances_expr == (('a+b,a', (-1, 2)), ('zero,a', (-1,)))
    assert t0.viblevelsdiff_expr == []

    t3 = Term2D(3, allterms_str[3])
    assert t3.viblevelsdiff_expr == ('a+c,b', 'b+c,a')
    assert t3.expression['non_averaged_props'][0] == ('F', ('a', 'c', 'b'))

    # for k in t0.expression:
    #     print(k, t0.expression[k])
    print(t0)
    print()

    # for k in t3.expression:
    #     print(k, t3.expression[k])

    print(t3)
    print()
    # print(t3.expression)
    # print(t3.vibstatesdiff_objs)



@require_asserts
def test_load_data():
    print()

    t0 = Term2D(0, allterms_str[0])
    parsed_data = parser.parse(linear_molecule=False)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data)

    before = set(t0.__dict__.keys())
    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)
    after = set(t0.__dict__.keys())
    new_attrs = after - before
    print('    ---->  New attributes after term.'
          'load_calc_data to term:', new_attrs)

    assert t0.harmonic_states[('zero',)] == 0.
    assert t0.harmonic_states_Eh[('zero',)] == 0.
    assert t0.allstates[("zero",)] == 0.
    assert t0.allstates_Eh[("zero",)] == 0.

    assert t0.harmonic_states[('0',)] == 2878.687 # indices are unchanged yet
    assert t0.harmonic_states[('2',)] == 1534.549 # indices are unchanged yet

    assert t0.allstates[("1",)] == 1794.540 # gaussian anharmonic freqs; indices are unchanged yet
    assert list(t0.properties_data.keys()) == ['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc']
    assert t0.mode_indices == [0, 1, 2, 3, 4, 5]

    parsed_data = parser.parse(linear_molecule=False)
    # vpt2 freqs now
    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data)

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)

    assert t0.allstates[('1',)] == 1794.5406564861917 # still unchanged indices

    # UPDATING INDICES NOW!
    parsed_data.upd_indices_several_parts(old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data)

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)
    assert t0.allstates[('3',)] == 1794.5406564861917


def test_amplitude_1term_single_point():
    print()

    t0 = Term2D(0, allterms_str[0])
    parsed_data = parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    print(allstates)
    print(harmonic_states)

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)

    amplitude_single = t0.get_intensity(2682.766, 3916.797, 3.8, 0.,
                                        collect_all=True, sel_abs=[(5,0)])

    # term.get_intensity is using:
    #       self.mode_indices;
    #       self.get_resonance_location(); self.get_intensity_ab()[0];
    # term.get_intensity_ab is using:
    #       self.term_label;
    #       self.get_full_factor(); self.get_res_factor(); self.get_factor_summed()
    #   result = (product_all*self.get_res_factor(w1, w2, a, b, Gamma_rc, condition))

    # assert
    print(t0.get_resonance_location(5, 0))
    print(amplitude_single)


def test_get_resonance_location_general_mock():
    print()

    t0 = Term2D(0, allterms_str[0])
    parsed_data = parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)
    # import numpy as np
    # axes_dict_1d = {1: np.array([2., 4., 8.]), 2: np.array([8., 16., 32.])}
    # x,y = np.meshgrid(axes_dict_1d[1], axes_dict_1d[2])
    # axes_dict = {1: x, 2: y}

    # amplitude_single = t0.get_intensity(2682.766, 3916.797, 3.8, 0.,
    #                                     collect_all=True, sel_abs=[(5,0)])
    import numpy as np

    a = {('a', 'b'): np.array([[0.25    , 0.125   , 0.0625  ],
       [0.125   , 0.0625  , 0.03125 ],
       [0.0625  , 0.03125 , 0.015625]]), ('a', 'b', 'c'): np.array([[[0.125      , 0.0625     , 0.03125    ],
        [0.0625     , 0.03125    , 0.015625   ],
        [0.03125    , 0.015625   , 0.0078125  ]],

       [[0.0625     , 0.03125    , 0.015625   ],
        [0.03125    , 0.015625   , 0.0078125  ],
        [0.015625   , 0.0078125  , 0.00390625 ]],

       [[0.03125    , 0.015625   , 0.0078125  ],
        [0.015625   , 0.0078125  , 0.00390625 ],
        [0.0078125  , 0.00390625 , 0.001953125]]])}
    b = {((1, 1), (2, 1), (1, 2)): np.array([[  12. ,  106.8,  298.8],
       [ 312. , 1249.2, 2575.2],
       [1000.8, 3655.2, 6990. ]]), ((1, 1), (2, 2), (1, 1)): np.array([[  12. ,  106.8,  298.8],
       [ 243.6, 1180.8, 2506.8],
       [ 766.8, 3421.2, 6756. ]]), ((1, 1), (2, 1), (1, 1)): np.array([[[  12. ,   31.8,   51.6],
        [  38.4,  106.8,  175.2],
        [  64.8,  181.8,  298.8]],

       [[  38.4,  106.8,  175.2],
        [ 129.6,  392.4,  655.2],
        [ 220.8,  678. , 1135.2]],

       [[  64.8,  181.8,  298.8],
        [ 220.8,  678. , 1135.2],
        [ 376.8, 1174.2, 1971.6]]])}
    c = {(1, -2): np.array([[ -6.,  -4.,   0.],
       [-14., -12.,  -8.],
       [-30., -28., -24.]]), (1,): np.array([[2., 4., 8.],
       [2., 4., 8.],
       [2., 4., 8.]])}
    d = {(0, 1): np.array([   -1.,  -100., -1000.]), (1, 2): np.array([[[  -19,  -199, -1999],
        [ -199,   -39,  -399],
        [-1999,  -399,   -79]],  # .reshape((-1,3))

       [[   80,  -100, -1900],
        [ -100,    60,  -300],
        [-1900,  -300,    20]],

       [[  980,   800, -1000],
        [  800,   960,   600],
        [-1000,   600,   920]]]), (1, 1): np.array([[   0,  -99, -999],
       [  99,    0, -900],
       [ 999,  900,    0]]), (0, 3): np.array([[[ -1.,  -2.,  -3.],
        [ -2.,  -4.,  -5.],
        [ -3.,  -5.,  -6.]],

       [[ -2.,  -4.,  -5.],
        [ -4.,  -7.,  -8.],
        [ -5.,  -8.,  -9.]],

       [[ -3.,  -5.,  -6.],
        [ -5.,  -8.,  -9.],
        [ -6.,  -9., -10.]]])}

    precalc_data = {'vibene_denoms': a,
                    'avrg_tensors': b,
                    'res_conds': c,
                    'vibdiffs': d}
    t0.precalc_data = precalc_data
    print(t0.get_resonance_location_general(0,2))


def test_get_resonance_location_general_real():
    print()

    t0 = Term2D(0, allterms_str[0])
    parsed_data = parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)
    import numpy as np
    te = TermsEvaluator([t0])
    freqs = np.array([t0.allstates[k] for k in t0.allstates if len(k)==1])
    Nnmodes = 6
    print(t0.properties_data.keys())
    data = {
        (1, 1): t0.properties_data['mu_Q'],
        (1, 2): t0.properties_data['mu_QQ'],
        (2, 1): t0.properties_data['alpha_Q'],
        (2, 2): t0.properties_data['alpha_QQ'],
    }
    avrg_terms = get_AlphaBetaGammaDelta_indices(num_f=4)

    axis1 = abst.spectralAxis({1: 1}, range_style='custom')
    axis1.range = np.arange(start1, end1, step1)
    axis2 = abst.spectralAxis({2: 1}, range_style='custom')
    axis2.range = np.arange(start2, end2, step2)
    axes = abst.spectralGrid({1: axis1, 2: axis2}, range_style='custom')

    # print('::::::: ', type(axes.a[1])) # wilson_main.abstractions.spectralAxis
    x,y = np.meshgrid(axes.a[1].range, axes.a[2].range)
    axes_dict = {1: x, 2: y}

    alldata = [freqs, Nnmodes, data, avrg_terms, axes_dict, t0.states_arrays] # todo: set this up better
    te.identify_to_precalculate()
    big_dict = te.precalculate(alldata)
    print('big_dict.keys()', big_dict.keys())

    # amplitude_single = t0.get_intensity(2682.766, 3916.797, 3.8, 0.,
    #                                     collect_all=True, sel_abs=[(5,0)])
    t0.precalc_data = big_dict
    print(t0.get_resonance_location_general(4, 4))


def test_axes():
    """
    spectralAxis, spectralGrid
    """
    import numpy as np
    axis1 = abst.spectralAxis({1: 1}, range_style='custom')
    axis1.range = np.arange(start1, end1, step1)
    axis2 = abst.spectralAxis({2: 1}, range_style='custom')
    axis2.range = np.arange(start2, end2, step2)
    axes = abst.spectralGrid({1: axis1, 2: axis2}, range_style='custom')

    r_expr = (('b,a', (-1, 2)), ('zero,a', (-1,)))
    res_conds = sorted([i[1] for i in r_expr],key=len)

    print(res_conds)
    # print(sorted([(2,), (1, 2), (-1, 3)],key=len)) # todo: test case
    print('axes', axes)
    print('axes.a', axes.a)
    for aa in axes.a:
        # print('aa, axes.a[aa], axes.a[aa].fv', aa, axes.a[aa], axes.a[aa].fv)
        print(axes.a[aa].range)


# def test_amplitude_1term_grid():
#     print()
#     parsed_data = parser.parse(linear_molecule=False)
#
#     parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
#     parsed_data.upd_indices_several_parts(old_new_dict)
#     deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func
#
#     t0 = Term2D(0, allterms_str[0])
#     t1 = Term2D(1, allterms_str[1])
#     t2 = Term2D(2, allterms_str[2])
#     t3 = Term2D(3, allterms_str[3])
#     terms = [t0, t1, t2, t3]
#
#     t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
#                       mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)
#     t1.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
#                       mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)
#     t2.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
#                       mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)
#     t3.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
#                       mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)
#
#     w1, w2 = np.arange(start1, end1, step1), np.arange(start2, end2, step2)
#     w1m, w2m = np.meshgrid(w1, w2)
#
#     amplitudes = 0.
#     for t in terms:
#         amplitudes += t.get_intensity(w1m, w2m, 3.8, 0.)
#
#     # print(amplitudes.shape)
#     # print(amplitudes)
