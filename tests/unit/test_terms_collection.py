"""
Each test creates own instances of Term2D

"""
import numpy as np
from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices
from wilson.spectrum.terms_collection import TermsEvaluator
from wilson.spectrum.term_nD import Term_nD
from wilson.spectrum.termeval_util_classes import VibStatesDiff
from wilson.utils import Conditions, prep_data_load

from tests.testing_utils import require_asserts

from CQCParse.parsing import GaussianParser, GaussianOutput
from CQCParse.relay import DataVault

import wilson.debug as debug
import CQCParse.debug as cqc_debug
debug.level = 0
cqc_debug.level = 0

print()

#
# allterms_str = { 0:
#                      # {'resonance': (('a+b,a', 'zero,a'), None),
#                      {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
#                       'vibenediff': None,
#                       'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_QQ', ('a', 'b',), ('G',))),
#                       'non_averaged_props': None,
#                       'vibene_denom': ('a','b',),
#                       'termB_pref': 1.,
#                       'termA_pref': 1/24},
#                  1:
#                      {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
#                       'vibenediff': None,
#                       'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_QQ', ('a', 'b',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
#                       'non_averaged_props': None,
#                       'vibene_denom': ('a','b',),
#                       'termB_pref': 1.,
#                       'termA_pref': 1/24},
#                  2:
#                      {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
#                       'vibenediff': ('a+b+c,zero', 'c,a+b'),
#                       'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('c',), ('G',))),
#                       'non_averaged_props': (('F', ('a', 'b', 'c',)),),
#                       'vibene_denom': ('a','b','c'),
#                       'termB_pref': 1.,
#                       'termA_pref': -1/48.},
#                  3:
#                      {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
#                       'vibenediff': ('a+c,b', 'b+c,a'),
#                       'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('c',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
#                       'non_averaged_props': (('F', ('a', 'c', 'b',)),), # non_averaged_props , no empty tuple
#                       'vibene_denom': ('a','b','c'),
#                       'termB_pref': 1.,
#                       'termA_pref': -1/48.},
#                  4:
#                      {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
#                       'vibenediff': ('a,a+b', 'b,zero'),
#                       'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('a',), ('G',))),
#                       'non_averaged_props': ('F', ('b', 'c', 'c',)),
#                       'vibene_denom': ('a','b','c'),
#                       'termB_pref': 0.5,
#                       'termA_pref': -1/48.},
#                  5:
#                      {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
#                       'vibenediff': ('b,a+b', 'a,zero'),
#                       'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
#                       'non_averaged_props': (('F', ('a', 'c', 'c',)),),
#                       'vibene_denom': ('a','b','c'),
#                       'termB_pref': 0.5,
#                       'termA_pref': -1 / 48.},
#                  6:
#                      {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
#                       'vibenediff': ('a,a+b', 'b,zero'),
#                       'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('a',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
#                       'non_averaged_props': (('F', ('b', 'c', 'c',)),),
#                       'vibene_denom': ('a','b','c'),
#                       'termB_pref': -0.5,
#                       'termA_pref': -1 / 48.},
#                  7:
#                      {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
#                       'vibenediff': ('b,a+b', 'a,zero'),
#                       'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
#                       'CFF': ('F', ('a', 'c', 'c',), tuple()), # fixme later
#                       'vibene_denom': ('a','b','c'),
#                       'termB_pref': -0.5,
#                       'termA_pref': -1 / 48.}
#             }
#
#
# gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)
#
# molecule, method, basis = 'FORM', 'B3LYP', 'cc_pVQZ'
#
# Gamma = 4.7
# diag_margin = 5.
#
# start1, end1 = 1000., 3150.
# step1 = 79.8
# start2, end2 = 1000., 6150.
# step2 = 79.8
#
# old_new_dict = {3:0, 5:1, 2:2, 1:3, 0:4, 4:5}
#
# elevels = 'anharm'
# enelvl = True
#
# #######################################################################
# ###     CQCParse use - getting calc data
# #######################################################################
#
# data_vault = DataVault("/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/files_database.csv")
# dataframe_gaussian = data_vault.getting_files_DB("gaussian")
#
# aa = dataframe_gaussian[(dataframe_gaussian['code'] == molecule)
#                         & (dataframe_gaussian['method'] == method)
#                         & (dataframe_gaussian['basis_set'] == basis)]['g16_3quanta_full']
# filename = aa.iloc[0]
# gout = GaussianOutput(molecule, method, basis, 'gaussian', filename)
#
# parser = GaussianParser(gout)
# parser.load()
#
#######################################################################


@require_asserts
def test_identify_to_precalculate(dict_8terms):
    print('\n\nTesting - identify_to_precalculate')

    t0 = Term_nD(0, dict_8terms[0])
    t1 = Term_nD(1, dict_8terms[1])
    t2 = Term_nD(2, dict_8terms[2])
    t3 = Term_nD(3, dict_8terms[3])
    terms = [t0, t1, t2, t3]

    tts = TermsEvaluator(terms)
    tts.identify_to_precalculate()

    print('>> For four EVV 2D IR terms:')
    print('Unique types of resonance conditions: ', tts.unique_res_conds)
    print('Unique indices sets in orient. avrg.: ', tts.unique_avrg_tensors_all)
    print('Unique vib. ene. denominators (1/omega_a/omega_b...): ', tts.unique_vibene_denoms)
    print('Unique vib diff types - tuples, all:\n', set(tts.mn_types))
    print('Unique vib diff types - tuples, all:')
    for k in set(tts.mn_types):
        print(k)
    print('\nUnique vib diff types - all', set([i for t in terms for i in t.vibdiff_symbolic]))

    assert set(tts.mn_types) == {VibStatesDiff((0, 1), True, (-1,), 'zero,a'),
                                 VibStatesDiff((1, 2), False),
                                 VibStatesDiff((2, 1), True, (-1, 2), 'a+b,a'),
                                 VibStatesDiff((1, 1), True, (-1, 2), 'b,a'),
                                 VibStatesDiff((3, 0), False)}
    print()

    assert sorted(tts.unique_res_conds) == sorted([('a+b,a', (-1, 2)), ('b,a', (-1, 2)), ('zero,a', (-1,))])
    assert tts.unique_avrg_tensors_all == {((1, 1), (2, 1), (1, 2)): 2,
                                           ((1, 1), (2, 2), (1, 1)): 2,
                                           ((1, 1), (2, 1), (1, 1)): 3}
    assert sorted(list(tts.unique_vibene_denoms)) == sorted(list({('a', 'b'), ('a', 'b', 'c')}))


@require_asserts
def test_outer_product_einsum():
    print()

    arr = np.array([1., 2., 4.])
    from wilson.spectrum.terms_collection import outer_product_einsum
    # print(repr(outer_product_einsum(1./arr, 3)))
    expected_2d = np.array([[1., 0.5, 0.25],
                            [0.5, 0.25, 0.125],
                            [0.25, 0.125, 0.0625]])
    expected_3d = np.array([[[1.      , 0.5     , 0.25    ],
                             [0.5     , 0.25    , 0.125   ],
                             [0.25    , 0.125   , 0.0625  ]],

                           [[0.5     , 0.25    , 0.125   ],
                            [0.25    , 0.125   , 0.0625  ],
                            [0.125   , 0.0625  , 0.03125 ]],

                           [[0.25    , 0.125   , 0.0625  ],
                            [0.125   , 0.0625  , 0.03125 ],
                            [0.0625  , 0.03125 , 0.015625]]])
    assert np.allclose(outer_product_einsum(1./arr, 2), expected_2d)
    assert np.allclose(outer_product_einsum(1./arr, 3), expected_3d)


@require_asserts
def test_precalc_vibene_denoms(dict_8terms):
    """
    """
    print()

    t0 = Term_nD(0, dict_8terms[0])
    t1 = Term_nD(1, dict_8terms[1])
    t2 = Term_nD(2, dict_8terms[2])
    t3 = Term_nD(3, dict_8terms[3])

    tts = TermsEvaluator([t0, t1, t2, t3])
    tts.identify_to_precalculate()

    freqs = np.array([2., 4., 8.])
    qstates = {1: freqs}
    res = tts.precalc_vibene_denoms(qstates)
    print(res)
    assert sorted(list(res.keys())) == sorted([('a', 'b', 'c'), ('a', 'b')])
    assert res[('a', 'b')][0,0] == 2.*2.
    assert res[('a', 'b')][0,1] == 2.*4.
    assert res[('a', 'b')][1,2] == 4.*8.
    assert res[('a', 'b')][1,2] == res[('a', 'b')][2,1]

    assert res[('a', 'b', 'c')][2,2,0] == 8.*8.*2.
    assert res[('a', 'b', 'c')][1,0,2] == 1.*4.*2.*8.
    assert res[('a', 'b', 'c')][1,0,2] == res[('a', 'b', 'c')][0,1,2]


@require_asserts
def test_precalc_avrg_tensors(dict_8terms):
    print()

    t0 = Term_nD(0, dict_8terms[0])
    tts = TermsEvaluator([t0])
    tts.identify_to_precalculate()

    gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)
    Nnmodes = 2

    data = {
        (1, 1): np.arange(Nnmodes * 3).reshape((Nnmodes, 3)),
        (1, 2): np.arange(Nnmodes * Nnmodes * 3).reshape((Nnmodes, Nnmodes, 3)),
        (2, 1): np.arange(Nnmodes * 3 * 3).reshape((Nnmodes, 3, 3)),
        (2, 2): np.arange(Nnmodes * Nnmodes * 3 * 3).reshape((Nnmodes, Nnmodes, 3, 3)),
    }

    stored = tts.precalc_avrg_tensors(Nnmodes, data, gammaCompsAll[:3])
    assert stored[((1, 1), (2, 1), (1, 2))][1,1] == 60.4 # term 0
    assert stored[((1, 1), (2, 1), (1, 2))][1,0] == 4.6 # term 0

    t1 = Term_nD(1, dict_8terms[1])
    t2 = Term_nD(2, dict_8terms[2])
    t3 = Term_nD(3, dict_8terms[3])

    tts = TermsEvaluator([t0, t1, t2, t3])
    tts.identify_to_precalculate()
    stored = tts.precalc_avrg_tensors(Nnmodes, data, gammaCompsAll)

    assert stored[((1, 1), (2, 1), (1, 1))][1,1,1] == 392.4
    assert stored[((1, 1), (2, 2), (1, 1))][0,0] == 12.
    assert stored[((1, 1), (2, 1), (1, 1))][1,0,1] == 106.8

    assert sorted(list(stored.keys())) == sorted([((1, 1), (2, 1), (1, 2)),
                                   ((1, 1), (2, 1), (1, 1)),
                                   ((1, 1), (2, 2), (1, 1))])


@require_asserts
def test_precalc_res_conds(dict_8terms):
    print('\n\nTesting - Precalculate Resonance Conditions')

    t0 = Term_nD(0, dict_8terms[0])
    t1 = Term_nD(1, dict_8terms[1])
    t2 = Term_nD(2, dict_8terms[2])
    t3 = Term_nD(3, dict_8terms[3])
    terms = [t0, t1, t2, t3]

    tts = TermsEvaluator(terms)
    tts.identify_to_precalculate()

    print('>> For four EVV 2D IR terms:')
    print('unique_res_conds', tts.unique_res_conds)
    assert sorted(tts.unique_res_conds) == sorted(tts.unique_res_conds)

    set_rc_types = set([i[1] for i in tts.unique_res_conds])
    print('set of res cond types', set_rc_types)
    assert sorted(list(set_rc_types)) == sorted(list({(-1, 2), (-1,)}))

    print('set(tts.mn_types)',  set(tts.mn_types))
    print('Quanta for states involved:', set([i for v in set(tts.mn_types) for i in v.diff_type if i>0 ]))

    # set_mn_types = set([i[0] for i in tts.unique_res_conds])
    # print('set of w_m,n types', set_mn_types)
    # Unique vib diff types - all
    #   {'b,a', 'b+c,a', 'zero,a', 'a+b+c,zero', 'a+c,b', 'a+b,a', 'c,a+b'}

    axes_dict_1d = {1: np.array([2., 4., 8.]), 2: np.array([8., 16., 32.])}
    x,y = np.meshgrid(axes_dict_1d[1], axes_dict_1d[2])
    axes_dict = {1: x, 2: y}
    print('axes_dict\n', axes_dict)

    freqs = np.array([2., 4., 8.])
    pf_types = tts.precalc_res_conds(axes_dict)

    print('\nresult of precalc\n', pf_types)
    assert np.allclose(pf_types[(1, -2)], np.array([[ -6.,  -4.,   0.],
                                                    [-14., -12.,  -8.],
                                                    [-30., -28., -24.]]))

    assert np.allclose(pf_types[(1,)], np.array([[2., 4., 8.],
                                                 [2., 4., 8.],
                                                 [2., 4., 8.]]))

    axes_dict = {1: 80, 2: 800}
    print('axes_dict\n', axes_dict)

    freqs = np.array([2., 4., 8.])
    pf_types = tts.precalc_res_conds(axes_dict)

    print('\nresult of precalc\n', pf_types)
    # assert np.allclose(pf_types[(1, -2)], np.array([[ -6.,  -4.,   0.],
    #                                                 [-14., -12.,  -8.],
    #                                                 [-30., -28., -24.]]))
    #
    # assert np.allclose(pf_types[(1,)], np.array([[2., 4., 8.],
    #                                              [2., 4., 8.],
    #                                              [2., 4., 8.]]))


@require_asserts
def test_precalc_vibdiffs(dict_8terms):
    """
    simple states indexing
    """
    print()

    t0 = Term_nD(0, dict_8terms[0])
    t1 = Term_nD(1, dict_8terms[1])
    t2 = Term_nD(2, dict_8terms[2])
    t3 = Term_nD(3, dict_8terms[3])
    terms = [t0, t1, t2, t3]

    tts = TermsEvaluator(terms)
    tts.identify_to_precalculate()

    states = {0: 0.,
              1: np.array([1, 100, 1000]),
              2: np.array([[20, 200, 2000],
                           [200, 40, 400],
                           [2000, 400, 80]]),
              3: np.array([[[ 1.,  2.,  3.],
                            [ 2.,  4.,  5.],
                            [ 3.,  5.,  6.]],

                           [[ 2.,  4.,  5.],
                            [ 4.,  7.,  8.],
                            [ 5.,  8.,  9.]],

                           [[ 3.,  5.,  6.],
                            [ 5.,  8.,  9.],
                            [ 6.,  9., 10.]]])}

    assert states[3][1,0,1] == states[3][1,1,0] == states[3][0,1,1]
    assert states[3][0,1,2] == states[3][1,0,2] == states[3][2,1,0] == states[3][2,0,1]

    rrr = tts.precalc_vibdiffs(states)
    assert sorted(list(rrr.keys())) == sorted([(0, 1), (1, 2), (1, 1), (0, 3)])
    assert rrr[(0, 1)].shape == (3,)
    assert rrr[(1, 1)].shape == (3,3)
    assert rrr[(1, 2)].shape == (3,3,3)
    assert rrr[(0, 3)].shape == (3,3,3)

    assert rrr[(0, 1)][1] == -states[1][1]
    assert rrr[(1, 1)][1,2] == states[1][1] - states[1][2]
    assert rrr[(1, 1)][0,2] == states[1][0] - states[1][2]
    assert rrr[(1, 1)][2,1] == states[1][2] - states[1][1]

    assert rrr[(0, 3)][1,2,0] == -states[3][1,0,2]
    assert rrr[(1, 2)][1,2,0] == states[1][1] - states[2][2,0]
    assert rrr[(1, 2)][1,2,1] == states[1][1] - states[2][2,1] # repeated index
    assert rrr[(1, 2)][1,2,0] == states[1][1] - states[2][0,2] # symmetry
    assert rrr[(1, 2)][1,2,0] != states[1][2] - states[2][2,1] # wrong indexing

    # print(t3.vibdiff_symbolic)

    print()


@require_asserts
def test_precalculate(dict_8terms):
    print()

    t0 = Term_nD(0, dict_8terms[0])
    t1 = Term_nD(1, dict_8terms[1])
    t2 = Term_nD(2, dict_8terms[2])
    t3 = Term_nD(3, dict_8terms[3])
    terms = [t0, t1, t2, t3]

    tts = TermsEvaluator(terms)
    tts.identify_to_precalculate()

    states = {0: 0.,
              1: np.array([1, 100, 1000]),
              2: np.array([[20, 200, 2000],
                           [200, 40, 400],
                           [2000, 400, 80]]),
              3: np.array([[[1., 2., 3.],
                            [2., 4., 5.],
                            [3., 5., 6.]],

                           [[2., 4., 5.],
                            [4., 7., 8.],
                            [5., 8., 9.]],

                           [[3., 5., 6.],
                            [5., 8., 9.],
                            [6., 9., 10.]]])}

    freqs = np.array([2., 4., 8.])

    axes_dict_1d = {1: np.array([2., 4., 8.]), 2: np.array([8., 16., 32.])}
    x,y = np.meshgrid(axes_dict_1d[1], axes_dict_1d[2])
    axes_dict = {1: x, 2: y}
    Nnmodes = 3
    data = {
        (1, 1): np.arange(Nnmodes * 3).reshape((Nnmodes, 3)),
        (1, 2): np.arange(Nnmodes * Nnmodes * 3).reshape((Nnmodes, Nnmodes, 3)),
        (2, 1): np.arange(Nnmodes * 3 * 3).reshape((Nnmodes, 3, 3)),
        (2, 2): np.arange(Nnmodes * Nnmodes * 3 * 3).reshape((Nnmodes, Nnmodes, 3, 3)),
    }
    avrg_terms = get_AlphaBetaGammaDelta_indices(num_f=4)

    alldata = [Nnmodes, data, avrg_terms, axes_dict, states, states]
    big_dict = tts.precalculate(alldata)

    print('\nPrecalculated stuff\n')
    for k in big_dict:
        print('   >>>', k)
        print(big_dict[k])
        print('---')


@require_asserts
def test_TE_compute_intensity():

    w1 = np.array([1185.29, 1501.59])
    w2 = np.array([2440.61, 3290.38])