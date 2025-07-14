"""
Testing TermsEvaluator.

Each test creates own instances of Term2D.
dict_8terms is a fixture dictionary defined in wilson_intensities/tests/conftest.py. 
It contains 8 terms expressions in dict format.
"""
import numpy as np
from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices
from wilson.spectrum.termsEvaluator import TermsEvaluator
from wilson.spectrum import TermND
from wilson.utils.spectrum_utils import VibStatesDiff

from tests.testing_utils import require_asserts


from wilson.utils import debug
import CQCParse.debug as cqc_debug
debug.level = 0
cqc_debug.level = 0

print()



@require_asserts
def test_identify_to_precalculate(dict_8terms: dict) -> None:
    print('\n\nTesting - identify_to_precalculate')

    t0 = TermND(0, dict_8terms[0])
    t1 = TermND(1, dict_8terms[1])
    t2 = TermND(2, dict_8terms[2])
    t3 = TermND(3, dict_8terms[3])
    terms = [t0, t1, t2, t3]

    tts = TermsEvaluator(terms)
    tts.identify_to_precalculate()

    print('>> For four EVV 2D IR terms:')
    print('Unique types of resonance conditions: ', tts.unique_res_conds)
    print('Unique indices sets in orient. avrg.: ', tts.unique_avrg_tensors_all_expr)
    print('Unique vib. ene. denominators (1/omega_a/omega_b...): ', tts.unique_vibene_denoms)
    print('Unique vib diff types - tuples, all:\n', set(tts.mn_types))
    print('Unique vib diff types - tuples, all:')
    for k in set(tts.mn_types):
        print(k)

    assert set(tts.mn_types) == {VibStatesDiff((0, 1), True, (-1,), 'zero,a'),
                                 VibStatesDiff((1, 2), False),
                                 VibStatesDiff((2, 1), True, (-1, 2), 'a+b,a'),
                                 VibStatesDiff((1, 1), True, (-1, 2), 'b,a'),
                                 VibStatesDiff((3, 0), False)}
    print()

    assert sorted(tts.unique_res_conds) == sorted([('a+b,a', (-1, 2)), ('b,a', (-1, 2)), ('zero,a', (-1,))])
    assert tts.unique_avrg_tensors_all_expr == {('dipgrad', 'diphess', 'polgrad'): 2,
                                                ('dipgrad', 'dipgrad', 'polhess'): 2,
                                                ('dipgrad', 'dipgrad', 'polgrad'): 3}
    assert sorted(list(tts.unique_vibene_denoms)) == sorted(list({('a', 'b'), ('a', 'b', 'c')}))


@require_asserts
def test_outer_product_einsum() -> None:
    print()

    arr = np.array([1., 2., 4.])
    from wilson.spectrum.termsEvaluator import outer_product_einsum
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
def test_precalc_vibene_denoms(dict_8terms: dict) -> None:
    """
    """
    print()

    t0 = TermND(0, dict_8terms[0])
    t1 = TermND(1, dict_8terms[1])
    t2 = TermND(2, dict_8terms[2])
    t3 = TermND(3, dict_8terms[3])

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
def test_precalc_avrg_tensors(dict_8terms: dict) -> None:
    print()

    t0 = TermND(0, dict_8terms[0])
    tts = TermsEvaluator([t0])
    tts.identify_to_precalculate()

    gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)
    Nnmodes = 2

    data = {
        'dipgrad': np.arange(Nnmodes * 3).reshape((Nnmodes, 3)),
        'diphess': np.arange(Nnmodes * Nnmodes * 3).reshape((Nnmodes, Nnmodes, 3)),
        'polgrad': np.arange(Nnmodes * 3 * 3).reshape((Nnmodes, 3, 3)),
        'polhess': np.arange(Nnmodes * Nnmodes * 3 * 3).reshape((Nnmodes, Nnmodes, 3, 3)),
    }

    stored = tts.precalc_avrg_tensors(Nnmodes, data, gammaCompsAll[:3])
    print('\nstored.keys()', stored.keys())

    assert stored[tuple(sorted(['dipgrad', 'polgrad', 'diphess']))][1,1] == 60.4 # term 0
    assert stored[tuple(sorted(['dipgrad', 'polgrad', 'diphess']))][1,0] == 4.6 # term 0

    t1 = TermND(1, dict_8terms[1])
    t2 = TermND(2, dict_8terms[2])
    t3 = TermND(3, dict_8terms[3])

    tts = TermsEvaluator([t0, t1, t2, t3])
    tts.identify_to_precalculate()
    stored = tts.precalc_avrg_tensors(Nnmodes, data, gammaCompsAll)

    assert stored[tuple(sorted(['dipgrad', 'polgrad', 'dipgrad']))][1,1,1] == 392.4
    assert stored[tuple(sorted(['dipgrad', 'polhess', 'dipgrad']))][0,0] == 12.
    assert stored[tuple(sorted(['dipgrad', 'polgrad', 'dipgrad']))][1,0,1] == 106.8

    # assert sorted(list(stored.keys())) == sorted([((1, 1), (2, 1), (1, 2)),
    #                                ((1, 1), (2, 1), (1, 1)),
    #                                ((1, 1), (2, 2), (1, 1))])
    assert sorted(list(stored.keys())) == [('dipgrad', 'dipgrad', 'polgrad'),
                                           ('dipgrad', 'dipgrad', 'polhess'),
                                           ('dipgrad', 'diphess', 'polgrad')]


@require_asserts
def test_precalc_res_conds(dict_8terms: dict) -> None:
    print('\n\nTesting - Precalculate Resonance Conditions')

    t0 = TermND(0, dict_8terms[0])
    t1 = TermND(1, dict_8terms[1])
    t2 = TermND(2, dict_8terms[2])
    t3 = TermND(3, dict_8terms[3])
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


    axes_dict_1d = {1: np.array([2., 4., 8.]), 2: np.array([8., 16., 32.])}
    x,y = np.meshgrid(axes_dict_1d[1], axes_dict_1d[2])
    axes_dict = {1: x, 2: y}
    print('axes_dict\n', axes_dict)

    # freqs = np.array([2., 4., 8.])
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

    # freqs = np.array([2., 4., 8.])
    pf_types = tts.precalc_res_conds(axes_dict)

    print('\nresult of precalc\n', pf_types)


@require_asserts
def test_precalc_vibdiffs(dict_8terms: dict) -> None:
    """
    simple states indexing
    """
    print()

    t0 = TermND(0, dict_8terms[0])
    t1 = TermND(1, dict_8terms[1])
    t2 = TermND(2, dict_8terms[2])
    t3 = TermND(3, dict_8terms[3])
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
def test_precalculate(dict_8terms: dict) -> None:
    print()

    t0 = TermND(0, dict_8terms[0])
    t1 = TermND(1, dict_8terms[1])
    t2 = TermND(2, dict_8terms[2])
    t3 = TermND(3, dict_8terms[3])
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

    # freqs = np.array([2., 4., 8.])

    axes_dict_1d = {1: np.array([2., 4., 8.]), 2: np.array([8., 16., 32.])}
    x,y = np.meshgrid(axes_dict_1d[1], axes_dict_1d[2])
    axes_dict = {1: x, 2: y}
    Nnmodes = 3
    props_data_ready = {
        'dipgrad': np.arange(Nnmodes * 3).reshape((Nnmodes, 3)),
        'diphess': np.arange(Nnmodes * Nnmodes * 3).reshape((Nnmodes, Nnmodes, 3)),
        'polgrad': np.arange(Nnmodes * 3 * 3).reshape((Nnmodes, 3, 3)),
        'polhess': np.arange(Nnmodes * Nnmodes * 3 * 3).reshape((Nnmodes, Nnmodes, 3, 3)),
    }
    avrg_terms = get_AlphaBetaGammaDelta_indices(num_f=4)

    from wilson.spectrum import DataForPrecalc
    alldata = DataForPrecalc(Nnmodes=Nnmodes,
                             props_data=props_data_ready,
                             avrg_terms=avrg_terms,
                             axes_dict=axes_dict,
                             states_arrays_Eh=states,
                             harmonic_arrays_Eh=states)

    precalc_dict = tts.precalculate(alldata)

    print('\nPrecalculated stuff\n')
    for k in precalc_dict:
        print('   >>>', k)
        print(precalc_dict[k])
        print('---')

