import numpy as np
from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices
from wilson.spectrum.term_evaluation import Term2D, TermsEvaluator
from wilson.utils import Conditions, prep_data_load

from CQCParse.parsing import GaussianParser, GaussianOutput
from CQCParse.relay import DataVault

print()

# allterms_str =     {
#     0: ((('a+b,a', 'zero,a'), None), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))), 1/24),
#     1: ((('b,a', 'zero,a'), None), (('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))), 1/24)),
#     2: ((('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc', 1.), -1/48.),
#     3: ((('b,a', 'zero,a'), ('a+c,b', 'b+c,a')), (('mu_Q', ('a',)), ('alpha_Q', ('c',)), ('mu_Q', ('b',)), 'acb', 1.), -1/48.),
#     4: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc', 0.5), -1/48.),
#     5: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc', 0.5), -1/48.),
#     6: ((('b,a', 'zero,a'), ('a,a+b', 'b,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc', -0.5), -1/48.),
#     7: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc', -0.5, -1/48.))
#     }

allterms_str = { 0:
                     # {'resonance': (('a+b,a', 'zero,a'), None),
                     {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': None,
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_QQ', ('a', 'b',), ('G',))),
                      'CFF': None,
                      'vibene_denom': ('a','b',),
                      'termB_pref': 1.,
                      'termA_pref': 1/24},
                 1:
                     # {'resonance': (('b,a', 'zero,a'), None),
                     {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': None,
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_QQ', ('a', 'b',), ('A', 'D')), ('mu_Q', ('b',), ('G',))),
                      'CFF': None,
                      'vibene_denom': ('a','b',),
                      'termB_pref': 1.,
                      'termA_pref': 1/24},
                 2:
                     # {'resonance': (('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')),
                     {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': ('a+b+c,zero', 'c,a+b'),
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('c',), ('G',)), 'abc', 1.),
                      'CFF': ('F', ('a', 'b', 'c',), tuple()),
                      'vibene_denom': ('a','b','c'),
                      'termB_pref': 1.,
                      'termA_pref': -1/48.},
                 3:
                     # {'resonance': (('b,a', 'zero,a'), ('a+c,b', 'b+c,a')),
                     {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': ('a+c,b', 'b+c,a'),
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('c',), ('A', 'D')), ('mu_Q', ('b',), ('G',)), 'acb', 1.),
                      'CFF': ('F', ('a', 'c', 'b',), tuple()),
                      'vibene_denom': ('a','b','c'),
                      'termB_pref': 1.,
                      'termA_pref': -1/48.},
                 4:
                     # {'resonance': (('b,a', 'zero,a'), ('b,a+b', 'a,zero')),
                     {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': ('a,a+b', 'b,zero'),
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('a',), ('G',)), 'bcc', 0.5),
                      'CFF': ('F', ('b', 'c', 'c',), tuple()),
                      'vibene_denom': ('a','b','c'),
                      'termB_pref': 0.5,
                      'termA_pref': -1/48.},
                 5:
                     # {'resonance': (('b,a', 'zero,a'), ('b,a+b', 'a,zero')),
                     {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': ('b,a+b', 'a,zero'),
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('b',), ('G',)), 'acc', 0.5),
                      'CFF': ('F', ('a', 'c', 'c',), tuple()),
                      'vibene_denom': ('a','b','c'),
                      'termB_pref': 0.5,
                      'termA_pref': -1 / 48.},
                 6:
                     # {'resonance': (('b,a', 'zero,a'), ('a,a+b', 'b,zero')),
                     {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': ('a,a+b', 'b,zero'),
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('a',), ('A', 'D')), ('mu_Q', ('b',), ('G',)), 'bcc', -0.5),
                      'CFF': ('F', ('b', 'c', 'c',), tuple()),
                      'vibene_denom': ('a','b','c'),
                      'termB_pref': -0.5,
                      'termA_pref': -1 / 48.},
                 7:
                     # {'resonance': (('b,a', 'zero,a'), ('b,a+b', 'a,zero')),
                     {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
                      'vibenediff': ('b,a+b', 'a,zero'),
                      'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('b',), ('G',)), 'acc', -0.5),
                      'CFF': ('F', ('a', 'c', 'c',), tuple()),
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
parsed_data = parser.parse(linear_molecule=False)

deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data)

#######################################################################


def test_instance():
    print()

    t0 = Term2D(0, allterms_str[0])

    assert t0.expression == allterms_str[0]
    assert t0.term_label == 'EL'
    assert t0.resonances_expr == ('a+b,a', 'zero,a')
    assert t0.viblevelsdiff_expr is None


def test_load_data():
    print()

    t0 = Term2D(0, allterms_str[0])
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data)

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)

    assert t0.harmonic_states[('zero',)] == 0.
    assert t0.harmonic_states_Eh[('zero',)] == 0.
    assert t0.allstates[("zero",)] == 0.
    assert t0.allstates_Eh[("zero",)] == 0.

    assert t0.harmonic_states[('0',)] == 2878.687 # indices are unchanged yet
    assert t0.harmonic_states[('2',)] == 1534.549 # indices are unchanged yet

    assert t0.allstates[("1",)] == 1794.540 # gaussian anharmonic freqs; indices are unchanged yet
    assert list(t0.properties_data.keys()) == ['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc']
    assert t0.mode_indices == [0, 1, 2, 3, 4, 5]

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
    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)
    amplitude_single = t0.get_intensity(2726., 3967., 3.8, 0.,
                                        collect_all=True, sel_abs=[(0,5)])
    print(t0.get_resonance_location(0, 5))
    print(amplitude_single)


def test_amplitude_1term_grid():
    print()

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0 = Term2D(0, allterms_str[0])
    t1 = Term2D(1, allterms_str[1])
    t2 = Term2D(2, allterms_str[2])
    t3 = Term2D(3, allterms_str[3])
    terms = [t0, t1, t2, t3]

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)
    t1.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)
    t2.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)
    t3.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=gammaCompsAll)

    w1, w2 = np.arange(start1, end1, step1), np.arange(start2, end2, step2)
    w1m, w2m = np.meshgrid(w1, w2)

    amplitudes = 0.
    for t in terms:
        amplitudes += t.get_intensity(w1m, w2m, 3.8, 0.)

    print(amplitudes.shape)
    print(amplitudes)


def test_identify_to_precalculate():
    print()

    t0 = Term2D(0, allterms_str[0])
    t1 = Term2D(1, allterms_str[1])
    t2 = Term2D(2, allterms_str[2])
    t3 = Term2D(3, allterms_str[3])
    terms = [t0, t1, t2, t3]

    tts = TermsEvaluator(terms)
    tts.identify_to_precalculate()

    print('Unique types of resonace conditions: ', tts.unique_res_conds)
    print('Unique indices sets in orient. avrg.: ', tts.unique_avrg_tensors_all)
    print('Unique vib. ene. denominators (1/omega_a/omega_b...): ', tts.unique_vibene_denoms)


def test_outer_product_einsum():
    print()

    arr = np.array([1., 2., 4.])
    from wilson.spectrum.term_evaluation import outer_product_einsum
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


def test_precalc_vibene_denoms():
    """
    later
    """
    pass

def test_precalc_avrg_tensors():
    print()
    import wilson.debug as debug
    debug.level = 0

    t0 = Term2D(0, allterms_str[0])
    tts = TermsEvaluator([t0])
    tts.identify_to_precalculate()

    gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)
    Nnmodes = 2

    dat = {
        (1, 1): np.arange(Nnmodes * 3).reshape((Nnmodes, 3)),
        (1, 2): np.arange(Nnmodes * Nnmodes * 3).reshape((Nnmodes, Nnmodes, 3)),
        (2, 1): np.arange(Nnmodes * 3 * 3).reshape((Nnmodes, 3, 3)),
        (2, 2): np.arange(Nnmodes * Nnmodes * 3 * 3).reshape((Nnmodes, Nnmodes, 3, 3)),
    }

    stored = tts.precalc_avrg_tensors(Nnmodes, dat, gammaCompsAll[:3])
    assert stored[0][1,1] == 60.4
    assert stored[0][1,0] == 4.6

    t1 = Term2D(1, allterms_str[1])
    t2 = Term2D(2, allterms_str[2])
    t3 = Term2D(3, allterms_str[3])

    tts = TermsEvaluator([t0, t1, t2, t3])
    tts.identify_to_precalculate()
    stored = tts.precalc_avrg_tensors(Nnmodes, dat, gammaCompsAll)

    assert stored[3][1,1,1] == 392.4
    assert stored[1][0,0] == 12.
    assert stored[3][1,0,1] == 106.8

    assert list(stored.keys()) == [0,1,3]


def test_precalc_res_conds():
    pass

def test_precalculate():
    pass
