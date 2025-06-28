"""
Each test creates own instances of Term2D

Duplicated intro in test_term_evaluation : UPD - now it's in pytest fixtured in conftest.py
"""
import numpy as np

from wilson.spectrum.termND import TermND
from wilson.spectrum.termsEvaluator import TermsEvaluator
from wilson.utils import Conditions, prep_data_load
from wilson_main import abstractions as abst

from tests.testing_utils import require_asserts

import wilson.debug as debug
import CQCParse.debug as cqc_debug

debug.level = 0
cqc_debug.level = 0

print()

def setup_term(term_id, terms_dict_setup, FORM_setup_parser, spectrum_setup):
    """
    Helper function to set up a TermND instance with parsed data and loaded calculations.
    """
    term = TermND(term_id, terms_dict_setup[term_id])
    parsed_data = FORM_setup_parser.parse(linear_molecule=False)
    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data)
    term.load_calc_data(
        properties_data=deriv_data,
        allstates=allstates,
        harmonic_states=harmonic_states,
        mode_indices=mode_indices,
        gammaCompsAll=spectrum_setup.gammaCompsAll
    )
    return term


@require_asserts
def test_instance(dict_8terms):
    print()

    t0 = TermND(0, dict_8terms[0])

    assert t0.expression == dict_8terms[0]
    assert t0.term_label == 'EL'
    assert t0.resonances_expr == (('a+b,a', (-1, 2)), ('zero,a', (-1,)))
    assert t0.viblevelsdiff_expr == []

    t3 = TermND(3, dict_8terms[3])
    assert t3.viblevelsdiff_expr == ('a+c,b', 'b+c,a')
    assert t3.expression['non_averaged_props'] == (('F', ('a', 'c', 'b')),)

    print(t0)
    print()

    print(t3)
    print()


@require_asserts
def test_load_data(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()

    t0 = TermND(0, dict_8terms[0])
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data)

    before = set(t0.__dict__.keys())
    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
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

    parsed_data = MOL_setup_parser.parse(linear_molecule=False)
    # vpt2 freqs now
    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data)

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)

    assert t0.allstates[('1',)] == 1794.5406564861917 # still unchanged indices

    # UPDATING INDICES NOW!
    parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data)

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
    assert t0.allstates[('3',)] == 1794.5406564861917


@require_asserts
def test_amplitude_1term_single_point(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()

    t0 = TermND(0, dict_8terms[0])
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)

    amplitude_single = t0.get_intensity(2682.766, 3916.797, 3.8, 0.,
                                        collect_all=True, sel_abs=[(5,0)])
    # term.get_intensity is using:
    #       self.mode_indices;
    #       self.get_resonance_location(); self.get_intensity_ab()[0];
    a,b = 5,0 # (4,5), (2,3), (2,5), (0,0)
    w1,w2 = t0.get_resonance_location(a,b)
    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    print(amplitude_single)
    ampl_single_ab = t0.get_intensity_ab(a,b, w1,w2, 3.8, condition=None)[0]
    print(ampl_single_ab)
    assert amplitude_single ==ampl_single_ab


@require_asserts
def test_amplitude_1term_single_point_ab(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()

    t0 = TermND(0, dict_8terms[0])
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
    #  ---->  New attributes after term.load_calc_data to term:
    #     {'harmonic_states', 'allstates_Eh',
    #      'allstates', 'harmonic_states_Eh',
    #      'properties_data', 'gammaCompsAll', 'mode_indices'}
    print('t0.precalc_data', t0.precalc_data)
    a,b = 0,0 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location(a,b)
    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_intensity_ab(a,b, w1,w2, 3.8, condition=None,
                      debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    print('----------------------------------------------')

    a,b = 0,1 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location(a,b)
    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_intensity_ab(a,b, w1,w2, 3.8, condition=None,
                      debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    print('----------------------------------------------')

    a,b = 2,3 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location(a,b)
    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_intensity_ab(a,b, w1,w2, 3.8, condition=None,
                      debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0


@require_asserts
def test_get_resonance_location_general_mock(dict_8terms):
    print()

    t0 = TermND(0, dict_8terms[0])

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
    print(t0.get_resonance_location_general((0,2)))
    # assert


@require_asserts
def test_get_resonance_location_general_real(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()

    t0 = TermND(0, dict_8terms[0])
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
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
    avrg_terms = spectrum_setup.gammaCompsAll

    axis1 = abst.spectralAxis({1: 1}, range_style='custom')
    axis1.range = np.arange(spectrum_setup.start1, spectrum_setup.end1, spectrum_setup.step1)
    axis2 = abst.spectralAxis({2: 1}, range_style='custom')
    axis2.range = np.arange(spectrum_setup.start2, spectrum_setup.end2, spectrum_setup.step2)
    axes = abst.spectralGrid({1: axis1, 2: axis2}, range_style='custom')

    # print('::::::: ', type(axes.a[1])) # wilson_main.abstractions.spectralAxis
    x,y = np.meshgrid(axes.a[1].range, axes.a[2].range)
    axes_dict = {1: x, 2: y}

    alldata = [Nnmodes, data, avrg_terms, axes_dict, t0.states_arrays, t0.harmonic_arrays] # todo: set this up better
    te.identify_to_precalculate()
    big_dict = te.precalculate(alldata)
    # print('big_dict.keys()', big_dict.keys())

    t0.precalc_data = big_dict
    print(t0.get_resonance_location_general((4, 4)))


@require_asserts
def test_amplitude_1term_single_point_ab_precalc(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()

    t0 = TermND(0, dict_8terms[0])
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)

    a,b = 0,0 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))

    te = TermsEvaluator([t0])
    freqs = np.array([t0.allstates[k] for k in t0.allstates if len(k)==1])
    Nnmodes = 6
    data = {
        (1, 1): t0.properties_data['mu_Q'],
        (1, 2): t0.properties_data['mu_QQ'],
        (2, 1): t0.properties_data['alpha_Q'],
        (2, 2): t0.properties_data['alpha_QQ'],
    }
    avrg_terms = spectrum_setup.gammaCompsAll
    axes_dict = {1: w1, 2: w2}

    print()
    alldata = [Nnmodes, data, avrg_terms, axes_dict, t0.states_arrays_Eh, t0.harmonic_arrays_Eh] # todo: set this up better
    te.identify_to_precalculate()
    big_dict = te.precalculate(alldata)
    print(t0.states_arrays[1][a],t0.states_arrays[1][b])
    # {'vibene_denoms': a,
    #  'avrg_tensors': b,
    #  'res_conds': c,
    #  'vibdiffs': d}
    t0.precalc_data = big_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_intensity_ab(a,b, w1,w2, 3.8, condition=None,
                      debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    print('----------------------------------------------')

    a,b = 0,1 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))
    axes_dict = {1: w1, 2: w2}

    alldata = [Nnmodes, data, avrg_terms, axes_dict, t0.states_arrays_Eh, t0.harmonic_arrays_Eh] # todo: set this up better
    te.identify_to_precalculate()
    big_dict = te.precalculate(alldata)
    t0.precalc_data = big_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_intensity_ab(a,b, w1,w2, 3.8, condition=None,
                      debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    print('----------------------------------------------')

    a,b = 2,3 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))
    axes_dict = {1: w1, 2: w2}

    alldata = [Nnmodes, data, avrg_terms, axes_dict, t0.states_arrays_Eh, t0.harmonic_arrays_Eh] # todo: set this up better
    te.identify_to_precalculate()
    big_dict = te.precalculate(alldata) # make it external to each term ,
    t0.precalc_data = big_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_intensity_ab(a,b, w1,w2, 3.8, condition=None,
                      debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0

@require_asserts
def test_amplitude_1term_single_point_ab_precalc_t2(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()

    t2 = TermND(2, dict_8terms[2])
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t2.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)

    a,b = 0,3 # (4,5), (2,3), (2,5)
    w1,w2 = t2.get_resonance_location_general((a,b))
    print(w1,w2)
    te = TermsEvaluator([t2])
    freqs = np.array([t2.allstates[k] for k in t2.allstates if len(k)==1])
    Nnmodes = 6
    data = {
        (1, 1): t2.properties_data['mu_Q'],
        (1, 2): t2.properties_data['mu_QQ'],
        (2, 1): t2.properties_data['alpha_Q'],
        (2, 2): t2.properties_data['alpha_QQ'],
    }
    avrg_terms = spectrum_setup.gammaCompsAll
    axes_dict = {1: w1, 2: w2}

    print()
    alldata = [Nnmodes, data, avrg_terms, axes_dict, t2.states_arrays_Eh, t2.harmonic_arrays_Eh] # todo: set this up better
    te.identify_to_precalculate()
    big_dict = te.precalculate(alldata)
    print(t2.states_arrays[1][a],t2.states_arrays[1][b])
    # {'vibene_denoms': a,
    #  'avrg_tensors': b,
    #  'res_conds': c,
    #  'vibdiffs': d}
    t2.precalc_data = big_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t2.get_intensity_ab(a,b, w1,w2, 3.8, condition=None,
                      debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    print('----------------------------------------------')

    a,b = 0,1 # (4,5), (2,3), (2,5)
    w1,w2 = t2.get_resonance_location_general((a,b))
    print(w1,w2)

    axes_dict = {1: w1, 2: w2}

    alldata = [Nnmodes, data, avrg_terms, axes_dict, t2.states_arrays_Eh, t2.harmonic_arrays_Eh] # todo: set this up better
    te.identify_to_precalculate()
    big_dict = te.precalculate(alldata)
    t2.precalc_data = big_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t2.get_intensity_ab(a,b, w1,w2, 3.8, condition=None,
                      debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    print('----------------------------------------------')

    a,b = 2,3 # (4,5), (2,3), (2,5)
    w1,w2 = t2.get_resonance_location_general((a,b))
    print(w1,w2)

    axes_dict = {1: w1, 2: w2}

    alldata = [Nnmodes, data, avrg_terms, axes_dict, t2.states_arrays_Eh, t2.harmonic_arrays_Eh] # todo: set this up better
    te.identify_to_precalculate()
    big_dict = te.precalculate(alldata) # make it external to each term ,
    t2.precalc_data = big_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t2.get_intensity_ab(a,b, w1,w2, 3.8, condition=None,
                      debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0


@require_asserts
def test_axes(spectrum_setup):
    """
    spectralAxis, spectralGrid: FIXME: should be in wilson_main
    """
    axis1 = abst.spectralAxis({1: 1}, range_style='custom')
    axis1.range = np.arange(spectrum_setup.start1, spectrum_setup.end1, spectrum_setup.step1)
    axis2 = abst.spectralAxis({2: 1}, range_style='custom')
    axis2.range = np.arange(spectrum_setup.start2, spectrum_setup.end2, spectrum_setup.step2)
    axes = abst.spectralGrid({1: axis1, 2: axis2}, range_style='custom')

    r_expr = (('b,a', (-1, 2)), ('zero,a', (-1,)))
    res_conds = sorted([i[1] for i in r_expr],key=len)

    print(res_conds)
    # print(sorted([(2,), (1, 2), (-1, 3)],key=len)) # todo: test case
    print('axes', axes)
    # print('axes.a', axes.a)
    # for aa in axes.a:
        # print('aa, axes.a[aa], axes.a[aa].fv', aa, axes.a[aa], axes.a[aa].fv)
        # print(axes.a[aa].range)


@require_asserts
def test_amplitude_4terms_grid(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0 = TermND(0, dict_8terms[0])
    t1 = TermND(1, dict_8terms[1])
    t2 = TermND(2, dict_8terms[2])
    t3 = TermND(3, dict_8terms[3])

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
    t1.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
    t2.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
    t3.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)

    ###########################################################################################################
    terms = [t0, t1, t2, t3]
    te = TermsEvaluator(terms)

    Nnmodes = 6
    print(t0.properties_data.keys())
    data = {
        (1, 1): t0.properties_data['mu_Q'],
        (1, 2): t0.properties_data['mu_QQ'],
        (2, 1): t0.properties_data['alpha_Q'],
        (2, 2): t0.properties_data['alpha_QQ'],
    }
    avrg_terms = spectrum_setup.gammaCompsAll

    w1 = np.arange(spectrum_setup.start1, spectrum_setup.end1, spectrum_setup.step1)
    w2 = np.arange(spectrum_setup.start2, spectrum_setup.end2, spectrum_setup.step2)
    w1m, w2m = np.meshgrid(w1, w2)

    # axis1 = abst.spectralAxis({1: 1}, range_style='custom')
    # axis1.range = np.arange(spectrum_setup.start1, spectrum_setup.end1, spectrum_setup.step1)
    # axis2 = abst.spectralAxis({2: 1}, range_style='custom')
    # axis2.range = np.arange(spectrum_setup.start2, spectrum_setup.end2, spectrum_setup.step2)
    # axes = abst.spectralGrid({1: axis1, 2: axis2}, range_style='custom')
    #
    # x,y = np.meshgrid(axes.a[1].range, axes.a[2].range)
    # axes_dict = {1: x, 2: y}
    axes_dict = {1: w1m, 2: w2m}

    alldata = [Nnmodes, data, avrg_terms, axes_dict, t2.states_arrays_Eh, t2.harmonic_arrays_Eh] # todo: set this up better
    te.identify_to_precalculate()
    big_dict = te.precalculate(alldata)
    for t in terms:
        t.precalc_data = big_dict
    ###########################################################################################################

    debug.level = 2
    # debug.debugfunc(t0.properties_data.keys(), 'self.properties_data.keys()')

    # print(w1m)
    amplitudes = 0.
    for t in terms:
        e = t.get_intensity(w1m, w2m, 3.8, 0., debugprint=False, collect_all=True)
        print('\n-----', t)
        print(e)
        print('any nan???? ', np.isnan(e).any())  # True if there's at least one NaN
        amplitudes += e
    debug.level = 0
    print('\n---- amplitudes')
    print(amplitudes.shape)
    print(np.max(np.abs(amplitudes)))
    print(f'{np.max(np.abs(amplitudes)**2):.2e}')
    print(spectrum_setup.start1, spectrum_setup.end1, spectrum_setup.step1)
    print(spectrum_setup.start2, spectrum_setup.end2, spectrum_setup.step2)
    print(w1m.shape)
    print(w1m)
    # print(amplitudes.shape)
    # print(amplitudes)


@require_asserts
def test_amplitude_1term_grid_t2(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    # t0 = TermND(0, terms_dict_setup[0])
    # t1 = TermND(1, terms_dict_setup[1])
    t2 = TermND(2, dict_8terms[2])
    # t3 = TermND(3, terms_dict_setup[3])

    # t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
    #                   mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
    # t1.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
    #                   mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
    t2.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
    # t3.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
    #                   mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)

    ###########################################################################################################
    # terms = [t0, t1, t2, t3]
    terms = [t2]
    te = TermsEvaluator(terms)

    Nnmodes = 6
    print(t2.properties_data.keys())
    data = {
        (1, 1): t2.properties_data['mu_Q'],
        (1, 2): t2.properties_data['mu_QQ'],
        (2, 1): t2.properties_data['alpha_Q'],
        (2, 2): t2.properties_data['alpha_QQ'],
    }
    avrg_terms = spectrum_setup.gammaCompsAll

    w1 = np.arange(spectrum_setup.start1, spectrum_setup.end1, spectrum_setup.step1)
    w2 = np.arange(spectrum_setup.start2, spectrum_setup.end2, spectrum_setup.step2)
    w1m, w2m = np.meshgrid(w1, w2)

    # axis1 = abst.spectralAxis({1: 1}, range_style='custom')
    # axis1.range = np.arange(spectrum_setup.start1, spectrum_setup.end1, spectrum_setup.step1)
    # axis2 = abst.spectralAxis({2: 1}, range_style='custom')
    # axis2.range = np.arange(spectrum_setup.start2, spectrum_setup.end2, spectrum_setup.step2)
    # axes = abst.spectralGrid({1: axis1, 2: axis2}, range_style='custom')
    # x,y = np.meshgrid(axes.a[1].range, axes.a[2].range)
    # axes_dict = {1: x, 2: y}

    a,b = 0,2 # (4,5), (2,3), (2,5)
    w1,w2 = t2.get_resonance_location_general((a,b))
    print(w1,w2)
    axes_dict = {1: w1, 2: w2}

    alldata = [Nnmodes, data, avrg_terms, axes_dict, t2.states_arrays_Eh, t2.harmonic_arrays_Eh] # todo: set this up better
    te.identify_to_precalculate()
    big_dict = te.precalculate(alldata)
    for t in terms:
        t.precalc_data = big_dict
    ###########################################################################################################

    debug.level = 2
    debug.debugfunc(t2.properties_data.keys(), 'self.properties_data.keys()')

    # print(w1m)
    amplitudes = 0.
    for t in terms:
        # e = t.get_intensity(w1m, w2m, 3.8, 0., debugprint=False, collect_all=True)
        e = t.get_intensity(w1, w2, 3.8, 0., debugprint=True, collect_all=True)
        print('\n-----', t)
        print(e)
        print('any nan???? ', np.isnan(e).any())  # True if there's at least one NaN
        amplitudes += e
    debug.level = 0
    print('\n---- amplitudes')
    print(amplitudes)
    print(np.max(amplitudes))


@require_asserts
def test_amplitude_4terms_single_point_ab_precalc(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()

    t0 = TermND(0, dict_8terms[0])
    t1 = TermND(1, dict_8terms[1])
    t2 = TermND(2, dict_8terms[2])
    t3 = TermND(3, dict_8terms[3])
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)

    a,b = 0,0 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))

    te = TermsEvaluator([t0, t1, t2, t3])
    freqs = np.array([t0.allstates[k] for k in t0.allstates if len(k)==1])
    Nnmodes = 6
    data = {
        (1, 1): t0.properties_data['mu_Q'],
        (1, 2): t0.properties_data['mu_QQ'],
        (2, 1): t0.properties_data['alpha_Q'],
        (2, 2): t0.properties_data['alpha_QQ'],
    }
    avrg_terms = spectrum_setup.gammaCompsAll
    axes_dict = {1: w1, 2: w2}

    print()
    alldata = [Nnmodes, data, avrg_terms, axes_dict, t0.states_arrays_Eh, t0.harmonic_arrays_Eh] # todo: set this up better
    te.identify_to_precalculate()
    big_dict = te.precalculate(alldata)
    print(t0.states_arrays[1][a],t0.states_arrays[1][b])
    # {'vibene_denoms': a,
    #  'avrg_tensors': b,
    #  'res_conds': c,
    #  'vibdiffs': d}
    t0.precalc_data = big_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_intensity_ab(a,b, w1,w2, 3.8, condition=None,
                      debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    print('----------------------------------------------')

    a,b = 0,1 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))
    axes_dict = {1: w1, 2: w2}

    alldata = [Nnmodes, data, avrg_terms, axes_dict, t0.states_arrays_Eh, t0.harmonic_arrays_Eh] # todo: set this up better
    te.identify_to_precalculate()
    big_dict = te.precalculate(alldata)
    t0.precalc_data = big_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_intensity_ab(a,b, w1,w2, 3.8, condition=None,
                      debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    print('----------------------------------------------')

    a,b = 2,3 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))
    axes_dict = {1: w1, 2: w2}

    alldata = [Nnmodes, data, avrg_terms, axes_dict, t0.states_arrays_Eh, t0.harmonic_arrays_Eh] # todo: set this up better
    te.identify_to_precalculate()
    big_dict = te.precalculate(alldata) # make it external to each term ,
    t0.precalc_data = big_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}, resonance in t0')
    debug.level = 1
    ampl = t0.get_intensity_ab(a,b, w1,w2, 3.8, condition=None,
                      debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    # print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    # debug.level = 1
    # ampl = 0.
    # for i, t in te.terms.items():
    #     ampl += t.get_intensity_ab(a,b, w1,w2, 3.8, condition=None,
    #                       debugprint=True)[0]
    # print(f'ampl = {ampl:.2e}')
    # debug.level = 0
