"""
Each test creates own instances of Term2D

Duplicated intro in test_term_evaluation : UPD - now it's in pytest fixtured in conftest.py
"""
import numpy as np

from wilson.spectrum.termND import TermND
from wilson.spectrum.termsEvaluator import TermsEvaluator
from wilson.utils import prep_data_load
# from wilson_main import abstractions as abst
from wilson.spectrum import DataForPrecalc

from tests.testing_utils import require_asserts

from wilson.spectrum import debug_mode
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
    MOL_setup_parser = MOL_setup_parser['FORM']
    spectrum_setup = spectrum_setup['FORM']

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
    assert list(t0.properties_data.keys()) == ['dipgrad', 'diphess', 'polgrad', 'polhess', 'cff']
    assert t0.mode_indices == [0, 1, 2, 3, 4, 5]

    parsed_data = MOL_setup_parser.parse(linear_molecule=False)
    # vpt2 freqs now
    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data)

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)

    assert t0.allstates[('1',)] == 1794.5406564861917 # still unchanged indices

    # UPDATING INDICES NOW!
    # parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data)

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
    # assert t0.allstates[('3',)] == 1794.5406564861917
    assert t0.allstates[('3',)] == 1185.288187960807


@require_asserts
def test_amplitude_1term_single_point(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()
    MOL_setup_parser = MOL_setup_parser['FORM']
    spectrum_setup = spectrum_setup['FORM']

    t0 = TermND(0, dict_8terms[0])
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    # parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)

    te = TermsEvaluator([t0])
    te.identify_to_precalculate()

    Nnmodes = 6
    # now here keys change; fixme: it the change needed??
    props_data_ready = {
        'dipgrad': t0.properties_data['dipgrad'],
        'diphess': t0.properties_data['diphess'],
        'polgrad': t0.properties_data['polgrad'],
        'polhess': t0.properties_data['polhess'],
    }
    from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices
    avrg_terms = get_AlphaBetaGammaDelta_indices(num_f=4)
    w1 = np.arange(spectrum_setup.start1,
                   spectrum_setup.end1, spectrum_setup.step1)
    w2 = np.arange(spectrum_setup.start2,
                   spectrum_setup.end2, spectrum_setup.step2)
    w1m, w2m = np.meshgrid(w1, w2, indexing='ij')
    axes_dict = {1: w1m, 2: w2m}
    from wilson.spectrum import DataForPrecalc
    alldata = DataForPrecalc(Nnmodes=Nnmodes,
                             props_data=props_data_ready,
                             avrg_terms=avrg_terms,
                             axes_dict=axes_dict,
                             states_arrays_Eh=t0.states_arrays_Eh,
                             harmonic_arrays_Eh=t0.harmonic_arrays_Eh)

    precalc_dict = te.precalculate(alldata)
    t0.precalc_data = precalc_dict

    a, b = 5, 0  # (4,5), (2,3), (2,5), (0,0)
    w1, w2 = t0.get_resonance_location_general((a, b))
    amplitude_single = t0.get_amplitudes(w1, w2, 3.8, 0.,
                                         collect_all=True, sel_abs=[(a,b)])

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    print(amplitude_single)
    ampl_single_ab = t0.get_amplitudes_ab((a, b), w1, w2, 3.8, condition=None)[0]
    print(ampl_single_ab)
    assert np.isclose(amplitude_single, ampl_single_ab)


@require_asserts
def test_amplitude_1term_single_point_ab(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()
    MOL_setup_parser = MOL_setup_parser['FORM']
    spectrum_setup = spectrum_setup['FORM']

    t0 = TermND(0, dict_8terms[0])
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    # parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)

    te = TermsEvaluator([t0])
    te.identify_to_precalculate()

    Nnmodes = 6
    # now here keys change; fixme: it the change needed??
    props_data_ready = {
        'dipgrad': t0.properties_data['dipgrad'],
        'diphess': t0.properties_data['diphess'],
        'polgrad': t0.properties_data['polgrad'],
        'polhess': t0.properties_data['polhess'],
    }
    from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices
    avrg_terms = get_AlphaBetaGammaDelta_indices(num_f=4)
    w1 = np.arange(spectrum_setup.start1,
                   spectrum_setup.end1, spectrum_setup.step1)
    w2 = np.arange(spectrum_setup.start2,
                   spectrum_setup.end2, spectrum_setup.step2)
    w1m, w2m = np.meshgrid(w1, w2, indexing='ij')
    axes_dict = {1: w1m, 2: w2m}
    from wilson.spectrum import DataForPrecalc
    alldata = DataForPrecalc(Nnmodes=Nnmodes,
                             props_data=props_data_ready,
                             avrg_terms=avrg_terms,
                             axes_dict=axes_dict,
                             states_arrays_Eh=t0.states_arrays_Eh,
                             harmonic_arrays_Eh=t0.harmonic_arrays_Eh)

    precalc_dict = te.precalculate(alldata)
    t0.precalc_data = precalc_dict

    print('t0.precalc_data', t0.precalc_data)
    a,b = 0,0 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))
    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_amplitudes_ab((a, b), w1, w2, 3.8, condition=None,
                                debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    print('----------------------------------------------')

    a,b = 0,1 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))
    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_amplitudes_ab((a, b), w1, w2, 3.8, condition=None,
                                debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    print('----------------------------------------------')

    a,b = 2,3 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))
    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_amplitudes_ab((a, b), w1, w2, 3.8, condition=None,
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
    MOL_setup_parser = MOL_setup_parser['FORM']
    spectrum_setup = spectrum_setup['FORM']

    t0 = TermND(0, dict_8terms[0])
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    # parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
    import numpy as np
    te = TermsEvaluator([t0])
    freqs = np.array([t0.allstates[k] for k in t0.allstates if len(k)==1])
    Nnmodes = 6
    print(t0.properties_data.keys())
    props_data_ready = {
        'dipgrad': t0.properties_data['dipgrad'],
        'diphess': t0.properties_data['diphess'],
        'polgrad': t0.properties_data['polgrad'],
        'polhess': t0.properties_data['polhess'],
    }
    avrg_terms = spectrum_setup.gammaCompsAll

    axis1 = np.arange(spectrum_setup.start1, spectrum_setup.end1, spectrum_setup.step1)
    axis2 = np.arange(spectrum_setup.start2, spectrum_setup.end2, spectrum_setup.step2)
    x,y = np.meshgrid(axis1, axis2)
    axes_dict = {1: x, 2: y}

    alldata = DataForPrecalc(Nnmodes=Nnmodes,
                             props_data=props_data_ready,
                             avrg_terms=avrg_terms,
                             axes_dict=axes_dict,
                             states_arrays_Eh=t0.states_arrays_Eh,
                             harmonic_arrays_Eh=t0.harmonic_arrays_Eh)
    te.identify_to_precalculate()
    precalc_dict = te.precalculate(alldata)

    t0.precalc_data = precalc_dict
    print(t0.get_resonance_location_general((4, 4)))


@require_asserts
def test_amplitude_1term_single_point_ab_precalc(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()
    MOL_setup_parser = MOL_setup_parser['FORM']
    spectrum_setup = spectrum_setup['FORM']

    t0 = TermND(0, dict_8terms[0])
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    # parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)

    a,b = 0,0 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))

    te = TermsEvaluator([t0])
    freqs = np.array([t0.allstates[k] for k in t0.allstates if len(k)==1])
    Nnmodes = 6
    props_data_ready = {
        'dipgrad': t0.properties_data['dipgrad'],
        'diphess': t0.properties_data['diphess'],
        'polgrad': t0.properties_data['polgrad'],
        'polhess': t0.properties_data['polhess'],
    }
    avrg_terms = spectrum_setup.gammaCompsAll
    axes_dict = {1: w1, 2: w2}

    print()
    alldata = DataForPrecalc(Nnmodes=Nnmodes,
                             props_data=props_data_ready,
                             avrg_terms=avrg_terms,
                             axes_dict=axes_dict,
                             states_arrays_Eh=t0.states_arrays_Eh,
                             harmonic_arrays_Eh=t0.harmonic_arrays_Eh)
    te.identify_to_precalculate()
    precalc_dict = te.precalculate(alldata)
    print(t0.states_arrays[1][a],t0.states_arrays[1][b])
    t0.precalc_data = precalc_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_amplitudes_ab((a, b), w1, w2, 3.8, condition=None,
                                debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    print('----------------------------------------------')

    a,b = 0,1 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))
    axes_dict = {1: w1, 2: w2}

    alldata = DataForPrecalc(Nnmodes=Nnmodes,
                             props_data=props_data_ready,
                             avrg_terms=avrg_terms,
                             axes_dict=axes_dict,
                             states_arrays_Eh=t0.states_arrays_Eh,
                             harmonic_arrays_Eh=t0.harmonic_arrays_Eh)
    te.identify_to_precalculate()
    precalc_dict = te.precalculate(alldata)
    t0.precalc_data = precalc_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_amplitudes_ab((a, b), w1, w2, 3.8, condition=None,
                                debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    print('----------------------------------------------')

    a,b = 2,3 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))
    axes_dict = {1: w1, 2: w2}

    alldata = DataForPrecalc(Nnmodes=Nnmodes,
                             props_data=props_data_ready,
                             avrg_terms=avrg_terms,
                             axes_dict=axes_dict,
                             states_arrays_Eh=t0.states_arrays_Eh,
                             harmonic_arrays_Eh=t0.harmonic_arrays_Eh)
    te.identify_to_precalculate()
    precalc_dict = te.precalculate(alldata) # make it external to each term ,
    t0.precalc_data = precalc_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_amplitudes_ab((a, b), w1, w2, 3.8, condition=None,
                                debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0



@require_asserts
def test_amplitude_4terms_grid(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()
    MOL_setup_parser = MOL_setup_parser['FORM']
    spectrum_setup = spectrum_setup['FORM']

    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    # parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
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
    props_data_ready = {
        'dipgrad': t0.properties_data['dipgrad'],
        'diphess': t0.properties_data['diphess'],
        'polgrad': t0.properties_data['polgrad'],
        'polhess': t0.properties_data['polhess'],
    }
    avrg_terms = spectrum_setup.gammaCompsAll

    w1 = np.arange(spectrum_setup.start1, spectrum_setup.end1, spectrum_setup.step1)
    w2 = np.arange(spectrum_setup.start2, spectrum_setup.end2, spectrum_setup.step2)
    w1m, w2m = np.meshgrid(w1, w2)

    axes_dict = {1: w1m, 2: w2m}

    alldata = DataForPrecalc(Nnmodes=Nnmodes,
                             props_data=props_data_ready,
                             avrg_terms=avrg_terms,
                             axes_dict=axes_dict,
                             states_arrays_Eh=t2.states_arrays_Eh,
                             harmonic_arrays_Eh=t2.harmonic_arrays_Eh)
    te.identify_to_precalculate()
    precalc_dict = te.precalculate(alldata)
    for t in terms:
        t.precalc_data = precalc_dict
    ###########################################################################################################

    # debug.level = 2
    # debug.debugfunc(t0.properties_data.keys(), 'self.properties_data.keys()')

    # print(w1m)
    amplitudes = 0.
    for t in terms:
        e = t.get_amplitudes(w1m, w2m, 3.8, 0., debugprint=False, collect_all=True)
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



@require_asserts
def test_amplitude_4terms_single_point_ab_precalc(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()
    MOL_setup_parser = MOL_setup_parser['FORM']
    spectrum_setup = spectrum_setup['FORM']

    t0 = TermND(0, dict_8terms[0])
    t1 = TermND(1, dict_8terms[1])
    t2 = TermND(2, dict_8terms[2])
    t3 = TermND(3, dict_8terms[3])
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    # parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)

    a,b = 0,0 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))

    te = TermsEvaluator([t0, t1, t2, t3])
    freqs = np.array([t0.allstates[k] for k in t0.allstates if len(k)==1])
    Nnmodes = 6
    props_data_ready = {
        'dipgrad': t0.properties_data['dipgrad'],
        'diphess': t0.properties_data['diphess'],
        'polgrad': t0.properties_data['polgrad'],
        'polhess': t0.properties_data['polhess'],
    }
    avrg_terms = spectrum_setup.gammaCompsAll
    axes_dict = {1: w1, 2: w2}

    print()
    alldata = DataForPrecalc(Nnmodes=Nnmodes,
                             props_data=props_data_ready,
                             avrg_terms=avrg_terms,
                             axes_dict=axes_dict,
                             states_arrays_Eh=t0.states_arrays_Eh,
                             harmonic_arrays_Eh=t0.harmonic_arrays_Eh)
    te.identify_to_precalculate()
    precalc_dict = te.precalculate(alldata)
    print(t0.states_arrays[1][a],t0.states_arrays[1][b])
    t0.precalc_data = precalc_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_amplitudes_ab((a, b), w1, w2, 3.8, condition=None,
                                debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    print('----------------------------------------------')

    a,b = 0,1 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))
    axes_dict = {1: w1, 2: w2}

    alldata = DataForPrecalc(Nnmodes=Nnmodes,
                             props_data=props_data_ready,
                             avrg_terms=avrg_terms,
                             axes_dict=axes_dict,
                             states_arrays_Eh=t0.states_arrays_Eh,
                             harmonic_arrays_Eh=t0.harmonic_arrays_Eh)
    te.identify_to_precalculate()
    precalc_dict = te.precalculate(alldata)
    t0.precalc_data = precalc_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}')
    debug.level = 1
    ampl = t0.get_amplitudes_ab((a, b), w1, w2, 3.8, condition=None,
                                debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0
    print('----------------------------------------------')

    a,b = 2,3 # (4,5), (2,3), (2,5)
    w1,w2 = t0.get_resonance_location_general((a,b))
    axes_dict = {1: w1, 2: w2}

    alldata = DataForPrecalc(Nnmodes=Nnmodes,
                             props_data=props_data_ready,
                             avrg_terms=avrg_terms,
                             axes_dict=axes_dict,
                             states_arrays_Eh=t0.states_arrays_Eh,
                             harmonic_arrays_Eh=t0.harmonic_arrays_Eh)
    te.identify_to_precalculate()
    precalc_dict = te.precalculate(alldata) # make it external to each term ,
    t0.precalc_data = precalc_dict

    print(f'\n(a,b) - {a,b}; w1,w2 - {w1:.2f}, {w2:.2f}, resonance in t0')
    debug.level = 1
    ampl = t0.get_amplitudes_ab((a, b), w1, w2, 3.8, condition=None,
                                debugprint=True)[0]
    print(f'ampl = {ampl:.2e}')
    debug.level = 0


def test_termevaluator():

    import json

    with open('/home/vlev/wilson-suite/tests/terms.json') as json_file:
        data = json.load(json_file)
        print("Type:", type(data))
        print(data)


def test_compute_vibdiff():
    print()
    from wilson.spectrum import compute_vibdiff
    print(compute_vibdiff((0,1), (3,)))
    print(compute_vibdiff((0,1), (6,)))
    print(compute_vibdiff((1,1), (3,2)))
    print(compute_vibdiff((1,1), (3,0)))
    print(compute_vibdiff((2,1), (3,0,1)))
    print(compute_vibdiff((2,1), (3,1,0)))



def test_dotspectrum_df(terms_collection, spectrum_setup, conditions):
    """
    get a spectrum figure
    """
    # terms_collection is a fixture
    te, precalc_dict = terms_collection['FORM']
    # spectrum_setup = spectrum_setup['FORM']
    # conditions = conditions['FORM']

    # expected_keys = ['vibene_denoms', 'avrg_tensors', 'res_conds', 'vibdiffs']
    # for key in expected_keys:
    #     assert key in precalc_dict, f"Key '{key}' missing in precalculated data"
    # assert precalc_dict['vibene_denoms'], "vibene_denoms data is empty"

    # from rich import print as rprint
    # print('\n')
    # rprint("[deep_pink3]Precalculated data[/deep_pink3]")
    # rprint(precalc_dict)
    # print('\n')

    with debug_mode(0):
        for id, term in te.terms.items():
            term.precalc_data = precalc_dict
            with np.printoptions(precision=2,legacy='1.25'):
                # formatted_resonances = {
                #     key: (f"{value[0]:.2f}", f"{value[1]:.2f}")
                #     for key, value in term.get_all_resonances(w2mw1=True).items()
                # }
                df, distances = term.get_dotspectrum_df(Gamma_rc=3.8, margin=1.)
                print(df)
                print('_____________________')
                print(term)
                # print(formatted_resonances)


from wilson.spectrum.termND import sum_over_suffixes


def test_get_factor_summed(terms_collection):
    print()

    te, precalc_dict = terms_collection['FORM']
    term3 = te.terms[3]
    term3.precalc_data = precalc_dict
    # print(' jimr', term3.precalc_data)

    ab_comb = (0,0)
    total = 0.
    remaining_length = term3.collective_n_idx_max - term3.collective_n_idx_rescond
    total2 = sum_over_suffixes(ab_comb,
                               remaining_length,
                               term3.mode_indices,
                               term3.get_full_factor)
    for c in term3.mode_indices:
        addition_2 = term3.get_full_factor((*ab_comb, c), False, debugprint=False)
        total += addition_2

    assert total2==total, "sum_over_suffixes didn't work out"

    ab_comb = (0,1)
    total = 0.
    remaining_length = term3.collective_n_idx_max - term3.collective_n_idx_rescond
    total2 = sum_over_suffixes(ab_comb,
                               remaining_length,
                               term3.mode_indices,
                               term3.get_full_factor)
    for c in term3.mode_indices:
        addition_2 = term3.get_full_factor((*ab_comb, c), False, debugprint=False)
        total += addition_2

    assert total2==total, "sum_over_suffixes didn't work out"

    ab_comb = (1,0)
    total = 0.
    remaining_length = term3.collective_n_idx_max - term3.collective_n_idx_rescond
    total2 = sum_over_suffixes(ab_comb,
                               remaining_length,
                               term3.mode_indices,
                               term3.get_full_factor)
    for c in term3.mode_indices:
        addition_2 = term3.get_full_factor((*ab_comb, c), False, debugprint=False)
        total += addition_2

    assert total2==total, "sum_over_suffixes didn't work out"

    ab_comb = (2,1)
    total = 0.
    remaining_length = term3.collective_n_idx_max - term3.collective_n_idx_rescond
    total2 = sum_over_suffixes(ab_comb,
                               remaining_length,
                               term3.mode_indices,
                               term3.get_full_factor)
    for c in term3.mode_indices:
        addition_2 = term3.get_full_factor((*ab_comb, c), False, debugprint=False)
        total += addition_2

    assert total2==total, "sum_over_suffixes didn't work out"

