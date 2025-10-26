import numpy as np
from wilson_suite.wilson_intensities.amplitudes import func_abstractions as f_abst
import wilson_suite.wilson_intensities.amplitudes.resonances


def test_solve_LSE_motif():
    motif1 = (((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))
    motif2 = (((('a', 'b'), ('a',)), ('A',)),)
    motif3 = (((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))
    motif4 = (((('',), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))

    from ...amplitudes.resonances import solve_LSE_motif
    params = f_abst.ParameterSet({'a': '1', 'b': '3', 'zero': 'zero'})
    vibdata = f_abst.VibStatesData(allstates=(f_abst.VibState(s={}, state_label='1', e=1234.),
                                              f_abst.VibState(s={}, state_label='3', e=3644.),
                                              f_abst.VibState(s={}, state_label='1+3', e=4164.),
                                              f_abst.VibState(s={}, state_label='zero', e=0.)),
                                   harmonic_osc_states_labels=('1', '3'))
    print()
    r1 = solve_LSE_motif(motif=motif1, parameters=params, vibdata=vibdata, unit='cm-1')
    print(r1, motif1, params)

    r2 = solve_LSE_motif(motif=motif2, parameters=params, vibdata=vibdata, unit='cm-1')
    print(r2, motif2, params)

    r3 = solve_LSE_motif(motif=motif3, parameters=params, vibdata=vibdata, unit='cm-1')
    print(r3, motif3, params)

    r4 = solve_LSE_motif(motif=motif4, parameters=params, vibdata=vibdata, unit='cm-1')
    print(r4, motif4, params)


def test_generate_RHS_motif():
    motif1 = (((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))
    motif2 = (((('a', 'b'), ('a',)), ('A',)),)
    motif3 = (((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))

    # (res_cond1, res_cond2, res_cond3, ...)
    # (res_cond1, (wibdiff_mn, axes), ...)
    # (res_cond1, ((m_inds, n_inds), (ax1, ax2, ax3, ...)), ...)
    # (res_cond1, (((m1, m2, ...), (n1, n2, ...)), (ax1, ax2, ax3, ...)), ...)

    # vibdiff: (m_inds, n_inds) <=== ((m1, m2, ...), (n1, n2, ...))
    motif4 = (((('',), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))

    from ...amplitudes.resonances import get_RHS_motif
    params = f_abst.ParameterSet({'a': '1', 'b': '3', 'zero': 'zero'})
    vibdata = f_abst.VibStatesData(allstates=(f_abst.VibState(s={}, state_label='1', e=1234.),
                                              f_abst.VibState(s={}, state_label='3', e=3644.),
                                              f_abst.VibState(s={}, state_label='1+3', e=4164.),
                                              f_abst.VibState(s={}, state_label='zero', e=0.)),
                                   harmonic_osc_states_labels=('1', '3'))

    r1 = get_RHS_motif(motif=motif1, parameters=params, vibdata=vibdata, unit='cm-1')
    print(r1, motif1, params)
    assert np.all(r1==np.array([-2930.0, -2410.0]))

    r2 = get_RHS_motif(motif=motif2, parameters=params, vibdata=vibdata, unit='cm-1')
    assert np.all(r2==np.array([-2930.0]))

    r3 = get_RHS_motif(motif=motif3, parameters=params, vibdata=vibdata, unit='cm-1')
    assert np.all(r3==np.array([1234.0, 1234.0]))

    r4 = get_RHS_motif(motif=motif4, parameters=params, vibdata=vibdata, unit='cm-1')
    assert np.all(r4==np.array([1234.0, -2410.0]))


def test_generate_LHS_motif():
    print()
    motif1 = (((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))
    motif2 = (((('a', 'b'), ('a',)), ('A',)),)
    motif3 = (((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))
    motif4 = (((('',), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))

    from ...amplitudes.resonances import generate_LHS_motif

    r1 = generate_LHS_motif(motif=motif1)
    assert np.allclose(r1, np.array([[-1.,  0.], [ 0., -1.]]))

    r2 = generate_LHS_motif(motif=motif2)
    assert np.allclose(r2, np.array([[-1.]])) # ???

    r3 = generate_LHS_motif(motif=motif3)
    assert np.allclose(r3, np.array([[ 0., -1.], [ -1.,  1.]]))

    r4 = generate_LHS_motif(motif=motif4)
    print(r4)


def test_is_location_in_window():
    loc1 = {'A': 12., 'B': 33.}
    window1 = {'A': (9., 14.), 'B': (22., 54.)}
    window2 = {'A': (12., 14.), 'B': (30., 54.)}
    window3 = {'A': (9., 14.), 'B': (41., 44.)}
    window4 = {'A': (11., 21.), 'B': (41., 44.)}

    print()
    r1 = wilson_suite.wilson_intensities.amplitudes.resonances.is_location_in_window(location=loc1, window=window1)
    assert r1

    r2 = wilson_suite.wilson_intensities.amplitudes.resonances.is_location_in_window(location=loc1, window=window2, margins={'A': (2., 2.)})
    assert r2

    r3 = wilson_suite.wilson_intensities.amplitudes.resonances.is_location_in_window(location=loc1, window=window3)
    assert not r3

    r4 = wilson_suite.wilson_intensities.amplitudes.resonances.is_location_in_window(location=loc1, window=window4, margins={'B':(10., 2.)})
    assert r4


def generate_only_res_cond_evv_term_selection():

    import wilson_suite.wilson_derive.abstractions as wa
    from fractions import Fraction

    ab_state = wa.HarmOscStateSymbolic(['a', 'b'])
    a_state = wa.HarmOscStateSymbolic(['a'])
    b_state = wa.HarmOscStateSymbolic(['b'])
    zero_state = wa.HarmOscStateSymbolic([''])

    vd_ab_a = wa.VibDiffTerm(sl = ab_state, sr = a_state)
    vd_0_a = wa.VibDiffTerm(sl=zero_state, sr=a_state)
    vd_b_a = wa.VibDiffTerm(sl=b_state, sr=a_state)

    rc_ab_a_w_A = wa.ResonanceCondition(diff = vd_ab_a, pf = ['A'])
    rc_b_a_w_B = wa.ResonanceCondition(diff=vd_b_a, pf=['B'])
    rc_0_a_w_B = wa.ResonanceCondition(diff=vd_0_a, pf=['B'])
    rc_0_a_w_AmB = wa.ResonanceCondition(diff=vd_0_a, pf=['A', '-B'])

    res_conds_a = [rc_ab_a_w_A, rc_b_a_w_B]
    res_conds_b = [rc_ab_a_w_A]
    res_conds_c = [rc_ab_a_w_A, rc_b_a_w_B]
    res_conds_d = [rc_0_a_w_B, rc_b_a_w_B]
    res_conds_e = [rc_0_a_w_B, rc_0_a_w_AmB]

    term_a = wa.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                 res = res_conds_a)

    term_b = wa.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                 res = res_conds_b)

    term_c = wa.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                 res = res_conds_c)

    term_d = wa.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                 res = res_conds_d)

    term_e = wa.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                 res = res_conds_e)

    return [term_a, term_b, term_c, term_d, term_e]


def test_identify_maximum_axes_in_terms():
    print()
    candidate_terms = generate_only_res_cond_evv_term_selection()

    wilson_suite.wilson_intensities.amplitudes.resonances.identify_maximum_axes_in_terms(candidate_terms)


def test_motifs_control():
    print()
    candidate_terms = generate_only_res_cond_evv_term_selection()

    r = wilson_suite.wilson_intensities.amplitudes.resonances.motifs_control(candidate_terms)
    print(r)


def test_find_resonance_locations_wrt_index_choices():
    print()

    motif1 = (((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))
    motif2 = (((('a', 'b'), ('a',)), ('A',)),)
    motif3 = (((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))
    motif4 = (((('',), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))

    from wilson_suite.wilson_intensities.amplitudes import func_abstractions as f_abst
    allstates = (f_abst.VibState(s={}, state_label='1', e=1234.),
                 f_abst.VibState(s={}, state_label='3', e=3644.),
                 f_abst.VibState(s={}, state_label='4', e=1621.),
                 f_abst.VibState(s={}, state_label='1+1', e=2514.),
                 f_abst.VibState(s={}, state_label='1+4', e=1904.),
                 f_abst.VibState(s={}, state_label='3+4', e=4129.),
                 f_abst.VibState(s={}, state_label='4+4', e=3022.),
                 f_abst.VibState(s={}, state_label='3+3', e=7344.),
                 f_abst.VibState(s={}, state_label='1+3', e=4364.))
    harm_labels = ('1', '3', '4')
    vibdata = f_abst.VibStatesData(allstates=allstates, harmonic_osc_states_labels=harm_labels)

    d = wilson_suite.wilson_intensities.amplitudes.resonances.find_resonance_locations_wrt_index_choices(motif=motif1, vibstates_data=vibdata)
    print(d)


def test_terms_for_motif():
    """
    HMM...
unique_motifs:
(((('a', 'b'), ('a',)), ('A',)),)  ------> ????
(((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))
(((('',), ('a',)), ('B',)), ((('b',), ('a',)), ('B',))) ------> ????
(((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))

    """
    candidate_terms = generate_only_res_cond_evv_term_selection()

    unique_motifs = wilson_suite.wilson_intensities.amplitudes.resonances.identify_unique_resmotifs(candidate_terms)

    terms_for_motif = wilson_suite.wilson_intensities.amplitudes.resonances.terms_for_motif(candidate_terms)

    assert  len(terms_for_motif[(((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))]) == 2
    assert  len(terms_for_motif[(((('a', 'b'), ('a',)), ('A',)),)]) == 1
    assert not(len(terms_for_motif[(((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))]) == 4)

    assert sorted(unique_motifs) == sorted(list(terms_for_motif.keys()))