import wilson_suite.wilson_intensities.spectrum.pre_eval_treatment as pet


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

    unique_motifs = pet.identify_unique_resmotifs(candidate_terms)

    terms_for_motif = pet.terms_for_motif(candidate_terms)

    assert  len(terms_for_motif[(((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))]) == 2
    assert  len(terms_for_motif[(((('a', 'b'), ('a',)), ('A',)),)]) == 1
    assert not(len(terms_for_motif[(((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))]) == 4)

    assert sorted(unique_motifs) == sorted(list(terms_for_motif.keys()))


def test_find_resonance_locations_wrt_index_choices():
    print()
    candidate_terms = generate_only_res_cond_evv_term_selection()

    unique_motifs = pet.identify_unique_resmotifs(candidate_terms)
    for i in unique_motifs:
        print(i)
    motif1 = (((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))
    motif2 = (((('a', 'b'), ('a',)), ('A',)),)
    motif3 = (((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))
    motif4 = (((('',), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))

    from wilson_suite.wilson_intensities.spectrum import func_abstractions as f_abst
    vibdata = f_abst.VibStatesData(allstates=(f_abst.VibState(s={}, state_label='1', e=1234.),
                                              f_abst.VibState(s={}, state_label='1+1', e=2514.),
                                              f_abst.VibState(s={}, state_label='3', e=3644.),
                                              f_abst.VibState(s={}, state_label='3+3', e=7344.),
                                              f_abst.VibState(s={}, state_label='1+3', e=4364.),
                                              f_abst.VibState(s={}, state_label='zero', e=0.)))
    
    d = pet.find_resonance_locations_wrt_index_choices(motif=motif1, vibstates_data=vibdata)
    print(d)

def test_motifs_control():
    print()
    candidate_terms = generate_only_res_cond_evv_term_selection()
    
    r = pet.motifs_control(candidate_terms)
    print(r)

def test_identify_maximum_axes():
    print()
    candidate_terms = generate_only_res_cond_evv_term_selection()

    pet.identify_maximum_axes(candidate_terms)

def test_single_motif_control():
    print()
    motif1 = (((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))
    motif2 = (((('a', 'b'), ('a',)), ('A',)),)
    motif3 = (((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))
    motif4 = (((('',), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))

    pet.single_motif_control(motif1, 2)
    pet.single_motif_control(motif2, 2)
    pet.single_motif_control(motif3, 2)
    pet.single_motif_control(motif4, 2)