import wilson_suite.wilson_derive.abstractions as wa

def generate_only_res_cond_evv_term_selection():

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


def test_ResonanceCondition():

    ab_state = wa.HarmOscStateSymbolic(['a', 'b'])
    a_state = wa.HarmOscStateSymbolic(['a'])
    b_state = wa.HarmOscStateSymbolic(['b'])
    zero_state = wa.HarmOscStateSymbolic([''])

    vd_ab_a = wa.VibDiffTerm(sl=ab_state, sr=a_state)
    vd_0_a = wa.VibDiffTerm(sl=zero_state, sr=a_state)
    vd_b_a = wa.VibDiffTerm(sl=b_state, sr=a_state)

    rc_ab_a_w_A = wa.ResonanceCondition(diff=vd_ab_a, pf=['A'])
    rc_b_a_w_B = wa.ResonanceCondition(diff=vd_b_a, pf=['B'])
    rc_0_a_w_B = wa.ResonanceCondition(diff=vd_0_a, pf=['B'])
    rc_0_a_w_AmB = wa.ResonanceCondition(diff=vd_0_a, pf=['A', '-B'])

    pass

def test_VibPerturbedTerm_repr():
    
    print()
    terms = generate_only_res_cond_evv_term_selection()
    # print(terms[0])

    print()
    # print(terms[1])
    
    terms[0].present_better()
