import wilson_suite.wilson_derive.abstractions as wa
import wilson_suite.wilson_derive.response_terms


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

    term_a = wilson_suite.wilson_derive.response_terms.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                                                        res = res_conds_a)

    term_b = wilson_suite.wilson_derive.response_terms.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                                                        res = res_conds_b)

    term_c = wilson_suite.wilson_derive.response_terms.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                                                        res = res_conds_c)

    term_d = wilson_suite.wilson_derive.response_terms.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                                                        res = res_conds_d)

    term_e = wilson_suite.wilson_derive.response_terms.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [],
                                                                        res = res_conds_e)

    return [term_a, term_b, term_c, term_d, term_e]

def generate_res_cond_objs():
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
    
    return [rc_ab_a_w_A, rc_b_a_w_B, rc_0_a_w_B, rc_0_a_w_AmB]


def test_VibPerturbedTerm_repr():
    
    print()
    terms = generate_only_res_cond_evv_term_selection()

    for i, t in enumerate(terms):
        print('\n ---', i)
        for res in t.res:
            print(res)
        
        for frt in t.freqterms:
            print(frt)

def test_VibPerturbedTerm_full_repr():
    
    print()
    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_list = get_terms_from_json()

    for i, t in enumerate(terms_fuller_list):
        print('\n ---', i)
        for res in t.res:
            print('in res cond', res.diff)
        
        for frt in t.freqterms:
            print(frt)

def test_RC_latex():
    res_conds = generate_res_cond_objs()
    latex_resconds = [rc.to_latex() for rc in res_conds]
    assert latex_resconds == ['(\\omega_{a+b,a} -A)', 
                              '(\\omega_{b,a} -B)', 
                              '(\\omega_{,a} -B)', 
                              '(\\omega_{,a} -A+B)']

def test_VibPerturbedTerm_latex():
    terms = generate_only_res_cond_evv_term_selection()
    latex_strs = [t.to_latex() for t in terms]

    assert latex_strs == ['\\frac{1}{4}\\frac{1}{(\\omega_{a+b,a} -A)(\\omega_{b,a} -B)}', 
                          '\\frac{1}{4}\\frac{1}{(\\omega_{a+b,a} -A)}', 
                          '\\frac{1}{4}\\frac{1}{(\\omega_{a+b,a} -A)(\\omega_{b,a} -B)}', 
                          '\\frac{1}{4}\\frac{1}{(\\omega_{,a} -B)(\\omega_{b,a} -B)}', 
                          '\\frac{1}{4}\\frac{1}{(\\omega_{,a} -B)(\\omega_{,a} -A+B)}']
    
    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_list = get_terms_from_json()

    latex_strs = [t.to_latex() for t in terms_fuller_list]
    assert latex_strs[0] == '\\frac{-1}{4}\\frac{1}{\\omega_{a,}\\omega_{b,}}\\frac{\\partial\\mu_{\\beta}}{\\partial Q_{a}}\\frac{\\partial\\mu_{\\gamma}}{\\partial Q_{b}}\\frac{\\partial^{2}\\alpha_{\\alpha\\delta}}{\\partial Q_{a}\\partial Q_{b}}\\frac{1}{(\\omega_{,a} +A-B)(\\omega_{b,a} -B)}'
    assert latex_strs[-1] == '\\frac{1}{8}\\frac{1}{\\omega_{a,}\\omega_{a+b+c,}\\omega_{b,}\\omega_{c,}}\\frac{\\partial\\alpha_{\\alpha\\delta}}{\\partial Q_{b}}\\frac{\\partial\\mu_{\\beta}}{\\partial Q_{a}}\\frac{\\partial\\mu_{\\gamma}}{\\partial Q_{c}}\\frac{\\partial^{3}E_{}}{\\partial Q_{a}\\partial Q_{b}\\partial Q_{c}}\\frac{1}{(\\omega_{,a} +A-B)(\\omega_{a+b,a} -B)}'
