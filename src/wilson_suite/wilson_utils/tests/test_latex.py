from ..latex_rendering import get_plt_latex

def test_plt_latex():
    expr1 = r'\frac{1}{4}\frac{1}{(\omega_{a+b,a} -A)(\omega_{b,a} -B)}'
    expr2 = r'\frac{1}{4}\frac{1}{(\omega_{a+b,a} -A)}'
    expr3 = r'\frac{1}{4}\frac{1}{(\omega_{a+b,a} -A)(\omega_{b,a} -B)}'
    expr4 = r'\frac{1}{4}\frac{1}{(\omega_{,a} -B)(\omega_{b,a} -B)}'
    expr5 = r'\frac{1}{4}\frac{1}{(\omega_{,a} -B)(\omega_{,a} -A+B)}'
    
    expr6 = r'\frac{-1}{4}\frac{1}{\omega_{a,}\omega_{b,}}\frac{\partial\mu_{\beta}}{\partial Q_{a}}\frac{\partial\mu_{\gamma}}{\partial Q_{b}}\frac{\partial^{2}\alpha_{\alpha\delta}}{\partial Q_{a}\partial Q_{b}}\frac{1}{(\omega_{,a} +A-B)(\omega_{b,a} -B)}'
    # get_plt_latex(expr1)
    # get_plt_latex(expr2)
    # get_plt_latex(expr3)
    # get_plt_latex(expr4)
    # get_plt_latex(expr5)
    # get_plt_latex(expr6)

def test_saved():
    from wilson_suite.fixtures import get_terms_from_json
    terms_fuller_flat = get_terms_from_json()

    latex_strs = [t.to_latex() for t in terms_fuller_flat]
    for n, t_latex in enumerate(latex_strs):
        get_plt_latex(t_latex, savename=f'term{n}.svg')

def generate_only_res_cond_evv_term_selection():
    """Five EVV-like terms with resonance conditions only (moved here from the deleted old test_resonances.py)"""
    import wilson_suite.wilson_derive.abstractions as wa
    import wilson_suite.wilson_derive.response_terms as wr

    from fractions import Fraction

    ab_state = wa.HarmOscStateSymbolic(['a', 'b'])
    a_state = wa.HarmOscStateSymbolic(['a'])
    b_state = wa.HarmOscStateSymbolic(['b'])
    zero_state = wa.HarmOscStateSymbolic([])

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

    return [wr.VibPerturbedTerm(coeff = Fraction(1, 4), props = [], freqterms = [], res = res_conds)
            for res_conds in (res_conds_a, res_conds_b, res_conds_c, res_conds_d, res_conds_e)]

def test_terms_custom_saved():
    terms = generate_only_res_cond_evv_term_selection()
    latex_strs = [t.to_latex() for t in terms]

    for n, t_latex in enumerate(latex_strs):
        get_plt_latex(t_latex, savename=f'term_custom{n}.svg')