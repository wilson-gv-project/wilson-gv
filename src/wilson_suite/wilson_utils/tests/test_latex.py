from ..latex_rendering import get_plt_latex

def test_plt_latex():
    expr1 = r'\frac{1}{4}\frac{1}{(\omega_{a+b,a} -A)(\omega_{b,a} -B)}'
    expr2 = r'\frac{1}{4}\frac{1}{(\omega_{a+b,a} -A)}'
    expr3 = r'\frac{1}{4}\frac{1}{(\omega_{a+b,a} -A)(\omega_{b,a} -B)}'
    expr4 = r'\frac{1}{4}\frac{1}{(\omega_{,a} -B)(\omega_{b,a} -B)}'
    expr5 = r'\frac{1}{4}\frac{1}{(\omega_{,a} -B)(\omega_{,a} -A+B)}'
    
    expr6 = r'\frac{-1}{4}\frac{1}{\omega_{a,}\omega_{b,}}\frac{\partial\mu_{\beta}}{\partial Q_{a}}\frac{\partial\mu_{\gamma}}{\partial Q_{b}}\frac{\partial^{2}\alpha_{\alpha\delta}}{\partial Q_{a}\partial Q_{b}}\frac{1}{(\omega_{,a} +A-B)(\omega_{b,a} -B)}'
    get_plt_latex(expr1)
    get_plt_latex(expr2)
    get_plt_latex(expr3)
    get_plt_latex(expr4)
    get_plt_latex(expr5)
    get_plt_latex(expr6)

def test_saved():
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_list = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)

    latex_strs = [t.to_latex() for t in terms_fuller_list]
    for n, t_latex in enumerate(latex_strs):
        get_plt_latex(t_latex, savename=f'term{n}.svg')

def test_terms_custom_saved():
    from ...wilson_intensities.tests.unit.test_resonances import generate_only_res_cond_evv_term_selection
    terms = generate_only_res_cond_evv_term_selection()
    latex_strs = [t.to_latex() for t in terms]

    for n, t_latex in enumerate(latex_strs):
        get_plt_latex(t_latex, savename=f'term_custom{n}.svg')