"""
latex_rendering.py: get_plt_latex draws a LaTeX formula with matplotlib's mathtext and saves it, e.g. as SVG.
A formula that mathtext cannot read raises ValueError when the figure is saved.
The formulas: VibPerturbedTerm.to_latex of five hand-built terms (resonance conditions only) and of the saved terms
(terms_fuller_flat.json).
"""
import matplotlib.pyplot as plt
import pytest

from ..latex_rendering import get_plt_latex


def generate_only_res_cond_evv_term_selection():
    """Five EVV-like terms with resonance conditions only (moved here from the deleted old test_resonances.py)"""
    from fractions import Fraction

    import wilson_suite.wilson_derive.abstractions as wa
    import wilson_suite.wilson_derive.response_terms as wr

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


# to_latex of the five terms above, in order (a and c have the same resonance conditions)
RES_COND_TERMS_LATEX = [
    r'\frac{1}{4}\frac{1}{(\omega_{a+b,a} -A)(\omega_{b,a} -B)}',
    r'\frac{1}{4}\frac{1}{(\omega_{a+b,a} -A)}',
    r'\frac{1}{4}\frac{1}{(\omega_{a+b,a} -A)(\omega_{b,a} -B)}',
    r'\frac{1}{4}\frac{1}{(\omega_{,a} -B)(\omega_{b,a} -B)}',
    r'\frac{1}{4}\frac{1}{(\omega_{,a} -B)(\omega_{,a} -A+B)}',
]
# to_latex of the first saved term: coefficient, frequencies, properties, resonance conditions
SAVED_TERM_0_LATEX = (r'\frac{-1}{4}\frac{1}{\omega_{a,}\omega_{b,}}\frac{\partial\mu_{\beta}}{\partial Q_{a}}'
                      r'\frac{\partial\mu_{\gamma}}{\partial Q_{b}}'
                      r'\frac{\partial^{2}\alpha_{\alpha\delta}}{\partial Q_{a}\partial Q_{b}}'
                      r'\frac{1}{(\omega_{,a} +A-B)(\omega_{b,a} -B)}')


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


## get_plt_latex ---------------------------------------------------------------

@pytest.mark.parametrize('expr', [*RES_COND_TERMS_LATEX, SAVED_TERM_0_LATEX])
def test_get_plt_latex_saves_the_formula_as_svg(expr, tmp_path):
    savename = tmp_path / 'formula.svg'

    get_plt_latex(expr, savename=str(savename))

    assert savename.read_text().startswith('<?xml')
    assert '<svg' in savename.read_text()
    assert [t.get_text() for t in plt.gcf().axes[0].texts] == [f'${expr}$']


def test_get_plt_latex_draws_in_the_given_color(tmp_path):
    get_plt_latex(r'\frac{1}{4}', savename=str(tmp_path / 'formula.svg'), color='red')

    assert plt.gcf().axes[0].texts[0].get_color() == 'red'


@pytest.mark.parametrize('expr', [r'\frac{1}{', r'\fracc{1}{4}', r'\omega_{a'],
                         ids=['frac without denominator', 'unknown command', 'brace not closed'])
def test_get_plt_latex_raises_for_a_formula_mathtext_cannot_read(expr, tmp_path):
    with pytest.raises(ValueError):
        get_plt_latex(expr, savename=str(tmp_path / 'formula.svg'))


## to_latex of the terms ------------------------------------------------------

def test_res_cond_terms_to_latex():
    assert [t.to_latex() for t in generate_only_res_cond_evv_term_selection()] == RES_COND_TERMS_LATEX


def test_saved_terms_to_latex_render_one_svg_per_term(tmp_path):
    """14 terms: each LaTeX string is one mathtext can read; each gives one file"""
    from wilson_suite.fixtures import get_terms_from_json
    terms = get_terms_from_json()

    for n, term in enumerate(terms):
        get_plt_latex(term.to_latex(), savename=str(tmp_path / f'term{n}.svg'))
        plt.close('all')

    assert len(terms) == 14
    assert terms[0].to_latex() == SAVED_TERM_0_LATEX
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f'term{n}.svg' for n in range(14))


def test_res_cond_terms_to_latex_render_one_svg_per_term(tmp_path):
    for n, term in enumerate(generate_only_res_cond_evv_term_selection()):
        get_plt_latex(term.to_latex(), savename=str(tmp_path / f'term_custom{n}.svg'))

    assert sorted(p.name for p in tmp_path.iterdir()) == [f'term_custom{n}.svg' for n in range(5)]
