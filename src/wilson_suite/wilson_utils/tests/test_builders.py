"""
builders.py:
    make_SpectralAxisSet     {label: pulse numbers} -> SpectralAxisSet
    show_valid_axis_combs    the axis choices of an experiment as text
    show_term_latex          a VibPerturbedTerm as LaTeX, whole or one part
The terms: the EVV experiment's 14 fully enhanced terms, before and after the translation to the axes
A = -w1, B = -w1 + w2.
"""
import wilson_suite as ws
from wilson_suite.wilson_experiment.indep_vars_and_axes import (
    IndependentVariableSet,
    SignedPulseTuple,
    SpectralAxis,
    SpectralAxisSet,
)

from ..builders import make_SpectralAxisSet, show_term_latex, show_valid_axis_combs

evv_exp = ws.fixtures.evv_experiment()
terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)

axis_choice = make_SpectralAxisSet({'A': [-1], 'B': [-1, 2]})
translated_terms = ws.derive.term_var_translate.translate_terms_to_axis_variables(terms, axis_choice)

flat_dict_orig = ws.utils.termdict_from_symb_term.derived_terms_flat(terms, tolistonly=False)
flat_dict = ws.utils.termdict_from_symb_term.derived_terms_flat(translated_terms, tolistonly=False)


## make_SpectralAxisSet --------------------------------------------------------

def test_make_SpectralAxisSet_one_axis_per_label_one_SignedPulseTuple_per_pulse():
    """A = -w1, B = -w1 + w2"""
    axis_set = make_SpectralAxisSet({'A': [-1], 'B': [-1, 2]})

    assert axis_set == SpectralAxisSet(axes=(
        SpectralAxis(label='A', var_set=IndependentVariableSet(var_set=(SignedPulseTuple(pulse_refs=(-1,)),))),
        SpectralAxis(label='B', var_set=IndependentVariableSet(var_set=(SignedPulseTuple(pulse_refs=(-1,)),
                                                                        SignedPulseTuple(pulse_refs=(2,))))),
    ))


## show_valid_axis_combs -------------------------------------------------------

# the EVV experiment: one phase matching condition, four axis sets
EVV_VALID_AXIS_COMBS_TEXT = '\n'.join([
    '0 -- axes choices for phase matching condition (-1, 2, 3)',
    'Independent variables are: (SignedPulseTuple(pulse_refs=(-1,)), SignedPulseTuple(pulse_refs=(2,)))',
    '',
    'SpectralAxisSet 0 with axes:',
    '---- SpectralAxis with label A and var_set (SignedPulseTuple(pulse_refs=(-1,)),)',
    '---- SpectralAxis with label B and var_set (SignedPulseTuple(pulse_refs=(-1,)), SignedPulseTuple(pulse_refs=(2,)))',
    '',
    'SpectralAxisSet 1 with axes:',
    '---- SpectralAxis with label A and var_set (SignedPulseTuple(pulse_refs=(-1,)),)',
    '---- SpectralAxis with label B and var_set (SignedPulseTuple(pulse_refs=(2,)),)',
    '',
    'SpectralAxisSet 2 with axes:',
    '---- SpectralAxis with label A and var_set (SignedPulseTuple(pulse_refs=(2,)),)',
    '---- SpectralAxis with label B and var_set (SignedPulseTuple(pulse_refs=(-1,)),)',
    '',
    'SpectralAxisSet 3 with axes:',
    '---- SpectralAxis with label A and var_set (SignedPulseTuple(pulse_refs=(2,)),)',
    '---- SpectralAxis with label B and var_set (SignedPulseTuple(pulse_refs=(-1,)), SignedPulseTuple(pulse_refs=(2,)))',
])


def test_show_valid_axis_combs_returns_the_text_with_output_true():
    assert show_valid_axis_combs(evv_exp.valid_axis_combs, output=True) == EVV_VALID_AXIS_COMBS_TEXT


def test_show_valid_axis_combs_prints_the_text_by_default(capsys):
    assert show_valid_axis_combs(evv_exp.valid_axis_combs) is None
    assert capsys.readouterr().out == EVV_VALID_AXIS_COMBS_TEXT + '\n'


## show_term_latex -------------------------------------------------------------

def test_show_term_latex_parts_of_a_translated_term():
    term = flat_dict['0_(1, 0)']

    assert show_term_latex(term, 'coeff') == r'\frac{-1}{4}'
    assert show_term_latex(term, 'freqterms') == r'\frac{1}{\omega_{a,}\omega_{b,}}'
    assert show_term_latex(term, 'props') == (r'\frac{\partial\mu_{\beta}}{\partial Q_{a}}'
                                              r'\frac{\partial\mu_{\gamma}}{\partial Q_{b}}'
                                              r'\frac{\partial^{2}\alpha_{\alpha\delta}}{\partial Q_{a}\partial Q_{b}}')
    assert show_term_latex(term, 'res') == r'\frac{1}{(\omega_{,a} -A)(\omega_{b,a} -B)}'


def test_show_term_latex_whole_term_is_the_parts_in_order():
    """coeff, freqterms, props, res"""
    for term in flat_dict.values():
        parts = [show_term_latex(term, part) for part in ('coeff', 'freqterms', 'props', 'res')]
        assert show_term_latex(term) == ''.join(parts)


def test_show_term_latex_resonance_conditions_before_and_after_the_translation():
    """before: pulse numbers (+1 = -(-w1), +1-2 = -(-w1 + w2)); after: the axis names A and B"""
    assert show_term_latex(flat_dict_orig['0_(1, 0)'], 'res') == r'\frac{1}{(\omega_{,a} +1)(\omega_{b,a} +1-2)}'
    assert show_term_latex(flat_dict['0_(1, 0)'], 'res') == r'\frac{1}{(\omega_{,a} -A)(\omega_{b,a} -B)}'


def test_show_term_latex_same_as_VibPerturbedTerm_to_latex():
    assert len(flat_dict) == len(flat_dict_orig) == 14
    for term in [*flat_dict.values(), *flat_dict_orig.values()]:
        assert show_term_latex(term) == term.to_latex()
