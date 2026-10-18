"""
Adapter to wilson_derive as it is on main. One thing the refactor uses is not there:
    term_to_str                         VibPerturbedTerm.to_str, which main does not have

Once derive has it, import it from there and delete this file.
The translation to axes and derive itself are not used here: the evaluator takes terms already translated
(data_for_tests/translated_terms, written by make_translated_terms.py there).
"""
from typing import TYPE_CHECKING

from wilson_suite.wilson_utils.prop_trivname import prop_trivname

if TYPE_CHECKING:
    from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm


def term_to_str(term: 'VibPerturbedTerm', nm_inds_for_sort=None) -> str:
    """
    One line per term, e.g. for an error message: coeff * props / freq terms / resonance conditions.
    Uses term.to_str when the term has it (debug_rendering's VibPerturbedTerm, the fake terms in test_plan.py).
    Sorts the term first if it is not sorted yet, as VibPerturbedTerm.to_str does.
    """
    if hasattr(term, 'to_str'):
        return term.to_str(nm_inds_for_sort)

    if not term.was_sorted:
        term.sort(nm_inds_for_sort)

    coeff = f'{term.coeff.numerator}/{term.coeff.denominator}'
    props = [f"{prop_trivname(ord_el=len(prop.ops), ord_geo=prop.dord)}_({','.join(prop.inds)})" for prop in term.props]
    freqterms = [f"<{ft.sl.q},{ft.sr.q}>" for ft in term.freqterms]
    res = [f'(<{r.diff.sl.q}{r.diff.sr.q}> - {r.pf} -iG)' for r in term.res]

    return coeff + ' * ' + ' * '.join(props) + ' / ' + ' / '.join(freqterms) + ' / ' + ' / '.join(res)
