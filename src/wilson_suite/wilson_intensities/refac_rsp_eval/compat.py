"""
Adapters to wilson_derive and wilson_experiment as they are on main. Three things the refactor uses are not there:
    translate_terms_to_axis_variables   a flat term list in and out; main's takes and gives {i: {j: [terms]}}
    term_to_str                         VibPerturbedTerm.to_str, which main does not have
    derive_terms                        VibExperiment.derive_terms (a flat term list), which main does not have

Once derive and the experiment have them, import them from there and delete this file.
"""
from collections.abc import Iterable
from typing import TYPE_CHECKING

from wilson_suite.wilson_derive import term_var_translate
from wilson_suite.wilson_utils.prop_trivname import prop_trivname

if TYPE_CHECKING:
    from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
    from wilson_suite.wilson_experiment.experiment_abstractions import VibExperiment
    from wilson_suite.wilson_experiment.indep_vars_and_axes import SpectralAxisSet


def translate_terms_to_axis_variables(terms: Iterable['VibPerturbedTerm'],
                                      chosen_axes: 'SpectralAxisSet') -> list['VibPerturbedTerm']:
    """
    Copies of the terms in the chosen axes, same order, e.g. the 14 EVV terms in pulse IDs -> the 14 terms in A, B.
    Main's function walks {i: {j: [terms]}}: the flat list goes in under one key i and one key j, and comes out there.
    """
    return term_var_translate.translate_terms_to_axis_variables({0: {0: list(terms)}}, chosen_axes)[0][0]


def derive_terms(experiment: 'VibExperiment') -> list['VibPerturbedTerm']:
    """
    The experiment's terms as derive writes them (pulse IDs), as one flat list, e.g. CARS: 4 terms.
    Main's derive gives {number of anharmonicities: {anharmonicity: [terms]}}; derived_terms_flat flattens it.
    Imported here, not at the top: derive is only needed by the callers of this function.
    """
    from wilson_suite.wilson_derive.derive import get_fully_enhanced_terms
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_flat

    return derived_terms_flat(get_fully_enhanced_terms(experiment=experiment), tolistonly=True)


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
