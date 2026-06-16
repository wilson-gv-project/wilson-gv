import pytest

from wilson_suite.wilson_derive.simplify import terms_simplify

def test_terms_simplify():

    # This unit test was Claude generated and reviewed

    import copy
    from fractions import Fraction
    from wilson_suite.fixtures import evv_experiment
    from wilson_suite.wilson_derive.derive import get_fully_enhanced_terms
    from wilson_suite.wilson_utils import common_labels as wu_common

    evv_exp = evv_experiment()
    terms = get_fully_enhanced_terms(experiment=evv_exp)

    el_terms = list(terms[1][(1, 0)])
    assert len(el_terms) == 2

    # Two identical terms should combine: coefficient doubles
    t0 = copy.deepcopy(el_terms[0])
    result_doubled = terms_simplify([t0, copy.deepcopy(el_terms[0])], wu_common.nm_inds)
    assert len(result_doubled) == 1
    combined = list(result_doubled.values())[0]
    assert combined.coeff == el_terms[0].coeff * 2

    # A term and its negative should cancel: result is empty
    t0_neg = copy.deepcopy(el_terms[0])
    t0_neg.coeff *= Fraction(-1)
    t0_neg.was_sorted = False
    result_cancelled = terms_simplify([copy.deepcopy(el_terms[0]), t0_neg], wu_common.nm_inds)
    assert len(result_cancelled) == 0

    # Two structurally distinct terms should both survive
    hashes_before = [i.h(also_sort=True, nm_inds=wu_common.nm_inds) for i in el_terms]
    result_distinct = terms_simplify(
        [copy.deepcopy(el_terms[0]), copy.deepcopy(el_terms[1])], wu_common.nm_inds
        )
    assert len(result_distinct) == 2
    hashes_after = [result_distinct[i].h(also_sort=True, nm_inds=wu_common.nm_inds) for i in result_distinct]
    assert hashes_before == hashes_after