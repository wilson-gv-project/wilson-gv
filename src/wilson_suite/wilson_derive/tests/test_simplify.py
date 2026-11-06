import pytest

from wilson_suite.wilson_derive.simplify import terms_simplify

def test_terms_simplify():

    # This unit test was Claude generated and reviewed

    import copy
    from fractions import Fraction
    from wilson_suite.wilson_derive.derive import get_terms_by_comp_pattern_and_magn_conds
    from wilson_suite.wilson_utils import common_labels as wu_common

    order = 3
    tot_anharm_limit = 1
    el_anharm_limit = 1
    mech_anharm_limit = 1
    comp_patt = (1, 1, 2)

    # At -k1 + k2 (+ k3): w1 is negative, w2 is positive, here constricting to w1 + w2 signif. > 0
    mc_plus = ((1,), (0, 1,),)
    mc_minus = ((0,),)

    # Getting terms
    # The assertions in this test depend on wilson-derive retaining ordering of terms
    terms = get_terms_by_comp_pattern_and_magn_conds(order=order, total_anharm_limit=tot_anharm_limit,
                                                     el_anharm_limit=el_anharm_limit,
                                                     mech_anharm_limit=mech_anharm_limit,
                                                     comp_pattern=comp_patt,
                                                     magn_conds=(mc_plus, mc_minus))

    el_terms = [terms[0], terms[1]]

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