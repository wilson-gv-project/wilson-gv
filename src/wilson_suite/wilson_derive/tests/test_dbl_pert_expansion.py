import pytest

from wilson_suite.wilson_derive.dbl_pert_expansion import expand_term, make_anharm_orders, make_anharm_orders_rec

def test_expand_term():

    # This unit test was Claude generated and reviewed

    import copy
    from fractions import Fraction
    from wilson_suite.fixtures import evv_experiment
    from wilson_suite.wilson_derive.vib_rsp_sos import get_vib_sos

    # Note: This is likely an appropriate truncation but that's not so important since testing on just one of these
    # terms should be sufficient
    evv_exp = evv_experiment()
    R_sos = get_vib_sos(evv_exp.order)
    R_sos_evv = []
    for int_seq in evv_exp.int_sequences:
        for term in R_sos:
            t = copy.deepcopy(term)
            t.dressWithPulseInteractions(int_seq)
            if t.allElRspEpochContained(evv_exp.epochs, 0) and t.allUVCancels(evv_exp.cfuv):
                R_sos_evv.append(t)
    assert len(R_sos_evv) == 4

    term = R_sos_evv[0]
    n_ints = len(term.ints)
    assert n_ints == 3
    orig_coeff = term.coeff

    # (0,0): harmonic - one term, all dord=1, coefficient unchanged
    result_00 = expand_term(term, 0, 0)
    assert len(result_00) == 1
    assert all(i.prop.dord == 1 for i in result_00[0].ints)
    assert result_00[0].coeff == orig_coeff

    # (1,0): electrical anharmonicity - integral k gets order 2 in position k, otherwise order 1 for other positions
    result_10 = expand_term(term, 1, 0)
    assert len(result_10) == n_ints
    for k, sub in enumerate(result_10):
        for j, intgr in enumerate(sub.ints):
            assert intgr.prop.dord == (2 if j == k else 1)
        assert sub.coeff == orig_coeff * Fraction(1, 2)

    # (0,1): mechanical anharmonicity — two sub-terms per integral (bra and ket perturbation)
    result_01 = expand_term(term, 0, 1)
    assert len(result_01) == 2 * n_ints
    for sub in result_01:
        assert len(sub.ints) == n_ints + 1
        cubic_ints = [i for i in sub.ints if i.prop.dord == 3]
        assert len(cubic_ints) == 1
        pert_wf_diffs = [fd for fd in sub.freqdiff if fd.is_pert_wf_diff]
        assert len(pert_wf_diffs) == 1
        assert sub.coeff == orig_coeff * Fraction(-1, 6)

    # Extra checks for integral/fd insertion correctness

    assert ([(k.bra.s, k.prop.dord, [m.o for m in k.prop.ops], k.ket.s) for k in result_01[0].ints] ==
            [('0', 3, [], 'A'), ('A', 1, [0, 3], 'n'), ('n', 1, [2], 'm'), ('m', 1, [1], '0')])

    assert [(k.sl.s, k.sr.s, k.is_pert_wf_diff) for k in result_01[0].freqdiff] == [('A', '0', True)]

    assert ([(k.bra.s, k.prop.dord, [m.o for m in k.prop.ops], k.ket.s) for k in result_01[1].ints] ==
            [('0', 1, [0, 3], 'A'), ('A', 3, [], 'n'), ('n', 1, [2], 'm'), ('m', 1, [1], '0')])

    assert [(k.sl.s, k.sr.s, k.is_pert_wf_diff) for k in result_01[1].freqdiff] == [('A', 'n', True)]

    assert ([(k.bra.s, k.prop.dord, [m.o for m in k.prop.ops], k.ket.s) for k in result_01[2].ints] ==
            [('0', 1, [0, 3], 'n'), ('n', 3, [], 'A'), ('A', 1, [2], 'm'), ('m', 1, [1], '0')])

    assert [(k.sl.s, k.sr.s, k.is_pert_wf_diff) for k in result_01[2].freqdiff] == [('A', 'n', True)]

    assert ([(k.bra.s, k.prop.dord, [m.o for m in k.prop.ops], k.ket.s) for k in result_01[3].ints] ==
            [('0', 1, [0, 3], 'n'), ('n', 1, [2], 'A'), ('A', 3, [], 'm'), ('m', 1, [1], '0')])

    assert [(k.sl.s, k.sr.s, k.is_pert_wf_diff) for k in result_01[3].freqdiff] == [('A', 'm', True)]

    assert ([(k.bra.s, k.prop.dord, [m.o for m in k.prop.ops], k.ket.s) for k in result_01[4].ints] ==
            [('0', 1, [0, 3], 'n'), ('n', 1, [2], 'm'), ('m', 3, [], 'A'), ('A', 1, [1], '0')])

    assert [(k.sl.s, k.sr.s, k.is_pert_wf_diff) for k in result_01[4].freqdiff] == [('A', 'm', True)]

    assert ([(k.bra.s, k.prop.dord, [m.o for m in k.prop.ops], k.ket.s) for k in result_01[5].ints] ==
            [('0', 1, [0, 3], 'n'), ('n', 1, [2], 'm'), ('m', 1, [1], 'A'), ('A', 3, [], '0')])

    assert [(k.sl.s, k.sr.s, k.is_pert_wf_diff) for k in result_01[5].freqdiff] == [('A', '0', True)]

def test_make_anharm_orders():


    # Case: (as for "regular EVV"): Up to 1st order el, mech, and up to 1st order total

    limit_total = 1
    limit_el = 1
    limit_mech = 1

    orders = make_anharm_orders(limit_total, limit_el, limit_mech)

    assert len(orders) == 2
    assert orders[0] == [(0, 0)]
    assert orders[1] == [(1, 0), (0, 1)]

    # Up to 1st order el, 2nd order mech, 3rd order total

    limit_total = 3
    limit_el = 1
    limit_mech = 2

    orders = make_anharm_orders(limit_total, limit_el, limit_mech)

    assert len(orders) == 4
    assert orders[0] == [(0, 0)]
    assert orders[1] == [(1, 0), (0, 1)]
    assert orders[2] == [(1, 1), (0, 2)]
    assert orders[3] == [(1, 2)]

    # Up to 5th order el, 2nd order mech, 4th order total

    limit_total = 4
    limit_el = 5
    limit_mech = 2

    orders = make_anharm_orders(limit_total, limit_el, limit_mech)

    assert len(orders) == 5
    assert orders[0] == [(0, 0)]
    assert orders[1] == [(1, 0), (0, 1)]
    assert orders[2] == [(2, 0), (1, 1), (0, 2)]
    assert orders[3] == [(3, 0), (2, 1), (1, 2)]
    assert orders[4] == [(4, 0), (3, 1), (2, 2)]

def test_make_anharm_orders_rec():

    # Test deferred: Only used by make_anharm_orders which is close to being a wrapper for _rec, so test of
    # make_anharm_orders deemed sufficient for now

    pass