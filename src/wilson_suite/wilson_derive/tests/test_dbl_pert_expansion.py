import pytest

from wilson_suite.wilson_derive.dbl_pert_expansion import expand_term, make_anharm_orders, make_anharm_orders_rec

def test_expand_term():



    pass

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