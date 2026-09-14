from fractions import Fraction

import pytest

from wilson_suite.wilson_derive.vib_rsp_sos import get_vib_sos

def test_get_vib_sos():

    # TODO: This test now for EVV terms. For general testing, need more complete testing at at least 3rd order and should have
    #  spot checks at some other orders (preferably up to order 6)

    # Test case: EVV relevant terms: "mu**2 alpha" terms with alpha operators restricted to [0,3]
    order = 3
    R_sos = get_vib_sos(order)

    # Selecting EVV relevant terms (i.e. "mu**2 alpha" terms with alpha operators restricted to [0,3]). NOTE: Further reduction
    # possible but not done here.
    # Since testing like this, then also test elsewhere that the other contribs are removed from consideration in the EVV results
    # Specifically: Verify that non-mu**2 alpha terms are removed and that non-epoch matching mu**2 alpha terms are removed

    R_sos_selected = []

    for i in R_sos:

        # Initialize matching flag
        matching = True

        # "mu**2 alpha" pattern
        mu2alpha_pattern = [1, 1, 2]

        # Matching against pattern and experiment epoch conditions
        for j in i.ints:

            # Test if matching epoch conditions
            if len(j.prop.ops) == 2:
                if not(sorted([k.o for k in j.prop.ops]) == [0, 3]):
                    matching = False

            # Test against mu**2 alpha pattern
            if len(j.prop.ops) in mu2alpha_pattern:
                mu2alpha_pattern.remove(len(j.prop.ops))
            else:
                matching = False

        if mu2alpha_pattern == [] and matching:
            R_sos_selected.append(i)

    # Should here have four terms
    assert(len(R_sos_selected) == 4)

    # Corresponds to derivation p.22 res. term 1
    t = R_sos_selected[0]

    assert t.coeff == Fraction(1)
    assert len(t.ints) == 3

    assert t.ints[0].bra.s == '0'
    assert len(t.ints[0].prop.ops) == 2
    assert t.ints[0].prop.ops[0].o == 0
    assert t.ints[0].prop.ops[1].o == 3
    assert t.ints[0].prop.dord == 0
    assert t.ints[0].ket.s == 'n'

    assert t.ints[1].bra.s == 'n'
    assert len(t.ints[1].prop.ops) == 1
    assert t.ints[1].prop.ops[0].o == 2
    assert t.ints[1].prop.dord == 0
    assert t.ints[1].ket.s == 'm'

    assert t.ints[2].bra.s == 'm'
    assert len(t.ints[2].prop.ops) == 1
    assert t.ints[2].prop.ops[0].o == 1
    assert t.ints[2].prop.dord == 0
    assert t.ints[2].ket.s == '0'

    assert len(t.res) == 2
    assert t.res[0].diff.sl.s == 'm'
    assert t.res[0].diff.sr.s == '0'
    assert t.res[0].pf == [1]

    assert t.res[1].diff.sl.s == 'n'
    assert t.res[1].diff.sr.s == '0'
    assert t.res[1].pf == [1, 2]

    # Corresponds to derivation p.21 res. term 2
    t = R_sos_selected[1]

    assert t.coeff == Fraction(-1)
    assert len(t.ints) == 3

    assert t.ints[0].bra.s == '0'
    assert len(t.ints[0].prop.ops) == 1
    assert t.ints[0].prop.ops[0].o == 2
    assert t.ints[0].prop.dord == 0
    assert t.ints[0].ket.s == 'n'

    assert t.ints[1].bra.s == 'n'
    assert len(t.ints[1].prop.ops) == 2
    assert t.ints[1].prop.ops[0].o == 0
    assert t.ints[1].prop.ops[1].o == 3
    assert t.ints[1].prop.dord == 0
    assert t.ints[1].ket.s == 'm'

    assert t.ints[2].bra.s == 'm'
    assert len(t.ints[2].prop.ops) == 1
    assert t.ints[2].prop.ops[0].o == 1
    assert t.ints[2].prop.dord == 0
    assert t.ints[2].ket.s == '0'

    assert len(t.res) == 2
    assert t.res[0].diff.sl.s == 'm'
    assert t.res[0].diff.sr.s == '0'
    assert t.res[0].pf == [1]

    assert t.res[1].diff.sl.s == 'm'
    assert t.res[1].diff.sr.s == 'n'
    assert t.res[1].pf == [1, 2]


    # Corresponds to derivation p.21 res. term 1
    t = R_sos_selected[2]

    assert t.coeff == Fraction(-1)
    assert len(t.ints) == 3

    assert t.ints[0].bra.s == '0'
    assert len(t.ints[0].prop.ops) == 1
    assert t.ints[0].prop.ops[0].o == 1
    assert t.ints[0].prop.dord == 0
    assert t.ints[0].ket.s == 'm'

    assert t.ints[1].bra.s == 'm'
    assert len(t.ints[1].prop.ops) == 2
    assert t.ints[1].prop.ops[0].o == 0
    assert t.ints[1].prop.ops[1].o == 3
    assert t.ints[1].prop.dord == 0
    assert t.ints[1].ket.s == 'n'

    assert t.ints[2].bra.s == 'n'
    assert len(t.ints[2].prop.ops) == 1
    assert t.ints[2].prop.ops[0].o == 2
    assert t.ints[2].prop.dord == 0
    assert t.ints[2].ket.s == '0'

    assert len(t.res) == 2
    assert t.res[0].diff.sl.s == '0'
    assert t.res[0].diff.sr.s == 'm'
    assert t.res[0].pf == [1]

    assert t.res[1].diff.sl.s == 'n'
    assert t.res[1].diff.sr.s == 'm'
    assert t.res[1].pf == [1, 2]

    # Corresponds to derivation p.22 res. term 2
    t = R_sos_selected[3]

    assert t.coeff == Fraction(1)
    assert len(t.ints) == 3

    assert t.ints[0].bra.s == '0'
    assert len(t.ints[0].prop.ops) == 1
    assert t.ints[0].prop.ops[0].o == 1
    assert t.ints[0].prop.dord == 0
    assert t.ints[0].ket.s == 'm'

    assert t.ints[1].bra.s == 'm'
    assert len(t.ints[1].prop.ops) == 1
    assert t.ints[1].prop.ops[0].o == 2
    assert t.ints[1].prop.dord == 0
    assert t.ints[1].ket.s == 'n'

    assert t.ints[2].bra.s == 'n'
    assert len(t.ints[2].prop.ops) == 2
    assert t.ints[2].prop.ops[0].o == 0
    assert t.ints[2].prop.ops[1].o == 3
    assert t.ints[2].prop.dord == 0
    assert t.ints[2].ket.s == '0'

    assert len(t.res) == 2
    assert t.res[0].diff.sl.s == '0'
    assert t.res[0].diff.sr.s == 'm'
    assert t.res[0].pf == [1]

    assert t.res[1].diff.sl.s == '0'
    assert t.res[1].diff.sr.s == 'n'
    assert t.res[1].pf == [1, 2]