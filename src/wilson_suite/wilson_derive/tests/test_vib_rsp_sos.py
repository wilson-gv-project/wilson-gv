import pytest

from wilson_suite.wilson_derive.vib_rsp_sos import get_vib_sos
from wilson_suite.wilson_derive.abstractions import QOperator, VibStateSymbolic
from wilson_suite.wilson_utils import common_labels as wu_common

def test_get_vib_sos():

    # TODO: Either go with this specific test or hard code own test for full 3rd order
    #  DECISION: Go with full 3rd order

    # Test case: EVV terms
    order = 3
    R_sos = get_vib_sos(order)

    # Selecting EVV relevant terms (i.e. "mu**2 alpha" terms with alpha operators restricted to [0,3])
    # If testing like this, then also test elsewhere that the other contribs are removed from consideration in the EVV results
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

    # NOTE: Can actually bring this down to one (sign of first interaction for 4 -> 2, pattern of max enhancement possible for 2 -> 1)

    print('len R_sos_selected', len(R_sos_selected))

    for i in R_sos_selected:
        print('\n\n')
        i.present()

    # For higher orders (preferably up to 6): With final pen/paper verification of method proof, can generate terms and
    # test get_vib_sos results against that (try to do this up to order 6)

    pass
