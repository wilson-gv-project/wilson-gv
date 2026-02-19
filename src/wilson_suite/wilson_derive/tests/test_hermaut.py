import pytest

from wilson_suite.wilson_derive.hermaut import all_uneq_walks, go_for_a_walk, do_hermaut
from wilson_suite.wilson_utils import common_labels as wu_common

def test_all_uneq_walks():

    for n in range(3):

        n_pairs = n + 1

        inds = wu_common.nm_inds
        first_walk = []
        walks = []
        closed = []

        all_uneq_walks(first_walk, n_pairs, inds[:n_pairs], closed, walks)

        print('Walks for n_pairs = ', n_pairs)
        print(walks)

        if n_pairs == 1:
            # Trivial (reading right to left): Up one and down one
            assert walks == [[['a', -1], ['a', 1]]]

        elif n_pairs == 2:
            # Two create/annihilate patterns (R to L): (D, U, D, U) and (D, D, U, U)
            # Starting from original indices: a, b, c, d
            # (D, U, D, U): Only a, a, b, b possible
            #   - Must have c = d, must have a = b, rename c to b
            #       - (Full summation over a and b)
            # (D, D, U, U):
            #   - If a != b:
            #       - Must have (c = a and d = b) or (c = b and d = a)
            #           - c = a and d = b:
            #               - a, b, a, b (Hermite coefficient 1/4)
            #           - c = b and d = a
            #               - a, b, b, a (Hermite coefficient 1/4)
            #   - If a = b:
            #       - Then must have c = d
            #           - a, a, a, a (Hermite coefficient 1/2)
            #       - Can be distributed (half each) into a != b summations (a, b, a, b) and (a, b, b, a)
            #           - This results in full summations (a, b) over these

            assert walks == [[['a', -1], ['a', 1], ['b', -1], ['b', 1]],
                             [['a', -1], ['b', -1], ['a', 1], ['b', 1]],
                             [['a', -1], ['b', -1], ['b', 1], ['a', 1]]]

        elif n_pairs == 3:
            assert walks == [[['a', -1], ['a', 1], ['b', -1], ['b', 1], ['c', -1], ['c', 1]],
                             [['a', -1], ['a', 1], ['b', -1], ['c', -1], ['b', 1], ['c', 1]],
                             [['a', -1], ['a', 1], ['b', -1], ['c', -1], ['c', 1], ['b', 1]],
                             [['a', -1], ['b', -1], ['a', 1], ['b', 1], ['c', -1], ['c', 1]],
                             [['a', -1], ['b', -1], ['a', 1], ['c', -1], ['b', 1], ['c', 1]],
                             [['a', -1], ['b', -1], ['a', 1], ['c', -1], ['c', 1], ['b', 1]],
                             [['a', -1], ['b', -1], ['b', 1], ['a', 1], ['c', -1], ['c', 1]],
                             [['a', -1], ['b', -1], ['b', 1], ['c', -1], ['a', 1], ['c', 1]],
                             [['a', -1], ['b', -1], ['b', 1], ['c', -1], ['c', 1], ['a', 1]],
                             [['a', -1], ['b', -1], ['c', -1], ['a', 1], ['b', 1], ['c', 1]],
                             [['a', -1], ['b', -1], ['c', -1], ['a', 1], ['c', 1], ['b', 1]],
                             [['a', -1], ['b', -1], ['c', -1], ['b', 1], ['a', 1], ['c', 1]],
                             [['a', -1], ['b', -1], ['c', -1], ['b', 1], ['c', 1], ['a', 1]],
                             [['a', -1], ['b', -1], ['c', -1], ['c', 1], ['a', 1], ['b', 1]],
                             [['a', -1], ['b', -1], ['c', -1], ['c', 1], ['b', 1], ['a', 1]]]






def test_go_for_a_walk():



    pass

def test_do_hermaut():

    pass

