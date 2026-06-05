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
            # Two create/annihilate patterns: (-, +, -, +) and (-, -, +, +)
            # Starting from original indices: a, b, c, d
            # (-, +, -, +): Only a, a, b, b possible
            #   - Must have c = d, must have a = b, rename c to b
            #       - (Full summation over a and b)
            #       - Hermite coefficient 1/4
            # (-, -, +, +):
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
            #           - This results in full summations (a, b) over these with Hermite coefficient 1/4

            assert walks == [[['a', -1], ['a', 1], ['b', -1], ['b', 1]],
                             [['a', -1], ['b', -1], ['a', 1], ['b', 1]],
                             [['a', -1], ['b', -1], ['b', 1], ['a', 1]]]

        elif n_pairs == 3:
            # Create/annihilate patterns:
            # 1 (-, -, -, +, +, +) (J, K, L, M, N, O)
            # 2 (-, -, +, -, +, +) (E, F, H, I)
            # 3 (-, +, -, -, +, +) (B, C)
            # 4 (-, -, +, +, -, +) (D, G)
            # 5 (-, +, -, +, -, +) (A)

            # Starting from original indices: a, b, c, d, e, f

            # Pattern 1:
            #   - Must have (d, e, f) = (a, b, c): Six permutations ("origins" of J, K, L, M, N, O)
            #   - Taking summation structure: sum(a, b, c) (all) =
            #     [all unequal] + (a = b, c != a) + (a = c, b != a) + (b = c, a != b) + (a = b = c)
            #   - All unequal: Straightforward 1/8 Hermite factor for each
            #   - (a = b, c != a):  (d, e, f) must be either (a, a, c), or (a, c, a), or (c, a, a)
            #   - Each leads to a 1/4 Hermite factor and can be distributed (half each) into resp.
            #     (J, L), (K, M), (N, O)
            #   - Analogous patterns for (a = c, b != a) + (b = c, a != b)
            #   - (a = b = c): (d, e, f) can only be (a, a, a), giving a 6/8 Hermite factor which can be
            #     distributed (1/6 each) into each of J, K, L, M, N, O
            #   - In total full a, b, c summation with Hermite factor 1/8
            #
            # Pattern 2:
            #   - Rename indices to a, b, d, c, e, f
            #   - Must have either f = c  or e = c
            #       - If f = c: e must be either b or a (and d must be the other of these): Origin of E and H
            #       - If e = c: f must be either b or a (and d must be the other of these): Origin of F and I
            #   - With summation structure as introduced in pattern 1
            #   - All unequal: Straightforward 1/8 Hermite factor for each
            #   - (a = b, c != a):  (d, e, f) must be either (a, c, a) or (a, a, c)
            #   - Each leads to a 1/4 factor distr. into resp (E, H) and (F, I)
            #   - Analogous patterns for (a = c, b != a) + (b = c, a != b)
            #   (a = b = c): (d, e, f) can only be (a, a, a), giving a 1/2 Hermite factor distr. (1/4 each) into
            #   each of (E, F, H, I)
            #   - Altogether full a, b, c summation with Hermite factor 1/8
            #
            # Pattern 3:
            #   - Rename indices to a, d, b, c, e, f
            #   - d must be a and (e, f) must be either (b, c) or (c, b) (origin of resp. B and C)
            #   - Same summation structure: All unequal straightforward 1/8 Hermite factor
            #   - (a = b, c != a): (d, e, f) must be either (a, a, c) or (a, c, a)
            #   - Each give a 1/8 factor aligning with resp. B and C
            #   - (a = c, b != a): Analogous pattern
            #   - (b = c, a != b): (d, e, f) must be (a, b, b) giving a 1/4 factor distr. into B and C
            #   - (a = b = c): (d, e, f) must be (a, a, a) giving a 1/4 factor distr. into B and C
            #   - Altogether full a, b, c summation with Hermite factor 1/8
            #
            # Pattern 4:
            #   - Rename indices to a, b, d, e, c, f
            #   - f must be c and (d, e) must be either (a, b) or (b, a): Origin of D and G
            #   - All unequal gives 1/8 factor
            #   - (a = b, c != a): (d, e, f) must be (a, a, c) with factor 1/4 distr. into resp. D and G
            #   - (a = c, c != a): (d, e, f) must be (a, b, a) or (b, a, a) each with 1/8 factor distr. into resp D and G
            #   - (b = c, a != b): (d, e, f) must be (a, b, b) or (b, a, b) each with 1/8 factor distr. into resp D and G
            #   - (a = b = c): (d, e, f) must be (a, a, a) giving 1/4 factor distr. into D and G
            #   - Altogether full a, b, c summation with Hermite factor 1/8
            #
            # Pattern 5:
            #   - Least complicated: Must have a = b, c = d, e = f
            #   - Rename from c to b, from e to c
            #   - Full summations over a, b, c follow almost directly and Hermite factor 1/8

            assert walks == [[['a', -1], ['a', 1], ['b', -1], ['b', 1], ['c', -1], ['c', 1]], # A: 5
                             [['a', -1], ['a', 1], ['b', -1], ['c', -1], ['b', 1], ['c', 1]], # B: 3
                             [['a', -1], ['a', 1], ['b', -1], ['c', -1], ['c', 1], ['b', 1]], # C: 3
                             [['a', -1], ['b', -1], ['a', 1], ['b', 1], ['c', -1], ['c', 1]], # D: 4
                             [['a', -1], ['b', -1], ['a', 1], ['c', -1], ['b', 1], ['c', 1]], # E: 2
                             [['a', -1], ['b', -1], ['a', 1], ['c', -1], ['c', 1], ['b', 1]], # F: 2
                             [['a', -1], ['b', -1], ['b', 1], ['a', 1], ['c', -1], ['c', 1]], # G: 4
                             [['a', -1], ['b', -1], ['b', 1], ['c', -1], ['a', 1], ['c', 1]], # H: 2
                             [['a', -1], ['b', -1], ['b', 1], ['c', -1], ['c', 1], ['a', 1]], # I: 2
                             [['a', -1], ['b', -1], ['c', -1], ['a', 1], ['b', 1], ['c', 1]], # J: 1
                             [['a', -1], ['b', -1], ['c', -1], ['a', 1], ['c', 1], ['b', 1]], # K: 1
                             [['a', -1], ['b', -1], ['c', -1], ['b', 1], ['a', 1], ['c', 1]], # L: 1
                             [['a', -1], ['b', -1], ['c', -1], ['b', 1], ['c', 1], ['a', 1]], # M: 1
                             [['a', -1], ['b', -1], ['c', -1], ['c', 1], ['a', 1], ['b', 1]], # N: 1
                             [['a', -1], ['b', -1], ['c', -1], ['c', 1], ['b', 1], ['a', 1]]] # O: 1






def test_go_for_a_walk():



    pass

def test_do_hermaut():

    pass

