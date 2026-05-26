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
            #   - (a = c, b != a): (d, e, f) must be (a, b, a) or (b, a, a) each with 1/8 factor distr. into resp D and G
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


# Claude generated, fixture to get selected terms for testing of hermaut functionality.
# The exact form of these terms is less important for their use in testing but here nevertheless are EVV relevant.
def _get_evv_filtered_sos():
    import copy
    from wilson_suite.fixtures import evv_experiment
    from wilson_suite.wilson_derive.vib_rsp_sos import get_vib_sos
    evv_exp = evv_experiment()
    R_sos = get_vib_sos(evv_exp.order)
    R_sos_evv = []
    for int_seq in evv_exp.int_sequences:
        for term in R_sos:
            t = copy.deepcopy(term)
            t.dressWithPulseInteractions(int_seq)
            if t.allElRspEpochContained(evv_exp.epochs, 0) and t.allUVCancels(evv_exp.cfuv):
                R_sos_evv.append(t)
    return R_sos_evv

# Claude generated, has been reviewed and amended
def test_go_for_a_walk():

    from fractions import Fraction
    from wilson_suite.wilson_derive.dbl_pert_expansion import expand_term
    from wilson_suite.wilson_derive.abstractions import HarmOscStateSymbolic

    R_sos_evv = _get_evv_filtered_sos()

    # El. anharm case: dord=4 (2+1+1), n_pairs=2, using D1D1E term
    # expand_term[0] has dord=2 on the first integral, dord=1 on the others
    term_10 = expand_term(R_sos_evv[0], 1, 0)[0]

    walk_a = [['a', -1], ['a', 1], ['b', -1], ['b', 1]]
    res_states, res_deriv_inds = go_for_a_walk(term_10, walk_a)

    # Ground state maps to empty harmonic oscillator
    assert res_states['0'].q == []
    # Intermediate state 'n' also ends up empty (annihilated by subsequent integral)
    assert res_states['n'].q == []
    # Intermediate state 'm' gets one quantum of 'b'
    assert res_states['m'].q == ['b']

    # First integral (dord=2) consumed walk steps at indices 0,1 → both 'a'
    assert res_deriv_inds[0] == ['a', 'a']
    # Remaining two integrals (dord=1 each) consumed steps 'b','b'
    assert res_deriv_inds[1] == ['b']
    assert res_deriv_inds[2] == ['b']

    # Malformed walk with double raising of 'a' index
    with pytest.raises(ValueError):
        walk_bogus = [['a', 1], ['a', 1], ['b', -1], ['b', 1]]
        res_states_bogus, res_deriv_inds_bogus = go_for_a_walk(term_10, walk_bogus)

    # Mech. anharm case: dord=6 (3+1+1+1), n_pairs=3
    # expand_term(0,1)[0] inserts a cubic integral before ints[0], perturbing the bra of the original first integral
    term_01 = expand_term(R_sos_evv[0], 0, 1)[0]
    walk_b = [['a', -1], ['a', 1], ['b', -1], ['b', 1], ['c', -1], ['c', 1]]
    res_states_01, res_deriv_inds_01 = go_for_a_walk(term_01, walk_b)

    assert res_states_01['0'].q == []
    assert res_states_01['n'].q == []
    assert res_states_01['m'].q == ['c']
    assert res_states_01['A'].q == ['b']

    # Cubic integral (dord=3) consumed walk steps for 'a','a','b'
    assert res_deriv_inds_01[0] == ['a', 'a', 'b']
    assert res_deriv_inds_01[1] == ['b']
    assert res_deriv_inds_01[2] == ['c']
    assert res_deriv_inds_01[3] == ['c']

    # Further test case: Another mech anharm term and a different walk
    term_01_2 = expand_term(R_sos_evv[0], 0, 1)[1]

    walk_b_2 = [['a', -1], ['b', -1], ['b', 1], ['c', -1], ['c', 1], ['a', 1]]
    res_states_01_2, res_deriv_inds_01_2 = go_for_a_walk(term_01_2, walk_b_2)

    assert res_states_01_2['0'].q == []
    assert res_states_01_2['n'].q == ['a', 'c']
    assert res_states_01_2['m'].q == ['a']
    assert res_states_01_2['A'].q == ['a']

    assert res_deriv_inds_01_2[0] == ['a']
    assert res_deriv_inds_01_2[1] == ['b', 'b', 'c']
    assert res_deriv_inds_01_2[2] == ['c']
    assert res_deriv_inds_01_2[3] == ['a']

    # Malformed walk with lowering before raising of 'b' index
    with pytest.raises(ValueError):
        walk_bogus = [['a', -1], ['a', 1], ['b', 1], ['b', -1], ['c', -1], ['c', 1]]
        res_states_bogus, res_deriv_inds_bogus = go_for_a_walk(term_01_2, walk_bogus)

# Claude generated, has been reviewed and amended
def test_do_hermaut():

    # Sketch (EVV relevant test):
    # Take one or a few properly prepared VibContribTerm instance (i.e. after expand_term;
    # follow get_fully_enhanced_terms) and do_hermaut with them
    # Include both at least one mech anharm and one el anharm case
    # Inspect results for veracity and build the requisite assertions
    # NOTE: Do this test last in this test file since all_uneq_walks and go_for_a_walk are then verified

    from fractions import Fraction
    from wilson_suite.wilson_derive.dbl_pert_expansion import expand_term
    from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
    from wilson_suite.wilson_utils import common_labels as wu_common

    R_sos_evv = _get_evv_filtered_sos()

    # Harmonic term: total dord=3 (odd) → must return empty list
    harm_term = expand_term(R_sos_evv[0], 0, 0)[0]
    assert do_hermaut(harm_term, wu_common.nm_inds) == []

    # El. anharm term: total dord=4, n_pairs=2 → 3 walks → 3 VibPerturbedTerm results
    el_term = expand_term(R_sos_evv[0], 1, 0)[0]
    el_results = do_hermaut(el_term, wu_common.nm_inds)
    assert len(el_results) == 3
    assert all(isinstance(r, VibPerturbedTerm) for r in el_results)

    for r in el_results:

        assert len(r.props) == 3
        assert len(r.freqterms) == 2
        assert not any(ft.is_pert_wf_diff for ft in r.freqterms)
        # new_coeff = el_term.coeff * 1/(2^2) = Fraction(1,2) * Fraction(1,4)
        assert r.coeff == el_term.coeff * Fraction(1, 4)

    # Verifying details for one term (walk here was [['a', -1], ['b', -1], ['b', 1], ['a', 1]])
    t = el_results[2]

    assert len(t.freqterms) == 2

    # 1/w_a
    assert not (t.freqterms[0].is_pert_wf_diff)
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_b
    assert not (t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == ['b']
    assert t.freqterms[1].sr.q == []

    assert len(t.props) == 3

    # d 2 alpha_{alpha delta}/ d Q_a d Q_b
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 2
    assert t.props[0].inds == ['a', 'b']

    # d mu_gamma/ d Q_b
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 2
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['b']

    # d mu_beta/ d Q_a
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 1
    assert t.props[2].dord == 1
    assert t.props[2].inds == ['a']

    assert len(t.res) == 2

    # 1/w_(a,0)^[-1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == ['a']
    assert t.res[0].diff.sr.q == []
    assert t.res[0].pf == [-1]

    # 1/w_(a + b,0)^[-1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == []
    assert t.res[1].pf == [-1, 2]


    # Mech. anharm term: total dord=6, n_pairs=3 → 15 walks → 15 results
    mech_term = expand_term(R_sos_evv[0], 0, 1)[0]
    mech_results = do_hermaut(mech_term, wu_common.nm_inds)
    assert len(mech_results) == 15
    assert all(isinstance(r, VibPerturbedTerm) for r in mech_results)

    for r in mech_results:
        assert len(r.props) == 4
        assert len(r.freqterms) == 4
        pert_wf_fts = [ft for ft in r.freqterms if ft.is_pert_wf_diff]
        regular_fts = [ft for ft in r.freqterms if not ft.is_pert_wf_diff]
        assert len(pert_wf_fts) == 1
        assert len(regular_fts) == 3
        # new_coeff = mech_term.coeff * 1/(2^3) = Fraction(-1,6) * Fraction(1,8)
        assert r.coeff == mech_term.coeff * Fraction(1, 8)

    # Verifying details for one term (walk here was [['a', -1], ['b', -1], ['b', 1], ['c', -1], ['c', 1], ['a', 1]])
    t = mech_results[8]

    assert len(t.freqterms) == 4

    # 1/(w_a - w_0) (pert WF frequency difference)
    assert t.freqterms[0].is_pert_wf_diff
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_a
    assert not(t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == ['a']
    assert t.freqterms[1].sr.q == []

    # 1/w_b
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == ['b']
    assert t.freqterms[2].sr.q == []

    # 1/w_c
    assert not (t.freqterms[3].is_pert_wf_diff)
    assert t.freqterms[3].sl.q == ['c']
    assert t.freqterms[3].sr.q == []

    assert len(t.props) == 4

    # F_abb
    assert len(t.props[0].ops) == 0
    assert t.props[0].dord == 3
    assert t.props[0].inds == ['a', 'b', 'b']

    # d alpha_{alpha delta}/ d Q_c
    assert len(t.props[1].ops) == 2
    assert t.props[1].ops[0].o == 0
    assert t.props[1].ops[1].o == 3
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['c']

    # d mu_gamma/ d Q_c
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 1
    assert t.props[2].inds == ['c']

    # d mu_beta/ d Q_a
    assert len(t.props[3].ops) == 1
    assert t.props[3].ops[0].o == 1
    assert t.props[3].dord == 1
    assert t.props[3].inds == ['a']

    assert len(t.res) == 2

    # 1/w_(a,0)^[-1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == ['a']
    assert t.res[0].diff.sr.q == []
    assert t.res[0].pf == [-1]

    # 1/w_(a+c,0)^[-1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'c']
    assert t.res[1].diff.sr.q == []
    assert t.res[1].pf == [-1, 2]

    # First bra not ground state
    with pytest.raises(NotImplementedError):
        bogus_term = expand_term(R_sos_evv[0], 1, 0)[0]
        # Goes by is_ground flag, not state name, I think this is appropriate but could lead to confusion if generated
        # manually (that is, a state with the label '0' is currently not understood here as the ground state unless the
        # is_ground flag is set to True, and any state label will be understood as the ground state if the is_ground
        # flag is set to True for that state)
        bogus_term.ints[0].bra.is_ground = False
        bogus_results = do_hermaut(bogus_term, wu_common.nm_inds)

    # Last ket not ground state
    with pytest.raises(NotImplementedError):
        bogus_term = expand_term(R_sos_evv[0], 1, 0)[0]
        bogus_term.ints[-1].ket.is_ground = False
        bogus_results = do_hermaut(bogus_term, wu_common.nm_inds)

    # Not telescopic (flipped integral)
    with pytest.raises(NotImplementedError):
        bogus_term = expand_term(R_sos_evv[0], 1, 0)[0]
        bogus_term.ints[1].bra.s = 'm'
        bogus_term.ints[1].ket.s = 'n'
        bogus_results = do_hermaut(bogus_term, wu_common.nm_inds)

    # Not telescopic (one state changed to weird index)
    with pytest.raises(NotImplementedError):
        bogus_term = expand_term(R_sos_evv[0], 1, 0)[0]
        bogus_term.ints[1].bra.s = 'm'
        bogus_term.ints[1].ket.s = 't'
        bogus_results = do_hermaut(bogus_term, wu_common.nm_inds)

    # Non-unique state progression
    with pytest.raises(AssertionError):
        bogus_term = expand_term(R_sos_evv[0], 1, 0)[0]
        bogus_term.ints[1].ket.s = 'n'
        bogus_term.ints[2].bra.s = 'n'
        bogus_results = do_hermaut(bogus_term, wu_common.nm_inds)


    #  Make bogus tests with non-telescopic state progression and nonunique state progression
    #  Raise NotImplementedError for non-ground state first bra and last ket
