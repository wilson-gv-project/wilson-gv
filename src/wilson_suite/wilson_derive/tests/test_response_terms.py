from fractions import Fraction

import pytest

from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm, VibContribTerm

# Helper function: Compare to selected template term (terms[1][(0, 1)][2] for EVV experiment)
# Optional normal mode index mask for comparisons after index switches
def _assert_term_is_template(t, nm_index_mask = {'a': 'a', 'b': 'b', 'c': 'c'}):

    assert t.coeff == Fraction(1, 16)
    assert len(t.props) == 4
    assert len(t.freqterms) == 4
    assert len(t.res) == 2

    assert t.coeff == 1 / 16

    # 1/w_a
    assert not (t.freqterms[0].is_pert_wf_diff)
    assert t.freqterms[0].sl.q == [nm_index_mask['a']]
    assert t.freqterms[0].sr.q == []

    # 1/w_b
    assert not (t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == [nm_index_mask['b']]
    assert t.freqterms[1].sr.q == []

    # 1/w_c
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == [nm_index_mask['c']]
    assert t.freqterms[2].sr.q == []

    # 1/(w_ab - w_a)  (pert WF frequency difference)
    assert t.freqterms[3].is_pert_wf_diff
    assert t.freqterms[3].sl.q == sorted([nm_index_mask['a'], nm_index_mask['b']])
    assert t.freqterms[3].sr.q == [nm_index_mask['a']]

    # d alpha_{alpha delta}/ d Q_a
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == [nm_index_mask['a']]

    # d mu_beta/ d Q_a
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 1
    assert t.props[1].dord == 1
    assert t.props[1].inds == [nm_index_mask['a']]

    # d mu_gamma/ d Q_b
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 1
    assert t.props[2].inds == [nm_index_mask['b']]

    # F_bcc
    assert len(t.props[3].ops) == 0
    assert t.props[3].dord == 3
    assert t.props[3].inds == sorted([nm_index_mask['b'], nm_index_mask['c'], nm_index_mask['c']])

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == [nm_index_mask['a']]
    assert t.res[0].pf == [1]

    # 1/w_(b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == [nm_index_mask['b']]
    assert t.res[1].diff.sr.q == [nm_index_mask['a']]
    assert t.res[1].pf == [1, 2]

    return True

def test_vib_perturbed_term():

    import copy
    from fractions import Fraction
    from wilson_suite.wilson_utils import common_labels as wu_common
    from wilson_suite.wilson_derive.derive import get_terms_by_comp_pattern_and_magn_conds

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

    # Constructor type checking
    # Known term from mech. anharm. part
    td = terms[4]

    with pytest.raises(TypeError):
        VibPerturbedTerm(1, td.props, td.freqterms, td.res)  # coeff not Fraction
    with pytest.raises(TypeError):
        VibPerturbedTerm(td.coeff, [object()], td.freqterms, td.res)  # bad prop type
    with pytest.raises(TypeError):
        VibPerturbedTerm(td.coeff, td.props, [object()], td.res)  # bad freqterm type
    with pytest.raises(TypeError):
        VibPerturbedTerm(td.coeff, td.props, td.freqterms, [object()])  # bad res type

    # Making new instance and checking (known) attributes
    t = VibPerturbedTerm(td.coeff, td.props, td.freqterms, td.res)

    assert _assert_term_is_template(t)

    # h() requires was_sorted=True first
    assert not(t.was_sorted)
    with pytest.raises(AssertionError):
        VibPerturbedTerm(t.coeff, t.props, t.freqterms, t.res).h()

    # Testing renaming and internal index tuple sorting
    tc = copy.deepcopy(t)
    # Here sending 'a' to original 'b', sending 'b' to original 'a', unchanged 'c'
    sort_mask = {'b': 'a', 'a': 'b', 'c': 'c'}
    tc.nmRenameAndInternalResort(sort_mask)
    assert _assert_term_is_template(tc, nm_index_mask=sort_mask)

    tc = copy.deepcopy(t)
    # Here sending 'c' to original 'a', sending 'a' to original 'b', sending 'b' to original 'c'
    sort_mask = {'a': 'c', 'b': 'a', 'c': 'b'}
    tc.nmRenameAndInternalResort(sort_mask)
    assert _assert_term_is_template(tc, nm_index_mask=sort_mask)

    # Testing sorting
    tc = copy.deepcopy(t)

    # Resonance conditions not in order of num pert freq
    tc.res = [copy.deepcopy(tc.res[1]), copy.deepcopy(tc.res[0])]
    tc.sort(wu_common.nm_inds)
    assert _assert_term_is_template(tc)

    # Normal mode indices not canonically named
    sort_mask = {'b': 'a', 'a': 'b', 'c': 'c'}
    tc.nmRenameAndInternalResort(sort_mask)
    tc.sort(wu_common.nm_inds)
    assert _assert_term_is_template(tc)

    # Operator reference out of numerolexical order
    tc.props[0].ops[0].o = 3
    tc.props[0].ops[1].o = 0

    # Properties not in order of geo diff. order
    # Properties at same geo diff. order not in numerolexical order
    tc.props = [copy.deepcopy(tc.props[2]), copy.deepcopy(tc.props[3]), copy.deepcopy(tc.props[0]), copy.deepcopy(tc.props[1])]

    tc.sort(wu_common.nm_inds)
    assert _assert_term_is_template(tc)

    # Criterion "FD 1.a)" Freq diff terms (internally) not greatest number of quanta in left-hand state
    # "FD 1.b)": Tied terms wrt. previous criterion not in lexical order of state tuples (tested further below)

    tc.freqterms[0].sl.q = []
    tc.freqterms[0].sr.q = ['a']

    tc.freqterms[2].sl.q = []
    tc.freqterms[2].sr.q = ['c']

    tc.freqterms[3].sl.q = ['a']
    tc.freqterms[3].sr.q = ['a', 'b']

    # Three sign-changing flips, so net sign change
    tc.coeff *= Fraction(-1)

    tc.sort(wu_common.nm_inds)
    assert _assert_term_is_template(tc)

    # "FD 2.a)": Freq diff terms (wrt. each other) not in lexical order of right-hand state tuples
    # "FD 2.b)": Tied terms wrt. previous criterion not in lexical order of left-hand state tuples (tested further below)
    tc.freqterms = [copy.deepcopy(tc.freqterms[2]), copy.deepcopy(tc.freqterms[3]), copy.deepcopy(tc.freqterms[0]),
                   copy.deepcopy(tc.freqterms[1])]

    tc.sort(wu_common.nm_inds)
    assert _assert_term_is_template(tc)

    # Combination of several of the previous disorderings
    tc.res = [copy.deepcopy(tc.res[1]), copy.deepcopy(tc.res[0])]

    tc.props[0].ops[0].o = 3
    tc.props[0].ops[1].o = 0
    tc.props = [copy.deepcopy(tc.props[2]), copy.deepcopy(tc.props[3]), copy.deepcopy(tc.props[0]),
                copy.deepcopy(tc.props[1])]

    tc.freqterms[0].sl.q = []
    tc.freqterms[0].sr.q = ['a']

    tc.freqterms[2].sl.q = []
    tc.freqterms[2].sr.q = ['c']

    tc.freqterms[3].sl.q = ['a']
    tc.freqterms[3].sr.q = ['a', 'b']

    # Three sign-changing flips, so net sign change
    tc.coeff *= Fraction(-1)

    tc.freqterms = [copy.deepcopy(tc.freqterms[2]), copy.deepcopy(tc.freqterms[3]), copy.deepcopy(tc.freqterms[0]),
                   copy.deepcopy(tc.freqterms[1])]

    tc.sort(wu_common.nm_inds)
    assert _assert_term_is_template(tc)

    # Comparing to original term hash, the new term's hash after being sorted back to
    # the original term should be unchanged
    assert td.h() == tc.h()

    # This test includes application of FD 1.b) and 2.b)

    # Copied for alterations
    t_special = copy.deepcopy(tc)

    t_special.freqterms[0].sl.q = ['a', 'c']
    t_special.freqterms[0].sr.q = ['a', 'c']

    # Will be flipped by FD 1.b)
    t_special.freqterms[1].sl.q = ['b']
    t_special.freqterms[1].sr.q = ['a']

    # Will be flipped by FD 1.a)
    t_special.freqterms[2].sl.q = []
    t_special.freqterms[2].sr.q = ['a']

    # Will be flipped by FD 1.b)
    t_special.freqterms[3].sl.q = ['a', 'c']
    t_special.freqterms[3].sr.q = ['a', 'b']

    coeff_sav = copy.deepcopy(t_special.coeff)

    # Result after FD 1.a)/b): [['a', 'c'], ['a', 'c']], [['a'], ['b']], [['a'], []], [['a','b'], ['a', 'c']]
    # Result after FD 2.a): [[['a'], []], [['a', 'c'], ['a', 'c']], [['a','b'], ['a', 'c']], [['a'], ['b']]] or
    #                       [[['a'], []], [['a', 'b'], ['a', 'c']], [['a','c'], ['a', 'c']], [['a'], ['b']]]
    # (sorting-algorithm-dependent at that point, in practice the first example was observed)
    # Result after FD 2.b): [[['a'], []], [['a', 'b'], ['a', 'c']], [['a','c'], ['a', 'c']], [['a'], ['b']]]

    t_special.sort(wu_common.nm_inds)

    # Note: The above alterations should lead to three sign-changing flips, so net sign change
    assert t_special.coeff == -1 *  coeff_sav

    assert t_special.freqterms[0].sl.q == ['a']
    assert t_special.freqterms[0].sr.q == []

    assert t_special.freqterms[1].sl.q == ['a', 'b']
    assert t_special.freqterms[1].sr.q == ['a', 'c']

    assert t_special.freqterms[2].sl.q == ['a', 'c']
    assert t_special.freqterms[2].sr.q == ['a', 'c']

    assert t_special.freqterms[3].sl.q == ['a']
    assert t_special.freqterms[3].sr.q == ['b']

    # Testing full_enhancement_possible
    # Is mostly a wrapper for couldBeResonantWithFieldByConditions (tested separately), so not much further testing
    # needed here

    # Same test term as before
    t = terms[4]

    mc_plus_int = ([2], [1, 2],)
    mc_minus_int = ([1],)

    # No magnitude conditions, should here not lead to any restriction
    assert t.full_enhancement_possible(mcp = (), mcm = ())

    # EVV magnitude conditions, should here not lead to any restriction
    assert t.full_enhancement_possible(mcp = mc_plus_int, mcm = mc_minus_int)

    # Cannot fulfill first resonance condition
    assert not(t.full_enhancement_possible(mcp = mc_minus_int, mcm = ()))

    # Can fulfill second resonance condition (direct finding from magnitude condition alone)
    assert t.full_enhancement_possible(mcp = (), mcm = mc_minus_int)

    # New test term ("a+b, a" term from el. anharm.)
    t = terms[3]

    # EVV magnitude conditions, should here not lead to any restriction
    assert t.full_enhancement_possible(mcp = mc_plus_int, mcm = mc_minus_int)

    # Magnitude conditions not applicable to these resonance conditions
    assert t.full_enhancement_possible(mcp=([3, 4],), mcm=([5, 7, 6],))


    # Anharm orders
    assert t.order_mech == 1
    assert t.order_el == 0

    # Summation and non-summation indices
    assert t.summ_ind == ['c']
    assert t.nonsumm_ind == ['a', 'b']

    # From different test, to be updated: Latex representation

    latex_str = t.to_latex()

    assert latex_str == ('\\frac{1}{8}' +
                        '\\frac{1}{\\omega_{a,}\\omega_{b,}\\omega_{c,}\\omega_{b+c,a}}' +
                        '\\frac{\\partial\\alpha_{\\alpha\\delta}}{\\partial Q_{c}}' +
                        '\\frac{\\partial\\mu_{\\beta}}{\\partial Q_{a}}' +
                        '\\frac{\\partial\\mu_{\\gamma}}{\\partial Q_{b}}' +
                        '\\frac{\\partial^{3}E_{}}{\\partial Q_{a}\\partial Q_{b}\\partial Q_{c}}' +
                        '\\frac{1}{(\\omega_{,a} -1)(\\omega_{b,a} -1-2)}')

def test_vib_contrib_term():

    import copy
    from wilson_suite.wilson_derive.vib_rsp_sos import get_vib_sos
    from wilson_suite.wilson_derive.abstractions import VibStateSymbolic, VibDiffTerm

    R_sos = get_vib_sos(3)

    # Taking an EVV relevant term
    t_choice = R_sos[21]

    # Constructor type checking
    with pytest.raises(TypeError):
        VibContribTerm(1, t_choice.ints, t_choice.res, (1, 1, 2))  # coeff not Fraction
    with pytest.raises(TypeError):
        VibContribTerm(t_choice.coeff, [object()], t_choice.res, (1, 1, 2))  # bad ints type
    with pytest.raises(TypeError):
        VibContribTerm(t_choice.coeff, t_choice.ints, [object()], (1, 1, 2))  # bad res type

    # Taking the EVV term and re-initializing a VibContribTerm
    t = VibContribTerm(t_choice.coeff, t_choice.ints, t_choice.res, comp_pattern=(1, 1, 2))

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
    assert len(t.freqdiff) == 0

    assert t.comp_pattern == (1, 1, 2)

    # Testing addFreqTerm
    td = copy.deepcopy(t)
    assert len(td.freqdiff) == 0

    symb_state_a = VibStateSymbolic('n')
    symb_state_b = VibStateSymbolic('p')
    td.addFreqTerm(VibDiffTerm(symb_state_a, symb_state_b, is_pert_wf_diff=True))

    assert len(td.freqdiff) == 1
    assert td.freqdiff[0].sl.s == 'n'
    assert td.freqdiff[0].sr.s == 'p'
    assert td.freqdiff[0].is_pert_wf_diff == True


