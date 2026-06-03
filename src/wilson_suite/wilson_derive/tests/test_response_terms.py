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

    # 1/w_(0,a)^[-1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == [nm_index_mask['a']]
    assert t.res[0].pf == [-1]

    # 1/w_(b,a)^[-1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == [nm_index_mask['b']]
    assert t.res[1].diff.sr.q == [nm_index_mask['a']]
    assert t.res[1].pf == [-1, 2]

    return True

def test_vib_perturbed_term():

    import copy
    from fractions import Fraction
    from wilson_suite.fixtures import evv_experiment
    from wilson_suite.wilson_derive.derive import get_fully_enhanced_terms
    from wilson_suite.wilson_utils import common_labels as wu_common

    evv_exp = evv_experiment()
    terms = get_fully_enhanced_terms(experiment=evv_exp)

    # Constructor type checking
    # Known term from mech. anharm. part
    td = terms[1][(0, 1)][2]
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



    # TODO:
    #  verify and possibly amend above init checks DONE
    #  nmRenameAndInternalResort testing DONE
    #  sort testing DONE
    #  more full_enhancement_possible testing

    # Testing full_enhancement_possible
    # Is mostly a wrapper for couldBeResonantWithFieldByConditions (tested separately), so not much further testing
    # needed here

    print('magn conds', evv_exp.magn_conditions)

    assert evv_exp.magn_conditions == ((-1, 2),)

    # Same test term as before
    t = terms[1][(0, 1)][2]

    t.present()

    # No magnitude conditions, should here not lead to any restriction
    assert t.full_enhancement_possible()

    # EVV magnitude conditions, should here not lead to any restriction
    assert t.full_enhancement_possible(magn_conditions=evv_exp.magn_conditions)

    # Cannot fulfill first resonance condition
    assert not(t.full_enhancement_possible(magn_conditions=((-1,),)))

    # Can fulfill second resonance condition (direct finding from magnitude condition alone)
    assert t.full_enhancement_possible(magn_conditions=((1, -2),))

    # Finding follows from both magnitude condition and previous resonance
    assert t.full_enhancement_possible(magn_conditions=((2),))

    # FIXME
    # CONTINUE HERE: It should be possible to determine non-fulfillment here: Try to improve couldBeResonant... for this
    # Cannot fulfill second resonance condition (finding must follow from both magnitude condition and previous resonance)
    assert not (t.full_enhancement_possible(magn_conditions=((-2),)))



    # New test term ("a+b, a" term from el. anharm.)
    t = terms[1][(1, 0)][1]

    t.present()

    # EVV magnitude conditions, should here not lead to any restriction
    assert t.full_enhancement_possible(magn_conditions=evv_exp.magn_conditions)

    # Magnitude conditions not applicable to these resonance conditions
    assert t.full_enhancement_possible(magn_conditions=((3, -4), (5, -7, 6)))

    # Cannot fulfill second resonance condition (direct finding from magnitude condition alone)
    assert not(t.full_enhancement_possible(magn_conditions=((1, -2),)))

    # Finding follows from both magnitude condition and previous resonance
    assert t.full_enhancement_possible(magn_conditions=((2),))

    # FIXME (related to l284 FIXME)
    # Cannot fulfill second resonance condition (finding must follow from both magnitude condition and previous resonance)
    assert not (t.full_enhancement_possible(magn_conditions=((-2),)))



def test_vib_contrib_term():

    import copy
    from wilson_suite.fixtures import evv_experiment
    from wilson_suite.wilson_derive.vib_rsp_sos import get_vib_sos
    from wilson_suite.wilson_derive.abstractions import TransitionIntegral, PolProp, VibStateSymbolic, ResonanceCondition, QOperator, VibDiffTerm

    evv_exp = evv_experiment()
    R_sos = get_vib_sos(evv_exp.order)

    # Taking an EVV relevant term
    t_choice = R_sos[21]

    # Constructor type checking
    with pytest.raises(TypeError):
        VibContribTerm(1, t_choice.ints, t_choice.res)  # coeff not Fraction
    with pytest.raises(TypeError):
        VibContribTerm(t_choice.coeff, [object()], t_choice.res)  # bad ints type
    with pytest.raises(TypeError):
        VibContribTerm(t_choice.coeff, t_choice.ints, [object()])  # bad res type

    # Taking the EVV term and re-initializing a VibContribTerm
    t = VibContribTerm(t_choice.coeff, t_choice.ints, t_choice.res)

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

    # Testing dressWithPulseInteractions
    td = copy.deepcopy(t)

    assert evv_exp.int_sequences == [({1: -1}, {2: 1}, {3: 1})]

    td.dressWithPulseInteractions(evv_exp.int_sequences[0])

    assert [k.o for k in td.ints[0].prop.ops] == [1]
    assert [k.o for k in td.ints[1].prop.ops] == [0, 3]
    assert [k.o for k in td.ints[2].prop.ops] == [2]

    assert td.res[0].pf == [-1]
    assert td.res[1].pf == [-1, 2]

    td = copy.deepcopy(t)
    td.dressWithPulseInteractions(({3: 1}, {2: -1}, {1: 1}))

    assert [k.o for k in td.ints[0].prop.ops] == [3]
    assert [k.o for k in td.ints[1].prop.ops] == [0, 1]
    assert [k.o for k in td.ints[2].prop.ops] == [2]

    assert td.res[0].pf == [3]
    assert td.res[1].pf == [3, -2]

    # Now testing allUVCancels and allElRspEpochContained

    # This is a CARS term, chosen for this testing since it has two extracted el. responses
    td = R_sos[3]
    # Dress according to -k1 + k2 + k3
    td.dressWithPulseInteractions(evv_exp.int_sequences[0])

    assert evv_exp.epochs == [[1], [2], [3]]
    assert evv_exp.cfuv == {1: 0.0, 2: 0.0, 3: 0.072}

    assert not(td.allElRspEpochContained(evv_exp.epochs, op_ind_omega = 0))
    assert not(td.allElRspEpochContained([[1], [2, 3]], op_ind_omega=0))
    assert not(td.allElRspEpochContained([[1, 3], [2]], op_ind_omega=0))
    assert td.allElRspEpochContained([[1, 2], [3]], op_ind_omega=0)
    assert not(td.allElRspEpochContained([[1, 0], [3]], op_ind_omega=2))
    assert td.allElRspEpochContained([[1, 2, 3]], op_ind_omega=0)

    assert td.allUVCancels(evv_exp.cfuv)
    assert td.allUVCancels({1: 0.0, 2: 0.0, 3: 0.0})
    assert not(td.allUVCancels({1: 0.072, 2: 0.0, 3: 0.032}))
    assert not (td.allUVCancels({1: 0.072, 2: 0.032, 3: 0.032}))
    assert td.allUVCancels({1: 0.072, 2: 0.072, 3: 0.032})
    assert td.allUVCancels({1: 0.072, 2: 0.072, 3: 0.072})

    # Extra epoch containment and UV cancellation testing

    td = R_sos[21]
    td.dressWithPulseInteractions(evv_exp.int_sequences[0])

    assert td.allElRspEpochContained(evv_exp.epochs, op_ind_omega=0)
    assert td.allElRspEpochContained([[1], [2, 0]], op_ind_omega=3)
    assert td.allElRspEpochContained([[1], [2], [0]], op_ind_omega=3)
    assert td.allElRspEpochContained([[1, 2], [3]], op_ind_omega=0)
    assert not(td.allElRspEpochContained([[1], [0], [2]], op_ind_omega=3))
    assert not(td.allElRspEpochContained([[1, 3], [2]], op_ind_omega=0))

    assert td.allUVCancels(evv_exp.cfuv)

