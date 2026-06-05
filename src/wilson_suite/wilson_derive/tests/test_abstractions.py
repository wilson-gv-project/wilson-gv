import pytest

from wilson_suite.wilson_derive.abstractions import (QOperator, HarmOscStateSymbolic, VibStateSymbolic, PolProp,
                                                     VibDiffTerm, ResonanceCondition, TransitionIntegral)

def test_q_operator():

    operator_a = QOperator(1)
    operator_b = QOperator(2, 'el_dipole')
    operator_c = QOperator(3, op_type='el_quadrupole', ax = (3, 3))

    assert operator_a.o == 1
    assert operator_a.op_type == None
    assert operator_a.ax == None

    assert operator_b.o == 2
    assert operator_b.op_type == 'el_dipole'
    assert operator_b.ax == None

    assert operator_c.o == 3
    assert operator_c.op_type == 'el_quadrupole'
    assert operator_c.ax == (3,3)

    # Non-integer o label
    with pytest.raises(TypeError):
        operator_bogus = QOperator('bogus')

    # Operator type specifier not string
    with pytest.raises(TypeError):
        operator_bogus = QOperator(1, op_type = 2)

    # Axis argument not tuple
    with pytest.raises(TypeError):
        operator_bogus = QOperator(1, 'dipole', [3, 3])

    # Axis argument element not integer
    with pytest.raises(TypeError):
        operator_bogus = QOperator(1, 'dipole', (3, 'bogus'))

def test_harm_osc_state_symbolic():

    ground_state = HarmOscStateSymbolic([])
    a_state = HarmOscStateSymbolic(['a'])
    bca_state = HarmOscStateSymbolic(['b', 'c', 'a'])

    assert ground_state.q == []
    assert a_state.q == ['a']
    # Will be sorted
    assert bca_state.q == ['a', 'b', 'c']

    # Quanta as tuple
    with pytest.raises(TypeError):
        bogus_state = HarmOscStateSymbolic(('a',))

    # Non-character quantum
    with pytest.raises(TypeError):
        bogus_state = HarmOscStateSymbolic((1, 'a'))

    # String but not character quantum
    with pytest.raises(TypeError):
        bogus_state = HarmOscStateSymbolic(('bogus', 'a'))

    pass

def test_vib_state_symbolic():

    state_a = VibStateSymbolic('A')

    state_b = VibStateSymbolic('B', mbu=['A'])

    state_c = VibStateSymbolic('G', is_ground=True )

    assert state_a.s == 'A'
    assert state_a.mbu == []
    assert state_a.is_ground == False

    assert state_b.s == 'B'
    assert state_b.mbu == ['A']
    assert state_b.is_ground == False

    assert state_c.s == 'G'
    assert state_c.mbu == []
    assert state_c.is_ground == True

    # Non-hashable state label
    with pytest.raises(TypeError):
        state_bogus = VibStateSymbolic(['A'])

    # Must-be-unequal argument not list
    with pytest.raises(TypeError):
        state_bogus = VibStateSymbolic('A', mbu='B')

    # is_ground not bool
    with pytest.raises(TypeError):
        state_bogus = VibStateSymbolic('A', is_ground='bogus')

def test_vib_diff_term():

    harm_osc_a = HarmOscStateSymbolic(['a'])
    harm_osc_bc = HarmOscStateSymbolic(['b', 'c'])

    symb_state_a = VibStateSymbolic('A')
    symb_state_b = VibStateSymbolic('B')

    vd_harm = VibDiffTerm(harm_osc_a, harm_osc_bc)
    vd_symb = VibDiffTerm(symb_state_a, symb_state_b, is_pert_wf_diff=True)

    assert vd_harm.sl.q == ['a']
    assert vd_harm.sr.q == ['b', 'c']
    assert vd_harm.is_pert_wf_diff == False

    assert vd_symb.sl.s == 'A'
    assert vd_symb.sr.s == 'B'
    assert vd_symb.is_pert_wf_diff == True

    # sl and sr not same type
    with pytest.raises(TypeError):
        vd_bogus = VibDiffTerm(harm_osc_a, symb_state_b)

    # Same but bogus type
    with pytest.raises(TypeError):
        vd_bogus = VibDiffTerm('bogus_a', 'bogus_b')

def test_resonance_condition():

    # Simple example: w_a - w_0 - (w_1 - w_2)
    harm_osc_a = HarmOscStateSymbolic(['a'])
    harm_osc_0 = HarmOscStateSymbolic([])
    vd_harm_a_0 = VibDiffTerm(harm_osc_a, harm_osc_0)

    pert_freq_1m2 = [1, -2]

    res_cond_a_0 = ResonanceCondition(vd_harm_a_0, pert_freq_1m2)

    assert res_cond_a_0.diff.sl.q == ['a']
    assert res_cond_a_0.diff.sr.q == []
    assert res_cond_a_0.pf == [1, -2]
    assert res_cond_a_0.id == None

    # Overall sign for state energy level difference must be positive
    assert res_cond_a_0.netStateSign() == 1

    # Another example: w_a - w_(b + c) - (w_1 - w_2 + w_3)
    harm_osc_bc = HarmOscStateSymbolic(['b', 'c'])
    vd_harm_a_bc = VibDiffTerm(harm_osc_a, harm_osc_bc)

    pert_freq_1m23 = [1, -2, 3]

    res_cond_a_bc = ResonanceCondition(vd_harm_a_bc, pert_freq_1m23, id=2)
    assert res_cond_a_bc.diff.sr.q == ['b', 'c']
    assert res_cond_a_bc.id == 2

    # Overall state energy lvl difference sign is here indeterminate without further information
    assert res_cond_a_bc.netStateSign() == -3

    # Now making an auxiliary state difference w_0 - w_c
    harm_osc_c = HarmOscStateSymbolic(['c'])
    vd_harm_0_c = VibDiffTerm(harm_osc_0, harm_osc_c)

    # Evaluating according to this trimmed resonance means incorporating further knowledge about equivalences between
    # energy levels (i.e. using vd_harm_0_c for this (w_a - w_(b + c)) term essentially means stating that w_a == w_b
    # The netStateSign routine should now return a definite finding of negative sign given this
    assert res_cond_a_bc.netStateSign(instead_trimmed= vd_harm_0_c) == -1

    # Testing couldBeResonantWithFieldByConditions
    # Note: This routine is not complete in the sense of always detecting "could not be resonant" and may
    # therefore return True when a resonance condition may in fact not be resonant. But since its use is in
    # aiding in ruling out terms from further consideration, the consequences of this are small:
    # It will at worst lead to evaluation of unimportant terms

    # Signifying that w1 - w2 > 0
    magn_conditions_1_m_m2_gt_0 = [[1, -2]]
    assert res_cond_a_0.couldBeResonantWithFieldByConditions(magn_conditions_1_m_m2_gt_0)

    # Signifying that w2 - w1 > 0: Resonance can here be ruled out since wa - w0 is positive and -w1 + w2 is positive,
    # i.e. w1 - w2 is negative, i.e. this resonance condition will have a positive number (the energy level difference)
    # minus a negative number (minus (w1 - w2) which is negative), ergo this will always be positive and > 0 so no
    # resonance possible
    magn_conditions_m2_m_1_gt_0 = [[-1, 2]]
    assert not(res_cond_a_0.couldBeResonantWithFieldByConditions(magn_conditions_m2_m_1_gt_0))

    # More tests of "could be resonant"

    harm_osc_ab = HarmOscStateSymbolic(['a', 'b'])
    # Energy level difference always negative, so if sum of pert freqs. is definitely positive (i.e, negative when
    # subtracted), then resonance can be ruled out
    vd_harm_a_ab = VibDiffTerm(harm_osc_a, harm_osc_ab)

    pert_freq_lrg = [1, -2, 3, -4, -5, 6]

    res_cond_lrg = ResonanceCondition(vd_harm_a_ab, pert_freq_lrg)

    magn_cond_a = [[1, -2], [3, -4]]
    magn_cond_b = [[1, -2], [3, -4, -5]]
    magn_cond_c = [[1, -2], [-3, 4, 5]]
    magn_cond_d = [[1, -2], [1, 3, -4, -5]]
    magn_cond_e = [[1, -2], [3, -4, 7]]
    magn_cond_f = [[1, -2, -4, -5], [3, -5]]

    # Should return True: While w1 - w2 > 0 and w3 - w4 > 0, there is no further information about w5 - w6, and they are
    # of opposing signs: Therefore resonance cannot be ruled out
    assert res_cond_lrg.couldBeResonantWithFieldByConditions(magn_cond_a)

    # Should return False: w1 - w2 > 0 and w3 - w4 - w5 > 0, and w6 > 0 by definition:
    # Therefore energy level difference - perturbing freqs is always negative: Cannot become zero: Resonance
    # is ruled out
    assert not(res_cond_lrg.couldBeResonantWithFieldByConditions(magn_cond_b))

    # Should return True: [1, -2] will be recognized with sign as is, but [-3, 4, 5] will be recognized with opposite sign
    assert res_cond_lrg.couldBeResonantWithFieldByConditions(magn_cond_c)

    # Should return True: Magnitude conditions are overlapping (w1) and only can be applied. Whichever is (it will be
    # the first), the result for the residual will be ambiguous
    assert res_cond_lrg.couldBeResonantWithFieldByConditions(magn_cond_d)

    # Should return True: One magnitude condition contains an unrepresented reference (w7) and so cannot be applied.
    # The residual after applying the only valid condition ([1, -2]) is ambiguous.
    assert res_cond_lrg.couldBeResonantWithFieldByConditions(magn_cond_e)

    # Should return True: The magnitude conditions are both applicable but overlapping.
    # The first listed will be applied first and the second will not be because of the overlap. However, the
    # resulting residual is w3 + w6 which aligns with the positive sign after applying the magnitude condition.
    assert not(res_cond_lrg.couldBeResonantWithFieldByConditions(magn_cond_f))

    # Setting up for "all residual" test 1

    pert_freq_lrg_2 = [-1, -2, -3, -4, -5]
    vd_harm_ab_a = VibDiffTerm(harm_osc_ab, harm_osc_a)
    res_cond_lrg_2 = ResonanceCondition(vd_harm_ab_a, pert_freq_lrg_2)
    magn_cond_mt = []

    # Should here go to full residual treatment and find that resonance can be ruled out
    assert not (res_cond_lrg_2.couldBeResonantWithFieldByConditions(magn_cond_mt))

    # "All residual" test 2

    pert_freq_lrg_2 = [1, 2, 3, 4, 5]
    vd_harm_ab_a = VibDiffTerm(harm_osc_ab, harm_osc_a)
    res_cond_lrg_2 = ResonanceCondition(vd_harm_ab_a, pert_freq_lrg_2)
    magn_cond_mt = []

    # Should here go to full residual treatment and find that resonance cannot be ruled out
    assert res_cond_lrg_2.couldBeResonantWithFieldByConditions(magn_cond_mt)

    # "All residual" test 3

    pert_freq_lrg_2 = [-1, -2, -3, 4, -5]
    vd_harm_ab_a = VibDiffTerm(harm_osc_ab, harm_osc_a)
    res_cond_lrg_2 = ResonanceCondition(vd_harm_ab_a, pert_freq_lrg_2)
    # Note present but empty resonance condition also considered valid input (will simply not be treated)
    magn_cond_mt = [[]]

    # Resonance cannot be ruled out (w4 opposite sign)
    assert res_cond_lrg_2.couldBeResonantWithFieldByConditions(magn_cond_mt)

    # Now testing application of previous resonances:

    # With magnitude conditions and residual freqs

    harm_osc_a = HarmOscStateSymbolic(['a'])
    harm_osc_c = HarmOscStateSymbolic(['c'])
    vd_harm_a_c = VibDiffTerm(harm_osc_a, harm_osc_c)
    pert_freq_1m3 = [1, -3]
    given_res_cond_a_c = ResonanceCondition(vd_harm_a_c, pert_freq_1m3)

    harm_osc_ab = HarmOscStateSymbolic(['a', 'b'])
    vd_harm_ab_c = VibDiffTerm(harm_osc_ab, harm_osc_c)
    pert_freq_1m2m34 = [1, -2, -3, 4]
    res_cond_ab_c = ResonanceCondition(vd_harm_ab_c, pert_freq_1m2m34)

    magn_conditions_m2_3_gt_0 = [[2, -4]]

    # Resonance can here be ruled out given that given_res_cond_a_c was resonant (but not otherwise with the present
    # magnitude conditions)
    assert not(res_cond_ab_c.couldBeResonantWithFieldByConditions(magn_conditions_m2_3_gt_0, given_res_cond_a_c))

    # With only residual freqs A: Resonance cannot be ruled out
    assert res_cond_ab_c.couldBeResonantWithFieldByConditions([[]], given_prev_res=given_res_cond_a_c)

    # With only residual freqs A: Resonance can be ruled out with modified pert freq argument
    pert_freq_12m34 = [1, -2, -3, -4]
    res_cond_ab_c = ResonanceCondition(vd_harm_ab_c, pert_freq_12m34)
    assert not(res_cond_ab_c.couldBeResonantWithFieldByConditions([[]],given_prev_res=given_res_cond_a_c))

    # Non-matching previous resonance
    harm_osc_b = HarmOscStateSymbolic(['b'])
    vd_harm_ab_b = VibDiffTerm(harm_osc_ab, harm_osc_b)
    pert_freq_1m2m34 = [1, -2, -3, 4]
    res_cond_ab_b = ResonanceCondition(vd_harm_ab_b, pert_freq_1m2m34)

    magn_conditions_m2_3_gt_0 = [[2, -4]]

    # Resonance can here not be ruled out but would be if the state difference had been w_(a + b) - w_c and not
    # (as used here) w_(a + b) - w_b
    assert res_cond_ab_b.couldBeResonantWithFieldByConditions(magn_conditions_m2_3_gt_0, given_res_cond_a_c)

    # Type-based early return (argument(s) not correct type(s) so returns True (cannot rule out resonance))

    # Perturbing freqs not signed pulse references
    pert_freq_a_mb = ['A', '-B']
    res_cond_a_0_axes = ResonanceCondition(vd_harm_a_0, pert_freq_a_mb)
    magn_conditions_m2_m_1_gt_0 = [[-1, 2]]
    assert res_cond_a_0_axes.couldBeResonantWithFieldByConditions(magn_conditions_m2_m_1_gt_0)

    # Magnitude conditions not list of lists of signed pulse references
    magn_conditions_dev_form = [['A', 2]]
    assert res_cond_a_0.couldBeResonantWithFieldByConditions(magn_conditions_dev_form)

    magn_conditions_dev_form = [-1, 2]
    assert res_cond_a_0.couldBeResonantWithFieldByConditions(magn_conditions_dev_form)

    # Previous resonance argument not ResonanceCondition
    assert res_cond_a_0.couldBeResonantWithFieldByConditions(magn_conditions_m2_m_1_gt_0, given_prev_res='deviating_form')

    # Extra test: Not HarmOscStateSymbolic states in VibDiffTerm
    # This is allowed but precludes the use of some methods
    vibstate_symb_a = VibStateSymbolic('A')
    vibstate_symb_b = VibStateSymbolic('B')
    vd_harm_a_0 = VibDiffTerm(vibstate_symb_a, vibstate_symb_b)
    pert_freq_1m2 = [1, -2]

    res_cond_a_0_symb = ResonanceCondition(vd_harm_a_0, pert_freq_1m2)

    assert res_cond_a_0_symb.diff.sl.s == 'A'
    assert res_cond_a_0_symb.diff.sr.s == 'B'

    with pytest.raises(TypeError):
        bogus = res_cond_a_0_symb.netStateSign()

    with pytest.raises(TypeError):
        magn_conditions_m2_m_1_gt_0 = [[-1, 2]]
        bogus = res_cond_a_0_symb.couldBeResonantWithFieldByConditions(magn_conditions_m2_m_1_gt_0)

    # Bogus:

    # Not VibDiffTerm
    with pytest.raises(TypeError):
        harm_osc_a = HarmOscStateSymbolic(['a'])
        harm_osc_0 = HarmOscStateSymbolic([])
        vd_harm_a_0_bogus = [harm_osc_a, harm_osc_0]
        pert_freq_1m2 = [1, -2]

        res_cond_a_0_bogus = ResonanceCondition(vd_harm_a_0_bogus, pert_freq_1m2)

    # Frequency labels not strings or integers

    # Identifier not integer
    with pytest.raises(TypeError):
        harm_osc_a = HarmOscStateSymbolic(['a'])
        harm_osc_0 = HarmOscStateSymbolic([])
        vd_harm_a_bc = VibDiffTerm(harm_osc_a, harm_osc_0)
        pert_freq_1m23 = [1, -2, 3]

        res_cond_a_0_bogus = ResonanceCondition(vd_harm_a_bc, pert_freq_1m23, id='2')

    # Not yet testing method couldBeResonantWithFieldByRanges (not finished and not currently used)

def test_line_shape():

    # Functionality not currently used, deferring tests

    pass

def test_pol_prop():

    operator_a = QOperator(2)
    operator_b = QOperator(3)
    operator_c = QOperator(5)

    operators = [operator_a, operator_b, operator_c]

    pol_prop_a = PolProp(operators)

    assert pol_prop_a.ops[0].o == 2
    assert pol_prop_a.ops[1].o == 3
    assert pol_prop_a.ops[2].o == 5
    assert pol_prop_a.dord == 0

    pol_prop_a = PolProp(operators, dord=3)
    assert pol_prop_a.dord == 3

    pol_prop_a.setDerivOrder(2)
    assert pol_prop_a.dord == 2

    # Testing setInds

    pol_prop_a.setInds(['b', 'a'])
    # Sorted at set
    assert pol_prop_a.inds == ['a', 'b']

    with pytest.raises(ValueError):
        pol_prop_a.setInds(['b', 'a', 'c'])

    # Testing epochContained
    epochs_a = [[1, 2, 3, 4, 5], [6]]
    epochs_b = [[1, 2, 3, 4], [5]]
    epochs_c = [[1], [4, 2, 3]]
    epochs_d = [[1], [2, 3], [4]]

    assert pol_prop_a.epochContained(epochs_a, op_ind_omega = 7)
    assert not(pol_prop_a.epochContained(epochs_b, op_ind_omega=6))
    assert pol_prop_a.epochContained(epochs_c, op_ind_omega = 5)
    assert not(pol_prop_a.epochContained(epochs_d, op_ind_omega=5))

    # Type errors:

    # Non-list operators argument
    with pytest.raises(TypeError):
        pol_prop_a = PolProp(tuple(operators))

    # Non-QOperator instances in operator list
    with pytest.raises(TypeError):
        pol_prop_a = PolProp([operator_a, 'bogus', operator_c])

    # Non-integer differentiation order
    with pytest.raises(TypeError):
        pol_prop_a = PolProp(operators, dord='bogus')

    with pytest.raises(TypeError):
        pol_prop_a = PolProp(operators)
        pol_prop_a.setDerivOrder('bogus')

    # setInds: inds not list
    with pytest.raises(TypeError):
        pol_prop_a = PolProp(operators)
        pol_prop_a.setDerivOrder(5)
        pol_prop_a.setInds('bogus')

    # setInds: inds not list of single-char strings
    with pytest.raises(TypeError):
        pol_prop_a = PolProp(operators)
        pol_prop_a.setDerivOrder(3)
        pol_prop_a.setInds(['a', 'bogus', 'c'])

    # epochContained: malformed epochs
    with pytest.raises(TypeError):
        pol_prop_a = PolProp(operators)
        epochs_bogus = ([1, 2, 3, 4, 5], [6])
        bogus = pol_prop_a.epochContained(epochs_bogus, op_ind_omega=7)

    with pytest.raises(TypeError):
        pol_prop_a = PolProp(operators)
        epochs_bogus = [(1, 2, 3, 4, 5), [6]]
        bogus = pol_prop_a.epochContained(epochs_bogus, op_ind_omega=7)

    with pytest.raises(TypeError):
        pol_prop_a = PolProp(operators)
        epochs_bogus = [[1, 2, 3, 'bogus', 5], [6]]
        bogus = pol_prop_a.epochContained(epochs_bogus, op_ind_omega=7)

    # epochContained: op_ind_omega wrong type
    with pytest.raises(TypeError):
        pol_prop_a = PolProp(operators)
        epochs_bogus = [[1, 2, 3, 4, 5], [6]]
        bogus = pol_prop_a.epochContained(epochs_bogus, op_ind_omega='bogus')

def test_pol_prop_sos_recursion():

    operator_a = QOperator(2)
    operator_b = QOperator(3)

    state_a = VibStateSymbolic('A')
    state_b = VibStateSymbolic('B', mbu=['A'])

    # FIXME: Return to this test once relevant attributes/methods settled during code pass/unit test work on
    # vib_rsp_sos
    pass

def test_transition_integral():

    state_a = VibStateSymbolic('A')
    state_b = VibStateSymbolic('B', mbu=['A'])

    operator_a = QOperator(2)
    operator_b = QOperator(3)
    operator_c = QOperator(5)
    operators = [operator_a, operator_b, operator_c]
    pol_prop_a = PolProp(operators, dord=3)

    t_int = TransitionIntegral(state_a, state_b, pol_prop_a)

    # Just a few spot checks since this is verified in separate test
    assert t_int.prop.ops[1].o == 3
    assert t_int.prop.dord == 3
    assert t_int.bra.s == 'A'
    assert t_int.ket.mbu == ['A']

    t_int.setBra(state_b)
    assert t_int.bra.s == 'B'

    t_int.setKet(state_a)
    assert t_int.ket.s == 'A'

    # Bogus

    # bra argument not VibStateSymbolic
    with pytest.raises(TypeError):
        t_int = TransitionIntegral('bogus', state_b, pol_prop_a)

    # ket argument not VibStateSymbolic
    with pytest.raises(TypeError):
        t_int = TransitionIntegral(state_a, 'bogus', pol_prop_a)

    # prop argument not PolProp
    with pytest.raises(TypeError):
        t_int = TransitionIntegral(state_a, state_b, 'bogus')

    # set_bra bra argument not VibStateSymbolic
    with pytest.raises(TypeError):
        t_int = TransitionIntegral(state_a, state_b, pol_prop_a)
        t_int.setBra('bogus')

    # set_ket ket argument not VibStateSymbolic
    with pytest.raises(TypeError):
        t_int = TransitionIntegral(state_a, state_b, pol_prop_a)
        t_int.setKet('bogus')