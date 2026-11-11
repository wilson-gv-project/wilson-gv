import pytest

from wilson_suite.wilson_derive.abstractions import (QOperator, HarmOscStateSymbolic, VibStateSymbolic, PolProp,
                                                     VibDiffTerm, ResonanceCondition, TransitionIntegral)

def test_q_operator():

    operator_a = QOperator(1)
    operator_b = QOperator(2, 'el_dipole')
    operator_c = QOperator(3, op_type='el_quadrupole', ax = (3, 3))

    assert operator_a.o == 1
    assert operator_a.op_type is None
    assert operator_a.ax is None

    assert operator_b.o == 2
    assert operator_b.op_type == 'el_dipole'
    assert operator_b.ax is None

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

    # Quanta malformed type which is not tuple or list
    with pytest.raises(TypeError):
        bogus_state = HarmOscStateSymbolic({('a',): 'a'})

    # Non-character quantum
    with pytest.raises(TypeError):
        bogus_state = HarmOscStateSymbolic((1, 'a'))

    # String but not character quantum
    with pytest.raises(TypeError):
        bogus_state = HarmOscStateSymbolic(('bogus', 'a'))

    pass

def test_vib_state_symbolic():

    state_a = VibStateSymbolic('A')
    state_b = VibStateSymbolic('B', mbu=[state_a])
    state_c = VibStateSymbolic('G', is_ground=True )

    ground_state = HarmOscStateSymbolic([])
    a_state = HarmOscStateSymbolic(['a'])
    abc_state = HarmOscStateSymbolic(['b', 'c', 'a'])
    bca_state = HarmOscStateSymbolic(['b', 'c', 'a'])

    assert state_a.s == 'A'
    assert state_a.mbu == []
    assert state_a.is_ground == False

    states_as_quanta = {'0': ground_state, 'A': a_state, 'B': abc_state}

    # Trivially fulfilled, no mbu requirement
    assert state_a.mbuFulfilled(states_as_quanta)

    assert state_b.s == 'B'
    assert len(state_b.mbu) == 1
    assert state_b.mbu[0].s == 'A'
    assert state_b.is_ground == False

    # Fulfilled, A's quanta are not == B's quanta
    states_as_quanta = {'0': ground_state, 'A': a_state, 'B': abc_state}
    assert state_b.mbuFulfilled(states_as_quanta)

    # Not fulfilled, A's quanta are (sorted and) == B's quanta
    states_as_quanta = {'0': ground_state, 'A': bca_state, 'B': abc_state}
    assert not(state_b.mbuFulfilled(states_as_quanta))

    # Bogus: Own state not in states_as_quanta
    with pytest.raises(AssertionError):
        states_as_quanta = {'0': ground_state, 'A': bca_state, 'bogus': abc_state}
        bogus = state_b.mbuFulfilled(states_as_quanta)

    # Bogus: MBU marked state not in states_as_quanta
    with pytest.raises(AssertionError):
        states_as_quanta = {'0': ground_state, 'boguz': bca_state, 'B': abc_state}
        bogus = state_b.mbuFulfilled(states_as_quanta)

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

    pert_freq_1m2 = [1, 2]

    res_cond_a_0 = ResonanceCondition(vd_harm_a_0, pert_freq_1m2)

    assert res_cond_a_0.diff.sl.q == ['a']
    assert res_cond_a_0.diff.sr.q == []
    assert res_cond_a_0.pf == [1, 2]
    assert res_cond_a_0.id is None

    # Overall sign for state energy level difference must be positive
    assert res_cond_a_0.netStateSign() == 1

    # Another example: w_a - w_(b + c) - (w_1 + w_2 + w_3)
    harm_osc_bc = HarmOscStateSymbolic(['b', 'c'])
    vd_harm_a_bc = VibDiffTerm(harm_osc_a, harm_osc_bc)

    pert_freq_123 = [1, 2, 3]

    res_cond_a_bc = ResonanceCondition(vd_harm_a_bc, pert_freq_123, id=2)
    assert res_cond_a_bc.diff.sr.q == ['b', 'c']
    assert res_cond_a_bc.id == 2

    latex_rescond = res_cond_a_bc.to_latex()
    assert latex_rescond == '(\\omega_{a,b+c} -1-2-3)'

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

    mc_12 = ([1, 2],)

    # Case: w2 + w1 < 0: Resonance can here be ruled out since wa - w0 is positive and w1 + w2 < 0,
    assert not res_cond_a_0.could_be_resonant_by_conditions(mc_plus=(), mc_minus=mc_12)

    # Case: w1 + w2 > 0: Resonance possible
    assert res_cond_a_0.could_be_resonant_by_conditions(mc_plus=mc_12, mc_minus=())

    # More tests of "could be resonant"

    harm_osc_ab = HarmOscStateSymbolic(['a', 'b'])
    # Energy level difference always negative, so if sum of pert freqs. is definitely positive (i.e, negative when
    # subtracted), then resonance can be ruled out
    vd_harm_a_ab = VibDiffTerm(harm_osc_a, harm_osc_ab)

    pert_freq_lrg = [1, 2, 3, 4, 5, 6]

    res_cond_lrg = ResonanceCondition(vd_harm_a_ab, pert_freq_lrg)

    assert res_cond_lrg.netStateSign() == -1

    magn_cond_a = ([1, 2], [3, 4])
    magn_cond_b = ([1, 2], [1, 2, 3, 4, 5, 6])
    magn_cond_c = ([3, 4, 5, 6],)
    magn_cond_d = ([1, 2], [3, 4, 5, 6])

    # Should return True: While w1 + w2 > 0 and w3 + w4 > 0, there is no further information about w5 + w6, therefore
    # resonance cannot be ruled out
    assert res_cond_lrg.could_be_resonant_by_conditions(mc_plus=magn_cond_a, mc_minus=())

    # Should return False: w1 + w2 > 0 and (crucially)  w1 + w2 + w3 + w4 + w5 + w6 > 0, even if w3 + w4 + w5 + w6 < 0
    assert not(res_cond_lrg.could_be_resonant_by_conditions(mc_plus=magn_cond_b, mc_minus=magn_cond_c))

    # Should return True: w1 + w2 > 0 and w3 + w4 + w5 + w6 > 0 so they together are enough information to make a
    # ruling of False, however the routine is not designed to recognize this (FieldAnalysis data would in this situation
    # also recognize w1 + w2 + w3 + w4 + w5 + w6 > 0 and include that)
    assert res_cond_lrg.could_be_resonant_by_conditions(mc_plus=magn_cond_d, mc_minus=())

    # Now testing application of previous resonances:

    # With magnitude conditions and residual freqs

    harm_osc_a = HarmOscStateSymbolic(['a'])
    harm_osc_c = HarmOscStateSymbolic(['c'])
    vd_harm_a_c = VibDiffTerm(harm_osc_a, harm_osc_c)
    pert_freq_12 = [1, 2]
    given_res_cond_a_c = ResonanceCondition(vd_harm_a_c, pert_freq_12)

    harm_osc_ab = HarmOscStateSymbolic(['a', 'b'])
    vd_harm_ab_c = VibDiffTerm(harm_osc_ab, harm_osc_c)
    pert_freq_1234 = [1, 2, 3, 4]
    res_cond_ab_c = ResonanceCondition(vd_harm_ab_c, pert_freq_1234)

    magn_conditions_34 = ([3, 4],)

    # Resonance can here be ruled out given that given_res_cond_a_c was resonant (but not otherwise with the present
    # magnitude conditions)
    assert not(res_cond_ab_c.could_be_resonant_by_conditions(mc_plus=(), mc_minus=magn_conditions_34,
                                                             given_prev_res=given_res_cond_a_c))

    # Non-covered previous resonance should give net effective bc, b resonance with given_res_cond_a_c
    harm_osc_b = HarmOscStateSymbolic(['b'])
    vd_harm_ab_b = VibDiffTerm(harm_osc_ab, harm_osc_b)
    pert_freq_1m2m34 = [1, 2, 3, 4]
    res_cond_ab_b = ResonanceCondition(vd_harm_ab_b, pert_freq_1m2m34)

    magn_conditions_m34 = ([3, 4],)

    # w3 + w4 < 0 and effective states 0, c should give
    assert not res_cond_ab_b.could_be_resonant_by_conditions(mc_plus=(), mc_minus=magn_conditions_m34,
                                                             given_prev_res=given_res_cond_a_c)

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