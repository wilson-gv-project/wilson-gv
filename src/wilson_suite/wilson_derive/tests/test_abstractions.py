import pytest

from wilson_suite.wilson_derive.abstractions import (QOperator, HarmOscStateSymbolic, VibStateSymbolic, PolProp,
                                                     VibDiffTerm, ResonanceCondition, TransitionIntegral, LineShape)

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

    # Simple example: w_a - w_(b + c) - (w_1 - w_2 + w_3)
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

    # Signifying that w1 - (-w2) > 0
    magn_conditions_1_m_m2_gt_0 = [[1, -2]]
    assert res_cond_a_0.couldBeResonantWithFieldByConditions(magn_conditions_1_m_m2_gt_0)

    # Signifying that -w2 - w1 > 0
    magn_conditions_m2_m_1_gt_0 = [[-1, 2]]
    assert not(res_cond_a_0.couldBeResonantWithFieldByConditions(magn_conditions_m2_m_1_gt_0))

    # More tests of "could be resonant": More magnitude conditions, residual freqs, no magnitude conditions, no
    # perturbing freqs, maybe some more

    # Not testing couldBeResonantWithFieldByRanges (not finished)

    # Bogus:


    # Not VibDiffTerm

    # Not HarmOscStateSymbolic states in VibDiffTerm

    # Frequency labels not strings or integers

    # Identifier not integer



def test_line_shape():

    # Functionality not currently used, deferring tests

    pass



def test_pol_prop():


    pass

def test_pol_prop_sos_recursion():


    pass

def test_transition_integral():

    pass
