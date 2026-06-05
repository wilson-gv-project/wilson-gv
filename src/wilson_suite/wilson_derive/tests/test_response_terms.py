from fractions import Fraction

import pytest

from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm, VibContribTerm

# NOTE: Methods in these classes are used in get_fully_enhanced_terms and once verified here alleviates test range there

# Here to EOF: Claude code, review and check

def test_vib_perturbed_term():

    # Sketch: Make EVV relevant term instance(s), test init (why not also datatype asserts) and methods
    # May need a few different cases for the methods

    import copy
    from fractions import Fraction
    from wilson_suite.fixtures import evv_experiment
    from wilson_suite.wilson_derive.derive import get_fully_enhanced_terms
    from wilson_suite.wilson_derive.abstractions import PolProp, VibDiffTerm, ResonanceCondition, HarmOscStateSymbolic
    from wilson_suite.wilson_utils import common_labels as wu_common

    evv_exp = evv_experiment()
    terms = get_fully_enhanced_terms(experiment=evv_exp)

    # Constructor type checking
    t = terms[1][(1, 0)][0]
    with pytest.raises(TypeError):
        VibPerturbedTerm(1, t.props, t.freqterms, t.res)  # coeff not Fraction
    with pytest.raises(TypeError):
        VibPerturbedTerm(t.coeff, [object()], t.freqterms, t.res)  # bad prop type
    with pytest.raises(TypeError):
        VibPerturbedTerm(t.coeff, t.props, [object()], t.res)  # bad freqterm type
    with pytest.raises(TypeError):
        VibPerturbedTerm(t.coeff, t.props, t.freqterms, [object()])  # bad res type

    t.present()

    # Known values for el. anharm term (verified in test_term_var_translate.py)
    assert t.coeff == Fraction(-1, 4)
    assert len(t.props) == 3
    assert len(t.freqterms) == 2
    assert len(t.res) == 2

    # Resonance conditions
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [-1]
    assert t.res[1].diff.sl.q == ['b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [-1, 2]

    # Freqterms
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []
    assert not t.freqterms[0].is_pert_wf_diff
    assert t.freqterms[1].sl.q == ['b']
    assert t.freqterms[1].sr.q == []
    assert not t.freqterms[1].is_pert_wf_diff

    # Properties
    assert t.props[0].ops[0].o == 1
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['a']
    assert t.props[1].ops[0].o == 2
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['b']
    assert t.props[2].dord == 2
    assert t.props[2].inds == ['a', 'b']
    assert sorted([op.o for op in t.props[2].ops]) == [0, 3]

    # h() requires was_sorted=True first
    with pytest.raises(AssertionError):
        VibPerturbedTerm(t.coeff, t.props, t.freqterms, t.res).h()

    # h() after sort returns an integer and is stable across calls
    t_copy = copy.deepcopy(t)
    t_copy.sort(wu_common.nm_inds)
    h1 = t_copy.h()
    h2 = t_copy.h()
    assert isinstance(h1, int)
    assert h1 == h2

    # TODO:
    #  verify and possibly amend above init checks
    #  nmRenameAndInternalResort testing
    #  sort testing
    #  more full_enhancement_possible testing

    # full_enhancement_possible with EVV magn_conditions
    assert t.full_enhancement_possible(magn_conditions=evv_exp.magn_conditions)


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

