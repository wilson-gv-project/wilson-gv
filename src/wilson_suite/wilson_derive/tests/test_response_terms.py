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

    # Sketch: Make EVV relevant term instance(s), test init (why not also datatype asserts) and methods
    # May need a few different cases for the methods

    # NOTE: Test allRspEpochContained for wider selection of cfgs; see if non-counting ordered pulse refs still give
    # correct result

    import copy
    from fractions import Fraction
    from wilson_suite.fixtures import evv_experiment
    from wilson_suite.wilson_derive.vib_rsp_sos import get_vib_sos
    from wilson_suite.wilson_derive.abstractions import TransitionIntegral, PolProp, VibStateSymbolic, ResonanceCondition, QOperator, VibDiffTerm
    from wilson_suite.wilson_utils import common_labels as wu_common

    evv_exp = evv_experiment()
    R_sos = get_vib_sos(evv_exp.order)

    # Constructor type checking on a raw SOS term
    raw_term = R_sos[0]
    with pytest.raises(TypeError):
        VibContribTerm(1, raw_term.ints, raw_term.res)  # coeff not Fraction
    with pytest.raises(TypeError):
        VibContribTerm(raw_term.coeff, [object()], raw_term.res)  # bad ints type
    with pytest.raises(TypeError):
        VibContribTerm(raw_term.coeff, raw_term.ints, [object()])  # bad res type

    # Dress and filter to get EVV terms
    R_sos_evv = []
    for int_seq in evv_exp.int_sequences:
        for term in R_sos:
            t = copy.deepcopy(term)
            t.dressWithPulseInteractions(int_seq)
            if t.allElRspEpochContained(evv_exp.epochs, 0) and t.allUVCancels(evv_exp.cfuv):
                R_sos_evv.append(t)
    assert len(R_sos_evv) == 4

    # Structural checks on EVV terms: each has 3 integrals, "mu^2 alpha" pattern = op counts [1,1,2]
    for term in R_sos_evv:
        op_counts = sorted([len(i.prop.ops) for i in term.ints])
        assert op_counts == [1, 1, 2]
        # The integral with 2 operators must contain the omega (detected) operator
        two_op_ints = [i for i in term.ints if len(i.prop.ops) == 2]
        assert len(two_op_ints) == 1
        assert 0 in [op.o for op in two_op_ints[0].prop.ops]

    # allUVCancels: EVV terms must cancel; a term with UV in resonance condition should not
    assert all(t.allUVCancels(evv_exp.cfuv) for t in R_sos_evv)

    # allElRspEpochContained: EVV terms pass; an EEE-type undressed term fails with separate epochs
    # (EEE has a single integral with ops [omega, op1, op2, op3]; after dressing, ops [0,1,2,3]
    # which puts ops from multiple epochs in one integral)
    R_sos_dressed_non_evv = []
    for int_seq in evv_exp.int_sequences:
        for term in R_sos:
            t = copy.deepcopy(term)
            t.dressWithPulseInteractions(int_seq)
            R_sos_dressed_non_evv.append(t)

    # Term at index 0 is EEE (all-E type): single integral with all 4 operators
    eee_term = R_sos_dressed_non_evv[0]
    assert len(eee_term.ints) == 1
    assert len(eee_term.ints[0].prop.ops) == 4
    assert not eee_term.allElRspEpochContained(evv_exp.epochs, 0)

    # Resonance conditions of the 4 EVV terms: first-order [[-1]] and second-order [[-1, 2]]
    for term in R_sos_evv:
        assert len(term.res) == 2
        assert term.res[0].pf == [-1]
        assert term.res[1].pf == [-1, 2]


    # TODO:
    #  verify and possibly amend above init related code
    #  addfreqterm test
    #  verify and amend above dresswithpulseinteractions code
    #  small alluvcancels test (mostly subcall to uvCancels which is already covered)
    #  small allelrspepochcontained test (mostly subcall to epochContained which is already covered)
