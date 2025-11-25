import pytest

from wilson_suite import fixtures as ws_fixtures

# FIXME: After separating classes from indep var and abstractions, split imports as appropriate between resp test files
from wilson_suite.wilson_experiment.abstractions import (SignedPulseTuple, PhaseMatchingCondition,
        IndependentVariableSet, IndependentVariableChoices, SpectralAxis, SpectralAxisSet, SpectralAxisChoices,
        find_subsets_making_orig, find_branching_indep_var_combs, find_indep_vars_for_one_phasematch,
        find_indep_exp_variables, find_axes_recursion, find_valid_axes_cfgs_for_one_phasematch,
        find_canonical_axes_for_one_phasematch, find_canonical_axes, find_valid_axes,
        SpecDetector, SpecScan, EmPulse, ElectricField, VibExperiment, get_carrier_freqs_uv, find_epochs, uv_cancels)

def test_signed_pulse_tuple():

    # Construct SignedPulseTuple with tuple of signed pulse references
    pulse_refs = (-1, 2, 3)
    spt = SignedPulseTuple(pulse_refs)
    assert spt.pulse_refs == (-1, 2, 3)

    # pulse references must be a tuple
    pulse_refs = [-1, 2]
    with pytest.raises(TypeError):
        spt = SignedPulseTuple(pulse_refs)

def test_phase_matching_condition():

    # -k1 + k2 + k3
    pulse_tuple_m1p2p3 = SignedPulseTuple((-1, 2, 3))
    phasematch_id = 0
    phasematch_m1p2p3 = PhaseMatchingCondition(pulse_tuple_m1p2p3, phasematch_id)

    assert phasematch_m1p2p3.id == 0
    assert phasematch_m1p2p3.pulses.pulse_refs == (-1, 2, 3)

    # Noninteger ID flag
    with pytest.raises(TypeError):
        phasematch_bogus_id = 'a'
        phasematch_m1p2p3_bogus = PhaseMatchingCondition(pulse_tuple_m1p2p3, phasematch_bogus_id)

    # Negative ID flag
    with pytest.raises(TypeError):
        phasematch_negative_id = -2
        phasematch_m1p2p3_bogus = PhaseMatchingCondition(pulse_tuple_m1p2p3, phasematch_negative_id)

def test_independent_variable_set():

    pulse_tuple_m1p2p3 = SignedPulseTuple((-1, 2, 3))
    pulse_tuple_p4m5 = SignedPulseTuple((4, -5))
    pulse_tuple_m1p6 = SignedPulseTuple((-1, 6))
    pulse_tuple_p1 = SignedPulseTuple((1,))

    ind_var_set = IndependentVariableSet((pulse_tuple_m1p2p3, pulse_tuple_p4m5))

    assert ind_var_set.var_set[0].pulse_refs == (-1, 2, 3)
    assert ind_var_set.var_set[1].pulse_refs == (4, -5)

    # Using list instead of tuple
    with pytest.raises(TypeError):
        ind_var_bogus = IndependentVariableSet([pulse_tuple_m1p2p3, pulse_tuple_p4m5])

    # Using tuple but elements are not SignedPulseTuple
    with pytest.raises(TypeError):
        ind_var_bogus = IndependentVariableSet(((-1, 2, 3), (4, -5)))

    # Repeating pulse refs
    with pytest.raises(ValueError):
        ind_var_bogus = IndependentVariableSet((pulse_tuple_m1p2p3, pulse_tuple_m1p6))

    # Repeating pulse refs (opposite sign)
    with pytest.raises(ValueError):
        ind_var_bogus = IndependentVariableSet((pulse_tuple_m1p2p3, pulse_tuple_p1))

def test_independent_variable_choices():

    # This test case consists of well-formed instances but is mock w.r.t. actual experiments

    # -k1 + k2 + k3
    pulse_tuple_m1p2p3 = SignedPulseTuple((-1, 2, 3))
    pulse_tuple_p4m5 = SignedPulseTuple((4, -5))
    pulse_tuple_m1p6 = SignedPulseTuple((-1, 6))

    phasematch_id = 2
    phasematch_m1p2p3 = PhaseMatchingCondition(pulse_tuple_m1p2p3, phasematch_id)

    ind_var_set_a = IndependentVariableSet((pulse_tuple_m1p2p3, pulse_tuple_p4m5))
    ind_var_set_b = IndependentVariableSet((pulse_tuple_m1p6,))

    ind_var_choices = IndependentVariableChoices(phasematch_m1p2p3, (ind_var_set_a, ind_var_set_b))

    assert isinstance(ind_var_choices.phasematch_cond, PhaseMatchingCondition)
    assert ind_var_choices.phasematch_cond.pulses.pulse_refs == (-1, 2, 3)
    assert ind_var_choices.phasematch_cond.id == 2
    assert ind_var_choices.var_groups[0].var_set[0].pulse_refs == (-1, 2, 3)
    assert ind_var_choices.var_groups[0].var_set[1].pulse_refs == (4, -5)
    assert ind_var_choices.var_groups[1].var_set[0].pulse_refs == (-1, 6)

    # None for one ind var set
    with pytest.raises(TypeError):
        ind_var_choices_bogus = IndependentVariableChoices(phasematch_m1p2p3, (None, ind_var_set_b))

    # Ind var sets as list
    with pytest.raises(TypeError):
        ind_var_choices_bogus = IndependentVariableChoices(phasematch_m1p2p3, [ind_var_set_a, ind_var_set_b])

    # phasematch_cond not PhaseMatchingCondition
    with pytest.raises(TypeError):
        ind_var_choices_bogus = IndependentVariableChoices('bogus', (ind_var_set_a, ind_var_set_b))

def test_spectral_axis():

    axis_label = 'A'

    pulse_tuple_m1p2p3 = SignedPulseTuple((-1, 2, 3))
    pulse_tuple_p4m5 = SignedPulseTuple((4, -5))
    ind_var_set = IndependentVariableSet((pulse_tuple_m1p2p3, pulse_tuple_p4m5))

    axis = SpectralAxis(axis_label, ind_var_set)

    assert axis.label == 'A'
    assert axis.var_set.var_set[0].pulse_refs == (-1, 2, 3)
    assert axis.var_set.var_set[1].pulse_refs == (4, -5)

    # Axis label as integer
    with pytest.raises(TypeError):
        axis_bogus = SpectralAxis(1, ind_var_set)

    # var_set not IndependentVariableSet
    with pytest.raises(TypeError):
        axis_bogus = SpectralAxis(axis_label, ['bogus'])


def test_spectral_axis_set():

    pulse_tuple_m1p2p3 = SignedPulseTuple((-1, 2, 3))
    pulse_tuple_p4m5 = SignedPulseTuple((4, -5))

    ind_var_set_a = IndependentVariableSet((pulse_tuple_m1p2p3, pulse_tuple_p4m5))
    ind_var_set_b = IndependentVariableSet((pulse_tuple_p4m5,))

    axis_label_a = 'A'
    axis_label_b = 'B'

    axis_a = SpectralAxis(axis_label_a, ind_var_set_a)
    axis_b = SpectralAxis(axis_label_b, ind_var_set_b)

    axis_set = SpectralAxisSet((axis_a, axis_b))

    assert axis_set.axes[0].label == 'A'
    assert axis_set.axes[0].var_set.var_set[0].pulse_refs == (-1, 2, 3)
    assert axis_set.axes[0].var_set.var_set[1].pulse_refs == (4, -5)

    assert axis_set.axes[1].label == 'B'
    assert axis_set.axes[1].var_set.var_set[0].pulse_refs == (4, -5)

    # Repeating axis names
    with pytest.raises(ValueError):

        axis_b_bogus = SpectralAxis(axis_label_a, ind_var_set_b)
        axis_set_bogus = SpectralAxisSet((axis_a, axis_b_bogus))

    # axes not tuple
    with pytest.raises(TypeError):
        axis_set_bogus = SpectralAxisSet([axis_a, axis_b])

    # Malformed axes element
    with pytest.raises(TypeError):
        axis_set_bogus = SpectralAxisSet(('bogus', axis_b))

def test_spectral_axis_choice_set():

    # This test case consists of well-formed instances but is mock w.r.t. actual experiments

    pulse_tuple_m1p2p3 = SignedPulseTuple((-1, 2, 3))
    pulse_tuple_p4m5 = SignedPulseTuple((4, -5))

    phasematch_id = 2
    phasematch_m1p2p3 = PhaseMatchingCondition(pulse_tuple_m1p2p3, phasematch_id)

    ind_var_set_a = IndependentVariableSet((pulse_tuple_m1p2p3, pulse_tuple_p4m5))
    ind_var_set_b = IndependentVariableSet((pulse_tuple_p4m5,))

    axis_label_a = 'A'
    axis_label_b = 'B'

    axis_a = SpectralAxis(axis_label_a, ind_var_set_a)
    axis_b = SpectralAxis(axis_label_b, ind_var_set_b)

    axis_set_1 = SpectralAxisSet((axis_a, axis_b))
    axis_set_2 = SpectralAxisSet((axis_b,))
    axis_set_3 = SpectralAxisSet((axis_b, axis_a))

    axis_choices = SpectralAxisChoices(phasematch_m1p2p3, ind_var_set_a, (axis_set_1, axis_set_2, axis_set_3))

    # Phase-matching direction -k1 + k2 + k3
    assert axis_choices.phasematch_cond.pulses.pulse_refs == (-1, 2, 3)
    assert axis_choices.phasematch_cond.id == 2

    # Axes for (mock) independent variables ((-w1 + w2 + w3), (w4 - w5))
    assert axis_choices.ind_vars.var_set[0].pulse_refs == (-1, 2, 3)
    assert axis_choices.ind_vars.var_set[1].pulse_refs == (4, -5)

    # First axis set: A: (-w1 + w2 + w3) + (w4 - w5), B: (w4 - w5)
    assert axis_choices.valid_axis_combs[0].axes[0].label == 'A'
    assert axis_choices.valid_axis_combs[0].axes[0].var_set.var_set[0].pulse_refs == (-1, 2, 3)
    assert axis_choices.valid_axis_combs[0].axes[0].var_set.var_set[1].pulse_refs == (4, -5)
    assert axis_choices.valid_axis_combs[0].axes[1].label == 'B'
    assert axis_choices.valid_axis_combs[0].axes[1].var_set.var_set[0].pulse_refs == (4, -5)

    # Second axis set: B: B: (w4 - w5)
    assert axis_choices.valid_axis_combs[1].axes[0].label == 'B'
    assert axis_choices.valid_axis_combs[1].axes[0].var_set.var_set[0].pulse_refs == (4, -5)

    # Third axis set: B: B: (w4 - w5),  A: (-w1 + w2 + w3) + (w4 - w5) (equivalent to first axis set but opposite order)
    assert axis_choices.valid_axis_combs[2].axes[0].label == 'B'
    assert axis_choices.valid_axis_combs[2].axes[0].var_set.var_set[0].pulse_refs == (4, -5)
    assert axis_choices.valid_axis_combs[2].axes[1].label == 'A'
    assert axis_choices.valid_axis_combs[2].axes[1].var_set.var_set[0].pulse_refs == (-1, 2, 3)
    assert axis_choices.valid_axis_combs[2].axes[1].var_set.var_set[1].pulse_refs == (4, -5)

    # phasematch_cond not PhaseMatchingCondition
    with pytest.raises(TypeError):
        axis_choices_bogus = SpectralAxisChoices(ind_var_set_a, ind_var_set_a, (axis_set_1, axis_set_2, axis_set_3))

    # ind_vars not IndependentVariableSet
    with pytest.raises(TypeError):
        axis_choices_bogus = SpectralAxisChoices(phasematch_m1p2p3, phasematch_m1p2p3, [axis_set_1, axis_set_2, axis_set_3])

    # valid_axis_combs not tuple
    with pytest.raises(TypeError):
        axis_choices_bogus = SpectralAxisChoices(phasematch_m1p2p3, ind_var_set_a, [axis_set_1, axis_set_2, axis_set_3])

    # malformed valid_axis_combs element
    with pytest.raises(TypeError):
        axis_choices_bogus = SpectralAxisChoices(phasematch_m1p2p3, ind_var_set_a,
                                                 (axis_set_1, 'bogus', axis_set_3))

def test_find_subsets_making_orig():



    pass

def test_find_branching_indep_var_combs():

    pass

def test_find_indep_vars_for_one_phasematch():

    pass


def test_find_indep_exp_variables():

    pass

def test_find_axes_recursion():

    pass


def test_find_valid_axes_cfgs_for_one_phasematch():

    pass

def test_find_canonical_axes_for_one_phasematch():

    pass

def test_find_canonical_axes():

    pass

def test_find_valid_axes():

    evv_exp = ws_fixtures.evv_experiment_pulse_1_and_2_coincident()

    print('relevant phase-matching conditions', evv_exp.relevant_phasematch)
    print('interaction sequences', evv_exp.int_sequences)
    print('independent variables', evv_exp.indep_vars)
    print('valid axis combinations', evv_exp.valid_axis_combs)
    print('canonical axes', evv_exp.canonical_axes)

    for i in evv_exp.valid_axis_combs:
        print('New axis combinations')
        print(i.phasematch_cond)
        for j in i.valid_axis_combs:
            for k in j.axes:
                print(k.label)
                for m in k.var_set.var_set:
                    print(m.pulse_refs)
        print('For independent variables')
        for j in i.ind_vars.var_set:
            print(j.pulse_refs)

    for i in evv_exp.canonical_axes:
        print('Canonical axes')
        print(i.phasematch_cond)
        for j in i.valid_axis_combs:
            for k in j.axes:
                print(k.label)
                for m in k.var_set.var_set:
                    print(m.pulse_refs)
        print('For independent variables')
        for j in i.ind_vars.var_set:
            print(j.pulse_refs)

    pass