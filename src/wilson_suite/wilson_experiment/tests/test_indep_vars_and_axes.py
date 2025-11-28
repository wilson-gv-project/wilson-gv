import copy
import pytest

from wilson_suite.wilson_experiment.indep_vars_and_axes import (SignedPulseTuple, PhaseMatchingCondition,
                IndependentVariableSet, IndependentVariableChoices, SpectralAxis, SpectralAxisSet, SpectralAxisChoices,
                find_subsets_making_orig, find_branching_indep_var_combs, find_indep_vars_for_one_phasematch,
                find_indep_exp_variables, find_axes_recursion, find_valid_axes_cfgs_for_one_phasematch,
                find_canonical_axes_for_one_phasematch, find_canonical_axes, find_valid_axes)

from wilson_suite.wilson_experiment.experiment_abstractions import EmPulse

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

    # Current recursion accumulator
    acc = []

    # Results accumulator
    uv_subs_res = []

    # Signed list of pulse IDs set to be UV/VIS range (in current epoch): The "original collection"
    uv_this = [-3, 1, 2]

    # Subsets of uv_this found to have cancelling UV/VIS range frequency components
    uv_superset_cancel = [(-3, 1, 2), (1, 2), (-3,)]

    # Tail-recursive
    find_subsets_making_orig(uv_superset_cancel, acc, uv_this, uv_subs_res)

    # Here, the signed collection of pulses 1, 2 and -3, and the collection (1, 2) and (-3) should both be found to
    # constitute the original collection
    assert sorted(uv_subs_res) == [[(-3, 1, 2)], [(1, 2), (-3,)]]

    # More extensive example
    uv_subs_res = []
    uv_this = [-4, -3, 1, 2, 5]
    uv_superset_cancel = [(-4, -3, 1, 2), (1, -3, 2), (-4,), (5,), (5, -4)]
    find_subsets_making_orig(uv_superset_cancel, [], uv_this, uv_subs_res)

    assert sorted(uv_subs_res) == [[(-4, -3, 1, 2), (5,)], [(1, -3, 2), (-4,), (5,)], [(1, -3, 2), (5, -4)]]

    # Malformed uv this ref
    uv_subs_res = []
    uv_this = [-4, 'a', -3, 1, 2]
    uv_superset_cancel = [(-4, -3, 1, 2), (-3, 1, 2), (-4,), (5,), (5, -4)]

    with pytest.raises(TypeError):
        find_subsets_making_orig(uv_superset_cancel, [], uv_this, uv_subs_res)

    # Malformed cancellation element
    uv_subs_res = []
    uv_this = [-4, 5, -3, 1, 2]
    uv_superset_cancel = [(-4, -3, 1, 2), (-3, 1, 2), (-4,), (5,), ('a', -4)]

    with pytest.raises(TypeError):
        find_subsets_making_orig(uv_superset_cancel, [], uv_this, uv_subs_res)

    # Both above cases: There would be cancelling set if elements were blindly read but TypeError is raised
    # because of malformation: This is neither a bug nor an intended feature
    uv_subs_res = []
    uv_this = [-4, 'a', -3, 1, 2]
    uv_superset_cancel = [(-4, -3, 1, 2), (-3, 1, 2), (-4,), (5,), ('a', -4)]

    with pytest.raises(TypeError):
        find_subsets_making_orig(uv_superset_cancel, [], uv_this, uv_subs_res)

def test_find_branching_indep_var_combs():

    # Current recursion accumulator
    seed_comb = []

    # Results accumulator
    ind_vars_p = []

    # Epoch counter
    curr_epoch = 0

    # Original identified independent variables where individual branching choices are structured in
    # Format:
    #   - Outermost list over epochs. In each epoch:
    #   - Tuples of IR range pulses (signed) in this epoch first, then a list
    #   of options for combinations of UV/VIS (signed) pulses in this epoch. For each such option:
    #   - A list of tuples describing collections of UV/VIS pulses that together constitute all the UV/VIS pulses
    #   in this epoch and where each tuple denotes a UV/VIS collection whose sum UV/VIS frequency component sums to 0

    # Here: Two epochs.
    # - Epoch 1: Pulses 4 and 5 are IR range pulses. Pulses -3, 1, 2 are UV/VIS pulses, where
    #   two partitionings exist so that in each partition in each partitioning, the UV/VIS freq components sum to zero:
    #   *  w1_UV + w2_UV - w3_UV sum to zero
    #   * -w3_UV sums to zero and w1_UV + w2_UV sum to zero (NOTE: Fictitious example since pulse 3
    #   would then in practice be classified as an IR range pulse by earlier routines, but this has no bearing on
    #   the functionality presently being tested)
    # - Epoch 2: Pulse 6 is an IR range pulse. Pulses -7, 8 are UV/VIS pulses, where
    #   two partitionings exist so that in each partition in each partitioning, the UV/VIS freq components sum to zero:
    #   * -w7_UV + w8_UV sums to zero
    #   * -w7_UV sums to zero and w8_UV sums to zero (NOTE: Same comment as for -w3_UV applies here)
    orig_vars = [[(4,), (5,), [[(-3, 1, 2)], [(-3,), (1, 2)]] ], [(6,), [[(-7,), (8,)], [(-7, 8)]] ]]

    find_branching_indep_var_combs(ind_vars_p, orig_vars, seed_comb, curr_epoch)

    # There are therefore two options for choices of UV/VIS combinations in each epoch, so we expect four combinations
    # altogether:
    assert ind_vars_p == [
                          [(4,), (5,), (-3, 1, 2), (6,), (-7,), (8,)],
                          [(4,), (5,), (-3, 1, 2), (6,), (-7, 8)],
                          [(4,), (5,), (-3,), (1, 2), (6,), (-7,), (8,)],
                          [(4,), (5,), (-3,), (1, 2), (6,), (-7, 8)]
                          ]

    # Simpler test cases

    # Two IR range pulses in one epoch: Should give same pulses back as ind vars
    seed_comb = []
    ind_vars_p = []
    orig_vars = [[(1,), (2,)]]
    find_branching_indep_var_combs(ind_vars_p, orig_vars, seed_comb, 0)
    assert ind_vars_p == [ [(1,), (2,)] ]

    # Two IR range pulses in different epochs: Should give same result as prev
    seed_comb = []
    ind_vars_p = []
    orig_vars = [[(1,)], [(2,)]]
    find_branching_indep_var_combs(ind_vars_p, orig_vars, seed_comb, 0)
    assert ind_vars_p == [ [(1,), (2,)] ]

    # One epoch, three UV/VIS pulses that cancel (and no more granular cancellations):
    # Should give same collection as the single ind var
    seed_comb = []
    ind_vars_p = []
    orig_vars = [[[[(1, 2, -3)]]]]
    find_branching_indep_var_combs(ind_vars_p, orig_vars, seed_comb, 0)
    assert ind_vars_p == [ [(1, 2, -3)] ]

def test_find_indep_vars_for_one_phasematch():

    # Four UV/VIS range pulses over two epochs
    pulse_a = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.072, id=1)
    pulse_b = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.072, id=2)
    pulse_c = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.144, id=3)
    pulse_d = EmPulse(env='impulsive', maxstr=1e-05, tc=120.0, cf=0.0, cf_uv=0.072, id=4)

    pulses = [pulse_a, pulse_b, pulse_c, pulse_d]

    # Three pulses in epoch 1, one in epoch 2
    epochs = [[1, 2, 3], [4]]

    # This phase-matching direction makes the UV/VIS component of the three epoch 1 pulses cancel
    pm_dir = SignedPulseTuple(pulse_refs=(1, 2, -3, 4))

    # Returns a list of one IndependentVariableSet instance
    indep_vars = find_indep_vars_for_one_phasematch(pulses, epochs, pm_dir)


    assert indep_vars[0].var_set[0].pulse_refs == (-3, 1, 2)

    # EVV with all pulses at different times
    pulse_a = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.0, id=1)
    pulse_b = EmPulse(env='impulsive', maxstr=1e-05, tc=20.0, cf=0.0, cf_uv=0.0, id=2)
    pulse_c = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.144, id=3)

    pulses = [pulse_a, pulse_b, pulse_c]
    epochs = [[1], [2], [3]]
    pm_dir = SignedPulseTuple(pulse_refs=(-1, 2, 3))

    indep_vars = find_indep_vars_for_one_phasematch(pulses, epochs, pm_dir)

    assert indep_vars[0].var_set[0].pulse_refs == (-1,)
    assert indep_vars[0].var_set[1].pulse_refs == (2,)

    # Complicated setup: Two IR pulses in first epoch, one IR pulse and four UV/VIS in 2nd epoch, where the UV/VIS
    # pulses have two different cancelling partitionings, and one UV/VIS pulse in the 3rd epoch not eligible to form
    # an independent variable
    pulse_a = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.0, id=1)
    pulse_b = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.0, id=2)
    pulse_c = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.072, id=3)
    pulse_d = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.072, id=4)
    pulse_e = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.144, id=5)
    pulse_f = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.144, id=6)
    pulse_g = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.0, id=7)
    pulse_h = EmPulse(env='impulsive', maxstr=1e-05, tc=50.0, cf=0.0, cf_uv=0.072, id=8)

    pulses = [pulse_a, pulse_b, pulse_c, pulse_d, pulse_e, pulse_f, pulse_g, pulse_h]
    epochs = [[1, 2], [3, 4, 5, 6, 7], [8]]
    pm_dir = SignedPulseTuple(pulse_refs=(-1, 2, 3, -4, 5, -6, 7, -8))

    indep_vars = find_indep_vars_for_one_phasematch(pulses, epochs, pm_dir)

    # Should return two sets of independent variables (differing in the UV/VIS partitioning)
    assert len(indep_vars) == 2
    assert isinstance(indep_vars[0], IndependentVariableSet)
    assert isinstance(indep_vars[1], IndependentVariableSet)

    # All sets are ordered by epoch, then internally by epoch with variables from IR pulses first, then
    # variables from UV/VIS pulses last. All UV/VIS combination (== len > 1) variables are sorted by pulse ID

    # First set: Five variables
    assert len(indep_vars[0].var_set) == 5
    assert indep_vars[0].var_set[0].pulse_refs == (-1,)
    assert indep_vars[0].var_set[1].pulse_refs == (2,)
    assert indep_vars[0].var_set[2].pulse_refs == (7,)
    assert indep_vars[0].var_set[3].pulse_refs == (-6, 5)
    assert indep_vars[0].var_set[4].pulse_refs == (-4, 3)

    # Second set: Four variables
    assert len(indep_vars[1].var_set) == 4
    assert indep_vars[1].var_set[0].pulse_refs == (-1,)
    assert indep_vars[1].var_set[1].pulse_refs == (2,)
    assert indep_vars[1].var_set[2].pulse_refs == (7,)
    assert indep_vars[1].var_set[3].pulse_refs == (-6, -4, 3, 5)

    # Two "Raman-like pulses" in first epoch, three UV/VIS pulses in 2nd epoch with two different ways of cancelling
    pulse_a = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.072, id=1)
    pulse_b = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.072, id=2)
    pulse_c = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.036, id=3)
    pulse_d = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.036, id=4)
    pulse_e = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.036, id=5)

    pulses = [pulse_a, pulse_b, pulse_c, pulse_d, pulse_e]
    epochs = [[1, 2], [3, 4, 5]]
    pm_dir = SignedPulseTuple(pulse_refs=(-1, 2, 3, 4, -5))

    indep_vars = find_indep_vars_for_one_phasematch(pulses, epochs, pm_dir)

    # Should return two sets of independent variables (differing by 5 cancelling against either 3 or 4)
    assert len(indep_vars) == 2

    assert len(indep_vars[0].var_set) == 2
    assert indep_vars[0].var_set[0].pulse_refs == (-1, 2)
    assert indep_vars[0].var_set[1].pulse_refs == (-5, 4)

    assert len(indep_vars[1].var_set) == 2
    assert indep_vars[1].var_set[0].pulse_refs == (-1, 2)
    assert indep_vars[1].var_set[1].pulse_refs == (-5, 3)


def test_find_indep_exp_variables():

    # Two IR pulses in first epoch, three UV/VIS pulses in 2nd epoch (same abs val UV/VIS freq component)
    pulse_a = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.0, id=1)
    pulse_b = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.0, id=2)
    pulse_c = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.036, id=3)
    pulse_d = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.036, id=4)
    pulse_e = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.036, id=5)

    pulses = [pulse_a, pulse_b, pulse_c, pulse_d, pulse_e]
    epochs = [[1, 2], [3, 4, 5]]

    # First phase-matching condition: -k1 + k2 + k3 + k4 - k5
    pm_cond_a = PhaseMatchingCondition(SignedPulseTuple(pulse_refs=(-1, 2, 3, 4, -5)))
    # Second condition: k1 - k2 + k3 + k4 + k5
    pm_cond_b = PhaseMatchingCondition(SignedPulseTuple(pulse_refs=(1, -2, -3, 4, -5)))
    # Third condition: -k1 + k2 + k3 + k4 + k5
    pm_cond_c = PhaseMatchingCondition(SignedPulseTuple(pulse_refs=(-1, 2, 3, 4, 5)))

    pm_conds = [pm_cond_a, pm_cond_b, pm_cond_c]

    indep_var_choices = find_indep_exp_variables(pulses, epochs, pm_conds)

    # Three sets of choices with three phase-matching conditions
    assert len(indep_var_choices) == 3
    for i in indep_var_choices:
        assert isinstance(i, IndependentVariableChoices)

    # Results for first phase-matching condition (5 can cancel against either 3 or 4).
    # See test_find_indep_vars_for_one_phasematch for more information about structure of variable groups
    assert indep_var_choices[0].phasematch_cond.pulses.pulse_refs == (-1, 2, 3, 4, -5)
    assert indep_var_choices[0].var_groups[0].var_set[0].pulse_refs == (-1,)
    assert indep_var_choices[0].var_groups[0].var_set[1].pulse_refs == (2,)
    assert indep_var_choices[0].var_groups[0].var_set[2].pulse_refs == (-5, 4)
    assert indep_var_choices[0].var_groups[1].var_set[0].pulse_refs == (-1,)
    assert indep_var_choices[0].var_groups[1].var_set[1].pulse_refs == (2,)
    assert indep_var_choices[0].var_groups[1].var_set[2].pulse_refs == (-5, 3)

    # Results for second phase-matching condition. Here 4 can cancel against either 5 or 3.
    assert indep_var_choices[1].phasematch_cond.pulses.pulse_refs == (1, -2, -3, 4, -5)
    assert indep_var_choices[1].var_groups[0].var_set[0].pulse_refs == (-2,)
    assert indep_var_choices[1].var_groups[0].var_set[1].pulse_refs == (1,)
    assert indep_var_choices[1].var_groups[0].var_set[2].pulse_refs == (-3, 4)
    assert indep_var_choices[1].var_groups[1].var_set[0].pulse_refs == (-2,)
    assert indep_var_choices[1].var_groups[1].var_set[1].pulse_refs == (1,)
    assert indep_var_choices[1].var_groups[1].var_set[2].pulse_refs == (-5, 4)

    # Results for third phase-matching condition: Here no subsets of (3, 4, 5) can cancel and so only the IR range
    # pulses can become variables. NOTE: This case is not encountered in practical calculations since no terms as found
    # by wilson-derive in this setup would survive filtering from exclusion of terms with resonance
    # conditions that can not be met by excitations in the vibrational manifold (there applying the  assumption that
    # the vibrational energy levels do not rise to the UV/VIS range)
    assert indep_var_choices[2].phasematch_cond.pulses.pulse_refs == (-1, 2, 3, 4, 5)
    assert len(indep_var_choices[2].var_groups) == 1
    assert len(indep_var_choices[2].var_groups[0].var_set) == 2
    assert indep_var_choices[2].var_groups[0].var_set[0].pulse_refs == (-1,)
    assert indep_var_choices[2].var_groups[0].var_set[1].pulse_refs == (2,)

def test_find_axes_recursion():

    # This routine (find_axes_recursion) works with basic datatypes

    # Two independent variables: -w1 and w2
    # Their ordering matters for the recursion, here showing one permutation:
    ind_vars = ((-1,), (2,))

    # Seed datatypes for recursion
    seed_ax_list = []
    seed_position = 0

    # Results are accumulated here
    res_axes = []

    find_axes_recursion(ind_vars, res_axes, seed_ax_list, seed_position)

    # Two choices of axes should be identified here:
    # - One choice with -w1 as axis 1 and w2 as axis 2
    # - One choice with -w1 as axis 1 and -w1 + w2 as axis 2
    assert res_axes == [
        [
            [(-1,)], [(2,)]
        ],
        [
            [(-1,)], [(-1,), (2,)]
        ]
    ]

    # Now doing the other permutation to identify further axes
    ind_vars = ((2,), (-1,))

    # Since res_axes here already contains results, the new axes identified from this call will be amended
    find_axes_recursion(ind_vars, res_axes, seed_ax_list, seed_position)

    # Two further choices of axes should be identified here:
    # - One choice with -w1 as axis 1 and w2 as axis 2
    # - One choice with -w1 as axis 1 and -w1 + w2 as axis 2
    assert res_axes == [
        [
            [(-1,)], [(2,)]
        ],
        [
            [(-1,)], [(-1,), (2,)]
        ],
        [
            [(2,)], [(-1,)]
        ],
        [
            [(2,)], [(-1,), (2,)]
        ]
    ]

    # Calling with the same permutation should not add new data to the result
    res_axes_ref = copy.deepcopy(res_axes)
    ind_vars = ((2,), (-1,))
    find_axes_recursion(ind_vars, res_axes, seed_ax_list, seed_position)
    assert res_axes == res_axes_ref

    # Larger example: Three independent variables, two of six possible permutations
    ind_vars = ((-1,), (-3, 2), (4,))

    seed_ax_list = []
    seed_position = 0
    res_axes = []

    find_axes_recursion(ind_vars, res_axes, seed_ax_list, seed_position)

    ind_vars = ((4,), (-3, 2), (-1,))

    find_axes_recursion(ind_vars, res_axes, seed_ax_list, seed_position)

    assert res_axes == [
        [
            [(-1,)], [(-3, 2)], [(4,)]
        ],
        [
            [(-1,)], [(-3, 2)], [(-3, 2), (4,)]
        ],
        [
            [(-1,)], [(-3, 2)], [(-3, 2), (-1,), (4,)]
        ],
        [
            [(-1,)], [(-3, 2)], [(-1,), (4,)]
        ],
        [
            [(-1,)], [(-3, 2), (-1,)], [(4,)]
        ],
        [
            [(-1,)], [(-3, 2), (-1,)], [(-3, 2), (4,)]
        ],
        [
            [(-1,)], [(-3, 2), (-1,)], [(-3, 2), (-1,), (4,)]
        ],
        [
            [(-1,)], [(-3, 2), (-1,)], [(-1,), (4,)]
        ],
        [
            [(4,)], [(-3, 2)], [(-1,)]
        ],
        [
            [(4,)], [(-3, 2)], [(-3, 2), (-1,)]
        ],
        [
            [(4,)], [(-3, 2)], [(-1,), (4,)]
        ],
        [
            [(4,)], [(-3, 2)], [(-3, 2), (-1,), (4,)]
        ],
        [
            [(4,)], [(-3, 2), (4,)], [(-1,)]
        ],
        [
            [(4,)], [(-3, 2), (4,)], [(-3, 2), (-1,)]
        ],
        [
            [(4,)], [(-3, 2), (4,)], [(-1,), (4,)]
        ],
        [
            [(4,)], [(-3, 2), (4,)], [(-3, 2), (-1,), (4,)]
        ]
    ]

def test_find_valid_axes_cfgs_for_one_phasematch():

    # EVV with all pulses at different times
    pulse_a = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.0, id=1)
    pulse_b = EmPulse(env='impulsive', maxstr=1e-05, tc=20.0, cf=0.0, cf_uv=0.0, id=2)
    pulse_c = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.144, id=3)

    pulses = [pulse_a, pulse_b, pulse_c]
    epochs = [[1], [2], [3]]
    pm_dir = [PhaseMatchingCondition(SignedPulseTuple(pulse_refs=(-1, 2, 3)), 1)]

    indep_var_choices = find_indep_exp_variables(pulses, epochs, pm_dir)

    axes_cfgs = find_valid_axes_cfgs_for_one_phasematch(indep_var_choices[0])

    # Results are organized as dictionaries with internal representation of independent variable set
    assert ((-1,), (2,)) in axes_cfgs

    # Four axis configurations should be found for this independent variable set
    # NOTE: The combinatorics here are not exhaustive: Internal permutations of 'A' and 'B' axes over a configuration
    # would yield more combinations, some of which can be equal to other configurations. This is disregarded for now
    # and can be pursued if relevant in the future.
    assert len(axes_cfgs[((-1,), (2,))]) == 4

    # First configuration: Axis 'A' is -w1, axis 'B' is -w1  + w2
    assert axes_cfgs[((-1,), (2,))][0].axes[0].label == 'A'
    assert len(axes_cfgs[((-1,), (2,))][0].axes[0].var_set.var_set) == 1
    assert axes_cfgs[((-1,), (2,))][0].axes[0].var_set.var_set[0].pulse_refs == (-1,)

    assert axes_cfgs[((-1,), (2,))][0].axes[1].label == 'B'
    assert len(axes_cfgs[((-1,), (2,))][0].axes[1].var_set.var_set) == 2
    assert axes_cfgs[((-1,), (2,))][0].axes[1].var_set.var_set[0].pulse_refs == (-1,)
    assert axes_cfgs[((-1,), (2,))][0].axes[1].var_set.var_set[1].pulse_refs == (2,)

    # Second configuration: Axis 'A' is -w1, axis 'B' is w2
    assert axes_cfgs[((-1,), (2,))][1].axes[0].label == 'A'
    assert len(axes_cfgs[((-1,), (2,))][1].axes[0].var_set.var_set) == 1
    assert axes_cfgs[((-1,), (2,))][1].axes[0].var_set.var_set[0].pulse_refs == (-1,)

    assert axes_cfgs[((-1,), (2,))][1].axes[1].label == 'B'
    assert len(axes_cfgs[((-1,), (2,))][1].axes[1].var_set.var_set) == 1
    assert axes_cfgs[((-1,), (2,))][1].axes[1].var_set.var_set[0].pulse_refs == (2,)

    # Third configuration: Axis 'A' is w2, axis 'B' is -w1
    assert axes_cfgs[((-1,), (2,))][2].axes[0].label == 'A'
    assert len(axes_cfgs[((-1,), (2,))][2].axes[0].var_set.var_set) == 1
    assert axes_cfgs[((-1,), (2,))][2].axes[0].var_set.var_set[0].pulse_refs == (2,)

    assert axes_cfgs[((-1,), (2,))][2].axes[1].label == 'B'
    assert len(axes_cfgs[((-1,), (2,))][2].axes[1].var_set.var_set) == 1
    assert axes_cfgs[((-1,), (2,))][2].axes[1].var_set.var_set[0].pulse_refs == (-1,)

    # Fourth configuration: Axis 'A' is w2, axis 'B' is -w1 + w2
    assert axes_cfgs[((-1,), (2,))][3].axes[0].label == 'A'
    assert len(axes_cfgs[((-1,), (2,))][3].axes[0].var_set.var_set) == 1
    assert axes_cfgs[((-1,), (2,))][3].axes[0].var_set.var_set[0].pulse_refs == (2,)

    assert axes_cfgs[((-1,), (2,))][3].axes[1].label == 'B'
    assert len(axes_cfgs[((-1,), (2,))][3].axes[1].var_set.var_set) == 2
    assert axes_cfgs[((-1,), (2,))][3].axes[1].var_set.var_set[0].pulse_refs == (-1,)
    assert axes_cfgs[((-1,), (2,))][0].axes[1].var_set.var_set[1].pulse_refs == (2,)


def test_find_valid_axes():

    # EVV with all pulses at different times
    pulse_a = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.0, id=1)
    pulse_b = EmPulse(env='impulsive', maxstr=1e-05, tc=20.0, cf=0.0, cf_uv=0.0, id=2)
    pulse_c = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.144, id=3)

    pulses = [pulse_a, pulse_b, pulse_c]
    epochs = [[1], [2], [3]]

    pm_dir_a = PhaseMatchingCondition(SignedPulseTuple(pulse_refs=(-1, 2, 3)), 1)
    pm_dir_b = PhaseMatchingCondition(SignedPulseTuple(pulse_refs=(1, -2, 3)), 2)

    indep_var_choices = find_indep_exp_variables(pulses, epochs, [pm_dir_a, pm_dir_b])

    axes_cfgs = find_valid_axes(indep_var_choices)

    # Two choice sets (two phase-matching directions and one set of independent variables each)
    assert len(axes_cfgs) == 2

    # First phase-matching direction
    assert axes_cfgs[0].phasematch_cond.pulses.pulse_refs == (-1, 2, 3)
    assert axes_cfgs[0].phasematch_cond.id == 1

    # Independent vars: -w1 and w2
    assert axes_cfgs[0].ind_vars.var_set[0].pulse_refs == (-1,)
    assert axes_cfgs[0].ind_vars.var_set[1].pulse_refs == (2,)

    # Four axis configurations
    assert len(axes_cfgs[0].valid_axis_combs) == 4

    # First configuration: A = -w1, B = -w1 + w2
    assert axes_cfgs[0].valid_axis_combs[0].axes[0].label == 'A'
    assert len(axes_cfgs[0].valid_axis_combs[0].axes[0].var_set.var_set) == 1
    assert axes_cfgs[0].valid_axis_combs[0].axes[0].var_set.var_set[0].pulse_refs == (-1,)

    assert axes_cfgs[0].valid_axis_combs[0].axes[1].label == 'B'
    assert len(axes_cfgs[0].valid_axis_combs[0].axes[1].var_set.var_set) == 2
    assert axes_cfgs[0].valid_axis_combs[0].axes[1].var_set.var_set[0].pulse_refs == (-1,)
    assert axes_cfgs[0].valid_axis_combs[0].axes[1].var_set.var_set[1].pulse_refs == (2,)

    # Second configuration: A = -w1, B = w2
    assert axes_cfgs[0].valid_axis_combs[1].axes[0].label == 'A'
    assert len(axes_cfgs[0].valid_axis_combs[1].axes[0].var_set.var_set) == 1
    assert axes_cfgs[0].valid_axis_combs[1].axes[0].var_set.var_set[0].pulse_refs == (-1,)

    assert axes_cfgs[0].valid_axis_combs[1].axes[1].label == 'B'
    assert len(axes_cfgs[0].valid_axis_combs[1].axes[1].var_set.var_set) == 1
    assert axes_cfgs[0].valid_axis_combs[1].axes[1].var_set.var_set[0].pulse_refs == (2,)

    # Third configuration: A = w2, B = -w1
    assert axes_cfgs[0].valid_axis_combs[2].axes[0].label == 'A'
    assert len(axes_cfgs[0].valid_axis_combs[2].axes[0].var_set.var_set) == 1
    assert axes_cfgs[0].valid_axis_combs[2].axes[0].var_set.var_set[0].pulse_refs == (2,)

    assert axes_cfgs[0].valid_axis_combs[2].axes[1].label == 'B'
    assert len(axes_cfgs[0].valid_axis_combs[2].axes[1].var_set.var_set) == 1
    assert axes_cfgs[0].valid_axis_combs[2].axes[1].var_set.var_set[0].pulse_refs == (-1,)

    # Fourth configuration: A = w2, B = -w1 + w2
    assert axes_cfgs[0].valid_axis_combs[3].axes[0].label == 'A'
    assert len(axes_cfgs[0].valid_axis_combs[3].axes[0].var_set.var_set) == 1
    assert axes_cfgs[0].valid_axis_combs[3].axes[0].var_set.var_set[0].pulse_refs == (2,)

    assert axes_cfgs[0].valid_axis_combs[3].axes[1].label == 'B'
    assert len(axes_cfgs[0].valid_axis_combs[3].axes[1].var_set.var_set) == 2
    assert axes_cfgs[0].valid_axis_combs[3].axes[1].var_set.var_set[0].pulse_refs == (-1,)
    assert axes_cfgs[0].valid_axis_combs[3].axes[1].var_set.var_set[1].pulse_refs == (2,)


    # Second phase-matching direction
    assert axes_cfgs[1].phasematch_cond.pulses.pulse_refs == (1, -2, 3)
    assert axes_cfgs[1].phasematch_cond.id == 2

    # Independent vars: w1 and -w2
    assert axes_cfgs[1].ind_vars.var_set[0].pulse_refs == (-2,)
    assert axes_cfgs[1].ind_vars.var_set[1].pulse_refs == (1,)

    # Four axis configurations
    assert len(axes_cfgs[1].valid_axis_combs) == 4

    # First configuration: A = -w2, B = -w2 + w1
    assert axes_cfgs[1].valid_axis_combs[0].axes[0].label == 'A'
    assert len(axes_cfgs[1].valid_axis_combs[0].axes[0].var_set.var_set) == 1
    assert axes_cfgs[1].valid_axis_combs[0].axes[0].var_set.var_set[0].pulse_refs == (-2,)

    assert axes_cfgs[1].valid_axis_combs[0].axes[1].label == 'B'
    assert len(axes_cfgs[1].valid_axis_combs[0].axes[1].var_set.var_set) == 2
    assert axes_cfgs[1].valid_axis_combs[0].axes[1].var_set.var_set[0].pulse_refs == (-2,)
    assert axes_cfgs[1].valid_axis_combs[0].axes[1].var_set.var_set[1].pulse_refs == (1,)

    # Second configuration: A = -w2, B = w1
    assert axes_cfgs[1].valid_axis_combs[1].axes[0].label == 'A'
    assert len(axes_cfgs[1].valid_axis_combs[1].axes[0].var_set.var_set) == 1
    assert axes_cfgs[1].valid_axis_combs[1].axes[0].var_set.var_set[0].pulse_refs == (-2,)

    assert axes_cfgs[1].valid_axis_combs[1].axes[1].label == 'B'
    assert len(axes_cfgs[1].valid_axis_combs[1].axes[1].var_set.var_set) == 1
    assert axes_cfgs[1].valid_axis_combs[1].axes[1].var_set.var_set[0].pulse_refs == (1,)

    # Third configuration: A = w1, B = -w2
    assert axes_cfgs[1].valid_axis_combs[2].axes[0].label == 'A'
    assert len(axes_cfgs[1].valid_axis_combs[2].axes[0].var_set.var_set) == 1
    assert axes_cfgs[1].valid_axis_combs[2].axes[0].var_set.var_set[0].pulse_refs == (1,)

    assert axes_cfgs[1].valid_axis_combs[2].axes[1].label == 'B'
    assert len(axes_cfgs[1].valid_axis_combs[2].axes[1].var_set.var_set) == 1
    assert axes_cfgs[1].valid_axis_combs[2].axes[1].var_set.var_set[0].pulse_refs == (-2,)

    # Fourth configuration: A = w1, B = -w2 + w1
    assert axes_cfgs[1].valid_axis_combs[3].axes[0].label == 'A'
    assert len(axes_cfgs[1].valid_axis_combs[3].axes[0].var_set.var_set) == 1
    assert axes_cfgs[1].valid_axis_combs[3].axes[0].var_set.var_set[0].pulse_refs == (1,)

    assert axes_cfgs[1].valid_axis_combs[3].axes[1].label == 'B'
    assert len(axes_cfgs[1].valid_axis_combs[3].axes[1].var_set.var_set) == 2
    assert axes_cfgs[1].valid_axis_combs[3].axes[1].var_set.var_set[0].pulse_refs == (-2,)
    assert axes_cfgs[1].valid_axis_combs[3].axes[1].var_set.var_set[1].pulse_refs == (1,)

    # NOTE: Can consider making more test cases here (e.g. more indep vars for one phase-matching condition)


def test_find_canonical_axes_for_one_phasematch():

    # EVV with all pulses at different times
    pulse_a = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.0, id=1)
    pulse_b = EmPulse(env='impulsive', maxstr=1e-05, tc=20.0, cf=0.0, cf_uv=0.0, id=2)
    pulse_c = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.144, id=3)

    pulses = [pulse_a, pulse_b, pulse_c]
    epochs = [[1], [2], [3]]
    pm_dir = [PhaseMatchingCondition(SignedPulseTuple(pulse_refs=(-1, 2, 3)), 1)]

    indep_var_choices = find_indep_exp_variables(pulses, epochs, pm_dir)

    axes_cfgs = find_canonical_axes_for_one_phasematch(indep_var_choices[0])

    # The canonical axes here are A: -w1 and B: w2
    assert len(axes_cfgs.axes) == 2
    assert axes_cfgs.axes[0].label == 'A'
    assert axes_cfgs.axes[0].var_set.var_set[0].pulse_refs == (-1,)
    assert axes_cfgs.axes[1].label == 'B'
    assert axes_cfgs.axes[1].var_set.var_set[0].pulse_refs == (2,)


    # Complicated setup: Two IR pulses in first epoch, one IR pulse and four UV/VIS in 2nd epoch, where the UV/VIS
    # pulses have two different cancelling partitionings, and one UV/VIS pulse in the 3rd epoch not eligible to form
    # an independent variable
    pulse_a = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.0, id=1)
    pulse_b = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.0, id=2)
    pulse_c = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.072, id=3)
    pulse_d = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.072, id=4)
    pulse_e = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.144, id=5)
    pulse_f = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.144, id=6)
    pulse_g = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.0, id=7)
    pulse_h = EmPulse(env='impulsive', maxstr=1e-05, tc=50.0, cf=0.0, cf_uv=0.072, id=8)

    pulses = [pulse_a, pulse_b, pulse_c, pulse_d, pulse_e, pulse_f, pulse_g, pulse_h]
    epochs = [[1, 2], [3, 4, 5, 6, 7], [8]]
    pm_dir = [PhaseMatchingCondition(SignedPulseTuple(pulse_refs=(-1, 2, 3, -4, 5, -6, 7, -8)), 1)]

    indep_var_choices = find_indep_exp_variables(pulses, epochs, pm_dir)
    axes_cfgs = find_canonical_axes_for_one_phasematch(indep_var_choices[0])

    # The canonical axes here are A: -w1, B: w2, C: w7 (IR before UV/VIS in epoch), D: w5 - w6, E: w3 - w4
    assert len(axes_cfgs.axes) == 5
    assert axes_cfgs.axes[0].label == 'A'
    assert axes_cfgs.axes[0].var_set.var_set[0].pulse_refs == (-1,)
    assert axes_cfgs.axes[1].label == 'B'
    assert axes_cfgs.axes[1].var_set.var_set[0].pulse_refs == (2,)
    assert axes_cfgs.axes[2].label == 'C'
    assert axes_cfgs.axes[2].var_set.var_set[0].pulse_refs == (7,)
    assert axes_cfgs.axes[3].label == 'D'
    assert axes_cfgs.axes[3].var_set.var_set[0].pulse_refs == (-6, 5)
    assert axes_cfgs.axes[4].label == 'E'
    assert axes_cfgs.axes[4].var_set.var_set[0].pulse_refs == (-4, 3)



def test_find_canonical_axes():

    # Not much extra to test here currently over test_find_canonical_axes_for_one_phasematch

    # EVV with all pulses at different times
    pulse_a = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.0, id=1)
    pulse_b = EmPulse(env='impulsive', maxstr=1e-05, tc=20.0, cf=0.0, cf_uv=0.0, id=2)
    pulse_c = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.144, id=3)

    pulses = [pulse_a, pulse_b, pulse_c]
    epochs = [[1], [2], [3]]
    pm_dir = [PhaseMatchingCondition(SignedPulseTuple(pulse_refs=(-1, 2, 3)), 1)]

    indep_var_choices = find_indep_exp_variables(pulses, epochs, pm_dir)

    axes_cfgs = find_canonical_axes(indep_var_choices)

    # The canonical axes here are A: -w1 and B: w2
    assert len(axes_cfgs.axes) == 2
    assert axes_cfgs.axes[0].label == 'A'
    assert axes_cfgs.axes[0].var_set.var_set[0].pulse_refs == (-1,)
    assert axes_cfgs.axes[1].label == 'B'
    assert axes_cfgs.axes[1].var_set.var_set[0].pulse_refs == (2,)


    # EVV with all pulses at different times
    pulse_a = EmPulse(env='impulsive', maxstr=1e-05, tc=10.0, cf=0.0, cf_uv=0.0, id=1)
    pulse_b = EmPulse(env='impulsive', maxstr=1e-05, tc=20.0, cf=0.0, cf_uv=0.0, id=2)
    pulse_c = EmPulse(env='impulsive', maxstr=1e-05, tc=30.0, cf=0.0, cf_uv=0.144, id=3)

    pulses = [pulse_a, pulse_b, pulse_c]
    epochs = [[1], [2], [3]]

    # Here two phase-matching directions
    pm_dir_a = PhaseMatchingCondition(SignedPulseTuple(pulse_refs=(-1, 2, 3)), 1)
    pm_dir_b = PhaseMatchingCondition(SignedPulseTuple(pulse_refs=(1, -2, 3)), 2)

    indep_var_choices = find_indep_exp_variables(pulses, epochs, [pm_dir_a, pm_dir_b])

    # Should fail because > 1 phase-matching directions not supported yet
    # NOTE: However, this case is an example of a situation where canonical axes could in fact be identified:
    # Canonical axes for pm_dir_a are identified as A: -w1 and B: w2, while for pm_dir_b, they are
    # identified as A: w1 and B: -w2 (i.e. just overall opposite sign). Future implementation should support this
    with pytest.raises(ValueError):
        axes_cfgs = find_canonical_axes(indep_var_choices)



