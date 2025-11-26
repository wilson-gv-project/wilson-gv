import pytest

from wilson_suite import fixtures as ws_fixtures


from wilson_suite.wilson_experiment.indep_vars_and_axes import (SignedPulseTuple, PhaseMatchingCondition,
                IndependentVariableSet, IndependentVariableChoices, SpectralAxis, SpectralAxisSet, SpectralAxisChoices,
                find_subsets_making_orig, find_branching_indep_var_combs, find_indep_vars_for_one_phasematch,
                find_indep_exp_variables, find_axes_recursion, find_valid_axes_cfgs_for_one_phasematch,
                find_canonical_axes_for_one_phasematch, find_canonical_axes, find_valid_axes)


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

    evv_exp = ws_fixtures.evv_experiment()

    evv_exp = ws_fixtures.experiment_beta_alpha_cars()

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