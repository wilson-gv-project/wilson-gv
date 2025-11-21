from wilson_suite import fixtures as ws_fixtures
from wilson_suite.wilson_experiment.abstractions import PhaseMatchingCondition, IndependentVariableChoiceSet, AxisChoiceSet


def test_signed_pulse_tuple():

    pass

def test_phase_matching_condition():

    phasematch_id = 0

    # -k1 + k2 + k3
    pulse_id_signs_m1p2p3 = {1: -1, 2: 1, 3: 1}

    phasematch_m1p2p3 = PhaseMatchingCondition(pulse_id_signs_m1p2p3, phasematch_id)

    assert phasematch_m1p2p3.phasematch_cond_id == 0

    i = 0
    for pulse in phasematch_m1p2p3.pulse_id_signs:

        if i == 0:
            assert pulse == 1
            assert phasematch_m1p2p3.pulse_id_signs[pulse] == -1

        if i == 1:
            assert pulse == 2
            assert phasematch_m1p2p3.pulse_id_signs[pulse] == 1

        if i == 2:
            assert pulse == 3
            assert phasematch_m1p2p3.pulse_id_signs[pulse] == 1

        i += 1


def test_independent_variable_set():

    pass

def test_independent_variable_choices():

    pass


def test_spectral_axis():

    pass

def test_spectral_axis_choice():

    pass

def test_spectral_axis_choice_set():

    pass


def test_find_subsets_making_orig():

    evv_exp = ws_fixtures.evv_experiment_pulse_1_and_2_coincident()

    print('relevant phase-matching conditions', evv_exp.relevant_phasematch)
    print('interaction sequences', evv_exp.int_sequences)
    print('independent variables', evv_exp.indep_vars)
    print('valid axis combinations', evv_exp.valid_axis_combs)
    print('canonical axes', evv_exp.canonical_axes)

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

    pass