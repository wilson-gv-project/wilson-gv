import pytest

from wilson_suite import fixtures as ws_fixtures

from wilson_suite.wilson_experiment.experiment_abstractions import (SpecDetector, SpecScan, EmPulse,
                        ElectricField, VibExperiment, get_carrier_freqs_uv, find_epochs, uv_cancels)

def test_spec_detector:

    pass


def test_spec_scan:

    pass

def test_em_pulse:

    pass

def test_electric_field:

    pass

def test_vib_experiment:

    pass

def test_get_carrier_freqs_uv:

    pass

def test_find_epochs:

    pass

def test_uv_cancels:

    pass

def test_dummy():

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
