from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import (
                                                                      build_machinery_from_experiment,
                                                                      build_machinery_from_terms,
                                                                      build_qc_context
                                                                      )
import wilson_suite as ws

def test_build_machinery_from_experiment():
    experiment = ws.fixtures.evv_experiment()
    evv_machine = build_machinery_from_experiment(experiment, 
                                                  axes_choice_dict={"A": [1], "B": [-1, 2]})
    qc_data = build_qc_context()
    
    evv_machine.feed_data(qc_data, gamma=10.)


def test_build_machinery_from_terms():
    
    terms = []
    evv_machine = build_machinery_from_terms(terms,
                                             axes_choice_dict={"A": [1], "B": [-1, 2]},
                                             polarization_avg_vector=[1.,1.,1.],
                                             magn_conditions=())

    qc_data = build_qc_context()
    
    evv_machine.feed_data(qc_data, gamma=10.)


