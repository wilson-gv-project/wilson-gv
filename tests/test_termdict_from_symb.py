import pytest

def setup_evv_terms_for_test():

    import wilson_suite as ws

    pulse_ir_1 = ws.experiment.abstractions.emPulse('ideal', 1.0e-5, tc=50.0, cf=0.00, wv=[0.0, 0.0, 1.0],
                                                    pol=[0.0, 0.0, 1.0], id=1)
    pulse_ir_2 = ws.experiment.abstractions.emPulse('impulsive', 1.0e-5, tc=100.0, cf=None, wv=[0.0, 0.0, 1.0],
                                                    pol=[0.0, 0.0, 1.0], id=2)
    pulse_uvvis_1 = ws.experiment.abstractions.emPulse('ideal', 1.0e-5, tc=120.0, cf=0.0, cf_uv=0.072,
                                                       wv=[0.0, 0.0, 1.0],
                                                       pol=[0.0, 0.0, 1.0], id=3)

    pulses = [pulse_ir_1, pulse_ir_2, pulse_uvvis_1]

    field_a = ws.experiment.abstractions.electricField(pulses)
    order = len(pulses)

    epochs = field_a.findEpochs()

    detector_a = ws.experiment.abstractions.specDetector('freq', detector_location=[0.0, 0.0, 1.0],
                                                         detection_polarization=[0.0, 0.0, 1.0],
                                                         detection_range=[0.003 + 0.0001 * i for i in range(101)],
                                                         wv_filter=[
                                                             {1: [-1], 2: [1], 3: [1]}])  # , {1: [-1], 2: [1], 3: [1]}

    # Push one carrier freq
    scan_obj_a = [['pulse', 1, 'cf', 1.0], ['detector', 0, 'detection_range', 1.0]]
    scan_range_a = [0.0001 * i for i in range(101)]
    scan_a = ws.experiment.abstractions.specScan(scan_obj_a, scan_range_a)

    experiment_a = ws.experiment.abstractions.vibExperiment(order, field_a, detector_a, [scan_a],
                                                            magn_conditions=[[-1, 2]])

    fully_enhanced_terms = ws.derive.main.get_fully_enhanced_terms(experiment_a)

    return fully_enhanced_terms


def test_dict_from_term():

    import wilson_suite as ws
    from fractions import Fraction

    my_terms = setup_evv_terms_for_test()

    reference_dict_a = {'termA_pref': Fraction(1, 4),
                        'termB_pref': 1.0,
                        'averaged_props': (('dipgrad', ('a',), ('B',)), ('dipgrad', ('b',), ('G',)), ('polhess', ('a', 'b'), ('A', 'D'))),
                        'non_averaged_props': None,
                        'vibene_denom': ('a', 'b'),
                        'vibenediff': None,
                        'resonances': (('zero,a', (-1,)), ('b,a', (-1, 2)))
                        }

    reference_dict_b = {'termA_pref': Fraction(-1, 16),
                        'termB_pref': 1.0,
                        'averaged_props': (('polgrad', ('a',), ('A', 'D')), ('dipgrad', ('b',), ('B',)), ('dipgrad', ('a',), ('G',))),
                        'non_averaged_props': (('cff', ('b', 'c', 'c')),),
                        'vibene_denom': ('a', 'b', 'c'),
                        'vibenediff': ('a+b,a',),
                        'resonances': (('zero,b', (-1,)), ('a,b', (-1, 2)))
                        }

    # Choosing a reasonably representative sample
    #my_terms[1][(1,0)][0].present()
    result_dict_a = ws.utils.termdict_from_symb_term.dict_from_term(my_terms[1][(1,0)][0])

    #my_terms[1][(0, 1)][5].present()
    result_dict_b = ws.utils.termdict_from_symb_term.dict_from_term(my_terms[1][(0, 1)][5])

    for i in reference_dict_a:
        try:
            assert reference_dict_a[i] == result_dict_a[i]
        except AssertionError:
            print(reference_dict_a[i], 'does not match', result_dict_a[i])

    for i in reference_dict_b:
        try:
            assert reference_dict_b[i] == result_dict_b[i]
        except AssertionError:
            print(reference_dict_b[i], 'does not match', result_dict_b[i])

    return