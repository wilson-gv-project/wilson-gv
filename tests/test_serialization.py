import wilson_experiment.abstractions as we_abst
from wilson_utils.serialization import check_if_jsonsafe

def test_dict_EmPulse():
    print()
    pulse_ir_1 = we_abst.EmPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, 
                                         wv=[0.0, 0.0, 1.0], 
                                         pol=[0.0, 0.0, 1.0], id=1)
    p1_d = pulse_ir_1.to_dict()
    p1_dict = {'env': 'ideal', 'tc': 50.0, 'maxstr': 1e-05, 'cf_uv': 0.0, 
                  'wv': [0.0, 0.0, 1.0], 'pol': [0.0, 0.0, 1.0], 'id': 1, 'cf': 0.0}
    assert p1_d == p1_dict
    p1_fromdict = we_abst.EmPulse.from_dict(p1_dict)
    assert pulse_ir_1.cf == p1_fromdict.cf
    assert pulse_ir_1.wv == p1_fromdict.wv
    assert p1_fromdict.env == 'ideal'
    assert p1_fromdict.cf == 0.00
    assert p1_fromdict.tc == 50.0
    assert p1_fromdict.wv == [0.0, 0.0, 1.0]
    assert p1_fromdict.pol == [0.0, 0.0, 1.0]

    pulse_uvvis_1 = we_abst.EmPulse('ideal', 1.0e-5, tc = 120.0, cf=0.0, 
                                            cf_uv=0.072, wv=[0.0, 0.0, 1.0], 
                                            pol=[0.0, 0.0, 1.0], id=3)    
    p2 = pulse_uvvis_1.to_dict()
    assert p2 == {'env': 'ideal', 'tc': 120.0, 'maxstr': 1e-05, 'cf_uv': 0.072, 
                  'wv': [0.0, 0.0, 1.0], 'pol': [0.0, 0.0, 1.0], 'id': 3, 'cf': 0.0}


def test_dict_ElectricField():
    print()
    pulse_ir_1 = we_abst.EmPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, 
                                         wv=[0.0, 0.0, 1.0], 
                                         pol=[0.0, 0.0, 1.0], id=1)
    pulse_ir_2 = we_abst.EmPulse('impulsive', 1.0e-5, tc = 100.0, cf=None, 
                                         wv=[0.0, 0.0, 1.0], 
                                         pol=[0.0, 0.0, 1.0], id=2)
    pulse_uvvis_1 = we_abst.EmPulse('ideal', 1.0e-5, tc = 120.0, cf=0.0, 
                                            cf_uv=0.072, wv=[0.0, 0.0, 1.0], 
                                            pol=[0.0, 0.0, 1.0], id=3)
    pulses = [pulse_ir_1, pulse_ir_2, pulse_uvvis_1]
    field_a = we_abst.ElectricField(pulses)

    field_a_dict = {'pulses': [{'env': 'ideal', 'tc': 50.0, 'maxstr': 1e-05, 
                                                'cf_uv': 0.0, 'wv': [0.0, 0.0, 1.0], 
                                                'pol': [0.0, 0.0, 1.0], 'id': 1, 'cf': 0.0}, 
                                            {'env': 'impulsive', 'tc': 100.0, 'maxstr': 1e-05, 
                                                'cf_uv': 0.0, 'wv': [0.0, 0.0, 1.0], 
                                                'pol': [0.0, 0.0, 1.0], 'id': 2}, 
                                            {'env': 'ideal', 'tc': 120.0, 'maxstr': 1e-05, 
                                                'cf_uv': 0.072, 'wv': [0.0, 0.0, 1.0], 
                                                'pol': [0.0, 0.0, 1.0], 'id': 3, 'cf': 0.0}]}
    assert field_a.to_dict() == field_a_dict

    EF_fromdict = we_abst.ElectricField.from_dict(field_a_dict)
    assert EF_fromdict.pulses[0].tc == 50.0
    assert EF_fromdict.pulses[0].maxstr == 1e-05
    assert EF_fromdict.pulses[2].tc == 120.0


def test_dict_SpecDetector():
    detector_a = we_abst.SpecDetector('freq', detector_location=[0.0, 0.0, 1.0],
                                              detection_polarization=[0.0, 0.0, 1.0],
                                              detection_range=[0.003 + 0.0001*i for i in range(101)],
                                              wv_filter=[{1: [-1], 2: [1], 3: [1]}]) #, {1: [-1], 2: [1], 3: [1]}
    
    dd = {'detection_method': 'freq', 'detector_location': [0.0, 0.0, 1.0], 'detection_polarization': [0.0, 0.0, 1.0], 'wv_filter': [{1: [-1], 2: [1], 3: [1]}], 'ignore_collinear': True, 'detection_range': [0.003, 0.0031, 0.0032, 0.0033, 0.0034000000000000002, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039000000000000003, 0.004, 0.0041, 0.004200000000000001, 0.0043, 0.0044, 0.0045000000000000005, 0.0046, 0.0047, 0.0048000000000000004, 0.0049, 0.005, 0.0051, 0.0052, 0.0053, 0.0054, 0.0055, 0.005600000000000001, 0.0057, 0.0058, 0.005900000000000001, 0.006, 0.0061, 0.006200000000000001, 0.0063, 0.0064, 0.006500000000000001, 0.0066, 0.0067, 0.0068000000000000005, 0.0069, 0.007, 0.0071, 0.007200000000000001, 0.0073, 0.0074, 0.007500000000000001, 0.0076, 0.0077, 0.0078000000000000005, 0.0079, 0.008, 0.0081, 0.0082, 0.0083, 0.008400000000000001, 0.0085, 0.0086, 0.0087, 0.0088, 0.0089, 0.009000000000000001, 0.0091, 0.0092, 0.0093, 0.0094, 0.009500000000000001, 0.009600000000000001, 0.0097, 0.0098, 0.0099, 0.01, 0.010100000000000001, 0.0102, 0.0103, 0.0104, 0.0105, 0.0106, 0.010700000000000001, 0.0108, 0.0109, 0.011, 0.011099999999999999, 0.011200000000000002, 0.011300000000000001, 0.0114, 0.0115, 0.0116, 0.011700000000000002, 0.011800000000000001, 0.0119, 0.012, 0.0121, 0.012199999999999999, 0.012300000000000002, 0.012400000000000001, 0.0125, 0.0126, 0.0127, 0.012799999999999999, 0.012900000000000002, 0.013000000000000001]}

    assert detector_a.to_dict() == dd
    assert detector_a.to_dict()['wv_filter'] == [{1: [-1], 2: [1], 3: [1]}]
    assert detector_a.to_dict()['wv_filter'] == detector_a.wv_filter


def test_dict_SpecScan():
    scan_obj_a = [['pulse', 1, 'cf', 1.0], ['detector', 0, 'detection_range', 1.0]]
    scan_range_a = [0.0001*i for i in range(101)]
    scan_a = we_abst.SpecScan(scan_obj_a, scan_range_a)

    print(scan_a)

def test_dict_VibExperiment():

    pulse_ir_1 = we_abst.EmPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=1)
    pulse_ir_2 = we_abst.EmPulse('impulsive', 1.0e-5, tc = 100.0, cf=None, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=2)
    pulse_uvvis_1 = we_abst.EmPulse('ideal', 1.0e-5, tc = 120.0, cf=0.0, cf_uv=0.072, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=3)

    pulses = [pulse_ir_1, pulse_ir_2, pulse_uvvis_1]

    field_a = we_abst.ElectricField(pulses)
    order = len(pulses)

    epochs = field_a.findEpochs()

    detector_a = we_abst.SpecDetector('freq', detector_location=[0.0, 0.0, 1.0],
                                      detection_polarization=[0.0, 0.0, 1.0],
                                      detection_range=[0.003 + 0.0001*i for i in range(101)],
                                      wv_filter=[{1: [-1], 2: [1], 3: [1]}]) #, {1: [-1], 2: [1], 3: [1]}

    # Push one carrier freq
    scan_obj_a = [['pulse', 1, 'cf', 1.0], ['detector', 0, 'detection_range', 1.0]]
    scan_range_a = [0.0001*i for i in range(101)]
    scan_a = we_abst.SpecScan(scan_obj_a, scan_range_a)

    experiment_a = we_abst.VibExperiment(order, field_a, detector_a, [scan_a], magn_conditions=[[-1, 2]])
    print('\nexperiment_a.__dict__', experiment_a.__dict__, '\n')

    pass
