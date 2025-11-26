from ...wilson_experiment import experiment_abstractions as we_abst
from ...wilson_utils.serialization import check_if_jsonsafe
from dataclasses import asdict

def test_dict_EmPulse():
    p1 = we_abst.EmPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, 
                                         wv=[0.0, 0.0, 1.0], 
                                         pol=[0.0, 0.0, 1.0], id=1)
    p1_d = asdict(p1)
    p1_dict_ref = {'env': 'ideal', 'maxstr': 1e-05, 'tc': 50.0, 'cf': 0.0, 'cf_uv': 0.0, 
               'dev': None, 'wv': [0.0, 0.0, 1.0], 'pol': [0.0, 0.0, 1.0], 'id': 1}
    assert p1_d == p1_dict_ref
    p1_fromdict = we_abst.EmPulse(**p1_d)
    assert p1.cf == p1_fromdict.cf
    assert p1.wv == p1_fromdict.wv
    assert p1_fromdict.env == 'ideal'
    assert p1_fromdict.cf == 0.00
    assert p1_fromdict.tc == 50.0
    assert p1_fromdict.wv == [0.0, 0.0, 1.0]
    assert p1_fromdict.pol == [0.0, 0.0, 1.0]

    pulse_uvvis_1 = we_abst.EmPulse('ideal', 1.0e-5, tc = 120.0, cf=0.0, 
                                            cf_uv=0.072, wv=[0.0, 0.0, 1.0], 
                                            pol=[0.0, 0.0, 1.0], id=3)    
    p2 = asdict(pulse_uvvis_1)
    p2_dict_ref = {'env': 'ideal', 'maxstr': 1e-05, 'tc': 120.0, 'cf': 0.0, 'cf_uv': 0.072, 
                   'dev': None, 'wv': [0.0, 0.0, 1.0], 'pol': [0.0, 0.0, 1.0], 'id': 3}
    assert p2 == p2_dict_ref


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

    p1_dict = {'env': 'ideal', 'maxstr': 1e-05, 'tc': 50.0, 'cf': 0.0, 'cf_uv': 0.0, 'dev': None, 'wv': [0.0, 0.0, 1.0], 'pol': [0.0, 0.0, 1.0], 'id': 1}
    p2_dict = {'env': 'impulsive', 'maxstr': 1e-05, 'tc': 100.0, 'cf': None, 'cf_uv': 0.0, 'dev': None, 'wv': [0.0, 0.0, 1.0], 'pol': [0.0, 0.0, 1.0], 'id': 2}
    p3_dict = {'env': 'ideal', 'maxstr': 1e-05, 'tc': 120.0, 'cf': 0.0, 'cf_uv': 0.072, 'dev': None, 'wv': [0.0, 0.0, 1.0], 'pol': [0.0, 0.0, 1.0], 'id': 3}
    field_a_dict = {'pulses': [p1_dict, p2_dict, p3_dict]}

    assert asdict(field_a) == field_a_dict
    
    field_a_dict_dc = {'pulses': [we_abst.EmPulse(**p1_dict), we_abst.EmPulse(**p2_dict), we_abst.EmPulse(**p3_dict)]}
    EF_fromdict = we_abst.ElectricField(**field_a_dict_dc)

    assert EF_fromdict.pulses[0].tc == 50.0
    assert EF_fromdict.pulses[0].maxstr == 1e-05
    assert EF_fromdict.pulses[2].tc == 120.0


def test_dict_SpecDetector():
    detector_a = we_abst.SpecDetector('freq', detector_location=[0.0, 0.0, 1.0],
                                              detection_polarization=[0.0, 0.0, 1.0],
                                              detection_range=[0.003 + 0.0001*i for i in range(10)],
                                              wv_filter=[{1: [-1], 2: [1], 3: [1]}]) #, {1: [-1], 2: [1], 3: [1]}
    dd = {'detection_method': 'freq', 'detector_location': [0.0, 0.0, 1.0], 'detection_polarization': [0.0, 0.0, 1.0], 
          'detection_range': [0.003, 0.0031, 0.0032, 0.0033, 0.0034000000000000002, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039000000000000003], 
          'wv_filter': [{1: [-1], 2: [1], 3: [1]}], 'ignore_collinear': True}
    assert asdict(detector_a) == dd
    assert asdict(detector_a)['wv_filter'] == [{1: [-1], 2: [1], 3: [1]}]
    assert asdict(detector_a)['wv_filter'] == detector_a.wv_filter


def test_dict_SpecScan():
    scan_obj_a = [['pulse', 1, 'cf', 1.0], ['detector', 0, 'detection_range', 1.0]]
    scan_range_a = [0.0001*i for i in range(10)]
    scan_a = we_abst.SpecScan(scan_obj_a, scan_range_a)

    print(asdict(scan_a))

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
