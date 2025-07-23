"""
check_if_jsonsafe - is it a format that can be written in JSON file with json library. 
Generally, it would also mean that the object is serialized into a dictionary here.
"""
import wilson_experiment.abstractions_dataclasses as we_abst_dataclass
import wilson_experiment.abstractions as we_abst
from wilson_utils.serialization import check_if_jsonsafe
from dataclasses import asdict

def test_SpecDetector():
    detector_reg = we_abst.SpecDetector('freq', detector_location=[0.0, 0.0, 1.0],
                                              detection_polarization=[0.0, 0.0, 1.0],
                                              detection_range=[0.003 + 0.0001*i for i in range(101)],
                                              wv_filter=[{1: [-1], 2: [1], 3: [1]}])
    assert check_if_jsonsafe(detector_reg.to_dict())

    detector_datacls = we_abst_dataclass.SpecDetector('freq', detector_location=[0.0, 0.0, 1.0],
                                            detection_polarization=[0.0, 0.0, 1.0],
                                            detection_range=[0.003 + 0.0001*i for i in range(101)],
                                            wv_filter=[{1: [-1], 2: [1], 3: [1]}])
    assert check_if_jsonsafe(asdict(detector_datacls))

def test_SpecScan():
    scan_obj = [['pulse', 1, 'cf', 1.0], ['detector', 0, 'detection_range', 1.0]]
    scan_range = [0.0001*i for i in range(101)]
    
    scan_reg = we_abst.SpecScan(scan_obj, scan_range)
    assert check_if_jsonsafe(scan_reg.to_dict())

    scan_datacls = we_abst_dataclass.SpecScan(scan_obj, scan_range)
    assert check_if_jsonsafe(asdict(scan_datacls))

def test_EmPulse():
    pulse_ir_1_reg = we_abst.EmPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, 
                                         wv=[0.0, 0.0, 1.0], 
                                         pol=[0.0, 0.0, 1.0], id=1)
    assert check_if_jsonsafe(pulse_ir_1_reg.to_dict())

    pulse_uvvis_1_reg = we_abst.EmPulse('ideal', 1.0e-5, tc = 120.0, cf=0.0, 
                                            cf_uv=0.072, wv=[0.0, 0.0, 1.0], 
                                            pol=[0.0, 0.0, 1.0], id=3)
    assert check_if_jsonsafe(pulse_uvvis_1_reg.to_dict())
    
    pulse_ir_1_datacls = we_abst_dataclass.EmPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, 
                                         wv=[0.0, 0.0, 1.0], 
                                         pol=[0.0, 0.0, 1.0], id=1)
    assert check_if_jsonsafe(asdict(pulse_ir_1_datacls))

    pulse_uvvis_1_datacls = we_abst_dataclass.EmPulse('ideal', 1.0e-5, tc = 120.0, cf=0.0, 
                                            cf_uv=0.072, wv=[0.0, 0.0, 1.0], 
                                            pol=[0.0, 0.0, 1.0], id=3)
    assert check_if_jsonsafe(asdict(pulse_uvvis_1_datacls))

def test_ElectricField():
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
    field_reg = we_abst.ElectricField(pulses)
    assert check_if_jsonsafe(field_reg.to_dict())


    pulse_ir_1_d = we_abst_dataclass.EmPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, 
                                         wv=[0.0, 0.0, 1.0], 
                                         pol=[0.0, 0.0, 1.0], id=1)
    pulse_ir_2_d = we_abst_dataclass.EmPulse('impulsive', 1.0e-5, tc = 100.0, cf=None, 
                                         wv=[0.0, 0.0, 1.0], 
                                         pol=[0.0, 0.0, 1.0], id=2)
    pulse_uvvis_1_d = we_abst_dataclass.EmPulse('ideal', 1.0e-5, tc = 120.0, cf=0.0, 
                                            cf_uv=0.072, wv=[0.0, 0.0, 1.0], 
                                            pol=[0.0, 0.0, 1.0], id=3)
    pulses_d = [pulse_ir_1_d, pulse_ir_2_d, pulse_uvvis_1_d]
    field_datacls = we_abst_dataclass.ElectricField(pulses_d)
    assert check_if_jsonsafe(asdict(field_datacls))

def test_VibExperiment():
    pulse_ir_1 = we_abst.EmPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=1)
    pulse_ir_2 = we_abst.EmPulse('impulsive', 1.0e-5, tc = 100.0, cf=None, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=2)
    pulse_uvvis_1 = we_abst.EmPulse('ideal', 1.0e-5, tc = 120.0, cf=0.0, cf_uv=0.072, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=3)
    pulses = [pulse_ir_1, pulse_ir_2, pulse_uvvis_1]
    field_a = we_abst.ElectricField(pulses)
    order = len(pulses)
    field_a.findEpochs()
    detector_a = we_abst.SpecDetector('freq', detector_location=[0.0, 0.0, 1.0],
                                      detection_polarization=[0.0, 0.0, 1.0],
                                      detection_range=[0.003 + 0.0001*i for i in range(101)],
                                      wv_filter=[{1: [-1], 2: [1], 3: [1]}])
    scan_obj_a = [['pulse', 1, 'cf', 1.0], ['detector', 0, 'detection_range', 1.0]]
    scan_range_a = [0.0001*i for i in range(101)]
    scan_a = we_abst.SpecScan(scan_obj_a, scan_range_a)
    experiment_a_reg = we_abst.VibExperiment(order, field_a, detector_a, [scan_a], magn_conditions=[[-1, 2]])
    # at this point gave up on custom to_dict() methods, too much trouble
    assert not check_if_jsonsafe(experiment_a_reg.to_dict())

    pulse_ir_1_d = we_abst_dataclass.EmPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=1)
    pulse_ir_2_d = we_abst_dataclass.EmPulse('impulsive', 1.0e-5, tc = 100.0, cf=None, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=2)
    pulse_uvvis_1_d = we_abst_dataclass.EmPulse('ideal', 1.0e-5, tc = 120.0, cf=0.0, cf_uv=0.072, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=3)
    pulses_d = [pulse_ir_1_d, pulse_ir_2_d, pulse_uvvis_1_d]
    field_a_d = we_abst_dataclass.ElectricField(pulses_d)
    order_d = len(pulses_d)
    field_a_d.findEpochs()
    detector_a_d = we_abst_dataclass.SpecDetector('freq', detector_location=[0.0, 0.0, 1.0],
                                      detection_polarization=[0.0, 0.0, 1.0],
                                      detection_range=[0.003 + 0.0001*i for i in range(101)],
                                      wv_filter=[{1: [-1], 2: [1], 3: [1]}])
    scan_obj_a_d = [['pulse', 1, 'cf', 1.0], ['detector', 0, 'detection_range', 1.0]]
    scan_range_a_d = [0.0001*i for i in range(101)]
    scan_a_d = we_abst_dataclass.SpecScan(scan_obj_a_d, scan_range_a_d)

    experiment_a_datacls = we_abst_dataclass.VibExperiment(order_d, field_a_d, detector_a_d, [scan_a_d], magn_conditions=[[-1, 2]])
    assert check_if_jsonsafe(asdict(experiment_a_datacls))

    from wilson_intensities.wilson.utils import get_package_root
    ws_root = get_package_root() + '/../../'

    import pickle
    with open(ws_root+"/wilson_experiment/tests/experiment_a_datacls.pkl", "wb") as f:
        pickle.dump(experiment_a_datacls, f)

    with open(ws_root+"/wilson_experiment/tests/experiment_a_datacls.pkl", "rb") as f:
        loaded_experiment_a_datacls = pickle.load(f)

    assert experiment_a_datacls.scans == loaded_experiment_a_datacls.scans
    assert experiment_a_datacls.detector == loaded_experiment_a_datacls.detector
    assert experiment_a_datacls.field == loaded_experiment_a_datacls.field
    assert experiment_a_datacls.order == loaded_experiment_a_datacls.order

    assert loaded_experiment_a_datacls.findInteractionSequences() == [[{1: -1}, {2: 1}, {3: 1}]]

    assert hasattr(loaded_experiment_a_datacls, 'tellDimensions')
    assert hasattr(loaded_experiment_a_datacls, 'findInteractionSequences')