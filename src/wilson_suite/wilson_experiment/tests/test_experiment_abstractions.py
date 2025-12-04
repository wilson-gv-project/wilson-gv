import pytest

from wilson_suite import fixtures as ws_fixtures

from wilson_suite.wilson_experiment.experiment_abstractions import (SpecDetector, ScanObject, SpecScan,
                    EmPulse, ElectricField, VibExperiment, get_carrier_freqs_uv, find_epochs, uv_cancels)

def test_spec_detector():

    # Detector suitable for EVV:
    #   - Detector is set to capture a spectrum ('freq')
    #   - Located at laboratory coordinates (0.0, 0.0, 1.0)
    #       - This keyword currently not in use
    #   - Set to detect light linearly polarized along the x-axis
    #   - Detection range (currently not in use) here set as the range (0.003, 0.013) a.u. in 101 equidistant spacings
    #   - "Wavevector filter" (for selecting phase-matching conditions) here -k1 + k2 +k3

    detector_a = SpecDetector(detection_method='freq',
                             detector_location=(0.0, 0.0, 1.0),
                             detection_polarization=(1.0, 0.0, 0.0),
                             detection_range=[0.003 + 0.0001 * i for i in range(101)],
                             wv_filter=[{1: -1, 2: 1, 3: 1}])

    assert detector_a.detection_method == 'freq'
    assert detector_a.detector_location == (0.0, 0.0, 1.0)
    assert detector_a.detection_polarization == (1.0, 0.0, 0.0)
    assert detector_a.detection_range == [0.003 + 0.0001 * i for i in range(101)]
    assert detector_a.wv_filter[0][1] == -1
    assert detector_a.wv_filter[0][2] == 1
    assert detector_a.wv_filter[0][3] == 1

    # Asserting default values of optional parameters not included
    assert detector_a.ignore_collinear
    assert detector_a.overall_phase == 1.0 + 0.0j

    # Unrecognized detection method
    with pytest.raises(ValueError):
        detector_bogus = SpecDetector(detection_method='bogus',
                                      detection_range=[0.003 + 0.0001 * i for i in range(101)])

    # Unsupported overall phase
    with pytest.raises(ValueError):
        detector_bogus = SpecDetector(detection_method='freq',
                                      overall_phase=3.0 + 2.0j)


def test_scan_object():

    # Here: A scan object referring to the carrier frequency of pulse 1, to be scanned at twice the
    # rapidity of range increments
    obj_a = ScanObject('pulse', 'cf', id=1, coeff=2.0)

    assert obj_a.category == 'pulse'
    assert obj_a.subcategory == 'cf'
    assert obj_a.id == 1
    assert obj_a.coeff == 2.0

    # Referring to the detection range of the detector, testing that optional parameters not included
    # have default values
    obj_b = ScanObject('detector', 'detection_range')
    assert obj_b.id == 0
    assert obj_b.coeff == 1.0

    # Unrecognized scan category
    with pytest.raises(ValueError):
        obj_bogus = ScanObject('bogus', 'cf')

    # Recognized scan category but unrecognized scan subcategory
    with pytest.raises(ValueError):
        obj_bogus = ScanObject('detector', 'bogus')

    # Recognized scan category and scan category that is a valid choice for a different category but not this one
    with pytest.raises(ValueError):
        obj_bogus = ScanObject('detector', 'cf')

def test_spec_scan():

    # Making a scan consisting of two objects:
    #   - Scanning pulse 1 carrier frequency by twice the increments of the range
    #   - Scanning the detector's detection range at the exact increments of the range (default coeff = 1.0)
    scan_obj_a = ScanObject('pulse', 'cf', id=1, coeff=2.0)
    scan_obj_b = ScanObject('detector', 'detection_range')
    scan_range = [0.0001 * i for i in range(101)]

    scan_objs = (scan_obj_a, scan_obj_b)

    scan_a = SpecScan(scan_objs, scan_range)

    assert scan_a.scan_objs[0].category == 'pulse'
    assert scan_a.scan_objs[0].subcategory == 'cf'
    assert scan_a.scan_objs[0].id == 1
    assert scan_a.scan_objs[0].coeff == 2.0

    assert scan_a.scan_objs[1].category == 'detector'
    assert scan_a.scan_objs[1].subcategory == 'detection_range'
    assert scan_a.scan_objs[1].id == 0
    assert scan_a.scan_objs[1].coeff == 1.0

    assert scan_a.range == [0.0001 * i for i in range(101)]

    # Scan objects not tuple
    with pytest.raises(TypeError):
        scan_objs = [scan_obj_a, scan_obj_b]
        scan_bogus = SpecScan(scan_objs, scan_range)

    # Non-ScanObject entry
    with pytest.raises(TypeError):
        scan_objs = ('bogus', scan_obj_b)
        scan_bogus = SpecScan(scan_objs, scan_range)

    # Non-iterable
    with pytest.raises(TypeError):
        scan_objs = (scan_obj_a, scan_obj_b)
        scan_bogus = SpecScan(scan_objs, 2.0)

def test_em_pulse():

    # An infrared pulse from the EVV experiment
    pulse_a = EmPulse(env='gaussian', maxstr=1.0e-5, tc = 100.0, cf_uv=0.0,
                      wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0), id=1)

    # A UV/VIS pulse from the EVV experiment
    pulse_b = EmPulse(env='gaussian', maxstr=1.0e-5, tc = 120.0, cf_uv=0.072,
                      wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0), id=3)


    pass

def test_electric_field():

    pass

def test_vib_experiment():

    pass

def test_get_carrier_freqs_uv():

    pass

def test_find_epochs():

    pass

def test_uv_cancels():

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
