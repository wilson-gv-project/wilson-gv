import pytest

from wilson_suite import fixtures as ws_fixtures

from wilson_suite.wilson_experiment.experiment_abstractions import (SpecDetector, ScanObject, SpecScan,
                    EmPulse, make_gaussian_pulse, make_impulsive_gaussian_pulse, make_cw_gaussian_pulse,
                    ElectricField, VibExperiment, get_carrier_freqs_uv, find_epochs, uv_cancels)

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

    # An infrared-range pulse with:
    # - Envelope strength maximum of 1e-5 a.u.
    # - Time envelope centerpoint at 100 a.u.
    # - Time envelope deviation parameter 2.0 a.u.
    # - Carrier frequency 0.005 a.u
    # - No UV/VIS component of carrier frequency
    # - Wavevector (0.0, 1.0, 0.0) (laboratory axes)
    # - Polarization vector (0.0, 0.0, 1.0) (laboratory axes)
    # - No overall phase shift (here replicating default)
    # - (Integer) identifier label: 1
    pulse_a = EmPulse(env='gaussian', tc = 100.0, cf = 0.005, dev = 2.0, cf_uv=0.0, maxstr=1.0e-5,
                      wv=(0.0, 1.0, 0.0), pol=(0.0, 0.0, 1.0), overall_phase = 1.0 + 0.0j, id=1)

    assert pulse_a.env == 'gaussian'
    assert pulse_a.tc == 100.0
    assert pulse_a.cf == 0.005
    assert pulse_a.dev == 2.0
    assert pulse_a.cf_uv == 0.0
    assert pulse_a.maxstr == 1.0e-5
    assert pulse_a.wv == (0.0, 1.0, 0.0)
    assert pulse_a.pol == (0.0, 0.0, 1.0)
    assert pulse_a.overall_phase == 1.0 + 0.0j
    assert pulse_a.id == 1

    # The choice of deviation parameter is non-limiting and should therefore result in the pulse being
    # neither impulsive-tending or continuous-wave-tending
    assert not(pulse_a.tendsImpulsive())
    assert not(pulse_a.tendsContinuous())

    # A UV/VIS impulsive-tending pulse. Leaving several parameters to their default values.
    # For impulsive-tending pulses, the carrier frequency argument is optional
    pulse_b = EmPulse(env='gaussian', tc = 120.0, dev=0.0, cf_uv = 0.072)
    assert pulse_b.env == 'gaussian'
    assert pulse_b.tc == 120.0
    assert pulse_b.cf == None # Default value
    assert pulse_b.dev == 0.0
    assert pulse_b.cf_uv == 0.072
    assert pulse_b.maxstr == 0.0 # Default
    assert pulse_b.wv == (0.0, 0.0, 1.0) # Default
    assert pulse_b.pol == (1.0, 0.0, 0.0) # Default
    assert pulse_b.overall_phase == 1.0 + 0.0j # Default
    assert pulse_b.id == None # Default

    assert pulse_b.tendsImpulsive()
    assert not(pulse_b.tendsContinuous())

    # A CW-tending IR range pulse. For continuous-wave-tending
    # pulses, the time centerpoint argument is optional
    from math import inf as infinity
    pulse_c = EmPulse(env='gaussian', cf = 0.003, dev = infinity)
    assert pulse_c.env == 'gaussian'
    assert pulse_c.tc == None # Default value
    assert pulse_c.cf == 0.003
    assert pulse_c.dev == infinity

    assert not(pulse_c.tendsImpulsive())
    assert pulse_c.tendsContinuous()

    # Unrecognized pulse envelope
    with pytest.raises((ValueError)):
        pulse_bogus = EmPulse(env='bogus', tc=120.0, dev=0.0, cf_uv=0.072)

    # Non-zero cf parameter with non-zero cf_uv parameter
    with pytest.raises((ValueError)):
        pulse_bogus = EmPulse(env='bogus', tc=120.0, cf=0.001, dev=2.0, cf_uv=0.072)

    # Negative cf_uv parameter
    with pytest.raises((ValueError)):
        pulse_bogus = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf_uv=-0.072)

    # Negative cf parameter
    with pytest.raises((ValueError)):
        pulse_bogus = EmPulse(env='gaussian', cf = -0.003, dev = infinity)

    # Missing required parameter for non-limiting pulse shape
    with pytest.raises((AssertionError)):
        pulse_bogus = EmPulse(env='gaussian', tc=120.0, dev=2.0, cf_uv=0.072)

    # Missing required parameter for non-limiting pulse shape
    with pytest.raises((AssertionError)):
        pulse_bogus = EmPulse(env='gaussian', tc=120.0, cf=0.001)

    # Missing required parameter for non-limiting pulse shape
    with pytest.raises((AssertionError)):
        pulse_bogus = EmPulse(env='gaussian', cf=0.001, dev=2.0)

    # Wavevector not len 3 tuple of floats
    with pytest.raises((AssertionError)):
        pulse_bogus = EmPulse(env='gaussian', tc=120.0, cf=0.001, dev=2.0,
                              wv=('bogus', 0.0, 0.0))

    # Polarization not len 3 tuple of floats
    with pytest.raises((AssertionError)):
        pulse_bogus = EmPulse(env='gaussian', tc=120.0, cf=0.001, dev=2.0,
                              pol=(1.0, 0.0, 0.0, 0.0))

    # Wavevector and polarization not orthogonal
    with pytest.raises((AssertionError)):
        pulse_bogus = EmPulse(env='gaussian', tc=120.0, cf=0.001, dev=2.0,
                              wv =(1.0, 0.0, 0.0), pol=(1.0, 0.0, 0.0))

    # Non-zero shift in overall phase
    with pytest.raises((AssertionError)):
        pulse_bogus = EmPulse(env='gaussian', tc=120.0, cf=0.001, dev=2.0,
                              overall_phase=0.0 + 1.0j)

def test_make_gaussian_pulse():

    # Same kind of pulse as first example in test_em_pulse
    pulse_a = make_gaussian_pulse(100.0, 0.005, 2.0, cf_uv=0.0, maxstr=1.0e-5,
                      wv=(0.0, 1.0, 0.0), pol=(0.0, 0.0, 1.0), overall_phase = 1.0 + 0.0j, id=1)

    assert pulse_a.env == 'gaussian'
    assert pulse_a.tc == 100.0
    assert pulse_a.cf == 0.005
    assert pulse_a.dev == 2.0
    assert pulse_a.cf_uv == 0.0
    assert pulse_a.maxstr == 1.0e-5
    assert pulse_a.wv == (0.0, 1.0, 0.0)
    assert pulse_a.pol == (0.0, 0.0, 1.0)
    assert pulse_a.overall_phase == 1.0 + 0.0j
    assert pulse_a.id == 1

    assert not(pulse_a.tendsImpulsive())
    assert not(pulse_a.tendsContinuous())

def test_make_impulsive_gaussian_pulse():

    # A UV/VIS impulsive-tending pulse. Leaving several parameters to their default values.
    pulse_b = make_impulsive_gaussian_pulse(tc = 120.0)
    assert pulse_b.env == 'gaussian'
    assert pulse_b.tc == 120.0
    assert pulse_b.cf == None # Default value
    assert pulse_b.dev == 0.0
    assert pulse_b.cf_uv == 0.0 # Default
    assert pulse_b.maxstr == 0.0 # Default
    assert pulse_b.wv == (0.0, 0.0, 1.0) # Default
    assert pulse_b.pol == (1.0, 0.0, 0.0) # Default
    assert pulse_b.overall_phase == 1.0 + 0.0j # Default
    assert pulse_b.id == None # Default

    assert pulse_b.tendsImpulsive()
    assert not(pulse_b.tendsContinuous())

def test_make_cw_gaussian_pulse():

    # A CW-tending IR range pulse. Only the (IR-range) carrier frequency argument needs to be specified.
    from math import inf as infinity
    pulse_c = make_cw_gaussian_pulse(0.003)
    assert pulse_c.env == 'gaussian'
    assert pulse_c.tc == None # Default value
    assert pulse_c.cf == 0.003
    assert pulse_c.dev == infinity

    assert not(pulse_c.tendsImpulsive())
    assert pulse_c.tendsContinuous()

def test_electric_field():

    # Simple 3-pulse field

    pulse_a = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf_uv=0.072, id=1)
    pulse_b = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf_uv=0.072, id=2)
    pulse_c = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf_uv=0.072, id=3)

    # Pulse ids out of order but that doesn't matter as long as all IDs from 1 to num of pulses are represented
    pulses = (pulse_a, pulse_c, pulse_b)

    field = ElectricField(pulses)

    # Not a lot to test here
    assert len(field.pulses) == 3
    assert isinstance(field.pulses[0], EmPulse)
    assert isinstance(field.pulses[1], EmPulse)
    assert isinstance(field.pulses[2], EmPulse)

    # Identifiers in sequence but don't start at 1
    with pytest.raises(ValueError):
        pulse_0 = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf_uv=0.072, id=0)
        pulses_bogus = (pulse_0, pulse_a, pulse_b)
        field_bogus = ElectricField(pulses_bogus)

    # Identifiers start at 1 but not in sequence
    with pytest.raises(ValueError):
        pulse_d = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf_uv=0.072, id=0)
        pulses_bogus = (pulse_a, pulse_b, pulse_d)
        field_bogus = ElectricField(pulses_bogus)


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


    pass
