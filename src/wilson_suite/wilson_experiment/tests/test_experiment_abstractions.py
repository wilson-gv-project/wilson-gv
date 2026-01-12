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

    pulse_ir_1 = make_impulsive_gaussian_pulse(tc=50.0, cf=0.0, cf_uv=0.0,
                                                    maxstr=1.0e-5, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0), id=1)

    pulse_ir_2 = make_impulsive_gaussian_pulse(tc=100.0, cf=0.0, cf_uv=0.0,
                                                    maxstr=1.0e-5, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0), id=2)

    pulse_uvvis_1 = make_impulsive_gaussian_pulse(tc=120.0, cf=0.0, cf_uv=0.072,
                                                       maxstr=1.0e-5, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0), id=3)

    pulses = (pulse_ir_1, pulse_ir_2, pulse_uvvis_1)

    field_a = ElectricField(pulses)

    detector_a = SpecDetector(detection_method='freq',
                                   detector_location=(0.0, 0.0, 1.0),
                                   detection_polarization=(1.0, 0.0, 0.0),
                                   detection_range=[0.003 + 0.0001 * i for i in range(101)],
                                   wv_filter=[{1: -1, 2: 1, 3: 1}])

    # Push one carrier freq
    scan_obj_a = ScanObject('pulse', 'cf', id=1, coeff=1.0)
    scan_obj_b = ScanObject('detector', 'detection_range', id=0, coeff=1.0)
    scan_range_a = [0.0001 * i for i in range(101)]
    scan_a = SpecScan(scan_objs=(scan_obj_a, scan_obj_b), range=scan_range_a)

    exp_a = VibExperiment(field=field_a, detector=detector_a, scans=(scan_a,), magn_conditions=((-1, 2),),)

    # Some attribute testing given a light touch since already covered by respective class tests
    assert len(exp_a.field.pulses) == 3
    assert exp_a.detector.detection_method == 'freq'
    assert len(exp_a.scans) == 1

    assert exp_a.magn_conditions == ((-1, 2),)

    # Assertions about post-init attributes: When these are obtained using VibExperiment methods, these assertions
    # also serve to cover those methods
    assert len(exp_a.relevant_phasematch) == 1
    assert exp_a.relevant_phasematch[0].id == 0
    assert exp_a.relevant_phasematch[0].pulses.pulse_refs == (-1, 2, 3)
    assert exp_a.dim == 2
    assert exp_a.epochs == [[1], [2], [3]] # Three epochs
    assert exp_a.int_sequences == [ ({1: -1}, {2: 1}, {3: 1}) ] # Only one interaction sequence: First -w1, then w2, then w3
    assert exp_a.cfuv == {1: 0.0, 2: 0.0, 3: 0.072} # Only pulse 3 has a nonzero UV/VIS freq component

    # One set of independent variables: -w1 and w2
    assert len(exp_a.indep_vars) == 1
    assert exp_a.indep_vars[0].phasematch_cond.id == 0
    assert exp_a.indep_vars[0].phasematch_cond.pulses.pulse_refs == (-1, 2, 3)
    assert exp_a.indep_vars[0].var_groups[0].var_set[0].pulse_refs == (-1,)
    assert exp_a.indep_vars[0].var_groups[0].var_set[1].pulse_refs == (2,)

    # See test_find_valid_axes for more information about structure
    assert len(exp_a.valid_axis_combs) == 1
    assert exp_a.valid_axis_combs[0].valid_axis_combs[0].axes[0].label == 'A'
    assert exp_a.valid_axis_combs[0].valid_axis_combs[0].axes[0].var_set.var_set[0].pulse_refs == (-1,)
    assert exp_a.valid_axis_combs[0].valid_axis_combs[0].axes[1].label == 'B'
    assert exp_a.valid_axis_combs[0].valid_axis_combs[0].axes[1].var_set.var_set[0].pulse_refs == (-1,)
    assert exp_a.valid_axis_combs[0].valid_axis_combs[0].axes[1].var_set.var_set[1].pulse_refs == (2,)

    assert exp_a.valid_axis_combs[0].valid_axis_combs[1].axes[0].label == 'A'
    assert exp_a.valid_axis_combs[0].valid_axis_combs[1].axes[0].var_set.var_set[0].pulse_refs == (-1,)
    assert exp_a.valid_axis_combs[0].valid_axis_combs[1].axes[1].label == 'B'
    assert exp_a.valid_axis_combs[0].valid_axis_combs[1].axes[1].var_set.var_set[0].pulse_refs == (2,)

    assert exp_a.valid_axis_combs[0].valid_axis_combs[2].axes[0].label == 'A'
    assert exp_a.valid_axis_combs[0].valid_axis_combs[2].axes[0].var_set.var_set[0].pulse_refs == (2,)
    assert exp_a.valid_axis_combs[0].valid_axis_combs[2].axes[1].label == 'B'
    assert exp_a.valid_axis_combs[0].valid_axis_combs[2].axes[1].var_set.var_set[0].pulse_refs == (-1,)

    assert exp_a.valid_axis_combs[0].valid_axis_combs[3].axes[0].label == 'A'
    assert exp_a.valid_axis_combs[0].valid_axis_combs[3].axes[0].var_set.var_set[0].pulse_refs == (2,)
    assert exp_a.valid_axis_combs[0].valid_axis_combs[3].axes[1].label == 'B'
    assert exp_a.valid_axis_combs[0].valid_axis_combs[3].axes[1].var_set.var_set[0].pulse_refs == (-1,)
    assert exp_a.valid_axis_combs[0].valid_axis_combs[3].axes[1].var_set.var_set[1].pulse_refs == (2,)

    # Canonical axes: A: -w1, B: w2
    assert exp_a.canonical_axes.axes[0].label == 'A'
    assert exp_a.canonical_axes.axes[0].var_set.var_set[0].pulse_refs == (-1,)
    assert exp_a.canonical_axes.axes[1].label == 'B'
    assert exp_a.canonical_axes.axes[1].var_set.var_set[0].pulse_refs == (2,)

    # All pulses (and detector polarization filter): Linearly polarized light in the x direction
    assert exp_a.all_polarizations == [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]

    # NOTE: May revise result after unit tests for avg vector
    assert exp_a.polarization_avg_vector == [1.0, 1.0, 1.0]

    # Wrong data for VibExperiment
    with pytest.raises(TypeError):
        exp_bogus = VibExperiment(field='bogus', detector=detector_a, scans=(scan_a,), magn_conditions=((-1, 2),),)

    with pytest.raises(TypeError):
        exp_a = VibExperiment(field=field_a, detector='bogus', scans=(scan_a,), magn_conditions=((-1, 2),), )

    with pytest.raises(TypeError):
        exp_a = VibExperiment(field=field_a, detector=detector_a, scans=('bogus',), magn_conditions=((-1, 2),), )

    with pytest.raises(TypeError):
        exp_a = VibExperiment(field=field_a, detector=detector_a, scans=(scan_a,), magn_conditions=([-1, 2],), )

    # Relevant phasematch all directions (no detector filter)
    # NOTE: Currently raises assertion error because the interaction sequence finder is not set up to work with
    # more than one phase-matching direction (and requires it to be specified in the detector)
    with pytest.raises(AssertionError):

        detector_a = SpecDetector(detection_method='freq',
                                       detector_location=(0.0, 0.0, 1.0),
                                       detection_polarization=(1.0, 0.0, 0.0),
                                       detection_range=[0.003 + 0.0001 * i for i in range(101)],
                                       )

        exp_a = VibExperiment(field=field_a, detector=detector_a, scans=(scan_a,), magn_conditions=((-1, 2),),)

    # More than one phase-matching direction in detector filter
    with pytest.raises(AssertionError):

        detector_a = SpecDetector(detection_method='freq',
                                  detector_location=(0.0, 0.0, 1.0),
                                  detection_polarization=(1.0, 0.0, 0.0),
                                  detection_range=[0.003 + 0.0001 * i for i in range(101)],
                                  wv_filter=[{1: -1, 2: 1, 3: 1}, {1: 1, 2: 1, 3: -1}])

        exp_a = VibExperiment(field=field_a, detector=detector_a, scans=(scan_a,), magn_conditions=((-1, 2),),)

    # More exotic setup, testing some of the things that are changed compared to the previous
    pulse_ir_3 = make_impulsive_gaussian_pulse(tc=120.0, cf=0.0, cf_uv=0.0,
                                                    maxstr=1.0e-5, wv=(0.0, 0.0, 1.0), pol=(0.0, 1.0, 0.0), id=4)

    pulse_uvvis_2 = make_impulsive_gaussian_pulse(tc=120.0, cf=0.0, cf_uv=0.072,
                                                       maxstr=1.0e-5, wv=(0.0, 0.0, 1.0), pol=(0.0, 1.0, 0.0), id=5)

    pulses = (pulse_ir_1, pulse_ir_2, pulse_uvvis_1, pulse_ir_3, pulse_uvvis_2)

    field_a = ElectricField(pulses)

    detector_a = SpecDetector(detection_method='freq',
                                   detector_location=(0.0, 0.0, 1.0),
                                   detection_polarization=(1.0, 0.0, 0.0),
                                   detection_range=[0.003 + 0.0001 * i for i in range(101)],
                                   wv_filter=[{1: -1, 2: 1, 3: 1, 4: 1, 5:-1}])

    # Scan IR pulse 1 carrier freq
    scan_obj_a = ScanObject('pulse', 'cf', id=1, coeff=1.0)
    scan_obj_b = ScanObject('detector', 'detection_range', id=0, coeff=1.0)

    # Scan a Raman-like freq. difference
    # NOTE: Not decided yet if this kind of scan should be expressed as scanning cf or cf_uv for UV/VIS pulses
    scan_obj_c = ScanObject('pulse', 'cf', id=3, coeff=0.5)
    scan_obj_d = ScanObject('pulse', 'cf', id=5, coeff=-0.5)

    scan_range_a = [0.0001 * i for i in range(101)]
    scan_range_b = [0.0001 * i for i in range(101)]

    scan_a = SpecScan(scan_objs=(scan_obj_a, scan_obj_b), range=scan_range_a)
    scan_b = SpecScan(scan_objs=(scan_obj_c, scan_obj_d, scan_obj_b), range=scan_range_b)

    exp_a = VibExperiment(field=field_a, detector=detector_a, scans=(scan_a, scan_b))

    # Three epochs
    assert exp_a.epochs == [[1], [2], [3, 4, 5]]

    # Six possible interaction sequences: All permutations of pulses 3, 4, 5
    assert exp_a.int_sequences == [
        ({1: -1}, {2: 1}, {3: 1}, {4: 1}, {5: -1}),
        ({1: -1}, {2: 1}, {3: 1}, {5: -1}, {4: 1}),
        ({1: -1}, {2: 1}, {4: 1}, {3: 1}, {5: -1}),
        ({1: -1}, {2: 1}, {4: 1}, {5: -1}, {3: 1}),
        ({1: -1}, {2: 1}, {5: -1}, {3: 1}, {4: 1}),
        ({1: -1}, {2: 1}, {5: -1}, {4: 1}, {3: 1})
    ]

    # Pulses 3 and 5 have non-zero UV/VIS freq components
    assert exp_a.cfuv == {1: 0.0, 2: 0.0, 3: 0.072, 4: 0.0, 5: 0.072}

    # Independent variables: -w1, w2, w4, and - w5 + w3
    assert len(exp_a.indep_vars) == 1
    assert exp_a.indep_vars[0].var_groups[0].var_set[0].pulse_refs == (-1,)
    assert exp_a.indep_vars[0].var_groups[0].var_set[1].pulse_refs == (2,)
    assert exp_a.indep_vars[0].var_groups[0].var_set[2].pulse_refs == (4,)
    assert exp_a.indep_vars[0].var_groups[0].var_set[3].pulse_refs == (-5, 3)

    # Canonical axes: A: -w1, B: w2, C: w4, D: w3 - w5
    assert len(exp_a.canonical_axes.axes) == 4
    assert exp_a.canonical_axes.axes[0].label == 'A'
    assert exp_a.canonical_axes.axes[0].var_set.var_set[0].pulse_refs == (-1,)
    assert exp_a.canonical_axes.axes[1].label == 'B'
    assert exp_a.canonical_axes.axes[1].var_set.var_set[0].pulse_refs == (2,)
    assert exp_a.canonical_axes.axes[2].label == 'C'
    assert exp_a.canonical_axes.axes[2].var_set.var_set[0].pulse_refs == (4,)
    assert exp_a.canonical_axes.axes[3].label == 'D'
    assert exp_a.canonical_axes.axes[3].var_set.var_set[0].pulse_refs == (-5, 3)

    # FIXME: Add test for polarization avg vector once unit tests for that are finished

def test_get_carrier_freqs_uv():

    # Simple four-pulse setup
    pulse_a = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf_uv=0.072, id=1)
    pulse_b = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf_uv=0.144, id=2)
    pulse_c = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf_uv=0.072, id=3)
    pulse_d = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf_uv=0.0, id=4)

    pulses = (pulse_a, pulse_b, pulse_c, pulse_d)
    field = ElectricField(pulses)

    cfuv = get_carrier_freqs_uv(field.pulses)

    # Uncomplicated function so not much to test
    assert isinstance(cfuv, dict)
    assert len(cfuv) == 4
    assert cfuv[1] == 0.072
    assert cfuv[2] == 0.144
    assert cfuv[3] == 0.072
    assert cfuv[4] == 0.0

def test_find_epochs():

    # Simple four-pulse setup all at same time
    pulse_a = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf_uv=0.072, id=1)
    pulse_b = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf_uv=0.144, id=2)
    pulse_c = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf_uv=0.072, id=3)
    pulse_d = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf_uv=0.0, id=4)

    pulses = (pulse_a, pulse_b, pulse_c, pulse_d)
    field = ElectricField(pulses)

    epochs_a = find_epochs(field)

    # Should return one epoch with all pulses
    assert epochs_a == [[1, 2, 3, 4]]

    # New setup with more variation
    pulse_a = EmPulse(env='gaussian', tc=10.0, dev=0.0, cf_uv=0.072, id=5)
    pulse_b = EmPulse(env='gaussian', tc=50.0, dev=0.0, cf_uv=0.144, id=2)
    pulse_c = EmPulse(env='gaussian', tc=50.0, dev=0.0, cf_uv=0.072, id=4)
    pulse_d = EmPulse(env='gaussian', tc=100.0, dev=0.0, cf_uv=0.0, id=3)
    pulse_e = EmPulse(env='gaussian', tc=100.0, dev=0.0, cf_uv=0.0, id=1)
    pulse_f = EmPulse(env='gaussian', tc=100.0, dev=0.0, cf_uv=0.0, id=6)
    pulse_g = EmPulse(env='gaussian', tc=100.00001, dev=0.0, cf_uv=0.0, id=7)

    pulses = (pulse_a, pulse_e, pulse_c, pulse_b, pulse_d, pulse_f, pulse_g)
    field = ElectricField(pulses)

    epochs_b = find_epochs(field)

    # Should return four epochs with p
    assert epochs_b == [[5], [2, 4], [1, 3, 6], [7]]

    # Nonzero tolerance for what is to be considered "simultaneous" or "same epoch" not yet supported
    with pytest.raises(ValueError):
        epochs_bogus = find_epochs(field, tol=0.0000001)

    # Non-impulsive-tending pulse
    with pytest.raises(AssertionError):
        pulse_d = EmPulse(env='gaussian', tc=100.0, dev=2.0, cf_uv=0.0, id=3)
        pulses = (pulse_a, pulse_e, pulse_c, pulse_b, pulse_d, pulse_f, pulse_g)
        field = ElectricField(pulses)
        epochs_bogus = find_epochs(field)


def test_uv_cancels():

    # Simple example
    cfs_uv = {1: 0.072, 2: 0.072}

    assert uv_cancels((1, -2), cfs_uv)
    assert not uv_cancels((1, 2), cfs_uv)

    # More involved situations
    cfs_uv = {1: 0.072, 2: 0.072, 3: 0.144, 4: 0.072, 5: 0.036, 6: 0.035999, 7: 0.035999999999}

    assert not uv_cancels((1, -3), cfs_uv)
    assert uv_cancels((2, -3, 1), cfs_uv)
    assert uv_cancels((-3, 2, 4), cfs_uv)
    assert uv_cancels((5, -7), cfs_uv)
    assert not uv_cancels((-5, 6), cfs_uv)

    # Difference small but above default 1e-10 tolerance
    assert not uv_cancels((1, 2, -3, 4, -5, -6), cfs_uv)

    # With raised tolerance, now accepted as cancelling
    assert uv_cancels((1, 2, -3, 4, -5, -6), cfs_uv, tol=0.001)

    # Negative tolerance
    with pytest.raises(ValueError):
        bogus = uv_cancels((1, -2), cfs_uv, tol=-0.002)


