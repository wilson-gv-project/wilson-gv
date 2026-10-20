import pytest

from wilson_suite.wilson_experiment.experiment_abstractions import (SpecDetector, ScanObject, SpecScan,
                                                                    EmPulse, make_gaussian_pulse,
                                                                    make_impulsive_gaussian_pulse,
                                                                    make_cw_gaussian_pulse,
                                                                    make_compound_gaussian_pulse,
                                                                    compound_pulses_in_list,
                                                                    compounding_patterns_at_order,
                                                                    pulse_time_ordering_valid,
                                                                    magnitude_conditions_for_ind_var_tuple,
                                                                    ElectricField, VibExperiment, FieldAnalysis)

def test_spec_detector():

    # Detector suitable for EVV:
    #   - Detector is set to capture a spectrum ('freq')
    #   - Set to detect light linearly polarized along the x-axis
    #   - Detection interval here set as the range [0.0, 5000.0] (units: cm^-1).
    #   - "Wavevector filter" (for selecting phase-matching conditions) here -k1 + k2 +k3

    detector_a = SpecDetector(detection_method='freq',
                              detection_polarization=(1.0, 0.0, 0.0),
                              detection_limited_to='same_order',
                              detection_interval=[0.0, 5000.0],
                              interaction_filter=[[-1, 2, 3]])

    assert detector_a.detection_method == 'freq'
    assert detector_a.detection_polarization == (1.0, 0.0, 0.0)
    assert detector_a.detection_interval == [0.0, 5000.0]
    assert detector_a.interaction_filter[0][0] == -1
    assert detector_a.interaction_filter[0][1] == 2
    assert detector_a.interaction_filter[0][2] == 3

    detector_a_face = SpecDetector(detection_method='freq',
                                   detection_polarization=(1.0, 0.0, 0.0),
                                   detection_limited_to='up_to_order',
                                   detection_interval=[0.0, 5000.0],
                                   detector_faces=(0.0, 0.0, 1.0),
                                   detector_max_order=4)

    assert detector_a_face.detector_faces == (0.0, 0.0, 1.0)
    assert detector_a_face.detector_max_order == 4
    assert detector_a_face.detection_limited_to == 'up_to_order'

    # Unrecognized detection method
    with pytest.raises(ValueError):
        detector_bogus = SpecDetector(detection_method='bogus',
                                      detection_polarization=(1.0, 0.0, 0.0),
                                      detection_limited_to='up_to_order',
                                      interaction_filter=[[-1, 2, 3]],
                                      detection_interval=[0.0, 5000.0])

    # Specifying both detector_faces and interaction_filter
    with pytest.raises(ValueError):
        detector_bogus = SpecDetector(detection_method='bogus',
                                      detection_polarization=(1.0, 0.0, 0.0),
                                      detection_limited_to='up_to_order',
                                      detector_faces=(0.0, 0.0, 1.0),
                                      interaction_filter=[[-1, 2, 3]],
                                      detection_interval=[0.0, 5000.0])

    # Specifying detector_max_order with interaction_filter
    with pytest.raises(ValueError):
        detector_bogus = SpecDetector(detection_method='bogus',
                                      detection_polarization=(1.0, 0.0, 0.0),
                                      detection_limited_to='up_to_order',
                                      interaction_filter=[[-1, 2, 3]],
                                      detector_max_order=3,
                                      detection_interval=[0.0, 5000.0])


def test_scan_object():

    # Here: A scan object referring to the carrier frequency of pulse 1, to be scanned at twice the
    # rapidity of range increments
    obj_a = ScanObject('pulse', 'cf', id=1, coeff=2.0)

    assert obj_a.category == 'pulse'
    assert obj_a.subcategory == 'cf'
    assert obj_a.id == 1
    assert obj_a.coeff == 2.0
    assert obj_a.scan_affects == 'integration'

    # Unrecognized scan category
    with pytest.raises(ValueError):
        obj_bogus = ScanObject('bogus', 'cf')

    # Recognized scan category but unrecognized scan subcategory
    with pytest.raises(ValueError):
        obj_bogus = ScanObject('pulse', 'bogus')

def test_spec_scan():

    # Making a scan consisting of two objects:
    #   - Scanning pulse 1 carrier frequency by twice the increments of the range
    #   - Scanning pulse 2 time maximum by 1 x the increments of the range
    scan_obj_a = ScanObject('pulse', 'cf', id=1, coeff=2.0)
    scan_obj_b = ScanObject('pulse', 'tc', id=2)
    scan_range = [2.0 * i for i in range(101)]

    scan_objs = (scan_obj_a, scan_obj_b)

    scan_a = SpecScan(scan_objs, scan_range)

    assert scan_a.scan_objs[0].category == 'pulse'
    assert scan_a.scan_objs[0].subcategory == 'cf'
    assert scan_a.scan_objs[0].id == 1
    assert scan_a.scan_objs[0].coeff == 2.0

    assert scan_a.scan_objs[1].category == 'pulse'
    assert scan_a.scan_objs[1].subcategory == 'tc'
    assert scan_a.scan_objs[1].id == 2
    assert scan_a.scan_objs[1].coeff == 1.0

    assert scan_a.range == [2.0 * i for i in range(101)]

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
    # - Time envelope centerpoint at 100 fs
    # - Time envelope deviation parameter 20.0 fs
    # - Carrier frequency 1000 cm^-1
    # - No UV/VIS component of carrier frequency
    # - Wavevector (0.0, 1.0, 0.0) (laboratory axes)
    # - Polarization vector (0.0, 0.0, 1.0) (laboratory axes)
    # - No overall phase shift (here replicating default)
    # - (Integer) identifier label: 1
    pulse_a = EmPulse(env='gaussian', tc = 100.0, cf = 1000.0, dev=20.0, maxstr=1.0e-5,
                      wv=(0.0, 1.0, 0.0), pol=(0.0, 0.0, 1.0), overall_phase = 1.0 + 0.0j, id=1)

    assert pulse_a.env == 'gaussian'
    assert pulse_a.tc == 100.0
    assert pulse_a.cf == 1000.0
    assert pulse_a.dev == 20.0
    assert pulse_a.maxstr == 1.0e-5
    assert pulse_a.wv == (0.0, 1.0, 0.0)
    assert pulse_a.pol == (0.0, 0.0, 1.0)
    assert pulse_a.overall_phase == 1.0 + 0.0j
    assert pulse_a.id == 1
    assert pulse_a.pol_comp is None
    assert pulse_a.id_comp is None
    assert not pulse_a.is_compound

    # The choice of deviation parameter is non-limiting and should therefore result in the pulse being
    # neither impulsive-tending or continuous-wave-tending
    assert not(pulse_a.tendsImpulsive())
    assert not(pulse_a.tendsContinuous())

    # A CW-tending IR range pulse. For continuous-wave-tending
    # pulses, the time centerpoint argument is optional
    from math import inf as infinity
    pulse_c = EmPulse(env='gaussian', cf = 2000.0, dev = infinity, wv=(0.0, 0.0, 1.0), id=1, pol=(1.0, 0.0, 0.0))
    assert pulse_c.env == 'gaussian'
    assert pulse_c.tc is None # Default value
    assert pulse_c.cf == 2000.0
    assert pulse_c.dev == infinity

    assert not(pulse_c.tendsImpulsive())
    assert pulse_c.tendsContinuous()

    # Compound pulse
    pulse_comp = EmPulse(env='gaussian', tc = 200.0, cf = 1000.0, dev=20.0, maxstr=1.0e-5,
                      wv=(0.0, 1.0, 0.0), pol_comp=((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), overall_phase = 1.0 + 0.0j, id_comp=(1, 3))

    assert pulse_comp.env == 'gaussian'
    assert pulse_comp.tc == 200.0
    assert pulse_comp.cf == 1000.0
    assert pulse_comp.dev == 20.0
    assert pulse_comp.maxstr == 1.0e-5
    assert pulse_comp.wv == (0.0, 1.0, 0.0)
    assert pulse_comp.overall_phase == 1.0 + 0.0j
    assert pulse_comp.id is None
    assert pulse_comp.pol is None
    assert pulse_comp.id_comp == ((1, 3))
    assert pulse_comp.pol_comp == ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert pulse_comp.is_compound

    # Nonorthogonal wv/polarization within tolerance
    pulse_a = EmPulse(env='gaussian', tc = 100.0, cf = 1000.0, dev=20.0, maxstr=1.0e-5,
                      wv=(0.0, 1.0, 0.02), pol=(0.0, 0.0, 1.0), overall_phase = 1.0 + 0.0j, id=1,
                      tol_wv_pol_nonort=5.0)

    assert pulse_a.wv == (0.0, 1.0, 0.02)
    assert pulse_a.pol == (0.0, 0.0, 1.0)

    # Unrecognized pulse envelope
    with pytest.raises((ValueError)):
        pulse_bogus = EmPulse(env='bogus', tc=120.0, dev=0.0, wv=(0.0, 0.0, 1.0), id=1, pol=(1.0, 0.0, 0.0))

    # Missing required parameter for non-limiting pulse shape
    with pytest.raises((AssertionError)):
        pulse_bogus = EmPulse(env='gaussian', tc=120.0, dev=2.0, wv=(0.0, 0.0, 1.0), id=1, pol=(1.0, 0.0, 0.0))

    # Missing required parameter for non-limiting pulse shape
    with pytest.raises((AssertionError)):
        pulse_bogus = EmPulse(env='gaussian', tc=120.0, cf=10000.0, wv=(0.0, 0.0, 1.0), id=1, pol=(1.0, 0.0, 0.0))

    # Missing required parameter for non-limiting pulse shape
    with pytest.raises((AssertionError)):
        pulse_bogus = EmPulse(env='gaussian', cf=1000.0, dev=2.0, wv=(0.0, 0.0, 1.0), id=1, pol=(1.0, 0.0, 0.0))

    # Wavevector not len 3 tuple of floats
    with pytest.raises((AssertionError)):
        pulse_bogus = EmPulse(env='gaussian', tc=120.0, cf=0.001, dev=2.0,
                              wv=('bogus', 0.0, 0.0), id=1, pol=(1.0, 0.0, 0.0))

    # Polarization not len 3 tuple of floats
    with pytest.raises((AssertionError)):
        pulse_bogus = EmPulse(env='gaussian', tc=120.0, cf=0.001, dev=2.0,
                              pol=(1.0, 0.0, 0.0, 0.0), wv=(0.0, 0.0, 1.0), id=1)

    # Wavevector and polarization not orthogonal
    with pytest.raises((AssertionError)):
        pulse_bogus = EmPulse(env='gaussian', tc=120.0, cf=0.001, dev=2.0,
                              wv =(1.0, 0.0, 0.0), pol=(1.0, 0.0, 0.0), id=1)

def test_make_gaussian_pulse():

    # Same kind of pulse as first example in test_em_pulse
    pulse_a = make_gaussian_pulse(100.0, 0.005, 2.0, maxstr=1.0e-5,
                      wv=(0.0, 1.0, 0.0), pol=(0.0, 0.0, 1.0), overall_phase = 1.0 + 0.0j, id=1)

    assert pulse_a.env == 'gaussian'
    assert pulse_a.tc == 100.0
    assert pulse_a.cf == 0.005
    assert pulse_a.dev == 2.0
    assert pulse_a.maxstr == 1.0e-5
    assert pulse_a.wv == (0.0, 1.0, 0.0)
    assert pulse_a.pol == (0.0, 0.0, 1.0)
    assert pulse_a.overall_phase == 1.0 + 0.0j
    assert pulse_a.id == 1

    assert not(pulse_a.tendsImpulsive())
    assert not(pulse_a.tendsContinuous())

def test_make_impulsive_gaussian_pulse():

    # A UV/VIS impulsive-tending pulse. Leaving several parameters to their default values.
    pulse_b = make_impulsive_gaussian_pulse(tc = 120.0, id=1)
    assert pulse_b.env == 'gaussian'
    assert pulse_b.tc == 120.0
    assert pulse_b.cf is None # Default value
    assert pulse_b.dev == 0.0
    assert pulse_b.maxstr == 0.0 # Default
    assert pulse_b.wv == (0.0, 0.0, 1.0) # Default
    assert pulse_b.pol == (1.0, 0.0, 0.0) # Default
    assert pulse_b.overall_phase == 0.0

    assert pulse_b.tendsImpulsive()
    assert not(pulse_b.tendsContinuous())

def test_make_cw_gaussian_pulse():

    # A CW-tending IR range pulse. Only the (IR-range) carrier frequency argument needs to be specified.
    from math import inf as infinity
    pulse_c = make_cw_gaussian_pulse(cf=0.003, id=1)
    assert pulse_c.env == 'gaussian'
    assert pulse_c.tc is None # Default value
    assert pulse_c.cf == 0.003
    assert pulse_c.id == 1
    assert pulse_c.dev == infinity

    assert not(pulse_c.tendsImpulsive())
    assert pulse_c.tendsContinuous()

def test_make_compound_gaussian_pulse():

    # Compounding two reasonable Gaussian IR-range pulses
    pulse_a = EmPulse(env='gaussian', tc = 100.0, cf = 1000.0, dev=100.0, maxstr=1.0e-5,
                      wv=(0.0, 1.0, 0.0), pol=(1.0, 0.0, 0.0), overall_phase = 1.0 + 0.0j, id=3)

    pulse_b = EmPulse(env='gaussian', tc = 200.0, cf = 2000.0, dev=60.0, maxstr=2.0e-5,
                      wv=(0.0, 1.0, 0.0), pol=(0.0, 0.0, 1.0), overall_phase = 0.5 + 0.0j, id=2)


    pulse_comp = make_compound_gaussian_pulse(pulse_a, pulse_b)

    # Combination attributes verified by formula
    assert pulse_comp.env == 'gaussian'
    # Time maximum is in-between
    assert abs(pulse_comp.tc - 173.5294) < 1e-4
    # Carrier freq is sum
    assert pulse_comp.cf == 3000.0
    # Deviation is narrower than each individual pulse's
    assert abs(pulse_comp.dev - 51.4496) < 1e-4
    assert abs(pulse_comp.maxstr - 1.3847e-10) < 1e-14
    assert pulse_comp.wv == (0.0, 2.0, 0.0)
    assert pulse_comp.pol is None
    assert pulse_comp.overall_phase == 1.5 + 0.0j
    assert pulse_comp.id is None
    # Compound polarization and IDs in corresponding ordering
    assert pulse_comp.pol_comp == ( (1.0, 0.0, 0.0),  (0.0, 0.0, 1.0))
    assert pulse_comp.id_comp == (3, 2)
    assert pulse_comp.is_compound

def test_compound_pulses_in_list():

    pulse_a = EmPulse(env='gaussian', tc = 100.0, cf = 1000.0, dev=100.0, maxstr=1.0e-5,
                      wv=(0.0, 1.0, 0.0), pol=(1.0, 0.0, 0.0), overall_phase = 1.0 + 0.0j, id=3)

    pulse_b = EmPulse(env='gaussian', tc = 200.0, cf = 2000.0, dev=60.0, maxstr=2.0e-5,
                      wv=(0.0, 1.0, 0.0), pol=(0.0, 0.0, 1.0), overall_phase = 0.5 + 0.0j, id=2)

    pulse_comp = compound_pulses_in_list([pulse_a, pulse_b])

    assert pulse_comp.env == 'gaussian'
    assert abs(pulse_comp.tc - 173.5294) < 1e-4
    assert pulse_comp.cf == 3000.0
    assert abs(pulse_comp.dev - 51.4496) < 1e-4
    assert abs(pulse_comp.maxstr - 1.3847e-10) < 1e-14
    assert pulse_comp.wv == (0.0, 2.0, 0.0)
    assert pulse_comp.pol is None
    assert pulse_comp.overall_phase == 1.5 + 0.0j
    assert pulse_comp.id is None
    assert pulse_comp.pol_comp == ( (1.0, 0.0, 0.0),  (0.0, 0.0, 1.0))
    assert pulse_comp.id_comp == (3, 2)
    assert pulse_comp.is_compound


def test_electric_field():

    # Simple 3-pulse field

    pulse_a = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf=0.072, id=1, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0))
    pulse_b = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf=0.072, id=2, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0))
    pulse_c = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf=0.072, id=3, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0))

    from math import inf as infinity
    pulse_cw_a = EmPulse(env='gaussian', cf=0.003, dev=infinity, id=1, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0))
    pulse_cw_b = EmPulse(env='gaussian', cf=0.003, dev=infinity, id=2, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0))
    pulse_cw_c = EmPulse(env='gaussian', cf=0.003, dev=infinity, id=3, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0))

    pulse_mix_c = EmPulse(env='gaussian', tc=120.0, dev=10.0, cf= 0.002, id=3, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0))

    # Pulse ids out of order but that doesn't matter as long as all IDs from 1 to num of pulses are represented
    pulses = (pulse_a, pulse_c, pulse_b)

    field = ElectricField(pulses)

    # Testing basic attributes
    assert len(field.pulses) == 3
    assert isinstance(field.pulses[0], EmPulse)
    assert isinstance(field.pulses[1], EmPulse)
    assert isinstance(field.pulses[2], EmPulse)
    assert field.impulsive_field
    assert not(field.cw_field)

    v_int = field.overlapping_pulses_for_interaction_pattern(pattern=(1, 2, 3))

    # All pulse subsets should overlap in time here
    assert v_int == [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

    # Identifiers in sequence but don't start at 1
    with pytest.raises(ValueError):
        pulse_0 = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf=0.072, id=0, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0))
        pulses_bogus = (pulse_0, pulse_a, pulse_b)
        field_bogus = ElectricField(pulses_bogus)

    # Identifiers start at 1 but not in sequence
    with pytest.raises(ValueError):
        pulse_d = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf=0.072, id=0, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0))
        pulses_bogus = (pulse_a, pulse_b, pulse_d)
        field_bogus = ElectricField(pulses_bogus)

    # Testing if continuous-wave field is recognized as such
    pulses_cw = (pulse_cw_a, pulse_cw_b, pulse_cw_c)
    field_cw = ElectricField(pulses_cw)
    assert field_cw.cw_field

    v_int_cw = field_cw.overlapping_pulses_for_interaction_pattern(pattern=[1, 2, 3])

    # Again, all pulses should overlap
    assert v_int_cw == [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

    # Mixes of impulsive, continuous and neither-type fields, should flag field as neither cw nor impulsive
    pulses_mix_1 = (pulse_cw_a, pulse_cw_b, pulse_c)
    field_mix_1 = ElectricField(pulses_mix_1)
    assert not(field_mix_1.cw_field)
    assert not(field_mix_1.impulsive_field)

    v_int_mix = field_mix_1.overlapping_pulses_for_interaction_pattern(pattern=[1, 2, 3])

    # Yet again, all pulses should overlap
    assert v_int_mix == [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

    pulses_mix_2 = (pulse_cw_a, pulse_b, pulse_mix_c)
    field_mix_2 = ElectricField(pulses_mix_2)
    assert not(field_mix_2.cw_field)
    assert not(field_mix_2.impulsive_field)

    # More general case: Complicated overlap configurations and testing of wavevector matching
    pulse_gen_b = EmPulse(env='gaussian', dev=infinity, cf=0.0, id=1, wv=(1.0, 0.0, 0.0), pol=(0.0, 1.0, 0.0))
    pulse_gen_c = EmPulse(env='gaussian', tc=300.01, dev=20.0, cf=0.02, id=2, wv=(1.0, 2.0, 0.0), pol=(0.0, 0.0, 1.0))
    pulse_gen_a = EmPulse(env='gaussian', tc=120.0, dev=20.0, cf=0.072, id=3, wv=(1.0, 0.0, 0.0), pol=(0.0, 1.0, 0.0))
    pulse_gen_d = EmPulse(env='gaussian', tc=20.0, dev=0.0, cf=0.072, id=4, wv=(0.0, 2.0, 4.0), pol=(1.0, 0.0, 0.0))
    pulse_gen_f = EmPulse(env='gaussian', tc=230.0, dev=30.0, cf=0.02, id=5, wv=(0.0, 1.0, -4.0), pol=(1.0, 0.0, 0.0))
    pulse_gen_e = EmPulse(env='gaussian', tc=120.0, dev=0.0, cf=0.07, id=6, wv=(1.0, 2.0, 0.0), pol=(0.0, 0.0, 2.0))

    field_gen = ElectricField((pulse_gen_a, pulse_gen_b, pulse_gen_c, pulse_gen_d, pulse_gen_e, pulse_gen_f))

    v_int_gen = field_gen.overlapping_pulses_for_interaction_pattern(pattern=(1, 2, 3, 4, 5, 6))

    assert v_int_gen == [(1,), (2,), (3,), (4,), (5,), (6,), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3),
                         (2, 5), (3, 4), (3, 5), (3, 6), (5, 6), (1, 2, 3), (1, 2, 5), (1, 3, 4), (1, 3, 5),
                         (1, 3, 6), (1, 5, 6), (2, 3, 5), (3, 5, 6), (1, 2, 3, 5), (1, 3, 5, 6)]

    # Lower tolerance (stricter) to rule overlap: Some of the patterns ruled in with default tolerance are now out
    v_int_gen_b = field_gen.overlapping_pulses_for_interaction_pattern(pattern=(1, 2, 3, 4, 5, 6), tol_n_dev=2.0)

    assert v_int_gen_b == [(1,), (2,), (3,), (4,), (5,), (6,), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                         (2, 5), (3, 6), (1, 2, 5), (1, 3, 6) ]

    # A couple more test cases for less "basic" interaction patterns
    v_int_gen_b = field_gen.overlapping_pulses_for_interaction_pattern(pattern=(1, -2, 3, 4, -5, 6), tol_n_dev=2.0)
    assert v_int_gen_b == [ (-5,), (-2,), (1,), (3,), (4,), (6,), (-5, -2), (-5, 1), (-2, 1), (1, 3), (1, 4), (1, 6),
                            (3, 6), (-5, -2, 1), (1, 3, 6)]

    v_int_gen_b = field_gen.overlapping_pulses_for_interaction_pattern(pattern=(1, -1, 1, -1, 3, -5), tol_n_dev=2.0)

    assert sorted(v_int_gen_b) == sorted([(-5,), (-1,), (1,), (3,), (-5, -1), (-5, 1), (-1, -1), (-1, 1), (-1, 3), (1, 1), (1, 3),
                           (-5, -1, -1), (-5, -1, 1), (-5, 1, 1), (-1, -1, 1), (-1, -1, 3),
                           (-1, 1, 1), (-1, 1, 3), (1, 1, 3), (-5, -1, -1, 1), (-5, -1, 1, 1), (-1, -1, 1, 1),
                           (-1, -1, 1, 3), (-1, 1, 1, 3), (-5, -1, -1, 1, 1), (-1, -1, 1, 1, 3)  ])

    # Testing wavevector matching routine
    wv_dir = (1.0, -4.0, -4.0) # 1 - 2 + 3 - 4
    matching_wv = field_gen.wavevectors_matching_direction(direction=wv_dir, max_order=4, filter='same_order')

    # NOTE: All the following matching_wv results are verified for positive ID but not for false negatives
    assert sorted(matching_wv) == sorted([(-4, -2, 1, 3), (-4, -2, 3, 3), (-6, -4, 3, 3), (-6, -4, 1, 3), (-4, -2, 1, 1), (-6, -4, 1, 1)])

    wv_dir = (1.0, 1.0, 4.0) # 1 - 1 + 6 - 5
    matching_wv = field_gen.wavevectors_matching_direction(direction=wv_dir, max_order=4, filter='same_order')
    assert sorted(matching_wv) == sorted([(-5, -1, 1, 6), (-5, -3, 2, 3), (-5, -3, 3, 6), (-5, -1, 2, 3), (-5, -1, 3, 6),
                           (-5, -3, 1, 2), (-5, -3, 1, 6), (-5, -1, 1, 2), (-5, -2, 2, 2), (-6, -5, 2, 2),
                           (-5, -5, 2, 2), (-5, -2, 2, 6), (-5, -4, 2, 4), (-6, -5, 2, 6), (-5, -5, 2, 6),
                           (-5, -5, 2, 5), (-5, -2, 6, 6), (-5, -4, 4, 6), (-6, -5, 6, 6), (-5, -5, 6, 6),
                           (-5, -5, 5, 6)])

    wv_dir = (0.0, 1.0, -4.0) # 1 - 3 + 5
    matching_wv = field_gen.wavevectors_matching_direction(direction=wv_dir, max_order=3, filter='up_to_order')
    assert sorted(matching_wv) == sorted([(-3, 1, 5), (5,), (5, 5), (-3, 3, 5), (-1, 3, 5), (-1, 1, 5), (-2, 2, 5),
                           (-6, 2, 5), (-2, 5, 6), (-4, 4, 5), (-6, 5, 6), (5, 5, 5), (-5, 5, 5)])


    # Testing resonance screening
    pulse_gen_a = EmPulse(env='gaussian', tc=0.0, dev=20.0, cf=9000., id=1, wv=(1.0, 0.0, 0.0), pol=(0.0, 0.0, 1.0))
    pulse_gen_b = EmPulse(env='gaussian', tc=400.0, dev=100.0, cf=15000., id=2, wv=(-1.0, 0.0, 0.0), pol=(0.0, 0.0, 1.0))
    pulse_gen_c = EmPulse(env='gaussian', tc=500.0, dev=30.0, cf=1500., id=3, wv=(1.0, 0.0, 0.0), pol=(0.0, 0.0, 1.0))
    pulse_gen_d = EmPulse(env='gaussian', tc=800., dev=40.0, cf=20000., id=4, wv=(0.0, 2.0, 0.0), pol=(0.0, 0.0, 1.0))

    field_gen = ElectricField((pulse_gen_a, pulse_gen_b, pulse_gen_c, pulse_gen_d))

    # Several pulses are overlapping in time (and are eligible as compound pulses) but some of them don't
    # have frequency envelopes that can create vibrational resonance

    ovl_pulses = field_gen.overlapping_pulses_for_interaction_pattern(pattern=(1, -2, 3, 4))
    assert sorted(ovl_pulses) == sorted([(-2,), (1,), (3,), (4,), (-2, 1), (-2, 3), (-2, 4), (3, 4), (-2, 3, 4)])

    screened_compound_pulses = field_gen.all_resonance_screened_compound_pulses(ovl_pulses)
    assert sorted(list(screened_compound_pulses)) == sorted(list(([(1,), (3,), (-2, 1), (-2, 4), (-2, 3, 4)])))


def test_compounding_patterns_at_order():

    # The compounding patterns are all ordered integer partitionings of the order N

    patterns = compounding_patterns_at_order(3)
    assert sorted(patterns) == sorted([(3,), (2, 1), (1, 2), (1, 1, 1)])

    patterns = compounding_patterns_at_order(5)
    assert sorted(patterns) == sorted([(5,), (4, 1), (3, 2), (2, 3), (1, 4), (3, 1, 1), (1, 3, 1), (1, 1, 3),
                                       (2, 2, 1), (2, 1, 2), (1, 2, 2),
                                       (2, 1, 1, 1), (1, 2, 1, 1), (1, 1, 2, 1), (1, 1, 1, 2), (1, 1, 1, 1, 1),])

def test_pulse_time_ordering_valid():

    # Three reasonably well-spaced pulses
    p1 = EmPulse(env='gaussian', tc=0.0, dev=20.0, cf=9000., id=1, wv=(1.0, 0.0, 0.0), pol=(0.0, 0.0, 1.0))
    p2 = EmPulse(env='gaussian', tc=300.0, dev=100.0, cf=15000., id=2, wv=(-1.0, 0.0, 0.0), pol=(0.0, 0.0, 1.0))
    p3 = EmPulse(env='gaussian', tc=1350.0, dev=30.0, cf=1500., id=3, wv=(1.0, 0.0, 0.0), pol=(0.0, 0.0, 1.0))

    # Singletons trivially valid
    assert pulse_time_ordering_valid([p1])
    assert pulse_time_ordering_valid([p2])

    # Default waiting time is 1000 fs, so 1 and 2 should be valid, also 2 and 3, but not 1 and 3
    assert pulse_time_ordering_valid([p1, p2])
    assert pulse_time_ordering_valid([p2, p3])
    assert not pulse_time_ordering_valid([p1, p3])

    # Therefore also 1, 2, 3 with default
    assert pulse_time_ordering_valid([p1, p2, p3])

    # Not 1, 3, 2
    assert not pulse_time_ordering_valid([p1, p3, p2])

def test_magnitude_conditions_for_ind_var_tuple():

    # A selection of typical pulses
    p1 = EmPulse(env='gaussian', tc=0.0, dev=20.0, cf=2000., id=1, wv=(1.0, 0.0, 0.0), pol=(0.0, 0.0, 1.0))
    p2 = EmPulse(env='gaussian', tc=300.0, dev=100.0, cf=15000., id=2, wv=(1.0, 0.0, 0.0), pol=(0.0, 0.0, 1.0))
    p3 = EmPulse(env='gaussian', tc=750.0, dev=80.0, cf=12000., id=3, wv=(1.0, 0.0, 0.0), pol=(0.0, 0.0, 1.0))

    field = ElectricField((p1, p2, p3))

    # Derived pulses
    pm1 = field.signed_pulses[-1]
    pm2 = field.signed_pulses[-2]
    pm3 = field.signed_pulses[-3]

    p12 = compound_pulses_in_list([p1, p2])
    p33 = compound_pulses_in_list([p3, p3])
    p1m1 = compound_pulses_in_list([p1, pm1])
    pm23 = compound_pulses_in_list([pm2, p3])
    pm2m2 = compound_pulses_in_list([pm2, pm2])

    # All relevant pulses plus a few extra
    all_pulses = {(1,): p1, (2,): p2, (-2,): pm2, (3,): p3, (-1, 1): p1m1, (1, 2): p12, (-2, 3): pm23, (-2, -2): pm2m2, (3, 3): p33}

    mc_p, mc_m = magnitude_conditions_for_ind_var_tuple(((1,), (1, 2,), (-2, 3,)), all_pulses, thres_magn=500.0, tol_n_dev=3.0)

    # Note (0, 2) (i.e. 1 + (-2, 3)) not included, mean is positive but spread most likely across zero
    # These interaction number combinations are deemed positive
    assert mc_p == ((0,), (1,), (0, 1), (1, 2), (0, 1, 2))
    # These interaction number combinations are deemed negative
    assert mc_m == ((2,),)

    # A bit more involved case
    mc_p, mc_m = magnitude_conditions_for_ind_var_tuple(((-1, 1), (2,), (-2,), (3,)), all_pulses, thres_magn=200.0)

    assert mc_p == ((1,), (3,), (0, 1), (0, 3), (1, 3), (0, 1, 3), (1, 2, 3), (0, 1, 2, 3))
    assert mc_m == ((2,), (0, 2), (2, 3), (0, 2, 3))


def test_field_analysis():

    pass


def present_field_analysis(e: FieldAnalysis):

    print(e)
    print('valid int patterns', e.valid_int_patterns)
    print('int pattern orders', e.int_pattern_orders)
    print('int patterns by order', e.int_patterns_by_order)
    print('time_ovl_tuples', e.time_ovl_tuples)
    print('res_and_time_ovl_tuples', e.res_and_time_ovl_tuples)

    print('comp_patterns_at_order', e.comp_patterns_at_order)
    print('pol_avg_vector_bank', e.pol_avg_vector_bank)
    print('int_sequences_by_order_comp_pol', e.int_sequences_by_order_comp_pol)

    print('magnitude_condition_bank', e.magnitude_condition_bank)
    print('int_sequences_by_order_comp_pol_magn', e.int_sequences_by_order_comp_pol_magn)

    return


def test_vib_experiment():

    # TODO: More extensive tests, covering more relevant cases

    # Rough wishlist:
    #
    # Overall setups:
    #
    # - EVV 2-scan experiment OK
    # - CARS OK
    # - "alpha, beta" cars OK
    # - SFG OK
    # - TRIVE variant (possibly delays)
    # - Ideal freq domain (with simplifications)
    # - "2D Raman" with/without cascading effects
    #
    # Situations (some may be covered by above setups):
    #
    # - Tolerances changing outcome: Pathways in/out etc.
    # - Different scan setups: Differing dimensionalities, scans making field analyses nonhomogeneous over scans
    # - Variations on pulses changing outcome: Resonance possibilities in/out
    # - More than one wavevector combination overlapping
    # - Testing both "same_order", "up_to_order", and "only_specified"
    # - Interacting more than one time with same pulse
    #
    # Specific attributes to verify (cannot test all exhaustively)
    #
    # - Compounding patterns identified
    # - Pol. avgs.
    # - Compound pulse bandwidths/positions in time
    #
    #


    # "Alpha-beta CARS"

    pulse_uvvis_1 = make_gaussian_pulse(tc=0.0, cf=15000.0, dev=120.0, maxstr=1.0e-5,
                                        wv=(0.0, 0.02, 1.0), pol=(1.0, 0.0, 0.0), id=1, tol_wv_pol_nonort=5.0)

    pulse_uvvis_2 = make_gaussian_pulse(tc=350.0, cf=15000.0, dev=10.0, maxstr=1.0e-5,
                                        wv=(0.0, 0.012, 1.0), pol=(0.0, 1.0, 0.0), id=2, tol_wv_pol_nonort=5.0)

    pulse_uvvis_3 = make_gaussian_pulse(tc=750.0, cf=15000.0, dev=120.0, maxstr=1.0e-5,
                                        wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0), id=3, tol_wv_pol_nonort=5.0)

    pulses = (pulse_uvvis_1, pulse_uvvis_2, pulse_uvvis_3)

    field_ab_cars = ElectricField(pulses)

    detector_ab_cars = SpecDetector(detection_method='freq',
                                 detection_polarization=(0.0, 0.0, 1.0),
                                 detection_limited_to='same_order',
                                 interaction_filter=[[1, -2, 3, 3]])

    exp_ab_cars = VibExperiment(field=field_ab_cars,
                          detector=detector_ab_cars,
                          scans=(),
                          tol_n_dev_t=3.0,
                          tol_n_dev_w=3.0,
                          thres_waiting_time=600.0,)

    present_field_analysis(exp_ab_cars.field_analyses[()])


    # EVV 2-scan

    pulse_ir_1 = make_gaussian_pulse(tc=100.0, cf=3000.0, dev=100.0, maxstr=1.0e-5,
                                     wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0), id=1)

    pulse_ir_2 = make_gaussian_pulse(tc=600.0, cf=1000.0, dev=100.0, maxstr=1.0e-5,
                                     wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0), id=2)

    pulse_uvvis_1 = make_gaussian_pulse(tc=1100.0, cf=15000.0, dev=100.0, maxstr=1.0e-5,
                                        wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0), id=3)

    pulses = (pulse_ir_1, pulse_ir_2, pulse_uvvis_1)

    field_evv = ElectricField(pulses)

    detector_evv = SpecDetector(detection_method='int',
                              detection_polarization=(1.0, 0.0, 0.0),
                              detection_limited_to = 'only_specified',
                              interaction_filter=[[-1, 2, 3]])

    # Push one carrier freq
    scan_obj_a = ScanObject('pulse', 'cf', id=1, coeff=1.0)
    scan_range_a = [50.0* i for i in range(10)]
    scan_a = SpecScan(scan_objs=(scan_obj_a,), range=scan_range_a)

    # Push another carrier freq
    scan_obj_b = ScanObject('pulse', 'cf', id=2, coeff=1.0)
    scan_range_b = [50.0* i for i in range(10)]
    scan_b = SpecScan(scan_objs=(scan_obj_b,), range=scan_range_b)

    exp_evv = VibExperiment(field=field_evv,
                          detector=detector_evv,
                          scans=(scan_a, scan_b),
                          tol_n_dev_t=2.0,
                          tol_n_dev_w=2.0,
                          thres_waiting_time=600.0)

    present_field_analysis(exp_evv.field_analyses[0,0])

    # Some attribute testing given a light touch since already covered by respective class tests
    assert len(exp_evv.field.pulses) == 3
    assert exp_evv.detector.detection_method == 'int'
    assert len(exp_evv.scans) == 2

    # Assertions about post-init attributes: When these are obtained using VibExperiment methods, these assertions
    # also serve to cover those methods
    assert exp_evv.dim == 2

    # CARS

    pulse_uvvis_1 = make_gaussian_pulse(tc=0.0, cf=15000.0, dev=120.0, maxstr=1.0e-5,
                                        wv=(0.0, 0.02, 1.0), pol=(1.0, 0.0, 0.0), id=1)

    pulse_uvvis_2 = make_gaussian_pulse(tc=350.0, cf=15000.0, dev=10.0, maxstr=1.0e-5,
                                        wv=(0.0, 0.012, 1.0), pol=(1.0, 0.0, 0.0), id=2)

    pulse_uvvis_3 = make_gaussian_pulse(tc=750.0, cf=15000.0, dev=120.0, maxstr=1.0e-5,
                                        wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0), id=3)

    pulses = (pulse_uvvis_1, pulse_uvvis_2, pulse_uvvis_3)

    field_cars = ElectricField(pulses)

    detector_cars = SpecDetector(detection_method='freq',
                                 detection_polarization=(1.0, 0.0, 0.0),
                                 detection_limited_to='same_order',
                                 interaction_filter=[[1, -2, 3]])

    exp_cars = VibExperiment(field=field_cars,
                          detector=detector_cars,
                          scans=(),
                          tol_n_dev_t=3.0,
                          tol_n_dev_w=3.0,
                          thres_waiting_time=600.0,)

    present_field_analysis(exp_cars.field_analyses[()])

    # "SFG-style"

    pulse_ir_1 = make_gaussian_pulse(tc=100.0, cf=1500.0, dev=100.0, maxstr=1.0e-5,
                                     wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0), id=1)


    pulse_uvvis_1 = make_gaussian_pulse(tc=500.0, cf=15000.0, dev=10.0, maxstr=1.0e-5,
                                        wv=(0.02, 0.0, 1.0), pol=(0.0, 1.0, 0.0), id=2)

    pulses = (pulse_ir_1, pulse_uvvis_1)

    field_sfg = ElectricField(pulses)

    detector_sfg = SpecDetector(detection_method='freq',
                                 detection_polarization=(0.0, 0.0, 1.0),
                                 detection_limited_to='same_order',
                                 interaction_filter=[[1, 2]])

    exp_sfg = VibExperiment(field=field_sfg,
                          detector=detector_sfg,
                          scans=(),
                          tol_n_dev_t=3.0,
                          tol_n_dev_w=3.0,
                          thres_waiting_time=600.0,)

    present_field_analysis(exp_sfg.field_analyses[()])
