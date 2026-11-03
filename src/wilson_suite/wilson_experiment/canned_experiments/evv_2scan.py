from wilson_suite.wilson_experiment.experiment_abstractions import (make_gaussian_pulse, ElectricField, SpecDetector,
                                                                    ScanObject, SpecScan, VibExperiment)

# Canned experiment: An EVV experiment with two scans, one for each IR carrier frequency
def evv_2scan():

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
                                detection_limited_to='only_specified',
                                interaction_filter=[[-1, 2, 3]])

    # Push one carrier freq
    scan_obj_a = ScanObject('pulse', 'cf', id=1, coeff=1.0)
    scan_range_a = [50.0 * i for i in range(10)]
    scan_a = SpecScan(scan_objs=(scan_obj_a,), range=scan_range_a)

    # Push another carrier freq
    scan_obj_b = ScanObject('pulse', 'cf', id=2, coeff=1.0)
    scan_range_b = [50.0 * i for i in range(20)]
    scan_b = SpecScan(scan_objs=(scan_obj_b,), range=scan_range_b)

    return VibExperiment(field=field_evv,
                            detector=detector_evv,
                            scans=(scan_a, scan_b),
                            tol_n_dev_t=2.0,
                            tol_n_dev_w=2.0,
                            thres_waiting_time=600.0)