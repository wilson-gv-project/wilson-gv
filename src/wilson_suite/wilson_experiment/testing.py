from . import experiment_abstractions as abst

pulse_ir_1 = abst.emPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=1)
pulse_ir_2 = abst.emPulse('ideal', 1.0e-5, tc = 150.0, cf=0.00, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=2)
pulse_uvvis_1 = abst.emPulse('ideal', 1.0e-5, tc = 200.0, cf=0.0, cf_uv=0.072, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=3)
pulse_uvvis_2 = abst.emPulse('ideal', 1.0e-5, tc = 220.0, cf=0.0, cf_uv=0.072, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=4)


field_a = abst.electricField([pulse_ir_1, pulse_ir_2, pulse_uvvis_1])

epochs = field_a.find_epochs()

print(epochs)

detector_a = abst.specDetector('freq', detector_location=[0.0, 0.0, 1.0], detection_polarization=[0.0, 0.0, 1.0],
                          detection_range=[0.003 + 0.0001*i for i in range(101)], wv_filter=[{1: [1], 2: [-1], 3: [1]}])

# Push one carrier freq
scan_obj_a = [['pulse', 1, 'cf', 1.0]]
scan_range_a = [0.0001*i for i in range(101)]
scan_a = abst.specScan(scan_obj_a, scan_range_a)

# Pull both carrier freqs apart
#scan_obj_a_2 = [['pulse', 1, 'cf', -0.5], ['pulse', 2, 'cf', 0.5]]

order = 3

experiment_a = abst.vibExperiment(order, field_a, detector_a, [scan_a])