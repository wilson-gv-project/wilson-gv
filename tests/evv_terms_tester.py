import wilson_suite as ws

pulse_ir_1 = ws.experiment.abstractions.emPulse('ideal', 1.0e-5, tc=50.0, cf=0.00, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=1)
pulse_ir_2 = ws.experiment.abstractions.emPulse('impulsive', 1.0e-5, tc=100.0, cf=None, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=2)
pulse_uvvis_1 = ws.experiment.abstractions.emPulse('ideal', 1.0e-5, tc=120.0, cf=0.0, cf_uv=0.072, wv=[0.0, 0.0, 1.0],
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

experiment_a = ws.experiment.abstractions.vibExperiment(order, field_a, detector_a, [scan_a], magn_conditions=[[-1, 2]])

fully_enhanced_terms = ws.derive.main.get_fully_enhanced_terms(experiment_a)

for i in fully_enhanced_terms:

	print('i', i)

	for j in fully_enhanced_terms[i]:

		print('j', j)

		if len(fully_enhanced_terms[i][j]) == 0:
			print('No terms')
			print(' ')

		else:
			for k in fully_enhanced_terms[i][j]:
				print('k', k)
				print('')
				print('New term')
				k.present()
				print('')

