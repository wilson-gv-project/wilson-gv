import wilson_suite as ws

pulse_ir_1 = ws.experiment.abstractions.EmPulse('ideal', 1.0e-5, tc=50.0, cf=0.00, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=1)
pulse_ir_2 = ws.experiment.abstractions.EmPulse('impulsive', 1.0e-5, tc=100.0, cf=None, wv=[0.0, 0.0, 1.0],
                                                pol=[0.0, 0.0, 1.0], id=2)
pulse_uvvis_1 = ws.experiment.abstractions.EmPulse('ideal', 1.0e-5, tc=120.0, cf=0.0, cf_uv=0.072, wv=[0.0, 0.0, 1.0],
                                                   pol=[0.0, 0.0, 1.0], id=3)

pulses = [pulse_ir_1, pulse_ir_2, pulse_uvvis_1]

field_a = ws.experiment.abstractions.ElectricField(pulses)
order = len(pulses)

epochs = field_a.findEpochs()

detector_a = ws.experiment.abstractions.SpecDetector('freq', detector_location=[0.0, 0.0, 1.0],
                                                     detection_polarization=[0.0, 0.0, 1.0],
                                                     detection_range=[0.003 + 0.0001 * i for i in range(101)],
                                                     wv_filter=[
                                                         {1: [-1], 2: [1], 3: [1]}])  # , {1: [-1], 2: [1], 3: [1]}

# Push one carrier freq
scan_obj_a = [['pulse', 1, 'cf', 1.0], ['detector', 0, 'detection_range', 1.0]]
scan_range_a = [0.0001 * i for i in range(101)]
scan_a = ws.experiment.abstractions.SpecScan(scan_obj_a, scan_range_a)

experiment_a = ws.experiment.abstractions.VibExperiment(order, field_a, detector_a, [scan_a], magn_conditions=[[-1, 2]])

fully_enhanced_terms = ws.derive.main.get_fully_enhanced_terms(experiment_a)

print('\n\n ### Printing fully_enhanced_terms...\n')
print('type(fully_enhanced_terms)', type(fully_enhanced_terms))

for i in fully_enhanced_terms:

	print('\n  --> i in fully_enhanced_terms: i is', i)
	print('  i is an int, an element of anharm_orders dict - sum of anharm orders'
		  '| anharm orders {1: [(1, 0), (0, 1)], 0: [(0, 0)]}')
	print('  value for key i is a dict')

	for j in fully_enhanced_terms[i]:

		print('\n  --> j in fully_enhanced_terms[i]: j is', j)
		print('  j is a tuple ')
		print('  value for key j is a dict')

		print('fully_enhanced_terms[i]', type(fully_enhanced_terms[i]))
		print('fully_enhanced_terms[i][j]', type(fully_enhanced_terms[i][j]))

		if len(fully_enhanced_terms[i][j]) == 0: # it's a list!
			print('\nlen(fully_enhanced_terms[i][j]) == 0  ---->  No terms')
			print(' ')

		else:
			# fully_enhanced_terms[i][j] is a list! of wilson_derive.abstractions.VibPerturbedTerm objects
			for k in fully_enhanced_terms[i][j]:
				print('\n  --> k in fully_enhanced_terms[i][j]: k is', k)
				print('  k is an element of', type(fully_enhanced_terms[i][j]))

				print('')
				print('New term')
				k.present()
				print('')

print('===========================================================================')
print('fully_enhanced_terms')
print()

for i in fully_enhanced_terms:
	for j in fully_enhanced_terms[i]:
		print(i, j, fully_enhanced_terms[i][j], '---', len(fully_enhanced_terms[i][j]))

one_term = fully_enhanced_terms[1][(1,0)][0]
print()
print([i for i in dir(one_term) if '_' not in i])
print()

one_term.present()

print()
print(one_term.props)
print()
print(one_term.freqterms)
print()
print(one_term.res)

print('===========================================================================')
print('fully_enhanced_terms\n')
print(fully_enhanced_terms)
