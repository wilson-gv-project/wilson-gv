import wilson_suite as ws

pulse_ir_1 = ws.experiment.abstractions.emPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=1)
pulse_ir_2 = ws.experiment.abstractions.emPulse('impulsive', 1.0e-5, tc = 100.0, cf=None, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=2)
pulse_uvvis_1 = ws.experiment.abstractions.emPulse('ideal', 1.0e-5, tc = 120.0, cf=0.0, cf_uv=0.072, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=3)

pulses = [pulse_ir_1, pulse_ir_2, pulse_uvvis_1]

field_a = ws.experiment.abstractions.electricField(pulses)
order = len(pulses)

epochs = field_a.findEpochs()

detector_a = ws.experiment.abstractions.specDetector('freq', detector_location=[0.0, 0.0, 1.0], detection_polarization=[0.0, 0.0, 1.0], detection_range=[0.003 + 0.0001*i for i in range(101)], wv_filter=[{1: [-1], 2: [1], 3: [1]}]) #, {1: [-1], 2: [1], 3: [1]} 

# Push one carrier freq
scan_obj_a = [['pulse', 1, 'cf', 1.0], ['detector', 0, 'detection_range', 1.0]]
scan_range_a = [0.0001*i for i in range(101)]
scan_a = ws.experiment.abstractions.specScan(scan_obj_a, scan_range_a)

experiment_a = ws.experiment.abstractions.vibExperiment(order, field_a, detector_a, [scan_a], magn_conditions=[[-1, 2]])


calc_setup = ws.main.abstractions.externalCalcSetup(program='gaussian', lvl_theory='B3LYP', basis='cc_pVQZ')

sim = ws.main.abstractions.wilsonSimulation()

sim.addExperiment(experiment_a)
sim.getTerms(ws.derive.main.get_fully_enhanced_terms)
sim.addSystem(ws.main.abstractions.molecularSystem(name='ACAC'))
sim.addVibAnaSetup(ws.main.abstractions.vibAnaSetup(vib_regime='GVPT2', vibana_prop_need='anharm', allow_skip_eigvec=True, external_fill_from=calc_setup))
sim.addPropEvalSetup(eval_uniform=calc_setup)

axis1 = ws.main.abstractions.spectralAxis({1: 1})
axis2 = ws.main.abstractions.spectralAxis({1: 1, 2: -1})
start = {1: 250, 2: 100}
end = {1: 3850, 2: 7550}
spacer = {1: 3.8, 2: 3.8}
spec_axes = ws.main.abstractions.spectralGrid({1: axis1, 2: axis2}, range_style='uniform', start=start, end=end, spacer=spacer)
evi = {'dynrange': 500, 'Gamma': 4.7, 'diag_margin': 5., 'maxmax': None}
rndi = {'num_level_ticks': 15}
eval_setup = ws.main.abstractions.specEvalSetup(axes=spec_axes, ev_info=evi, rnd_info=rndi)
sim.addSpecEvalSetup(eval_setup)

sim.findPropsAndMaxStateLvl()
sim.dressPropsWithSetup()
sim.makeCalculationBatches()
sim.getResultsFromCalculationBatches(source_type='vault', source_loc=ws.intensities.utils.get_package_root() + '/tests/test_database/mini_files_database.csv' )

sim.evaluate(ws.intensities.spectrum.wilsonmain_integration.spectrum2D)

sim.render(ws.intensities.wilsonmain_render_integration.render_spectrum)

'''



#fully_enhanced_terms = wdrv_main.get_fully_enhanced_terms(experiment_a)

for i in sim.terms:

	for j in sim.terms[i]:
	
		print('Anharmonicity order', j)
	
		if len(sim.terms[i][j]) == 0:
			print('No terms')
			print(' ')
			
		else:
			for k in sim.terms[i][j]:
				print('')
				print('New term')
				k.present()
				print('')
	 
'''
