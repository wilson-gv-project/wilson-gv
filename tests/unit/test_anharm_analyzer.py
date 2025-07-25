from wilson.utils.debug import printtest

def evv_experiment():
    import wilson_suite as ws
    pulse_ir_1 = ws.experiment.abstractions.EmPulse('ideal', 1.0e-5, tc = 50.0, cf=0.00, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=1)
    pulse_ir_2 = ws.experiment.abstractions.EmPulse('impulsive', 1.0e-5, tc = 100.0, cf=None, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=2)
    pulse_uvvis_1 = ws.experiment.abstractions.EmPulse('ideal', 1.0e-5, tc = 120.0, cf=0.0, cf_uv=0.072, wv=[0.0, 0.0, 1.0], pol=[0.0, 0.0, 1.0], id=3)

    pulses = [pulse_ir_1, pulse_ir_2, pulse_uvvis_1]

    field_a = ws.experiment.abstractions.ElectricField(pulses)
    order = len(pulses)

    field_a.findEpochs()

    detector_a = ws.experiment.abstractions.SpecDetector('freq', detector_location=[0.0, 0.0, 1.0],
                                                        detection_polarization=[0.0, 0.0, 1.0],
                                                        detection_range=[0.003 + 0.0001*i for i in range(101)],
                                                        wv_filter=[{1: [-1], 2: [1], 3: [1]}]) #, {1: [-1], 2: [1], 3: [1]}

    # Push one carrier freq
    scan_obj_a = [['pulse', 1, 'cf', 1.0], ['detector', 0, 'detection_range', 1.0]]
    scan_range_a = [0.0001*i for i in range(101)]
    scan_a = ws.experiment.abstractions.SpecScan(scan_obj_a, scan_range_a)
    experiment_a = ws.experiment.abstractions.VibExperiment(order, field_a, detector_a, [scan_a], magn_conditions=[[-1, 2]])
    return experiment_a

def test_anharm_analyzer():
    from wilson_intensities.wilson.spectrum.anharmonic_analyzer import anharm_analyzer
    import wilson_suite as ws
    mol_system = ws.main.abstractions.MolecularSystem(name='FORM', natoms=4)
    calc_setup = ws.main.abstractions.ExternalCalcSetup(program='gaussian', lvl_theory='B3LYP', basis='cc_pVQZ')
    # for gvpt2 need both vib_regime and vibana_prop_need='all', otherwise no coriolis and rot
    # omg, need vibana_prop_need='anharm' for nc_sqrt_eigval to be retrieved
    vibana = ws.main.abstractions.VibAnaSetup(system=mol_system, vib_regime='GVPT2', vibana_prop_need='anharm',
                                                    allow_skip_eigvec=True, external_fill_from=calc_setup)
    printtest(vibana.vibana_prop_need)

    experiment_a = evv_experiment()

    sim = ws.main.abstractions.WilsonSimulation()
    sim.addExperiment(experiment_a)
    sim.getTerms(ws.derive.main.get_fully_enhanced_terms) # here terms are derived
    
    sim.addSystem(mol_system)
    sim.addPropEvalSetup(eval_uniform=calc_setup)
    sim.addVibAnaSetup(vib_ana_setup=vibana)
    sim.addPropEvalSetup(eval_uniform=calc_setup)
    # should take care of when props are there, because needed for vib analyzer
    sim.findPropsAndMaxStateLvl() # setting up self.props/sim.props
    printtest([i.triv_name for i in sim.props])
    sim.dressPropsWithSetup()
    sim.makeCalculationBatches()
    sim.getResultsFromCalculationBatches(source_type='vault',
                                        source_loc=ws.intensities.utils.get_package_root()
                                                    + '/../tests/test_database/mini_files_database.csv' )
    printtest(sim.props)
    printtest(sim.props[0].__dict__.keys())
    printtest([i.triv_name for i in sim.props])
    for i in sim.props:
        if i.triv_name == 'cff':
            printtest(i.vals.shape)
    printtest(sim.vib_ana_setup.regime)
    printtest(sim.vib_ana_setup.vibana_prop_need)
    
    # ['prop_spec', 'triv_name', 'vals', 'in_basis', 'in_units', 
    # 'system', 'calc_setup', 'target_basis', 'target_units']
    printtest(sim.vib_ana_setup.nc_sqrt_eigval)
    printtest(len(sim.vib_ana_setup.nc_sqrt_eigval))

    sim.vib_ana_setup.doAnharmonicAnalysis(sim.props, anharmonic_analyzer=anharm_analyzer)