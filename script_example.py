import wilson_suite as ws
import wilson_suite.wilson_experiment.experiment_abstractions as wexp


vib_regime = 'GVPT2'
vibana_own_analysis = 'anharm'
dynamic_range = 1000


def run():

    molecular_system = ws.main.abstractions.MolecularSystem(name='formaldehyde', natoms=4)
    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian',
                                                     base_file_loc='data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')
    
    """
    identifier = (molecular_system.name + '_' + 
                molecular_system.conformer + '_' + 
                calc_setup.lvl_theory.replace('-', '_') + '_' + 
                calc_setup.basis_set.replace('-', '_') + f"_{vib_regime}" + 
                f'_dr{str(dynamic_range).replace('.', 'p')}')
    """
    
    # ---------  VibExperiment setup 
    pulse_ir_1 = wexp.make_impulsive_gaussian_pulse(tc=50.0, cf=0.0, cf_uv=0.0,
                                                           maxstr=1.0e-5, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0), id=1)

    pulse_ir_2 = wexp.make_impulsive_gaussian_pulse(tc=100.0, cf=0.0, cf_uv=0.0,
                                                           maxstr=1.0e-5, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0), id=2)

    pulse_uvvis_1 = wexp.make_impulsive_gaussian_pulse(tc=120.0, cf=0.0, cf_uv=0.072,
                                                           maxstr=1.0e-5, wv=(0.0, 0.0, 1.0), pol=(1.0, 0.0, 0.0), id=3)

    pulses = (pulse_ir_1, pulse_ir_2, pulse_uvvis_1)

    field_a = wexp.ElectricField(pulses)

    detector_a = wexp.SpecDetector(detection_method='freq',
                                   detector_location=(0.0, 0.0, 1.0),
                                   detection_polarization=(1.0, 0.0, 0.0),
                                   detection_range=[0.003 + 0.0001 * i for i in range(101)],
                                   wv_filter=[{1: -1, 2: 1, 3: 1}])

    # Push one carrier freq
    scan_obj_a = wexp.ScanObject('pulse', 'cf', id=1, coeff=1.0)
    scan_obj_b = wexp.ScanObject('detector', 'detection_range', id=0, coeff=1.0)
    scan_range_a = [0.0001 * i for i in range(101)]
    scan_a = wexp.SpecScan(scan_objs=(scan_obj_a, scan_obj_b), range=scan_range_a)

    experiment = wexp.VibExperiment(field=field_a, detector=detector_a, scans=(scan_a,), magn_conditions=((-1, 2),),)


    # ---------  SpecEvalSetup 
    window_bounds = {
                        "A": (50.0, 3850.0),
                        "B": (0.0, 3850.0)
                    }
    spec_box = ws.intensities.amplitudes.spectrum_composition.Box(bounds=window_bounds)
    spec_window = ws.intensities.amplitudes.spectrum_composition.SpectralWindow(box=spec_box)

    # SpectralAxisSet convenient builder - new axes labels to collection of independent vars, 
    #                                                      here - laser pulses with chosen signs
    spec_axes = ws.utils.some_reprs.make_SpectralAxisSet({"A": [1], "B": [-1, 2]})

    eval_info = ws.main.spectrum_abstractions.EvaluationInfo(Gamma=10.0, Gamma_unit='cm-1', 
                                                            margins={'diag_margin': 5.0}, 
                                                            spectral_window=spec_window, 
                                                            grid_resolution={'A': 20, 'B': 20}, dynamic_range=1000, 
                                                            spectral_axes=spec_axes, box_range_safety_margin=0.1, 
                                                            scale_wrt_max_intensity=False, minimum_box_padding=30.0, 
                                                            apply_exp_magn_conditions_eval=True, 
                                                            apply_exp_magn_conditions_render=False, 
                                                            exp_magn_conditions=None, magn_conditions_margin=0.1)

    mgn_conds = None
    if eval_info.apply_exp_magn_conditions_eval and eval_info.apply_exp_magn_conditions_render:
        mgn_conds = 'render'
    if eval_info.apply_exp_magn_conditions_eval and not eval_info.apply_exp_magn_conditions_render:
        mgn_conds = 'eval'

    rnd_info = ws.main.spectrum_abstractions.RenderingInfo()

    eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=eval_info, rnd_info=rnd_info)


    # ---------  VibAnaSetup 
    vib_ana = ws.main.abstractions.VibAnaSetup(system=molecular_system, 
                                                regime=vib_regime, 
                                                vibana_own_analysis=vibana_own_analysis)

    # ---------- WilsonSimulation
    sim = ws.main.workflow_abstractions.WilsonSimulation()


    # -- setting attributes
    sim.addExperiment(experiment=experiment)

    DERIVED_EVV_TERMS = ws.derive.derive.get_fully_enhanced_terms(experiment=experiment)
    sim.addTerms(terms=DERIVED_EVV_TERMS)
    sim.addSystem(system=molecular_system)
    sim.addVibAnaSetup(vib_ana)
    sim.addPropEvalSetup(eval_uniform=calc_setup)
    sim.addSpecEvalSetup(eval_setup)

    sim.setPropsAndMaxStateLvl()
    sim.dressPropsWithSetup()
    sim.setAxisChoiceAndTranslateTerms(eval_setup.ev_info.spectral_axes)

    #### ---- GETTING CALC DATA FROM QC OUTPUTS
    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    sim.getResults(obtainer=wilson_data_obtainer)

    # ------- preparing states
    sim.vib_ana_setup.exclude_modes = []
    sim.vib_ana_setup.set_include_modes_list()


    ws.main.main_functions.do_anharmonic_analysis(vib_ana=sim.vib_ana_setup, 
                                                props=sim.props, 
                                                anharmonic_analyzer=ws.intensities.anharmonic_treatment.anharmonic_analyzer.anharm_analyzer_data)

    ## -- pre-evaluation config
    if mgn_conds is not None:
        sim.apply_exp_magn_conditions(where=mgn_conds) # options: eval, render

    ## -- evaluation
    sim.evaluate()

    ## -- further analysis

    return sim

if __name__ == '__main__':
    sim = run()
    print(sim.spec)
