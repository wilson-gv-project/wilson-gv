import wilson_suite as ws
from wilson_suite.wilson_main.abstractions import DataOriginInfo

def test_getting_data():
    """
    complete_info_keys = ['cff', 'anharmonic_states', 'nc_sqrt_eigval', 'dipgrad', 'B', 'polgrad', 'coriolis', 'polhess', 'qff', 'diphess']
    compl_data = complete_info_keys + ['harmonic_states']

    sim.getResults(obtainer=wilson_data_obtainer)    
    anharm own analysis - ['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff', 'qff', 'B', 'coriolis', 'nc_sqrt_eigval']
    complete_info_keys = ['cff', 'anharmonic_states', 'nc_sqrt_eigval', 'dipgrad', 'B', 'polgrad', 'coriolis', 'polhess', 'qff', 'diphess']
    """
    from ...fixtures import evv_experiment

    evv_exp = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)
    mol_system = ws.main.abstractions.MolecularSystem(name='h2o', natoms=3)
    calc_setup_blank = ws.main.abstractions.DataOriginInfo()

    # -------- vibana_own_analysis='none'
    sim = ws.main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(evv_exp)
    sim.addTerms(terms=terms) # terms

    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='none')
    # dict_keys(['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff', 'nc_sqrt_eigval', 'anharmonic_states'])
    
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vib_ana)

    sim.addPropEvalSetup(eval_uniform=calc_setup_blank)
    
    sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    sim.dressPropsWithSetup()
    
    rq_none_keys = list(sim.requestData().keys())

    # -------- vibana_own_analysis='anharm'
    sim = ws.main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(evv_exp)
    sim.addTerms(terms=terms) # terms

    vib_ana1 = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='anharm')
    # dict_keys(['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff', 'qff', 'B', 'coriolis', 'nc_sqrt_eigval'])
    
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vib_ana1)

    sim.addPropEvalSetup(eval_uniform=calc_setup_blank)
    
    sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    sim.dressPropsWithSetup()

    rq_anh_keys = list(sim.requestData().keys())

    # -------- vibana_own_analysis='full'
    sim = ws.main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(evv_exp)
    sim.addTerms(terms=terms) # terms

    vib_ana1 = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='full')
    # dict_keys(['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff', 'qff', 'B', 'coriolis', 'nc_sqrt_eigval'])
    
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vib_ana1)

    sim.addPropEvalSetup(eval_uniform=calc_setup_blank)
    
    sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    sim.dressPropsWithSetup()

    rq_full_keys = list(sim.requestData().keys())

    missing_from_A = set(rq_anh_keys) - set(rq_none_keys)
    missing_from_B = set(rq_none_keys) - set(rq_anh_keys)
    missing_from_C = set(rq_anh_keys) - set(rq_full_keys)
    missing_from_D = set(rq_full_keys) - set(rq_anh_keys)

    # missing_from rq_none but in rq_anharm
    assert missing_from_A == {'coriolis', 'qff', 'B'}
    # missing_from rq_anharm but in rq_none
    assert missing_from_B == {'anharmonic_states'}
    # missing_from rq_full but in rq_anharm
    assert missing_from_C == {'nc_sqrt_eigval'}
    # missing_from rq_anharm but in rq_full
    assert missing_from_D == {'hess'}

    union_none_anharm = list(set(rq_none_keys) | set(rq_anh_keys))
    print(union_none_anharm, len(union_none_anharm))
    print(len(rq_none_keys), len(rq_anh_keys), len(rq_full_keys))

    
    data_orig_g16 = DataOriginInfo(source_type='gaussian',
                                base_file_loc='/home/vlev/monorepo/src/../data_for_tests/g16_h2o_HF_STO3G.out')
    complete_info_keys = ['cff', 'anharmonic_states', 'nc_sqrt_eigval', 'dipgrad', 'B', 'polgrad', 'coriolis', 'polhess', 'qff', 'diphess']
    complete_rq = dict.fromkeys(complete_info_keys, data_orig_g16)
    
    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    compl_data = wilson_data_obtainer(complete_rq)

    assert sorted(list(compl_data.keys())) == sorted(complete_info_keys+['harmonic_states', 'cff_rc', 'qff_rc'])
    assert all(v is not None for v in compl_data.values())
