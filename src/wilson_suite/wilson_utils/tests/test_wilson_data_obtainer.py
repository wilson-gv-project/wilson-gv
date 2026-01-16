import wilson_suite as ws
from wilson_suite.wilson_main.abstractions import DataOriginInfo

def test_getting_data():
    from ...fixtures import evv_experiment
    from wilson_suite.wilson_utils.paths import SUITE_ROOT

    evv_exp = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)

    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian',
                                                     lvl_theory='HF', 
                                                     basis_set='STO-3G', 
                                                     base_file_loc=SUITE_ROOT+'/../data_for_tests/g16_h2o_HF_STO3G.out')
    calc_setup_blank = ws.main.abstractions.DataOriginInfo()

    sim = ws.main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(evv_exp)
    sim.addTerms(terms=terms) # terms

    mol_system = ws.main.abstractions.MolecularSystem(name='h2o', natoms=3)

    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='none')
    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='anharm')
    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='full')
    
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vib_ana)
    sim.addPropEvalSetup(eval_uniform=calc_setup_blank)
    
    sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    sim.dressPropsWithSetup()

    print(sim.requestData().keys())

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    """
    sim.getResults(obtainer=wilson_data_obtainer)

        rq = {'dipgrad': DataOriginInfo(source_type='gaussian', lvl_theory='HF', basis_set='STO-3G', base_file_loc='/home/vlev/monorepo/src/../data_for_tests/g16_h2o_HF_STO3G.out'), 
          'polhess': DataOriginInfo(source_type='gaussian', lvl_theory='HF', basis_set='STO-3G', base_file_loc='/home/vlev/monorepo/src/../data_for_tests/g16_h2o_HF_STO3G.out'), 
          'polgrad': DataOriginInfo(source_type='gaussian', lvl_theory='HF', basis_set='STO-3G', base_file_loc='/home/vlev/monorepo/src/../data_for_tests/g16_h2o_HF_STO3G.out'), 
          'diphess': DataOriginInfo(source_type='gaussian', lvl_theory='HF', basis_set='STO-3G', base_file_loc='/home/vlev/monorepo/src/../data_for_tests/g16_h2o_HF_STO3G.out'), 
          'cff': DataOriginInfo(source_type='gaussian', lvl_theory='HF', basis_set='STO-3G', base_file_loc='/home/vlev/monorepo/src/../data_for_tests/g16_h2o_HF_STO3G.out'), 
          'nc_sqrt_eigval': DataOriginInfo(source_type='gaussian', lvl_theory='HF', basis_set='STO-3G', base_file_loc='/home/vlev/monorepo/src/../data_for_tests/g16_h2o_HF_STO3G.out'), 
          'anharmonic_states': DataOriginInfo(source_type='gaussian', lvl_theory='HF', basis_set='STO-3G', base_file_loc='/home/vlev/monorepo/src/../data_for_tests/g16_h2o_HF_STO3G.out')}
    
    anharm own analysis - ['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff', 'qff', 'B', 'coriolis', 'nc_sqrt_eigval']
    """
    data_orig_g16 = DataOriginInfo(source_type='gaussian',base_file_loc='/home/vlev/monorepo/src/../data_for_tests/g16_h2o_HF_STO3G.out')
    rq_none = ['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff', 'nc_sqrt_eigval', 'anharmonic_states']
    rq = dict.fromkeys(rq_none, data_orig_g16)

    anh_keys = ['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff', 'qff', 'B', 'coriolis', 'nc_sqrt_eigval']
    rq_anharm = dict.fromkeys(anh_keys, data_orig_g16)
    ra = wilson_data_obtainer(rq_anharm)
    # print(ra)
    
    # r = wilson_data_obtainer(rq)
    # print(r)

    rq_full = ['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff', 'hess', 'qff', 'B', 'coriolis']
    print('len(rq_full)', len(rq_full))

    union = list(set(rq_none) | set(rq_anharm))
    intersection = list(set(rq_none) & set(rq_anharm))
    print('rq_none, rq_anharm', len(rq_none), len(rq_anharm))
    print(len(intersection))
    
    missing_from_A = set(rq_anharm) - set(rq_none)
    missing_from_B = set(rq_none) - set(rq_anharm)
    missing_from_C = set(rq_anharm) - set(rq_full)
    missing_from_D = set(rq_full) - set(rq_anharm)

    # missing_from rq_none but in rq_anharm
    assert missing_from_A == {'B', 'normal_modes', 'atoms', 'qff', 'coriolis'}
    # missing_from rq_anharm but in rq_none
    assert missing_from_B == {'anharmonic_states'}
    # missing_from rq_full but in rq_anharm
    assert missing_from_C == {'nc_sqrt_eigval', 'atoms', 'normal_modes'}
    # missing_from rq_anharm but in rq_full
    assert missing_from_D == {'hess'}

    print()
    print(union, len(union))