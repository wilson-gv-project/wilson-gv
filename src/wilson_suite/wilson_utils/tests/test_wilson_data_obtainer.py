import wilson_suite as ws
from wilson_suite.wilson_main.abstractions import DataOriginInfo
import numpy as np

def test_getting_data():
    """
    complete_info_keys = ['cff', 'anharmonic_states', 'nc_sqrt_eigval', 'dipgrad', 'B', 'polgrad', 'coriolis', 'polhess', 'qff', 'diphess']
    compl_data = complete_info_keys + ['harmonic_states']
    """
    from ...fixtures import evv_experiment
    from wilson_suite.wilson_utils.paths import SUITE_ROOT

    evv_exp = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)

    sim = ws.main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(evv_exp)
    sim.addTerms(terms=terms) # terms

    mol_system = ws.main.abstractions.MolecularSystem(name='h2o', natoms=3)
    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='none')
    # dict_keys(['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff', 'nc_sqrt_eigval', 'anharmonic_states'])
    # vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='anharm')
    # dict_keys(['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff', 'qff', 'B', 'coriolis', 'nc_sqrt_eigval'])
    # vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='full')
    # dict_keys(['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff', 'hess', 'qff', 'B', 'coriolis'])
    
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vib_ana)

    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian',
                                                     lvl_theory='HF', 
                                                     basis_set='STO-3G', 
                                                     base_file_loc=SUITE_ROOT+'/../data_for_tests/g16_h2o_HF_STO3G.out')
    calc_setup_blank = ws.main.abstractions.DataOriginInfo()
    sim.addPropEvalSetup(eval_uniform=calc_setup_blank)
    
    sim.setPropsAndMaxStateLvl() # setting up self.props/sim.props
    sim.dressPropsWithSetup()

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

    complete_info_keys = ['cff', 'anharmonic_states', 'nc_sqrt_eigval', 'dipgrad', 'B', 'polgrad', 'coriolis', 'polhess', 'qff', 'diphess']
    """
    data_orig_g16 = DataOriginInfo(source_type='gaussian',
                                   base_file_loc='/home/vlev/monorepo/src/../data_for_tests/g16_h2o_HF_STO3G.out')
    rq_none_keys = ['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff', 'nc_sqrt_eigval', 'anharmonic_states']
    # request_dict = dict.fromkeys(rq_none_keys, data_orig_g16)

    rq_anh_keys = ['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff', 'qff', 'B', 'coriolis', 'nc_sqrt_eigval']
    # rq_anharm = dict.fromkeys(rq_anh_keys, data_orig_g16)

    rq_full_keys = ['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff', 'hess', 'qff', 'B', 'coriolis']
    
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

    complete_info_keys = ['cff', 'anharmonic_states', 'nc_sqrt_eigval', 'dipgrad', 'B', 'polgrad', 'coriolis', 'polhess', 'qff', 'diphess']
    complete_rq = dict.fromkeys(complete_info_keys, data_orig_g16)
    compl_data = wilson_data_obtainer(complete_rq)

    assert sorted(list(compl_data.keys())) == sorted(complete_info_keys+['harmonic_states'])
    assert all(v is not None for v in compl_data.values())

    # pickling data dict - testing save_obtained_data
    from wilson_suite.wilson_utils.paths import SUITE_ROOT
    filename = '/test_compl_data.pkl'
    filepath = SUITE_ROOT+'/wilson_suite/wilson_utils/tests'+ filename

    ws.utils.save_obtained_data(compl_data, format='pkl', filename=filepath)

    unpkl_compl_data = ws.utils.serialization.unpickle_smth_from(filepath)
    assert_equal(unpkl_compl_data, compl_data)


def assert_equal(a, b):
    """
    assert equality of complete obtained data arrays

    some values are np.ndarrays, others are dicts with float values
    """

    assert type(a) is type(b)

    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_equal(a[k], b[k])

    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)

    # elif isinstance(a, (list, tuple)):
    #     print('a, (list, tuple)', a)
    #     assert len(a) == len(b)
    #     for x, y in zip(a, b):
    #         assert_equal(x, y)

    else:
        # float comparison
        assert a == b