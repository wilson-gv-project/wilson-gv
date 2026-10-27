import wilson_suite as ws


def test_request_data_per_vibana_mode():
    """
    The names WilsonSimulation.requestData asks for in each vibana_own_analysis mode of VibAnaSetup (the old
    workflow). Moved from wilson_utils/tests/test_wilson_data_obtainer.py, which tests the obtainer. Needs the
    evv_experiment fixture for the terms.
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
