import numpy as np
import wilson_suite
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
from wilson_suite.wilson_intensities.utils.utils import pairwise_differences

import logging
from wilson_suite.wilson_utils.logger import setup_logger
setup_logger("wilson", level=logging.WARNING)


def test_amplitude_mock_singlepoint_one_elterm():

    from wilson_suite.wilson_utils.useful_shortcuts import bare_wsim_for_EVVpGVPT2, makeSpecSetup2D
    from wilson_suite.wilson_main.abstractions import VibAnaSetup, DataOriginInfo, MolecularSystem, SpectralAxis, SpecEvalSetup
    from wilson_suite.wilson_utils.abstractions import VibState
    sim_conf_dict = {'vib_ana_setup': VibAnaSetup(regime='GVPT2', vibana_own_analysis='none'), 
                     'eval_uniform': DataOriginInfo('cfour', 'lvl1', 'basis1'), 
                     'system': MolecularSystem('name', 3)}
    sim = bare_wsim_for_EVVpGVPT2(**sim_conf_dict, 
                                    silent=True)
    del sim.terms[1][(0,1)]
    del sim.terms[0]

    # props, vib_ana_setup
    states = [VibState(s={('0',): 1.}, e=500.),
              VibState(s={('1',): 1.}, e=700.),
              VibState(s={('2',): 1.}, e=1300.),
              VibState(s={('0','0'): 1.}, e=950.),
              VibState(s={('0','1'): 1.}, e=1150.),
              VibState(s={('0','2'): 1.}, e=1985.),
              VibState(s={('1','1'): 1.}, e=1380.),
              VibState(s={('1','2'): 1.}, e=1996.),
              VibState(s={('2','2'): 1.}, e=2650.)]
    sim.vib_ana_setup.setStates(states)
    sim.vib_ana_setup.nc_sqrt_eigval = {0: 500., 1: 700., 2: 1300.}

    sim.findPropsAndMaxStateLvl()

    ggff = np.zeros((3,3,3,3))
    gf = np.zeros((3,3))
    ggf = np.zeros((3,3,3))
    gff = np.zeros((3,3,3))

    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    termsdicts = derived_terms_dict_to_dicts(sim.terms)
    assert termsdicts[1] == {'averaged_props': (('polgrad', ('b',), ('A', 'D')), ('dipgrad', ('a',), ('B',)), ('diphess', ('a', 'b'), ('G',))), 
                             'non_averaged_props': None, 'termA_pref': 0.25, 'termB_pref': 1.0, 
                             'vibene_denom': ('a', 'b'), 'vibenediff': None, 
                             'resonances': (('zero,a', (-1,)), ('a+b,a', (-1, 2))), 
                             'lvl_anharm': 1, 'anharm_tuple': (1, 0)}

    gf[0,0] = 1.
    gff[1,0,0] = 1.
    ggf[0,1,0] = 1.
    ggf[1,0,0] = ggf[0,1,0]
    
    sim.props[0].addValues(gf)
    sim.props[1].addValues(ggff)
    sim.props[2].addValues(gff)
    sim.props[3].addValues(ggf)

    start = {'x': 500., 'y': 1150.}
    end = {'x': 513., 'y': 1163.}
    spacer = {'x': 10., 'y': 10.}

    axis1 = SpectralAxis({'w1': 1})
    axis2 = SpectralAxis({'w2': 1})

    axes = {'x': axis1, 'y': axis2}
    specevalsetup: SpecEvalSetup = makeSpecSetup2D(start, end, spacer, axes, configs={})
    specevalsetup.ev_info.Gamma = 1.
    
    sim.addSpecEvalSetup(specevalsetup)
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    reference = -3./15 * 0.25 / convNu2Ene(500.) / convNu2Ene(700.) / (convNu2Ene(1.))**2

    from wilson_suite.wilson_intensities.spectrum.evaluators import terms_evaluator
    sim.evaluateAsResponseFunctionWithDiagnostics(evaluator=terms_evaluator)
    
    assert np.round(reference/sim.spec[0,0], 6) == 1.


def test_amplitude_mock_singlepoint_one_elterm_Gamma():

    from wilson_suite.wilson_utils.useful_shortcuts import bare_wsim_for_EVVpGVPT2, makeSpecSetup2D
    from wilson_suite.wilson_main.abstractions import VibAnaSetup, DataOriginInfo, MolecularSystem, SpectralAxis, SpecEvalSetup
    from wilson_suite.wilson_utils.abstractions import VibState
    sim_conf_dict = {'vib_ana_setup': VibAnaSetup(regime='GVPT2', vibana_own_analysis='none'), 
                     'eval_uniform': DataOriginInfo('cfour', 'lvl1', 'basis1'), 
                     'system': MolecularSystem('name', 3)}
    sim = bare_wsim_for_EVVpGVPT2(**sim_conf_dict, 
                                    silent=True)
    del sim.terms[1][(0,1)]
    del sim.terms[0]

    # props, vib_ana_setup
    states = [VibState(s={('0',): 1.}, e=500.),
              VibState(s={('1',): 1.}, e=700.),
              VibState(s={('2',): 1.}, e=1300.),
              VibState(s={('0','0'): 1.}, e=950.),
              VibState(s={('0','1'): 1.}, e=1150.),
              VibState(s={('0','2'): 1.}, e=1985.),
              VibState(s={('1','1'): 1.}, e=1380.),
              VibState(s={('1','2'): 1.}, e=1996.),
              VibState(s={('2','2'): 1.}, e=2650.)]
    sim.vib_ana_setup.setStates(states)
    sim.vib_ana_setup.nc_sqrt_eigval = {0: 500., 1: 700., 2: 1300.}

    sim.findPropsAndMaxStateLvl()

    ggff = np.zeros((3,3,3,3))
    gf = np.zeros((3,3))
    ggf = np.zeros((3,3,3))
    gff = np.zeros((3,3,3))

    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    termsdicts = derived_terms_dict_to_dicts(sim.terms)
    assert termsdicts[1] == {'averaged_props': (('polgrad', ('b',), ('A', 'D')), ('dipgrad', ('a',), ('B',)), ('diphess', ('a', 'b'), ('G',))), 
                             'non_averaged_props': None, 'termA_pref': 0.25, 'termB_pref': 1.0, 
                             'vibene_denom': ('a', 'b'), 'vibenediff': None, 
                             'resonances': (('zero,a', (-1,)), ('a+b,a', (-1, 2))), 
                             'lvl_anharm': 1, 'anharm_tuple': (1, 0)}

    gf[0,0] = 1.
    gff[1,0,0] = 1.
    ggf[0,1,0] = 1.
    ggf[1,0,0] = ggf[0,1,0]
    
    sim.props[0].addValues(gf)
    sim.props[1].addValues(ggff)
    sim.props[2].addValues(gff)
    sim.props[3].addValues(ggf)

    start = {'x': 500., 'y': 1150.}
    end = {'x': 513., 'y': 1163.}
    spacer = {'x': 10., 'y': 10.}

    axis1 = SpectralAxis({'w1': 1})
    axis2 = SpectralAxis({'w2': 1})

    Gamma = 1003.

    axes = {'x': axis1, 'y': axis2}
    specevalsetup: SpecEvalSetup = makeSpecSetup2D(start, end, spacer, axes, configs={})
    specevalsetup.ev_info.Gamma = Gamma
    
    sim.addSpecEvalSetup(specevalsetup)
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    reference = -3./15 * 0.25 / convNu2Ene(500.) / convNu2Ene(700.) / (convNu2Ene(Gamma))**2

    from wilson_suite.wilson_intensities.spectrum.evaluators import terms_evaluator
    sim.evaluateAsResponseFunctionWithDiagnostics(evaluator=terms_evaluator)

    assert np.round(reference/sim.spec[0,0], 6) == 1.

def test_amplitude_mock_singlepoint_one_mechterm():
    print()

    from wilson_suite.wilson_utils.useful_shortcuts import bare_wsim_for_EVVpGVPT2, makeSpecSetup2D
    from wilson_suite.wilson_main.abstractions import VibAnaSetup, DataOriginInfo, MolecularSystem, SpectralAxis, SpecEvalSetup
    from wilson_suite.wilson_utils.abstractions import VibState
    sim_conf_dict = {'vib_ana_setup': VibAnaSetup(regime='GVPT2', vibana_own_analysis='none'), 
                     'eval_uniform': DataOriginInfo('cfour', 'lvl1', 'basis1'), 
                     'system': MolecularSystem('name', 3)}
    sim = bare_wsim_for_EVVpGVPT2(**sim_conf_dict, 
                                    silent=True)
    del sim.terms[1][(1,0)]
    del sim.terms[0]
    sim.terms[1][(0,1)] = [sim.terms[1][(0,1)][6], sim.terms[1][(0,1)][11]]

    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    termsdicts = derived_terms_dict_to_dicts(sim.terms)

    assert termsdicts[0] == {'averaged_props': (('polgrad', ('b',), ('A', 'D')), ('dipgrad', ('a',), ('B',)), ('dipgrad', ('c',), ('G',))), 
                             'non_averaged_props': (('cff', ('a', 'b', 'c')),), 
                             'termA_pref': 0.125, 'termB_pref': 1.0, 'vibene_denom': ('a', 'b', 'c'), 
                             'vibenediff': ('a+b,c',), 'resonances': (('zero,a', (-1,)), ('a+b,a', (-1, 2))), 
                             'lvl_anharm': 1, 'anharm_tuple': (0, 1)}
    assert termsdicts[1] == {'averaged_props': (('polgrad', ('b',), ('A', 'D')), ('dipgrad', ('a',), ('B',)), ('dipgrad', ('c',), ('G',))), 
                             'non_averaged_props': (('cff', ('a', 'b', 'c')),), 
                             'termA_pref': -0.125, 'termB_pref': 1.0, 'vibene_denom': ('a', 'b', 'c'), 
                             'vibenediff': ('a+b+c,zero',), 'resonances': (('zero,a', (-1,)), ('a+b,a', (-1, 2))), 
                             'lvl_anharm': 1, 'anharm_tuple': (0, 1)}
    for t in termsdicts:
        print(t)

    # props, vib_ana_setup
    states = [VibState(s={('0',): 1.}, e=500.),
              VibState(s={('1',): 1.}, e=700.),
              VibState(s={('2',): 1.}, e=1300.),
              VibState(s={('0','0'): 1.}, e=950.),
              VibState(s={('0','1'): 1.}, e=1150.),
              VibState(s={('0','2'): 1.}, e=1985.),
              VibState(s={('1','1'): 1.}, e=1380.),
              VibState(s={('1','2'): 1.}, e=1996.),
              VibState(s={('2','2'): 1.}, e=2650.),
              VibState(s={('0','0','0'): 1.}, e=1450.),
              VibState(s={('0','0','1'): 1.}, e=1600.),
              VibState(s={('0','0','2'): 1.}, e=2200.),
              VibState(s={('0','1','1'): 1.}, e=1800.),
              VibState(s={('0','1','2'): 1.}, e=2400.),
              VibState(s={('0','2','2'): 1.}, e=3050.),
              VibState(s={('1','1','1'): 1.}, e=2000.),
              VibState(s={('1','1','2'): 1.}, e=2650.),
              VibState(s={('1','2','2'): 1.}, e=3250.),
              VibState(s={('2','2','2'): 1.}, e=3800.),

              ]
    sim.vib_ana_setup.setStates(states)
    sim.vib_ana_setup.nc_sqrt_eigval = {0: 500., 1: 700., 2: 1300.}

    sim.findPropsAndMaxStateLvl()

    gf = np.zeros((3,3))
    gff = np.zeros((3,3,3))
    ggg = np.zeros((3,3,3))

    gf[0,0] = gf[2,0] = 1.
    gff[1,0,0] = 1.


    ggg[0,1,0] = ggg[1,0,0] = ggg[0,0,1] = 1.
    ggg[0,1,2] = 1.3
    
    ggg[0,2,1] = ggg[1,0,2] = ggg[1,2,0] = ggg[2,0,1] = ggg[2,1,0] = 0.

    props_dict = {sim.props[i].trivial_name: i for i in range(len(sim.props))}

    sim.props[props_dict['dipgrad']].addValues(gf, in_basis='nm')
    sim.props[props_dict['polgrad']].addValues(gff, in_basis='nm')
    sim.props[props_dict['cff']].addValues(ggg, in_basis='nm', in_units='au')
    
    start = {'x': 500., 'y': 1150.}
    end = {'x': 513., 'y': 1163.}
    spacer = {'x': 10., 'y': 10.}

    axis1 = SpectralAxis({'w1': 1})
    axis2 = SpectralAxis({'w2': 1})

    Gamma = 1.

    axes = {'x': axis1, 'y': axis2}
    specevalsetup: SpecEvalSetup = makeSpecSetup2D(start, end, spacer, axes, configs={})
    specevalsetup.ev_info.Gamma = Gamma
    
    sim.addSpecEvalSetup(specevalsetup)
    ref1a = 3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(500.) / (convNu2Ene(Gamma))**2  \
                * (1./ (convNu2Ene(1600.)))
    ref1b = 3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(500.) / (convNu2Ene(Gamma))**2  \
                * (1./ (convNu2Ene(500.-1150.)))
    
    ref2a = 3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(1300.) / (convNu2Ene(Gamma))**2  \
                * (1./ (convNu2Ene(2400.))) * 1.3
    ref2b = 3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(1300.) / (convNu2Ene(Gamma))**2  \
                * (1./ (convNu2Ene(1300.-1150.))) * 1.3
    
    reference = ref1a + ref1b + ref2a + ref2b
    print('>>>>>> ref res', reference)
    print(f'>>>>>> ref1 ene fac {1 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(500.):.3e}')
    print(f'>>>>>> ref2 ene fac {1 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(1300.):.3e}')
    print(f'ref1 vibdiff1 {1./ (convNu2Ene(1600.))}')
    print(f'ref2 vibdiff1 {1./ (convNu2Ene(2400.))}')
    print(f'ref1 vibdiff2 { 1./ (convNu2Ene(500.-1150.))}')
    print(f'ref2 vibdiff2 {1./ (convNu2Ene(1300.-1150.))}')

    print(f'prodall ref1a {ref1a * (convNu2Ene(Gamma))**2:.3e}')
    print(f'prodall ref1b {ref1b * (convNu2Ene(Gamma))**2:.3e}')
    print(f'prodall ref2a {ref2a * (convNu2Ene(Gamma))**2:.3e}')
    print(f'prodall ref2b {ref2b * (convNu2Ene(Gamma))**2:.3e}')

    print('ref1', ref1a+ref1b)
    print('ref2', ref2a+ref2b)
    print('>>>>>> reference', reference)
    print(f'>>>>>> ref res {1./(convNu2Ene(Gamma))**2:.3e}')

    from wilson_suite.wilson_intensities.spectrum.evaluators import terms_evaluator
    sim.evaluateAsResponseFunctionWithDiagnostics(evaluator=terms_evaluator)

    assert np.round(reference/sim.spec[0,0], 6) == 1.


def test_amplitude_mock_singlepoint_one_mechterm_ba():
    print()

    from wilson_suite.wilson_utils.useful_shortcuts import bare_wsim_for_EVVpGVPT2, makeSpecSetup2D
    from wilson_suite.wilson_main.abstractions import VibAnaSetup, DataOriginInfo, MolecularSystem, SpectralAxis, SpecEvalSetup
    from wilson_suite.wilson_utils.abstractions import VibState
    sim_conf_dict = {'vib_ana_setup': VibAnaSetup(regime='GVPT2', vibana_own_analysis='none'), 
                     'eval_uniform': DataOriginInfo('cfour', 'lvl1', 'basis1'), 
                     'system': MolecularSystem('name', 3)}
    sim = bare_wsim_for_EVVpGVPT2(**sim_conf_dict, 
                                    silent=True)
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts

    del sim.terms[1][(1,0)]
    del sim.terms[0]
    sim.terms[1][(0,1)] = [sim.terms[1][(0,1)][1], sim.terms[1][(0,1)][4]]

    termsdicts = derived_terms_dict_to_dicts(sim.terms)

    for i, t in enumerate(termsdicts):
        print(i, t)

    # props, vib_ana_setup
    states = [VibState(s={('0',): 1.}, e=500.),
              VibState(s={('1',): 1.}, e=700.),
              VibState(s={('2',): 1.}, e=1300.),
              VibState(s={('0','0'): 1.}, e=950.),
              VibState(s={('0','1'): 1.}, e=1150.),
              VibState(s={('0','2'): 1.}, e=1985.),
              VibState(s={('1','1'): 1.}, e=1380.),
              VibState(s={('1','2'): 1.}, e=1996.),
              VibState(s={('2','2'): 1.}, e=2650.),
              VibState(s={('0','0','0'): 1.}, e=1450.),
              VibState(s={('0','0','1'): 1.}, e=1600.),
              VibState(s={('0','0','2'): 1.}, e=2200.),
              VibState(s={('0','1','1'): 1.}, e=1800.),
              VibState(s={('0','1','2'): 1.}, e=2400.),
              VibState(s={('0','2','2'): 1.}, e=3050.),
              VibState(s={('1','1','1'): 1.}, e=2000.),
              VibState(s={('1','1','2'): 1.}, e=2650.),
              VibState(s={('1','2','2'): 1.}, e=3250.),
              VibState(s={('2','2','2'): 1.}, e=3800.),

              ]
    sim.vib_ana_setup.setStates(states)
    sim.vib_ana_setup.nc_sqrt_eigval = {0: 500., 1: 700., 2: 1300.}

    sim.findPropsAndMaxStateLvl()

    gf = np.zeros((3,3))
    gff = np.zeros((3,3,3))
    ggg = np.zeros((3,3,3))

    gf[0,0] = gf[1,0] = 1.
    gff[2,0,0] = gff[0,0,0] = 1.


    ggg[0,1,0] = ggg[1,0,0] = ggg[0,0,1] = 1.
    ggg[0,1,2] = 1.3
    
    ggg[0,2,1] = ggg[1,0,2] = ggg[1,2,0] = ggg[2,0,1] = ggg[2,1,0] = 0.

    props_dict = {sim.props[i].trivial_name: i for i in range(len(sim.props))}

    print(f'props_dict \n{props_dict}')

    sim.props[props_dict['dipgrad']].addValues(gf, in_basis='nm')
    sim.props[props_dict['polgrad']].addValues(gff, in_basis='nm')
    sim.props[props_dict['cff']].addValues(ggg, in_basis='nm', in_units='au')
    
    start = {'x': 500., 'y': 700.}
    end = {'x': 513., 'y': 713.}
    spacer = {'x': 10., 'y': 10.}

    axis1 = SpectralAxis({'w1': 1})
    axis2 = SpectralAxis({'w2': 1})

    Gamma = 1.

    axes = {'x': axis1, 'y': axis2}
    specevalsetup: SpecEvalSetup = makeSpecSetup2D(start, end, spacer, axes, configs={})
    specevalsetup.ev_info.Gamma = Gamma
    
    sim.addSpecEvalSetup(specevalsetup)
    # 0, 1, 0
    ref1a = -3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(500.) / (-1j*convNu2Ene(Gamma))**2  \
                * (1./ (convNu2Ene(1150.-500.)))
    ref1b = -3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(500.) / (-1j*convNu2Ene(Gamma))**2  \
                * (1./ (convNu2Ene(950.-700.)))
    # 0, 1, 2
    ref2a = -3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(1300.) / (-1j*convNu2Ene(Gamma))**2  \
                * (1./ (convNu2Ene(1996.-500.))) * 1.3
    ref2b = -3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(1300.) / (-1j*convNu2Ene(Gamma))**2  \
                * (1./ (convNu2Ene(1985.-700.))) * 1.3
    
    print('vibdif ref1a', 1./ (convNu2Ene(1150.-500.)))
    print('vibdif ref1b', 1./ (convNu2Ene(950.-700.)))

    print('vibdif ref2a', 1./ (convNu2Ene(1996.-500.)))
    print('vibdif ref2b', 1./ (convNu2Ene(1985.-700.)))

    reference = ref1a + ref1b + ref2a + ref2b
    print('>>>>>> ref res', reference)
    print(f'>>>>>> ref1 ene fac {1 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(500.):.3e}')
    print(f'>>>>>> ref2 ene fac {1 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(1300.):.3e}')

    print(f'prodall ref1a {ref1a * (convNu2Ene(Gamma))**2:.3e}')
    print(f'prodall ref1b {ref1b * (convNu2Ene(Gamma))**2:.3e}')
    print(f'prodall ref2a {ref2a * (convNu2Ene(Gamma))**2:.3e}')
    print(f'prodall ref2b {ref2b * (convNu2Ene(Gamma))**2:.3e}')

    print('ref1', ref1a+ref1b)
    print('ref2', ref2a+ref2b)
    print('>>>>>> reference', reference)
    print(f'>>>>>> ref res {1./(convNu2Ene(Gamma))**2:.3e}')
    

    from wilson_suite.wilson_intensities.spectrum.evaluators import terms_evaluator
    sim.evaluateAsResponseFunctionWithDiagnostics(evaluator=terms_evaluator)
    print(f'sim.spec[0,0] {sim.spec[0,0]:.5e}')
    print(sim.diagn)
    print('ratio', reference/sim.spec[0,0])

    assert np.round(reference/sim.spec[0,0], 12) == 1.



def test_amplitude_mock_singlepoint_one_mechterm_Gamma():
    print()

    from wilson_suite.wilson_utils.useful_shortcuts import bare_wsim_for_EVVpGVPT2, makeSpecSetup2D
    from wilson_suite.wilson_main.abstractions import VibAnaSetup, DataOriginInfo, MolecularSystem, SpectralAxis, SpecEvalSetup
    from wilson_suite.wilson_utils.abstractions import VibState
    sim_conf_dict = {'vib_ana_setup': VibAnaSetup(regime='GVPT2', vibana_own_analysis='none'), 
                     'eval_uniform': DataOriginInfo('cfour', 'lvl1', 'basis1'), 
                     'system': MolecularSystem('name', 3)}
    sim = bare_wsim_for_EVVpGVPT2(**sim_conf_dict, 
                                    silent=True)
    del sim.terms[1][(1,0)]
    del sim.terms[0]
    sim.terms[1][(0,1)] = [sim.terms[1][(0,1)][6], sim.terms[1][(0,1)][11]]

    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    termsdicts = derived_terms_dict_to_dicts(sim.terms)

    assert termsdicts[0] == {'averaged_props': (('polgrad', ('b',), ('A', 'D')), ('dipgrad', ('a',), ('B',)), ('dipgrad', ('c',), ('G',))), 
                             'non_averaged_props': (('cff', ('a', 'b', 'c')),), 
                             'termA_pref': 0.125, 'termB_pref': 1.0, 'vibene_denom': ('a', 'b', 'c'), 
                             'vibenediff': ('a+b,c',), 'resonances': (('zero,a', (-1,)), ('a+b,a', (-1, 2))), 
                             'lvl_anharm': 1, 'anharm_tuple': (0, 1)}
    assert termsdicts[1] == {'averaged_props': (('polgrad', ('b',), ('A', 'D')), ('dipgrad', ('a',), ('B',)), ('dipgrad', ('c',), ('G',))), 
                             'non_averaged_props': (('cff', ('a', 'b', 'c')),), 
                             'termA_pref': -0.125, 'termB_pref': 1.0, 'vibene_denom': ('a', 'b', 'c'), 
                             'vibenediff': ('a+b+c,zero',), 'resonances': (('zero,a', (-1,)), ('a+b,a', (-1, 2))), 
                             'lvl_anharm': 1, 'anharm_tuple': (0, 1)}
    for t in termsdicts:
        print(t)

    # props, vib_ana_setup
    states = [VibState(s={('0',): 1.}, e=500.),
              VibState(s={('1',): 1.}, e=700.),
              VibState(s={('2',): 1.}, e=1300.),
              VibState(s={('0','0'): 1.}, e=950.),
              VibState(s={('0','1'): 1.}, e=1150.),
              VibState(s={('0','2'): 1.}, e=1985.),
              VibState(s={('1','1'): 1.}, e=1380.),
              VibState(s={('1','2'): 1.}, e=1996.),
              VibState(s={('2','2'): 1.}, e=2650.),
              VibState(s={('0','0','0'): 1.}, e=1450.),
              VibState(s={('0','0','1'): 1.}, e=1600.),
              VibState(s={('0','0','2'): 1.}, e=2200.),
              VibState(s={('0','1','1'): 1.}, e=1800.),
              VibState(s={('0','1','2'): 1.}, e=2400.),
              VibState(s={('0','2','2'): 1.}, e=3050.),
              VibState(s={('1','1','1'): 1.}, e=2000.),
              VibState(s={('1','1','2'): 1.}, e=2650.),
              VibState(s={('1','2','2'): 1.}, e=3250.),
              VibState(s={('2','2','2'): 1.}, e=3800.),

              ]
    sim.vib_ana_setup.setStates(states)
    sim.vib_ana_setup.nc_sqrt_eigval = {0: 500., 1: 700., 2: 1300.}

    sim.findPropsAndMaxStateLvl()

    gf = np.zeros((3,3))
    gff = np.zeros((3,3,3))
    ggg = np.zeros((3,3,3))

    gf[0,0] = gf[2,0] = 1.
    gff[1,0,0] = 1.


    ggg[0,1,0] = ggg[1,0,0] = ggg[0,0,1] = 1.
    ggg[0,1,2] = 1.3
    
    ggg[0,2,1] = ggg[1,0,2] = ggg[1,2,0] = ggg[2,0,1] = ggg[2,1,0] = 0.

    props_dict = {sim.props[i].trivial_name: i for i in range(len(sim.props))}

    sim.props[props_dict['dipgrad']].addValues(gf, in_basis='nm')
    sim.props[props_dict['polgrad']].addValues(gff, in_basis='nm')
    sim.props[props_dict['cff']].addValues(ggg, in_basis='nm', in_units='au')
    
    start = {'x': 500., 'y': 1150.}
    end = {'x': 513., 'y': 1163.}
    spacer = {'x': 10., 'y': 10.}

    axis1 = SpectralAxis({'w1': 1})
    axis2 = SpectralAxis({'w2': 1})

    Gamma = 1003.

    axes = {'x': axis1, 'y': axis2}
    specevalsetup: SpecEvalSetup = makeSpecSetup2D(start, end, spacer, axes, configs={})
    specevalsetup.ev_info.Gamma = Gamma
    
    sim.addSpecEvalSetup(specevalsetup)
    ref1a = 3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(500.) / (convNu2Ene(Gamma))**2  \
                * (1./ (convNu2Ene(1600.)))
    ref1b = 3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(500.) / (convNu2Ene(Gamma))**2  \
                * (1./ (convNu2Ene(500.-1150.)))
    
    ref2a = 3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(1300.) / (convNu2Ene(Gamma))**2  \
                * (1./ (convNu2Ene(2400.))) * 1.3
    ref2b = 3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(1300.) / (convNu2Ene(Gamma))**2  \
                * (1./ (convNu2Ene(1300.-1150.))) * 1.3
    
    reference = ref1a + ref1b + ref2a + ref2b
    print('>>>>>> ref res', reference)
    print(f'>>>>>> ref1 ene fac {1 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(500.):.3e}')
    print(f'>>>>>> ref2 ene fac {1 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(1300.):.3e}')
    print(f'ref1 vibdiff1 {1./ (convNu2Ene(1600.))}')
    print(f'ref2 vibdiff1 {1./ (convNu2Ene(2400.))}')
    print(f'ref1 vibdiff2 { 1./ (convNu2Ene(500.-1150.))}')
    print(f'ref2 vibdiff2 {1./ (convNu2Ene(1300.-1150.))}')

    print(f'prodall ref1a {ref1a * (convNu2Ene(Gamma))**2:.3e}')
    print(f'prodall ref1b {ref1b * (convNu2Ene(Gamma))**2:.3e}')
    print(f'prodall ref2a {ref2a * (convNu2Ene(Gamma))**2:.3e}')
    print(f'prodall ref2b {ref2b * (convNu2Ene(Gamma))**2:.3e}')

    print('ref1', ref1a+ref1b)
    print('ref2', ref2a+ref2b)
    print('>>>>>> reference', reference)
    print(f'>>>>>> ref res {1./(convNu2Ene(Gamma))**2:.3e}')

    from wilson_suite.wilson_intensities.spectrum.evaluators import terms_evaluator
    sim.evaluateAsResponseFunctionWithDiagnostics(evaluator=terms_evaluator)

    assert np.round(reference/sim.spec[0,0], 6) == 1.


def test_amplitude_mock_offresonance_one_elterm():

    from wilson_suite.wilson_utils.useful_shortcuts import bare_wsim_for_EVVpGVPT2, makeSpecSetup2D
    from wilson_suite.wilson_main.abstractions import VibAnaSetup, DataOriginInfo, MolecularSystem, SpectralAxis, SpecEvalSetup
    from wilson_suite.wilson_utils.abstractions import VibState
    sim_conf_dict = {'vib_ana_setup': VibAnaSetup(regime='GVPT2', vibana_own_analysis='none'), 
                     'eval_uniform': DataOriginInfo('cfour', 'lvl1', 'basis1'), 
                     'system': MolecularSystem('name', 3)}
    sim = bare_wsim_for_EVVpGVPT2(**sim_conf_dict, 
                                    silent=True)
    del sim.terms[1][(0,1)]
    del sim.terms[0]

    # props, vib_ana_setup
    states = [VibState(s={('0',): 1.}, e=500.),
              VibState(s={('1',): 1.}, e=700.),
              VibState(s={('2',): 1.}, e=1300.),
              VibState(s={('0','0'): 1.}, e=950.),
              VibState(s={('0','1'): 1.}, e=1150.),
              VibState(s={('0','2'): 1.}, e=1985.),
              VibState(s={('1','1'): 1.}, e=1380.),
              VibState(s={('1','2'): 1.}, e=1996.),
              VibState(s={('2','2'): 1.}, e=2650.)]
    sim.vib_ana_setup.setStates(states)
    sim.vib_ana_setup.nc_sqrt_eigval = {0: 500., 1: 700., 2: 1300.}

    sim.findPropsAndMaxStateLvl()

    ggff = np.zeros((3,3,3,3))
    gf = np.zeros((3,3))
    ggf = np.zeros((3,3,3))
    gff = np.zeros((3,3,3))

    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    termsdicts = derived_terms_dict_to_dicts(sim.terms)
    assert termsdicts[1] == {'averaged_props': (('polgrad', ('b',), ('A', 'D')), ('dipgrad', ('a',), ('B',)), ('diphess', ('a', 'b'), ('G',))), 
                             'non_averaged_props': None, 'termA_pref': 0.25, 'termB_pref': 1.0, 
                             'vibene_denom': ('a', 'b'), 'vibenediff': None, 
                             'resonances': (('zero,a', (-1,)), ('a+b,a', (-1, 2))), 
                             'lvl_anharm': 1, 'anharm_tuple': (1, 0)}

    gf[0,0] = 1.
    gff[1,0,0] = 1.
    ggf[0,1,0] = 1.
    ggf[1,0,0] = ggf[0,1,0]
    
    sim.props[0].addValues(gf)
    sim.props[1].addValues(ggff)
    sim.props[2].addValues(gff)
    sim.props[3].addValues(ggf)

    start = {'x': 505., 'y': 1158.}
    end = {'x': 513., 'y': 1163.}
    spacer = {'x': 10., 'y': 10.}

    axis1 = SpectralAxis({'w1': 1})
    axis2 = SpectralAxis({'w2': 1})

    Gamma = 1.

    axes = {'x': axis1, 'y': axis2}
    specevalsetup: SpecEvalSetup = makeSpecSetup2D(start, end, spacer, axes, configs={})
    specevalsetup.ev_info.Gamma = Gamma
    
    sim.addSpecEvalSetup(specevalsetup)
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    w1, w1mw2 = 505., -653.

    reference = 3./15 * 0.25 / convNu2Ene(500.) / convNu2Ene(700.) / \
                (convNu2Ene(-500. + w1) - 1j*convNu2Ene(Gamma)) / (convNu2Ene(1150.-500 + w1mw2) - 1j*convNu2Ene(Gamma))

    from wilson_suite.wilson_intensities.spectrum.evaluators import terms_evaluator
    from functools import partial
    # eval_selected = partial(terms_evaluator, selected_combs=[(0,1)], collect_all=True)
    eval_selected = partial(terms_evaluator, collect_all=True)
    sim.evaluateAsResponseFunctionWithDiagnostics(evaluator=eval_selected)
    # sim.evaluateAsResponseFunctionWithDiagnostics(evaluator=terms_evaluator)
    print(sim.diagn)
    print(f'reference {reference:.5e}')
    print(f'sim.spec[0,0] {sim.spec[0,0]:.5e}')
    print('ratio', reference/sim.spec[0,0])

    assert np.round(reference/sim.spec[0,0], 6) == 1.


def test_amplitude_mock_offresonance_one_mechterm():
    print()

    from wilson_suite.wilson_utils.useful_shortcuts import bare_wsim_for_EVVpGVPT2, makeSpecSetup2D
    from wilson_suite.wilson_main.abstractions import VibAnaSetup, DataOriginInfo, MolecularSystem, SpectralAxis, SpecEvalSetup
    from wilson_suite.wilson_utils.abstractions import VibState
    sim_conf_dict = {'vib_ana_setup': VibAnaSetup(regime='GVPT2', vibana_own_analysis='none'), 
                     'eval_uniform': DataOriginInfo('cfour', 'lvl1', 'basis1'), 
                     'system': MolecularSystem('name', 3)}
    sim = bare_wsim_for_EVVpGVPT2(**sim_conf_dict, 
                                    silent=True)
    del sim.terms[1][(1,0)]
    del sim.terms[0]
    sim.terms[1][(0,1)] = [sim.terms[1][(0,1)][6], sim.terms[1][(0,1)][11]]

    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    termsdicts = derived_terms_dict_to_dicts(sim.terms)

    assert termsdicts[0] == {'averaged_props': (('polgrad', ('b',), ('A', 'D')), ('dipgrad', ('a',), ('B',)), ('dipgrad', ('c',), ('G',))), 
                             'non_averaged_props': (('cff', ('a', 'b', 'c')),), 
                             'termA_pref': 0.125, 'termB_pref': 1.0, 'vibene_denom': ('a', 'b', 'c'), 
                             'vibenediff': ('a+b,c',), 'resonances': (('zero,a', (-1,)), ('a+b,a', (-1, 2))), 
                             'lvl_anharm': 1, 'anharm_tuple': (0, 1)}
    assert termsdicts[1] == {'averaged_props': (('polgrad', ('b',), ('A', 'D')), ('dipgrad', ('a',), ('B',)), ('dipgrad', ('c',), ('G',))), 
                             'non_averaged_props': (('cff', ('a', 'b', 'c')),), 
                             'termA_pref': -0.125, 'termB_pref': 1.0, 'vibene_denom': ('a', 'b', 'c'), 
                             'vibenediff': ('a+b+c,zero',), 'resonances': (('zero,a', (-1,)), ('a+b,a', (-1, 2))), 
                             'lvl_anharm': 1, 'anharm_tuple': (0, 1)}
    for t in termsdicts:
        print(t)

    # props, vib_ana_setup
    states = [VibState(s={('0',): 1.}, e=500.),
              VibState(s={('1',): 1.}, e=700.),
              VibState(s={('2',): 1.}, e=1300.),
              VibState(s={('0','0'): 1.}, e=950.),
              VibState(s={('0','1'): 1.}, e=1150.),
              VibState(s={('0','2'): 1.}, e=1985.),
              VibState(s={('1','1'): 1.}, e=1380.),
              VibState(s={('1','2'): 1.}, e=1996.),
              VibState(s={('2','2'): 1.}, e=2650.),
              VibState(s={('0','0','0'): 1.}, e=1450.),
              VibState(s={('0','0','1'): 1.}, e=1600.),
              VibState(s={('0','0','2'): 1.}, e=2200.),
              VibState(s={('0','1','1'): 1.}, e=1800.),
              VibState(s={('0','1','2'): 1.}, e=2400.),
              VibState(s={('0','2','2'): 1.}, e=3050.),
              VibState(s={('1','1','1'): 1.}, e=2000.),
              VibState(s={('1','1','2'): 1.}, e=2650.),
              VibState(s={('1','2','2'): 1.}, e=3250.),
              VibState(s={('2','2','2'): 1.}, e=3800.),

              ]
    sim.vib_ana_setup.setStates(states)
    sim.vib_ana_setup.nc_sqrt_eigval = {0: 500., 1: 700., 2: 1300.}

    sim.findPropsAndMaxStateLvl()

    gf = np.zeros((3,3))
    gff = np.zeros((3,3,3))
    ggg = np.zeros((3,3,3))

    gf[0,0] = gf[2,0] = 1.
    gff[1,0,0] = 1.


    ggg[0,1,0] = ggg[1,0,0] = ggg[0,0,1] = 1.
    ggg[0,1,2] = 1.3
    
    ggg[0,2,1] = ggg[1,0,2] = ggg[1,2,0] = ggg[2,0,1] = ggg[2,1,0] = 0.

    props_dict = {sim.props[i].trivial_name: i for i in range(len(sim.props))}

    sim.props[props_dict['dipgrad']].addValues(gf, in_basis='nm')
    sim.props[props_dict['polgrad']].addValues(gff, in_basis='nm')
    sim.props[props_dict['cff']].addValues(ggg, in_basis='nm', in_units='au')
    
    start = {'x': 505., 'y': 1158.}
    end = {'x': 513., 'y': 1163.}
    spacer = {'x': 10., 'y': 10.}

    axis1 = SpectralAxis({'w1': 1})
    axis2 = SpectralAxis({'w2': 1})

    Gamma = 1.

    axes = {'x': axis1, 'y': axis2}
    specevalsetup: SpecEvalSetup = makeSpecSetup2D(start, end, spacer, axes, configs={})
    specevalsetup.ev_info.Gamma = Gamma
    
    sim.addSpecEvalSetup(specevalsetup)

    w1, w1mw2 = 505., -653.

    ref1a = 3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(500.) \
                / (convNu2Ene(-500. + w1) - 1j*convNu2Ene(Gamma)) / (convNu2Ene(1150.-500 + w1mw2) - 1j*convNu2Ene(Gamma)) \
                * (1./ (convNu2Ene(1600.)))
    ref1b = 3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(500.) \
                / (convNu2Ene(-500. + w1) - 1j*convNu2Ene(Gamma)) / (convNu2Ene(1150.-500 + w1mw2) - 1j*convNu2Ene(Gamma)) \
                * (1./ (convNu2Ene(500.-1150.)))
    
    ref2a = 3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(1300.)  \
                / (convNu2Ene(-500. + w1) - 1j*convNu2Ene(Gamma)) / (convNu2Ene(1150.-500 + w1mw2) - 1j*convNu2Ene(Gamma)) \
                * (1./ (convNu2Ene(2400.))) * 1.3
    ref2b = 3./15 * 0.125 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(1300.) \
                / (convNu2Ene(-500. + w1) - 1j*convNu2Ene(Gamma)) / (convNu2Ene(1150.-500 + w1mw2) - 1j*convNu2Ene(Gamma)) \
                * (1./ (convNu2Ene(1300.-1150.))) * 1.3
    
    reference = -(ref1a + ref1b + ref2a + ref2b)

    #TODO: check sign in reference

    print(f'>>>>>> ref res {reference:.5e}')
    print(f'>>>>>> ref1 ene fac {1 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(500.):.3e}')
    print(f'>>>>>> ref2 ene fac {1 / convNu2Ene(500.) / convNu2Ene(700.) / convNu2Ene(1300.):.3e}')
    print(f'ref1 vibdiff1 {1./ (convNu2Ene(1600.))}')
    print(f'ref2 vibdiff1 {1./ (convNu2Ene(2400.))}')
    print(f'ref1 vibdiff2 { 1./ (convNu2Ene(500.-1150.))}')
    print(f'ref2 vibdiff2 {1./ (convNu2Ene(1300.-1150.))}')

    print(f'prodall ref1a {ref1a * (convNu2Ene(Gamma))**2:.5e}')
    print(f'prodall ref1b {ref1b * (convNu2Ene(Gamma))**2:.5e}')
    print(f'prodall ref2a {ref2a * (convNu2Ene(Gamma))**2:.5e}')
    print(f'prodall ref2b {ref2b * (convNu2Ene(Gamma))**2:.5e}')

    print('ref1', ref1a+ref1b)
    print('ref2', ref2a+ref2b)
    print('>>>>>> reference', reference)
    print(f'>>>>>> ref res {1./(convNu2Ene(Gamma))**2:.5e}')


    from wilson_suite.wilson_intensities.spectrum.evaluators import terms_evaluator
    from functools import partial
    eval_selected = partial(terms_evaluator, selected_combs=[(0,1)], collect_all=True)
    sim.evaluateAsResponseFunctionWithDiagnostics(evaluator=eval_selected)

    print(f'sim.spec[0,0] {sim.spec[0,0]:.5e}')
    print('ratio', reference/sim.spec[0,0])

    assert np.round(reference/sim.spec[0,0], 6) == 1.



"""
test_amplitude_mock_singlepoint_one_mechterm ++
different Gamma ++
off-resonance point ++

multiple features test - M
small real system - H2O - M

test for the used mode combinations/unused

test_amplitude_mock_singlepoint_one_mechterm_one_elterm
mock with diff number of normal modes
non-triavial orient. avrg. tensor
differnt ene lvls for harm and anharm
summation over c

different axes

"""

def test_pairwise_differences():
    print()
    qstates_Eh = {1: convNu2Ene(np.array([500., 700., 1300.])),
                  2: convNu2Ene(np.array([[950., 1150., 1985.],
                      [1150., 1380., 1996.],
                      [1985., 1996., 2650.]]))}
    a = pairwise_differences(qstates_Eh[1], qstates_Eh[2])
    print(a)
    assert a[2, 0, 1] == qstates_Eh[1][2] - qstates_Eh[2][(0,1)]
    assert a[0, 1, 0] == qstates_Eh[1][0] - qstates_Eh[2][(1,0)]

    print(1/a[(0,1,2)])
    print(1/a[(2,0,1)])
    q = np.array([[[-0.00205035, -0.00296162, -0.00676616],                         
                                        [-0.00296162,                              
                                -0.00400958, -0.00681628],                         
                                        [-0.00676616,                              
                                -0.00681628, -0.00979612]],                        
                                                                                    
                                        [[-0.00113908,                              
                                -0.00205035, -0.00585489],                         
                                        [-0.00205035,                              
                                -0.00309831, -0.00590501],                         
                                        [-0.00585489,                              
                                -0.00590501, -0.00888485]],                        
                                                                                    
                                        [[ 0.00159472,                              
                                0.00068345, -0.00312109],                          
                                        [ 0.00068345,                              
                                -0.00036451, -0.00317121],                         
                                        [-0.00312109,                              
                                -0.00317121, -0.00615105]]])
    assert np.allclose(q, a)