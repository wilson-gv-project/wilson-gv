"""
Goal:
- construct a calculation using mixed sources of data

[x] making calc batches
[ ] making inputs
[ ] putting results to Storage ?
[ ] getting data for batches
"""
import numpy as np
from wilson_utils.logger import setup_logger
import wilson_main.abstractions as wm_abst

import wilson_main.calculationManagement as manage
import wilson_main.externalDataProcessor as dataprc

import sys
import os
# to get wilson_fixtures import working
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from wilson_fixtures.fixtures import evv_terms

import logging
logger = logging.getLogger("wilson."+__name__)
setup_logger("wilson", level=logging.DEBUG)

def run():
    logger.info('wm_abst.namelogger')
    logger.info(wm_abst.namelogger)


    storage = dataprc.CalcDataStorage()

    mol1 = wm_abst.MolecularSystem(name='mol1', natoms=3)
    mol2 = wm_abst.MolecularSystem(name='mol2', natoms=5)

    setup1 = wm_abst.ExternalCalcSetup('p1', 'lvl1', 'b1')
    setup2 = wm_abst.ExternalCalcSetup('p1', 'lvl2', 'b2')
    setup3 = wm_abst.ExternalCalcSetup('p1', 'lvl2', 'b3')
    setup4 = wm_abst.ExternalCalcSetup('p2', 'lvl3', 'b2')

    np3b3 = np.array([[0.48, 0.53, 0.52],
                    [0.42, 0.81, 0.47],
                    [0.23, 0.66, 0.8 ]])
    np1b3 = np.array([[0.81, 0.51, 0.3 ]])
    np3b3b3 = np.array([[[0.21, 0.44, 0.16],
                        [0.96, 0.98, 0.43],
                        [0.69, 0.5 , 0.05]],

                        [[0.21, 0.68, 0.11],
                        [0.55, 0.22, 0.61],
                        [0.34, 0.11, 0.32]],
                        
                        [[0.2 , 0.63, 0.52],
                        [0.95, 0.49, 0.22],
                        [0.17, 0.84, 0.27]]])
    np3b3b3b3 = np.array([[[[0.93, 0.84, 0.13],
                            [0.96, 0.52, 0.5 ],
                            [0.38, 0.71, 0.16]],

                            [[0.38, 0.35, 0.03],
                            [0.49, 0.54, 0.47],
                            [0.56, 0.95, 0.25]],

                            [[0.47, 0.81, 0.13],
                            [0.12, 0.6 , 0.97],
                            [0.55, 0.15, 0.05]]],

                        [[[0.89, 0.13, 0.08],
                            [0.48, 0.45, 0.14],
                            [0.33, 0.15, 0.78]],

                            [[0.38, 0.6 , 0.82],
                            [0.36, 0.64, 0.58],
                            [0.83, 0.52, 0.05]],

                            [[0.14, 0.89, 0.69],
                            [0.88, 0.95, 0.64],
                            [0.21, 0.14, 0.4 ]]],

                        [[[0.03, 0.59, 0.85],
                            [0.82, 0.2 , 0.09],
                            [0.14, 0.37, 0.36]],

                            [[0.23, 0.25, 0.  ],
                            [0.43, 0.9 , 0.47],
                            [0.47, 0.37, 0.35]],

                            [[0.3 , 0.8 , 0.54],
                            [0.18, 0.08, 0.38],
                            [0.57, 0.22, 0.06]]]])

    np5b3 = np.array([[0.29, 0.23, 0.52],
                    [0.42, 0.15, 0.23],
                    [0.33, 0.26, 0.04],
                    [0.21, 0.79, 0.21],
                    [0.66, 0.93, 0.71]])

    datadict1 = {'system': mol1, 'calc_setup': setup1, 
                'B': (np1b3, None, 'cm-1'), 'coriolis': (np3b3, 'bu', 'cm-1'),
                'hess': (np3b3, 'bu', 'cm-1'), 'cff': (np3b3b3, 'bu', 'cm-1'), 'qff': (np3b3b3b3, 'bu', 'cm-1'), 
                'dipgrad': (np5b3, 'bu', 'cm-1'), 'diphess': (np5b3, 'bu', 'cm-1'), 
                'polgrad': (np5b3, 'bu', 'cm-1'), 'polhess': (np5b3, 'bu', 'cm-1'),
                'harmonic_states': {(3,):4, (5,):2, (6,):46}, 'anharmonic_states': {(3,):14, (5,):32, (6,):96}}

    datadict2 = {'system': mol2, 'calc_setup': setup2, 
                'B': None, 'coriolis': None,
                'hess': None, 'cff': None, 'qff': None, 
                'dipgrad': None, 'diphess': None, 'polgrad': None, 'polhess': None,
                'harmonic_states': {}, 'anharmonic_states': {}}

    datadict3 = {'system': mol1, 'calc_setup': setup2, 
                'B': None, 'coriolis': None,
                'hess': None, 'cff': None, 'qff': None, 
                'dipgrad': (np5b3, 'bu', 'cm-1'), 'diphess': (np5b3, 'bu', 'cm-1'), 
                'polgrad': None, 'polhess': None,
                'harmonic_states': {}, 'anharmonic_states': {}}

    datadict4 = {'system': mol1, 'calc_setup': setup3, 
                'B': (np1b3, None, 'cm-1'), 'coriolis': (np3b3, 'bu', 'cm-1'),
                'hess': (np3b3, 'bu', 'cm-1'), 'cff': (np3b3b3, 'bu', 'cm-1'), 'qff': (np3b3b3b3, 'bu', 'cm-1'), 
                'dipgrad': (np5b3, 'bu', 'cm-1'), 'diphess': (np5b3, 'bu', 'cm-1'), 
                'polgrad': (np5b3, 'bu', 'cm-1'), 'polhess': (np5b3, 'bu', 'cm-1'),
                'harmonic_states': {(3,):4, (5,):2, (6,):46}, 'anharmonic_states': {(3,):14, (5,):32, (6,):96}}

    datadict5 = {'system': mol2, 'calc_setup': setup4, 
                'B': None, 'coriolis': None,
                'hess': None, 'cff': None, 'qff': None, 
                'dipgrad': None, 'diphess': None, 'polgrad': None, 'polhess': None,
                'harmonic_states': {}, 'anharmonic_states': {}}


    # NOTE dataprc.CalculatedDataFromOutput(datadict1) - then need keys in dict to be in order
    # NOTE dataprc.CalculatedDataFromOutput(**datadict1) - doesn't need ordered keys

    cd1 = dataprc.CalculatedDataFromOutput(**datadict1)
    cd2 = dataprc.CalculatedDataFromOutput(**datadict2)
    cd3 = dataprc.CalculatedDataFromOutput(**datadict3)
    cd4 = dataprc.CalculatedDataFromOutput(**datadict4)
    cd5 = dataprc.CalculatedDataFromOutput(**datadict5)

    storage.addResult(cd1)
    storage.addResult(cd2)
    storage.addResult(cd3)
    storage.addResult(cd4)
    storage.addResult(cd5)

    # QC calculations/vibana parameters
    vibanasetup = wm_abst.VibAnaSetup(system=mol1, regime='GVPT2', 
                                    vibana_prop_need='anharm', # very confusing name for understanding what is going on....
                                    allow_skip_eigvec=True, external_fill_from=setup1)

    eval_prop_specify = {'cff': setup1, 'qff': setup1, 
                        'dipgrad': setup2, 'diphess': setup2, 
                        'polgrad': setup3, 'polhess': setup3,
                        'B': setup1, 'coriolis': setup2}

    terms = evv_terms()

    needed_props, vibanasetup = manage.findPropsAndMaxStateLvlNeeded(terms, vibanasetup, freqs='static')

    wm_abst.dressPropsWithSetup(props=needed_props, eval_by_prop_name=eval_prop_specify)

    logger.debug('      needed_props after drssing')
    logger.debug(needed_props)

    g = manage.groupDataForCalcSetups(vibanasetup=vibanasetup, calc_props_setup=eval_prop_specify, props_needed=needed_props)

    logger.debug('      grouped calc setups:')
    logger.debug(g)

    calcbatches = manage.makeBatchesFromGroups(mol1, g)

    logger.debug('      calcbatches:')
    logger.debug(calcbatches)

    # Get results from calculation batches
    # register vib_ana_setup_to_fill.states
    manage.getVibAnaValsFromStorage(system=mol1, vib_ana_setup_to_fill=vibanasetup, calcdatasets=storage)

    logger.debug('      vibanasetup')
    logger.debug(vibanasetup)

    # register props_to_fill values
    manage.getPropValsFromStorage(system=mol1, props_to_fill=needed_props, eval_by_prop_name=eval_prop_specify, calcdatasets=storage)

    logger.debug('      needed_props')
    logger.debug(needed_props)

    logger.debug(needed_props[0].vals)
    logger.debug(needed_props[0].serial_vals)

    logger.debug('      calcbatches after getting vals:')
    logger.debug(calcbatches)

    # maybe get info/hints of what one can do from this point? or any other point - the workflow is fairly complex
    # TODO hints for notebook use