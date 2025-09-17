"""
Anharmonic analyzer for wilson_main.VibAnaSetup.doAnharmonicAnalysis()

"""
from ...wilson_utils import abstractions as wu_abst
from ...wilson_main import abstractions as wm_abst
from ..spectrum.vpt2 import anharm_corr_energies

import logging
logger = logging.getLogger("wilson."+__name__)

def anharm_analyzer_data(system:wm_abst.MolecularSystem = None, props: list[wm_abst.MolecularProperty] = None, 
                    nc_sqrt_eigval: dict = None, 
                    regime: str = None, regime_subinfo: dict = None, 
                    exclude_modes: list = None) -> tuple[list[wu_abst.VibState], dict]:
    """
    Basically a wrapper for analyser; passes data to anharm_corr_energies where analysis happens...
        then puts into list[VibState] form.
    This way it is more clean

    returns self.states for VibAnaSetup, and states are list[VibState]
    
    NOTE: Pure function -  will assume that inputs are always valid
    """
    logger.info('Starting anharm_analyzer()')
    
    # modes exclusion list update
    if exclude_modes is None:
        exclude_modes = []
    
    prop_dict = {i.trivial_name: i.vals for i in props}
    logger.debug(f'prop_dict {prop_dict.keys()}')
    
    # corrected_levels : funds, over2q, combo2q, over3q, combo3q
    corrected_levels, fermi_resonances = anharm_corr_energies(nc_sqrt_eigval,
                                                             prop_dict['cff'], prop_dict['qff'], prop_dict['B'], prop_dict['coriolis'],
                                                             regime, exclude_modes)
    # assembling data for return
    all_states_corr = {}

    for i in range(len(nc_sqrt_eigval)):
        all_states_corr[(str(i),)] = corrected_levels[0][i]

        for j in range(i + 1):
            if i == j:
                all_states_corr[tuple([str(i), str(i)])] = corrected_levels[1][i]
            else:
                all_states_corr[tuple([str(el) for el in sorted([i, j])])] = corrected_levels[2][i, j]

            for k in range(len(nc_sqrt_eigval)):
                if i == j == k:
                    all_states_corr[tuple([str(i), str(i), str(i)])] = corrected_levels[3][i]
                else:
                    key = tuple([str(el) for el in sorted([i, j, k])])
                    if key not in all_states_corr:
                        if corrected_levels[4][i, j, k] != 0.:
                            all_states_corr[key] = corrected_levels[4][i, j, k]

    logger.debug(f'all_states_corr: {all_states_corr}')

    vibstates = []
    for st in all_states_corr:
        # if len(st) <= max_state_lvl: # ?
        # TODO: Exclusion based on mode index or freq cutoff
        # FIXME: Change to integer indexing - in s dict?
        vibstates.append(wu_abst.VibState(s={st: 1.0}, e=all_states_corr[st]))

    logger.debug('GVPT2 anharm corrected:')
    logger.debug(f'vibstates: {vibstates}')

    diagnostics = {'fermi_resonances': fermi_resonances}
    
    return vibstates, diagnostics

