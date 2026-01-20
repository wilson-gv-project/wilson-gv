"""
Anharmonic analyzer for wilson_main.VibAnaSetup.doAnharmonicAnalysis()

"""
from ...wilson_main import abstractions as wm_abst
from ..anharmonic_treatment.vpt2 import anharm_corr_energies

import logging
logger = logging.getLogger("wilson_suite."+__name__)

def anharm_analyzer_data(props: list[wm_abst.MolecularProperty] = None, 
                         nc_sqrt_eigval: dict = None, 
                         regime: str = None,
                         exclude_modes: list = None) -> tuple[list[wm_abst.VibState], dict]:
    """
    Basically a wrapper for analyser; passes data to anharm_corr_energies where analysis happens...
        then puts into list[VibState] form.
    This way it is more clean

    returns self.states for VibAnaSetup, and states are list[VibState]
    
    NOTE: Pure function -  will assume that inputs are always valid
    """
    logger.info('Starting anharm_analyzer()')
    if regime not in ['GVPT2', 'VPT2', 'DVPT2']:
        raise NotImplementedError(f"Vibrational anharmonic analysis regime not supported: {regime}. Choose from: GVPT2, VPT2, DVPT2")
    
    # modes exclusion list update
    if exclude_modes is None:
        exclude_modes = []
    
    # prop_dict = {i.trivial_name: i.serial_vals for i in props}
    prop_dict = {i.trivial_name: i.vals for i in props}
    for i in props: 
        if i.trivial_name in ['cff', 'qff']:
            prop_dict[i.trivial_name] = i.extra_data
    logger.debug(f'prop_dict {prop_dict.keys()}')
    logger.debug(prop_dict)

    # FIXME: Convertors from au to rec cm of cff, qff, (MR: B, coriolis)
    # from wilson_suite.wilson_utils.unit_convertor import 

    # corrected_levels : funds, over2q, combo2q, over3q, combo3q
    corrected_levels, fermi_resonances = anharm_corr_energies(harmonic_energies=nc_sqrt_eigval,
                                                             cubic_forcefield=prop_dict['cff'], 
                                                             quartic_forcefield=prop_dict['qff'], 
                                                             rotational_constant=prop_dict['B'], 
                                                             coriolis_constant=prop_dict['coriolis'],
                                                             anharmonic_type=regime, 
                                                             list2exclude=exclude_modes)
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
        vibstates.append(wm_abst.VibState(harm_quanta_coeffs={st: 1.0}, energy=all_states_corr[st]))

    logger.debug('GVPT2 anharm corrected:')
    logger.debug(f'vibstates: {vibstates}')

    diagnostics = {'fermi_resonances': fermi_resonances}
    
    return vibstates, diagnostics

