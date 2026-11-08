"""
Anharmonic analyzer

"""
import logging

from ...wilson_system.system_data import MolPropsCollection, VibState
from ..anharmonic_treatment.vpt2 import anharm_corr_energies

logger = logging.getLogger("wilson_suite."+__name__)

def anharm_analyzer_data(props: MolPropsCollection, 
                         nc_sqrt_eigval: dict, 
                         regime: str,
                         exclude_modes: list) -> tuple[list[VibState], dict]:
    """
    Basically a wrapper for analyser; passes data to anharm_corr_energies where analysis happens...
        then puts into list[VibState] form.
    This way it is more clean

    returns self.states for VibAnaSetup, and states are list[VibState]

    exclude_modes: modes left out of VPT2, as if the molecule did not have them. The other states
        change, and no state of an excluded mode is returned. An unknown mode raises ValueError.
    diagnostics: {'fermi_resonances': [...], 'exclude_modes': the excluded modes, sorted, e.g. (1,)}

    NOTE: Pure function -  will assume that inputs are always valid
    """
    logger.info('Starting anharm_analyzer()')
    if regime not in ['GVPT2', 'VPT2', 'DVPT2']:
        raise NotImplementedError(f"Vibrational anharmonic analysis regime not supported: {regime}. Choose from: GVPT2, VPT2, DVPT2")
    
    # modes exclusion list update
    if exclude_modes is None:
        exclude_modes = []
    unknown = sorted(set(exclude_modes) - set(nc_sqrt_eigval))
    if unknown:
        raise ValueError(f'exclude_modes {unknown}: no such mode, the modes are {sorted(nc_sqrt_eigval)}')

    # force constants in cm-1 (cff_rc, qff_rc), as anharm_corr_energies takes them; cff and qff are in atomic units
    prop_dict = {name: props[name].vals for name in ('cff_rc', 'qff_rc', 'B', 'coriolis')}

    logger.debug(f'prop_dict {prop_dict.keys()}')
    logger.debug(prop_dict)

    # FIXME: Convertors from au to rec cm of cff, qff, (MR: B, coriolis)
    # from wilson_suite.wilson_utils.unit_convertor import 

    # corrected_levels : funds, over2q, combo2q, over3q, combo3q
    corrected_levels, fermi_resonances = anharm_corr_energies(harmonic_energies=nc_sqrt_eigval,
                                                             cubic_forcefield=prop_dict['cff_rc'],
                                                             quartic_forcefield=prop_dict['qff_rc'],
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
                all_states_corr[(str(i), str(i))] = corrected_levels[1][i]
            else:
                all_states_corr[(str(j), str(i))] = corrected_levels[2][i, j]     # j < i: numeric order

            for k in range(len(nc_sqrt_eigval)):
                if i == j == k:
                    all_states_corr[(str(i), str(i), str(i))] = corrected_levels[3][i]
                else:
                    key = tuple([str(el) for el in sorted([i, j, k])])
                    if key not in all_states_corr and corrected_levels[4][i, j, k] != 0.:
                        all_states_corr[key] = corrected_levels[4][i, j, k]

    logger.debug(f'all_states_corr: {all_states_corr}')

    # VPT2 keeps a slot for each excluded mode and leaves it at 0 cm-1, the ground-state energy;
    # a state that contains an excluded mode is left out, so no such 0 is read as a real energy.
    # whole mode labels are compared: excluding '1' keeps ('0', '10')
    excluded = {str(m) for m in exclude_modes}
    vibstates = []
    for st, energy in all_states_corr.items():
        if excluded.intersection(st):
            continue
        # if len(st) <= max_state_lvl: # ?
        # TODO: Exclusion based on freq cutoff
        # FIXME: Change to integer indexing - in s dict?
        vibstates.append(VibState(harm_quanta_coeffs={st: 1.0}, energy=energy))

    logger.debug('GVPT2 anharm corrected:')
    logger.debug(f'vibstates: {vibstates}')

    # the energies of the kept states depend on which modes VPT2 left out, and the states alone
    # do not show those modes, so the diagnostics name them. Sorted tuple: same modes in any order, same record
    diagnostics = {'fermi_resonances': fermi_resonances, 'exclude_modes': tuple(sorted(exclude_modes))}
    
    return vibstates, diagnostics

