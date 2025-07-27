"""
(self.system, props, self.regime, self.regime_subinfo, self.nc_sqrt_eigval, self.nc_eigvec)
"""
import copy
import numpy as np
from wilson_main.abstractions import VibState, MolecularSystem, MolecularProperty
from wilson.spectrum.vpt2 import anharm_corr_energies
from wilson.spectrum.vpt2 import identify_fermi, adjust_for_fermi_resonance, get_X

import logging
logger = logging.getLogger("wilson."+__name__)

def anharm_analyzer_data(system:MolecularSystem = None, props: list[MolecularProperty] = None, 
                    nc_sqrt_eigval: dict = None, 
                    regime: dict = None, regime_subinfo: dict = None, 
                    exclude_modes: list = None) -> list[VibState]:
    """
    Basically a wrapper for analyser; passes data to anharm_corr_energies where analysis happens...
        then puts into list[VibState] form

    returns self.states for VibAnaSetup, 
        and states: list[VibState]
    
            Returns VPT2 corrected energy levels of all states as a dictionary : {str(int): float}

    """
    logger.info('Starting anharm_analyzer()')
    
    # modes exclusion list update
    if exclude_modes is None:
        exclude_modes = []

    prop_dict = {i.trivial_name: i.vals for i in props}
    logger.debug(f'prop_dict {prop_dict.keys()}')

    # corrected_levels : funds, over2q, combo2q, over3q, combo3q
    # corrected_levels = anharm_corr_energies(upd_harmonic_energies,
    corrected_levels, fermi_resonance = anharm_corr_energies(nc_sqrt_eigval,
                                                             prop_dict['cff'], prop_dict['qff'], prop_dict['B'], prop_dict['coriolis'],
                                                             regime, exclude_modes)
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
                            all_states_corr[key] = corrected_levels[4][
                                i, j, k]

    all_states = copy.deepcopy(all_states_corr)
    one = {i: all_states[i] for i in all_states if len(i) == 1}
    two = {i: all_states[i] for i in all_states if len(i) == 2}

    logger.info('GVPT2 anharm corrected:')
    logger.info(repr(dict(sorted(one.items()))))
    logger.info(repr(dict(sorted(two.items()))))

    return all_states, fermi_resonance


def anharm_analyzer(system:MolecularSystem = None, props: list[MolecularProperty] = None, 
                    nc_sqrt_eigval: dict = None, 
                    regime: dict = None, regime_subinfo: dict = None, 
                    exclude_modes: list = None) -> list[VibState]:
# def anharm_corr_energies(harmonic_energies, cubic_forcefield, quartic_forcefield,
#                          rotational_constant, coriolis_constant, anharmonic_type,
#                          list2exclude):
    """
    Takes in cm-1 unit for all the arguments:
    UPD! harmonic_energies is a dictionary - parserObj.fundamentals_harmonic_int

        harmonic_energies, cubic_forcefield, quartic_forcefield, rotational_constant, coriolis_constant(unit?)
        (nmodes,);   (nmodes, nmodes, nmodes);   (nmodes, nmodes, nmodes, nmodes);   [x,y,z];   (nmodes, nmodes)

    anharmonic_type options:
            'VPT2'                        - don't do_res, don't do_var
            'DVPT2'                       - do_res, don't do_var
            'GVPT2'                       - do_res, do_var
    returns:
        fundamental, overtones, combotones, over3q, combo3q
    """

    if anharmonic_type == 'GVPT2':
        do_variational_correction = True
        do_resonance_checks = True
    elif anharmonic_type == 'VPT2':
        do_resonance_checks = False
        do_variational_correction = False
    elif anharmonic_type == 'DVPT2':
        do_resonance_checks = True
        do_variational_correction = False
    else:
        raise AssertionError("Unrecornized anharmonic type or it isn't specified")

    harmonic_energies = list(nc_sqrt_eigval.values())

    original_len_ene = len(harmonic_energies)
    harmonic_energies = {k: v for k, v in harmonic_energies.items() if k not in exclude_modes}

    fundamental = np.zeros((original_len_ene))
    overtones = np.zeros((original_len_ene))
    combotones = np.zeros((original_len_ene, original_len_ene))
    over3q = np.zeros((original_len_ene))
    combo3q = np.zeros((original_len_ene, original_len_ene, original_len_ene))

    fermi_resonance = identify_fermi(harmonic_energies, cubic_forcefield, do_resonance_checks)
    # selecting resonances fermi_resonance = [fermi_resonance[0]]
    X, X_cubic, X_quartic, X_coriolis = get_X(harmonic_energies, cubic_forcefield, quartic_forcefield,
                                              rotational_constant, coriolis_constant, do_resonance_checks,
                                              fermi_resonance, original_len_ene)


    if fermi_resonance: # if not an empty list
        logger.debug(f'Fermi resonances identified - {len(fermi_resonance)}: {fermi_resonance}')

    funds_corrections = np.zeros((original_len_ene))
    # for i in range(len(harmonic_energies)):
    for i in harmonic_energies:
        fundamental[i] += harmonic_energies[i] + 2 * X[i][i]

        fscr = 0
        # for j in range(len(harmonic_energies)):
        for j in harmonic_energies:
            if j != i:
                fscr += 0.5 * X[i][j]

        fundamental[i] += fscr
        funds_corrections[i] += 2 * X[i][i] + fscr

    overtones_corrections = np.zeros((original_len_ene))
    combotones_corrections = np.zeros((original_len_ene, original_len_ene))
    over3q_corrections = np.zeros((original_len_ene))
    combo3q_corrections = np.zeros((original_len_ene, original_len_ene, original_len_ene))

    # for i in range(len(harmonic_energies)):
    for i in harmonic_energies:
        overtones[i] += 2 * fundamental[i] + 2 * X[i][i]
        overtones_corrections[i] += 2 * X[i][i]

        over3q[i] += 3 * fundamental[i] + 6 * X[i][i]
        over3q_corrections[i] += 6 * X[i][i]

        # for j in range(len(harmonic_energies)):
        for j in harmonic_energies:
            if i == j:
                # for k in range(len(harmonic_energies)):
                for k in harmonic_energies:
                    if k != i:
                        combo3q[i][i][k] += 2 * fundamental[i] + 2 * X[i][i] + fundamental[k] + 2 * X[i][k]
                        combo3q_corrections[i][i][k] += 2 * X[i][i] + 2 * X[i][k]

            else:
                combotones[i][j] += fundamental[i] + fundamental[j] + X[i][j]
                combotones_corrections[i][j] += X[i][j]

                # for k in range(len(harmonic_energies)):
                for k in harmonic_energies:
                    if k == i or k == j:
                        continue
                    combo3q[i][j][k] += fundamental[i] + fundamental[j] + fundamental[k] + X[i][j] + X[i][k] + X[j][k]
                    combo3q_corrections[i][j][k] += X[i][j] + X[i][k] + X[j][k]

    if do_variational_correction:
        selectedFR = range((len(fermi_resonance)))
        adjusted_fundamental, adjusted_overtones, adjusted_combotones = \
            adjust_for_fermi_resonance(fundamental, overtones, combotones, over3q, combo3q, cubic_forcefield,
                                       [fermi_resonance[i] for i in selectedFR])
        return (adjusted_fundamental, adjusted_overtones, adjusted_combotones, over3q, combo3q), fermi_resonance

    else:
        return (fundamental, overtones, combotones, over3q, combo3q), fermi_resonance
