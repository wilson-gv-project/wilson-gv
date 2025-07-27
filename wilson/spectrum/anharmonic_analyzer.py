"""
Anharmonic analyzer for wilson_main.VibAnaSetup.doAnharmonicAnalysis()

"""
from wilson_utils.abstractions import VibState
from wilson_main.abstractions import MolecularSystem, MolecularProperty
from wilson.spectrum.vpt2 import anharm_corr_energies

import logging
logger = logging.getLogger("wilson."+__name__)

def anharm_analyzer_data(system:MolecularSystem = None, props: list[MolecularProperty] = None, 
                    nc_sqrt_eigval: dict = None, 
                    regime: str = None, regime_subinfo: dict = None, 
                    exclude_modes: list = None) -> tuple[list[VibState], dict]:
    """
    Basically a wrapper for analyser; passes data to anharm_corr_energies where analysis happens...
        then puts into list[VibState] form

    returns self.states for VibAnaSetup, and states are list[VibState]
    
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
        vibstates.append(VibState(s={st: 1.0}, e=all_states_corr[st]))

    logger.debug('GVPT2 anharm corrected:')
    logger.debug(f'vibstates: {vibstates}')

    diagnostics = {'fermi_resonances': fermi_resonances}
    
    return vibstates, diagnostics


# def anharm_analyzer(system:MolecularSystem = None, props: list[MolecularProperty] = None,
#                     nc_sqrt_eigval: dict = None,
#                     regime: str = None, regime_subinfo: dict = None,
#                     exclude_modes: list = None) -> list[VibState]:
#     """
#     Takes in cm-1 unit for all the arguments:
#     UPD! harmonic_energies is a dictionary - parserObj.fundamentals_harmonic_int

#         harmonic_energies, cubic_forcefield, quartic_forcefield, rotational_constant, coriolis_constant(unit?)
#         (nmodes,);   (nmodes, nmodes, nmodes);   (nmodes, nmodes, nmodes, nmodes);   [x,y,z];   (nmodes, nmodes)

#     anharmonic_type options:
#             'VPT2'                        - don't do_res, don't do_var
#             'DVPT2'                       - do_res, don't do_var
#             'GVPT2'                       - do_res, do_var
#     returns:
#         fundamental, overtones, combotones, over3q, combo3q
#     """

#     if regime == 'GVPT2':
#         do_variational_correction = True
#         do_resonance_checks = True
#     elif regime == 'VPT2':
#         do_resonance_checks = False
#         do_variational_correction = False
#     elif regime == 'DVPT2':
#         do_resonance_checks = True
#         do_variational_correction = False
#     else:
#         raise AssertionError("Unrecornized anharmonic type or it isn't specified")

#     if exclude_modes is None:
#         exclude_modes = []

#     original_len_ene = len(nc_sqrt_eigval)
#     harmonic_energies = {k: v for k, v in nc_sqrt_eigval.items() if k not in exclude_modes}

#     fundamental = np.zeros((original_len_ene))
#     overtones = np.zeros((original_len_ene))
#     combotones = np.zeros((original_len_ene, original_len_ene))
#     over3q = np.zeros((original_len_ene))
#     combo3q = np.zeros((original_len_ene, original_len_ene, original_len_ene))

#     prop_dict = {i.trivial_name: i.vals for i in props}
#     logger.debug(f'prop_dict {prop_dict.keys()}')

#     cubic_forcefield = prop_dict['cff']
#     quartic_forcefield = prop_dict['qff']
#     rotational_constant = prop_dict['B']
#     coriolis_constant = prop_dict['coriolis']

#     fermi_resonance = identify_fermi(harmonic_energies, cubic_forcefield, do_resonance_checks)
#     # selecting resonances fermi_resonance = [fermi_resonance[0]]
#     X, X_cubic, X_quartic, X_coriolis = get_X(harmonic_energies, cubic_forcefield, quartic_forcefield,
#                                               rotational_constant, coriolis_constant, do_resonance_checks,
#                                               fermi_resonance, original_len_ene)

#     if fermi_resonance: # if not an empty list
#         logger.debug(f'Fermi resonances identified - {len(fermi_resonance)}: {fermi_resonance}')

#     funds_corrections = np.zeros((original_len_ene))
#     for i in harmonic_energies:
#         fundamental[i] += harmonic_energies[i] + 2 * X[i][i]

#         fscr = 0
#         for j in harmonic_energies:
#             if j != i:
#                 fscr += 0.5 * X[i][j]

#         fundamental[i] += fscr
#         funds_corrections[i] += 2 * X[i][i] + fscr

#     overtones_corrections = np.zeros((original_len_ene))
#     combotones_corrections = np.zeros((original_len_ene, original_len_ene))
#     over3q_corrections = np.zeros((original_len_ene))
#     combo3q_corrections = np.zeros((original_len_ene, original_len_ene, original_len_ene))

#     for i in harmonic_energies:
#         overtones[i] += 2 * fundamental[i] + 2 * X[i][i]
#         overtones_corrections[i] += 2 * X[i][i]

#         over3q[i] += 3 * fundamental[i] + 6 * X[i][i]
#         over3q_corrections[i] += 6 * X[i][i]

#         for j in harmonic_energies:
#             if i == j:
#                 for k in harmonic_energies:
#                     if k != i:
#                         combo3q[i][i][k] += 2 * fundamental[i] + 2 * X[i][i] + fundamental[k] + 2 * X[i][k]
#                         combo3q_corrections[i][i][k] += 2 * X[i][i] + 2 * X[i][k]

#             else:
#                 combotones[i][j] += fundamental[i] + fundamental[j] + X[i][j]
#                 combotones_corrections[i][j] += X[i][j]

#                 for k in harmonic_energies:
#                     if k == i or k == j:
#                         continue
#                     combo3q[i][j][k] += fundamental[i] + fundamental[j] + fundamental[k] + X[i][j] + X[i][k] + X[j][k]
#                     combo3q_corrections[i][j][k] += X[i][j] + X[i][k] + X[j][k]

#     if do_variational_correction:
#         selectedFR = range((len(fermi_resonance)))
#         adjusted_fundamental, adjusted_overtones, adjusted_combotones = \
#             adjust_for_fermi_resonance(fundamental, overtones, combotones, over3q, combo3q, cubic_forcefield,
#                                        [fermi_resonance[i] for i in selectedFR])
#         return (adjusted_fundamental, adjusted_overtones, adjusted_combotones, over3q, combo3q), fermi_resonance

#     else:
#         return (fundamental, overtones, combotones, over3q, combo3q), fermi_resonance
