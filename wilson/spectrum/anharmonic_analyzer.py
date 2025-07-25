"""
(self.system, props, self.regime, self.regime_subinfo, self.nc_sqrt_eigval, self.nc_eigvec)
"""
import copy
from wilson_main.abstractions import VibState, MolecularSystem, MolecularProperty
from wilson.utils.debug import infoprint, separator_print, debugprint

def anharm_analyzer(system:MolecularSystem = None, props: list[MolecularProperty] = None, 
                    nc_sqrt_eigval: dict = None, 
                    regime: dict = None, regime_subinfo: dict = None, 
                    exclude_modes: list = None) -> list[VibState]:
    """
    returns self.states for VibAnaSetup, 
        and states: list[VibState]
    
            Returns VPT2 corrected energy levels of all states as a dictionary : {str(int): float}

    """
    separator_print('anharm_analyzer')
    from wilson.spectrum.vpt2 import anharm_corr_energies

    if exclude_modes is None:
        exclude_modes = []

    prop_dict = {i.triv_name: i.vals for i in props}
    debugprint(f'prop_dict {prop_dict.keys()}')

    # list, not associated to normal mode indices

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

    infoprint('GVPT2 anharm corrected:')
    infoprint('\n'+repr(dict(sorted(one.items()))))
    infoprint('\n'+repr(dict(sorted(two.items()))))

    return all_states, fermi_resonance
