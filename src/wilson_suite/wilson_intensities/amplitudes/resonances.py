"""
RESONANCE LOCATIONS for ResonanceMotif/VibPerturbedTerm
RESONANCES in VibPerturbedTerm
"""
import numpy as np
from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import ResLocGeoObject
from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, VibStatesData, ResonanceMotif
from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiff, VibDiffCache


def is_location_in_window(location: dict, window: dict, margins: dict=None):
    """
    location = {'A': np.float64(485.0), 'B': np.float64(-2023.0)}
    window = {'A': (min, max), 'B': (min, max)}
    margin = {'B': (margmin, margmax)} # positive means extention of the boundary, negative means shrinking
    """
    if not isinstance(location, dict) or not isinstance(window, dict):
        raise ValueError("Both 'location' and 'window' must be dictionaries.")
    if margins is not None and not isinstance(margins, dict):
        raise ValueError("'margin' must be a dictionary or None.")

    adjusted_window = {
        axis: (
            window[axis][0] - margins[axis][0] if margins and axis in margins else window[axis][0],
            window[axis][1] + margins[axis][1] if margins and axis in margins else window[axis][1]
        )
        for axis in window
    }

    for axis, value in location.items():
        if axis not in adjusted_window:
            raise ValueError(f"Axis '{axis}' in location is not defined in the window.")

        min_val, max_val = adjusted_window[axis]
        if not (min_val < value < max_val):
            return False
    return True


def generate_LHS_motif(motif: ResonanceMotif):
    """
    motif is a tuple/collection of res_conditions
        res_conditions is a tuple of (vib_difference, axes)
            vib_difference is a tuple of states indices
    """
    from wilson_suite.wilson_utils.common_labels import num_cap_alpha_labels
    # maximum different normal mode index across all tuples
    max_different_freq_axes = motif.get_max_different_freq_axes()
    num_axes = len(max_different_freq_axes)

    if num_axes == 1:
        num_axes = len(motif)

    # to identify coeff matrix shape
    coeff_matrix = np.zeros((num_axes,num_axes))

    for i, r_condition in enumerate(motif):
        axis_tupleID: tuple[str] = tuple(r_condition.pf)

        # axis_tupleID = ('A', '-B') --> {'A': 1, 'B': -1} better?
        # coeffs {'A': 1, 'B': -1}
        coeffs = {var.strip('-') : 1 if '-' not in var else -1 for var in axis_tupleID}

        for alpha_label, coefficient in coeffs.items():
             # Reverse the sign (FIXME???) and place it in the correct position
             coeff_matrix[i, num_cap_alpha_labels[alpha_label]] = -1 * np.sign(coefficient)

    return coeff_matrix


# def get_RHS_motif(motif: tuple[tuple,...],
def get_RHS_motif(motif: ResonanceMotif,
            parameters: ParameterSet, vibdata: VibStatesData,
            vibdiff_cache: VibDiffCache,
            unit: str='Eh'):
    """
    making a constants vector from a list of tuples
    resonance_tuples = [(1, (-1,)), (2, (-1, 2)), (3, (-2, 3))]
    ind_tuple = (1, 2, 3) --- 
    vibdiffbank: VibDiffBank instance

    output: [5, -3, 2]
    """
    constants = []

    for res_cond in motif:
        vib_diff_w_value = VibDiff.from_symbolic(res_cond.diff, 
                                                    parameters, 
                                                    vibdata)
        vib_diff_w_value.cache_it(vibdiff_cache=vibdiff_cache)
        constants.append((-1)*vib_diff_w_value.energy_difference(au=(unit=='Eh')))

    return constants


# works with .func_abstractions
def solve_LSE_motif(motif: ResonanceMotif,
                    parameters: ParameterSet, vibdata: VibStatesData,
                    vibdiff_cache: VibDiffCache,
                    unit: str='Eh') -> ResLocGeoObject:
    """
    solving a linear system of equations
    coeff_matrix = [[1, 0, 0], [1, -1, 0], [0, 1, -1]]
    constants = [5, -3, 2]
    output: [5. 2. 0.]

    returns a dict {f'w{i+1}': solution}
    """

    coeff_matrix = generate_LHS_motif(motif)

    constants = get_RHS_motif(motif, parameters, vibdata, vibdiff_cache, unit)

    A = np.array(coeff_matrix)
    b = np.array(constants)

    try:
        solution = np.linalg.solve(A, b)
        from wilson_suite.wilson_utils.common_labels import num_cap_alpha_labels
        num_to_ax = {v:k for k,v in num_cap_alpha_labels.items()}

        return ResLocGeoObject({num_to_ax[i]: float(val) for i, val in enumerate(solution)})
    except np.linalg.LinAlgError as e:
        print("Error solving linear system:", e)


def _generate_index_choices(motif: ResonanceMotif, vibstates_data: 'VibStatesData'):
    """
    Generate all possible index combinations for the given motif.
    """
    from ..amplitudes.utils import generate_index_choices_general
    indlabels_in_motif = sorted(list(motif.get_nm_indices()))
    labels = vibstates_data.harmonic_osc_states_labels
    return generate_index_choices_general(indlabels_in_motif=indlabels_in_motif, labels=labels)


def find_resonance_locations_wrt_index_choices(motif: ResonanceMotif,
                                               vibstates_data: 'VibStatesData',
                                               vibdiff_cache: 'VibDiffCache',
                                               spec_window=None) -> dict[ResonanceMotif,dict[ResLocGeoObject,list]]:
    """
    """
    from ..amplitudes.term_parts import ParameterSet

    # Use or adapt solve_LSE_resonance with information from motif to get resonance locations

    # If necessary, format resulting data to the below structure

    # {motif 1: {(500., 1200.): [(1, 2), (1, 3)],
    #           (500., 1400.): [(1, 4)], ...}}

    results: dict[ResonanceMotif,dict[ResLocGeoObject,list]] = {motif: {}}

    index_choices = _generate_index_choices(motif, vibstates_data)
    for idxs in index_choices:
        parameters = ParameterSet(idxs)
        location_key = solve_LSE_motif(motif, parameters, vibstates_data, vibdiff_cache, unit='cm-1')

        if spec_window is None or is_location_in_window(location_key, window=spec_window, margin={}):
            results[motif].setdefault(location_key, []).append(idxs)
    return results



def get_indlabels_in_resmotif(motif: tuple):
    """
    for ResonanceMotif
    """
    indlabels_list = set([indlabels for rcond in motif for indlabels in rcond[0]])
    return set(label for labels in indlabels_list for label in labels)



def identify_unique_resmotifs(list_of_terms: list['VibPerturbedTerm']) -> set[ResonanceMotif]:
    """
    """
    return set(ResonanceMotif(term.res) for term in list_of_terms)


def identify_maximum_axes_in_terms(list_of_terms: list['VibPerturbedTerm']):
    """
    """
    unique = identify_unique_resmotifs(list_of_terms)
    axes: list[tuple[str]] = [cond.pf for motif in unique for cond in motif]

    axes_in_these_terms = set([ax_tuple[0].strip('-') for ax_tuple in axes])
    return len(axes_in_these_terms)




