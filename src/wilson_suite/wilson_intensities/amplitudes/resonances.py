"""
RESONANCE LOCATIONS for ResonanceMotif/VibPerturbedTerm
RESONANCES in VibPerturbedTerm
"""
import numpy as np
from wilson_suite.wilson_analysis.render.render import get_axes_in_resmotif
from wilson_suite.wilson_derive.abstractions import ResonanceCondition, VibPerturbedTerm
from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, VibStatesData
from wilson_suite.wilson_intensities.amplitudes.vibene_differences import get_vibdiff_motif
import copy

from wilson_suite.wilson_intensities.amplitudes.utils import initialize_resonance_dict

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


def generate_LHS_motif(motif: tuple[tuple,...]):
    """
    motif is a tuple/collection of res_conditions
        res_conditions is a tuple of (vib_difference, axes)
            vib_difference is a tuple of states indices
    """
    from wilson_suite.wilson_utils.common_labels import num_cap_alpha_labels
    # maximum variable index across all tuples
    max_var_index = max([len(rc[1]) for rc in motif])

    if max_var_index == 1:
        max_var_index = len(motif)

    # to identify coeff matrix shape
    coeff_matrix = np.zeros((max_var_index, max_var_index))

    for i, r_condition in enumerate(motif):
        axis_tupleID: tuple[str] = r_condition[1]

        # axis_tupleID = ('A', '-B') --> {'A': 1, 'B': -1} better?
        # coeffs {'A': 1, 'B': -1}
        coeffs = {var.strip('-') : 1 if '-' not in var else -1 for var in axis_tupleID}

        for alpha_label, coefficient in coeffs.items():
             # Reverse the sign and place it in the correct position
             coeff_matrix[i, num_cap_alpha_labels[alpha_label]] = -1 * np.sign(coefficient)

    return coeff_matrix


def get_RHS_motif(motif: tuple[tuple,...],
            parameters: ParameterSet, vibdata: VibStatesData,
            unit: str='Eh',
            eval_mode: str = 'on-the-fly'):
    """
    making a constants vector from a list of tuples
    resonance_tuples = [(1, (-1,)), (2, (-1, 2)), (3, (-2, 3))]
    ind_tuple = (1, 2, 3) --- 
    vibdiffbank: VibDiffBank instance

    output: [5, -3, 2]
    """
    if eval_mode == 'on-the-fly':
        constants = [(-1)*get_vibdiff_motif(vibdiff_symb=rc[0], parameters=parameters,
                                            allstates_map=vibdata.allstates_map, unit=unit) for rc in motif]
    else:
        raise NotImplementedError('RHS can be only "on-the-fly" now')
    return constants


# works with .func_abstractions
def solve_LSE_motif(motif: tuple[tuple,...],
                    parameters: ParameterSet, vibdata: VibStatesData,
                    unit: str='Eh',
                    eval_mode: str = 'on-the-fly'):
    """
    solving a linear system of equations
    coeff_matrix = [[1, 0, 0], [1, -1, 0], [0, 1, -1]]
    constants = [5, -3, 2]
    output: [5. 2. 0.]

    returns a dict {f'w{i+1}': solution}
    """
    coeff_matrix = generate_LHS_motif(motif)
    constants = get_RHS_motif(motif, parameters, vibdata, unit, eval_mode)

    A = np.array(coeff_matrix)
    b = np.array(constants)

    try:
        solution = np.linalg.solve(A, b)
        from wilson_suite.wilson_utils.common_labels import num_cap_alpha_labels
        num_to_ax = {v:k for k,v in num_cap_alpha_labels.items()}

        return {num_to_ax[i]: val for i, val in enumerate(solution)}
    except np.linalg.LinAlgError as e:
        print("Error solving linear system:", e)


def _generate_index_choices(motif, vibstates_data: 'VibStatesData'):
    """
    Generate all possible index combinations for the given motif.
    """
    from ..amplitudes.utils import generate_index_choices_general
    indlabels_in_motif = sorted(list(get_indlabels_in_resmotif(motif)))
    labels = vibstates_data.harmonic_osc_states_labels
    return generate_index_choices_general(indlabels_in_motif=indlabels_in_motif, labels=labels)


def find_resonance_locations_wrt_index_choices(motif: tuple[tuple,...],
                                               vibstates_data: 'VibStatesData',
                                               spec_window=None) -> dict:
    """
    """
    from ..amplitudes.term_parts import ParameterSet
    res_loc_dict = initialize_resonance_dict(motif)
    # print('res_loc_dict', res_loc_dict)

    # Use or adapt solve_LSE_resonance with information from motif to get resonance locations

    # If necessary, format resulting data to the below structure

    # {motif 1: {(500., 1200.): [(1, 2), (1, 3)],
    #           (500., 1400.): [(1, 4)], ...}}

    results: dict[dict,list] = {}

    index_choices = _generate_index_choices(motif, vibstates_data)

    for idxs in index_choices:
        parameters = ParameterSet(idxs)
        location_d = solve_LSE_motif(motif, parameters, vibstates_data, unit='cm-1')
        location_key = tuple(location_d.items())

        if spec_window is None or is_location_in_window(location_d, window=spec_window, margin={}):
            results.setdefault(location_key, []).append(idxs)

    return results


def crop_resonances_to_window(resonances: tuple[dict], spec_window: dict, margins) -> list[dict]:
    """
    take collection of res loc dicts and return a new collection of only ones in the window
    """
    return [resloc for resloc in resonances if is_location_in_window(resloc, window=spec_window, margins=margins)]


def make_resonance_motif(res_conds: list['ResonanceCondition']) -> tuple:
    """
    """
    conditions = []

    for cond in res_conds:

        new_pf = tuple(cond.pf)
        new_diff = tuple([tuple(cond.diff.sl.q), tuple(cond.diff.sr.q)])

        conditions.append(tuple([new_diff, new_pf]))

    return tuple(conditions)


def get_indlabels_in_resmotif(motif: tuple):
    """
    for ResonanceMotif
    """
    indlabels_list = set([indlabels for rcond in motif for indlabels in rcond[0]])
    return set(label for labels in indlabels_list for label in labels)


def identify_unique_resmotifs(list_of_terms: list['VibPerturbedTerm']) -> set[tuple]:
    """
    """
    return set(make_resonance_motif(term.res) for term in list_of_terms)


def identify_maximum_axes_in_terms(list_of_terms: list['VibPerturbedTerm']):
    """
    """
    unique = identify_unique_resmotifs(list_of_terms)
    axes: list[tuple[str]] = [cond[1] for motif in list(unique) for cond in motif]

    axes_in_these_terms = set([ax_tuple[0].strip('-') for ax_tuple in axes])
    return len(axes_in_these_terms)


def motifs_control(list_of_terms: list['VibPerturbedTerm']):
    """
    what axes are in the motifs and give info and suggestions
    """
    total_num_axes = identify_maximum_axes_in_terms(list_of_terms)
    unique = identify_unique_resmotifs(list_of_terms)
    axes_per_motif = {m: get_axes_in_resmotif(motif=m) for m in unique}

    return axes_per_motif, total_num_axes


def terms_for_motif(terms: list['VibPerturbedTerm']) -> dict[tuple, list]:
    """
    """
    terms_for_motif: dict[tuple, list] = {}

    for t in terms:
        this_term_motif = make_resonance_motif(t.res)

        if this_term_motif in terms_for_motif:
            terms_for_motif[this_term_motif].append(copy.deepcopy(t))
        else:
            terms_for_motif[this_term_motif] = [copy.deepcopy(t)]

    return terms_for_motif