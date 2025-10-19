from wilson_suite.wilson_derive.abstractions import ResonanceCondition, VibPerturbedTerm, PolProp, VibDiffTerm
from wilson_suite.wilson_intensities.spectrum import func_abstractions as f_abst 
import copy
import numpy as np
"""
Extra info on top of VibPerturbedTerm and its components:
 - [+] resonances motif - per list[ResonanceCondition]
 - [] list of axes (vars of eval function)

 for a collection of VibPerturbedTerms: 
    - identify valid choices of axes
    - axes choice
    - use the chosen axes in terms
    - find resonance motifs
"""
from dataclasses import dataclass
# @dataclass
# class PropsCollection:
#     props: list[PolProp]

#     def __eq__(self, other):
#         if isinstance(other, PropsCollection):
#             return all([p in other.props for p in self.props])
#         return False
# @dataclass
# class ResonanceMotif:
#     resonance_conditions: list[ResonanceCondition]

def make_vibdiff_motif(freqterms: list[VibDiffTerm]):
    """
    """
    return

def simple_prop_ID(property: PolProp) -> tuple[tuple, int]:
    """
    """
    operators = tuple([op.o for op in property.ops])
    return (operators, property.dord)

def make_avrg_props_motif(props: list[PolProp]) -> set[tuple]:
    """
    indices below are concrete, after | but could be others, main part of ID is in the numerator
    {((0, 3), 1),  ---- \\frac{\\partial\\alpha_{\\alpha\\delta}} | e.g. {\\partial Q_{b}}
     ((2,), 1),    ---- \\frac{\\partial\\mu_{\\gamma}} | e.g. {\\partial Q_{b}}
     ((1,), 1)}    ---- \\frac{\\partial\\mu_{\\beta}} | e.g. {\\partial Q_{a}}
    """
    return tuple(simple_prop_ID(prop) for prop in props if prop.ops)

def make_resonance_motif(res_conds: list[ResonanceCondition]) -> tuple:
    """
    """
    conditions = []

    for cond in res_conds:

        new_pf = tuple(cond.pf)
        new_diff = tuple([tuple(cond.diff.sl.q), tuple(cond.diff.sr.q)])

        conditions.append(tuple([new_diff, new_pf]))

    return tuple(conditions)

def get_axes_in_resmotif(motif: tuple):
    """
    for ResonanceMotif
    """
    return set([ax.strip('-') for rcond in motif for ax in rcond[1]])

def get_indlabels_in_resmotif(motif: tuple):
    """
    for ResonanceMotif
    """
    indlabels_list = set([indlabels for rcond in motif for indlabels in rcond[0]])
    return set(label for labels in indlabels_list for label in labels)

def identify_unique_resmotifs(list_of_terms: list[VibPerturbedTerm]) -> set[tuple]:
    """
    """
    return set(make_resonance_motif(term.res) for term in list_of_terms)

def identify_unique_avrgmotifs(list_of_terms: list[VibPerturbedTerm]):
    """
    """
    return set(make_avrg_props_motif(term.props) for term in list_of_terms)

def identify_maximum_axes_in_terms(list_of_terms: list[VibPerturbedTerm]):
    """
    """
    unique = identify_unique_resmotifs(list_of_terms)
    axes: list[tuple[str]] = [cond[1] for motif in list(unique) for cond in motif]

    axes_in_these_terms = set([ax_tuple[0].strip('-') for ax_tuple in axes])
    return len(axes_in_these_terms)

def motifs_control(list_of_terms: list[VibPerturbedTerm]):
    """
    what axes are in the motifs and give info and suggestions
    """
    total_num_axes = identify_maximum_axes_in_terms(list_of_terms)
    unique = identify_unique_resmotifs(list_of_terms)
    axes_per_motif = {m: get_axes_in_resmotif(motif=m) for m in unique}

    return axes_per_motif, total_num_axes

def terms_for_motif(terms: list[VibPerturbedTerm]) -> dict[tuple, list]:
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

def initialize_resonance_dict(motif):
    """
    Initialize a dictionary with axes in the motif as keys.
    """
    axes_in_motif = sorted(get_axes_in_resmotif(motif))
    return {ax: None for ax in axes_in_motif}

def generate_index_choices(motif, vibstates_data: f_abst.VibStatesData):
    """
    Generate all possible index combinations for the given motif.
    """
    indlabels_in_motif = sorted(list(get_indlabels_in_resmotif(motif)))
    labels = vibstates_data.harmonic_osc_states_labels
    import itertools
    return [dict(zip(indlabels_in_motif, combo)) for combo in itertools.product(labels, repeat=len(indlabels_in_motif))]


def find_resonance_locations_wrt_index_choices(motif: tuple[tuple,...], 
                                               vibstates_data: f_abst.VibStatesData, 
                                               spec_window=None) -> dict:
    """
    """
    from ..spectrum.func_evaluation import solve_LSE_motif, ParameterSet
    res_loc_dict = initialize_resonance_dict(motif)
    print('res_loc_dict', res_loc_dict)

    # Use or adapt solve_LSE_resonance with information from motif to get resonance locations

    # If necessary, format resulting data to the below structure

    # {motif 1: {(500., 1200.): [(1, 2), (1, 3)],
    #           (500., 1400.): [(1, 4)], ...}}

    results: dict[dict,list] = {}    
    
    index_choices = generate_index_choices(motif, vibstates_data)

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

def is_location_in_window(location: dict, window: dict, margins: dict=None):
    """
    location = {'A': np.float64(485.0), 'B': np.float64(-2023.0)}
    window = {'A': (min, max), 'B': (min, max)}
    margin = {'B': (margmin, margmax)}
    """
    if not isinstance(location, dict) or not isinstance(window, dict):
        raise ValueError("Both 'location' and 'window' must be dictionaries.")
    if margins is not None and not isinstance(margins, dict):
        raise ValueError("'margin' must be a dictionary or None.")
    
    adjusted_window = {
        axis: (
            window[axis][0] + margins[axis][0] if margins and axis in margins else window[axis][0],
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

def calculate_term_coeffs_for_indices(terms, motif_res_loc):

    # Suggestion sketch for overall steps

    # Identify all unique parts that can be precalculated for a sensible partitioning of the term parts
    # One function here

    # Perform the precalculation and keep in a structure to yet be decided
    # One function here

    # Current implementation target for the previous two fns: Make them work in the general care and
    # don't worry about optimization yet

    # Calculate the coefficients for the terms by combining the precalculated parts
    # One function here

    # Would return a structure {term: {resonance index tuple: term coeff, ...}, ...}

    pass


def identify_precalc_unique_terms_parts(terms: list[VibPerturbedTerm]):
    """
    Identify all unique parts that can be precalculated for 
            a sensible partitioning of the term parts

    1. orient. averaged props - identify unique patterns
    2. non-orient. avrg. props. - skip further if zero
    2. vibdiffs_bank - ?
    """

    return

'''
def evaluate_term_coeffs(term, relevant_indices):

   # Evaluate the coefficient part of the term 'term' for all of the indices in 'relevant_index_tuples'

    # Return data or associate with term instance question not settled yet

    pass
'''

def find_domain_groups_by_distance(res_locations, distance_threshold):
    """
    using scikit-learn to cluster points with distance threshold
    """
    from sklearn import cluster
    ward = cluster.AgglomerativeClustering(linkage="ward", 
                                           distance_threshold=distance_threshold, 
                                           n_clusters=None)
    labels = ward.fit_predict(res_locations)
    groups: dict[int, list] = {}
    for i, label in enumerate(labels):
        groups.setdefault(int(label), []).append(res_locations[i])
    return groups

def find_distance_threshold(dynamic_range, Gamma_axes: dict):
    """
    Gamma is a dictionary {'A': Gamma_A, 'B': Gamma_B, ...}

    at Gamma   - 1/2 of maximum
    at Gamma/2 - 4/5 of maximum
    """
    multiplier = np.sqrt((dynamic_range-1)/dynamic_range)
    gammas = [-1j*G for G in Gamma_axes.values()]
    dist_ax = [G*multiplier for G in Gamma_axes.values()]
    print(dist_ax)
    gamma_prod = np.prod(gammas)
    base_intensity = 1./gamma_prod
    min_to_show = base_intensity/dynamic_range
    
    raise NotImplementedError

def determine_domains_and_features(features_to_draw):
    """
    features_to_draw is a dict:
        {motif 1: {(500., 1200.): [(1, 2), (1, 3)],
                (500., 1400.): [(1, 4)], ...},
        motif 2: {}}
    
    features_to_draw[i][(state_tuple), (location_tuple)] for i in res_motifs = coeff as float
    """
    return

def get_domain_grids(domains_with_features):

    pass