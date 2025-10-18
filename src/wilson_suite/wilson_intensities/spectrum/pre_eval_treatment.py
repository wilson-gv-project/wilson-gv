from wilson_suite.wilson_derive.abstractions import ResonanceCondition, VibPerturbedTerm
import copy
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
def make_resonance_motif(res_conds: list[ResonanceCondition]) -> tuple:
    """
    """
    conditions = []

    for cond in res_conds:

        new_pf = tuple(cond.pf)
        new_diff = tuple([tuple(cond.diff.sl.q), tuple(cond.diff.sr.q)])

        conditions.append(tuple([new_diff, new_pf]))

    return tuple(conditions)

def get_axes_in_motif(motif: tuple):
    """
    """
    return set([ax.strip('-') for rc in motif for ax in rc[1]])
    
def identify_unique_resmotifs(list_of_terms: list[VibPerturbedTerm]) -> set[tuple]:
    """
    """
    return set(make_resonance_motif(term.res) for term in list_of_terms)

def identify_maximum_axes(list_of_terms: list[VibPerturbedTerm]):
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
    total_num_axes = identify_maximum_axes(list_of_terms)
    unique = identify_unique_resmotifs(list_of_terms)
    axes_per_motif = {m: get_axes_in_motif(motif=m) for m in unique}

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

def find_resonance_locations_wrt_index_choices(motif, vibstates_data, spec_window=None) -> dict:
    from ..spectrum.func_evaluation import solve_LSE_motif, ParameterSet
    axes_in_motif = get_axes_in_motif(motif)
    res_loc_dict = {ax: None for ax in axes_in_motif}

    # Use or adapt solve_LSE_resonance with information from motif to get resonance locations

    # If necessary, format resulting data to the below structure

    # {motif 1: {(500., 1200.): [(1, 2), (1, 3)],
    #           (500., 1400.): [(1, 4)], ...}}

    results: dict[dict,list] = {}

    labels = ['1', '3'] # should be input (?) with vibstates_data(?)
    indices = ['a', 'b'] # should get from motif
    
    import itertools
    index_choices = [dict(zip(indices, combo)) for combo in itertools.product(labels, repeat=len(indices))]
    print(index_choices)

    for idxs in index_choices:
        parameters = ParameterSet(idxs)
        location_d = solve_LSE_motif(motif, parameters, vibstates_data, unit='cm-1')
        if tuple(location_d.items()) not in results:
            results[tuple(location_d.items())] = [idxs]
        else:
            results[tuple(location_d.items())].append(idxs)

    return results


def crop_resonances_to_window(resonances, spec_window, tolerance):

    pass

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

'''
def evaluate_term_coeffs(term, relevant_indices):

   # Evaluate the coefficient part of the term 'term' for all of the indices in 'relevant_index_tuples'

    # Return data or associate with term instance question not settled yet

    pass
'''

def determine_domains_and_features(features_to_draw):

    pass

def get_domain_grids(domains_with_features):

    pass