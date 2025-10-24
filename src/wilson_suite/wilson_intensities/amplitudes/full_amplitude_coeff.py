from typing import TYPE_CHECKING

from wilson_suite.wilson_intensities.amplitudes.averaged_props import identify_unique_avrgmotifs
from wilson_suite.wilson_intensities.amplitudes.vibene_differences import identify_unique_vibdiff_motifs
if TYPE_CHECKING:
    from wilson_suite.wilson_derive.abstractions import VibPerturbedTerm


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



"""
FULL TERM COEFFICIENT for VibPerturbedTerm
"""
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

def identify_precalc_unique_coeff_parts(terms: list['VibPerturbedTerm']):
    """
    Identify all unique parts that can be precalculated for 
            a sensible partitioning of the term parts

    1. orient. averaged props - identify unique patterns
    2. non-orient. avrg. props. - skip further if zero
    2. vibdiffs_bank - ?
    """
    return {'avrg_motifs': identify_unique_avrgmotifs(terms), 
            'vibdiff_motifs': identify_unique_vibdiff_motifs(terms)}


'''
def evaluate_term_coeffs(term, relevant_indices):

   # Evaluate the coefficient part of the term 'term' for all of the indices in 'relevant_index_tuples'

    # Return data or associate with term instance question not settled yet

    pass
'''


