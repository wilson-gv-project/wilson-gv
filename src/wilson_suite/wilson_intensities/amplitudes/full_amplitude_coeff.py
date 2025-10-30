from typing import TYPE_CHECKING, Any

from wilson_suite.wilson_intensities.amplitudes.vibene_differences import identify_unique_vibdiff_motifs
from wilson_suite.wilson_intensities.amplitudes.resonances import identify_unique_resmotifs
import wilson_suite.wilson_intensities.amplitudes.averaged_props as avrgprops
import wilson_suite.wilson_intensities.amplitudes.vibene_differences as vediff

from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet
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
def calculate_term_coeffs_for_indices(terms: list['VibPerturbedTerm'], 
                                      motif_res_loc, settings: dict):

    # Suggestion sketch for overall steps

    # Identify all unique parts that can be precalculated for a sensible partitioning of the term parts
    # One function here
    need_to_precalc = identify_precalc_unique_coeff_parts(terms)

    # Perform the precalculation and keep in a structure to yet be decided
    # One function here
    precalculate_unique_coeff_parts(need_to_precalc, settings)
    
    # Current implementation target for the previous two fns: Make them work in the general care and
    # don't worry about optimization yet

    # Calculate the coefficients for the terms by combining the precalculated parts
    # One function here

    # Would return a structure {term: {resonance index tuple: term coeff, ...}, ...}

    pass

def identify_precalc_unique_coeff_parts(terms: list['VibPerturbedTerm']) -> dict[str, Any]:
    """
    Identify all unique parts that can be precalculated for 
            a sensible partitioning of the term parts

    1. orient. averaged props - identify unique patterns +???
    2. non-orient. avrg. props. - skip further if zero
    2. vibdiffs_bank - will be cached - calculated on the fly and saved
    """
    avrg_expressions = [avrgprops.PropsCollection(props=term.props).get_averaged_props().sort() for term in terms]

    return {
            'avrg_tensors': avrgprops.identify_unique_avrg_tensors(avrg_expressions), 
            'avrg_expr_tensor_mapping': avrgprops.make_unique_avrg_tensors_mapping(avrg_expressions),
            'vibenedenoms_tensors': vediff.identify_vibenedenoms(terms),
            'vibdiff_motifs': identify_unique_vibdiff_motifs(terms)
            # 'resonances_motifs': identify_unique_resmotifs(terms)
            }

def precalculate_unique_coeff_parts(need_to_precalc: dict, settings: dict):
    """
        data = {'avrg_tensors': {}}

    """
    data = {'avrg_tensors': {}, 
            'avrg_expr_tensor_mapping': need_to_precalc['avrg_expr_tensor_mapping'],
            'vibenedenoms_tensors': {}}

    for avrg_tensor in need_to_precalc['avrg_tensors']:
        data['avrg_tensors'][avrg_tensor] = avrgprops.calculate_avrg_tensor(avrg_expression=avrg_tensor, 
                                                                            polarization=settings['polarization'],
                                                                            number_of_nmodes=settings['number_of_nmodes'],
                                                                            props_data=settings['props_data'])
    
    for ve_denom in need_to_precalc['vibenedenoms_tensors']:
        data['vibenedenoms_tensors'][ve_denom] = vediff.calculate_vibenedenom_tensor(vibenedenom_inds=ve_denom, 
                                                                                     vibstates_data=settings['vibstates_data'])
    return data


def evaluate_term_coeffs(term: 'VibPerturbedTerm', relevant_indices: list[dict], pre_eval_data):
    """
    safety function to check relevant_indices?
    """
    # Evaluate the coefficient part of the term 'term' for all of the indices in 'relevant_index_tuples'

    # Return data or associate with term instance question not settled yet
    results = {}

    avrg_expr = avrgprops.PropsCollection(props=term.props).get_averaged_props()
    non_avrg_expr = avrgprops.PropsCollection(props=term.props).get_non_averaged_props()
    print('non_avrg_expr', non_avrg_expr)

    avrg_tensor_expr = pre_eval_data['avrg_expr_tensor_mapping'][avrg_expr]
    avrg_tensor = pre_eval_data['avrg_tensors'][avrg_tensor_expr]
    
    freqterms = vediff.FreqTermsCollection(freqterms=term.freqterms)
    extra_freqterms = freqterms.get_pert_wf_diff()

    # print('extra_freqterms', extra_freqterms)

    vibenedenoms_tensor = pre_eval_data['vibenedenoms_tensors'][freqterms.get_num_indices_vibenedenom()]

    # print('vibenedenoms_tensor.shape', vibenedenoms_tensor.shape)

    # print('avrg_expr', avrg_expr)
    # print('avrg_tensor_expr', avrg_tensor_expr)

    idx_summ, idx_nonsumm = term.tellNonSummSummIndices()
    term_idx_all = sorted(idx_summ + idx_nonsumm)

    # print('\nterm.tellNonSummSummIndices()', idx_summ, idx_nonsumm)
    # print('term_idx_all', term_idx_all, '\n')

    for index_dict in relevant_indices:
        if not all([index in index_dict for index in term_idx_all]):
            raise ValueError('index_dict is missing values for some indices')

        for non_avrg_prop in non_avrg_expr:
            print('non_avrg_prop', non_avrg_prop)
        sorted_index_dict = dict(sorted(index_dict.items()))
        
        # works if all unique indices are in index_dict
        index_tuple = tuple([v for v in sorted_index_dict.values()]) # ???

        avrg_index_tuple = avrgprops.get_ind_tuple_from_base(expr=avrg_expr, base_expr=avrg_tensor_expr, index_dict=index_dict)
        
        AVRG = avrg_tensor[avrg_index_tuple]
        # print('avrg_index_tuple', avrg_index_tuple)
        # print('index_tuple', index_tuple)

        # should be a single float result?
        results[ParameterSet(index_dict)] = float(term.coeff) * avrg_tensor[avrg_index_tuple] * vibenedenoms_tensor[index_tuple]
    return results
