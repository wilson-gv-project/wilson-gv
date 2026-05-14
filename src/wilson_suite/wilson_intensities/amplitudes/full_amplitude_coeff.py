"""
Docstring for wilson_suite.wilson_intensities.amplitudes.full_amplitude_coeff

All functions are used in evaluation
"""

from typing import TYPE_CHECKING, Any

import wilson_suite.wilson_intensities.amplitudes.averaged_props as avrgprops
import wilson_suite.wilson_intensities.amplitudes.vibene_differences as vediff
from wilson_suite.wilson_utils.prop_trivname import prop_trivname
import numpy as np
from typing import List, Dict, Tuple

from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, EvaluationDataAndConfigs, FreqTermsCollection, PrecalculatedData
if TYPE_CHECKING:
    from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm

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
            'vibdiff_motifs': vediff.identify_unique_vibdiff_motifs(terms)
            }

def precalculate_unique_coeff_parts(need_to_precalc: dict, 
                                    data_and_configs: EvaluationDataAndConfigs):
    """
        data = {'avrg_tensors': {}}

    """
    avrg_tensors = {}
    for avrg_tensor in need_to_precalc['avrg_tensors']:
        avrg_tensors[avrg_tensor] = avrgprops.calculate_avrg_tensor(avrg_expression=avrg_tensor, 
                                                                        pulse_polarization_vector=data_and_configs.pulse_polarization_vector,
                                                                        props_data=data_and_configs.props_data,
                                                                        number_of_nmodes=data_and_configs.number_of_nmodes,
                                                                        nm_inds_choices=data_and_configs.nm_inds_choices)   
    vibenedenoms_tensors = {}
    for ve_denom in need_to_precalc['vibenedenoms_tensors']:
        vibenedenoms_tensors[ve_denom] = vediff.calculate_vibenedenom_tensor(vibenedenom_inds=ve_denom, 
                                                                             nc_sqrt_eigval=data_and_configs.nc_sqrt_eigval)
    
    return PrecalculatedData(vibdiff_cache=vediff.VibDiffCache(),
                             avrg_tensors=avrg_tensors, 
                             avrg_expr_tensor_mapping=need_to_precalc['avrg_expr_tensor_mapping'],
                             vibenedenoms_tensors=vibenedenoms_tensors)

# NOTE: Have a more thorough look at this together
# def evaluate_term_coeffs(term: 'VibPerturbedTerm', 
#                          relevant_indices: list[dict], 
#                          necessary_data: tuple[EvaluationDataAndConfigs, PrecalculatedData], 
#                          zero_tol: float = 1e-18) -> dict[ParameterSet, float]:
#     """
#     safety function to check relevant_indices?

#     necessary_data keys: avrg_expr_tensor_mapping, 
#                          avrg_tensors, 
#                          vibenedenoms_tensors, 
#                          props_data (will be given separatelly, in principle),
#                          vibdiff_cache,
#                          vibstates_data (given separatelly)
#     relevant_indices - smth like [{'a': 0, 'b': 0, 'c': 1}, {'a': 0, 'b': 0, 'c': 2}, {'a': 0, 'b': 1, 'c': 1}]
#     zero_tol - what is considered zero as value
#     """
#     # Evaluate the coefficient part of the term 'term' for all of the indices in 'relevant_index_tuples'

#     # Return data or associate with term instance question not settled yet
#     results = {}
#     data_and_configs, precalculated_data = necessary_data
    
#     # AVRG and NON_AVRG
#     avrg_expr = avrgprops.PropsCollection(props=term.props).get_averaged_props().sort()
#     non_avrg_expr = avrgprops.PropsCollection(props=term.props).get_non_averaged_props()
#     # freq terms and freq terms with pert_wf_diff
#     freqterms = FreqTermsCollection(freqterms=term.freqterms)
#     extra_freqterms = freqterms.get_pert_wf_diff()

#     idx_summ, idx_nonsumm = term.tellNonSummSummIndices()
#     term_idx_all = sorted(idx_summ + idx_nonsumm)

#     for index_dict in relevant_indices:

#         if not all([index in index_dict for index in term_idx_all]):
#             # index_dict {'a': 0, 'b': 0}
#             to_summ_over = [index for index in term_idx_all if index not in index_dict]
#             # to_summ ['c']
#             n_modes = data_and_configs.number_of_nmodes
#             nm_idxs = list(range(n_modes))
            
#             # Create new index dictionaries for all combinations
#             inds_w_summ_over = []
#             import itertools
            
#             # Generate all possible combinations for missing indices
#             value_combinations = itertools.product(nm_idxs, repeat=len(to_summ_over))

#             for values in value_combinations:
#                 new_dict = index_dict.copy()  # Create a copy of original dict
#                 for idx, summ_index in enumerate(to_summ_over):
#                     new_dict[summ_index] = values[idx]
#                 inds_w_summ_over.append(new_dict)

#             computed = evaluate_term_coeffs(term=term, 
#                                     relevant_indices=inds_w_summ_over, 
#                                     necessary_data=necessary_data)
#             result = sum(list(computed.values()))
#             results[ParameterSet(index_dict)] = result
#             continue
        
#         NON_AVRG = eval_non_avrg_per_indexdict(non_avrg_expr, index_dict, data_and_configs, zero_tol)
#         AVRG = eval_avrg_per_indexdict(avrg_expr, index_dict, precalculated_data, zero_tol)
#         VIBDIFF_TERMS = eval_vibdiff_pert_wf_diff(extra_freqterms, index_dict, precalculated_data, data_and_configs)
#         VIBENE_DENOM = eval_vibenedenom(freqterms, index_dict, precalculated_data)

#         product_all = NON_AVRG*AVRG*VIBDIFF_TERMS*VIBENE_DENOM

#         # should be a single float result always?
#         results[ParameterSet(index_dict)] = float(term.coeff) * float(product_all)
#     return results


def evaluate_term_coeffs(term: 'VibPerturbedTerm', 
                         relevant_indices: List[Dict], 
                         necessary_data: Tuple['EvaluationDataAndConfigs', 'PrecalculatedData'], 
                         zero_tol: float = 1e-18) -> Dict['ParameterSet', float]:
    """
    Evaluate the coefficient part of the term 'term' for all of the indices in 'relevant_indices',
    handling hierarchical summation over indices.

    example_relevant_indices = [
        {'e': 0},  # Sum over 'a', 'b', 'c', 'd'
        {'e': 1},  # Sum over 'a', 'b', 'c', 'd'
        {'e': 2},  # Sum over 'a', 'b', 'c', 'd'
    ]

    Parameters:
        term: VibPerturbedTerm
            The term to evaluate, containing properties and frequency terms.
        relevant_indices: List[Dict]
            List of dictionaries specifying the relevant indices for evaluation.
        necessary_data: Tuple[EvaluationDataAndConfigs, PrecalculatedData]
            Tuple containing data and configurations required for evaluation.
        zero_tol: float
            Tolerance for considering a value as zero.
    Returns:
        Dict[ParameterSet, float]: A dictionary mapping ParameterSet to computed coefficients.
    """
    results = {}
    data_and_configs, precalculated_data = necessary_data
    
    # extract AVRG and NON_AVRG expressions
    avrg_expr = avrgprops.PropsCollection(props=term.props).get_averaged_props().sort()
    non_avrg_expr = avrgprops.PropsCollection(props=term.props).get_non_averaged_props()
    
    # extract frequency terms and their differences
    freqterms = FreqTermsCollection(freqterms=term.freqterms)
    extra_freqterms = freqterms.get_pert_wf_diff()
    
    # Get all indices
    idx_summ, idx_nonsumm = term.tellNonSummSummIndices()
    term_idx_all = sorted(idx_summ + idx_nonsumm)
    
    def hierarchical_sum(index_dict: Dict, remaining_indices: List[str], dict_of_sum: dict) -> float:
        """
        Perform hierarchical summation over the remaining indices.
        Parameters:
            index_dict: Dict
                The current index dictionary with some indices fixed.
            remaining_indices: List[str]
                The list of indices that still need to be summed over.
        Returns:
            float: The result of the summation for the given index dictionary.
            
            dict_of_sum: top level: 
                    {ParameterSet(index_dict): {}}
        """
        # Base case: no remaining indices to sum over
        if not remaining_indices:
            # print('dict_of_sum', dict_of_sum)
            value, contribs = evaluate_single_index_dict(term, index_dict, avrg_expr, non_avrg_expr, extra_freqterms, freqterms, data_and_configs, precalculated_data, zero_tol)
            dict_of_sum[ParameterSet(index_dict)] = contribs
            # returns coef and dict with param contribs

            return value, dict_of_sum
        
        # Get the next index to sum over
        current_index = remaining_indices[0]
        remaining = remaining_indices[1:]
        
        # Perform summation over all possible values for the current index
        n_modes = data_and_configs.number_of_nmodes
        total_sum = 0.0

        for value in range(n_modes):
            # Update the index dictionary with the current value
            new_index_dict = index_dict.copy()
            new_index_dict[current_index] = value
            
            parent_key = ParameterSet(index_dict)
            child_key = ParameterSet(new_index_dict)
            # if parent_key not in dict_of_sum:
            #     dict_of_sum[parent_key] = {}
            # dict_of_sum[parent_key][child_key] = {}
            
            # Recursively compute the sum for the remaining indices
            recurse_res = hierarchical_sum(new_index_dict, remaining, dict_of_sum[ParameterSet(index_dict)])
            total_sum += recurse_res[0]
        
        return total_sum, dict_of_sum
    
    # Iterate over the relevant indices -- 
    for index_dict in relevant_indices:
        # Identify missing indices
        missing_indices = [index for index in term_idx_all if index not in index_dict]
        
        dict_of_sum = {ParameterSet(index_dict): {}}
        
        # Perform hierarchical summation for the current index_dict
        result = hierarchical_sum(index_dict, missing_indices, dict_of_sum)
        results[ParameterSet(index_dict)] = result
    
    return results


def evaluate_single_index_dict(term: 'VibPerturbedTerm', 
                               index_dict: Dict, 
                               avrg_expr, 
                               non_avrg_expr, 
                               extra_freqterms, 
                               freqterms, 
                               data_and_configs, 
                               precalculated_data, 
                               zero_tol: float) -> tuple[float, dict]:
    """
    Evaluate the term for a single index dictionary.
    Parameters:
        (same as evaluate_term_coeffs)
    
    Returns:
        float: The computed coefficient for the given index dictionary.
    """
    # Evaluate NON_AVRG
    NON_AVRG = eval_non_avrg_per_indexdict(non_avrg_expr, index_dict, data_and_configs, zero_tol)
    if NON_AVRG == 0.0:
        # print('\nNON_AVRG zero - ', non_avrg_expr, index_dict, '\n\n')
        AVRG = eval_avrg_per_indexdict(avrg_expr, index_dict, precalculated_data, zero_tol)
        return 0.0, {'NON_AVRG': NON_AVRG, 'AVRG': AVRG}
    # Evaluate AVRG
    AVRG = eval_avrg_per_indexdict(avrg_expr, index_dict, precalculated_data, zero_tol)
    if AVRG == 0.0:
        # print('---- avrg_expr, index_dict', avrg_expr, index_dict)
        # print('\nAVRG zero - ', avrg_expr, index_dict, '\n\n')
        return 0.0, {'AVRG': AVRG}
    # Evaluate VIBDIFF_TERMS
    VIBDIFF_TERMS = eval_vibdiff_pert_wf_diff(extra_freqterms, index_dict, precalculated_data, data_and_configs)
    if VIBDIFF_TERMS == 0.0:
        # print('\nVIBDIFF_TERMS zero - ', extra_freqterms, index_dict, '\n\n')
        return 0.0, {'VIBDIFF_TERMS': VIBDIFF_TERMS}
    # Evaluate VIBENE_DENOM
    VIBENE_DENOM = eval_vibenedenom(freqterms, index_dict, precalculated_data)
    if VIBENE_DENOM == 0.0:
        # print('\nVIBENE_DENOM zero - ', freqterms, index_dict, '\n\n')
        return 0.0, {'VIBENE_DENOM': VIBENE_DENOM}
    # Compute the product
    product_all = NON_AVRG * AVRG * VIBDIFF_TERMS * VIBENE_DENOM

    # print('\nindex_dict', index_dict)
    # print('NON_AVRG', NON_AVRG)
    # print('AVRG', AVRG)
    # print('VIBDIFF_TERMS', VIBDIFF_TERMS)
    # print('VIBENE_DENOM', VIBENE_DENOM, '\n')

    dict_contribs = {'NON_AVRG': NON_AVRG, 'AVRG': AVRG, 'VIBDIFF_TERMS': VIBDIFF_TERMS, 'VIBENE_DENOM': VIBENE_DENOM}

    return float(term.coeff) * float(product_all), dict_contribs

# TODO: error handling for missing or invalid data for all functions below
def eval_non_avrg_per_indexdict(non_avrg_expr: avrgprops.PropsCollection, 
                                index_dict: dict, 
                                data_and_configs: EvaluationDataAndConfigs, 
                                zero_tol: float = 1e-18):
    """
    non_avrg_expr - extracted part of VibPerturbed term 

    order of indices generally: a,b,c,... 
    E.g. in CFF tensor index tuple is (a,b,c)
    """
    product_all = 1.
    
    for non_avrg_prop in non_avrg_expr:
        # accessing values for non-averaged properties from data
        na_prop_inds = tuple([index_dict[i] for i in non_avrg_prop.inds])
        triv_name = prop_trivname(ord_el=len(non_avrg_prop.ops), ord_geo=non_avrg_prop.dord)

        NON_AVRG = data_and_configs.props_data.get(triv_name).vals[na_prop_inds]

        if np.isclose(NON_AVRG, zero_tol):
            return 0.
        else:
            product_all *= NON_AVRG
    
    return product_all

def eval_avrg_per_indexdict(avrg_expr: avrgprops.PropsCollection, 
                            index_dict: dict, 
                            precalculated_data: PrecalculatedData,
                            zero_tol: float = 1e-18):
    avrg_tensor_expr = precalculated_data.avrg_expr_tensor_mapping[avrg_expr]
    
    avrg_tensor = precalculated_data.avrg_tensors[avrg_tensor_expr]
    
    avrg_index_tuple = avrgprops.get_ind_tuple_from_base(expr=avrg_expr, 
                                                         base_expr=avrg_tensor_expr, 
                                                         index_dict=index_dict)
    if np.isclose(avrg_tensor[avrg_index_tuple], zero_tol, atol=1e-20): # FIXME: against abs()
        return 0.
    return avrg_tensor[avrg_index_tuple]

def eval_vibdiff_pert_wf_diff(extra_freqterms: FreqTermsCollection,
                              index_dict: dict,
                              precalculated_data: PrecalculatedData,
                              data_and_configs: EvaluationDataAndConfigs):
    product_all = 1.

    for vibdiff in extra_freqterms:
        vib_diff_w_value = vediff.VibDiff.from_symbolic(vibdiff, index_dict, 
                                                        data_and_configs.vibstates_data)
        vib_diff_w_value.cache_it(vibdiff_cache=precalculated_data.vibdiff_cache)

        product_all *= 1./ vib_diff_w_value.energy_difference(au=True)
    
    return product_all

def eval_vibenedenom(freqterms: FreqTermsCollection,
                     index_dict: dict,
                     precalculated_data: PrecalculatedData):
    
    vibenedenoms_tensor = precalculated_data.vibenedenoms_tensors[freqterms.get_num_indices_vibenedenom()]

    vibeneden_index_tuple = tuple([index_dict[i] for i in freqterms.get_num_indices_vibenedenom()])

    return vibenedenoms_tensor[vibeneden_index_tuple]