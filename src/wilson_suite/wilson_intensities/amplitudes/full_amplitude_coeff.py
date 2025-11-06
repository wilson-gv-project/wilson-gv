from typing import TYPE_CHECKING, Any

from wilson_suite.wilson_intensities.amplitudes.resonances import identify_unique_resmotifs
import wilson_suite.wilson_intensities.amplitudes.averaged_props as avrgprops
import wilson_suite.wilson_intensities.amplitudes.vibene_differences as vediff
from wilson_suite.wilson_utils.prop_trivname import prop_trivname
import numpy as np
import copy

from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, ResonanceMotif, EvaluationDataAndConfigs
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
                                      motif_res_loc, data_and_configs: EvaluationDataAndConfigs):

    # Suggestion sketch for overall steps

    # Identify all unique parts that can be precalculated for a sensible partitioning of the term parts
    # One function here
    need_to_precalc = identify_precalc_unique_coeff_parts(terms)

    # Perform the precalculation and keep in a structure to yet be decided
    # One function here
    precalculated_data = precalculate_unique_coeff_parts(need_to_precalc, data_and_configs)
    
    # Current implementation target for the previous two fns: Make them work in the general care and
    # don't worry about optimization yet

    results = {}
    # Calculate the coefficients for the terms by combining the precalculated parts
    # One function here
    for term in terms:
        this_res_motif = ResonanceMotif(term.res)
        rel_inds = [...]
        evaluate_term_coeffs(term, relevant_indices=rel_inds, necessary_data=precalculated_data)

    # Would return a structure {term: {resonance index tuple: term coeff, ...}, ...}

    return results

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

def precalculate_unique_coeff_parts(need_to_precalc: dict, data_and_configs: EvaluationDataAndConfigs):
    """
        data = {'avrg_tensors': {}}

    """

    data = copy.deepcopy(data_and_configs)
    data.avrg_tensors = {}
    data.avrg_expr_tensor_mapping = need_to_precalc['avrg_expr_tensor_mapping']
    data.vibenedenoms_tensors = {}

    for avrg_tensor in need_to_precalc['avrg_tensors']:
        data.avrg_tensors[avrg_tensor] = avrgprops.calculate_avrg_tensor(avrg_expression=avrg_tensor, 
                                                                        polarization=data_and_configs.polarization,
                                                                        #pulse_polarization_vector=data_and_configs.pulse_polarization_vector,
                                                                        number_of_nmodes=data_and_configs.number_of_nmodes,
                                                                        props_data=data_and_configs.props_data)   
    for ve_denom in need_to_precalc['vibenedenoms_tensors']:
        data.vibenedenoms_tensors[ve_denom] = vediff.calculate_vibenedenom_tensor(vibenedenom_inds=ve_denom, 
                                                                                    vibstates_data=data_and_configs.vibstates_data)
    data.vibdiff_cache = vediff.VibDiffCache()
    return data


def evaluate_term_coeffs(term: 'VibPerturbedTerm', 
                         relevant_indices: list[dict], 
                         necessary_data: EvaluationDataAndConfigs, 
                         zero_tol: float = 1e-18) -> dict[ParameterSet, float]:
    """
    safety function to check relevant_indices?

    necessary_data keys: avrg_expr_tensor_mapping, 
                         avrg_tensors, 
                         vibenedenoms_tensors, 
                         props_data (will be given separatelly, in principle),
                         vibdiff_cache,
                         vibstates_data (given separatelly)
    relevant_indices - smth like [{'a': 0, 'b': 0, 'c': 1}, {'a': 0, 'b': 0, 'c': 2}, {'a': 0, 'b': 1, 'c': 1}]
    zero_tol - what is considered zero as value
    """
    # Evaluate the coefficient part of the term 'term' for all of the indices in 'relevant_index_tuples'

    # Return data or associate with term instance question not settled yet
    results = {}

    # AVRG and NON_AVRG
    avrg_expr = avrgprops.PropsCollection(props=term.props).get_averaged_props().sort()
    non_avrg_expr = avrgprops.PropsCollection(props=term.props).get_non_averaged_props()

    avrg_tensor_expr = necessary_data.avrg_expr_tensor_mapping[avrg_expr]
    avrg_tensor = necessary_data.avrg_tensors[avrg_tensor_expr]
    
    freqterms = vediff.FreqTermsCollection(freqterms=term.freqterms)
    extra_freqterms = freqterms.get_pert_wf_diff()

    vibenedenoms_tensor = necessary_data.vibenedenoms_tensors[freqterms.get_num_indices_vibenedenom()]

    idx_summ, idx_nonsumm = term.tellNonSummSummIndices()
    term_idx_all = sorted(idx_summ + idx_nonsumm)

    for index_dict in relevant_indices:

        if not all([index in index_dict for index in term_idx_all]):
            # index_dict {'a': 0, 'b': 0}
            to_summ_over = [index for index in term_idx_all if index not in index_dict]
            # to_summ ['c']
            n_modes = necessary_data.number_of_nmodes
            nm_idxs = list(range(n_modes))
            
            # Create new index dictionaries for all combinations
            inds_w_summ_over = []
            import itertools
            
            # Generate all possible combinations for missing indices
            value_combinations = itertools.product(nm_idxs, repeat=len(to_summ_over))

            for values in value_combinations:
                new_dict = index_dict.copy()  # Create a copy of original dict
                for idx, summ_index in enumerate(to_summ_over):
                    new_dict[summ_index] = values[idx]
                inds_w_summ_over.append(new_dict)

            computed = evaluate_term_coeffs(term=term, 
                                    relevant_indices=inds_w_summ_over, 
                                    necessary_data=necessary_data)
            result = sum(list(computed.values()))
            results[ParameterSet(index_dict)] = result
            continue
            # raise ValueError(f'index_dict - {index_dict} - is missing values for some indices')
        
        product = 1.
        
        for non_avrg_prop in non_avrg_expr:
            # accessing values for non-averaged properties from data
            na_prop_inds = tuple([index_dict[i] for i in non_avrg_prop.inds])
            triv_name = prop_trivname(ord_el=len(non_avrg_prop.ops), ord_geo=non_avrg_prop.dord)

            NON_AVRG = necessary_data.props_data.get(triv_name).vals[na_prop_inds]

            if np.isclose(NON_AVRG, zero_tol):
                results[ParameterSet(index_dict)] =  0.
                continue
            else:
                product *= NON_AVRG

        avrg_index_tuple = avrgprops.get_ind_tuple_from_base(expr=avrg_expr, base_expr=avrg_tensor_expr, index_dict=index_dict)
        AVRG = avrg_tensor[avrg_index_tuple]
        
        if np.isclose(AVRG, zero_tol):
            results[ParameterSet(index_dict)] =  0.
            continue
        else:
            product *= AVRG

        # VIBDIFF_TERMS is_pert_wf_diff
        for vibdiff in extra_freqterms:
            vib_diff_w_value = vediff.VibDiff.from_symbolic(vibdiff, index_dict, 
                                                            necessary_data.vibstates_data)
            vib_diff_w_value.cache_it(vibdiff_cache=necessary_data.vibdiff_cache)

            product *= 1./ vib_diff_w_value.energy_difference(au=True)

        # VIBENE_DENOM
        vibeneden_index_tuple = tuple([index_dict[i] for i in freqterms.get_num_indices_vibenedenom()])

        product *= vibenedenoms_tensor[vibeneden_index_tuple]

        # should be a single float result always?
        results[ParameterSet(index_dict)] = float(term.coeff) * float(product)
    return results
