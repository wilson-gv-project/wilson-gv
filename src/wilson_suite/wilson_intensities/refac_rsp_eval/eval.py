"""
===
purification of abstractions
===

rsp_evaluator


"""

from dataclasses import dataclass
from typing import TYPE_CHECKING
from collections.abc import Sequence

from wilson_suite.wilson_derive.abstractions import ResonanceCondition, HarmOscStateSymbolic, PolProp, VibDiffTerm

if TYPE_CHECKING:
    from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm

@dataclass
class PropsCollection:
    """
    Collects several PolProp class instances and enables some group operations:
        get_avegaded_props - extract averaged properties
        get_non_avegaded_props - extract non-averaged properties
        get_cart_axes
        get_mode_indices
        get_total_difforder
    """
    props: Sequence[PolProp]

    def __post_init__(self):
        self.props = tuple(self.props)

    def __iter__(self):
        yield from self.props

    def __hash__(self):
        # return hash(tuple([tuple(self.get_cart_axes()), self.get_total_difforder()]))
        return hash( (self.get_cart_axes(), self.get_mode_indices()) )
    
    def __eq__(self, other):
        """
        Now depends on comparison of PolProp instances.
        Now PolProp instances are considered equal if the have the same lists of operators (ops)
            (further, equality of QOperator instances) and same differentiation order (dord)
        """
        if isinstance(other, PropsCollection):
            return all([p in other.props for p in self.props])
        return False
    
    def get_averaged_props(self):
        return PropsCollection(props=[p for p in self.props if p.ops])
    def get_non_averaged_props(self):
        return PropsCollection(props=[p for p in self.props if not p.ops])
    
    def get_cart_axes(self):
        return tuple(op.o for p in self.props for op in p.ops)
    def get_mode_indices(self):
        groups = [p.inds if p.inds is not None else [] for p in self.props]
        return tuple(idx for p_inds in groups for idx in p_inds)
    
    # UNUSED
    def get_mode_indices_grouped(self):
        return [p.inds if p.inds is not None else [] for p in self.props]
    
    def get_mode_indices_group_template(self):
        return [len(p.inds) if p.inds is not None else [] for p in self.props]
    
    # UNUSED
    def get_total_difforder(self):
        return sum([p.dord for p in self.props])
    
    def _set_attr_for_all_props(self, attr, value):
        for prop in self.props:
            prop.__setattr__(attr, value)
    
    def identify_avrg_motif(self):
        """
        Averaged properties motif/ID - inds will be set to None.
        Indices will be added later, when several terms with with avrg motifs are concidered
        """
        averaged = copy.deepcopy(self.get_averaged_props())
        if averaged.props:
            averaged._set_attr_for_all_props('inds', None)
            return averaged
    
    def sort(self):
        """
        non-averaged props will be in the end of the tuple
        """
        self.props = sorted(self.props, key=lambda j: j.ops[0].o if j.ops else float('inf'))
        return self

    def __repr__(self):
        inds_all = [len(p.inds) if p.inds else 0 for p in self.props]
        full_string = [f'{prop_trivname(ord_geo=inds_all[i], ord_el=len(p.ops))}{p.inds}{[i.o for i in p.ops]}_d{p.dord}' for i, p in enumerate(self.props)]
        return ' * '.join(full_string)

@dataclass
class FreqTermsCollection:
    freqterms: list[VibDiffTerm]
    
    def __post_init__(self):
        self.freqterms = tuple(self.freqterms)

    def __iter__(self):
        yield from self.freqterms

    def __hash__(self):
        return hash(self.freqterms)
    
    def __eq__(self, other):
        """
        Now depends on comparison of PolProp instances.
        Now PolProp instances are considered equal if the have the same lists of operators (ops)
            (further, equality of QOperator instances) and same differentiation order (dord)
        """
        if isinstance(other, FreqTermsCollection):
            return all([ft in other.freqterms for ft in self.freqterms])
        return False

    def get_vibenedenom(self):
        return FreqTermsCollection(freqterms=[ft for ft in self.freqterms if not ft.is_pert_wf_diff])
    
    def get_pert_wf_diff(self):
        return FreqTermsCollection(freqterms=[ft for ft in self.freqterms if ft.is_pert_wf_diff])
    
    def get_num_indices_vibenedenom(self):
        """
        """
        # these vibdiffterms have only sl, sr is zero
        return tuple(sorted(set(i for vd in self.get_vibenedenom() for i in vd.sl.q)))


@dataclass
class ResonanceMotif:
    """
    Collects ResonanceCondition instances into a "resonance motif"
        get_vibdiffs
        get_freq_axes
    """
    resonance_conditions: list[ResonanceCondition]
    
    def __iter__(self):
        yield from self.resonance_conditions

    def __eq__(self, other):
        if isinstance(other, ResonanceMotif):
            return self._tuplify() == other._tuplify()
        return False
    def __hash__(self):
        return hash(self._tuplify())
    
    def _tuplify(self):
        conditions = []
        for cond in self.resonance_conditions:
            new_pf = tuple(cond.pf)
            new_diff = tuple([tuple(cond.diff.sl.q), tuple(cond.diff.sr.q)])

            conditions.append(tuple([new_diff, new_pf]))
        return tuple(conditions)
    
    def to_str(self):
        """
        EVV / paper1 spectific here
        """
        strings = []
        for cond in self.resonance_conditions:
            if 'B' in cond.pf[0]:
                state = []
                for i in [cond.diff.sl.q, cond.diff.sr.q]:
                    if len(i)!=0:
                        state.append('+'.join(i))
                    else:
                        state.append('.')
                strings.append(f'{','.join(state)}')
        return ' x '.join(strings)

    # UNUSED?
    @classmethod
    def from_tuples(cls, tupleOfTuples):
        """
        motif1 = (((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))
        motif2 = (((('a', 'b'), ('a',)), ('A',)),)
        motif3 = (((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))
        motif4 = (((('',), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))
        """
        r_conditions = []
        for rc_tuple in tupleOfTuples:
            print('rc tuple', rc_tuple)
            rc = ResonanceCondition(diff=VibDiffTerm(sl=HarmOscStateSymbolic(q=rc_tuple[0][0]),
                                                     sr=HarmOscStateSymbolic(q=rc_tuple[0][1])), pf=rc_tuple[1])
            r_conditions.append(rc)
        return cls(r_conditions)

    # UNUSED
    @classmethod
    def from_dicts(cls, res_conds_listdict: list[dict]):
        """
        motif3 = (
                  ((left-('',), right-('a',)), pert_freqs-('B',)), 
                  ((left-('',), right-('a',)), pert_freqs-('A', '-B')))

        res_conds_dict = [{'left': tuple, 'right': tuple, 'pert_freqs': tuple},
                          {'left': tuple, 'right': tuple, 'pert_freqs': tuple}]
        """
        r_conditions = []

        for rc_dict in res_conds_listdict:
            rc = ResonanceCondition(diff=VibDiffTerm(sl=HarmOscStateSymbolic(q=rc_dict['left']),
                                                     sr=HarmOscStateSymbolic(q=rc_dict['right'])), 
                                                     pf=rc_dict['pert_freqs'])
            r_conditions.append(rc)
        return cls(r_conditions)

    def __repr__(self):
        return f'{self.resonance_conditions}'
    
    def __len__(self):
        """
        Returns the number of elements in the container.
        """
        return len(self.resonance_conditions)
    
    # UNUSED
    @property
    def resonance_location_class(self, total_num_axes):
        return total_num_axes - len(self.resonance_conditions)
    
    # UNUSED
    def get_vibdiffs(self):
        return {i: cond.diff for i, cond in enumerate(self.resonance_conditions)}
    # UNUSED
    def get_freq_axes(self):
        return {i: tuple(cond.pf) for i, cond in enumerate(self.resonance_conditions)}
    
    def get_max_different_freq_axes(self):
       return set([i.strip('-') for cond in self.resonance_conditions for i in cond.pf])
    
    def get_nm_indices(self):
        return set([label for cond in self.resonance_conditions for i in cond.diff for label in i.q])




@dataclass
class RspEvalTerm:

    term_id: int | str          # provenance back to the symbolic term

    rot_avrg_props: PropsCollection | None
    rot_invr_props: PropsCollection | None

    res_conds: ResonanceMotif | None

    vibdiffs: FreqTermsCollection | None
    ene_prefac: FreqTermsCollection | None
    num_coeff: FreqTermsCollection | None

    summation_indices: tuple[str, ...] | None  # from tellNonSummSummIndices
    non_summation_indices: tuple[str, ...] | None
    all_indices: tuple[str, ...] | None       # sorted union



def parse_vibpert_term(term: 'VibPerturbedTerm') -> RspEvalTerm:
    
    # extract AVRG and NON_AVRG expressions
    avrg_expr = PropsCollection(props=term.props).get_averaged_props().sort()
    non_avrg_expr = PropsCollection(props=term.props).get_non_averaged_props()

    res_conds = ResonanceMotif(term.res)
    
    # extract frequency terms and their differences
    freqterms_all = FreqTermsCollection(freqterms=term.freqterms)
    extra_freqterms = freqterms_all.get_pert_wf_diff() # fixme

    ene_prefac = freqterms_all.get_vibenedenom() # fixme
    """
    precalculated_data.vibenedenoms_tensors[freqterms.get_num_indices_vibenedenom()]
    """

    num_coeff = float(term.coeff)

    idx_summ, idx_nonsumm = term.tellNonSummSummIndices()

    return RspEvalTerm(rot_avrg_props=avrg_expr,
                       rot_invr_props=non_avrg_expr,
                       res_conds=res_conds,
                       vibdiffs=extra_freqterms,
                       ene_prefac=ene_prefac,
                       num_coeff=num_coeff,
                       summation_indices=idx_summ,
                       non_summation_indices=idx_nonsumm)


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