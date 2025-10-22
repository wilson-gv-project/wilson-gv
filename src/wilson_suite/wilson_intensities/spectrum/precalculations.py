import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wilson_suite.wilson_derive.abstractions import VibPerturbedTerm

def precalc_unique_coeff_parts(data_to_precalc: dict):
    return

def precalc_avrg_tensors(needed_data: dict, 
                         avrg_terms_calc: tuple[list, float],
                         terms: list[VibPerturbedTerm],
                         avrg_motifs: tuple[tuple]) -> dict[tuple: np.ndarray]:
    """
    3 motifs:
    {(((0, 3), 1), ((1,), 1), ((2,), 1)), 
     (((1,), 1), ((2,), 1), ((0, 3), 2)), 
     (((0, 3), 1), ((1,), 1), ((2,), 2))}
    
    avrg tensor coding:
    ((1, 1), (2, 1), (1, 2)) - mu_Q, alpha_Q, mu_QQ
    ((1, 1), (2, 2), (1, 1)) - mu_Q, alpha_QQ, mu_Q
    ((1, 1), (2, 1), (1, 1)) - mu_Q, alpha_Q, mu_Q

    (0, 3), 1) - alpha_Q
    ((1,), 1)  - mu_Q


    requires:
        self.unique_avrg_tensors_tID; self.seq_tuples; self.unique_avrg_tensors_all_expr;
        self.terms[tID] so it's a dict;

    HOW TO SET UP FORMULA FOR ORIENATIONAL AVERAGING???? - avrg_terms_calc
    should be based on pulses (experiment info)
    """
    import string
    from ..utils.tools import combinations_with_permutations
    from ..utils.spectrum_utils import greek_list

    # Nnmodes: int, data: dict
    avrg_sum_terms, prefactorAvrg = avrg_terms_calc
    storage_tensors = {}

    # for tID in self.unique_avrg_tensors_tID:
    for motif in avrg_motifs:
        print('avrg_motifs', motif)
        # simple_prop_tuple = self.seq_tuples.vk[terms[tID]]
        number_of_indices = motif[1]
        shape = (needed_data['Nnmodes'],) * number_of_indices
        # logger.warning(f'self.unique_avrg_tensors_all_expr {self.unique_avrg_tensors_all_expr}')
        # logger.warning(f'shape precalc_avrg_tensors {shape}')
        avrg_tensor_to_fill = np.zeros(shape)
        indices_combs = combinations_with_permutations(range(needed_data['Nnmodes']), number_of_indices)

        for indices_choice in indices_combs:
            total = 0.

            names = list(string.ascii_lowercase)
            indices_names = names[:len(indices_choice)]
            ind_mapping = {var: val for var, val in zip(indices_names, indices_choice)}

            for cart_axes in avrg_sum_terms:
                
                greek_dict = {L: n for L, n in zip(greek_list[:len(cart_axes)], cart_axes)}

                product = 1.

                for prop_tuple in motif:
                    el_operators, differentiation_order = prop_tuple
                    prop_tuple_key = (len(el_operators), differentiation_order)
                    nm_inds = prop_tuple[0]
                    cart_inds = prop_tuple[0]
                    all_inds = (nm_inds, cart_inds)

                # for i, input_tuple in enumerate(terms[tID].avrg_props_expr):
                    # prop_key, idxs_key = get_data_keys(input_tuple, ind_mapping, greek_dict)
                    # retrieve data for preperty (prop_key) and idxs_key which is (tuple(mode inds), tuple(cart inds))
                    product *= needed_data[prop_tuple_key][all_inds]

                total += product

            if abs(total)<1e-28:
                total = 0.
            else:
                total *= prefactorAvrg
            avrg_tensor_to_fill[indices_choice] = total

        storage_tensors[motif] = avrg_tensor_to_fill

    return storage_tensors

def get_data_keys(input_tuple: tuple, variables: dict, greek_dict: dict) -> tuple[str, tuple]:
    """
    tuple_input = ((1, 1), ('B',), ('a',))

    input_tuples = [
        ((1, 1), ('B',), ('a',)),
        ((2, 1), ('A', 'D'), ('a',)),
        ((2, 2), ('A', 'D'), ('a', 'b'))
    ]
    in term:
    (('mu_Q', ('a',), ('B',)),
     ('alpha_Q', ('b',), ('A', 'D')),
     ('mu_Q', ('c',), ('G',)))

    prop_der_key is a trivial name string
    second_part contains normal mode indices
    third part contains cartesian axes Greek indices
    """

    prop_der_key, second_part, third_part = input_tuple

    second_part = tuple([variables[v] for v in second_part])
    third_part = tuple([greek_dict[L] for L in third_part])
    # combine third_part and second_part to make the second-level index
    idxs_key = tuple(second_part) + tuple(third_part)

    return prop_der_key, idxs_key
