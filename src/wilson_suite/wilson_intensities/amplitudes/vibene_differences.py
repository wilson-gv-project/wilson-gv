"""
VIB DIFFERENCES in VibPerturbedTerm
"""
import numpy as np
import itertools
from wilson_suite.wilson_derive.abstractions import VibPerturbedTerm
from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, VibStatesData
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene


def identify_unique_vibdiff_motifs(list_of_terms: list['VibPerturbedTerm']):
    all_vibdiffs = []

    for term in list_of_terms:
        for res in term.res:
            all_vibdiffs.append(sorted([len(set(res.diff.sl.q)), len(set(res.diff.sr.q))]))
            # all_vibdiffs.append(tuple([tuple(res.diff.sl.q), tuple(res.diff.sr.q)]))            

        for frt in term.freqterms:
            all_vibdiffs.append(sorted([len(set(frt.sl.q)), len(set(frt.sr.q))]))
            # all_vibdiffs.append(tuple([tuple(frt.sl.q), tuple(frt.sr.q)]))

    return set(tuple(vd) for vd in all_vibdiffs)


class VibDiffBank:
    def __init__(self, indices: tuple|list, max_quanta: int,
                 state_value_func: callable,
                 dense_threshold=1e5, mode: str = None):
        """
        indices: list of numeric indices
        max_quanta: max number of quanta per state
        state_value_func: function(state_tuple) -> float
        dense_threshold: max number of differences to store in dense mode

        Attributes:
            - state_values
            - bank
            - state_to_idx
        """
        self.indices = indices
        self.max_quanta = max_quanta
        self.state_value_func = state_value_func

        self.mode = mode

        # generate all states
        self.states = self._generate_all_states()
        self.states.append('zero')

        self.state_to_idx = {s: i for i, s in enumerate(self.states)}

        if mode is None:
            # decide on dense vs on-demand
            num_diffs = len(self.states) ** 2

            if num_diffs <= dense_threshold:
                self.mode = "dense"
            else:
                self.mode = "ondemand"

        if self.mode == "ondemand":
            self._build_energy_bank()
        elif self.mode == "dense":
            self._build_dense_bank()

    def _generate_all_states(self):
        states = []
        for r in range(1, self.max_quanta + 1):
            states.extend(itertools.product(self.indices, repeat=r))
        return [tuple(s) for s in states]

    def _build_dense_bank(self):
        values = []
        for s in self.states:
            if s == 'zero':
                values.append(0.0)
            else:
                s = sorted(s)
                values.append(float(self.state_value_func(s)))
        values = np.array(values, dtype=float)
        self.bank = values[:, None] - values[None, :]

    def _build_energy_bank(self):
        self.state_values = {s: self.state_value_func(s) for s in self.states if s != 'zero'}
        self.state_values['zero'] = 0.

    def get_vibdiff_number(self, ind_diff_str: str, ind_tuple: tuple):
        """
        ind_diff_str: string like 'a+b,a' or 'zero,a'
        ind_tuple: tuple of numeric indices, e.g. (1,2,3,4,5)
        """
        letter_to_pos = {chr(ord('a') + i): i for i in range(len(ind_tuple))}

        def parse_state_ref(ref: str):
            """Parsing label of a vib state with symbolic indices: a+b; b+c; a; c+a"""
            return tuple(sorted(ind_tuple[letter_to_pos[ch.strip()]] for ch in ref.split('+')))

        left_str, right_str = ind_diff_str.split(',')

        if left_str!='zero':
            s1 = parse_state_ref(left_str)
        else:
            s1 = 'zero'
        if right_str!='zero':
            s2 = parse_state_ref(right_str)
        else:
            s2 = 'zero'

        if self.mode == "dense":
            return self.bank[self.state_to_idx[s1], self.state_to_idx[s2]]
        else:  # ondemand
            return self.state_values[s1] - self.state_values[s2]


def get_vibdiff_motif(vibdiff_symb: tuple[tuple],
                      parameters: ParameterSet,
                      allstates_map: dict, unit='Eh') -> float:
    """
    left, right - vibrational states labels for left and right state
    eval_mode - 'full-stored' or 'on-the-fly'
    """
    leftToNum = [parameters[alpha_ind] for alpha_ind in vibdiff_symb[0]]
    rightToNum = [parameters[alpha_ind] for alpha_ind in vibdiff_symb[1]]

    left_num = '+'.join(sorted(leftToNum))
    right_num = '+'.join(sorted(rightToNum))

    if unit=='Eh':
        return convNu2Ene(allstates_map[left_num] - allstates_map[right_num])
    elif unit=='cm-1':
        return allstates_map[left_num] - allstates_map[right_num]
    else:
        raise NotImplementedError('This unit of energy is not supported')

def calculate_vibenedenom_tensor(vibenedenom_inds: set, 
                                 vibstates_data: VibStatesData):
    """
    should be using harmonic uncorrected vib ene levels!!!
    """
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    
    vector = convNu2Ene(np.array(list(vibstates_data.get_harmonic_osc_states().values())))
    
    # 'i,j,k->ijk'
    letters = ['i', 'j', 'k', 'l', 'n', 'n', 'o', 'p']
    einsum_str = ','.join(letters[:len(vibenedenom_inds)])+'->'+''.join(letters[:len(vibenedenom_inds)])

    return 1. / np.einsum(einsum_str, *(vector,) * len(vibenedenom_inds))


def calculate_vibenedenoms(unique_vibenedenoms: list[set], 
                           vibstates_data: VibStatesData):
    """
    can be done as vector multiplication
    """
    results = {}
    
    for u_vediff in unique_vibenedenoms:
        results[tuple(sorted(u_vediff))] = calculate_vibenedenom_tensor(u_vediff, vibstates_data)
    
    return results

from wilson_suite.wilson_intensities.amplitudes.term_parts import FreqTermsCollection
def identify_vibenedenoms(terms: list['VibPerturbedTerm']):
    """
    """
    return set([FreqTermsCollection(freqterms=t.freqterms).get_num_indices_vibenedenom() for t in terms])