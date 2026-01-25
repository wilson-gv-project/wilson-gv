"""
VIB DIFFERENCES in VibPerturbedTerm
"""
import numpy as np
import itertools
from wilson_suite.wilson_derive.abstractions import VibDiffTerm
from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_main.abstractions import VibState
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, VibStatesData

def identify_unique_vibdiff_motifs(list_of_terms: list['VibPerturbedTerm']):
    all_vibdiffs = []

    for term in list_of_terms:
        for res in term.res:
            all_vibdiffs.append(sorted([len(set(res.diff.sl.q)), len(set(res.diff.sr.q))]))

        for frt in term.freqterms:
            all_vibdiffs.append(sorted([len(set(frt.sl.q)), len(set(frt.sr.q))]))

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
                      parameters: 'ParameterSet',
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

def calculate_vibenedenom_tensor(vibenedenom_inds: tuple, 
                                 vibstates_data: 'VibStatesData'):
    """
    should be using harmonic uncorrected vib ene levels!!!
    """
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    
    vector = np.zeros((vibstates_data.number_of_nmodes,))
    for i in vibstates_data.harmonic_osc_states_labels:
        vector[i] = convNu2Ene(vibstates_data.get_harmonic_osc_states()[i])
    
    # 'i,j,k->ijk'
    letters = ['i', 'j', 'k', 'l', 'n', 'n', 'o', 'p']
    einsum_str = ','.join(letters[:len(vibenedenom_inds)])+'->'+''.join(letters[:len(vibenedenom_inds)])

    denominator = np.einsum(einsum_str, *(vector,) * len(vibenedenom_inds))

    # to not have RuntimeWarning: divide by zero encountered in divide
    return np.divide(1., denominator, where=denominator != 0)


def calculate_vibenedenoms(unique_vibenedenoms: list[set], 
                           vibstates_data: 'VibStatesData'):
    """
    can be done as vector multiplication
    """
    results = {}
    
    for u_vediff in unique_vibenedenoms:
        results[tuple(sorted(u_vediff))] = calculate_vibenedenom_tensor(u_vediff, vibstates_data)
    
    return results

def identify_vibenedenoms(terms: list['VibPerturbedTerm']):
    """
    """
    from wilson_suite.wilson_intensities.amplitudes.term_parts import FreqTermsCollection
    return set([FreqTermsCollection(freqterms=t.freqterms).get_num_indices_vibenedenom() for t in terms])


def make_vibdiff_key(vibdiff_term: VibDiffTerm, index_dict: dict) -> tuple[str, str]:
    """
    Non-sorted key for VibDiffBank_cache

    returns keys for vibdiff bank for vib states expression and choice of indices
    """
    left_state_symb = vibdiff_term.sl.q
    right_state_symb = vibdiff_term.sr.q


    left_state_label = ','.join(sorted([str(index_dict[i]) for i in left_state_symb]))
    right_state_label = ','.join(sorted([str(index_dict[i]) for i in right_state_symb]))
    
    if left_state_label == '':
        left_state_label = 'zero'
    if right_state_label == '':
        right_state_label = 'zero'
    
    return (left_state_label, right_state_label)


@dataclass
class VibDiff:
    """
    Represents difference between two vibrational states.
    Numerical representation that holds values, as opposed to VibDiffTerm which is symbolic.
    Handles special case of zero states (ground state) in comparisons.
    """
    left: VibState
    right: VibState
    
    def is_zero_state(self, state: VibState) -> bool:
        """
        Check if state is a zero (ground) state.

        #TODO more criteria?
        """
        return state.state_label == 'zero'
    
    def normalized(self) -> 'VibDiff':
        """
        Return normalized form where left <= right.
        Zero states are considered smaller than any other state.
        """
        left_is_zero = self.is_zero_state(self.left)
        right_is_zero = self.is_zero_state(self.right)
        
        # If both are zero states or neither is zero, use standard comparison
        if left_is_zero == right_is_zero:
            if self.left < self.right:
                return VibDiff(self.left, self.right)
            return VibDiff(self.right, self.left)
            
        # Zero state should always be on the left
        if left_is_zero:
            return VibDiff(self.left, self.right)
        return VibDiff(self.right, self.left)
    
    def energy_difference(self, *, au=False) -> float:
        """
        Calculate energy difference between states.
        For zero states, energy is considered to be 0.0
        """
        left_energy = 0.0 if self.is_zero_state(self.left) else self.left.energy
        right_energy = 0.0 if self.is_zero_state(self.right) else self.right.energy
        if au:
            return convNu2Ene(left_energy - right_energy)
        else:
            return left_energy - right_energy

    @classmethod
    def from_symbolic(cls, 
                    vibdiff_term_symb: VibDiffTerm,
                    index_dict: dict,
                    vibstates_data: 'VibStatesData') -> 'VibDiff':
        """Construct VibDiff from symbolic representation."""
        # Get state labels from symbolic term
        left_label, right_label = make_vibdiff_key(vibdiff_term_symb, index_dict)
        # Look up states in vibstates_data
        left_state = (
            VibState(harm_quanta_coeffs={}, state_label='zero', energy=0.0)
            if left_label == 'zero'
            else vibstates_data.get_state_by_label(left_label)
        )
        
        right_state = (
            VibState(harm_quanta_coeffs={}, state_label='zero', energy=0.0)
            if right_label == 'zero'
            else vibstates_data.get_state_by_label(right_label)
        )

        return cls(left=left_state, right=right_state)

    def cache_it(self, vibdiff_cache: 'VibDiffCache'):
        """Ensure this VibDiff's energy is cached."""
        if vibdiff_cache.get(self) is None:
            energy = self.energy_difference()
            vibdiff_cache.add(self, energy)

@dataclass
class VibDiffCache:
    """
    bank keys
    ('0', '2')   -> sorted version is the key --- ('0', '2')
    ('0,2', '2') -> sorted version is the key --- ('2', '0,2')
    ('1,2', '3') -> sorted version is the key --- ('3', '1,2')
    ('1,2,4', '3,1') -> sorted version is the key --- ('1,3', '1,2,4')
    ('4,1,2', '3,1') -> sorted version is the key --- ('1,3', '1,2,4')

    """
    def __init__(self):
        self._cache: dict[tuple[str, str], float] = {}
    
    def __repr__(self):
        return str(self._cache)
    
    def get(self, vib_diff: VibDiff) -> float | None:
        """Get cached energy difference"""
        key = (vib_diff.left.state_label, vib_diff.right.state_label)
        norm_diff = vib_diff.normalized()
        norm_key = (norm_diff.left.state_label, norm_diff.right.state_label)
        
        if key in self._cache:
            return self._cache[key]
        if norm_key in self._cache:
            return -self._cache[norm_key] if key != norm_key else self._cache[norm_key]
        return None
        
    def add(self, vib_diff: VibDiff, energy: float):
        """Cache energy difference"""
        norm_diff = vib_diff.normalized()
        self._cache[(norm_diff.left.state_label, norm_diff.right.state_label)] = energy


def normalize_state_key(state_str: str) -> tuple[int, ...]:
    """Convert state string to normalized form for comparison while preserving multiplicity
    
    Args:
        state_str: String representation of state like '5,7' or '5,5,7' or 'zero'
    
    Returns:
        Tuple of sorted integers representing the state, or empty tuple for 'zero'
        
    Examples:
        '5,7'    -> (5,7)
        '7,5'    -> (5,7)      # Same as above - normalized order
        '5,5,7'  -> (5,5,7)    # Preserves multiplicity
        'zero'   -> ()         # Empty tuple for ground state

    Claude Sonnet 3.5
    """
    if state_str == 'zero' or state_str == '':
        return tuple()
    return tuple(sorted(int(x) for x in state_str.split(',')))

def compare_vibdiff_states(left: str, right: str) -> int:
    """Compare two vibrational states and return ordering value
    
    Args:
        left: First state string ('5,7' or 'zero' etc)
        right: Second state string
        
    Returns:
        -1 if left < right
         0 if left == right 
         1 if left > right
    """
    left_tuple = normalize_state_key(left)
    right_tuple = normalize_state_key(right)
    
    # Compare by length first
    if len(left_tuple) != len(right_tuple):
        return -1 if len(left_tuple) < len(right_tuple) else 1
        
    # Empty tuples (zero states) are equal
    if not left_tuple and not right_tuple:
        return 0
        
    # Compare by minimum element
    if left_tuple and right_tuple:
        left_min = min(left_tuple)
        right_min = min(right_tuple)
        if left_min != right_min:
            return -1 if left_min < right_min else 1
    
    # Compare full tuples lexicographically
    if left_tuple < right_tuple:
        return -1
    elif left_tuple > right_tuple:
        return 1
    return 0

def is_vibdiff_sorted(left: str, right: str) -> bool:
    """
    Check if vibration difference key is in canonical order
        -1 if left < right  - no need to reorder
         0 if left == right - no need to reorder
         1 if left > right  - need to reorder
    """
    return compare_vibdiff_states(left, right) <= 0

def make_sorted_vibdiff_key(left: str, right: str) -> tuple[str, str]:
    """Create canonical sorted form of vibdiff key tuple"""
    if is_vibdiff_sorted(left, right):
        return (left, right)
    return (right, left)


def compute_vibdiff(diff_tuple_label: tuple[str, str], vibstates_data: 'VibStatesData'):
    """
    
    """
    left, right = diff_tuple_label
    return vibstates_data.allenergies_map[left] - vibstates_data.allenergies_map[right]