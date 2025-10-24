import numpy as np
import itertools
from typing import Iterable, Generator, ClassVar, Dict, Any, Self
from dataclasses import dataclass, field
import re
from collections import Counter


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


from .func_abstractions import ParameterSet, VibStatesData

# works with .func_abstractions
def solve_LSE_motif(motif: tuple[tuple,...], 
                    parameters: ParameterSet, vibdata: VibStatesData,
                    unit: str='Eh',
                    eval_mode: str = 'on-the-fly'):
    """
    solving a linear system of equations
    coeff_matrix = [[1, 0, 0], [1, -1, 0], [0, 1, -1]]
    constants = [5, -3, 2]
    output: [5. 2. 0.]

    returns a dict {f'w{i+1}': solution}
    """
    coeff_matrix = generate_LHS_motif(motif)
    constants = get_RHS_motif(motif, parameters, vibdata, unit, eval_mode)

    A = np.array(coeff_matrix)
    b = np.array(constants)
    
    try:
        solution = np.linalg.solve(A, b)
        from wilson_suite.wilson_utils.common_labels import num_cap_alpha_labels
        num_to_ax = {v:k for k,v in num_cap_alpha_labels.items()}

        return {num_to_ax[i]: val for i, val in enumerate(solution)}
    except np.linalg.LinAlgError as e:
        print("Error solving linear system:", e)

def generate_LHS_motif(motif: tuple[tuple,...]):
    """
    motif is a tuple/collection of res_conditions
        res_conditions is a tuple of (vib_difference, axes)
            vib_difference is a tuple of states indices
    """
    from wilson_suite.wilson_utils.common_labels import num_cap_alpha_labels
    # maximum variable index across all tuples
    max_var_index = max([len(rc[1]) for rc in motif])
    
    if max_var_index == 1:
        max_var_index = len(motif)
    
    # to identify coeff matrix shape
    coeff_matrix = np.zeros((max_var_index, max_var_index))

    for i, r_condition in enumerate(motif):
        axis_tupleID: tuple[str] = r_condition[1]

        # axis_tupleID = ('A', '-B') --> {'A': 1, 'B': -1} better?
        # coeffs {'A': 1, 'B': -1}
        coeffs = {var.strip('-') : 1 if '-' not in var else -1 for var in axis_tupleID}

        for alpha_label, coefficient in coeffs.items():
             # Reverse the sign and place it in the correct position
             coeff_matrix[i, num_cap_alpha_labels[alpha_label]] = -1 * np.sign(coefficient)
    
    return coeff_matrix

def get_RHS_motif(motif: tuple[tuple,...], 
            parameters: ParameterSet, vibdata: VibStatesData,
            unit: str='Eh',
            eval_mode: str = 'on-the-fly'):
    """
    making a constants vector from a list of tuples
    resonance_tuples = [(1, (-1,)), (2, (-1, 2)), (3, (-2, 3))]
    ind_tuple = (1, 2, 3) --- 
    vibdiffbank: VibDiffBank instance

    output: [5, -3, 2]
    """
    if eval_mode == 'on-the-fly':
        constants = [(-1)*get_vibdiff_motif(vibdiff_symb=rc[0], parameters=parameters,
                                            allstates_map=vibdata.allstates_map, unit=unit) for rc in motif]
    else:
        raise NotImplementedError('RHS can be only "on-the-fly" now')
    return constants

from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

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

@dataclass(frozen=True)
class Resonance:
    """
    pattern: tuple                   # e.g. ('b,a', (-1,2))
    location: tuple                  # coordinates or other representation
    producers = field(default_factory=lambda: list())  # list of dicts: {"term": str, "assignment": tuple, "value": float}

    """
    location: tuple
    producers: list = field(default_factory=lambda: list())
    
    def __hash__(self):
        # make assignment hashable by converting to tuple
        return hash((self.pattern, tuple(sorted(self.assignment.items())), self.location))

    def __eq__(self, other):
        return (
            isinstance(other, Resonance) and
            self.location == other.location
        )
    
    def add_producer(self, term_id, term_res_pattern, assignment, value=None):
        """
        term_id=term.short_id - string e.g. T001(1_0)
        term_res_pattern=term.resonances - tuple expression e.g. (('b,a', (-1,2)), ('a+b,a', (-1)))
        assignment=comb - (a,b,c) - numbers-indices

        !conflict with `frozen=True`
        """

        self.producers.append({"term": term_id, "pattern": term_res_pattern,
                               "assignment": assignment, "value": value})


def compress_terms_strlabel(terms):
    """Compress consecutive terms into ranges, preserving suffixes like (0_1)."""
    # Extract prefix, number, and suffix from each term
    parsed = []
    for t in terms:
        m = re.match(r"([A-Za-z]+)(\d+)(\(.*\))", t)
        if not m:
            parsed.append((t, None, None))  # fallback if it doesn't match
        else:
            prefix, num, suffix = m.groups()
            parsed.append((prefix, int(num), suffix))

    # Group by prefix+suffix
    groups = {}
    for prefix, num, suffix in parsed:
        key = (prefix, suffix)
        groups.setdefault(key, []).append(num)

    # For each group, sort and compress into ranges
    compressed = []
    for (prefix, suffix), nums in groups.items():
        nums = sorted(nums)
        start = prev = nums[0]
        for n in nums[1:] + [None]:  # add sentinel
            if n is None or n != prev + 1:
                # flush range
                if start == prev:
                    compressed.append(f"{prefix}{start}{suffix}")
                else:
                    compressed.append(f"{prefix}{start}–{prev}{suffix}")
                start = n
            prev = n

    return ",".join(compressed)


def resonance_to_str(resonance: Resonance) -> str:
    loc = resonance.location
    producers = resonance.producers

    # collect all terms
    terms = [p["term"] for p in producers]

    # assume all patterns/assignments/values are the same → take from first
    if producers:
        pattern = producers[0]["pattern"]
    else:
        pattern = None

    # build compact string
    return (
        f"Resonance @ ({loc[0]:.2f}, {loc[1]:.2f}); "
        f"terms={compress_terms_strlabel(terms)}; "
        f"pattern={pattern}; "
    )

def make_state_value_func(vibstates):
    """
    Returns a closure that maps index-tuples/lists to their vibrational energy
    using vibstates.

    Handles special cases like 'zero'.
    """
    # Pre-build a dictionary for fast lookup
    state_map = {}
    for state in vibstates:
        # state.serial_s is a dict like {'0,1,2': count}
        for key in state.serial_s.keys():
            state_map[key] = state.e

    def state_value_func(indices):
        """
        indices can be:
          - 'zero'
          - tuple/list/set of ints (mode indices)
        Returns float energy.
        """
        if indices == 'zero':
            return 0.0

        # normalize: tuple -> list -> sorted -> str
        if isinstance(indices, (tuple, list, set)):
            strtuple = ','.join(str(i) for i in sorted(indices))
        elif isinstance(indices, str):
            # If it's already a str like "0,1,2"
            strtuple = indices
        else:
            raise TypeError(f"Unsupported indices type: {type(indices)}")

        return state_map.get(strtuple, None)

    return state_value_func


@dataclass(frozen=True)
class EvalTerm:
    """
    EvalTerm with global registry to avoid duplicates.
    
    Example usage:
        {'resonances': (('b,a', (-1, 2)), ('zero,a', (-1,))),
        'vibenediff': ('b,a+b', 'a,zero'),
        'averaged_props': (('dipgrad', ('a',), ('B',)),
                            ('polgrad', ('b',), ('A', 'D')),
                            ('dipgrad', ('b',), ('G',))),
        'non_averaged_props': (('F', ('a', 'c', 'c',)),),
        'vibene_denom': ('a','b','c'),
        'termB_pref': 0.5,
        'termA_pref': -1/8.,
        'lvl_anharm': 2,
        'anharm_tuple': (1, 0)}
    """
    resonances: tuple
    vibenediff: tuple
    averaged_props: tuple
    non_averaged_props: tuple
    vibene_denom: tuple
    termB_pref: float
    termA_pref: float
    lvl_anharm: int
    anharm_tuple: tuple
    
    # Class variables for global registry
    _global_counter: ClassVar[int] = 0
    _registry: ClassVar[Dict[tuple, 'EvalTerm']] = {}
    
    def __new__(cls, *args, **kwargs):
        # Create a temporary instance to get the hash key
        if args:
            field_names = ['resonances', 'vibenediff', 'averaged_props', 
                          'non_averaged_props', 'vibene_denom', 'termB_pref', 
                          'termA_pref', 'lvl_anharm', 'anharm_tuple']
            kwargs.update(dict(zip(field_names, args)))
        
        # Create a key for the registry based on all field values
        key = cls._make_registry_key(kwargs)
        
        # Check if this exact term already exists
        if key in cls._registry:
            return cls._registry[key]
        
        # Create new instance using normal dataclass constructor
        instance = super().__new__(cls)
        return instance
    
    def __post_init__(self):
        # Create registry key and check if we need to register this instance
        key = self._make_registry_key(self.__dict__)
        
        if key not in EvalTerm._registry:
            EvalTerm._global_counter += 1
            object.__setattr__(self, '_seq_num', EvalTerm._global_counter)
            EvalTerm._registry[key] = self
        else:
            # This shouldn't happen due to __new__, but just in case ---???
            existing = EvalTerm._registry[key]
            object.__setattr__(self, '_seq_num', existing._seq_num)
    
    @staticmethod
    def _make_registry_key(field_dict):
        """Create a hashable key from the field values."""
        # Convert dict to sorted tuple of (key, value) pairs
        # Handle nested structures by converting to strings for hashing
        def make_hashable(obj):
            if isinstance(obj, (list, tuple)):
                return tuple(make_hashable(item) for item in obj)
            elif isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            else:
                return obj
        
        relevant_fields = {k: v for k, v in field_dict.items() 
                          if not k.startswith('_')}
        return tuple(sorted((k, make_hashable(v)) for k, v in relevant_fields.items()))
    
    @property
    def short_id(self) -> str:
        anharm_tuple_str = '_'.join([str(i) for i in self.anharm_tuple])
        return f"T{self._seq_num:03d}({anharm_tuple_str})"
    
    @classmethod
    def get_registry_stats(cls):
        """Get statistics about the global registry."""
        return {
            'total_unique_terms': len(cls._registry),
            'global_counter': cls._global_counter,
            'terms': {term.short_id: term for term in cls._registry.values()}
        }
    
    @classmethod
    def clear_registry(cls):
        """Clear the global registry (useful for testing)."""
        cls._registry.clear()
        cls._global_counter = 0

