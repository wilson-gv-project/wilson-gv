"""
Levels of summation:

- vibstates_labels
- terms


result_grid

"""
import numpy as np
import itertools
from typing import Iterable, Generator, ClassVar, Dict, Any, Self
from dataclasses import dataclass, field
from collections import Counter

from collections.abc import Mapping
import copy

class ParameterSet(Mapping):
    """
    Dict-like holder of "parameter label -> index value" mapping

    index value should be in VibState label space, so it's a string likely
    """
    def __init__(self, parameters):

        if not isinstance(parameters, dict):
            raise TypeError("ParameterSet must be initialized with a dictionary.")
        parameters = copy.deepcopy(parameters)
        
        if 'zero' not in parameters:
            parameters['zero'] = 'zero'
        self._parameters = dict(parameters)
        self._hash = hash(frozenset(self._parameters.items()))

    def parameter_labels(self):
        labels = list(self._parameters.keys())
        labels.remove("zero")
        return labels
    
    def indices(self):
        inds_all = list(self._parameters.values())
        inds_all.remove("zero")
        return inds_all

    def __getitem__(self, key):
        if key=='':
            key = 'zero'
        return self._parameters[key]

    def __iter__(self):
        return iter(self._parameters)

    def __len__(self):
        return len(self._parameters)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"{self.__class__.__name__}({self._parameters})"

    def __eq__(self, other):
        if isinstance(other, ParameterSet):
            return self._parameters == other._parameters
        return False
    
    def to_dict(self):
        return self._parameters

    @classmethod
    def from_dict(cls, parameters):
        return cls(parameters)

from wilson_suite.wilson_utils.abstractions import VibState
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

@dataclass
class VibStatesData:
    """
    Holds vib states data and can compute vib states energy differences
    """
    allstates: tuple[VibState]
    harmonic_osc_states_labels: tuple

    def __post_init__(self):
        tmp_allstates = list(self.allstates)
        tmp_allstates.append(VibState(s={}, state_label='zero', e=0.))
        self.allstates = tuple(tmp_allstates)
        
        self.allstates_map = {i.state_label: i.e for i in self.allstates}
        self._storage = dict()

    def _fill_storage(self):
        for vlabel_a, energy_a in self.allstates_map:
            for vlabel_b, energy_b in self.allstates_map:
                self._storage[(vlabel_a, vlabel_b)] = convNu2Ene(energy_a - energy_b)


@dataclass
class EvalualtionLayer:
    term_label: list | tuple # pointer to the term via its termID, has symbolic vibstates_label
    vibstates_label: list | tuple # 
    resonance: list | tuple
    freq_input: np.ndarray
    func_of_freq_input: np.ndarray


@dataclass(frozen=True)
class ResonancePoint:
    """
    pattern: tuple                   # e.g. ('b,a', (-1,2))
    location: tuple                  # coordinates or other representation
    producers = field(default_factory=lambda: list())  # list of dicts: {"term": str, "assignment": tuple, "value": float}
    """
    location: tuple
    term_id: str # term_res_pattern: str -- will be implied? should pass whole term?
    parameters: ParameterSet # {'a': 1, 'b': 0, 'c': 3, ...}
    factor_value: float
    Gamma: tuple


class ResonanceWaveMatch(Mapping):
    __slots__ = ("_items",)

    def __init__(self, wavematching: dict):
        self._items = tuple(sorted(wavematching.items()))

    def __getitem__(self, key):
        for k, v in self._items:
            if k == key:
                return v
        raise KeyError(key)

    def __iter__(self):
        return (k for k, _ in self._items)

    def __len__(self):
        return len(self._items)

    def __hash__(self):
        return hash(self._items)

    def __eq__(self, other):
        if not isinstance(other, ResonanceWaveMatch):
            return NotImplemented
        return self._items == other._items

    def __repr__(self):
        return f"ResonanceWaveMatch({dict(self._items)})"


@dataclass(frozen=True)
class VibDiffSymbolic:
    """
    attributes of vibrational difference
    """
    left: str
    right: str
    wavematching: ResonanceWaveMatch = None # [-1, 2] is {'1': -1, '2': 1} - is this better?

    def resolve_states(self, parameters: ParameterSet):
        """Get the actual vibrational states from parameters."""
        return parameters[self.left], parameters[self.right]

    def evaluate(self, parameters: ParameterSet, vibdata: VibStatesData,
                 eval_mode: str = 'on-the-fly'):
        if not self.wavematching is not None:
            raise ValueError('Cannot compute expression with frequency variables')
        
        left_state, right_state = self.resolve_states(parameters)
        return vibdata.get_vibdiff(left_state, right_state, eval_mode)



@dataclass(frozen=True)
class VibEneSymbolic:
    resonances: tuple[VibDiffSymbolic, ...]
    energy_differences: tuple[VibDiffSymbolic, ...]
    denominators: tuple[str, ...]


@dataclass(frozen=True)
class MolPropertySymbolic:
    """
    Abstraction

    MolPropertySymbolic(name='mu_Q', cart_axes=('B',), nm_indices=('a',)),
    MolPropertySymbolic(name='mu_QQ', cart_axes=('G',), nm_indices=('a', 'b'))
    MolPropertySymbolic(name='alpha_Q', cart_axes=('A', 'D'), nm_indices=('b',)),
    MolPropertySymbolic(name='CFF', cart_axes=(), nm_indices=('a', 'b', 'c'))

    """

    trivial_name: str
    cart_axes: tuple[str]
    nm_indices: tuple[str]
    tensor: np.ndarray = 0.

    @property
    def simple_tuple(self) -> tuple:
        """
        len(self.cart_axes) - number of EL perturbations
        len(self.nm_indices) - number of normal coordinates derivatives
        """
        return tuple([len(self.cart_axes), len(self.nm_indices)])


@dataclass(frozen=True)
class GroupPropsSymbolic:
    """
    props together in one tuple; is a key for precalc dict

    self.property_simple_tuples = tuple([p.simple_tuple for p in self.properties])
    self.nice_props = AveragedProps(self.properties)

    'averaged_props': (('dipgrad', ('a',), ('B',)),
                        ('polgrad', ('b',), ('A', 'D')),
                        ('dipgrad', ('b',), ('G',))),
    'non_averaged_props': (('F', ('a', 'c', 'c',)),),
    """
    
    props: tuple[MolPropertySymbolic]

    @property
    def cart_axes(self: Self) -> tuple:
        """Cartesian axes indices (strs)."""
        return tuple([p.cart_axes for p in self.props])

    @property
    def nm_indices(self: Self) -> tuple:
        """Normal modes indices (strs)."""
        return tuple([p.nm_indices for p in self.props])


    def __eq__(self: Self, other: Self):
        if not isinstance(other, GroupPropsSymbolic):
            return False
        return (
            self.cart_axes == other.cart_axes and
            sorted(self.nm_indices) == sorted(other.nm_indices)
        )

    def __hash__(self):
        return hash((self.cart_axes, frozenset(Counter(self.nm_indices).items())))

    def __repr__(self):
        return f'GroupPropsSymbolic:\n   {self.cart_axes}\n   {self.nm_indices}\n'


@dataclass(frozen=True)
class PropertiesGrouped:
    """
    'averaged_props': (('dipgrad', ('a',), ('B',)),
                        ('polgrad', ('b',), ('A', 'D')),
                        ('dipgrad', ('b',), ('G',))),
    'non_averaged_props': (('F', ('a', 'c', 'c',)),),
    """
    averaged: GroupPropsSymbolic
    non_averaged: GroupPropsSymbolic

@dataclass(frozen=True)
class TermCoefficients:
    term_a: float
    term_b: float

@dataclass(frozen=True)
class AnharmonicLevelInfo:
    level: int
    el_mech: tuple[int, int]


@dataclass(frozen=True)
class EvaluationTerm:
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
    vib_structure: VibEneSymbolic
    properties: PropertiesGrouped
    coefficients: TermCoefficients
    anharmonicity: AnharmonicLevelInfo

    # Class variables for global registry
    _global_counter: ClassVar[int] = 0
    _registry: ClassVar[dict[tuple, 'EvaluationTerm']] = {}
    
    def __new__(cls, *args, **kwargs):
        # Create a temporary instance to get the hash key
        if args:
            field_names = ['vib_structure', 'properties', 'coefficients', 'anharmonicity']
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
        
        if key not in EvaluationTerm._registry:
            EvaluationTerm._global_counter += 1
            object.__setattr__(self, '_seq_num', EvaluationTerm._global_counter)
            EvaluationTerm._registry[key] = self
        else:
            # This shouldn't happen due to __new__, but just in case ---???
            existing = EvaluationTerm._registry[key]
            object.__setattr__(self, '_seq_num', existing._seq_num)
    
    @staticmethod
    def _make_registry_key(field_dict: dict[str,Any]):
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
        anharm_tuple_str = '_'.join([str(i) for i in self.anharmonicity.el_mech])
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

