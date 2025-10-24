from typing import Iterable, Generator, ClassVar, Dict, Any, Self
from dataclasses import dataclass, field
import re
from collections import Counter




@dataclass(frozen=True)
class Resonance:
    """
    pattern: tuple                   # e.g. ('b,a', (-1,2))
    location: tuple                  # coordinates or other representation
    producers = field(default_factory=lambda: list())  # list of dicts: {"term": str, "assignment": tuple, "value": float}

    """
    location: tuple
    producers: list = field(default_factory=lambda: list())
    
    # def __hash__(self):
    #     # make assignment hashable by converting to tuple
    #     return hash((self.pattern, tuple(sorted(self.assignment.items())), self.location))

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
                    compressed.append(f"{prefix}{start}-{prev}{suffix}")
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

