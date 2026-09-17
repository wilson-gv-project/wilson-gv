"""
ONLY importer of wilson_derive
  
  in:  terms IN axis choice -- as VibPertTermsCollection
       magn_conds IN axis choice -- 
                            -w1 + w2 is always > 0 ==> magn_conds = ((-1, 2),)
                            MagnConditions = tuple[tuple[int|str, ...], ...]

  
  out: EvalPlan + WorkManifest
  
  holds: PropsCollection, FreqTermsCollection, ResonanceMotif,
         parse_vibpert_term, index bookkeeping, motif keys

---

[] 
[] 
---

==> tuple[CompiledTerm]
"""

"""
=================================
int or tuple of ints would be independent vars.

independent vars here are -1 and 2:
    {'A': [-1], 'B': [2]} or {'A': [-1], 'B': [-1, 2]}
independent vars here are -1+2 and 3:
    {'A': [(-1,2),], 'B': [3,]} or {'A': [(-1,2),], 'B': [(-1, 2), 3]}

SpectralAxisSetDict = dict[str, tuple[int|tuple[int,...],...]]

----

# -w1 + w2 is always > 0 ==> magn_conds = ((-1, 2),)
MagnConditions = tuple[tuple[int|str, ...], ...]
"""

import copy
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from wilson_suite.wilson_derive.abstractions import (
    PolProp,  # here and term_parts
    ResonanceCondition,  # here and term_parts
    VibDiffTerm,  # here and term_parts and vibene_differences
)
from wilson_suite.wilson_utils.prop_trivname import prop_trivname

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
        """
        keeping copied instances in the collection
        """
        self.props = tuple(copy.deepcopy(self.props))

    def __iter__(self):
        yield from self.props

    def __hash__(self):
        return hash( (self.get_cart_axes(), self.get_mode_indices()) )
    
    def __eq__(self, other):
        """
        Now depends on comparison of PolProp instances.
        Now PolProp instances are considered equal if the have the same lists of operators (ops)
            (further, equality of QOperator instances) and same differentiation order (dord)
        """
        if isinstance(other, PropsCollection):
            return all(p in other.props for p in self.props)
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

    """
    Mutation after key insertion. sort() mutates self.props, returns self, and reassigns a list into a field __post_init__ had normalized to a tuple. A collection used as a key can be mutated after insertion. Frozen, with sorted() returning a new instance.
    """
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
    """
    VibDiffTerm's states are HarmOscStateSymbolic instances
    """
    freqterms: Sequence[VibDiffTerm]
    
    def __post_init__(self):
        """
        keeping copied instances in the collection
        """
        self.freqterms = tuple(copy.deepcopy(self.freqterms))

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
            return all(ft in other.freqterms for ft in self.freqterms)
        return False

    def get_vibenedenom(self):
        """
        freqterm.sl. and freqterm.sr. should be HarmOscStateSymbolic instances
        """
        def sr_or_sl_only(freqterm: VibDiffTerm):
            return (freqterm.sl.q == []) or (freqterm.sr.q == [])

        return FreqTermsCollection(freqterms=[ft for ft in self.freqterms if not ft.is_pert_wf_diff or sr_or_sl_only(ft)])
    
    def get_pert_wf_diff(self):
        return FreqTermsCollection(freqterms=[ft for ft in self.freqterms if ft.is_pert_wf_diff])
    
    def get_num_indices_vibenedenom(self):
        """
        these vibdiffterms have only sl, sr is zero
        """
        return tuple(sorted({i for vd in self.get_vibenedenom() for i in vd.sl.q}))


@dataclass(frozen=True)
class ResCondKey:
    """One resonance condition reduced to identity: quanta labels + perturbing freqs.
    `diff` is (left quanta, right quanta); ground state is ()."""
    diff: tuple[tuple[str, ...], tuple[str, ...]]
    pf: tuple[str, ...]

    @property
    def left(self):
        return self.diff[0]

    @property
    def right(self):
        return self.diff[1]


@dataclass(frozen=True)
class ResonanceMotif:
    """Plan-level dedup key: a motif is its canonical tuple, nothing more.
    Immutable, so it never aliases derive-owned ResonanceConditions and never copies them."""
    conditions: tuple[ResCondKey, ...]

    def __post_init__(self):
        object.__setattr__(self, 'conditions', tuple(self.conditions))

    @classmethod
    def from_conditions(cls, conditions: Sequence[ResonanceCondition]) -> 'ResonanceMotif':
        return cls(tuple(
            ResCondKey(diff=(tuple(c.diff.sl.q), tuple(c.diff.sr.q)), pf=tuple(c.pf))
            for c in conditions
        ))

    @classmethod
    def from_tuples(cls, tuple_of_tuples) -> 'ResonanceMotif':
        """sorted() reproduces HarmOscStateSymbolic's normalisation of q."""
        return cls(tuple(
            ResCondKey(diff=(tuple(sorted(sl)), tuple(sorted(sr))), pf=tuple(pf))
            for (sl, sr), pf in tuple_of_tuples
        ))

    def _tuplify(self):
        return tuple((c.diff, c.pf) for c in self.conditions)

    def __iter__(self):
        yield from self.conditions

    def __len__(self):
        return len(self.conditions)

    def get_max_different_freq_axes(self):
        return {ax.strip('-') for c in self.conditions for ax in c.pf}

    def get_nm_indices(self):
        return {label for c in self.conditions for quanta in c.diff for label in quanta}

    def __repr__(self):
        return f'{self.conditions}'


"""
ParameterSet has a type that lies. Declared Mapping[str, int], but __init__ injects params['zero'] = 'zero' (a str value) and __getitem__ remaps '' → 'zero'; __lt__ hardcodes the alphabet ('a'..'h'). A generic index-assignment type that secretly knows vibrational-state labelling conventions. Decide which it is: if generic, the zero sentinel and ordering are policy living in a labelling module; if domain, name it (IndexAssignment) and make the conventions explicit and tested. The ground state currently spelled three ways ('', 'zero', state_label == 'zero') is that ambiguity leaking.
"""
# numerical
@dataclass(frozen=True)
class ParameterSet(Mapping[str, int]):
    """
    Immutable mapping of parameter label -> index value.
    """
    _parameters: Mapping[str, int]

    def __init__(self, parameters: Mapping[str, int]):
        if not isinstance(parameters, Mapping):
            raise TypeError("ParameterSet must be initialized with a mapping.")

        params = dict(parameters)

        if 'zero' not in params:
            params['zero'] = 'zero'

        object.__setattr__(
            self,
            "_parameters",
            MappingProxyType(params),
        )

    # --- Mapping interface ---

    def __getitem__(self, key: str):
        if key == '':
            key = 'zero'
        return self._parameters[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameters)

    def __len__(self) -> int:
        return len(self._parameters)

    # --- Equality & hashing ---

    def __eq__(self, other):
        if not isinstance(other, ParameterSet):
            return NotImplemented
        return self._parameters == other._parameters

    def __hash__(self):
        # Order-independent, value-based hash
        return hash(frozenset(self._parameters.items()))

    def __lt__(self, other):
        if not isinstance(other, ParameterSet):
            return NotImplemented
        
        sort_keys = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
        relevant_keys = [k for k in sort_keys if k in self or k in other]

        # Sort keys to ensure we compare 'a', then 'b', then 'c' 
        # regardless of insertion order.
        self_vals = tuple(self.get(k, 0) for k in relevant_keys)
        other_vals = tuple(other.get(k, 0) for k in relevant_keys)
        
        return self_vals < other_vals

    # --- Convenience ---

    def parameter_labels(self):
        return [k for k in self._parameters if k != 'zero']

    def indices(self):
        return [v for v in self._parameters.values() if v != 'zero']

    def to_dict(self):
        return dict(self._parameters)

    def __repr__(self):
        repr_d = {k: v for k, v in self._parameters.items() if k != 'zero'}
        return f"{self.__class__.__name__}({repr_d})"

    # --- Pickle support ---
    def __getstate__(self):
        # Return plain dict instead of mappingproxy
        return {'_parameters': dict(self._parameters)}

    def __setstate__(self, state):
        object.__setattr__(self, "_parameters", MappingProxyType(state['_parameters']))



## --------------------------------------------------------------------
@dataclass
class CompiledTerm:
    """
    In Axes.

    full = 1.num_fact * 2.(avrg_props * nonavrg_props) * 3.freq_denom * 4.res_motf
        cmp_props = (avrg_props * nonavrg_props)
    """
    cmp_props: PropsCollection
    cmp_resmotf: ResonanceMotif
    cmp_freqdenom: FreqTermsCollection
    frac_factor: float

    @classmethod
    def from_VibPertTerm(cls, term: 'VibPerturbedTerm'):

        frac_factor = float(term.coeff)
        properties = PropsCollection(term.props)
        freq_denom = FreqTermsCollection(term.freqterms)
        res_conds = ResonanceMotif.from_conditions(term.res)

        return cls(properties, res_conds, freq_denom, frac_factor)


## --------------------------------------------------------------------

def compile_terms(terms: Sequence['VibPerturbedTerm']):

    compiled = []
    for t in terms:
        compiled.append(CompiledTerm.from_VibPertTerm(t))

    return compiled