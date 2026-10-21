"""
From derive's terms to compiled terms: the symbolic side, no molecular data.

  in:  terms (VibPerturbedTerm) already translated to axes: pf ('-A', 'B'), not pulse ids (-1, 2).
       ResCondKey raises ValueError on pulse ids.
  out: list[CompiledTerm] (compile_terms)

  holds: PropsCollection, FreqTermsCollection, ResCondKey, ResonanceMotif, generate_LHS_motif, ParameterSet,
         ResLocPoint, split_props, CompiledTerm

The only module of the package (tests not counted) that imports wilson_derive at run time:
PolProp, ResonanceCondition and VibDiffTerm from wilson_derive.abstractions.
"""

import copy
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from wilson_suite.wilson_derive.abstractions import (
    PolProp,
    ResonanceCondition,
    VibDiffTerm,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.compat import term_to_str
from wilson_suite.wilson_utils.prop_trivname import prop_trivname

if TYPE_CHECKING:
    from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm


@dataclass
class PropsCollection:
    """
    A tuple of PolProps, copied when the collection is made: changing the originals does not change it.
        get_averaged_props       the properties with field operators (ops), as a new collection
        get_non_averaged_props   the properties without field operators, as a new collection
        sort                     a new collection, sorted by each property's first Cartesian slot
        names                    the data-array name of each property, from dord and len(ops)
        get_cart_axes            the Cartesian slot (op.o) of every field operator, in order
        get_mode_indices         the mode labels (inds) of every property, in order
    """
    props: Sequence[PolProp]

    def __post_init__(self):
        self.props = tuple(copy.deepcopy(self.props))

    def __iter__(self):
        yield from self.props

    def _key(self) -> tuple:
        """One (slots, inds, dord) per property, sorted: the order of the properties does not count, repeats do."""
        return tuple(sorted((tuple(op.o for op in p.ops), tuple(p.inds or ()), p.dord) for p in self.props))

    def __hash__(self):
        return hash(self._key())
    
    def __eq__(self, other):
        """
        Equal if the properties are the same, in any order; repeats count, e.g.
            PropsCollection([p1, p2]) == PropsCollection([p2, p1])        True
            PropsCollection([p1, p1]) == PropsCollection([p1])            False
        A property compares by its slots (op.o, in order), inds and dord (_key); __hash__ reads the same key.
        """
        if not isinstance(other, PropsCollection):
            return NotImplemented
        return self._key() == other._key()
    
    def get_averaged_props(self):
        """
        The properties with field operators (ops, derive's QOperators), e.g. polgrad[b] dipgrad[a] dipgrad[c]:
        the orientation average reads them.
        """
        return PropsCollection(props=[p for p in self.props if p.ops])
    
    def get_non_averaged_props(self):
        """
        The properties without field operators, e.g. cff[a,b,c] (dord 3): read as they are, not averaged.
        """
        return PropsCollection(props=[p for p in self.props if not p.ops])
    
    @cached_property
    def names(self) -> tuple[str, ...]:
        """
        The trivial name of each property, in order, one per property (a name twice if the property is there twice),
            e.g. polgrad[b] * dipgrad[a] * dipgrad[c] -> ('polgrad', 'dipgrad', 'dipgrad')
        The name picks the data array the property reads: geometric derivative order (dord) and number of field
        operators (ops). Made once per collection: the evaluation reads the names for every index set.
        """
        return tuple(prop_trivname(ord_geo=p.dord, ord_el=len(p.ops)) for p in self.props)

    def get_cart_axes(self):
        return tuple(op.o for p in self.props for op in p.ops)
    
    def get_mode_indices(self):
        groups = [p.inds if p.inds is not None else [] for p in self.props]
        return tuple(idx for p_inds in groups for idx in p_inds)

    def sort(self) -> 'PropsCollection':
        """
        A new collection, sorted by the Cartesian slot (op.o) of each property's first field operator; properties
        without field operators go last, e.g. dipgrad[a] (slot 1), polgrad[b] (slots 0, 3) -> polgrad[b], dipgrad[a].
        The collection it is called on is not changed.
        """
        return PropsCollection(sorted(self.props, key=lambda j: j.ops[0].o if j.ops else float('inf')))

    def __repr__(self):
        """
        Name (from dord, as names), mode indices, slots (op.o) and dord of each property, e.g.
            polgrad['b'][0, 3]_d1 * dipgrad['a'][1]_d1 * dipgrad['c'][2]_d1 * cff['a', 'b', 'c'][]_d3
        """
        return ' * '.join(f'{name}{p.inds}{[op.o for op in p.ops]}_d{p.dord}'
                          for name, p in zip(self.names, self.props))


@dataclass
class FreqTermsCollection:
    """
    A tuple of VibDiffTerms (denominators), copied when the collection is made: changing the originals does not
    change it. A VibDiffTerm's states sl and sr are HarmOscStateSymbolic instances, read by their quanta q,
    e.g. sl.q ['a', 'b'], sr.q ['c'] for 1/(E_ab - E_c).
        get_vibenedenom, get_pert_wf_diff   the 1/omega / the energy-difference denominators, as a new collection
        get_max_state_lvl                   the most quanta in one state
    """
    freqterms: Sequence[VibDiffTerm]

    def __post_init__(self):
        self.freqterms = tuple(copy.deepcopy(self.freqterms))

    def __iter__(self):
        yield from self.freqterms

    def _key(self) -> tuple:
        """One (sl.q, sr.q, is_pert_wf_diff) per denominator, sorted: the order does not count, repeats do."""
        return tuple(sorted((tuple(ft.sl.q), tuple(ft.sr.q), ft.is_pert_wf_diff)  # type: ignore
                            for ft in self.freqterms))

    def __hash__(self):
        return hash(self._key())
    
    def __eq__(self, other):
        """
        Equal if the denominators are the same, in any order; repeats count. A denominator compares by its quanta
        and is_pert_wf_diff (_key), not by memory address (VibDiffTerm has no __eq__), so two collections made
        from the same term are equal. __hash__ reads the same key.
        """
        if not isinstance(other, FreqTermsCollection):
            return NotImplemented
        return self._key() == other._key()

    def get_vibenedenom(self):
        """
        The 1/omega denominators (is_pert_wf_diff False), e.g. 1/omega_a: one mode in sl, the ground state in sr
        (harmonic_denom checks both).
        """
        return FreqTermsCollection(freqterms=[ft for ft in self.freqterms if not ft.is_pert_wf_diff])
    
    def get_pert_wf_diff(self):
        """The energy-difference denominators (is_pert_wf_diff True), e.g. 1/(E_ab - E_c), as a new collection."""
        return FreqTermsCollection(freqterms=[ft for ft in self.freqterms if ft.is_pert_wf_diff])

    def get_max_state_lvl(self):
        """The most quanta in one state; 0 without freqterms (the ground state has no quanta)."""
        return max((max(len(vd.sl.q), len(vd.sr.q)) for vd in self.freqterms), default=0) # type: ignore


@dataclass(frozen=True, order=True)
class ResCondKey:
    """
    One resonance condition as plain values: the quanta of its two states, and its pf (perturbing frequencies).
    The resonance is where the signed sum of pf equals E(left) - E(right).
        diff  (left quanta, right quanta); the ground state is (), e.g. (('a', 'b'), ('a',)): states ab and a
        pf    signed axes, sorted (a sum, so the order carries no meaning), e.g. ('-A', 'B') for -A + B
    e.g. ResCondKey(diff=((), ('a',)), pf=('-A',)): -A = E_0 - E_a, so the peak is at A = E_a - E_0.
    Frozen and ordered, so conditions can be compared and sorted (ResonanceMotif sorts them).
    raises ValueError on pulse IDs (int), e.g. pf (-1, 2) as derive writes it: the evaluator takes terms already
    translated to axes, e.g. pf ('-A', 'B').
    """
    diff: tuple[tuple[str, ...], tuple[str, ...]]
    pf: tuple[str, ...]

    def __post_init__(self):
        if not all(isinstance(ax, str) for ax in self.pf):
            raise ValueError(f'resonance condition with pulse IDs, pf {self.pf}: the evaluator takes terms translated '
                             f"to axes, e.g. pf ('-A', 'B')")
        object.__setattr__(self, 'pf', tuple(sorted(self.pf)))

    @property
    def left(self):
        return self.diff[0]

    @property
    def right(self):
        return self.diff[1]


@dataclass(frozen=True)
class ResonanceMotif:
    """
    The resonance conditions of one term, as a sorted tuple of ResCondKeys: the same conditions in any order
    give the same motif. e.g. the two conditions
        ResCondKey(diff=((), ('a',)), pf=('-A',))          -A = E_0 - E_a
        ResCondKey(diff=(('a', 'b'), ('a',)), pf=('B',))    B = E_ab - E_a
    fix one point: A = E_a - E_0, B = E_ab - E_a. ResonanceMotif(()) is the motif of a term without conditions.
    Frozen and hashable, so a dict key, e.g. one location per (motif, index set).
    Holds only quanta labels and pf: no derive objects.
    """
    conditions: tuple[ResCondKey, ...]

    def __post_init__(self):
        object.__setattr__(self, 'conditions', tuple(sorted(self.conditions)))

    @classmethod
    def from_conditions(cls, conditions: Sequence[ResonanceCondition]) -> 'ResonanceMotif':
        """A motif from derive's ResonanceConditions: reads the quanta (diff.sl.q, diff.sr.q) and pf of each."""
        return cls(tuple(
            ResCondKey(diff=(tuple(c.diff.sl.q), tuple(c.diff.sr.q)), pf=tuple(c.pf)) # type: ignore
            for c in conditions
        ))

    @classmethod
    def from_tuples(cls, tuple_of_tuples) -> 'ResonanceMotif':
        """
        A motif from plain tuples, one ((left quanta, right quanta), pf) per condition, e.g.
            [((('a',), ()), ('A',)), ((('a', 'b'), ('a',)), ('B',))]
        The quanta are sorted, as HarmOscStateSymbolic sorts q: ('b', 'a') and ('a', 'b') are the same state.
        """
        return cls(tuple(
            ResCondKey(diff=(tuple(sorted(sl)), tuple(sorted(sr))), pf=tuple(pf))
            for (sl, sr), pf in tuple_of_tuples
        ))

    def __iter__(self):
        yield from self.conditions

    def __len__(self):
        return len(self.conditions)

    def get_max_different_freq_axes(self):
        """The distinct axes of the conditions, without sign, e.g. pf ('-A',) and ('A', 'B') -> {'A', 'B'}."""
        return {ax.strip('-') for c in self.conditions for ax in c.pf}

    def get_nm_indices(self):
        """The mode labels in the conditions' states, e.g. diff ((), ('a',)) and (('a', 'b'), ('a',)) -> {'a', 'b'}."""
        return {label for c in self.conditions for quanta in c.diff for label in quanta}

    def get_max_state_lvl(self):
        """The most quanta in one state; 0 without conditions (a flat background)."""
        return max((max(len(c.diff[0]), len(c.diff[1])) for c in self.conditions), default=0)

    def __repr__(self):
        return f'{self.conditions}'


def generate_LHS_motif(motif: ResonanceMotif,
                       axes: Iterable[str] | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    """
    The left-hand side of the motif's resonance conditions as a matrix.
        motif  its conditions (ResCondKey), pf in axes, e.g. pf ('-A',) and ('B',)
        axes   the columns; None -> the distinct axes of the motif
    returns (coeff_matrix, axes): one row per condition, in the motif's order, and one column per axis in
    `axes` (sorted). Axes the motif does not mention get an all-zero column. An entry is minus the axis
    sign in pf (see the loop), e.g. for the pf above, in that order: axes ('A', 'B'), matrix [[1, 0], [0, -1]].
    raises ValueError if `axes` misses an axis of the motif, e.g. axes ('A',) for pf ('B',).
    Reads only the axes (pf) of the conditions, not the states, so CompiledTerm.from_VibPertTerm can check
    a motif without molecular data.
    """
    motif_axes = motif.get_max_different_freq_axes()
    all_axes = tuple(sorted(motif_axes if axes is None else set(axes)))
    if not motif_axes <= set(all_axes):
        raise ValueError(f"axes {all_axes} do not cover the axes of {motif}: {sorted(motif_axes)}")

    col = {ax: j for j, ax in enumerate(all_axes)}

    coeff_matrix = np.zeros((len(motif), len(all_axes)))

    for i, r_cond_key in enumerate(motif):
        # e.g. ('A', '-B') --> {'A': 1, 'B': -1}
        coeffs = {var.strip('-') : 1 if '-' not in var else -1 for var in r_cond_key.pf}

        for alpha_label, coefficient in coeffs.items():
             # Minus the axis sign: 'A' -> -1, '-B' -> +1. With get_RHS_motif's -E the row reads
             # -(signed sum of pf) = -E, i.e. E - (signed sum of pf) = 0 (derive's convention)
             coeff_matrix[i, col[alpha_label]] = -1 * np.sign(coefficient)

    return coeff_matrix, all_axes


@dataclass(frozen=True)
class ParameterSet(Mapping[str, int]):
    """
    The mode numbers chosen for a term's mode labels (one index set), e.g.
    ParameterSet({'a': 0, 'b': 1, 'c': 4}): label a is mode 0, b is mode 1, c is mode 4.
    Immutable and hashable, unlike a dict, so it can be a dict key
    (FeatureResult.factors_by_index_set, ContributionTable.group_by).
    Holds exactly the given labels. The ground state needs no entry: it has no quanta
    (VibDiff.from_quanta, make_state_label(()) -> 'zero').
    == and hash ignore the order of the labels; there is no <, so sorted() raises TypeError.
    """
    _parameters: Mapping[str, int]

    def __init__(self, parameters: Mapping[str, int]):
        if not isinstance(parameters, Mapping):
            raise TypeError("ParameterSet must be initialized with a mapping.")

        object.__setattr__(
            self,
            "_parameters",
            MappingProxyType(dict(parameters)),
        )

    # --- Mapping interface ---

    def __getitem__(self, key: str):
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

    # --- Convenience ---

    def parameter_labels(self):
        return list(self._parameters)

    def indices(self):
        return list(self._parameters.values())

    def to_dict(self):
        return dict(self._parameters)

    def __repr__(self):
        return f"{self.__class__.__name__}({dict(self._parameters)})"

    # --- Pickle support ---
    def __getstate__(self):
        # Return plain dict instead of mappingproxy
        return {'_parameters': dict(self._parameters)}

    def __setstate__(self, state):
        object.__setattr__(self, "_parameters", MappingProxyType(state['_parameters']))


CoordValue = float
Coordinates = tuple[tuple[str, CoordValue], ...]

@dataclass(frozen=True, slots=True, init=False)
class ResLocPoint:
    """
    The location of a resonance: one value per axis, axes sorted, e.g. Point(A=1864.0, B=900.0).
    ResLocPoint({}) is the location without axes of a flat background.
    Frozen; == and hash come from the coordinates, so a point can be a dict key (features.py: group_by).
    """
    coordinates: Coordinates            # The only stored data: sorted (axis, value) pairs, e.g. (('A', 1864.0),)

    def __init__(self, coord_dict: Mapping[str, CoordValue]):
        # A new tuple: a later change to the caller's dict does not change the point
        object.__setattr__(self, 'coordinates', tuple(sorted(coord_dict.items())))

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self.coordinates)

    @property
    def values(self) -> tuple[CoordValue, ...]:
        return tuple(v for _, v in self.coordinates)

    @property
    def dimensionality(self) -> int:
        """The number of axes."""
        return len(self.coordinates)

    def as_dict(self) -> dict[str, CoordValue]:
        """A new dict {axis: value}; changing it does not change the point."""
        return dict(self.coordinates)

    def __getitem__(self, axis: str) -> CoordValue:
        for k, v in self.coordinates:
            if k == axis:
                return v
        raise KeyError(f"Axis {axis} not found")

    def __repr__(self) -> str:
        """The coordinates, e.g. Point(A=1864.0, B=900.0); Point() for the location without axes."""
        coords = ", ".join(f"{k}={v}" for k, v in self.coordinates)
        return f"Point({coords})"


## --------------------------------------------------------------------
def split_props(term: 'VibPerturbedTerm') -> tuple[PropsCollection, PropsCollection]:
    """
    The averaged properties (sorted) and the non-averaged properties of a term: CompiledTerm.avrg_props and
    .non_avrg_props, e.g. (polgrad[b] dipgrad[a] dipgrad[c],  cff[a,b,c]): the properties with field operators
    are averaged. Works on terms in pulse ids too: the translation to axes changes only the resonance conditions.
    """
    properties = PropsCollection(term.props)
    return properties.get_averaged_props().sort(), properties.get_non_averaged_props()


@dataclass(frozen=True)
class CompiledTerm:
    """
    One term in the form the evaluation reads: its parts split, pf in axes (not pulse ids).
    At one index set, the term's coefficient is the product of (evaluate_full_index_dict)
        frac_factor      the number in front, derive's coeff as a float, e.g. -0.125
        avrg_props       the properties with field operators, orientation-averaged; sorted (PropsCollection.sort),
                         e.g. polgrad[b] dipgrad[a] dipgrad[c]
        non_avrg_props   the properties without, e.g. cff[a,b,c]
        cmp_freqdenom    the denominators, e.g. 1/omega_a, 1/omega_b, 1/omega_c, 1/(E_ab - E_c)
                         (split into freq_vibenedenom and freq_pert_wf_diff)
    cmp_resmotf, the resonance conditions, puts the coefficient on the spectrum (solve_LSE_motif),
        e.g. two conditions, pf ('-A',) and ('B',).
    idx_nonsumm are the mode labels the conditions fix, e.g. ('a', 'b'); idx_summ are summed over, e.g. ('c',).
    """
    avrg_props: PropsCollection
    non_avrg_props: PropsCollection

    cmp_resmotf: ResonanceMotif
    cmp_freqdenom: FreqTermsCollection
    frac_factor: float
    idx_summ: tuple[str, ...]
    idx_nonsumm: tuple[str, ...]


    @classmethod
    def from_VibPertTerm(cls, term: 'VibPerturbedTerm') -> 'CompiledTerm':
        """
        The CompiledTerm of one VibPerturbedTerm already translated to axes, e.g. pf ('-A', 'B').
        raises ValueError on pulse ids, e.g. pf (-1, 2) (ResCondKey).

        Also raises ValueError if the term's resonance conditions do not fix exactly one point.
        One point needs as many conditions as axes, all independent (rank). Checked here, without data:
        the condition matrix needs only the axes, not the energies.
        e.g. conditions on A and on B -> a point in 2D; on B only -> a point in 1D (fine);
             one condition on A + B -> a line; two conditions on B only -> no point or the same one twice.
        A term without conditions passes: it is a flat background, the same on every grid
        point (solve_LSE_motif gives it the location without axes).
        A term may also lack its non-averaged properties or its energy-difference denominators: each then gives
        the factor 1.
        """
        frac_factor = float(term.coeff)
        freq_denom = FreqTermsCollection(term.freqterms)      # Copies derive's VibDiffTerms
        res_conds = ResonanceMotif.from_conditions(term.res)  # Quanta labels and pf only, no derive objects
        if len(res_conds) > 0:
            lhs, axes = generate_LHS_motif(res_conds)
            rank = int(np.linalg.matrix_rank(lhs))
            if not len(res_conds) == len(axes) == rank:
                raise ValueError(f"resonance conditions do not fix one point: {len(res_conds)} conditions, "
                                 f"axes {axes}, rank {rank}\n{term_to_str(term)}")
        idx_nonsumm, idx__summ = term.tellNonSummSummIndices()

        avrg_props, non_avrg_props = split_props(term)

        return cls(avrg_props, non_avrg_props, res_conds, freq_denom, frac_factor, tuple(idx__summ), tuple(idx_nonsumm)) # type: ignore

    @cached_property
    def all_props(self) -> PropsCollection:
        """
        The averaged and the non-averaged properties together, e.g. for building the data request.
        Made once per term: a new PropsCollection deepcopies its properties.
        """
        return PropsCollection([*self.avrg_props, *self.non_avrg_props])

    @cached_property
    def prop_names(self) -> tuple[str, ...]:
        """The trivial names of the term's properties, each once, e.g. ('polgrad', 'dipgrad', 'cff')."""
        return tuple(dict.fromkeys(self.all_props.names))

    # The two kinds of denominator, split once per term and kept: evaluate_full_index_dict reads them for every
    # index set, and each split makes a new FreqTermsCollection (a deepcopy).
    @cached_property
    def freq_vibenedenom(self) -> FreqTermsCollection:
        """The 1/omega denominators (is_pert_wf_diff False), e.g. 1/omega_a: omega from the Hessian eigenvalues."""
        return self.cmp_freqdenom.get_vibenedenom()

    @cached_property
    def freq_pert_wf_diff(self) -> FreqTermsCollection:
        """The energy-difference denominators (is_pert_wf_diff True), e.g. 1/(E_ab - E_a): from the chosen states."""
        return self.cmp_freqdenom.get_pert_wf_diff()

    @property
    def max_state_lvl(self):
        """The most quanta in one state of the term, over its denominators and its conditions, e.g. 2 for state ab."""
        return max(self.cmp_freqdenom.get_max_state_lvl(), self.cmp_resmotf.get_max_state_lvl())


## --------------------------------------------------------------------

def compile_terms(terms: Sequence['VibPerturbedTerm']) -> list[CompiledTerm]:
    """
    CompiledTerm.from_VibPertTerm of each term, in the same order.
    raises ValueError on the first term in pulse ids, or whose resonance conditions do not fix one point.
    """
    compiled = []
    for t in terms:
        compiled.append(CompiledTerm.from_VibPertTerm(t))

    return compiled

