"""
From derive's terms to compiled terms: the symbolic side, no molecular data (todo.txt H6).

  in:  terms (VibPerturbedTerm) already translated to axes (data_for_tests/translated_terms)
  out: list[CompiledTerm] (compile_terms)

  holds: PropsCollection, FreqTermsCollection, ResCondKey, ResonanceMotif, generate_LHS_motif, ParameterSet,
         ResLocPoint, split_props, CompiledTerm

The one module that imports wilson_derive at run time: abstractions, the parts of a term (todo.txt H8).
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
        """
        ops is a list of QOperator instances - electromagnetic field coupling operators
        """
        return PropsCollection(props=[p for p in self.props if p.ops])
    def get_non_averaged_props(self):
        """
        also, dord is geometry differentiation order, so dord=0 for non-averaged props
        """
        return PropsCollection(props=[p for p in self.props if not p.ops])
    
    @cached_property
    def names(self) -> tuple[str, ...]:
        """
        The trivial name of each property, in order, one per property (a name twice if the property is there twice),
            e.g. term 8's averaged part polgrad[b] * dipgrad[a] * dipgrad[c] -> ('polgrad', 'dipgrad', 'dipgrad')
        The name picks the data array the property reads: geometric derivative order (dord) and number of field
        operators (ops). Made once per collection: the evaluation reads the names for every index set.
        """
        return tuple(prop_trivname(ord_geo=p.dord, ord_el=len(p.ops)) for p in self.props)

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

    """
    Left from a review: PropsCollection is not frozen, and its == is loose (todo.txt D1).
    sort() returns a new instance since 2026-10-12, so it no longer changes a collection after use (or its names).
    """
    def sort(self) -> 'PropsCollection':
        """
        A new collection, sorted by the Cartesian slot of each property's first operator;
        non-averaged props will be in the end of the tuple. This collection stays as it is.
        """
        return PropsCollection(sorted(self.props, key=lambda j: j.ops[0].o if j.ops else float('inf')))

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
        Only one vib state energy.

        freqterm.sl and freqterm.sr should be HarmOscStateSymbolic instances
        """

        return FreqTermsCollection(freqterms=[ft for ft in self.freqterms if not ft.is_pert_wf_diff])
    
    def get_pert_wf_diff(self):
        return FreqTermsCollection(freqterms=[ft for ft in self.freqterms if ft.is_pert_wf_diff])

    def get_max_state_lvl(self):
        return max(max(len(vd.sl.q), len(vd.sr.q)) for vd in self.freqterms) # type: ignore


def sr_or_sl_only(freqterm: VibDiffTerm):
    return (freqterm.sl.q == []) or (freqterm.sr.q == []) # type: ignore


@dataclass
class ResCondCollection:
    resconds: Sequence[ResonanceCondition]


@dataclass(frozen=True, order=True)
class ResCondKey:
    """One resonance condition reduced to identity: quanta labels + perturbing freqs.
    `diff` is (left quanta, right quanta); ground state is ().
    `pf` is a sum of signed axes, so its order carries no meaning and is sorted.
    raises ValueError on pulse IDs (int), e.g. pf (-1, 2) as derive writes it: the evaluator takes terms already
    translated to axes, e.g. pf ('-A', 'B') (data_for_tests/translated_terms)."""
    diff: tuple[tuple[str, ...], tuple[str, ...]]
    pf: tuple[str, ...]

    def __post_init__(self):
        if not all(isinstance(ax, str) for ax in self.pf):
            raise ValueError(f'resonance condition with pulse IDs, pf {self.pf}: the evaluator takes terms translated '
                             f"to axes, e.g. pf ('-A', 'B') (data_for_tests/translated_terms)")
        object.__setattr__(self, 'pf', tuple(sorted(self.pf)))

    @property
    def left(self):
        return self.diff[0]

    @property
    def right(self):
        return self.diff[1]


@dataclass(frozen=True)
class ResonanceMotif:
    """Plan-level dedup key: a motif is its canonical tuple, nothing more.
    Conditions are sorted, so the same conditions in any order are the same motif.
    Immutable, so it never aliases derive-owned ResonanceConditions and never copies them."""
    conditions: tuple[ResCondKey, ...]

    def __post_init__(self):
        object.__setattr__(self, 'conditions', tuple(sorted(self.conditions)))

    @classmethod
    def from_conditions(cls, conditions: Sequence[ResonanceCondition]) -> 'ResonanceMotif':
        return cls(tuple(
            ResCondKey(diff=(tuple(c.diff.sl.q), tuple(c.diff.sr.q)), pf=tuple(c.pf)) # type: ignore
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

    def get_max_state_lvl(self):
        return max(max(len(c.diff[0]), len(c.diff[1])) for c in self.conditions) # type: ignore

    def __repr__(self):
        return f'{self.conditions}'


def generate_LHS_motif(motif: ResonanceMotif,
                       axes: Iterable[str] | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    """
    motif is a tuple/collection of res_conditions
        res_conditions is a tuple of (vib_difference, axes)
            vib_difference is a tuple of states indices

    returns (coeff_matrix, axes): one row per condition, one column per axis in `axes`
    (sorted; defaults to the distinct axes of the motif). Axes the motif does not
    mention get an all-zero column.
    Reads only the axes (pf) of the conditions, no states: here, not in evaluate.py,
    so that CompiledTerm.from_VibPertTerm can check the motif without data.
    """
    motif_axes = motif.get_max_different_freq_axes()
    all_axes = tuple(sorted(motif_axes if axes is None else set(axes)))
    if not motif_axes <= set(all_axes):
        raise ValueError(f"axes {all_axes} do not cover the axes of {motif}: {sorted(motif_axes)}")

    col = {ax: j for j, ax in enumerate(all_axes)}

    coeff_matrix = np.zeros((len(motif), len(all_axes)))

    for i, r_cond_key in enumerate(motif):
        # ('A', '-B') --> {'A': 1, 'B': -1}
        coeffs = {var.strip('-') : 1 if '-' not in var else -1 for var in r_cond_key.pf}

        for alpha_label, coefficient in coeffs.items():
             # minus the axis sign: 'A' -> -1, '-B' -> +1. With get_RHS_motif's -E the row reads
             # -(signed sum of pf) = -E, i.e. E - (signed sum of pf) = 0 (derive's convention)
             coeff_matrix[i, col[alpha_label]] = -1 * np.sign(coefficient)

    return coeff_matrix, all_axes


@dataclass(frozen=True)
class ParameterSet(Mapping[str, int]):
    """
    Immutable mapping of label -> mode number: one index set of a term, e.g. ParameterSet({'a': 0, 'b': 1, 'c': 4}).
    It holds exactly the given labels. The ground state needs no entry: it has no quanta, so nothing is looked up
    (VibDiff.from_quanta, make_state_label(()) -> 'zero'). Hashable, so a dict key (leaves,
    ContributionTable.by_params); not ordered (todo.txt D3).
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

class ResLocPoint:
    """
    Represents geometric objects in N-dimensional space that are hashable.

    Examples:
        Point:      (('A', 1864.0), ('B', 900.0))

    """
    def __init__(self, coord_dict: dict[str, CoordValue]):
        # own copy: a later change to the caller's dict must not change this point
        self._coord_dict = dict(coord_dict)
        # dict to sorted tuple of tuples for consistent hashing
        self.coordinates: Coordinates = tuple(sorted(coord_dict.items()))

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self.coordinates)

    @property
    def values(self) -> tuple[CoordValue, ...]:
        return tuple(v for _, v in self.coordinates)

    @property
    def dimensionality(self) -> int:
        """Returns dimensionality of the point"""
        return len(self._coord_dict)

    def as_dict(self) -> dict[str, CoordValue]:
        """{axis: value} as a new dict; changing it does not change the point."""
        return dict(self._coord_dict)

    def __getitem__(self, axis: str) -> CoordValue:
        for k, v in self.coordinates:
            if k == axis:
                return v
        raise KeyError(f"Axis {axis} not found")

    def __eq__(self, other) -> bool:
        if not isinstance(other, ResLocPoint):
            return NotImplemented
        return self.coordinates == other.coordinates

    def __hash__(self) -> int:
        return hash(self.coordinates)

    def __repr__(self) -> str:
        """
        Returns a string representation showing type and coordinates.

        Examples:
            Point(A=1864.0, B=900.0)
        """
        coords = ", ".join(f"{k}={v}" for k, v in self.coordinates)
        return f"Point({coords})"


## --------------------------------------------------------------------
def split_props(term: 'VibPerturbedTerm') -> tuple[PropsCollection, PropsCollection]:
    """
    (averaged props, sorted; non-averaged props) of a term: CompiledTerm.avrg_props and .non_avrg_props.
        e.g. EVV term 8: (polgrad[b] dipgrad[a] dipgrad[c],  cff[a,b,c]): the props with field operators are averaged
    Reads only the props: the translation to axes changes only the resonance conditions, so the props are the
    ones derive wrote. prop_names and make_polarization_linear_comb read them through here.
    """
    properties = PropsCollection(term.props)
    return properties.get_averaged_props().sort(), properties.get_non_averaged_props()


@dataclass(frozen=True)
class CompiledTerm:
    """
    In Axes.

    full = 1.num_fact * 2.(avrg_props * nonavrg_props) * 3.freq_denom * 4.res_motf
        cmp_props = (avrg_props * nonavrg_props)
    
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
        VibPerturbedTerm - in axes.

        raises ValueError if the term's resonance conditions do not fix exactly one point
        (B7): as many conditions as axes, all independent (rank). Checked here, without data:
        the condition matrix needs only the axes, not the energies.
        e.g. conditions on A and on B -> a point in 2D; on B only -> a point in 1D (fine);
             one condition on A + B -> a line; two conditions on B only -> no point or the same one twice.
        A term without conditions passes: it is a flat background (todo.txt E12), the same on every grid
        point (solve_LSE_motif gives it the location without axes).

        TODO: enable possibility of missing parts
        """
        frac_factor = float(term.coeff)
        freq_denom = FreqTermsCollection(term.freqterms) # states
        res_conds = ResonanceMotif.from_conditions(term.res) # states
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
        avrg + non-avrg props together, e.g. for building the data request.
        Made once per term: a new PropsCollection deepcopies its props.
        """
        return PropsCollection([*self.avrg_props, *self.non_avrg_props])

    @cached_property
    def prop_names(self) -> tuple[str, ...]:
        """the trivial names of the term's properties, each once, e.g. term 8: ('polgrad', 'dipgrad', 'cff')"""
        return tuple(dict.fromkeys(self.all_props.names))

    # The two kinds of denominator, split once per term and kept (todo.txt F3).
    # evaluate_full_index_dict reads them for every index set; splitting there made a new FreqTermsCollection,
    # with a deepcopy, each time: 11 of 22 s of build_contributions on ACDM (42 modes).
    @cached_property
    def freq_vibenedenom(self) -> FreqTermsCollection:
        """the 1/omega denominators (is_pert_wf_diff False), e.g. 1/omega_a: omega from the Hessian eigenvalues"""
        return self.cmp_freqdenom.get_vibenedenom()

    @cached_property
    def freq_pert_wf_diff(self) -> FreqTermsCollection:
        """the energy-difference denominators (is_pert_wf_diff True), e.g. 1/(E_ab - E_a): from the chosen states"""
        return self.cmp_freqdenom.get_pert_wf_diff()

    @property
    def max_state_lvl(self):
        return max(self.cmp_freqdenom.get_max_state_lvl(), self.cmp_resmotf.get_max_state_lvl())


## --------------------------------------------------------------------

def compile_terms(terms: Sequence['VibPerturbedTerm']) -> list[CompiledTerm]:

    compiled = []
    for t in terms:
        compiled.append(CompiledTerm.from_VibPertTerm(t))

    return compiled

