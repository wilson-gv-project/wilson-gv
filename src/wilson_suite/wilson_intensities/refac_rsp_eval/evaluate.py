"""

[] 
[] 
---

==> list[SpectralFeature]
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from wilson_suite.wilson_derive.abstractions import (
    PolProp,
    VibDiffTerm,  # here and term_parts and vibene_differences
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ParameterSet,
)
from wilson_suite.wilson_utils.prop_trivname import prop_trivname
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

if TYPE_CHECKING:
    from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
        CompiledTerm,
        FreqTermsCollection,
        PropsCollection,
    )



@dataclass(frozen=True)
class DataOriginInfo:
    """
    Class to represent computational setups for properties obtained external to Wilson
    Does not need to pertain to an actual program and could also be used for "get from no specific calculation"/
    "get from file"

    ----
    source_type: String: Options: gaussian, cfour, wilson
    lvl_theory: String: Level of theory
    basis_set: String: Basis set
    base_file_loc: String: path to the base file
    """
    # Strings
    source_type: str = ''
    
    lvl_theory: str = ''
    basis_set: str = ''

    base_file_loc: str = ''


    def __hash__(self):
        def to_tuple(x):
            return tuple(sorted(x.items())) if isinstance(x, dict) else x
        return hash((self.source_type, self.lvl_theory, self.basis_set, to_tuple(self.base_file_loc)))

    def __eq__(self, other):
        if not isinstance(other, DataOriginInfo):
            return False
        
        return (
            self.source_type == other.source_type and
            self.lvl_theory == other.lvl_theory and
            self.basis_set == other.basis_set and
            self.base_file_loc == other.base_file_loc
        )



@dataclass
class VibState:
    """
    Class to represent a vibrational state.
    This is for a "concrete" vibrational state and not the same as its symbolic namesake in wilson-derive.

    ----
    s: dictionary {(harm. quanta): coeff, (harm. quanta): coeff, ...}: Specify the state in terms of harm. osc. WFs
    e: float: State energy level
    d: type not specified: Should be some form of vector to represent displacement in terms of atomic coordinates

    UPD:
    dictionary self.s is not JSON-serializable (tuples can't be keys), but self.serial_s is.
    self.serial_s is set up in post_init; deserialize_state_dict will return original self.s based on self.serial_s.

    Notes:
    s: InitVar[dict] = field(repr=False) - means that this atribute will not be in repr() of the class instance
    InitVar - is an init-only variable
    This seems to be okay for now, but should mind this feature
    """
    harm_quanta_coeffs: dict[tuple[int, ...], float]
    energy: float = 0.0
    displacement: Any = None
    serial_harm_quanta_coeffs: dict[str, float] = field(init=False)
    state_label: str = ""
    harmonic_WF: bool = False

    def __post_init__(self) -> None:
        """Convert tuple keys to comma-separated strings for JSON serialization."""
        self.serial_harm_quanta_coeffs = {
            ",".join(map(str, k)): v
            for k, v in self.harm_quanta_coeffs.items()
        }

    def deserialize_state_dict(self) -> dict[tuple[int, ...], float]:
        """Convert serialized dictionary back to original format with tuple keys."""
        return {
            tuple(int(x) for x in k.split(",")): v
            for k, v in self.serial_harm_quanta_coeffs.items()
        }

    def __eq__(self, other: 'VibState') -> bool:
        if not isinstance(other, VibState):
            return NotImplemented
        return self.state_label == other.state_label and bool(np.isclose(self.energy, other.energy))

    def __lt__(self, other: 'VibState') -> bool:
        if not isinstance(other, VibState):
            return NotImplemented
        return self.state_label < other.state_label
    
    @classmethod
    def get_1q_states(cls, states: list['VibState']):
        return [s for s in states if len(s.state_label.split(','))==1]


@dataclass
class VibStatesData:
    """
    Holds vib states data and can compute vib states energy differences
    """
    allstates: tuple[VibState,...]
    harmonic_osc_states_labels: tuple[int,...] = ()
    number_of_nmodes: int = 0
    
    def __post_init__(self):
        tmp_allstates = list(self.allstates)
        tmp_allstates.append(VibState(harm_quanta_coeffs={}, state_label='zero', energy=0.))
        self.allstates = tuple(tmp_allstates)
        
        self.allenergies_map = {i.state_label: i.energy for i in self.allstates}
        self.allstates_map = {i.state_label: i for i in self.allstates}
        self._storage = {}


    def get_harmonic_osc_states(self):
        """
        i.state_label - TODO: make a convention, rules how to describe vibstates
        now i.state_label is str
        """
        harm_states_str = [i for i in self.allstates if ',' not in i.state_label and i.state_label!='zero']
        harm_states = {int(i.state_label): i.energy for i in harm_states_str if int(i.state_label) in self.harmonic_osc_states_labels}

        return dict(sorted(harm_states.items()))
    
    def get_state_by_label(self, state_label):
        if state_label in self.allstates_map:
            return self.allstates_map.get(state_label)
        else:
            raise ValueError(f'Requested state label - {state_label} - is not in VibStatesData')
    
    # UNUSED
    def get_energy_by_label(self, state_label):
        if state_label in self.allstates_map:
            return self.allenergies_map.get(state_label)
        else:
            raise ValueError(f'Requested state label - {state_label} - is not in VibStatesData')


def make_vibdiff_key(vibdiff_term: VibDiffTerm, index_dict: dict) -> tuple[str, str]:
    """
    Non-sorted key for VibDiffBank_cache

    returns keys for vibdiff bank for vib states expression and choice of indices
    """
    left_state_symb = vibdiff_term.sl.q # type: ignore
    right_state_symb = vibdiff_term.sr.q # type: ignore


    left_state_label = ','.join([str(i) for i in sorted([index_dict[i] for i in left_state_symb])])
    right_state_label = ','.join([str(i) for i in sorted([index_dict[i] for i in right_state_symb])])
    
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
    left: VibState | None
    right: VibState | None
    
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
        if self.left is None or self.right is None:
            raise ValueError("Both left and right states must be provided for normalization.")
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
        if self.left is None or self.right is None:
            raise ValueError("Both left and right states must be provided to calculate energy difference.")
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

    @classmethod
    def from_quanta(cls, left_q, right_q, index_dict, vibstates_data: 'VibStatesData') -> 'VibDiff':
        """Same as from_symbolic, but from quanta labels instead of a VibDiffTerm —
        so callers holding a ResCondKey don't need a derive object."""
        def label(quanta):
            return ','.join(str(i) for i in sorted(index_dict[i] for i in quanta)) or 'zero'
        zero = VibState(harm_quanta_coeffs={}, state_label='zero', energy=0.0)
        ll, rl = label(left_q), label(right_q)
        return cls(left=zero if ll == 'zero' else vibstates_data.get_state_by_label(ll),
                   right=zero if rl == 'zero' else vibstates_data.get_state_by_label(rl))


@dataclass(frozen=True)
class MolSystemData:
    """
    data and holding abstractions.

    would be constructed outside of evaluation, and passed in to evaluation functions
    """
    name: str
    eigenvals: np.ndarray | None
    eigenvecs: np.ndarray | None
    mol_props: 'MolPropsCollection'
    states: 'VibStatesData'
    natoms: int = 0
    geo: Any = None
    geo_extra: Any = None
    linear: bool = False
    data_origin: DataOriginInfo | None = None

    @property
    def data_filled(self) -> bool:
        """Check if all properties have values."""
        return self.mol_props.is_filled


    @classmethod
    def from_datadict(cls, 
                      mol_props: 'MolPropsCollection',  # to be filled with data
                      data_dict: dict,                  # data obtained from obtainer
                      states_choice: str = 'harmonic'):
        """
        loading data into self.props (and optionally to self.vib_ana_setup)

        data_dict: dict - {data_name: values}

        """
        mol_props.fill_from(data_dict)

        harm_states, anharm_states = _make_hq_states_from_datadict(data_dict)
        if states_choice == 'harmonic':
            labels = tuple(int(i[0]) for i in harm_states if len(i))
            states = VibStatesData(allstates=harm_states, harmonic_osc_states_labels=labels)

        elif states_choice == 'anharmonic':
            labels = tuple(int(i[0]) for i in anharm_states if len(i))
            states = VibStatesData(allstates=anharm_states, harmonic_osc_states_labels=labels)

        geo = data_dict.get('geo', None)
        natoms = len(geo) if geo is not None else 0

        return cls(name='from_data_dict',
                   eigenvals=data_dict.get('nc_sqrt_eigval', None), # FIXME: none values should be handled better
                   eigenvecs=data_dict.get('eigenvecs', None), # FIXME: none values should be handled better
                   mol_props=mol_props,
                   natoms=natoms,
                   states=states,
                   geo=geo,
                   geo_extra=data_dict.get('geo_extra', None),
                   linear=data_dict.get('linear', False),
                   data_origin=data_dict.get('data_origin', None))


def _make_hq_states_from_datadict(data_dict: dict[str, Any]) -> tuple[tuple, tuple]:
    """
    Construct tuple['VibState'] from data_dict.
    """

    harm_states = ()
    if 'harmonic_states' in data_dict:
        states_dict: dict = data_dict['harmonic_states']

        for state, energy in states_dict.items():
            harm_states += (VibState(harm_quanta_coeffs={state: 1.0}, energy=energy, state_label=','.join(state)),)

    anharm_states = ()
    if 'anharmonic_states' in data_dict:
        states_dict: dict = data_dict['anharmonic_states']

        for state, energy in states_dict.items():
            anharm_states += (VibState(harm_quanta_coeffs={state: 1.0}, energy=energy, state_label=','.join(state)),)

    return harm_states, anharm_states



@dataclass
class MolecularProperty:
    """
    Class to represent a molecular (energy derivative or similar) property
    Can both be used "head only" (only prop_spec, target_basis, target_units) to specify only the concept of a property
    and "full" (system, calc_setup) for a particular realization (optional with/without values)
    
    ----
    prop_spec: Dictionary {'attr name': val, ...}: Info like perturbing operators, frequencies etc. (all values must be hashable)
    triv_name: String: Trivial name For simplified reference
    vals: Form not specified: Values of properties - could be array or dictionary
    system: MolecularSystem instance: For which system?
    calc_setup: DataOriginInfo instance: For which calculation setup?

    see more in test_main_dataclasses.py::test_MolecularProperty
    """
    # FIXME: Improve on prop_spec name; settle more consistently what the attributes will be and what must be default
    prop_spec: dict
    trivial_name: str | None = None
    system: MolSystemData | None = None
    calc_setup: DataOriginInfo | None = None
    vals: Any = field(default=None, repr=False)
    extra_data: dict | None = None

    def to_dict(self):
        return {
            "prop_spec": self.prop_spec,
            "trivial_name": self.trivial_name,
        }

    @classmethod
    def from_polprop(cls, polprop: 'PolProp', 
                     freqs: str = 'static') -> 'MolecularProperty':
        """
        Construct MolecularProperty from a PolProp instance.
        """
        ops = []

        ord_geo = polprop.dord
        for _ in range(ord_geo):
            ops.append('g')

        ord_el = len(polprop.ops)
        for _ in range(ord_el):
            ops.append('f')

        if freqs == 'static':
            pdict = {'ops': tuple(ops), 'freq': tuple([0.0 * k for k in range(len(ops))])}
        else:
            raise AssertionError('Managing electronic properties for non-static frequencies not yet implemented')

        return cls(prop_spec=pdict, trivial_name=prop_trivname(ord_geo=ord_geo, ord_el=ord_el))


@dataclass
class MolPropsCollection:
    properties: list[MolecularProperty]

    def get(self, trivial_name: str):
        d = {prop.trivial_name: prop for prop in self.properties}
        if trivial_name not in d:
            raise ValueError(f'trivial_name {trivial_name} is not in MolPropsCollection')
        return d.get(trivial_name)

    def __getitem__(self, trivial_name):
        """Allow coll[name] syntax."""
        return self.get(trivial_name)

    def __contains__(self, trivial_name: str) -> bool:
        """Allow `name in coll` syntax."""
        return any(p.trivial_name == trivial_name for p in self.properties)

    def __iter__(self):
        """Allow `for p in coll` syntax."""
        return iter(self.properties)

    def __len__(self) -> int:
        return len(self.properties)

    def names(self) -> list[str]:
        """All trivial names."""
        return [p.trivial_name for p in self.properties if p.trivial_name is not None]

    def filter(self, predicate: Callable[[MolecularProperty], bool]) -> 'MolPropsCollection':
        return MolPropsCollection([p for p in self.properties if predicate(p)])

    def without_values(self) -> 'MolPropsCollection':
        """Properties still awaiting data — useful for finding what's missing."""
        return self.filter(lambda p: p.vals is None)

    def by_calc_setup(self, origin: DataOriginInfo) -> 'MolPropsCollection':
        """All properties computed with a given setup."""
        return self.filter(lambda p: p.calc_setup == origin)

    def of_order(self, order: int) -> 'MolPropsCollection':
        """Properties of a specific differentiation order, e.g. 1 for dipole, 2 for polarizability."""
        return self.filter(lambda p: sum([1 for i in p.prop_spec['ops'] if i=='g']) == order)

    def group_by_calc_setup(self) -> dict[DataOriginInfo, 'MolPropsCollection']:
        """Bucket properties by which setup they use. For batching QC jobs."""
        from collections import defaultdict
        groups = defaultdict(list)
        for p in self.properties:
            groups[p.calc_setup].append(p)
        return {k: MolPropsCollection(v) for k, v in groups.items()}

    def dress(self, uniform: DataOriginInfo | None = None, 
            by_name: dict[str, DataOriginInfo] | None = None):
        """Attach DataOriginInfo to each property. 
        by_name takes precedence; uniform is the fallback."""
        if uniform is None and by_name is None:
            raise ValueError("Provide `uniform` or `by_name` (or both).")
        
        for p in self.properties:
            if by_name and p.trivial_name in by_name:
                p.calc_setup = by_name[p.trivial_name]
            elif uniform is not None:
                p.calc_setup = uniform
            else:
                raise ValueError(f"No setup for property {p}")

    @property
    def are_dressed(self) -> bool:
        return all(isinstance(p.calc_setup, DataOriginInfo) for p in self.properties)

    def build_request_dict(self) -> dict[str, DataOriginInfo]:
        """Build a {name: DataOriginInfo} shopping list."""
        if not self.are_dressed:
            raise RuntimeError("Collection must be dressed before requesting data.")

        result: dict[str, DataOriginInfo] = {}
        for p in self.properties:
            if p.trivial_name is None:
                raise ValueError("All properties must have a trivial name.")
            if p.calc_setup is None:
                raise ValueError("All properties must have a calc setup.")
            result[p.trivial_name] = p.calc_setup
        return result

    def fill_from(self, data_dict: dict):
        """Load obtained data into each property's .vals."""
        for p in self.properties:
            if p.trivial_name in data_dict:
                p.vals = data_dict[p.trivial_name]

    @property
    def is_filled(self) -> bool:
        return all(p.vals is not None for p in self.properties)


## ------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationDataAndConfigs:
    """
    data holding abstractions are used here
    """
    props_data: MolPropsCollection
    vibstates_data: 'VibStatesData'
    number_of_nmodes: int
    nm_inds_choices: Sequence[int]
    pulse_polarization_vector: list
    nc_sqrt_eigval: dict


@dataclass()
class PrecalculatedData:
    avrg_tensors: dict
    avrg_expr_tensor_mapping: dict
    vibenedenoms_tensors: dict


def evaluate_term_coeffs(compl_term: 'CompiledTerm',
                         relevant_indices: list[dict],
                         precalculated_data: 'PrecalculatedData',
                         molsys_data: 'MolSystemData',
                         zero_tol: float = 1e-18):
    idx_summ, idx_nonsumm = compl_term.idx_summ_nonsumm
    term_idx_all = sorted(idx_summ + idx_nonsumm)
    n_modes = len(molsys_data.eigenvals) if molsys_data.eigenvals is not None else 0

    def sum_over(index_dict, remaining, leaves):
        if not remaining:
            value, contribs = evaluate_single_index_dict(compl_term, index_dict, molsys_data, precalculated_data, zero_tol)
            leaves[ParameterSet(index_dict)] = contribs
            return value
        current, rest = remaining[0], remaining[1:]
        return sum(sum_over({**index_dict, current: v}, rest, leaves) for v in range(n_modes))

    results = {}
    for index_dict in relevant_indices:
        missing = [i for i in term_idx_all if i not in index_dict]
        leaves = {}
        results[ParameterSet(index_dict)] = (sum_over(index_dict, missing, leaves), leaves)
    return results


def evaluate_single_index_dict(compl_term: 'CompiledTerm',    # term
                               index_dict: dict,            # index choice (a,b,c...)
                               molsys_data: MolSystemData,
                               precalculated_data: PrecalculatedData | None,
                               zero_tol: float) -> tuple[float, dict]:
    """
    Evaluate the term for a single index dictionary.
    Parameters:
        (same as evaluate_term_coeffs)
    
    Returns:
        float: The computed coefficient for the given index dictionary.
    """
    # Evaluate NON_AVRG
    non_avrg_expr = compl_term.cmp_props.get_non_averaged_props()
    avrg_expr = compl_term.cmp_props.get_averaged_props().sort()

    NON_AVRG = eval_non_avrg_per_indexdict(non_avrg_expr, index_dict, molsys_data, zero_tol)
    if NON_AVRG == 0.0:
        if precalculated_data is not None:
            AVRG = eval_avrg_per_indexdict(avrg_expr, index_dict, precalculated_data, zero_tol)
        return 0.0, {'NON_AVRG': NON_AVRG, 'AVRG': AVRG}

    # Evaluate AVRG
    if precalculated_data is not None:
        AVRG = eval_avrg_per_indexdict(avrg_expr, index_dict, precalculated_data, zero_tol)
    if AVRG == 0.0:
        return 0.0, {'AVRG': AVRG}

    # Evaluate VIBDIFF_TERMS
    extra_freqterms = compl_term.cmp_freqdenom.get_pert_wf_diff()

    # VIBDIFF_TERMS = eval_vibdiff_pert_wf_diff(extra_freqterms, index_dict, precalculated_data, molsys_data)
    VIBDIFF_TERMS = otf_vibdiffdenom(extra_freqterms, index_dict, molsys_data)
    if VIBDIFF_TERMS == 0.0:
        return 0.0, {'VIBDIFF_TERMS': VIBDIFF_TERMS}

    # Evaluate VIBENE_DENOM
    freqterms = compl_term.cmp_freqdenom.get_vibenedenom()

    if precalculated_data is None or precalculated_data.vibenedenoms_tensors is None:
        VIBENE_DENOM = otf_vibdiffdenom(freqterms, index_dict, molsys_data)
    else:
        VIBENE_DENOM = eval_vibenedenom(freqterms, index_dict, precalculated_data)

    if VIBENE_DENOM == 0.0:
        return 0.0, {'VIBENE_DENOM': VIBENE_DENOM}

    # Compute the product
    product_all = NON_AVRG * AVRG * VIBDIFF_TERMS * VIBENE_DENOM

    dict_contribs = {'NON_AVRG': NON_AVRG, 'AVRG': AVRG, 'VIBDIFF_TERMS': VIBDIFF_TERMS, 'VIBENE_DENOM': VIBENE_DENOM}

    return float(compl_term.frac_factor) * float(product_all), dict_contribs


def eval_non_avrg_per_indexdict(non_avrg_expr: 'PropsCollection', 
                                index_dict: dict, 
                                molsys_data: MolSystemData,
                                zero_tol: float = 1e-18):
    """
    non_avrg_expr - extracted part of VibPerturbed term 

    order of indices generally: a,b,c,... 
    E.g. in CFF tensor index tuple is (a,b,c)

    TODO: error handling for missing or invalid data for all functions below
    """
    product_all = 1.
    
    for non_avrg_prop in non_avrg_expr:
        # accessing values for non-averaged properties from data
        if non_avrg_prop.inds is None or len(non_avrg_prop.inds) == 0:
            # If there are no indices, we assume it's a scalar property
            na_prop_inds = ()
        else:
            na_prop_inds = tuple([index_dict[i] for i in non_avrg_prop.inds])
        
        triv_name = prop_trivname(ord_el=len(non_avrg_prop.ops), ord_geo=non_avrg_prop.dord)

        if triv_name not in molsys_data.mol_props.names():
            raise ValueError(f"Trivial name {triv_name} not found in props_data.")

        prop = molsys_data.mol_props.get(triv_name)
        if prop is not None:
            NON_AVRG = prop.vals[na_prop_inds]

        if np.isclose(NON_AVRG, 0.0, atol=zero_tol, rtol=0.0):
            return 0.
        else:
            product_all *= NON_AVRG
    
    return product_all


def get_ind_tuple_from_base(expr: 'PropsCollection', base_expr: 'PropsCollection', index_dict: dict):
    """Map expr to indices according to base expression's unique symbols."""
    base_unique = sorted(set(base_expr.get_mode_indices()))
    expr_inds = expr.get_mode_indices()

    if len(base_unique) < len(expr_inds):
        # walk through only base_unique ind labels
        return tuple(index_dict[sym] for sym in base_unique)
    elif len(base_unique) == len(expr_inds):
        # walk through all expr_inds labels, there are repeated labels
        return tuple(index_dict[sym] for sym in expr_inds)
    else:
        # this should not be possible in the worflow
        raise ValueError('This base_expr cannot be a base expression for this expr')


def eval_avrg_per_indexdict(avrg_expr: 'PropsCollection', 
                            index_dict: dict, 
                            precalculated_data: PrecalculatedData,
                            zero_tol: float = 1e-18):
    """
    with precalculated_data
    """
    avrg_tensor_expr = precalculated_data.avrg_expr_tensor_mapping[avrg_expr]
    
    avrg_tensor = precalculated_data.avrg_tensors[avrg_tensor_expr]
    
    avrg_index_tuple = get_ind_tuple_from_base(expr=avrg_expr, 
                                                         base_expr=avrg_tensor_expr, 
                                                         index_dict=index_dict)
    if np.isclose(avrg_tensor[avrg_index_tuple], 0.0, atol=zero_tol, rtol=0.0):
        return 0.
    return avrg_tensor[avrg_index_tuple]


def make_gen_func_to_compute_avrg(*,
                              avrg_expression: 'PropsCollection',
                              pulse_polarization_vector: list) -> Callable[[dict, 'MolPropsCollection'], float]:
    """
    for an expression with properties data values,
    compute average with given polarization setup for a choice of normal mode indices
    """
    num_pulses = len(avrg_expression.get_cart_axes())  # should this be a set?

    from wilson_suite.wilson_intensities.amplitudes.averaging import (
        getGeneralPolarizationAveragingExpression,
    )

    polarization_linear_comb = getGeneralPolarizationAveragingExpression(rank = num_pulses,
                                                                        laser_pol = pulse_polarization_vector)

    def compute_for_idx_choice(index_choices: dict, props_data: 'MolPropsCollection') -> float:
        """
        index_choices: dict, props_data: 'MolPropsCollection'
        """

        if not isinstance(props_data, MolPropsCollection):
            raise TypeError(
                f"props_data must be a MolPropsCollection, got {type(props_data).__name__}"
            )

        # Validate index_choices has all required keys
        required_inds = {i for prop in avrg_expression for i in prop.inds} # type: ignore
        missing = required_inds - index_choices.keys()
        if missing:
            raise KeyError(
                f"index_choices is missing required mode indices: {missing}"
            )

        from wilson_suite.wilson_utils.prop_trivname import prop_trivname

        total = 0.

        for cart_axes in polarization_linear_comb:

            # Comment (MR): Noting that I considered if there would be any issues with this in generalized routine,
            # couldn't think of any but want to discuss and double check for safety

            product = 1.

            for prop in avrg_expression:

                prop_tuple_key = prop_trivname(ord_el=len(prop.ops), ord_geo=prop.dord)

                nm_inds = tuple([index_choices[i] for i in prop.inds]) # type: ignore
                cart_inds = tuple([cart_axes[i.o] for i in prop.ops])
                all_inds = (*nm_inds, *cart_inds)

                # retrieve data for preperty (prop_key) and idxs_key which is (tuple(mode inds), tuple(cart inds))
                getprop = props_data.get(prop_tuple_key)
                if getprop:
                    product *= getprop.vals[all_inds]

            # if product != 0.:
            #     logger.debug(f"Avrg prop contribution for indices {index_choices} and cart axes {cart_axes} with coefficient {polarization_linear_comb[cart_axes]}: {product}")

            total += product * polarization_linear_comb[cart_axes]

        return total

    return compute_for_idx_choice


def calculate_avrg_tensor(avrg_expression: 'PropsCollection',
                          pulse_polarization_vector: list,
                          props_data: 'MolPropsCollection',
                          number_of_nmodes: int,
                          modes_to_fill: list[int] | None = None) -> np.ndarray:
    """
    Precalculating the full tensor for given avrg_expression

    modes_to_fill - could be generated with for all normal modes with:
        modes_to_fill: list[int] = list(range(number_of_nmodes))

    """
    if modes_to_fill is None:
        modes_to_fill = list(range(number_of_nmodes))
    if max(modes_to_fill) >= number_of_nmodes:
        raise ValueError(f"modes_to_fill contains indices exceeding number_of_nmodes ({number_of_nmodes})")
    
    # so indices are in alphabetical order in full_tensor below
    mode_inds = sorted(set(avrg_expression.get_mode_indices()))  # list, deterministic order

    from wilson_suite.wilson_intensities.amplitudes.utils import (
        generate_index_choices_general,
    )

    ind_choices: list[dict[str, int]] = generate_index_choices_general(indlabels_in_motif=mode_inds, labels=modes_to_fill)

    # Indicating generalized version for updating
    func_general = make_gen_func_to_compute_avrg(avrg_expression=avrg_expression, pulse_polarization_vector=pulse_polarization_vector)

    full_tensor = np.zeros((number_of_nmodes,)*len(mode_inds))

    for idx in ind_choices:
        # so indices are in alphabetical order
        full_tensor[tuple(idx[k] for k in mode_inds)] = func_general(idx, props_data)

    return full_tensor


def eval_vibenedenom(freqterms: 'FreqTermsCollection',
                     index_dict: dict,
                     precalculated_data: PrecalculatedData):
    """
    try without precalculated data
    """
    vibenedenoms_tensor = precalculated_data.vibenedenoms_tensors[freqterms.get_num_indices_vibenedenom()]

    vibeneden_index_tuple = tuple([index_dict[i] for i in freqterms.get_num_indices_vibenedenom()])

    return vibenedenoms_tensor[vibeneden_index_tuple]

def otf_vibdiffdenom(freqterms: 'FreqTermsCollection',
                     index_dict: dict,
                     molsys_data: MolSystemData):
    """
    without precalculated data
    """
    product_all = 1.

    for vibdiff in freqterms:
        vib_diff_w_value = VibDiff.from_symbolic(vibdiff, index_dict,
                                                        molsys_data.states)

        product_all *= 1./ vib_diff_w_value.energy_difference(au=True)

    return product_all