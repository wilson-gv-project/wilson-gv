import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from wilson_suite.wilson_utils.prop_trivname import prop_trivname
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

if TYPE_CHECKING:
    from wilson_suite.wilson_derive.abstractions import (
        PolProp,
        VibDiffTerm,  # here and term_parts and vibene_differences
    )
    from wilson_suite.wilson_intensities.refac_rsp_eval.plan import CompiledTerm


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
    allstates: Sequence[VibState]
    harmonic_osc_states_labels: Sequence[int] = ()
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

def make_state_label(modes) -> str:
    """
    (10, 2) -> '2,10';  ('1', '0') -> '0,1';  () -> 'zero'
    
    assert state_label(('10', '2')) == state_label([2, 10]) == '2,10'

    modes is Sequence[int | str]
    """
    return ','.join(str(m) for m in sorted(int(m) for m in modes)) or 'zero'


def _make_vibdiff_key(vibdiff_term: 'VibDiffTerm', index_dict: dict) -> tuple[str, str]:
    """
    Non-sorted key for VibDiffBank_cache

    returns keys for vibdiff bank for vib states expression and choice of indices
    """

    return (make_state_label(index_dict[q] for q in vibdiff_term.sl.q), # type: ignore
            make_state_label(index_dict[q] for q in vibdiff_term.sr.q)) # type: ignore



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
                    vibdiff_term_symb: 'VibDiffTerm',
                    index_dict: dict,
                    vibstates_data: 'VibStatesData') -> 'VibDiff':
        """Construct VibDiff from symbolic representation."""
        
        return cls.from_quanta(vibdiff_term_symb.sl.q, vibdiff_term_symb.sr.q, 
                               index_dict, vibstates_data)

    @classmethod
    def from_quanta(cls, left_q, right_q, index_dict, vibstates_data: 'VibStatesData') -> 'VibDiff':
        """Same as from_symbolic, but from quanta labels instead of a VibDiffTerm —
        so callers holding a ResCondKey don't need a derive object."""

        ll = make_state_label(index_dict[q] for q in left_q)
        rl = make_state_label(index_dict[q] for q in right_q)
        return cls(left=vibstates_data.get_state_by_label(ll),
                right=vibstates_data.get_state_by_label(rl))


@dataclass(frozen=True)
class MolSystemData:
    """
    data and holding abstractions.

    would be constructed outside of evaluation, and passed in to evaluation functions

    SHOULD HAVE DATA FILLED FOR EVALUATION (ALL ATTRIBUTES CONTAIN NUMBERS)
    """
    name: str
    eigenvals: dict[int, float] | None
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
                      states_choice: str = 'harmonic',
                      name: str = 'from_data_dict'):
        """
        loading data into self.props (and optionally to self.vib_ana_setup)

        data_dict: dict - {data_name: values}

        """
        # resets values, because props in collection shold be from the same source
        
        mol_props = mol_props.empty_copy()
        mol_props.fill_from(data_dict)

        harm_states, anharm_states = _make_hq_states_from_datadict(data_dict)
        if states_choice == 'harmonic':
            labels = tuple(int(i.state_label.split(',')[0]) for i in harm_states if len(i.state_label.split(','))==1)
            states = VibStatesData(allstates=harm_states, harmonic_osc_states_labels=labels)

        elif states_choice == 'anharmonic':
            labels = tuple(int(i.state_label.split(',')[0]) for i in harm_states if len(i.state_label.split(','))==1)
            states = VibStatesData(allstates=anharm_states, harmonic_osc_states_labels=labels)
        else:
            raise ValueError('states_choice can be harmonic or anharmonic.')
        
        # VibStatesData could be empty if no data

        geo = data_dict.get('geo', None)
        natoms = len(geo) if geo is not None else 0

        return cls(name=name,
                   eigenvals=data_dict.get('nc_sqrt_eigval', None), # FIXME: none values should be handled better
                   eigenvecs=data_dict.get('normal_modes', None), # FIXME: none values should be handled better
                   mol_props=mol_props,
                   natoms=natoms,
                   states=states,
                   geo=geo,
                   geo_extra=data_dict.get('geo_extra', None),
                   linear=data_dict.get('linear', False),
                   data_origin=data_dict.get('data_origin', None))


def _sys_info_request(data_origin: DataOriginInfo):
    keys = ['anharmonic_states', 'harmonic_states', 'nc_sqrt_eigval', 'normal_modes',
            'atoms', 'equilibrium_geometry']
    return dict.fromkeys(keys, data_origin)

def build_data_request_for_term(term: 'CompiledTerm', data_origin: DataOriginInfo) -> dict:
    request = term.avrg_props.build_request_dict(calc_setup=data_origin)
    request.update(term.non_avrg_props.build_request_dict(calc_setup=data_origin))
    request.update(_sys_info_request(data_origin))
    return request

def _make_hq_states_from_datadict(data_dict: dict[str, Any]) -> tuple[tuple, tuple]:
    """
    Construct tuple['VibState'] from data_dict.
    """

    harm_states = ()
    if 'harmonic_states' in data_dict:
        states_dict: dict = data_dict['harmonic_states']

        for state, energy in states_dict.items():
            harm_states += (VibState(harm_quanta_coeffs={state: 1.0}, energy=energy, state_label=make_state_label(state)),)

    anharm_states = ()
    if 'anharmonic_states' in data_dict:
        states_dict: dict = data_dict['anharmonic_states']

        for state, energy in states_dict.items():
            anharm_states += (VibState(harm_quanta_coeffs={state: 1.0}, energy=energy, state_label=make_state_label(state)),)

    return harm_states, anharm_states


@dataclass
class MolecularProperty:
    """
    Class to represent a molecular (energy derivative or similar) property

    holds name (key), values; 
        does not hold provenance info but would be retireved from MolSystemData with this info
    
    ----
    triv_name: String: Trivial name For simplified reference
    vals: Form not specified: Values of properties - could be array or dictionary
    """
    trivial_name: str | None = None
    vals: Any = field(default=None, repr=False)
    extra_data: dict | None = None

    def to_dict(self):
        return {
            "trivial_name": self.trivial_name,
            "extra_data": self.extra_data,
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

        return cls(trivial_name=prop_trivname(ord_geo=ord_geo, ord_el=ord_el),
                   extra_data=pdict)


@dataclass
class MolPropsCollection:
    properties: Sequence[MolecularProperty]


    def get(self, trivial_name: str) -> MolecularProperty | None:
        return next((p for p in self.properties if p.trivial_name == trivial_name), None)

    def __getitem__(self, trivial_name: str) -> MolecularProperty:
        prop = self.get(trivial_name)
        if prop is None:
            raise KeyError(f'trivial_name {trivial_name} is not in MolPropsCollection')
        return prop

    def __contains__(self, item: str | MolecularProperty) -> bool:
        """Allow `name in coll` syntax."""
        if isinstance(item, MolecularProperty):
            return item in self.properties
        return any(p.trivial_name == item for p in self.properties)

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

    def fill_from(self, data_dict: dict):
        """Load obtained data into each property's .vals."""
        for p in self.properties:
            # resets values, because props in collection shold be from the same source
            p.vals = None
            if p.trivial_name in data_dict:
                p.vals = data_dict[p.trivial_name]
    
    @property
    def is_filled(self) -> bool:
        return all(p.vals is not None for p in self.properties)


    def empty_copy(self) -> 'MolPropsCollection':
        """Same property names, new objects, no values. The template is not touched."""
        return MolPropsCollection([MolecularProperty(trivial_name=p.trivial_name,
                                                    extra_data=copy.deepcopy(p.extra_data))
                                for p in self.properties])
## ----------------------------------------------------
