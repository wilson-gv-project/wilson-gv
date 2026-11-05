import copy
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from wilson_suite.wilson_utils.prop_trivname import prop_trivname
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

if TYPE_CHECKING:
    from wilson_suite.wilson_derive.abstractions import (
        VibDiffTerm,
    )

# What every evaluation reads besides the properties: MolSystemData.load requests these from every source.
EVALUATION_DATA_KEYS = ('nc_sqrt_eigval', 'harmonic_states', 'anharmonic_states')

# Property name -> (g, e): g = number of derivatives by normal modes, e = number of derivatives by the electric field.
# The array has one index of length n (number of modes) per g, then one index of length 3 (x, y, z) per e.
# Example: 'polhess' (polarizability, 2nd derivative by modes) -> (2, 2) -> shape (n, n, 3, 3).
PROP_INDEX_COUNTS = {prop_trivname(ord_geo=g, ord_el=e): (g, e) for g in range(7) for e in range(7)}

# Symmetry in the mode indices: values that differ by less than this count as equal. 1e-18 is the evaluator's zero_tol (au).
SYMMETRY_RTOL, SYMMETRY_ATOL = 1e-12, 1e-18

# Label of the ground state (no quanta): make_state_label(()) gives it; VibStatesData adds the state with energy 0.
ZERO_STATE_LABEL = 'zero'


@dataclass(frozen=True)
class DataOriginInfo:
    """
    Class to represent computational setups for properties obtained external to Wilson
    Does not need to pertain to an actual program and could also be used for "get from no specific calculation"/"get from file"

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
    harm_quanta_coeffs: dictionary {(modes with quanta): coeff, ...}: Specify the state in terms of harm. osc. WFs,
        e.g. {('0', '1'): 1.0}: one quantum in mode 0 and one in mode 1. From the data dict the modes are str
    energy: float: State energy level in cm-1
    displacement: type not specified: Should be some form of vector to represent displacement in terms of atomic coordinates
    state_label: str: The modes with quanta, sorted, one entry per quantum (make_state_label), e.g. '2,10' for one quantum
        in mode 2 and one in mode 10, '1,1' for two quanta in mode 1, ZERO_STATE_LABEL for no quanta (ground state)
    harmonic_WF: bool: True if the state is a harmonic oscillator state, False otherwise
    ----

    Dictionary self.harm_quanta_coeffs is not JSON-serializable (tuples can't be keys), but self.serial_harm_quanta_coeffs is.
    self.serial_harm_quanta_coeffs is set up in post_init; deserialize_state_dict rebuilds the tuple keys from it, with
    int modes: ('0', '1') comes back as (0, 1).
    """
    harm_quanta_coeffs: dict[tuple[int | str, ...], float]
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

    @classmethod
    def get_1q_states(cls, states: list['VibState']):
        """Get states with exactly one quantum in any mode. The ground state ('zero', no quanta) is not one."""
        return [s for s in states if s.state_label != ZERO_STATE_LABEL and ',' not in s.state_label]


@dataclass
class VibStatesData:
    """
    Holds the vibrational states of one state set (e.g. the harmonic states), looked up by label.
    Adds the ground state itself ('zero', energy 0). VibDiff computes the energy differences.
    """
    allstates: Sequence[VibState]
    harmonic_osc_states_labels: Sequence[int] = ()

    def __post_init__(self):
        tmp_allstates = list(self.allstates)
        tmp_allstates.append(VibState(harm_quanta_coeffs={}, state_label=ZERO_STATE_LABEL, energy=0.))
        self.allstates = tuple(tmp_allstates)

        self.allstates_map = {i.state_label: i for i in self.allstates}


    def get_harmonic_osc_states(self) -> dict[int, float]:
        """
        Get harmonic oscillator states from allstates, filtering by harmonic_osc_states_labels.
        Returns {mode: energy} of these 1-quantum states, sorted by mode, e.g. {0: 1000., 1: 1500.}.
        """
        harm_states_str = [i for i in self.allstates if ',' not in i.state_label and i.state_label != ZERO_STATE_LABEL]
        harm_states = {int(i.state_label): i.energy for i in harm_states_str if int(i.state_label) in self.harmonic_osc_states_labels}

        return dict(sorted(harm_states.items()))
    
    def get_state_by_label(self, state_label: str) -> VibState | None:
        if state_label in self.allstates_map:
            return self.allstates_map.get(state_label)
        else:
            raise ValueError(f'Requested state label - {state_label} - is not in VibStatesData')


def make_state_label(modes: Sequence[int | str]) -> str:
    """
    (10, 2) -> '2,10';  ('1', '0') -> '0,1';  () -> 'zero'
    
    assert make_state_label(('10', '2')) == make_state_label([2, 10]) == '2,10'
    """
    return ','.join(str(m) for m in sorted(int(m) for m in modes)) or ZERO_STATE_LABEL


@dataclass
class VibDiff:
    """
    Represents difference between two vibrational states.
    Numerical representation that holds values, as opposed to VibDiffTerm which is symbolic.
    Handles special case of zero states (ground state) in energy_difference.
    """
    left: VibState | None
    right: VibState | None
    
    def is_zero_state(self, state: VibState) -> bool:
        """
        Check if state is a zero (ground) state.
        """
        return state.state_label == ZERO_STATE_LABEL
    
    def energy_difference(self, *, au=False) -> float:
        """
        Calculate energy difference between states in cm-1 (in Eh if au=True).
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
        
        return cls.from_quanta(vibdiff_term_symb.sl.q, vibdiff_term_symb.sr.q,  # type: ignore
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
    The data of one molecule for the evaluation: harmonic frequencies, property values and state energies.
    Built before the evaluation, then passed to it (e.g. build_contributions). Two ways to build one:
        MolSystemData.load(origin, ['polgrad', 'dipgrad', 'cff'])    reads the data from one source, e.g. a file
        MolSystemData.from_datadict(mol_props, data_dict)             from a data dict already read

    The evaluator reads three fields:
        eigenvals   the harmonic frequency of each mode in cm-1 (data key 'nc_sqrt_eigval'), e.g. {0: 2878.7,
                    1: 1820.4}. Gives the number of modes, and the frequency where a term needs one.
        mol_props   the property values, e.g. polgrad, dipgrad, cff. Each array has the mode indices first, then
                    x, y, z (PROP_INDEX_COUNTS). A property with no values has vals None; check_props raises if a
                    term needs it.
        states      one VibStatesData per state set, e.g. {'harmonic': ..., 'anharmonic': ...}, energies in cm-1.
                    Gives the energy differences where a term needs them.
                    The data holds no choice: the evaluation takes one (states_choice) and reads
                    states[states_choice]. The choice must follow the vib analysis regime: GVPT2, VPT2 ->
                    'anharmonic'; harmonic -> 'harmonic'. The two differ, e.g. in 272 of 340 formaldehyde
                    coefficients of the 14 EVV terms.
    The evaluator does not read the other fields: name (a label), eigenvecs (the normal modes; load does not read
    them), natoms, geo, geo_extra, linear, data_origin (where the data comes from).
    """
    name: str
    eigenvals: dict[int, float]
    eigenvecs: np.ndarray | None
    mol_props: 'MolPropsCollection'
    states: dict[str, 'VibStatesData']
    natoms: int = 0
    geo: Any = None
    geo_extra: Any = None
    linear: bool = False
    data_origin: DataOriginInfo | None = None

    @property
    def data_filled(self) -> bool:
        """Check if all properties have values."""
        return self.mol_props.is_filled

    def check_states_choice(self, states_choice: str) -> None:
        """
        raises ValueError if the data has no state set states_choice,
            e.g. 'anharmonic' for data read from a file without anharmonic states, or a regime name like 'GVPT2'
        """
        if states_choice not in self.states:
            raise ValueError(f'states_choice {states_choice!r}: the data has the state sets {sorted(self.states)}')

    def check_props(self, prop_names: Iterable[str]) -> None:
        """
        raises ValueError if a named property has no values here: not in mol_props, or there with vals None,
            e.g. 'polhess' for data loaded for ['polgrad', 'dipgrad', 'cff'] only.
        prop_names: what the terms need, e.g. from CompiledTerm.prop_names
        """
        have = [p.trivial_name for p in self.mol_props if p.vals is not None]
        missing = [name for name in dict.fromkeys(prop_names) if name not in have]
        if missing:
            raise ValueError(f'no values for {missing}: the data has values for {have}')


    @classmethod
    def from_datadict(cls,
                      mol_props: 'MolPropsCollection',  # to be filled with data
                      data_dict: dict,                  # data obtained from obtainer
                      *,
                      name: str = 'from_data_dict'):
        """
        Build a MolSystemData from a data dict already read (load reads one and calls this).

        data_dict: dict - {data_name: values}
        states: one VibStatesData per state list in data_dict, by states choice:
            'harmonic_states' -> states['harmonic'], 'anharmonic_states' -> states['anharmonic'].
            'anharmonic_states' may be missing: no entry (check_states_choice raises if the evaluation asks for it).
            Both sets get the modes of the harmonic fundamentals as harmonic_osc_states_labels.
        raises ValueError if data_dict does not fit what the evaluator reads (check_data_dict), e.g. without
            'nc_sqrt_eigval'. A property of mol_props that data_dict lacks is no error here: its vals stay None
            (check_props).
        """
        check_data_dict(data_dict, mol_props.names())

        mol_props = mol_props.empty_copy()
        mol_props.fill_from(data_dict)

        harm_states, anharm_states = _make_hq_states_from_datadict(data_dict)
        labels = tuple(int(i.state_label.split(',')[0]) for i in harm_states if len(i.state_label.split(','))==1)
        states = {choice: VibStatesData(allstates=allstates, harmonic_osc_states_labels=labels)
                  for choice, allstates in (('harmonic', harm_states), ('anharmonic', anharm_states))
                  if allstates}

        geo = data_dict.get('geo', None)
        natoms = len(geo) if geo is not None else 0

        return cls(name=name,
                   eigenvals=data_dict['nc_sqrt_eigval'],
                   eigenvecs=data_dict.get('normal_modes', None),
                   mol_props=mol_props,
                   natoms=natoms,
                   states=states,
                   geo=geo,
                   geo_extra=data_dict.get('geo_extra', None),
                   linear=data_dict.get('linear', False),
                   data_origin=data_dict.get('data_origin', None))

    @classmethod
    def load(cls, origin: DataOriginInfo, prop_names: Iterable[str], *,
             obtainer: Callable[[dict], dict] | None = None,
             name: str = 'loaded') -> 'MolSystemData':
        """
        Read the named properties from one source, in one obtainer call, and build a MolSystemData.
        Also reads what every evaluation needs: the harmonic frequencies (nc_sqrt_eigval) and both state lists.
            e.g. MolSystemData.load(origin, ['polgrad', 'dipgrad', 'cff'])
                 -> request {'polgrad', 'dipgrad', 'cff', 'nc_sqrt_eigval', 'harmonic_states', 'anharmonic_states'}
        Nothing else is requested: the evaluator does not use normal_modes, atoms or the geometry, and the reader
        fails on normal_modes for some files.

        prop_names  trivial names, e.g. pipeline.prop_names(terms)
        obtainer    request {name: DataOriginInfo} -> data dict; default wilson_data_obtainer.
                    A test can pass a function that returns a dict, with no file.
        raises ValueError if the data does not fit what the evaluator reads (check_data_dict), or if the source
            has no values for a requested property. Both messages name the source; so do check_data_dict's warnings.
        """
        if obtainer is None:
            # imported here, not at the top: wilson_data_obtainer imports DataOriginInfo from this module
            from wilson_suite.wilson_utils.wilson_data_obtainer import (
                wilson_data_obtainer,
            )
            obtainer = wilson_data_obtainer

        names = list(dict.fromkeys(prop_names))        # each name once, in the given order
        request = dict.fromkeys([*names, *EVALUATION_DATA_KEYS], origin)
        data_dict = obtainer(request)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                molsys = cls.from_datadict(MolPropsCollection([MolecularProperty(trivial_name=n) for n in names]),
                                           data_dict, name=name)
            except ValueError as err:
                raise ValueError(f'the data from {origin} does not fit: {err}') from err
        for w in caught:            # again, with the source, at the line that called load
            warnings.warn(f'the data from {origin}: {w.message}', w.category, stacklevel=2)

        # fill_from leaves vals None for a name the source lacks. The same check as build_contributions makes,
        # but here it means the source failed to return a property: it fails at the reading, naming the source
        try:
            molsys.check_props(names)
        except ValueError as err:
            raise ValueError(f'the source did not return every requested property, {origin}: {err}') from err
        return molsys


def check_data_dict(data_dict: dict, prop_names: Iterable[str]) -> None:
    """
    raises ValueError if data_dict does not fit what the evaluator reads, with every problem found:
        names         'nc_sqrt_eigval' (harmonic frequencies, cm-1) and 'harmonic_states' are there.
                      'anharmonic_states' may be missing: MolSystemData.check_states_choice raises if it is asked for.
        modes         nc_sqrt_eigval has the keys 0 .. n-1, e.g. {0: 2878.7, 1: 1820.4}: n modes
        shapes        each property: mode indices first, then x, y, z (PROP_INDEX_COUNTS),
                      e.g. n = 6: polhess (6, 6, 3, 3), cff (6, 6, 6), dipgrad (6, 3)
        full tensors  each property is symmetric in its mode indices, e.g. cff[0, 2, 2] == cff[2, 0, 2]
                      (SYMMETRY_RTOL, SYMMETRY_ATOL)
        fundamentals  'harmonic_states' has one key per mode, e.g. ('0',), ('1',), ... ('5',) for n = 6
        ground state  a state list may hold the ground state, key (), but only at 0 cm-1: energies count from it
    prop_names  the properties to check, e.g. ['polgrad', 'cff']. A name data_dict lacks is no error here
                (MolSystemData.check_props); a name not in PROP_INDEX_COUNTS (e.g. 'coriolis') gets no shape check.
    warns (UserWarning), no error: an excited state at or below 0 cm-1, the ground state, in any state list,
        e.g. ACDM's anharmonic fundamental of mode 41 at -136.1 cm-1.
    Not checked (no check is possible from the values): units, coordinates.
    """
    problems = []

    for key in ('harmonic_states', 'anharmonic_states'):
        excited = {state: energy for state, energy in (data_dict.get(key) or {}).items() if state != ()}
        low = sorted((energy, state) for state, energy in excited.items() if energy <= 0.)
        if low:
            lowest = ', '.join(f'{state} {energy:.1f}' for energy, state in low[:3])
            warnings.warn(f'{key}: {len(low)} of {len(excited)} state energies are at or below 0 cm-1, '
                          f'the ground state; lowest: {lowest}', stacklevel=3)
        ground = (data_dict.get(key) or {}).get(())
        if ground is not None and ground != 0.:
            problems.append(f"'{key}': the ground state () at {ground} cm-1; it must be at 0, as the energies "
                            f"count from it")

    freqs = data_dict.get('nc_sqrt_eigval')
    n = None
    if freqs is None:
        problems.append("no 'nc_sqrt_eigval' (the harmonic frequencies)")
    elif not isinstance(freqs, dict) or not freqs or set(freqs) != set(range(len(freqs))):
        keys = list(freqs) if isinstance(freqs, dict) else type(freqs).__name__
        problems.append(f"'nc_sqrt_eigval': needs a dict with the modes 0 .. n-1 as keys, got {keys}")
    else:
        n = len(freqs)

    harmonic = data_dict.get('harmonic_states')
    if harmonic is None:
        problems.append("no 'harmonic_states'")
    elif n is not None:
        have = {int(key[0]) for key in harmonic if len(key) == 1}
        missing = [m for m in range(n) if m not in have]
        if missing:
            problems.append(f"'harmonic_states' lacks the fundamentals of modes {missing}, e.g. the key "
                            f"('{missing[0]}',)")

    for name in dict.fromkeys(prop_names):
        if n is None or data_dict.get(name) is None or name not in PROP_INDEX_COUNTS:
            continue
        n_modes, n_xyz = PROP_INDEX_COUNTS[name]
        vals = np.asarray(data_dict[name])
        expected = (n,) * n_modes + (3,) * n_xyz
        if vals.shape != expected:
            problems.append(f'{name}: shape {vals.shape}, expected {expected} (mode indices first, then x, y, z)')
            continue
        for i in range(n_modes - 1):             # symmetric under every swap of neighbours: under every order
            swapped = np.swapaxes(vals, i, i + 1)
            if not np.allclose(vals, swapped, rtol=SYMMETRY_RTOL, atol=SYMMETRY_ATOL):
                idx = np.unravel_index(np.argmax(np.abs(vals - swapped)), vals.shape)
                other = list(idx)
                other[i], other[i + 1] = other[i + 1], other[i]
                problems.append(f'{name}: not symmetric in its mode indices, {name}{list(map(int, idx))} = '
                                f'{vals[idx]:.6g}, {name}{list(map(int, other))} = {vals[tuple(other)]:.6g}')
                break

    if problems:
        raise ValueError('the data dict does not fit what the evaluator reads:\n  - ' + '\n  - '.join(problems))


def _make_hq_states_from_datadict(data_dict: dict[str, Any]) -> tuple[tuple, tuple]:
    """
    Construct tuple['VibState'] from data_dict.
    The ground state key () is left out: VibStatesData adds the ground state itself, with energy 0.
    """

    harm_states = ()
    if 'harmonic_states' in data_dict:
        states_dict: dict = data_dict['harmonic_states']

        for state, energy in states_dict.items():
            if state != ():
                harm_states += (VibState(harm_quanta_coeffs={state: 1.0}, energy=energy,
                                         state_label=make_state_label(state)),)

    anharm_states = ()
    if 'anharmonic_states' in data_dict:
        states_dict: dict = data_dict['anharmonic_states']

        for state, energy in states_dict.items():
            if state != ():
                anharm_states += (VibState(harm_quanta_coeffs={state: 1.0}, energy=energy,
                                           state_label=make_state_label(state)),)

    return harm_states, anharm_states


@dataclass
class MolecularProperty:
    """
    Class to represent a molecular (energy derivative or similar) property

    holds name (key), values; 
        does not hold provenance info but would be retrieved from MolSystemData with this info

    ----
    trivial_name: String: Trivial name For simplified reference
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
        """
        Allow `name in coll` syntax. A MolecularProperty counts only as that very object: == would compare the
        values, and two numpy arrays there raise ValueError.
        """
        if isinstance(item, MolecularProperty):
            return any(p is item for p in self.properties)
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
            # resets values, because props in collection should be from the same source
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
