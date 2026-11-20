import copy
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from wilson_suite.wilson_utils.prop_trivname import prop_trivname
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene

if TYPE_CHECKING:
    from wilson_suite.wilson_derive.abstractions import (
        VibDiffTerm,
    )

# What every evaluation reads besides the properties: from_output_file requests these from every source.
# When wilson computes the anharmonic states, from_output_file requests VPT2_INPUT_KEYS instead of 'anharmonic_states'.
FREQUENCY_AND_STATE_KEYS = ('nc_sqrt_eigval', 'harmonic_states', 'anharmonic_states')

# What MolSystemData.with_wilson_anharmonic_states reads from mol_props, besides the harmonic frequencies:
# cubic and quartic force constants in cm-1, rotational constants, Coriolis constants.
VPT2_INPUT_KEYS = ('cff_rc', 'qff_rc', 'B', 'coriolis')

# Array layout of each property: one letter per index, in order. 'm': a normal mode (length n, the number of
# modes); 'x': x, y or z (length 3). Every property name the data can have is here.
#   derivatives by normal modes and electric field: mode indices first, e.g. 'polhess' -> 'mmxx', shape (n, n, 3, 3)
#   force constants in cm-1, as VPT2 takes them: 'cff_rc' -> 'mmm', 'qff_rc' -> 'mmmm'
#   rotational constants: 'B' -> 'x';  Coriolis constants: 'coriolis' -> 'xmm', shape (3, n, n)
PROP_LAYOUT = {**{prop_trivname(ord_geo=g, ord_el=e): 'm' * g + 'x' * e for g in range(7) for e in range(7)},
               'cff_rc': 'mmm', 'qff_rc': 'mmmm', 'B': 'x', 'coriolis': 'xmm'}

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
    VibDiff computes the energy differences.
    The energies count from the ground state ('zero'). Checked for every state set, however it is made:
        a ground state, if given, must be at 0 cm-1 (ValueError); without one, it is added at 0
        an excited state at or below 0 cm-1 gives a warning (UserWarning)

    regime: str - the vibrational analysis regime, e.g. 'harmonic', 'GVPT2', 'VPT2';
        'unknown' for anharmonic states read from a file: the file's regime is not read
    computed_by: str - 'file': read from the data (the program is in MolSystemData.data_origin);
        'wilson': computed by wilson (MolSystemData.with_wilson_anharmonic_states)
    """
    allstates: Sequence[VibState]
    regime: str
    computed_by: str
    harmonic_osc_states_labels: Sequence[int] = ()

    def __post_init__(self):
        name = f'the states (regime {self.regime!r}, computed by {self.computed_by})'
        off_zero = [s.energy for s in self.allstates if s.state_label == ZERO_STATE_LABEL and s.energy != 0.]
        if off_zero:
            raise ValueError(f'{name}: the ground state ({ZERO_STATE_LABEL!r}) at {off_zero[0]} cm-1; it must be '
                             f'at 0, as the energies count from it')
        excited = [s for s in self.allstates if s.state_label != ZERO_STATE_LABEL]
        low = sorted((s.energy, s.state_label) for s in excited if s.energy <= 0.)
        if low:
            lowest = ', '.join(f'{label} {energy:.1f}' for energy, label in low[:3])
            warnings.warn(f'{name}: {len(low)} of {len(excited)} state energies are at or below 0 cm-1, '
                          f'the ground state; lowest: {lowest}', stacklevel=3)      # at the line that built the set

        self.allstates = (*excited, VibState(harm_quanta_coeffs={}, state_label=ZERO_STATE_LABEL, energy=0.))
        self.allstates_map = {i.state_label: i for i in self.allstates}


    def get_harmonic_osc_states(self) -> dict[int, float]:
        """
        Get harmonic oscillator states from allstates, filtering by harmonic_osc_states_labels.
        Returns {mode: energy} of these 1-quantum states, sorted by mode, e.g. {0: 1000., 1: 1500.}.
        """
        harm_states_str = [i for i in self.allstates if ',' not in i.state_label and i.state_label != ZERO_STATE_LABEL]
        harm_states = {int(i.state_label): i.energy for i in harm_states_str if int(i.state_label) in self.harmonic_osc_states_labels}

        return dict(sorted(harm_states.items()))
    
    def get_state_by_label(self, state_label: str) -> VibState:
        if state_label in self.allstates_map:
            return self.allstates_map[state_label]
        else:
            raise ValueError(f'Requested state label - {state_label} - is not in VibStatesData')


def make_state_label(modes: Sequence[int | str]) -> str:
    """
    (10, 2) -> '2,10';  ('1', '0') -> '0,1';  () -> 'zero'
    
    assert make_state_label(('10', '2')) == make_state_label([2, 10]) == '2,10'
    """
    return ','.join(str(m) for m in sorted(int(m) for m in modes)) or ZERO_STATE_LABEL


def check_modes_exist(modes: Iterable, n_modes: int) -> None:
    """Raises ValueError for a mode not in 0 .. n_modes-1, e.g. 6, -1 or '1' when there are 6 modes."""
    unknown = [m for m in dict.fromkeys(modes) if m not in range(n_modes)]
    if unknown:
        raise ValueError(f'modes {unknown} do not exist: the modes are 0 .. {n_modes - 1}')


@dataclass
class VibDiff:
    """
    Represents difference between two vibrational states.
    Numerical representation that holds values, as opposed to VibDiffTerm which is symbolic.
    """
    left: VibState
    right: VibState

    def energy_difference(self, *, au=False) -> float:
        """
        Energy of left minus energy of right, in cm-1 (in Eh if au=True).
        The ground state is at 0: VibStatesData holds it there.
        """
        difference = self.left.energy - self.right.energy
        return convNu2Ene(difference) if au else difference

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
        MolSystemData.from_output_file(origin, ['polgrad', 'dipgrad', 'cff'])   reads one output file
        MolSystemData.from_data_dict(mol_props, data_dict)                      from a data dict already read

    The evaluator reads three fields:
        eigenvals   the harmonic frequency of each mode in cm-1 (data key 'nc_sqrt_eigval'), e.g. {0: 2878.7,
                    1: 1820.4}. Gives the number of modes, and the frequency where a term needs one.
        mol_props   the property values, e.g. polgrad, dipgrad, cff. Each array has the index order PROP_LAYOUT
                    gives, e.g. polhess (n, n, 3, 3), coriolis (3, n, n). A property with no values has vals None;
                    MolPropsCollection.require_values raises if a term needs it.
        states      one VibStatesData per state set, e.g. {'harmonic': ..., 'anharmonic': ...}, energies in cm-1.
                    Gives the energy differences where a term needs them.
                    The data holds no choice: the evaluation takes one (states_choice) and reads
                    states[states_choice]. 'anharmonic' holds the file's states, or wilson's own after
                    from_output_file(..., wilson_regime=...) or with_wilson_anharmonic_states;
                    VibStatesData.regime says which.
                    'harmonic' and 'anharmonic' differ, e.g. in 272 of 340 formaldehyde coefficients of the
                    14 EVV terms.
    The evaluator does not read the other fields: name (a label), eigenvecs (the normal modes; from_output_file
    does not read them), natoms, geo, geo_extra, linear, data_origin (where the data comes from).
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

    def __post_init__(self):
        """
        Checks that the fields fit together, however the data is built, with every problem in one ValueError:
            eigenvals  a dict with the modes 0 .. n-1 as keys: n modes
            mol_props  each name has a layout in PROP_LAYOUT; each array with values has the shape that its layout
                       and n give, and is symmetric in its mode indices (not coriolis)
        The state sets check themselves (VibStatesData).
        """
        problems, n = [], None
        freqs = self.eigenvals
        if not isinstance(freqs, dict) or not freqs or set(freqs) != set(range(len(freqs))):
            keys = list(freqs) if isinstance(freqs, dict) else type(freqs).__name__
            problems.append(f"eigenvals (data key 'nc_sqrt_eigval'): needs a dict with the modes 0 .. n-1 as keys, "
                            f"got {keys}")
        else:
            n = len(freqs)

        for prop in self.mol_props:
            name = prop.trivial_name
            if name not in PROP_LAYOUT:
                problems.append(f'{name}: no layout in PROP_LAYOUT, so its shape cannot be checked')
            elif n is not None and prop.vals is not None:
                problem = _layout_problem(name, np.asarray(prop.vals), PROP_LAYOUT[name], n)
                if problem:
                    problems.append(problem)

        if problems:
            raise ValueError(f'MolSystemData {self.name!r} does not fit together:\n  - ' + '\n  - '.join(problems))

    @property
    def data_filled(self) -> bool:
        """Check if all properties have values."""
        return self.mol_props.is_filled

    def state_set(self, states_choice: str) -> 'VibStatesData':
        """
        The state set states[states_choice], e.g. states_choice 'anharmonic'.
        raises ValueError if the data has no such set,
            e.g. 'anharmonic' for data read from a file without anharmonic states, or a regime name like 'GVPT2'
        """
        if states_choice not in self.states:
            raise ValueError(f'states_choice {states_choice!r}: the data has the state sets {sorted(self.states)}')
        return self.states[states_choice]

    def with_wilson_anharmonic_states(self, regime: str) -> 'MolSystemData':
        """
        A copy whose states['anharmonic'] are wilson's own, from compute_vpt2_states with no modes left out.
        Reads VPT2_INPUT_KEYS from mol_props, e.g.
            MolSystemData.from_output_file(origin, ['polgrad', *VPT2_INPUT_KEYS]).with_wilson_anharmonic_states('GVPT2')
        regime: a key of VPT2_REGIMES, e.g. 'GVPT2'
        raises ValueError for another regime, or if one of VPT2_INPUT_KEYS has no values
        """
        from wilson_suite.wilson_system.anharmonic_treatment.anharmonic_analyzer import (
            compute_vpt2_states,
        )

        states, _ = compute_vpt2_states(self.mol_props, self.eigenvals, regime, exclude_modes=[])
        for s in states:                    # each state has one key, e.g. {('0', '1'): 1.0}
            s.state_label = make_state_label(next(iter(s.harm_quanta_coeffs)))
        anharmonic = VibStatesData(allstates=states, regime=regime, computed_by='wilson',
                                   harmonic_osc_states_labels=self.state_set('harmonic').harmonic_osc_states_labels)
        return replace(self, states={**self.states, 'anharmonic': anharmonic})


    @classmethod
    def from_data_dict(cls,
                       mol_props: 'MolPropsCollection',  # to be filled with data
                       data_dict: dict,                  # data read by read_from_sources
                       *,
                       name: str = 'from_data_dict'):
        """
        Build a MolSystemData from a data dict already read. Reads no file and computes nothing;
        from_output_file calls this.

        data_dict: dict - {data_name: values}
        states: one VibStatesData per state list in data_dict, by states choice:
            'harmonic_states' -> states['harmonic'], 'anharmonic_states' -> states['anharmonic'].
            Either may be missing: no entry (state_set raises if the evaluation asks for it).
            Both sets get the modes of the harmonic fundamentals as harmonic_osc_states_labels.
        raises ValueError if the result does not fit together (MolSystemData, e.g. without 'nc_sqrt_eigval'), or a
            state set breaks its rules (VibStatesData). A property of mol_props that data_dict lacks is no error
            here: its vals stay None (MolPropsCollection.require_values).
        """
        mol_props = mol_props.empty_copy()
        mol_props.fill_from(data_dict)

        harm_states, anharm_states = _make_hq_states_from_data_dict(data_dict)
        labels = tuple(int(s.state_label) for s in VibState.get_1q_states(list(harm_states)))

        states = {}
        if harm_states:
            states['harmonic'] = VibStatesData(allstates=harm_states, regime='harmonic', computed_by='file',
                                               harmonic_osc_states_labels=labels)
        if anharm_states:
            states['anharmonic'] = VibStatesData(allstates=anharm_states, regime='unknown', computed_by='file',
                                                 harmonic_osc_states_labels=labels)

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
    def from_output_file(cls, origin: DataOriginInfo, prop_names: Iterable[str], *,
                         wilson_regime: str | None = None,
                         name: str = 'loaded') -> 'MolSystemData':
        """
        Read the data from one source and build a MolSystemData.
        Reads the named properties and what every evaluation needs, in one read_from_sources call: the harmonic
        frequencies (nc_sqrt_eigval) and both state lists.
            e.g. MolSystemData.from_output_file(origin, ['polgrad', 'dipgrad', 'cff'])
                 -> request {'polgrad', 'dipgrad', 'cff', 'nc_sqrt_eigval', 'harmonic_states', 'anharmonic_states'}
        Nothing else is requested: the evaluator does not use normal_modes, atoms or the geometry.

        prop_names     trivial names, e.g. pipeline.prop_names(terms)
        wilson_regime  where the anharmonic states come from:
                       None                      the source's 'anharmonic_states'
                       a key of VPT2_REGIMES     wilson's own (with_wilson_anharmonic_states). Reads
                       ('VPT2', 'DVPT2', 'GVPT2')  VPT2_INPUT_KEYS as properties, instead of the source's
                                                 'anharmonic_states':
            e.g. MolSystemData.from_output_file(origin, ['polgrad'], wilson_regime='GVPT2')
                 -> request {'polgrad', 'cff_rc', 'qff_rc', 'B', 'coriolis', 'nc_sqrt_eigval', 'harmonic_states'},
                    then with_wilson_anharmonic_states('GVPT2')
        The result's data_origin is origin. A requested property the source lacks is no error here: its vals stay
        None (MolPropsCollection.require_values).
        raises ValueError for an unknown wilson_regime, before the source is read; if the data does not fit
            together (MolSystemData, VibStatesData); with a wilson_regime, if one of VPT2_INPUT_KEYS has no values.
            The errors and warnings after the read name the source.
        """
        from wilson_suite.wilson_system.anharmonic_treatment.vpt2 import VPT2_REGIMES
        from wilson_suite.wilson_system.data_obtainer import read_from_sources

        # before the read: compute_vpt2_states checks the regime too, but only after the read
        if wilson_regime is not None and wilson_regime not in VPT2_REGIMES:
            raise ValueError(f"wilson_regime {wilson_regime!r}: expected None (the file's states) or one of "
                             f"{', '.join(VPT2_REGIMES)}")

        # 1. what to read: with a regime, the analysis keys instead of the source's anharmonic states
        names = list(dict.fromkeys(prop_names))        # each name once, in the given order
        data_keys = list(FREQUENCY_AND_STATE_KEYS)
        if wilson_regime is not None:
            names = list(dict.fromkeys([*names, *VPT2_INPUT_KEYS]))
            data_keys.remove('anharmonic_states')

        # 2. read
        data_dict = {**read_from_sources(dict.fromkeys([*names, *data_keys], origin)), 'data_origin': origin}

        # 3. build, 4. with a regime, wilson's own anharmonic states; errors and warnings name the source
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                molsys = cls.from_data_dict(MolPropsCollection([MolecularProperty(trivial_name=n) for n in names]),
                                            data_dict, name=name)
                if wilson_regime is not None:
                    molsys = molsys.with_wilson_anharmonic_states(wilson_regime)
            except ValueError as err:
                raise ValueError(f'the data from {origin} does not fit: {err}') from err
        # again, each distinct warning once, with the source, at the line that called from_output_file
        for w in {(w.category, str(w.message)): w for w in caught}.values():
            warnings.warn(f'the data from {origin}: {w.message}', w.category, stacklevel=2)
        return molsys


def _layout_problem(name: str, vals: np.ndarray, layout: str, n: int) -> str | None:
    """
    The problem of one property array, if any: its shape, then its symmetry in the mode indices.
        e.g. n = 6: polhess 'mmxx' -> shape (6, 6, 3, 3), polhess[0, 2] == polhess[2, 0]
    Values that differ by less than SYMMETRY_RTOL, SYMMETRY_ATOL count as equal.
    """
    expected = tuple(n if index == 'm' else 3 for index in layout)
    if vals.shape != expected:
        return f'{name}: shape {vals.shape}, expected {expected} (layout {layout!r}: m a mode, x one of x, y, z)'
    # coriolis: no symmetry check. The Coriolis constant changes sign when its two modes swap, but the reader
    # gives the same sign; VPT2 squares it, so the sign changes no state. Every other layout has its modes first.
    if name == 'coriolis':
        return None
    for i in range(layout.count('m') - 1):     # symmetric under every swap of neighbours: under every order
        swapped = np.swapaxes(vals, i, i + 1)
        if not np.allclose(vals, swapped, rtol=SYMMETRY_RTOL, atol=SYMMETRY_ATOL):
            idx = np.unravel_index(np.argmax(np.abs(vals - swapped)), vals.shape)
            other = list(idx)
            other[i], other[i + 1] = other[i + 1], other[i]
            return (f'{name}: not symmetric in its mode indices, {name}{list(map(int, idx))} = '
                    f'{vals[idx]:.6g}, {name}{list(map(int, other))} = {vals[tuple(other)]:.6g}')
    return None


def _make_hq_states_from_data_dict(data_dict: dict[str, Any]) -> tuple[tuple, tuple]:
    """
    Construct tuple['VibState'] from data_dict.
    The ground state key () becomes the state 'zero'; VibStatesData checks that it is at 0.
    """

    harm_states = ()
    if data_dict.get('harmonic_states') is not None:
        states_dict: dict = data_dict['harmonic_states']

        for state, energy in states_dict.items():
            harm_states += (VibState(harm_quanta_coeffs={state: 1.0}, energy=energy,
                                     state_label=make_state_label(state)),)

    anharm_states = ()
    if data_dict.get('anharmonic_states') is not None:
        states_dict: dict = data_dict['anharmonic_states']

        for state, energy in states_dict.items():
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

    def require_values(self, names: Iterable[str]) -> None:
        """
        Raises ValueError if a named property has no values here: not in the collection, or there with vals None,
            e.g. 'polhess' in a collection read for ['polgrad', 'dipgrad', 'cff'] only.
        names: what a computation reads, e.g. the properties of the terms, or VPT2_INPUT_KEYS
        """
        have = [p.trivial_name for p in self.properties if p.vals is not None]
        missing = [name for name in dict.fromkeys(names) if name not in have]
        if missing:
            raise ValueError(f'no values for {missing}: the data has values for {have}')

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
