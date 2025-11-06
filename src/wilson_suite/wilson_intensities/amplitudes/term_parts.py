from wilson_suite.wilson_derive.abstractions import ResonanceCondition, HarmOscStateSymbolic, PolProp, VibDiffTerm
from dataclasses import dataclass, field
from wilson_suite.wilson_utils.prop_trivname import prop_trivname
from ...wilson_main.abstractions import MolecularProperty, MolPropsCollection
from ...wilson_derive.abstractions import VibPerturbedTerm
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..amplitudes.vibene_differences import VibDiffCache

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
    props: list[PolProp]

    def __post_init__(self):
        self.props = tuple(self.props)

    def __iter__(self):
        for prop in self.props:
            yield prop

    def __hash__(self):
        # return hash(tuple([tuple(self.get_cart_axes()), self.get_total_difforder()]))
        return hash(tuple([tuple(self.get_cart_axes()), tuple(self.get_mode_indices())]))
    
    def __eq__(self, other):
        """
        Now depends on comparison of PolProp instances.
        Now PolProp instances are considered equal if the have the same lists of operators (ops)
            (further, equality of QOperator instances) and same differentiation order (dord)
        """
        if isinstance(other, PropsCollection):
            return all([p in other.props for p in self.props])
        return False
    
    def get_averaged_props(self):
        return PropsCollection(props=[p for p in self.props if p.ops])
    def get_non_averaged_props(self):
        return PropsCollection(props=[p for p in self.props if not p.ops])
    
    def get_cart_axes(self):
        return [op.o for p in self.props for op in p.ops]
    def get_mode_indices(self):
        groups = [p.inds if p.inds is not None else [] for p in self.props]
        return [idx for p_inds in groups for idx in p_inds]
    
    def get_mode_indices_grouped(self):
        return [p.inds if p.inds is not None else [] for p in self.props]
    
    def get_mode_indices_group_template(self):
        return [len(p.inds) if p.inds is not None else [] for p in self.props]
    
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
    freqterms: list[VibDiffTerm]
    
    def __post_init__(self):
        self.freqterms = tuple(self.freqterms)

    def __iter__(self):
        for freqt in self.freqterms:
            yield freqt

    def __hash__(self):
        return hash(self.freqterms)
    
    def __eq__(self, other):
        """
        Now depends on comparison of PolProp instances.
        Now PolProp instances are considered equal if the have the same lists of operators (ops)
            (further, equality of QOperator instances) and same differentiation order (dord)
        """
        if isinstance(other, FreqTermsCollection):
            return all([ft in other.freqterms for ft in self.freqterms])
        return False

    def get_vibenedenom(self):
        return FreqTermsCollection(freqterms=[ft for ft in self.freqterms if not ft.is_pert_wf_diff])
    
    def get_pert_wf_diff(self):
        return FreqTermsCollection(freqterms=[ft for ft in self.freqterms if ft.is_pert_wf_diff])
    
    def get_num_indices_vibenedenom(self):
        """
        """
        # these vibdiffterms have only sl, sr is zero
        return tuple(sorted(set(i for vd in self.get_vibenedenom() for i in vd.sl.q)))

@dataclass
class ResonanceMotif:
    """
    Collects ResonanceCondition instances into a "resonance motif"
        get_vibdiffs
        get_freq_axes
    """
    resonance_conditions: list[ResonanceCondition]
    
    def __iter__(self):
        for condition in self.resonance_conditions:
            yield condition

    def __eq__(self, other):
        if isinstance(other, ResonanceMotif):
            return self._tuplify() == other._tuplify()
        return False
    def __hash__(self):
        return hash(self._tuplify())
    
    def _tuplify(self):
        conditions = []
        for cond in self.resonance_conditions:
            new_pf = tuple(cond.pf)
            new_diff = tuple([tuple(cond.diff.sl.q), tuple(cond.diff.sr.q)])

            conditions.append(tuple([new_diff, new_pf]))
        return tuple(conditions)
    
    @classmethod
    def from_tuples(cls, tupleOfTuples):
        r_conditions = []
        for rc_tuple in tupleOfTuples:
            rc = ResonanceCondition(diff=VibDiffTerm(sl=HarmOscStateSymbolic(q=rc_tuple[0][0]),
                                                     sr=HarmOscStateSymbolic(q=rc_tuple[0][1])), pf=rc_tuple[1])
            r_conditions.append(rc)
        return cls(r_conditions)

    def __repr__(self):
        return f'{self.resonance_conditions}'
    
    def __len__(self):
        """
        Returns the number of elements in the container.
        """
        return len(self.resonance_conditions)
    
    @property
    def resonance_location_class(self, total_num_axes):
        return total_num_axes - len(self.resonance_conditions)
    
    def get_vibdiffs(self):
        # return {i: tuple([tuple(cond.diff.sl.q), tuple(cond.diff.sr.q)]) for i, cond in enumerate(self.resonance_conditions)}
        return {i: cond.diff for i, cond in enumerate(self.resonance_conditions)}
    def get_freq_axes(self):
        return {i: tuple(cond.pf) for i, cond in enumerate(self.resonance_conditions)}
    
    def get_max_different_freq_axes(self):
       return set([i.strip('-') for cond in self.resonance_conditions for i in cond.pf])
    
    def get_nm_indices(self):
        return set([label for cond in self.resonance_conditions for i in cond.diff for label in i.q])

# class ResonanceCondValue:

@dataclass(frozen=True)
class EvalVibPerturbedTerm:
    """
    properties - both averaged and non-averaged together generally
    """
    properties: PropsCollection
    resonance_motif: ResonanceMotif


@dataclass(frozen=True)
class EvalTermCollection:
    """
    """
    terms: list[EvalVibPerturbedTerm]

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
        return [i for i in list(self._parameters.keys()) if i!='zero']
    
    def indices(self):
        return [i for i in list(self._parameters.values()) if i!='zero']

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
        repr_d = {k:v for k,v in self._parameters.items() if k!='zero'}
        return f"{self.__class__.__name__}({repr_d})"

    def __eq__(self, other):
        if isinstance(other, ParameterSet):
            return self._parameters == other._parameters
        return False
    
    def to_dict(self):
        return self._parameters

    @classmethod
    def from_dict(cls, parameters):
        return cls(parameters)

from wilson_suite.wilson_main.abstractions import VibState
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
        tmp_allstates.append(VibState(harm_quanta_coeffs={}, state_label='zero', energy=0.))
        self.allstates = tuple(tmp_allstates)
        
        self.allenergies_map = {i.state_label: i.energy for i in self.allstates}
        self.allstates_map = {i.state_label: i for i in self.allstates}
        self._storage = dict()

    def _fill_storage(self):
        for vlabel_a, energy_a in self.allenergies_map:
            for vlabel_b, energy_b in self.allenergies_map:
                self._storage[(vlabel_a, vlabel_b)] = convNu2Ene(energy_a - energy_b)


    def get_harmonic_osc_states(self):
        """
        i.state_label - TODO: make a convention, rules how to describe vibstates
        now i.state_label is str
        """
        harm_states = {int(i.state_label): i.energy for i in self.allstates if i.harmonic_WF}
        return dict(sorted(harm_states.items()))
    
    def get_state_by_label(self, state_label):
        if state_label in self.allstates_map:
            return self.allstates_map.get(state_label)
        else:
            raise ValueError(f'Requested state label - {state_label} - is not in VibStatesData')
    
    def get_energy_by_label(self, state_label):
        if state_label in self.allstates_map:
            return self.allenergies_map.get(state_label)
        else:
            raise ValueError(f'Requested state label - {state_label} - is not in VibStatesData')

@dataclass()
class EvaluationDataAndConfigs:
    # props_data: list['MolecularProperty']
    props_data: MolPropsCollection
    vibstates_data: 'VibStatesData'
    number_of_nmodes: int
    nm_inds_choices: list[int]
    vibdiff_cache: 'VibDiffCache' = None
    avrg_tensors: dict = None
    avrg_expr_tensor_mapping: dict = None
    vibenedenoms_tensors: dict = None
    pulse_polarization_vector: list = None


@dataclass(frozen=True)
class TermParametersChoice:
    """
    term_key - hash(VibPerturbedTerm)
    now those terms would have the same res_motif
    """
    term_keys: tuple[int]
    states_parameters: tuple[ParameterSet]

    def __hash__(self) -> int:
        return hash((self.term_keys, self.states_parameters))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TermParametersChoice):
            return False
        return (self.term_keys == other.term_keys and 
                self.states_parameters == other.states_parameters)

@dataclass
class SpectralFeature:
    location: 'GeometricObject'
    term_contributions: tuple[TermParametersChoice] # grouped by res_motif
    lineshape_parameter: dict = None
    amplitude_coeff: float = None

    def __hash__(self) -> int:
        return hash((self.location, self.term_contributions))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, SpectralFeature):
            return False
        return (self.location == other.location and 
                self.term_contributions == other.term_contributions)

    def union(self, other: 'SpectralFeature'):
        if self.location == other.location:
            return SpectralFeature(location=self.location, 
                                   term_contributions=self.term_contributions+other.term_contributions,
                                   amplitude_coeff=self.amplitude_coeff+other.amplitude_coeff)
        else:
            raise ValueError('Cannot make a union of SpectralFeatures is location is not the same')

from typing import Union, Literal, Tuple, Optional, List

# Type aliases
CoordValue = Union[float, Literal['all']]
Coordinates = Tuple[Tuple[str, CoordValue], ...]

class GeometricObject:
    """
    Represents geometric objects in N-dimensional space that are hashable.
    
    Examples:
        Point:      (('A', 1864.0), ('B', 900.0))
        Line:       (('A', 1864.0), ('B', 'all'))
        Plane:      (('A', 'all'), ('B', 'all'), ('C', 1200.0))
    """
    def __init__(self, coord_dict: dict[str, CoordValue]):
        # Convert dict to sorted tuple of tuples for consistent hashing
        self.coordinates: Coordinates = tuple(sorted(coord_dict.items()))
    
    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self.coordinates)
    
    @property
    def values(self) -> tuple[CoordValue, ...]:
        return tuple(v for _, v in self.coordinates)
    
    @property
    def dimensionality(self) -> int:
        """Returns dimensionality of the object (0=point, 1=line, 2=plane, etc)"""
        return sum(1 for v in self.values if v == 'all')
    
    def __getitem__(self, axis: str) -> CoordValue:
        for k, v in self.coordinates:
            if k == axis:
                return v
        raise KeyError(f"Axis {axis} not found")
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, GeometricObject):
            return NotImplemented
        return self.coordinates == other.coordinates
    
    def __hash__(self) -> int:
        return hash(self.coordinates)
    
    def is_point(self) -> bool:
        return self.dimensionality == 0
    
    def is_line(self) -> bool:
        return self.dimensionality == 1
    
    def is_plane(self) -> bool:
        return self.dimensionality == 2
    
    def __repr__(self) -> str:
        """
        Returns a string representation showing type and coordinates.
        
        Examples:
            Point(A=1864.0, B=900.0)
            Line(A=1864.0, B=all)
            Plane(A=all, B=all, C=1200.0)
        """
        type_name = "Point" if self.is_point() else "Line" if self.is_line() else "Plane" if self.is_plane() else "Object"
        coords = ", ".join(f"{k}={v}" for k, v in self.coordinates)
        return f"{type_name}({coords})"


# Source - https://stackoverflow.com/questions/60590442/abstract-dataclass-without-abstract-methods-in-python-prohibit-instantiation
# Posted by Jundiaius
# Retrieved 11/5/2025, License - CC-BY-SA 4.0

# @dataclass
# class SpectralWindow(ABC): 
#     def __new__(cls, *args, **kwargs): 
#         if cls == SpectralWindow or cls.__bases__[0] == SpectralWindow: 
#             raise TypeError("Cannot instantiate abstract class.") 
#         return super().__new__(cls)

# @dataclass
# class RectanglelWindow(SpectralWindow):
    

# -------------------------------------------------------

from dataclasses import dataclass, field
# from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union
import numpy as np

@dataclass
class SpectralWindow():
    """N-dimensional rectangular domain."""



@dataclass
class RectangularDomain():
    """N-dimensional rectangular domain."""
    shape: Tuple[int, ...]
    # bounds: Optional[Tuple[Tuple[float, float], ...]] = None
    bounds: SpectralWindow = None
    labels: Optional[Tuple[str, ...]] = None
    full_features: List['SpectralFeature'] = field(default_factory=list)

    def __post_init__(self):
        from wilson_suite.wilson_utils.common_labels import cap_alpha_labels

        # --- bounds ---
        if self.bounds is None:
            self.bounds = tuple((0.0, float(n)) for n in self.shape)
        elif len(self.bounds) != len(self.shape):
            raise ValueError("bounds must match shape dimensionality")

        # --- labels ---
        if self.labels is None:
            if len(self.shape) > len(cap_alpha_labels):
                raise ValueError(f"Not enough predefined labels for {len(self.shape)} dimensions.")
            self.labels = tuple(cap_alpha_labels[:len(self.shape)])
        elif len(self.labels) != len(self.shape):
            raise ValueError("labels must match shape dimensionality")

        # --- label-index map ---
        self._label_to_index = {label: i for i, label in enumerate(self.labels)}
    
    @classmethod
    def from_features(cls, features: List['SpectralFeature'], padding: float = 0.0):
        """Create a rectangular window that bounds given features."""
        if not features:
            raise ValueError("Feature list cannot be empty")

        # Collect all numeric coords per axis
        axis_vals: dict[str, list[float]] = {}
        for f in features:
            for axis, val in f.location.coordinates:
                if val != 'all':
                    axis_vals.setdefault(axis, []).append(float(val))

        # Determine bounds and shape
        bounds = []
        for axis, vals in sorted(axis_vals.items()):
            min_val, max_val = min(vals) - padding, max(vals) + padding
            bounds.append((min_val, max_val))
        shape = tuple(len(axis_vals[a]) for a in sorted(axis_vals))

        # Construct the window and assign features
        window = cls(shape=shape, bounds=tuple(bounds))
        window.add_full_features(features)
        return window
    
    # -----------------------
    # Basic properties
    # -----------------------
    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def extent(self) -> Tuple[float, ...]:
        return tuple(max_val - min_val for min_val, max_val in self.bounds)

    @property
    def volume(self) -> float:
        return np.prod(self.extent)

    @property
    def center(self) -> Tuple[float, ...]:
        return tuple((min_val + max_val) / 2 for min_val, max_val in self.bounds)

    # -----------------------
    # Axis utilities
    # -----------------------
    def axis_index(self, key: Union[int, str]) -> int:
        return self._label_to_index[key] if isinstance(key, str) else key

    def axis_bounds(self, key: Union[int, str]) -> Tuple[float, float]:
        return self.bounds[self.axis_index(key)]

    def axis_extent(self, key: Union[int, str]) -> float:
        min_val, max_val = self.axis_bounds(key)
        return max_val - min_val

    def axis_coords(self, key: Union[int, str]) -> np.ndarray:
        i = self.axis_index(key)
        n = self.shape[i]
        min_val, max_val = self.bounds[i]
        return np.linspace(min_val, max_val, n, endpoint=False)

    def meshgrids(self) -> Tuple[np.ndarray, ...]:
        coords_1d = [self.axis_coords(i) for i in range(self.ndim)]
        return np.meshgrid(*coords_1d, indexing='ij')

    def grid_flat(self) -> np.ndarray:
        grids = self.meshgrids()
        return np.stack([g.ravel() for g in grids], axis=1)

    def contains(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        if points.shape[-1] != self.ndim:
            raise ValueError(f"Points must have {self.ndim} coordinates in last dimension.")
        
        inside = np.ones(points.shape[:-1], dtype=bool)
        
        # iterate over dimensions(axes) of the domain
        for i, (min_val, max_val) in enumerate(self.bounds):
            # in-place "addition" of arrays with booleans - "addition" with AND operator
            inside &= (points[..., i] >= min_val) & (points[..., i] < max_val)
        return inside

    # -----------------------
    # Abstract interface
    # -----------------------
    def generate(self) -> np.ndarray:
        return np.ones(self.shape)

    def __call__(self) -> np.ndarray:
        return self.generate()

    def __getitem__(self, key):
        """Access axis coordinates via label or tuple of labels."""
        if isinstance(key, str):
            return self.axis_coords(key)
        elif isinstance(key, (list, tuple)):
            return tuple(self.axis_coords(k) for k in key)
        raise TypeError("Key must be a label or sequence of labels.")

    # -----------------------
    # Feature handling (unchanged)
    # -----------------------
    def add_full_feature(self, feature: 'SpectralFeature') -> None:
        self.full_features.append(feature)

    def add_full_features(self, features: list['SpectralFeature']) -> None:
        self.full_features.extend(features)

    def features_in_bounds(self) -> list['SpectralFeature']:
        filtered = []
        for f in self.full_features:
            coords = f.location
            inside = True
            for label, (min_val, max_val) in zip(self.labels, self.bounds):
                val = coords[label]
                if val == "all":
                    continue
                if not (min_val <= val < max_val):
                    inside = False
                    break
            if inside:
                filtered.append(f)
        return filtered

