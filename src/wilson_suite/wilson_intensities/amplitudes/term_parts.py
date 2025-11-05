from wilson_suite.wilson_derive.abstractions import ResonanceCondition, HarmOscStateSymbolic, PolProp, VibDiffTerm
from dataclasses import dataclass, field
from wilson_suite.wilson_utils.prop_trivname import prop_trivname
from ...wilson_main.abstractions import MolecularProperty, MolPropsCollection
from ...wilson_derive.abstractions import VibPerturbedTerm
from typing import TYPE_CHECKING, Union, Literal, Tuple
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
    polarization: str
    number_of_nmodes: int
    vibdiff_cache: 'VibDiffCache' = None
    avrg_tensors: dict = None
    avrg_expr_tensor_mapping: dict = None
    vibenedenoms_tensors: dict = None


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

@dataclass(frozen=True)
class SpectralFeature:
    location: tuple[float, ...]
    term_contributions: list[TermParametersChoice] # grouped by res_motif

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
                                   term_contributions=self.term_contributions+other.term_contributions)
        else:
            raise ValueError('Cannot make a union of SpectralFeatures is location is not the same')

from typing import Union, Literal, Tuple

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


# @dataclass
# class SpectralWindow:

# from abc import ABC

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

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

@dataclass
class SpectralWindow(ABC):
    """Abstract base class for N-dimensional spectral windows."""
    shape: Tuple[int, ...]
    
    def __new__(cls, *args, **kwargs):
        if cls == SpectralWindow:
            raise TypeError("Cannot instantiate abstract class.")
        return super().__new__(cls)
    
    @abstractmethod
    def generate(self) -> np.ndarray:
        """Generate the N-dimensional window coefficients."""
        pass
    
    def __call__(self) -> np.ndarray:
        """Allow the window to be called like a function."""
        return self.generate()
    
    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return len(self.shape)


@dataclass
class RectangularWindow(SpectralWindow):
    """Rectangular (boxcar) window - all coefficients are 1 in N dimensions.
    
    Attributes:
        shape: Tuple defining the N-dimensional shape
        bounds: Optional tuple of (min, max) pairs for each dimension.
                If None, defaults to (0, shape[i]) for each dimension.
        labels: Optional tuple of strings labeling each axis.
                If None, defaults to capital letters A, B, C, etc.
    """
    bounds: Tuple[Tuple[float, float], ...] = None
    labels: Tuple[str, ...] = None
    _label_to_index: dict[str, int] = field(init=False, repr=False)
    
    def __post_init__(self):
        """Initialize bounds and labels if not provided."""
        # Initialize bounds
        if self.bounds is None:
            object.__setattr__(self, 'bounds', tuple((0.0, float(n)) for n in self.shape))
        elif len(self.bounds) != len(self.shape):
            raise ValueError(
                f"bounds dimensionality ({len(self.bounds)}) must match "
                f"shape dimensionality ({len(self.shape)})"
            )
            
        # Initialize labels
        if self.labels is None:
            from wilson_suite.wilson_utils.common_labels import cap_alpha_labels
            if len(self.shape) > len(cap_alpha_labels):
                raise ValueError(f"Not enough predefined labels for {len(self.shape)} dimensions.")
            object.__setattr__(self, 'labels', tuple(cap_alpha_labels[:len(self.shape)]))
        elif len(self.labels) != len(self.shape):
            raise ValueError(
                f"Labels length ({len(self.labels)}) must match shape length ({len(self.shape)})"
            )
            
        # Create label to index mapping
        object.__setattr__(self, '_label_to_index', 
                          {label: i for i, label in enumerate(self.labels)})
    
    def generate(self) -> np.ndarray:
        """Generate N-dimensional rectangular window coefficients."""
        return np.ones(self.shape)
    
    def axis_index(self, key: Union[int, str]) -> int:
        """Resolve axis index from label or integer."""
        if isinstance(key, str):
            return self._label_to_index[key]
        return key

    def axis_bounds(self, key: Union[int, str]) -> Tuple[float, float]:
        """Return bounds for given axis (by label or index)."""
        return self.bounds[self.axis_index(key)]
    
    def axis_extent(self, key: Union[int, str]) -> float:
        """Return extent for given axis."""
        min_val, max_val = self.axis_bounds(key)
        return max_val - min_val

    def axis_coords(self, key: Union[int, str]) -> np.ndarray:
        """Return coordinate values along a labeled axis."""
        i = self.axis_index(key)
        n = self.shape[i]
        min_val, max_val = self.bounds[i]
        return np.linspace(min_val, max_val, n, endpoint=False)

    def meshgrids(self) -> Tuple[np.ndarray, ...]:
        """Generate coordinate grids for each dimension."""
        coords_1d = [self.axis_coords(i) for i in range(self.ndim)]
        return np.meshgrid(*coords_1d, indexing='ij')
    
    def grid_flat(self) -> np.ndarray:
        """
        Generate flattened coordinate grid.
        flat like a stack of coordinates:
            [[0. 0.]
            [0. 1.]
            [0. 2.]
            [1. 0.]
            [1. 1.]
            [1. 2.]]
        """
        grids = self.meshgrids()
        return np.stack([g.ravel() for g in grids], axis=1)
    
    def contains(self, points: np.ndarray) -> np.ndarray:
        """Check if points are within the window bounds."""
        points = np.asarray(points)
        if points.shape[-1] != self.ndim:
            raise ValueError(
                f"Points must have {self.ndim} coordinates in last dimension, "
                f"got {points.shape[-1]}"
            )
        
        inside = np.ones(points.shape[:-1], dtype=bool)
        for i, (min_val, max_val) in enumerate(self.bounds):
            inside &= (points[..., i] >= min_val) & (points[..., i] < max_val)
        
        return inside
    
    @property
    def extent(self) -> Tuple[float, ...]:
        """Get the extent (width) of each dimension."""
        return tuple(max_val - min_val for min_val, max_val in self.bounds)
    
    @property
    def volume(self) -> float:
        """Get the total volume (area for 2D, length for 1D) of the window."""
        return np.prod(self.extent)
    
    @property
    def center(self) -> Tuple[float, ...]:
        """Get the center coordinates of the window."""
        return tuple((min_val + max_val) / 2 for min_val, max_val in self.bounds)
    
    def __getitem__(self, key):
        """Allow accessing axis coordinates using labels or sequences of labels."""
        if isinstance(key, str):
            return self.axis_coords(key)
        elif isinstance(key, (list, tuple)):
            return tuple(self.axis_coords(k) for k in key)
        raise TypeError("Key must be a label or sequence of labels.")
