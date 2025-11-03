from wilson_suite.wilson_derive.abstractions import ResonanceCondition, HarmOscStateSymbolic, PolProp, VibDiffTerm
from dataclasses import dataclass
from wilson_suite.wilson_utils.prop_trivname import prop_trivname
from ...wilson_main.abstractions import MolecularProperty, MolPropsCollection
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
        return f"{self.__class__.__name__}({self._parameters})"

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