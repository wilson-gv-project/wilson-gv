from wilson_suite.wilson_derive.abstractions import ResonanceCondition, VibPerturbedTerm, PolProp, VibDiffTerm
from dataclasses import dataclass
from wilson_suite.wilson_utils.prop_trivname import prop_trivname
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...wilson_main.abstractions import MolecularProperty
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
        return hash(tuple([tuple(self.get_cart_axes()), self.get_total_difforder()]))
    
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
        return [idx for p in self.props for idx in p.inds]
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
        averaged = self.get_averaged_props()
        if averaged.props:
            averaged._set_attr_for_all_props('inds', None)
            return averaged
    def __repr__(self):
        full_string = [f'{prop_trivname(ord_geo=len(p.inds), ord_el=len(p.ops))}{p.inds}{[i.o for i in p.ops]}' for p in self.props]
        return ' * '.join(full_string)


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
    
    def _tuplify(self):
        conditions = []
        for cond in self.resonance_conditions:
            new_pf = tuple(cond.pf)
            new_diff = tuple([tuple(cond.diff.sl.q), tuple(cond.diff.sr.q)])

            conditions.append(tuple([new_diff, new_pf]))
        return tuple(conditions)
    
    @property
    def resonance_location_class(self, total_num_axes):
        return total_num_axes - len(self.resonance_conditions)
    
    def get_vibdiffs(self):
        return {i: tuple([tuple(cond.diff.sl.q), tuple(cond.diff.sr.q)]) for i, cond in enumerate(self.resonance_conditions)}
    def get_freq_axes(self):
        return {i: tuple(cond.pf) for i, cond in enumerate(self.resonance_conditions)}

@dataclass
class VibDiffMotif:
    """
    """
    left_len: str
    right_len: str

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
