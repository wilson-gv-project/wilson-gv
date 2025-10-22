from wilson_suite.wilson_derive.abstractions import ResonanceCondition, VibPerturbedTerm, PolProp, VibDiffTerm
from dataclasses import dataclass

@dataclass
class PropsCollection:
    props: list[PolProp]

    def __hash__(self):
        return hash(tuple([tuple(self.get_cart_axes()), self.get_total_difforder()]))
    
    def __eq__(self, other):
        if isinstance(other, PropsCollection):
            return all([p in other.props for p in self.props])
        return False
    
    def get_avegaded_props(self):
        return PropsCollection(props=[p for p in self.props if p.ops])
    def get_non_avegaded_props(self):
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

@dataclass
class ResonanceMotif:
    resonance_conditions: list[ResonanceCondition]

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
    
    def get_vibdiffs(self):
        return {i: tuple([tuple(cond.diff.sl.q), tuple(cond.diff.sr.q)]) for i, cond in enumerate(self.resonance_conditions)}
    def get_freq_axes(self):
        return {i: tuple(cond.pf) for i, cond in enumerate(self.resonance_conditions)}

@dataclass
class VibDiffMotif:
    left_len: str
    right_len: str

@dataclass
class EvalVibPerturbedTerm:
    properties: PropsCollection
    resonance_motif: ResonanceMotif