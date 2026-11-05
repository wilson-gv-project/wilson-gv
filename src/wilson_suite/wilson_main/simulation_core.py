import copy
from dataclasses import dataclass

from wilson_system.system_data import MolPropsCollection, MolecularProperty
from ..wilson_experiment.experiment_abstractions import VibExperiment
from ..wilson_derive.derive import ResponseTermsCollection
import numpy as np

@dataclass
class SimulationCore():

    experiment: VibExperiment
    total_anharm_limit: int
    mech_anharm_limit: int
    el_anharm_limit: int

    def __post_init__(self):

        terms_by_order_comp_magn = {}

        props_set = set()

        max_state_lvl = 0
        pair_quanta = set()

        for scan_elem in np.ndindex(self.experiment.field_analyses.shape):
            int_seq = self.experiment.field_analyses[scan_elem].int_sequences_by_order_comp_pol_magn

            for order in int_seq:
                if not order in terms_by_order_comp_magn:
                    terms_by_order_comp_magn[order] = {}

                for comp in int_seq[order]:
                    if not comp in terms_by_order_comp_magn[order]:
                        terms_by_order_comp_magn[order][comp] = {}

                    for pol in int_seq[order][comp]:
                        for magn in int_seq[order][comp][pol]:
                            if not magn in terms_by_order_comp_magn[order][comp]:

                                new_collection = ResponseTermsCollection(order = order,
                                                            total_anharm_limit=self.total_anharm_limit,
                                                            el_anharm_limit=self.el_anharm_limit,
                                                            mech_anharm_limit=self.mech_anharm_limit,
                                                            comp_pattern=comp,
                                                            magn_conds=magn)

                                terms_by_order_comp_magn[order][comp][magn] = copy.deepcopy(new_collection)

                                props_set = props_set.union(set(new_collection.props.names()))

                                if new_collection.max_state_lvl > max_state_lvl:
                                    max_state_lvl = new_collection.max_state_lvl

                                pair_quanta = pair_quanta.union(new_collection.pair_quanta)

        self.props = MolPropsCollection(MolecularProperty(i) for i in props_set)

pass