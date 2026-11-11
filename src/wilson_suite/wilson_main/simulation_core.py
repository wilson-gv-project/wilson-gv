import copy
from dataclasses import dataclass

from ..wilson_system.system_data import MolPropsCollection, MolecularProperty
from ..wilson_experiment.experiment_abstractions import VibExperiment
from ..wilson_derive.derive import ResponseTermsCollection
import numpy as np

@dataclass
class SimulationCore():
    """
    Class to collect experiment and terms information, and determine collected term-related information
    over all scan points of experiment.

    experiment: VibExperiment: The requested experiment.

    total_anharm_limit: Maximum total (electrical + mechanical) order of anharmonicity to be considered

    el_anharm_limit: Maximum order of electrical anharmonicity to be considered

    mech_anharm_limit: Maximum order of mechanical anharmonicity to be considered
    """

    experiment: VibExperiment
    total_anharm_limit: int
    el_anharm_limit: int
    mech_anharm_limit: int

    def __post_init__(self):
        """
        Process terms and analyze collectively, determining the following derived attributes:

        terms_by_order_comp_magn: Organized as nested dictionaries over perturbation order, compounding pattern
        and collective magnitude condition references: Terms (as ResponseTermCollection instances) for
        each such unique entry.

        coll_magn_bank: "Collective magnitude conditions": Dictionary organized by unique magnitude condition setups
        encountered over all scans. Values are the respective magnitude condition setups

        props: (Derivative) properties needed to evaluate the whole collection of identified terms,
        as MolPropsCollection instance consisting of Molecular property instances without values but with trivnames
        registered

        max_state_lvl: Integer describing the maximum number of quanta in states encountered over all (collective) terms

        pair_quanta: Set of 2-tuples describing unique encountered pair quanta in
        resonance condition VibDiff instances over all collected terms: For use
        if individual (state-pair specific) relaxation parameters will be used during evalutaion
        """

        # To hold all unique sets of resulting terms
        terms_by_order_comp_magn = {}

        # Collective magnitude condition bank and key counter
        # Need collective bank because individual FieldAnalysis have their own banks and entries at one scan point
        # may differ
        coll_magn_bank = {}
        coll_magn_bank_ctr = 0

        # For bookkeeping magnitude conditions encountered for each order and compounding pattern
        magn_enc_for_ord_comp = {}

        # To hold all derivative properties needed to evaluate the collective terms in terms_by_order_comp_magn
        props_set = set()

        # Maximum number of vibrational quanta encountered over all terms
        max_state_lvl = 0

        # All pair quanta encountered over all terms
        pair_quanta = set()

        # Iterating over scan points
        for scan_elem in np.ndindex(self.experiment.field_analyses.shape):

            # Considering valid interaction sequences at this scan point
            int_seq = self.experiment.field_analyses[scan_elem].int_sequences_by_order_comp_pol_magn

            # Order of perturbation
            for order in int_seq:

                if not order in terms_by_order_comp_magn:
                    terms_by_order_comp_magn[order] = {}
                    magn_enc_for_ord_comp[order] = {}

                # Compounding pattern
                for comp in int_seq[order]:

                    if not comp in terms_by_order_comp_magn[order]:
                        terms_by_order_comp_magn[order][comp] = {}
                        magn_enc_for_ord_comp[order][comp] = []

                    # Polarization setup and magnitude conditions
                    for pol in int_seq[order][comp]:
                        for magn in int_seq[order][comp][pol]:

                            curr_magn = self.experiment.field_analyses[scan_elem].magnitude_condition_bank[magn]

                            # Has this magn. cond. has been treated before for this order and compounding pattern?
                            # If not then now treat it and register that
                            if not curr_magn in magn_enc_for_ord_comp[order][comp]:

                                # Checking if this magn. cond. was already in the collective bank; if not make a
                                # new entry for it; in both cases mark the location
                                found_in_bank = False

                                coll_bank_loc = 0
                                for i in coll_magn_bank:
                                    if coll_magn_bank[i] == curr_magn:
                                        found_in_bank = True
                                        break
                                    coll_bank_loc += 1

                                if not found_in_bank:
                                    coll_magn_bank[coll_magn_bank_ctr] = curr_magn
                                    coll_bank_loc = coll_magn_bank_ctr
                                    coll_magn_bank_ctr += 1

                                magn_enc_for_ord_comp[order][comp].append(curr_magn)

                                new_collection = ResponseTermsCollection(order = order,
                                                            total_anharm_limit=self.total_anharm_limit,
                                                            el_anharm_limit=self.el_anharm_limit,
                                                            mech_anharm_limit=self.mech_anharm_limit,
                                                            comp_pattern=comp,
                                                            magn_conds=curr_magn)

                                # Register the obtained collection
                                terms_by_order_comp_magn[order][comp][coll_bank_loc] = new_collection

                                # Amending (if new entries) the set of necessary derivative properties
                                props_set = props_set.union(set(new_collection.props.names()))

                                # Updating/amending (if needed) the max state level and pair quanta
                                if new_collection.max_state_lvl > max_state_lvl:
                                    max_state_lvl = new_collection.max_state_lvl

                                pair_quanta = pair_quanta.union(new_collection.pair_quanta)




        # Setting collected attributes over all scan points
        self.terms_by_order_comp_magn = terms_by_order_comp_magn
        self.coll_magn_bank = coll_magn_bank
        self.props = MolPropsCollection(properties=tuple(MolecularProperty(i) for i in props_set))
        self.max_state_lvl = max_state_lvl
        self.pair_quanta = pair_quanta

pass