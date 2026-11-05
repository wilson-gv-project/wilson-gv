from ..wilson_experiment.experiment_abstractions import VibExperiment
from . import dbl_pert_expansion
from . import hermaut
from . import vib_rsp_sos
from dataclasses import dataclass
from .response_terms import VibPerturbedTerm, VibContribTerm
from . import simplify
import copy
from ..wilson_utils import common_labels as wu_common
from ..wilson_utils import prop_trivname as pt
from ..wilson_system.system_data import MolecularProperty, MolPropsCollection

@dataclass
class ResponseTermsCollection:
    """
    Data holder class to collect terms under a common setup:

    order: The order of perturbation considered
    total_anharm_limit: Upper limit to total order of anharmonicity (electrical + mechanical)
    el_anharm_limit: Upper limit to electrical order of anharmonicity
    mech_anharm_limit: Upper limit to mechanical order of anharmonicity

    """

    order: int
    total_anharm_limit: int
    el_anharm_limit: int
    mech_anharm_limit: int
    # Keys are compounding patterns, values are magnitude conditions under each comp. pattern
    comp_patterns_with_magn_conditions: dict[tuple, tuple]

    def __post_init__(self):

        # Derived properties from original arguments if any

        # Get terms
        self.terms = get_terms_by_comp_patterns(self.order,
                                           self.total_anharm_limit, self.el_anharm_limit, self.mech_anharm_limit,
                                           self.comp_patterns_with_magn_conditions)

        props = self.__determine_needed_props()

        self.props = props

        max_state_lvl, pair_quanta = self.__find_max_state_lvl_and_pair_quanta()

        self.max_state_lvl = max_state_lvl
        self.pair_quanta = pair_quanta

    def __determine_needed_props(self):

        # Walk through all terms and determine which (derivative) properties would be needed to evaluate them
        # Make a MolPropsCollection instance with all the needed derivatives
        # Determine the maximum number of quanta used in state energy levels
        # Determine the different (sorted) pair number of quanta used in relaxation parameters

        props = []

        for p in self.terms:
            for m in self.terms[p]:
                for t in self.terms[p][m]:
                    for pp in t.props:
                        trivname = pt.prop_trivname(ord_geo=pp.dord, ord_el=len(pp.ops), ord_rot=0)
                        if not trivname in props:
                            props.append(trivname)

        return MolPropsCollection([MolecularProperty(i) for i in props])

    def __find_max_state_lvl_and_pair_quanta(self):

        max_state_lvl = 0
        pair_quanta = []

        for p in self.terms:
            for m in self.terms[p]:
                for t in self.terms[p][m]:

                    for j in t.freqterms:

                        if len(j.sl.q) > max_state_lvl:
                            max_state_lvl = len(j.sl.q)
                        if len(j.sr.q) > max_state_lvl:
                            max_state_lvl = len(j.sr.q)

                    for j in t.res:

                        len_l = len(j.diff.sl.q)
                        len_r = len(j.diff.sr.q)

                        if len_l > max_state_lvl:
                            max_state_lvl = len_l

                        if len_r > max_state_lvl:
                            max_state_lvl = len_r

                        if not (len_l, len_r) in pair_quanta:
                            pair_quanta.append((len_l, len_r))

        return max_state_lvl, sorted(pair_quanta)

def do_dbl_pert_expand_and_hermaut(sos_terms: list[VibContribTerm],
                                   order_el: int, order_mech: int) -> list[VibPerturbedTerm]:
    """
    Take a list of SOS vibrational contribution response terms, perform a double perturbation expansion according to the
    requested orders of electrical and mechanical anharmonicity, carry out Hermite walk expansions for the resulting
    terms and simplify these results by coefficient factoring
    """

    # Do double perturbation expansion
    R_dbl_pert = []
    for k in sos_terms:
        R_dbl_pert.extend(dbl_pert_expansion.expand_term(k, order_el=order_el, order_mech=order_mech))

    # Do Hermite walk expansion
    full_hermaut_terms = []
    for k in R_dbl_pert:
        full_hermaut_terms.extend(hermaut.do_hermaut(k, wu_common.nm_inds))

    # Simplify
    return list(simplify.terms_simplify(full_hermaut_terms, wu_common.nm_inds).values())

def get_terms_by_comp_patterns(order: int, total_anharm_limit: int, el_anharm_limit: int, mech_anharm_limit: int,
                              comp_patterns_with_magn_conditions: dict[tuple, tuple]) -> dict[tuple, list[VibPerturbedTerm]]:
    """
    Take a specification of the order of perturbation, degree of anharmonicity, requested compounding patterns,
    and magnitude conditions and return fully enhanceable terms

    May return some terms that are not fully enhanceable depending on completeness of magnitude condition handling
    """

    # Get all SOS terms that match requested compounding patterns
    # NOTE: Can if needed at very high orders replace get_vib_sos by something that prunes
    # compounding patterns during the derivation
    sos_terms = vib_rsp_sos.get_vib_sos(order)

    # For all requested compounding patterns, group SOS terms under each pattern
    sos_terms_at_comp_pattern = {i: [] for i in list(comp_patterns_with_magn_conditions.keys())}

    for i in sos_terms:

        if i.comp_pattern in sos_terms_at_comp_pattern:
            sos_terms_at_comp_pattern[i.comp_pattern].append(copy.deepcopy(i))

    # Then get anharm orders
    anharm_orders = dbl_pert_expansion.make_anharm_orders(total_anharm_limit, el_anharm_limit, mech_anharm_limit)

    # Do double perturbation expansion, Hermite expansion and simplification

    # Group post-Hermite terms under each requested compounding pattern
    ph_terms_at_comp_pattern = {i: [] for i in list(sos_terms_at_comp_pattern.keys())}

    for tot_ord in anharm_orders:
        for ind_ord in anharm_orders[tot_ord]:
            for i in sos_terms_at_comp_pattern:
                ph_terms_at_comp_pattern[i].extend(do_dbl_pert_expand_and_hermaut(sos_terms_at_comp_pattern[i],
                                                   ind_ord[0], ind_ord[1]))

    # Filter by magnitude conditions

    # Initialize result holder
    final_terms_at_comp_patt_and_magn = {i:
                                         {j: [] for j in comp_patterns_with_magn_conditions[i]}
                                         for i in list(comp_patterns_with_magn_conditions.keys())}

    # Walk through post-Hermite term and screen/group by magnitude conditions
    for p in final_terms_at_comp_patt_and_magn:

        # Build dict to tell which interaction indices each compounding pattern refers to
        comp_patt_to_int = {}
        int_ctr = 1
        comp_ctr = 0

        for c in p:
            new_ints = []

            for j in range(c):
                new_ints.append(int_ctr)
                int_ctr += 1

            comp_patt_to_int[comp_ctr] = copy.deepcopy(new_ints)

            comp_ctr += 1

        for m in final_terms_at_comp_patt_and_magn[p]:

            # Translate magn cond to interaction indices

            mcp_int = []
            for j in m[0]:
                new_mcp = []
                for k in j:
                    new_mcp.extend(comp_patt_to_int[k])
                mcp_int.append(copy.deepcopy(new_mcp))

            mcm_int = []
            for j in m[1]:
                new_mcm = []
                for k in j:
                    new_mcm.extend(comp_patt_to_int[k])
                mcm_int.append(copy.deepcopy(new_mcm))

            for t in ph_terms_at_comp_pattern[p]:

                if t.full_enhancement_possible(mcp = mcp_int, mcm = mcm_int):
                    final_terms_at_comp_patt_and_magn[p][m].append(copy.deepcopy(t))

    return final_terms_at_comp_patt_and_magn
