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

    order: Integer: The order of perturbation considered
    total_anharm_limit: Integer: Upper limit to total order of anharmonicity (electrical + mechanical)
    el_anharm_limit: Integer: Upper limit to electrical order of anharmonicity
    mech_anharm_limit: Integer: Upper limit to mechanical order of anharmonicity
    comp_pattern: Tuple of integers: The compounding pattern to consider
    magn_conds: Tuple: The magnitude conditions set to consider, each a tuple of integers

    """

    order: int
    total_anharm_limit: int
    el_anharm_limit: int
    mech_anharm_limit: int
    comp_pattern: tuple[int]
    magn_conds: tuple[tuple[int]]

    def __post_init__(self):

        # Obtain terms for the requested setup parameters
        self.terms = get_terms_by_comp_pattern_and_magn_conds(self.order,
                                           self.total_anharm_limit, self.el_anharm_limit, self.mech_anharm_limit,
                                           self.comp_pattern, self.magn_conds)

        # Determine which (derivative) properties are needed to evaluate these terms
        props = self.__determine_needed_props()

        self.props = props

        # Determine the maximum number of quanta in the states and pair quanta in energy level differences
        # The latter is mainly relevant for determining the scope of relaxation parameters
        max_state_lvl, pair_quanta = self.__find_max_state_lvl_and_pair_quanta()

        self.max_state_lvl = max_state_lvl
        self.pair_quanta = pair_quanta

    def __determine_needed_props(self) -> MolPropsCollection:
        """
        Walk through all terms and determine which (derivative) properties would be needed to evaluate them
        Make a MolPropsCollection instance with all the needed derivatives
        """

        props = []

        for t in self.terms:
            for pp in t.props:
                trivname = pt.prop_trivname(ord_geo=pp.dord, ord_el=len(pp.ops), ord_rot=0)
                if not trivname in props:
                    props.append(trivname)

        return MolPropsCollection([MolecularProperty(i) for i in props])


    def __find_max_state_lvl_and_pair_quanta(self) ->  tuple[int, tuple[tuple]] :
        """
        Determine the maximum number of quanta used in state energy levels

        Determine the different (sorted) pair number of quanta in level diffs. (relevant for relaxation parameters)

        Returns and integer (max state level) and a tuple of 2-tuples (each a pair quanta tuple)
        """

        max_state_lvl = 0
        pair_quanta = set()

        for t in self.terms:

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
                    pair_quanta.add((len_l, len_r))

        return max_state_lvl, pair_quanta


def do_dbl_pert_expand_and_hermaut(sos_terms: list[VibContribTerm],
                                   order_el: int, order_mech: int) -> list[VibPerturbedTerm]:
    """
    Take a list of SOS vibrational contribution response terms, perform a double perturbation expansion according to the
    requested orders of electrical and mechanical anharmonicity, carry out Hermite walk expansions for the resulting
    terms and simplify these results by coefficient factoring

    sos_terms: a list of VibContribTerm instances to be expanded

    order_el: Integer: Order of electrical anharmonicity

    order_mech: Integer: Order of mechanical anharmonicity

    Returns: List of VibPerturbedTerm instances: The expanded terms
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

def get_terms_by_comp_pattern_and_magn_conds(order: int, total_anharm_limit: int, el_anharm_limit: int,
                                             mech_anharm_limit: int, comp_pattern: tuple,
                                             magn_conds: tuple[tuple]) -> list[VibPerturbedTerm]:
    """
    Take a specification of the order of perturbation, degree of anharmonicity, requested compounding patterns,
    and magnitude conditions and return fully enhanceable terms up to the requested anharmonicity setup

    order: Integer: Order of perturbation

    total_anharm_limit: Integer: Max. total (el. + mech.) order of anharmonicity to be considered

    el_anharm_limit: Integer: Max. order of electrical anharmonicity

    mech_anharm_limit: Integer: Max order of mechanical anharmonicity

    comp_pattern: Time-ordered compounding pattern: Tuple of integers referring to the level of compounding
    at each interaction index

    magn_conds: Len 2 tuple (positive and negative combinations): Each entry another tuple: Each such entry a tuple of
    interaction indices each corresponding to an interaction index combination found to be either positive or negative

    Returns: list of VibPerturbedTerm instances: Terms found to be possibly fully enhanceable at the requested setup

    May return some terms that are not fully enhanceable depending on completeness of magnitude condition handling
    """

    # Get all SOS terms that match requested compounding pattern
    # NOTE: Can if needed at very high orders replace get_vib_sos by something that prunes
    # compounding patterns during the derivation
    sos_terms = vib_rsp_sos.get_vib_sos(order)

    # For requested compounding pattern, group SOS terms under each pattern
    sos_terms_at_comp_pattern = []

    for i in sos_terms:
        if i.comp_pattern == comp_pattern:
            sos_terms_at_comp_pattern.append(copy.deepcopy(i))

    # Then get anharm orders
    anharm_orders = dbl_pert_expansion.make_anharm_orders(total_anharm_limit, el_anharm_limit, mech_anharm_limit)

    # Do double perturbation expansion, Hermite expansion and simplification

    # Get post-Hermite terms
    ph_terms = []

    for tot_ord in anharm_orders:
        for ind_ord in anharm_orders[tot_ord]:
            ph_terms.extend(do_dbl_pert_expand_and_hermaut(sos_terms_at_comp_pattern,
                                               ind_ord[0], ind_ord[1]))

    # Filter by magnitude conditions

    # Initialize result holder
    final_terms = []

    # Walk through post-Hermite terms and screen by magnitude conditions

    # Build dict to tell which interaction indices the compounding pattern refers to
    comp_patt_to_int = {}
    int_ctr = 1
    comp_ctr = 0

    for c in comp_pattern:
        new_ints = []

        for j in range(c):
            new_ints.append(int_ctr)
            int_ctr += 1

        comp_patt_to_int[comp_ctr] = copy.deepcopy(new_ints)
        comp_ctr += 1

    # Translate magnitude condition to interaction indices
    mcp_int = []

    for m in magn_conds[0]:
        new_mcp = []
        for j in m:
            new_mcp.extend(comp_patt_to_int[j])

        mcp_int.append(copy.deepcopy(new_mcp))

    mcm_int = []

    for m in magn_conds[1]:
        new_mcm = []
        for j in m:
            new_mcm.extend(comp_patt_to_int[j])

        mcm_int.append(copy.deepcopy(new_mcm))

    # Keep terms that are not clearly impossible to fully enhance
    for t in ph_terms:

        if t.full_enhancement_possible(mcp = mcp_int, mcm = mcm_int):
            final_terms.append(copy.deepcopy(t))

    return final_terms
