from ..wilson_experiment.experiment_abstractions import VibExperiment
from . import dbl_pert_expansion
from . import hermaut
from . import vib_rsp_sos
from . import response_terms
from . import simplify
import copy
from ..wilson_utils import common_labels as wu_common

def get_dressed_vib_sos_with_exp_filtering(order: int, int_sequences: list, epochs: list,
                                           cfuv: dict) -> list[response_terms.VibContribTerm]:
    """
    Take an order parameter and an experiment's a) interaction sequences, b) epochs, and c) UV carrier frequency info
    and 1) Get the sum-over-states expression for the vibrational contribution to this order's response function with
    dummy interaction indices, 2) dress the terms with the pulse references for each valid interaction sequence, and
    3) filter the resulting terms by i) discarding terms where any pulses involved in an electronic response are not
    confined to the same epoch, ii) discarding terms where carrier frequencies of pulses in an electronic response
    do not have UV frequency components that together sum to zero

    Returns: A list of VibContribTerm instances constituting the filtered SOS expression
    """

    R_sos = vib_rsp_sos.get_vib_sos(order)

    R_sos_filtered = []

    for i in int_sequences:
        for j in R_sos:
            new_R_sos = copy.deepcopy(j)
            new_R_sos.dressWithPulseInteractions(i)
            if new_R_sos.allElRspEpochContained(epochs, 0):
                if new_R_sos.allUVCancels(cfuv):
                    R_sos_filtered.append(new_R_sos)

    return R_sos_filtered

def do_dbl_pert_expand_and_hermaut_with_enh_filtering(sos_terms:
                                                      list[response_terms.VibContribTerm],
                                                      order_el: int, order_mech: int,
                                                      magn_conditions: tuple) -> list[response_terms.VibPerturbedTerm]:
    """
    Take a list of SOS vibrational contribution response terms, perform a double perturbation expansion according to the
    requested orders of electrical and mechanical anharmonicity, carry out Hermite walk expansions for the resulting
    terms, simplify these results by coefficient factoring, and filter those results according to whether full
    enhancement is possible with the imposed magnitude conditions of the experiment.
     
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
    simplified_hermaut_terms = simplify.terms_simplify(full_hermaut_terms, wu_common.nm_inds)

    # Filter by possibility of full enhancement
    filtered_terms = []
    for k in simplified_hermaut_terms:
        if (simplified_hermaut_terms[k].full_enhancement_possible(magn_conditions=magn_conditions)):
            filtered_terms.append(copy.deepcopy(simplified_hermaut_terms[k]))

    return filtered_terms

def get_fully_enhanced_terms(experiment: VibExperiment, total_anharm_limit: int=1, el_anharm_limit: int=1,
                             mech_anharm_limit: int=1) -> dict:
    """
    Take an experiment instance and requested maximal order(s) of anharmonicity and return
    all terms that may be fully enhanced under this experiment at the requested orders of anharmonicity
    or lower. Returns a dictionary {0: {[0,0]: [...], ...}, 1: {[1,0]: [...], ...}, ...} where [...]
    is a list of VibPerturedTerm instances

    experiment: VibExperiment instance from wilson-experiment

    total: integer: Limitation on total order of anharmonicity (default = 1)
    el_anharm_limit: integer: Limitation on electrical order of anharmonicity (default = 1)
    mech_anharm_limit: integer: Limitation on mechanical order of anharmonicity (default = 1)
    """

    # Get interaction-dressed and epoch-/UV carrier freq.-filtered SOS vibrational response function expression
    R_sos = get_dressed_vib_sos_with_exp_filtering(experiment.order, experiment.int_sequences, experiment.epochs,
                                                   experiment.cfuv)


    # Get representation of relevant anharmonicity orders
    anharm_orders = dbl_pert_expansion.make_anharm_orders(total_anharm_limit, el_anharm_limit, mech_anharm_limit)

    # For each such anharmonicity order, do double pert. expansion, Hermite walk expansion and filtering
    final_terms = {}
    for i in anharm_orders:
        final_terms[i] = {}
        for j in anharm_orders[i]:
            final_terms[i][j] = do_dbl_pert_expand_and_hermaut_with_enh_filtering(R_sos, j[0], j[1], experiment.magn_conditions)

    return final_terms

# TODO: For future work, add function to dress generated VibPerturbedTerm instances with specific operator types
# according to choice of electromagnetic multipole expansion regime