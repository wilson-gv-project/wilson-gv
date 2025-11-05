from ..wilson_experiment.abstractions import VibExperiment
from . import abstractions as abst
from . import dbl_pert_expansion
from . import hermaut
from . import vib_rsp_sos
from . import simplify
import copy
from ..wilson_utils import common_labels as wu_common

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

    op_omega = abst.QOperator(wu_common.op_omega_label_int)
    ops_pert = tuple([abst.QOperator(wu_common.op_labels_int[i]) for i in range(experiment.order)])

    symbolic_vib_states = [abst.VibStateSymbolic(wu_common.ground_state_label, is_ground=True)]
    symbolic_vib_states.extend([abst.VibStateSymbolic(wu_common.state_labels[i]) for i in range(experiment.order)])

    R_sos: list[abst.VibContribTerm] = vib_rsp_sos.get_vib_sos(op_omega, ops_pert, experiment.order, symbolic_vib_states, noncomb=True)

    R_sos_int = []

    for i in experiment.int_sequences:
        for j in R_sos:
            new_R_sos = copy.deepcopy(j)
            new_R_sos.dressWithPulseInteractions(i)
            if (new_R_sos.allElRspEpochContained(experiment.epochs, 0)):
                if (new_R_sos.allUVCancels(experiment.cfuv)):
                    R_sos_int.append(new_R_sos)

    R_sos = R_sos_int

    anharm_orders = dbl_pert_expansion.make_anharm_orders(total_anharm_limit, el_anharm_limit, mech_anharm_limit)

    final_terms = {}

    for i in anharm_orders:

        final_terms[i] = {}

        for j in anharm_orders[i]:

            final_terms[i][j] = []

            R_dbl_pert = []

            for k in R_sos:
                R_dbl_pert.extend(dbl_pert_expansion.expand_term(k, order_el=j[0], order_mech=j[1]))

            full_hermaut_terms = []

            for k in R_dbl_pert:
                full_hermaut_terms.extend(hermaut.do_hermaut(k, wu_common.nm_inds))

            simplified_hermaut_terms = simplify.terms_simplify(full_hermaut_terms, wu_common.nm_inds)

            for k in simplified_hermaut_terms:
                if (simplified_hermaut_terms[k].full_enhancement_possible(magn_conditions=experiment.magn_conditions)):
                    final_terms[i][j].append(copy.deepcopy(simplified_hermaut_terms[k]))

    return final_terms

# TODO: For future work, add function to dress generated VibPerturbedTerm instances with specific operator types
# according to choice of electromagnetic multipole expansion regime