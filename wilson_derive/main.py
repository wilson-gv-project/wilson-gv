from wilson_experiment.abstractions import VibExperiment
from . import abstractions as abst
from . import dbl_pert_expansion
from . import hermaut
from . import vib_rsp_sos
from . import simplify
import copy


# Operator and state labels initialization
# FIXME: Consider moving these common definitions to utils

nm_inds = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

states = [abst.VibStateSymbolic('0', is_ground=True), abst.VibStateSymbolic('m'), abst.VibStateSymbolic('n'), abst.VibStateSymbolic('p'), abst.VibStateSymbolic('q'),
          abst.VibStateSymbolic('r'), abst.VibStateSymbolic('s'), abst.VibStateSymbolic('t'), abst.VibStateSymbolic('u'), abst.VibStateSymbolic('v')]

op_omega = abst.QOperator(0, 1)

ops_pert = (
abst.QOperator(1, 1),
abst.QOperator(2, 1),
abst.QOperator(3, 1),
abst.QOperator(4, 1),
abst.QOperator(5, 1),
abst.QOperator(6, 1),
abst.QOperator(7, 1),
abst.QOperator(8, 1),
)

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

    R_sos = vib_rsp_sos.get_vib_sos(op_omega, ops_pert, experiment.order, states, noncomb=True)

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
                full_hermaut_terms.extend(hermaut.do_hermaut(k, nm_inds))

            simplified_hermaut_terms = simplify.terms_simplify(full_hermaut_terms, nm_inds)

            for k in simplified_hermaut_terms:
                if (simplified_hermaut_terms[k].full_enhancement_possible(magn_conditions=experiment.magn_conditions)):
                    final_terms[i][j].append(copy.deepcopy(simplified_hermaut_terms[k]))

    return final_terms

