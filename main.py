
from . import abstractions as abst
from . import dbl_pert_expansion
from . import hermaut
from . import vib_rsp_sos
from . import simplify
import copy


# Operator and state labels initialization

nm_inds = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

states = [abst.vibState('0', is_ground=True), abst.vibState('m'), abst.vibState('n'), abst.vibState('p'), abst.vibState('q'),
          abst.vibState('r'), abst.vibState('s'), abst.vibState('t'), abst.vibState('u'), abst.vibState('v')]

op_omega = abst.qOperator(0, 1)

ops_pert = (
abst.qOperator(1, 1),
abst.qOperator(2, 1),
abst.qOperator(3, 1),
abst.qOperator(4, 1),
abst.qOperator(5, 1),
abst.qOperator(6, 1),
abst.qOperator(7, 1),
abst.qOperator(8, 1),
)

def make_anharm_orders_rec(total, limit_el, limit_mech, new_entry, orders):

    if total == 0:

        if not(sum(new_entry) in orders):
            orders[sum(new_entry)] = [tuple(new_entry)]

        else:
            if not(new_entry in orders[sum(new_entry)]):
                orders[sum(new_entry)].append(tuple(new_entry))

    else:

        if not(new_entry[0] > limit_el):

            # Give to el
            next_entry = copy.deepcopy(new_entry)
            next_entry[0] += 1
            make_anharm_orders_rec(total - 1, limit_el, limit_mech, next_entry, orders)

        if not(new_entry[1] > limit_mech):

            # Give to mech
            next_entry = copy.deepcopy(new_entry)
            next_entry[1] += 1
            make_anharm_orders_rec(total - 1, limit_el, limit_mech, next_entry, orders)

        # Do nothing (for registering lower orders)
        next_entry = copy.deepcopy(new_entry)
        make_anharm_orders_rec(total - 1, limit_el, limit_mech, next_entry, orders)

    return

def make_anharm_orders(total, limit_el, limit_mech):

    orders = {}
    new_entry = [0, 0]

    make_anharm_orders_rec(total, limit_el, limit_mech, new_entry, orders)

    return orders

def get_fully_enhanced_terms(experiment, total_anharm_limit=1, el_anharm_limit=1, mech_anharm_limit=1):

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

    anharm_orders = make_anharm_orders(total_anharm_limit, el_anharm_limit, mech_anharm_limit)
    print('anharm orders', anharm_orders)

    final_terms = {}

    for i in anharm_orders:

        final_terms[i] = {}

        for j in anharm_orders[i]:

            print('entry i j', i, j)

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

