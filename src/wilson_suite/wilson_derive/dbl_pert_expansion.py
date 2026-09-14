import copy
from .abstractions import PolProp, VibStateSymbolic, TransitionIntegral, VibDiffTerm
from .response_terms import VibContribTerm
from fractions import Fraction

# TODO: Implement arbitrary-order electrical/mechanical expansion

def expand_term(term: VibContribTerm, order_el: int=0, order_mech: int=0):
    """
    Expand a "sum-over-states" vibrational contribution term according to normal coordinate Taylor expansions
    of the term's electronic polarization properties and the vibrational potential

    term: VibContribTerm instance: The term to be expanded
    order_el: integer (default: 0): Order of electrical anharmonicity (sum anharmonicity in polarization properties)
    order_mech: integer (default: 0): Order of mechanical anharmonicity (anharmonicity in vibrational potential)
    """

    result_terms_el = []

    # First do electrical expansion
    if (order_el == 0):

        new_term = copy.deepcopy(term)
        for i in new_term.ints:
            i.prop.setDerivOrder(1)
        result_terms_el.append(copy.deepcopy(new_term))

    elif (order_el == 1):

        new_term_harm = copy.deepcopy(term)

        for i in new_term_harm.ints:
            i.prop.setDerivOrder(1)

        for i in range(len(new_term_harm.ints)):
            new_term = copy.deepcopy(new_term_harm)
            new_term.ints[i].prop.setDerivOrder(2)
            new_term.coeff *= Fraction(1, 2)
            result_terms_el.append(copy.deepcopy(new_term))

    elif (order_el == 2):

        new_term_harm = copy.deepcopy(term)

        for i in new_term_harm.ints:
            i.prop.setDerivOrder(1)

        for i in range(len(new_term_harm.ints)):
            for j in range(i, len(new_term_harm.ints)):

                new_term = copy.deepcopy(new_term_harm)

                if (i == j):

                    new_term.ints[i].prop.setDerivOrder(3)
                    new_term.coeff *= Fraction(1, 6)
                    result_terms_el.append(copy.deepcopy(new_term))

                else:

                    new_term.ints[i].prop.setDerivOrder(2)
                    new_term.ints[j].prop.setDerivOrder(2)
                    new_term.coeff *= Fraction(1, 4)
                    result_terms_el.append(copy.deepcopy(new_term))

    else:
        raise NotImplementedError('Electrical anharmonicity of order' + str(order_el) + ' not implemented')

    # Then do mechanical expansion or return the electrically harmonic result if no mechanical anharmonicity
    if (order_mech == 0):
        return result_terms_el

    elif (order_mech == 1):
        result_terms = []

        for i in result_terms_el:

            for j in range(len(i.ints)):

                for bk in ['bra', 'ket']:

                    new_term = copy.deepcopy(i)
                    new_term.coeff *= Fraction(-1, 6)
                    vcubic = PolProp([], dord=3)

                    if (bk == 'bra'):

                        orig_state = copy.deepcopy(new_term.ints[j].bra)
                        new_state = VibStateSymbolic('A', mbu=[orig_state])
                        new_term.ints[j].setBra(copy.deepcopy(new_state))

                        new_int = TransitionIntegral(copy.deepcopy(orig_state), copy.deepcopy(new_state),
                                                          copy.deepcopy(vcubic))

                        new_term.ints.insert(j, new_int)


                    elif (bk == 'ket'):
                        orig_state = copy.deepcopy(new_term.ints[j].ket)
                        new_state = VibStateSymbolic('A', mbu=[orig_state])
                        new_term.ints[j].setKet(copy.deepcopy(new_state))

                        new_int = TransitionIntegral(copy.deepcopy(new_state), copy.deepcopy(orig_state),
                                                          copy.deepcopy(vcubic))

                        new_term.ints.insert(j + 1, new_int)

                    new_term.addFreqTerm(VibDiffTerm(copy.deepcopy(new_state), copy.deepcopy(orig_state), is_pert_wf_diff=True))

                    result_terms.append(new_term)


        return result_terms

    else:
        raise NotImplementedError('Mechanical anharmonicity of order' + str(order_mech) + ' not implemented')

def make_anharm_orders_rec(total: int, limit_el: int, limit_mech: int, new_entry: list, orders: dict):
    """
    Tail recursion to generate anharmonic orders:

    total: integer: "remaining" orders to potentially fill
    limit_el: integer: Limitation on electrical order of anharmonicity
    limit_mech: integer: Limitation on mechanical order of anharmonicity
    new_entry: list: [a, b] where a and b are currently accumulated orders of el. resp. mech. anharmonicity
    orders: dictionary: To hold results
    """

    if total == 0:

        if not(sum(new_entry) in orders):
            orders[sum(new_entry)] = [tuple(new_entry)]

        else:

            if not(tuple(new_entry) in orders[sum(new_entry)]):
                orders[sum(new_entry)].append(tuple(new_entry))

    else:

        if not(new_entry[0] >= limit_el):

            # Give to el
            next_entry = copy.deepcopy(new_entry)
            next_entry[0] += 1
            make_anharm_orders_rec(total - 1, limit_el, limit_mech, next_entry, orders)

        if not(new_entry[1] >= limit_mech):

            # Give to mech
            next_entry = copy.deepcopy(new_entry)
            next_entry[1] += 1
            make_anharm_orders_rec(total - 1, limit_el, limit_mech, next_entry, orders)

        # Do "nothing" (for registering lower orders)
        next_entry = copy.deepcopy(new_entry)
        make_anharm_orders_rec(total - 1, limit_el, limit_mech, next_entry, orders)

    return

def make_anharm_orders(total: int, limit_el: int, limit_mech: int) -> dict:
    """
    Generate anharmonic orders as dict {0: [[0, 0]], 1: [[1, 0], [0, 1], ...], ...} limited by order arguments:

    total: integer: Limitation on total order of anharmonicity
    limit_el: integer: Limitation on electrical order of anharmonicity
    limit_mech: integer: Limitation on mechanical order of anharmonicity
    """

    orders = {}
    new_entry = [0, 0]

    make_anharm_orders_rec(total, limit_el, limit_mech, new_entry, orders)

    return orders
