import copy
from .abstractions import polProp, vibState, transitionIntegral, vibDiffTerm
from fractions import Fraction

# TODO: Implement arbitrary-order electrical/mechanical expansion

def expand_term(term, order_el=0, order_mech=0):

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

    # Then do mechanical expansion or return the mechanically harmonic result
    if (order_mech == 0):
        return result_terms_el

    elif (order_mech == 1):
        result_terms = []

        for i in result_terms_el:

            for j in range(len(i.ints)):

                for bk in ['bra', 'ket']:

                    new_term = copy.deepcopy(i)
                    new_term.coeff *= Fraction(-1, 6)
                    vcubic = polProp([], dord=3)

                    if (bk == 'bra'):

                        orig_state = copy.deepcopy(new_term.ints[j].bra)
                        new_state = vibState('A', mbu=[orig_state])
                        new_term.ints[j].setBra(copy.deepcopy(new_state))

                        new_int = transitionIntegral(copy.deepcopy(orig_state), copy.deepcopy(new_state),
                                                          copy.deepcopy(vcubic))

                        new_term.ints.insert(j, new_int)


                    elif (bk == 'ket'):
                        orig_state = copy.deepcopy(new_term.ints[j].ket)
                        new_state = vibState('A', mbu=[orig_state])
                        new_term.ints[j].setKet(copy.deepcopy(new_state))

                        new_int = transitionIntegral(copy.deepcopy(new_state), copy.deepcopy(orig_state),
                                                          copy.deepcopy(vcubic))

                        new_term.ints.insert(j + 1, new_int)

                    new_term.addFreqTerm(vibDiffTerm(copy.deepcopy(new_state), copy.deepcopy(orig_state)))

                    result_terms.append(new_term)


        return result_terms

    else:
        raise NotImplementedError('Electrical anharmonicity of order' + str(order_el) + ' not implemented')


