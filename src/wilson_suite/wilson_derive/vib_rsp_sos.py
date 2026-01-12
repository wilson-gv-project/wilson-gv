from .abstractions import (VibDiffTerm, ResonanceCondition, PolProp, TransitionIntegral, QOperator, VibStateSymbolic)
from .response_terms import VibContribTerm
from fractions import Fraction
from wilson_suite.wilson_utils import common_labels as wu_common
import copy

def get_vib_sos(maxord: int) -> list[VibContribTerm]:
    """
    Get vibrational SOS expressions at order 'maxord'

    maxord: Integer: Requested order of sum-over-states expression

    Returns a list of VibContribTerm instances corresponding to the vibrational SOS expressions at the 'maxord' order
    of perturbation
    """

    def which_int_has_omega(ints: list[TransitionIntegral], op_omega: QOperator) -> int:

        found = False

        for i in range(len(ints)):

            if op_omega.o in [j.o for j in ints[i].prop.ops]:

                if not found:
                    omega_int_ind = i
                    found = True

                else:
                    raise AssertionError('Omega operator found in more than one integral')

        if not found:
            raise AssertionError('Omega operator not found in any integral')

        return omega_int_ind

    def which_op_is_omega(integral: TransitionIntegral, op_omega: QOperator) -> int:

        found = False

        ops = [j.o for j in integral.prop.ops]

        for i in range(len(ops)):

            if ops[i] == op_omega.o:

                if not found:
                    omega_op_ind = i
                    found = True

                else:
                    raise AssertionError('Omega operator found in more than one position in integral')

        if not found:
            raise AssertionError('Omega operator not found in an integral that should contain it')

        return omega_op_ind

    # Get operator and state symbols using common labels
    op_omega = QOperator(wu_common.op_omega_label_int)
    ops = tuple([QOperator(wu_common.op_labels_int[i]) for i in range(maxord)])
    states = [VibStateSymbolic(wu_common.ground_state_label, is_ground=True)]
    states.extend([VibStateSymbolic(wu_common.state_labels[i]) for i in range(maxord)])

    # Initialize with order 0 term
    vib_sos_terms = [[VibContribTerm(Fraction(1), [TransitionIntegral(states[0], states[0], PolProp([op_omega]))], [])]]

    # Walk through orders and build terms
    for p in range(maxord):

        # To hold the terms at the new order
        vib_sos_terms.append([])

        # For each term at the preceding order, carry out the manipulations to get the new-order terms
        for r in vib_sos_terms[p]:

            # Identify which integral has the omega operator
            omega_ind = which_int_has_omega(r.ints, op_omega)

            # E type terms: New state as electronic
            r_e = copy.deepcopy(r)

            # D type terms: New state as vibrational
            r_d_1 = copy.deepcopy(r)
            r_d_2 = copy.deepcopy(r)

            # New resonance conditions for "D" type terms
            r_d_1.coeff *= -1
            r_d_1.res.append(ResonanceCondition(VibDiffTerm(states[p + 1], r.ints[omega_ind].bra), [(j + 1) for j in range(p + 1)]))

            r_d_2.coeff *= 1
            r_d_2.res.append(ResonanceCondition(VibDiffTerm(r.ints[omega_ind].ket, states[p + 1]), [(j + 1) for j in range(p + 1)]))

            r_e.ints[omega_ind].prop.ops.append(ops[p])

            # Making new integral for D1 term
            r_D_1_new = copy.deepcopy(r_d_1.ints[omega_ind])
            # Remove the omega operator from this integral (it will be "re-initialized in a new integral according to the recursive scheme)
            del r_D_1_new.prop.ops[which_op_is_omega(r_D_1_new, op_omega)]

            r_D_1_new.bra = states[p + 1]
            r_D_1_new.prop.ops.append(ops[p])

            # Here is the re-initizalization
            r_d_1.ints[omega_ind] = TransitionIntegral(r_d_1.ints[omega_ind].bra, states[p + 1], PolProp([op_omega]))
            # Insert the new D1 integral in the position after the re-initialized "omega integral"
            r_d_1.ints.insert(omega_ind + 1, r_D_1_new)

            # Making new integral for D2 term
            r_D_2_new = copy.deepcopy(r_d_2.ints[omega_ind])
            del r_D_2_new.prop.ops[which_op_is_omega(r_D_2_new, op_omega)]
            r_D_2_new.ket = states[p + 1]
            r_D_2_new.prop.ops.append(ops[p])

            r_d_2.ints[omega_ind] = TransitionIntegral(states[p + 1], r_d_2.ints[omega_ind].ket, PolProp([op_omega]))
            # Insert the new D2 integral in the position before the re-initialized "omega integral"
            r_d_2.ints.insert(omega_ind, r_D_2_new)

            # Putting results in next order terms holder
            vib_sos_terms[p + 1].append(r_e)
            vib_sos_terms[p + 1].append(r_d_1)
            vib_sos_terms[p + 1].append(r_d_2)

    return vib_sos_terms[maxord]

