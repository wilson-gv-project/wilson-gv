
from .abstractions import vibDiffTerm, resonanceCondition, vibContribTerm, polProp, transitionIntegral
from fractions import Fraction
import itertools
import copy

class rspTermSOSRecursion:

    def __init__(self, a, rsp_omega, b, freq, k):

        # Integrals ("properties") to the left of the integral containing the operator omega
        self.a = a
        # An integral containing the operator omega
        self.rsp_omega = rsp_omega
        # Integrals ("properties") to the right of the integral containing the operator omega
        self.b = b
        # Resonance conditions
        self.freq = freq
        # Integral operator order argument (for combinatorics summation)
        self.k = k

        self.hbar = 0
        self.coeff = 1

    # Print information about self
    def presentab(self):

        print('Overall coefficient', self.coeff)

        for i in self.a:
            i.presentProp()

        self.rsp_omega.presentProp()

        for i in self.b:
            i.presentProp()

        for i in self.freq:
            i.present()

# (Electronic) polarization property for use in SOS recursion
class polPropSOSRecursion:

    # Takes set of operators, bra and ket vibrational states left and right
    def __init__(self, order, left, right):
        self.order = order
        # FIXME: Is omega here vestigial?
        self.omega = 1
        self.left = left
        self.right = right
        self.ops = []

    def addOperator(self, op):
        self.ops.append(op)


    # Print information about self
    def presentProp(self):

        if self.omega:
            print('('  + str(self.order + 1) + ')' +  ' ' + self.left.s + ',' + self.right.s + ' (a)')

        else:
            print('(' + str(self.order) + ')' + ' ' + self.left.s + ',' + self.right.s)

        print('Operators:', [i.o for i in self.ops])

# Get "uncombinatorized" form of SOS vibrational contributions up to order 'maxord' with vibrational states 'states'
def vib_contribs_abstract(maxord, states, ops):

    # Initial order 0 term
    R = [[rspTermSOSRecursion([], polPropSOSRecursion(0, states[0], states[0]), [], [], [maxord])]]

    for p in range(maxord):

        # To hold the terms at the new order
        R.append([])

        # For each term at the preceding order, carry out the manipulations to get the new-order terms
        for r in R[p]:

            # E type terms: New state as electronic
            r_E = copy.deepcopy(r)

            # D type terms: New state as vibrational
            r_D_1 = copy.deepcopy(r)
            r_D_2 = copy.deepcopy(r)

            # New resonance conditions for "D" type terms
            r_D_1.hbar += 1
            r_D_1.coeff *= -1
            r_D_1.freq.append(resonanceCondition(vibDiffTerm(states[p + 1], r.rsp_omega.left), [(j + 1) for j in range(p + 1)]))

            r_D_2.hbar += 1
            r_D_2.coeff *= 1
            r_D_2.freq.append(resonanceCondition(vibDiffTerm(r.rsp_omega.right, states[p + 1]), [(j + 1) for j in range(p + 1)]))

            # FIXME: Create the new terms according to recursion in manuscript - document when manuscript updated
            if (r_E.rsp_omega.order < 1):
                r_E.k.append(1)

            else:
                r_E.k[len(r_E.k) - 1] += 1

            r_E.k[0] -= 1
            r_E.rsp_omega.order += 1

            r_E.rsp_omega.addOperator(ops[p])

            # Making new integral for D1 term
            r_D_1_new = copy.deepcopy(r_D_1.rsp_omega)

            if (r_D_1_new.order < 1):
                r_D_1.k.append(1)

            else:
                r_D_1.k[len(r_D_1.k) - 1] += 1




            r_D_1_new.omega = 0
            r_D_1_new.order += 1
            r_D_1_new.left = states[p + 1]

            # NEW
            r_D_1_new.addOperator(ops[p])
            # END NEW

            r_D_1.b = [r_D_1_new] + r_D_1.b
            r_D_1.rsp_omega = polPropSOSRecursion(0, r_D_1.rsp_omega.left, states[p + 1])
            r_D_1.k[0] -= 1

            # Making new integral for D2 term
            r_D_2_new = copy.deepcopy(r_D_2.rsp_omega)

            if (r_D_2_new.order < 1):
                r_D_2.k.append(1)

            else:
                r_D_2.k[len(r_D_2.k) - 1] += 1



            r_D_2_new.order += 1
            r_D_2_new.omega = 0
            r_D_2_new.right = states[p + 1]

            # NEW
            r_D_2_new.addOperator(ops[p])
            # END NEW

            r_D_2.a = r_D_2.a + [r_D_2_new]
            r_D_2.rsp_omega = polPropSOSRecursion(0, states[p + 1], r_D_2.rsp_omega.right)
            r_D_2.k[0] -= 1

            R[p + 1].append(r_E)
            R[p + 1].append(r_D_1)
            R[p + 1].append(r_D_2)

        p += 1

        if (p == maxord):
            return R

    return R

# FIXME: I need to reacquaint myself with this function (see manuscript): It could be obsolete (only first comb needed)
# Get "permutation" combinations of perturbing operators
def get_op_combs_rec_t(ops, curr, k, lvl, res, noncomb=False):

    if (lvl == len(k)):
        res.append(copy.deepcopy(curr))

    else:
        new_list = itertools.combinations(ops, k[lvl])

        for i in new_list:
            new_curr = copy.deepcopy(curr)
            new_curr.append(copy.deepcopy(list(i)))
            get_op_combs_rec_t(make_op_sel_set(ops, list(i)), new_curr, k, lvl + 1, res, noncomb)
            if noncomb:
                return


# Recursion helper function
def make_op_sel_set(all_ops, taken):
    sel_set = []

    for i in all_ops:
        found = 0

        for j in taken:
            if ((j[0].o, j[1]) == (i[0].o, i[1])):
                found = 1

        if not found:
            sel_set.append(i)

    return sel_set

# Get vibrational SOS expressions at order 'maxord' for omega operator 'op_omega' and perturbing operators 'ops'"
# with vibrational states 'states'
def get_vib_sos(op_omega, ops, maxord, states, noncomb=False):

    # Frequency references and (operator, freq) pairs structure
    # FIXME: Possible disconnect between op keys and "template" (1,2,3..) freq args in vib_contribs_abstract
    freqs = tuple([i + 1 for i in range(len(ops))])
    op_freq_pairs = tuple([(ops[i], freqs[i]) for i in range(len(ops[:maxord]))])

    # Get "uncombinatorized" SOS expressions
    R = vib_contribs_abstract(maxord, states, ops)
    R_comb = []


    # Get operator combinatorics information
    for i in range(len(R[maxord])):
        c = []
        res = []
        get_op_combs_rec_t(op_freq_pairs, c, R[maxord][i].k, 1, res, noncomb=noncomb)
        R_comb.append(copy.deepcopy(res))

    R_vibContrib = []

    # Loop over identified terms and associated combinatorics and make SOS vibContribTerm instances
    for i in range(len(R[maxord])):

        for k in range(len(R_comb[i])):

            # Frequency mask for combinations
            freq_mask = {}
            oc = 0

            for m in range(len(R_comb[i][k])):
                for n in range(len(R_comb[i][k][m])):
                    freq_mask[oc + 1] = R_comb[i][k][m][n][1]
                    oc += 1

            new_ints = []
            oc = 0

            # FIXME 2024: Clean up after October fixes (operator permutations, no symmetry)

            # Make transition integrals for a, omega, b parts

            for m in R[maxord][i].a:

                new_ints.append(
                    transitionIntegral(m.left, m.right,
                                       polProp(m.ops))
                                       )

                oc += 1

            # FIXME: Possible ordering mix-up for non-uniform (usually non-all electric dipole) operators
            if (R[maxord][i].rsp_omega.order > 0):

                new_ops = [op_omega]
                new_ops.extend(R[maxord][i].rsp_omega.ops)

                new_ints.append(
                    transitionIntegral(R[maxord][i].rsp_omega.left, R[maxord][i].rsp_omega.right,
                                       polProp(new_ops))
                )
                oc += 1

            else:
                new_ints.append(
                    transitionIntegral(R[maxord][i].rsp_omega.left, R[maxord][i].rsp_omega.right,
                                       polProp([op_omega])
                                       )
                )

            for m in R[maxord][i].b:
                new_ints.append(
                    transitionIntegral(m.left, m.right,
                                       polProp(m.ops))
                                       )
                oc += 1

            new_res = []

            # Resonance conditions
            for m in range(len(R[maxord][i].freq)):
                one_new_res = copy.deepcopy(R[maxord][i].freq[m])
                one_new_res.permute(freq_mask)
                new_res.append(one_new_res)

            R_vibContrib.append(vibContribTerm(
                Fraction(R[maxord][i].coeff),
                new_ints,
                new_res
            ))

    return R_vibContrib