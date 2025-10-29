
import numpy as np
import copy


def isotropic_average_for_props_and_field(term, experiment):
    """
    Take a collection of properties and an experiment and determine the appropriate orientational average
    """

    # Outline of routine:

    # The average is formed as the contraction A * f * M * g * P (see JCP 67, 5026)
    # A is a tensor describing a laboratory-frame quantity: Here it represents the polarization of the incident/detected radiation
    # P is a tensor describing a molecule-frame quantity: Here it represents some relevant part of a response property (e.g.,
    # for four-wave mixing 2D-IR, this is one term's collection of polarization properties
    # f and g are collections resulting from evaluating strings of Kronecker deltas and Levi-Civita symbols according to the
    # JCP 67, 5026 procedure. The result is a collection of references to tensor components of A (f) or P (g) that fulfill the conditions as
    # dictated by each Kronecker/Levi-Civita string (one string leads to a given number of tensor components, and the collection is over all strings)
    # The f and g collections are created from tabulated strings. They have a "positive" and "negative" part. The positive part signifies that the
    # components specified inside are to be added. The negative part (only relevant if there was a Levi-Civita symbol during the evaluation (which happens at
    # odd orders) signifies that the components are to be subtracted
    # M is a matrix collecting coefficients associated with the orientational averaging (it is here tabulated for orders up to 6 but can in principle
    # be calculated).

    # 1: Fails:
    # a) Fail if the order is > 6
    # b) Fail if not all of the polarization properties are electric dipole polarization properties

    # 2: Get f, M, g
    # When properties are all electric dipole properties (not sure what happens if not), then f and g are the same thing
    # (except that f applies to A and g applies to P)

    # The exact organization of 3, 4, 5 may be adjusted

    # 3. Form K = M * g * p

    # 4. Form L = f * K

    # 5. Form the result A * L

    # Need to find out:
    # - How does A represent general polarization setups?
    # - Could einsum be a smart thing to use here?
    # - How to best bring in polarization information from the experiment?
    # - How to make sure that the ranks of the microscopic and macroscopic tensors (and their combination on averaging)
    # are properly kept track of/aligned?



    pass

# Calculate transposed 'laser polarization term' (the term (A * f) in (A * f * M * g * P) in JCP 141, 204103)
# The argument pol is a list of vectors
def get_pol_laser(pol):

    A = 1.0

    A = get_pol_tensor(A, pol)
    A = np.reshape(A, tuple([len(pol[i]) for i in range(len(pol))]))

    f = get_iso_f(len(pol))

    pl = np.zeros((len(f)))

    for i in range(len(f)):

        for j in range(len(f[i][0])):
            pl[i] += A[tuple(f[i][0][j])]

        for j in range(len(f[i][1])):
            pl[i] -= A[tuple(f[i][1][j])]

    return np.transpose(pl)


# Create polarization tensor from individual 3D polarization vectors of incident light
# Polarization vector elements are in general complex-valued
# Inital value of pol_tensor is 1.0
def get_pol_tensor(pol_tensor, pol):

    if len(pol) == 0:
        return pol_tensor

    else:
        return get_pol_tensor(np.kron(pol_tensor, np.array(copy.deepcopy(pol[len(pol) - 1]))), pol[0:len(pol) -1])

def mdk(a, b):

    return [
        {(a - 1): 0, (b - 1): 0},
        {(a - 1): 1, (b - 1): 1},
        {(a - 1): 2, (b - 1): 2}
    ]

def mdl(a, b, c):

    return [
        [{(a - 1): 0, (b - 1): 1, (c - 1): 2},
         {(a - 1): 1, (b - 1): 2, (c - 1): 0},
         {(a - 1): 2, (b - 1): 0, (c - 1): 1}],
        [{(a - 1): 2, (b - 1): 1, (c - 1): 0},
         {(a - 1): 1, (b - 1): 0, (c - 1): 2},
         {(a - 1): 0, (b - 1): 2, (c - 1): 1}]
    ]

def make_iso_f(n, kron, lc):

    # Make two lists of lists in iso_f: One for addition and another for subtraction
    iso_f = []

    iso_f_first = meso_iso_f(kron, [[0*i for i in range(n)]])
    iso_f.append(iso_f_first)

    # Are there only Kronecker deltas to take care of? If so, then no subtraction
    if len(lc) == 0:
        print('iso f', iso_f)
        return [iso_f[0], []]

    # If not, proceed to do Levi-Civita handling
    iso_f.append(copy.deepcopy(iso_f[0]))

    bperm = [[0], [1]]

    for i in range(len(bperm)):

        this_lc = []
        for j in range(len(bperm[i])):
            this_lc.append(lc[j][bperm[i][j]])

        iso_f[sum(bperm[i]) % 2] = meso_iso_f(this_lc, iso_f[sum(bperm[i]) % 2])

    print('iso f', iso_f)

    return iso_f

def meso_iso_f(dicts, iso_f):

        if len(dicts) > 0:

            new_iso_f = []

            for i in range(len(iso_f)):

                this_iso_f = copy.deepcopy(iso_f[i])
                for j in range(len(dicts[0])):
                    curr_iso_f = copy.deepcopy(this_iso_f)
                    for k in dicts[0][j].keys():

                        curr_iso_f[k] = dicts[0][j][k]

                    new_iso_f.append(copy.deepcopy(curr_iso_f))

            iso_f = copy.deepcopy(new_iso_f)
            result = meso_iso_f(dicts[1:len(dicts)], iso_f)
            return result

        else:

            return iso_f


# Currently only 3D
# Maybe necessary to rewrite for higher dimensions for e.g. quadrupole effects
def get_iso_f(n):

    if n == 2:

        return [
            make_iso_f(2, [mdk(0, 1)], []),
        ]


    elif n == 3:

        return [
            make_iso_f(3, [], [mdl(0, 1, 2)]),
        ]

    elif n == 4:

        return [
            make_iso_f(4, [mdk(0, 1), mdk(2, 3)], []),
            make_iso_f(4, [mdk(0, 2), mdk(1, 3)], []),
            make_iso_f(4, [mdk(0, 3), mdk(1, 2)], [])
        ]

    elif n == 5:

        return [
            make_iso_f(5, [mdk(3, 4)], [mdl(0, 1, 2)]),
            make_iso_f(5, [mdk(2, 4)], [mdl(0, 1, 3)]),
            make_iso_f(5, [mdk(2, 3)], [mdl(0, 1, 4)]),
            make_iso_f(5, [mdk(1, 4)], [mdl(0, 2, 3)]),
            make_iso_f(5, [mdk(1, 3)], [mdl(0, 2, 4)]),
            make_iso_f(5, [mdk(1, 2)], [mdl(0, 3, 4)])
        ]

    elif n == 6:

        return [
            make_iso_f(6, [mdk(0, 1), mdk(2, 3), mdk(4, 5)], []),
            make_iso_f(6, [mdk(0, 1), mdk(2, 4), mdk(3, 5)], []),
            make_iso_f(6, [mdk(0, 1), mdk(2, 5), mdk(3, 4)], []),
            make_iso_f(6, [mdk(0, 2), mdk(1, 3), mdk(4, 5)], []),
            make_iso_f(6, [mdk(0, 2), mdk(1, 4), mdk(3, 5)], []),
            make_iso_f(6, [mdk(0, 2), mdk(1, 5), mdk(3, 4)], []),
            make_iso_f(6, [mdk(0, 3), mdk(1, 2), mdk(4, 5)], []),
            make_iso_f(6, [mdk(0, 3), mdk(1, 4), mdk(2, 5)], []),
            make_iso_f(6, [mdk(0, 3), mdk(1, 5), mdk(2, 4)], []),
            make_iso_f(6, [mdk(0, 4), mdk(1, 2), mdk(3, 5)], []),
            make_iso_f(6, [mdk(0, 4), mdk(1, 3), mdk(2, 5)], []),
            make_iso_f(6, [mdk(0, 4), mdk(1, 5), mdk(2, 3)], []),
            make_iso_f(6, [mdk(0, 5), mdk(1, 2), mdk(3, 4)], []),
            make_iso_f(6, [mdk(0, 5), mdk(1, 3), mdk(2, 4)], []),
            make_iso_f(6, [mdk(0, 5), mdk(1, 4), mdk(2, 3)], [])
        ]

    else:

        raise ValueError('Unsupported get_iso_f order:', n)


def get_iso_mat(n):

    if n == 2:
        #FIXME: Factor 1/3? NOW UPD

        return 1.0/3.0

    elif n == 3:

        # FIXME: Factor 1/6? NOW UPD

        return 1.0/6.0

    elif n == 4:

        return np.array([
            [ 4, -1, -1],
            [-1,  4, -1],
            [-1, -1,  4]
        ])/30.0

    elif n == 5:

        return np.array([
            [ 3, -1, -1,  1,  1,  0],
            [-1,  3, -1, -1,  0,  1],
            [-1, -1,  3,  0, -1, -1],
            [ 1, -1,  0 , 3, -1,  1],
            [ 1,  0, -1, -1,  3, -1],
            [ 0,  1, -1,  1,  -1, 3]
        ])/30.0

    elif n == 6:

        return np.array([
            [ 16, -5, -5, -5,  2,  2, -5,  2,  2,  2,  2, -5,  2,  2, -5],
            [ -5, 16, -5,  2, -5,  2,  2,  2, -5, -5,  2,  2,  2, -5,  2],
            [ -5, -5, 16,  2,  2, -5,  2, -5,  2,  2, -5,  2, -5,  2,  2],
            [ -5,  2,  2, 16, -5, -5, -5,  2,  2,  2, -5,  2,  2, -5,  2],
            [  2, -5,  2, -5, 16, -5,  2, -5,  2, -5,  2,  2,  2,  2, -5],
            [  2,  2, -5, -5, -5, 16,  2,  2, -5,  2,  2, -5, -5,  2,  2],
            [ -5,  2,  2, -5,  2,  2, 16, -5, -5, -5,  2,  2, -5,  2,  2],
            [  2,  2, -5,  2, -5,  2, -5, 16, -5,  2, -5,  2,  2,  2, -5],
            [  2, -5,  2,  2,  2, -5, -5, -5, 16,  2,  2, -5,  2, -5,  2],
            [  2, -5,  2,  2, -5,  2, -5,  2,  2, 16, -5, -5, -5,  2,  2],
            [  2,  2, -5, -5,  2,  2,  2, -5,  2, -5, 16, -5,  2, -5,  2],
            [ -5,  2,  2,  2,  2, -5,  2,  2, -5, -5, -5, 16,  2,  2, -5],
            [  2,  2, -5,  2,  2, -5, -5,  2,  2, -5,  2,  2, 16, -5, -5],
            [  2, -5,  2, -5,  2,  2,  2,  2, -5,  2, -5,  2, -5, 16, -5],
            [ -5,  2,  2,  2, -5,  2,  2, -5,  2,  2,  2, -5, -5, -5, 16]
        ])/210.0

    else:

        raise ValueError('Unsupported get_iso_mat order:', n)


# Author: Magnus Ringholm
# this used to be in a separate file - mtRspfuncs.py

class mtRspfuncs:

# Class containing information and routine for fetching value of derivatives used in microscopic terms

    def __init__(self, operators, modes):

# The lists 'operators' and 'modes_rsp' contain lists specifying the operators and normal
# modes involved in the differentiation
# Let's say the operators are mu_alpha, mu_beta, and mu_delta diff. w.r.t. modes a (twice)
# and b (once). This is d**3beta / (da**2 * db). This would give
# operators = [0, 1, 3] and modes = [1, 0]
# UPDATE 2014: Identify by name, not number (example above would be ['a', 'b', 'd'] and ['b', 'a']
# Also: allow for several quantities, so each of operators and modes are lists of lists
# FURTHER UPDATE: Go back to identify by number (do mapping from identifiers used in sympy)

        self.operators = operators
        self.modes = modes
        self.ind_cache = {}
        self.value_cache = {}
        self.cachesize = 0

    def val(self, d, tensors, mode_indices, pl, iso_mat, iso_f):


        for i in range(self.cachesize):
            if self.ind_cache[i] == mode_indices:
                return self.value_cache[i]

        # Not sure about proper dimensions, original assignment below commented out, use len(iso_f) instead for now
        #P = np.zeros(sum([len(self.operators[i]) for i in range(len(self.operators))]))
        P = np.zeros(len(pl))

        for i in range(len(iso_f)):

            # First add
            for j in range(len(iso_f[i][0])):

                new_val = 1.0

                for k in range(len(self.operators)):

                    if not(self.operators[k][0] == 'z'):
                        this_ind = tuple([iso_f[i][0][j][m] for m in [alphanum[p] for p in self.operators[k]]])

                    else:
                        this_ind = (0,)

                    new_val = new_val * tensors.tensor_value(d, self.operators[k], this_ind, [mode_indices[m] for m in self.modes[k]])

                P[i] += new_val

            # Then subtract
            for j in range(len(iso_f[i][1])):

                new_val = 1.0

                for k in range(len(self.operators)):

                    if not(self.operators[k][0] == 'z'):
                        this_ind = tuple([iso_f[i][1][j][m] for m in [alphanum[p] for p in self.operators[k]]])

                    else:
                        this_ind = (0,)

                    new_val = new_val * tensors.tensor_value(d, self.operators[k], this_ind, [mode_indices[m] for m in self.modes[k]])

                P[i] -= new_val

        ans = np.dot(pl, np.dot(iso_mat, P))

        self.ind_cache[self.cachesize] = copy.deepcopy(mode_indices)
        self.value_cache[self.cachesize] = ans
        self.cachesize += 1

        return ans


def get_AlphaBetaGammaDelta_indices(num_f: int) -> np.ndarray:
    """
    Now is set for the EVV experiment and for ZZZZ polarization.

    pol_g is a list of lists of 2 lists where the second one is empty
          but first one contains the lists of interest

    :param num_f: number of pulses
    :return: array_of_4greekIndices - an array of arrays of 4 greek indices for second hyperpolarizability :
             [alpha, beta, gamma, delta]
    """
    pol_g = get_iso_f(num_f)
    array_of_4greekIndices = np.array([pol[0] for pol in pol_g], dtype='object').reshape(-1, num_f)
    return array_of_4greekIndices


def getPolarizationAveragingExpression(num_pulses: int, polarization: str):
    """
    Get the arrays of indices to be summed and a prefactor of the averaging expression

    """
    if len(polarization) != num_pulses:
        raise ValueError("Polarization choice string should have the length of 'num_pulses'")
    if num_pulses==4:
        if polarization=="ZZZZ":
            return get_AlphaBetaGammaDelta_indices(num_f=num_pulses), 1./15

