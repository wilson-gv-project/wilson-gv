
import numpy as np
import copy


def isotropic_average_for_props_and_field(term, experiment):

    # Need to find out:
    # - How does A represent general polarization setups?
    #   - I think I have a decent idea now: For e.g. a beam in the z direction, use the x and y components to define the polarization
    #   - Should be able to consider both linear and circularly polarized light: 3110: Yes indeed it looks so
    #   - FIXME: Find out magnitude of polarization vector: Should be some kind of unit-like vector
    #   - For beams whose wavevector is not purely in a Cartesian direction, will need to be some extra angle stuff but should be manageable
    # - Could einsum be a smart thing to use here?
    # - How to best bring in polarization information from the experiment?
    #   - I think take the entire experiment including the detector. I suppose the detector will also need to be extended to involve a polarization filter
    #   - Not sure how to handle "mixed" polarization setups (i.e. beaming/detecting light with several polarizations), but leave that for now
    # - How to make sure that the ranks of the microscopic and macroscopic tensors (and their combination on averaging)
    # are properly kept track of/aligned?

    # Outside tasks:
    # - Change detector_location attribute to be "detector_facing"? NO (OK)
    # - Choose convention for polarization definition OK
    # - Generate complete laboratory-axis polarization vectors OK
    # - Change in integration testing scripts: Make linear polarization but in a choice of direction WAIT
    # - Change wilson-intensities start to determine laser polarization term for experiment (or compute and send from main)
    # - Harmonize the ranks of pulses to the ranks of polarization property Greek indices


    # Technical/physics questions:
    # - Polarization changes for negative wavevectors?
    #           - I think (from literature) that the polarization then is complex conjugated:
    #               - Overall pulse phase is simple complex conjugation
    #               - Since I am limiting to linear polarization there is no question about any changes in polarization unit vector
    #               - See if Mukamel offers any more on this
    # - Polarization/overall phase for detection?
    #   - Overall phase: Not sure and also not sure if it matters
    #   - Polarization vector: I presume to be handled by polarization filter (in general more than one choice detectable)
    # - Taking real parts of incident/outgoing field modes?
    # - All the polarization considerations w.r.t. evaluating as response function
    #   - Recall that the microscopic parts of the response function are the same (although selected for by phase-matching
    #   and spectral domain considerations, and that the polarization/"tabletop" pulse considerations
    #   a) properly recombine them (which is a polarization direction matter only and is not complicated with linear polarization), and
    #   b) I don't think other effects (overall phase, specific pulse time envelopes) effect changes that are important to us:
    #       - They would likely have to do with some propagation of overall phase
    #       - Field strength considerations would be more or less a simple multiplication
    #       - Pulse shape considerations are not in the present scope and it's an established limitation to us
    # - Check Mukamel for full understanding of wave-mixing's emergence of phase-matching direction OK
    #   - No changes there to above findings
    #   - negative k vectors appear to have c.c. polarization

    # CONCLUSION 251103: Can proceed with implementation and settle the remaining details at a later stage
    # Recap of all known remaining details:'
    # - Overall phase now enforced as zero for each pulse, any limitations? Phase of outgoing signal?
    # - Correct in polarization filter thinking?
    # - Taking real part?
    # - Full "evaluating as response function" check for inconsistencies
    # - c.c. polarization for negative k vectors dbl chk (no changes upon c.c. with my zero overall phase and linear pol.?)
    #   - for prev. and "real part", cos(-wt) and cos(wt) are same but sin(-wt) and sin(wt) are opposite phase. Is that an issue here?
    # - Unit vectorization for polarization? Probably already settled but can double check





    # The main routine should take:
    # - The laser polarization term (a vector)
    # - An order parameter n

    # The result of the main routine should be:
    # - A structure associating a tuple of Greek/electromagnetic perturbation
    #   operator index: axis value pairs with a coefficient, defining the orientational average.
    # E.g.: for a quadratic response function with Cartesian index labels 123 (alpha beta gamma),
    # the sum 0.5 * Beta_zzz + 2.0 * Beta_xzy would be expressed (x = 0, y = 1, z = 2) in a form like
    # [(3: 2, 1: 2, 2: 2): 0.5, (1: 0, 3: 2, 2: 3): 2.0] (orderings deliberately mixed for this example)
    # Alternatively, return Cartesian rank mapping as separate dictionary and keep one Cartesian tuple in coeff list
    # I like this last option better



    pass

# FIXME: Type hints for all

# FIXME: Verify this routine thoroughly with tests
def get_pol_laser(pol):
    """
    Calculate transposed 'laser polarization term' (the term (A * f) in (A * f * M * g * P) in JCP 141, 204103)
    pol: list of len 3 lists specifying polarization vectors
    """

    A = 1.0

    A = get_pol_tensor(A, pol)
    A = np.reshape(A, tuple([len(pol[i]) for i in range(len(pol))]))

    #print('A', A)

    f = get_iso_f(len(pol))

    #print('f', f)

    # NOTE: This is real-valued for now since it only currently deals with linear polarization and assumed overall
    # phase of zero. If any of these conditions are relaxed this must be changed to be complex-valued
    pl = np.zeros((len(f)))

    for i in range(len(f)):

        for j in range(len(f[i][0])):
            pl[i] += A[tuple(f[i][0][j])]

        for j in range(len(f[i][1])):
            pl[i] -= A[tuple(f[i][1][j])]

    #print('A * f', np.transpose(pl))

    return [float(i) for i in pl]


def get_pol_tensor(pol_tensor, pol):
    """
    Create polarization tensor from individual 3D polarization vectors of incident light by forming Kronecker product
    Polarization vector elements are in general complex-valued
    Inital value of pol_tensor is 1.0

    pol: 3D polarization vectors
    Tail-recursive: Return is a rank len(pol) 3 x 3 x 3 ... array
    """

    if len(pol) == 0:
        return pol_tensor

    else:
        return get_pol_tensor(np.kron(pol_tensor, np.array(copy.deepcopy(pol[len(pol) - 1]))), pol[0:len(pol) -1])

def mdk(a: int, b: int):
    """
    Kronecker delta over three spatial dimensions

    Takes integers a and b for ranks and returns list of dictionaries each of valid pairs of axes for these ranks
    """

    return [
        {a: 0, b: 0},
        {a: 1, b: 1},
        {a: 2, b: 2}
    ]

def mdl(a: int, b: int, c: int):
    """
    Levi-Civita terms

    Takes integers a, b, c for ranks and returns two lists, each consisting of dictionaries each of valid triples of
    axes for these ranks:
    - The first list contains elements to be considered with a factor +1 ("forwards" permutations)
    - The second list contains elements to be considered with a factor -1 ("backwards" permutations)
    """

    return [
        [{a: 0, b: 1, c: 2},
         {a: 1, b: 2, c: 0},
         {a: 2, b: 0, c: 1}],
        [{a: 2, b: 1, c: 0},
         {a: 1, b: 0, c: 2},
         {a: 0, b: 2, c: 1}]
    ]

def make_iso_f_element(n, kron, lc):
    """
    Determine full set of valid axis values for each rank given a string of Kronecker deltas and Levi-Civita symbols
    for individual doubles/triples of ranks. This is an element of the f (g) vector

    n: Total order (strictly speaking redundant when all ranks are represented in the Kronecker/Levi-Civita collection)
    kron: List of Kronecker delta valid axis values for given ranks (see mdk())
    lc: List of two lists (coefficient +1 and -1): Levi-civita valid axis values for given ranks (see mdl())

    Returns two lists of lists:
    - The first list contains elements to be considered with a factor +1 ("forwards" permutations)
    - The second list contains elements to be considered with a factor -1 ("backwards" permutations)
    - Note that the second list may be empty if no elements are to be subtracted (i.e. no Levi-Civita)
    """

    # Make two lists of lists in iso_f: One for addition and another for subtraction
    iso_f = []

    # Call meso_iso_f: iso_f argument is a dummy recursion seed
    iso_f.append(meso_iso_f_element(kron, [['dummy' for i in range(n)]]))

    # Are there only Kronecker deltas to take care of? If so, then no subtraction: Make tuples of valid axis combs. and return
    if len(lc) == 0:

        # Check if all ranks were covered
        for i in iso_f[0]:
             if 'dummy' in i:
                 raise AssertionError('Not all ranks were covered by Kronecker dictated combinations')

        return [[tuple(i) for i in iso_f[0]], []]


    # If not, proceed to do Levi-Civita handling
    iso_f.append(copy.deepcopy(iso_f[0]))

    # FIXME: Test this to end of fn specifically: No big reason to suspect problems but I don't recall how this worked
    bperm = [[0], [1]]
    for i in range(len(bperm)):

        this_lc = []
        for j in range(len(bperm[i])):
            this_lc.append(lc[j][bperm[i][j]])

        iso_f[sum(bperm[i]) % 2] = meso_iso_f_element(this_lc, iso_f[sum(bperm[i]) % 2])

    # Check if all ranks were covered
    for i in iso_f[0]:
        if 'dummy' in i:
            raise AssertionError('Not all ranks were covered by Kronecker dictated combinations')

    for i in iso_f[1]:
        if 'dummy' in i:
            raise AssertionError('Not all ranks were covered by Kronecker dictated combinations')

    # Make tuples of valid axis combs. and return
    return [[tuple(i) for i in iso_f[0]], [tuple(i) for i in iso_f[1]]]

def meso_iso_f_element(dicts, iso_f):
    """
    Helper function: Take list of Kronecker or Levi-Civita (resp. mdk() or mdl(), see
    return structure there) dictionaries of rank : element pairs and return a set of valid collected axis combinations

    Returns a list of lists: (Partially or fully covered (depending on completeness of dicts in covering ranks)) lists of
    valid axis combinations
    """

    # FIXME: This appears to fill ranks according to the dictionaries
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
        result = meso_iso_f_element(dicts[1:len(dicts)], iso_f)
        return result

    else:

        return iso_f

def get_iso_f(n):
    """
    Get an f (or also called g when applied to microscopic part) term

    n: Requested tensor rank

    Returns: List of f elements (each of make_iso_f_element structure, see that routine for specification)
    """

    if n == 2:

        return [
            make_iso_f_element(2, [mdk(0, 1)], []),
        ]


    elif n == 3:

        return [
            make_iso_f_element(3, [], [mdl(0, 1, 2)]),
        ]

    elif n == 4:

        return [
            make_iso_f_element(4, [mdk(0, 1), mdk(2, 3)], []),
            make_iso_f_element(4, [mdk(0, 2), mdk(1, 3)], []),
            make_iso_f_element(4, [mdk(0, 3), mdk(1, 2)], [])
        ]

    elif n == 5:

        return [
            make_iso_f_element(5, [mdk(3, 4)], [mdl(0, 1, 2)]),
            make_iso_f_element(5, [mdk(2, 4)], [mdl(0, 1, 3)]),
            make_iso_f_element(5, [mdk(2, 3)], [mdl(0, 1, 4)]),
            make_iso_f_element(5, [mdk(1, 4)], [mdl(0, 2, 3)]),
            make_iso_f_element(5, [mdk(1, 3)], [mdl(0, 2, 4)]),
            make_iso_f_element(5, [mdk(1, 2)], [mdl(0, 3, 4)])
        ]

    elif n == 6:

        return [
            make_iso_f_element(6, [mdk(0, 1), mdk(2, 3), mdk(4, 5)], []),
            make_iso_f_element(6, [mdk(0, 1), mdk(2, 4), mdk(3, 5)], []),
            make_iso_f_element(6, [mdk(0, 1), mdk(2, 5), mdk(3, 4)], []),
            make_iso_f_element(6, [mdk(0, 2), mdk(1, 3), mdk(4, 5)], []),
            make_iso_f_element(6, [mdk(0, 2), mdk(1, 4), mdk(3, 5)], []),
            make_iso_f_element(6, [mdk(0, 2), mdk(1, 5), mdk(3, 4)], []),
            make_iso_f_element(6, [mdk(0, 3), mdk(1, 2), mdk(4, 5)], []),
            make_iso_f_element(6, [mdk(0, 3), mdk(1, 4), mdk(2, 5)], []),
            make_iso_f_element(6, [mdk(0, 3), mdk(1, 5), mdk(2, 4)], []),
            make_iso_f_element(6, [mdk(0, 4), mdk(1, 2), mdk(3, 5)], []),
            make_iso_f_element(6, [mdk(0, 4), mdk(1, 3), mdk(2, 5)], []),
            make_iso_f_element(6, [mdk(0, 4), mdk(1, 5), mdk(2, 3)], []),
            make_iso_f_element(6, [mdk(0, 5), mdk(1, 2), mdk(3, 4)], []),
            make_iso_f_element(6, [mdk(0, 5), mdk(1, 3), mdk(2, 4)], []),
            make_iso_f_element(6, [mdk(0, 5), mdk(1, 4), mdk(2, 3)], [])
        ]

    else:

        raise ValueError('Unsupported get_iso_f order:', n)


def get_iso_mat(n):
    """
    Get the matrix M used in orientational averaging between macroscopic and microscopic vectors (here tabulated)

    n: Orientational average rank parameter
    """

    if n == 2:
        #FIXME: Factor 1/3? NOW UPD

        return np.array([[1.0]]) / 3.0

    elif n == 3:

        # FIXME: Factor 1/6? NOW UPD
        return np.array([[1.0]]) / 6.0

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

def getGeneralPolarizationAveragingExpression(rank: int, laser_pol: np.array):
    """
    Get the arrays of indices to be summed and a prefactor of the averaging expression
    Functioning for non-electric dipole polarization properties not investigated/supported

    rank: Integer: The rank of the orientational average. For electric dipole polarization properties, the rank is
    certainly equal to the number of ranks underlying the laser polarization term. NOTE: There might be further issues
    to consider for non-dipole approximation operators and so I keep the rank argument for now, noting that it might
    be found redundant after this consideration.

    laser_pol: The "laser polarization" term: A term of the form A * f from JCP 67, 5026. Concerns the "macroscopic"
    part of the averaging and is an experiment-specific (pulse-set-up-specific) quantity.

    The purpose of this routine is to determine which elements of a "microscopic" tensor w.r.t "molecular axes" must
    be combined (and with which coefficients) to form the appropriate orientational average combination under the extant
    polarization setup of the experiement; i.e., in the contraction A * f * M * g * P (see JCP 67, 5026), form the
    product A * f * M * g expressed a linear combination of elements of rank N tensor P over Cartesian axis elements.
    A routine possessing P can then assemble the finished average by evaluating this linear combination.

    Returns: A dictionary {Component tuple 1: coefficient, Component tuple 2: coefficient, ...}
    """

    # 1: Fails:
    # Fail if the order is > 6
    if rank > 6:
        raise ValueError('Averaging ranks > 6 not supported')
    elif rank < 2:
        raise ValueError('Averaging ranks < 2 not valid')

    # 2: Get M, g (f)

    iso_mat_m = get_iso_mat(rank)
    iso_vec_f = get_iso_f(rank)

    #print('Laser polarization term:', laser_pol)
    #print('M:', iso_mat_m)
    #print('f:', iso_vec_f)

    # 3. Form the linear combination recipe dot(A * f, M * g)

    linear_combination = {}

    for i in range(len(iso_vec_f)):
        for j in range(len(iso_vec_f)):

            # Elements for addition
            for k in iso_vec_f[j][0]:

                # If already registered, update coefficient
                if k in linear_combination:
                    linear_combination[k] += laser_pol[i] * iso_mat_m[i, j]

                # Otherwise make new entry
                else:
                    linear_combination[k] = laser_pol[i] * iso_mat_m[i, j]

            # Elements for subtraction
            for k in iso_vec_f[j][1]:

                # If already registered, update coefficient
                if k in linear_combination:
                    linear_combination[k] -= laser_pol[i] * iso_mat_m[i, j]

                # Otherwise make new entry
                else:
                    linear_combination[k] = -1.0 * laser_pol[i] * iso_mat_m[i, j]

    #print('Linear combination result:', linear_combination)

    # Prune zero elements
    marked_for_deletion = []

    for i in linear_combination:
        if linear_combination[i] == 0.0:
            marked_for_deletion.append(i)

    for i in marked_for_deletion:
        del linear_combination[i]

    return linear_combination

