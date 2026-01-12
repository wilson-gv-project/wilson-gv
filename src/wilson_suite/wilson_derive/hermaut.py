import copy
from .abstractions import HarmOscStateSymbolic, VibDiffTerm, ResonanceCondition
from .response_terms import VibPerturbedTerm, VibContribTerm
from fractions import Fraction

# FIXME: The routines in this file need verification and harmonization with respect to theory manuscript

def all_uneq_walks(curr_walk: list, N: int, unstarted: list, closed: list, result: list):
    """
    Generate all Hermite "walks" for a given number of pairs of unique raise/lower operators
    Tail-recursive

    curr_walk: List of [a, b] pairs: Walk currently being generated (a is the normal mode
    index and b is "raise or lower" flag (resp. 1 and -1)
    N: Total number of unique raise/lower pairs
    Unstarted: List of pair references not yet encountered in current walk
    Closed: List of pair references already represented with both raise and lower in current walk  ("closed")
    Result: List: Results accumulator for all curr_walks at end of recursion
    """

    # Termination condition
    if len(curr_walk) == N * 2:

        result.append(curr_walk)

    # Otherwise recurse further
    else:

        for i in curr_walk:

            if not (i[0] in closed):
                new_walk = copy.deepcopy(curr_walk)
                new_closed = copy.deepcopy(closed)

                new_walk.append([i[0], 1])
                new_closed.append(i[0])
                all_uneq_walks(new_walk, N, unstarted, new_closed, result)

        if not (unstarted == []):
            new_walk = copy.deepcopy(curr_walk)
            new_walk.append([unstarted[0], -1])
            new_unstarted = copy.deepcopy(unstarted)
            new_unstarted.pop(0)

            all_uneq_walks(new_walk, N, new_unstarted, closed, result)

    return

def go_for_a_walk(term: VibContribTerm, walk):
    """
    Traverse a Hermite walk with a term and return a collection of (vibState, harmOscState) pairs
    and normal mode indices for each derivative

    term: VibContribTerm instance being "walked"
    walk: The raise/lower progression (the "walk") w.r.t. which term will be subjected (see curr_walk)
    definition in all_uneq_walks
    """

    res_states = {}
    res_deriv_inds = []

    # Assign ground-state vibState to ground-state harmOscState
    res_states[term.ints[-1].ket.s] = HarmOscStateSymbolic([])

    # Reversing order of integrals for walk
    ints = copy.deepcopy(term.ints)
    ints.reverse()

    # Walk location counter
    w = round(sum([i.prop.dord for i in term.ints]))

    # Walk through each integral
    # Start from end of walk
    for i in ints:

        # "Starting" quanta from the ket state of this integral
        quanta = copy.deepcopy(res_states[i.ket.s].q)
        this_deriv_inds = []

        # Now go through the derivative order for this integral and change the quanta according
        # to the appropriate part of the walk to determine the bra state
        for j in range(i.prop.dord):

            # If there was already one excitation corresponding to this index, then
            # the next encounter must be deexcitation, so remove this index
            if walk[w-1][0] in quanta:
                quanta.remove(walk[w-1][0])

            # Otherwise make a new entry for a quantum of this index
            else:
                quanta.append(walk[w-1][0])

            this_deriv_inds.insert(0, walk[w-1][0])

            # Decrement the walk location
            w -= 1

        # If at end verify that the resulting state is the ground state and do not make a new state entry
        if (w == 0):

            if not quanta == []:
                raise AssertionError('Error: Final bra state was not walked back to ground state')

        # Otherwise make a new state entry
        else:
            res_states[i.bra.s] = HarmOscStateSymbolic(quanta)

        # Record the used derivative indices for this integral
        res_deriv_inds.insert(0, copy.deepcopy(this_deriv_inds))

    return res_states, res_deriv_inds

def do_hermaut(term: VibContribTerm, inds: list):
    """
    Do a Hermite evaluation of a VibContribTerm to generate a list of valid VibPerturbedTerm instances resulting
    from the "Hermite walks" evaluation

    TODO: Rework for re-resolved states: Make this keep track of orig states too
    (instead of pure harm osc, go with "character of...")
    Evaluator must then loop over states and their character (so both loop over states and nm indices)
    Non-variationally-re-resolved states will trivially reduce (state, nm) summation to nm like before

    term: VibContribTerm: the term to be evaluated
    inds: list: Symbolic normal mode indices

    Returns a list of VibPerturbedTerm instances
    """

    # Count up derivative orders: Return empty if odd

    sumord = round(sum([i.prop.dord for i in term.ints]))

    if not ((sumord % 2) == 0):
        return []

    # Test if first bra and last ket are ground state
    if not term.ints[-1].ket.is_ground:
        raise NotImplementedError('Having the last integral ket state not be the ground state is not implemented')

    if not term.ints[0].bra.is_ground:
        raise NotImplementedError('Having the first integral bra state not be the ground state is not implemented')

    # Test if state progression is telescopic and state indices are unique
    is_telescopic = True
    are_states_unique = True
    encountered_states = []

    for i in range(len(term.ints)):

        if not(term.ints[i-1].ket.s == term.ints[i].bra.s):
            is_telescopic = False

        if term.ints[i-1].ket.s in encountered_states:
            are_states_unique = False
        else:
            encountered_states.append(term.ints[i-1].ket.s)

    if not (is_telescopic):
        raise NotImplementedError('Non-telescopic state progressions in Hermite walk not yet implemented')
    if not (are_states_unique):
        raise AssertionError('Error: More than one state has the same index')

    # If even, then get all walks
    first_walk = []
    walks = []
    closed = []
    all_uneq_walks(first_walk, round(sumord/2), inds[:round(sumord/2)], closed, walks)

    finished_terms = []

    # Go on each walk and construct a VibPerturbedTerm instance
    for i in walks:

        # Go on this walk
        res_states, res_deriv_inds = go_for_a_walk(term, i)

        # Go through all integrals and perform mbu check
        # Since assumed telescopic it is sufficient to check all bras but will redundantly
        # check kets too just to be on the safe side (if missing mbu parameter for a bra or a ket of them)
        # After this check then all clear for the rest of treatment since every state is represented in the integrals
        for j in term.ints:
            if not(j.bra.mbuFulfilled(res_states)):
                raise AssertionError('Not all must-be-unequal-to states are unequal to their assumed-unequal-to states')
            if not(j.ket.mbuFulfilled(res_states)):
                raise AssertionError('Not all must-be-unequal-to states are unequal to their assumed-unequal-to states')

        # Prepare VibPerturbedTerm instance

        # Make properties and insert the identified differentiation indices
        new_props = []

        for j in range(len(term.ints)):
            new_prop = copy.deepcopy(term.ints[j].prop)
            new_prop.setInds(res_deriv_inds[j])
            new_props.append(new_prop)

        # Make frequency (difference) terms
        new_freqterms = []

        # Remake existing frequency (difference) terms in terms of harmonic oscillator quanta
        for j in term.freqdiff:
            new_freqterms.append(VibDiffTerm(copy.deepcopy(res_states[j.sl.s]), copy.deepcopy(res_states[j.sr.s]), is_pert_wf_diff=j.is_pert_wf_diff))

        # Add new freq factors from harm osc treatment
        for j in inds[:round(sumord/2)]:
            new_freqterms.append(VibDiffTerm(HarmOscStateSymbolic([j]), HarmOscStateSymbolic([])))

        # Update coefficient w.r.t. the 1/(2 w_a)-style factors
        new_coeff = term.coeff * Fraction(1, 2**(round(sumord/2)))

        # Remake resonance conditions in terms of harmonic oscillator quanta
        new_res = []

        for j in term.res:
            new_res.append(ResonanceCondition(
                VibDiffTerm(copy.deepcopy(res_states[j.diff.sl.s]), copy.deepcopy(res_states[j.diff.sr.s])),
                j.pf, j.id))

        # Create the VibPerturbedTerm instance
        finished_terms.append(VibPerturbedTerm(new_coeff, new_props, new_freqterms, new_res))

    return finished_terms
