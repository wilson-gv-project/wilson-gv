import copy
from .abstractions import harmOscState, vibPerturbedTerm, vibDiffTerm, resonanceCondition
from fractions import Fraction

# Walk L to R (not R to L)
def all_uneq_walks(curr_walk, N, unstarted, closed, result):
    if len(curr_walk) == N * 2:

        result.append(curr_walk)

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

# Traverse a Hermite walk with a term and return a collection of (vibState, harmOscState) pairs
# and normal mode indices for each derivative
def go_for_a_walk(term, walk):

    res_states = {}
    res_deriv_inds = []

    # Assign ground-state vibState to ground-state harmOscState
    res_states[term.ints[-1].ket.s] = harmOscState([])

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
            res_states[i.bra.s] = harmOscState(quanta)

        # Record the used derivative indices for this integral
        res_deriv_inds.insert(0, copy.deepcopy(this_deriv_inds))

    return res_states, res_deriv_inds


# TODO: Rework: Make this keep track of orig states too (instead of pure harm osc, go with "character of...")
# Evaluator must loop over states and their character (so both loop over states and nm indices)
# Non-variationally-re-resolved states will trivially reduce (state, nm) summation to nm like before
def do_hermaut(term, inds):

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

    # Go on each walk and construct a vibPerturbedTerm instance
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

        # Prepare vibPerturbedTerm instance

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
            new_freqterms.append(vibDiffTerm(copy.deepcopy(res_states[j.sl.s]), copy.deepcopy(res_states[j.sr.s]), is_pert_wf_diff=j.is_pert_wf_diff))

        # Add new freq factors from harm osc treatment
        for j in inds[:round(sumord/2)]:
            new_freqterms.append(vibDiffTerm(harmOscState([j]), harmOscState([])))

        # Update coefficient w.r.t. the 1/(2 w_a)-style factors
        new_coeff = term.coeff * Fraction(1, 2**(round(sumord/2)))

        # Remake resonance conditions in terms of harmonic oscillator quanta
        new_res = []

        for j in term.res:
            new_res.append(resonanceCondition(
                vibDiffTerm(copy.deepcopy(res_states[j.diff.sl.s]), copy.deepcopy(res_states[j.diff.sr.s])),
                j.pf, j.id))

        # Create the vibPerturbedTerm instance
        finished_terms.append(vibPerturbedTerm(new_coeff, new_props, new_freqterms, new_res))


    return finished_terms
