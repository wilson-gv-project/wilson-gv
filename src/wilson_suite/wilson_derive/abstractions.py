import copy

from dataclasses import dataclass

@dataclass
class QOperator:
    """
    Quantum-mechanical operator denoting an interaction with the field

    o: Operator label (integer)

    The next arguments are optional since derivations may be carried out without their specification and generating
    the specific terms arising from a choice of multipole expansion regime can be carried out at a later stage
    (avoiding repetitive derivations of similarly structured terms)

    op_type: String (default None): Optional specification of operator type
    ax: Tuple of integers (default None): Optional specification of Cartesian axis composition of operator (for the
    electric dipole operator, this would be (3,), while for e.g. the electric quadrupole operator, this would
    be (3, 3))
    """

    o: int
    op_type: str = None
    ax: tuple = None

    def __post_init__(self):

        if not(isinstance(self.o, int)):
            raise TypeError('All operator labels in qOperator must be integers')

        if self.op_type is not None:
            if not(isinstance(self.op_type, str)):
                raise TypeError('Operator type argument must be string if specified')

        if self.ax is not None:
            if not(isinstance(self.ax, tuple)):
                raise TypeError('Axis argument must, if specified, be a tuple of integers')
            else:
                for i in self.ax:
                    if not isinstance(i, int):
                        raise TypeError('Axis argument must, if specified, be a tuple of integers')

    def __repr__(self):
        return f'QOperator(o = {self.o}, op_type = {self.op_type}, ax = {self.ax})'
    def __hash__(self):
        return hash( ( self.o, self.op_type, self.ax ) )
    def __eq__(self, other):
        if isinstance(other, QOperator):
            return self.o == other.o and self.ax == other.ax and self.op_type == other.op_type
        return False
    
class HarmOscStateSymbolic:
    """
    Symbol-described harmonic oscillator state class
    """

    def __init__(self, q: list|tuple):
        """
        q: list (or tuple) of normal mode index quanta
        Ground state: q = []
        One-quantum b: q = ['b']
        Three-quantum a,a,b: q = ['a', 'a', 'b']
        """

        if not (isinstance(q, list) or isinstance(q, tuple)):
            raise TypeError('Harmonic oscillator state quanta must be represented as a list or tuple of characters')

        for i in q:

            if not isinstance(i, str):
                raise TypeError('Harmonic oscillator state quanta must be represented as a list or tuple of characters')

            if not len(i) == 1:
                raise TypeError('Harmonic oscillator state quanta must be represented as a list or tuple of characters')

        # Sort
        self.q = sorted(q)

    def __repr__(self):
        return f'omega_{self.q}'

    def h(self):
        """
        Hash function
        """
        return hash(tuple(self.q))

class VibStateSymbolic:
    """
    Symbol-described vibrational state class (less specific than HarmOscStateSymbolic)
    """

    def __init__(self, s: str, mbu: list=[], is_ground: bool=False):
        """
        s: State label (string)
        mbu: The present state must be unequal to state(s) whose labels are given in this list
        is_ground: Is this state always the ground vibrational state?
        """

        try:
            h = hash(s)
        except TypeError:
            raise TypeError('State label must be hashable')

        self.s = s

        if not isinstance(mbu, list):
            raise TypeError('Must-be-unequal argument must be a list')

        self.mbu = mbu

        if not isinstance(is_ground, bool):
            raise TypeError('The is_ground argument must be a Boolean')

        self.is_ground = is_ground

    def mbuFulfilled(self, states_as_quanta: dict) -> bool:
        """
        Check if states wrt. which the present state is required to be unequal are in fact unequal

        states_as_quanta: dictionary {state_label: quanta, ...}
        """

        if self.mbu == []:
            return True

        if not self.s in states_as_quanta:
            raise AssertionError('Must-be-unequal check with insufficient information about self state')

        for i in self.mbu:

            if not i.s in states_as_quanta:
                raise AssertionError('Must-be-unequal check with insufficient information about target state')

            if sorted(states_as_quanta[i.s].q) == sorted(states_as_quanta[self.s].q):
                return False

        return True

    def h(self):
        """
        Hash function
        """
        return hash( ( self.s, tuple([i.s for i in self.mbu]), self.is_ground ) )

class VibDiffTerm:
    """
    Class representing (inverse of) vibrational energy level difference
    """

    def __init__(self, sl=None, sr=None, is_pert_wf_diff=False):
        """
        sl: VibStateSymbolic or HarmOscStateSymbolic instance: Bra ("left-hand") state
        sr: VibStateSymbolic or HarmOscStateSymbolic instance: Ket ("right-hand") state
        is_pert_wf_diff: Boolean: Flag: Does this term come from an expression for
        a perturbed (vibrational) wavefunction? (Alternative: from Hermite integration)
        """

        # Must be both vibState or both harmOscState instances
        if not (isinstance(sl, VibStateSymbolic) and isinstance(sr, VibStateSymbolic) or
                isinstance(sl, HarmOscStateSymbolic) and isinstance(sr, HarmOscStateSymbolic)):
            raise TypeError('Both sl and sr must be either both VibStateSymbolic instances or both HarmOscStateSymbolic instances')

        self.sl = sl
        self.sr = sr

        if not isinstance(is_pert_wf_diff, bool):
            raise TypeError('is_pert_wf_diff must be a Boolean')

        self.is_pert_wf_diff = is_pert_wf_diff

    def __repr__(self):
        return f'[sl={self.sl}, sr={self.sr}], pert_wf={str(self.is_pert_wf_diff)[0]}'

    def __hash__(self):
        return hash( ( self.sl.h(), self.sr.h() ) )
    
    def __iter__(self):
        """Make the class iterable over sl and sr"""
        yield self.sl
        yield self.sr

    def to_latex(self):
        if isinstance(self.sl, VibStateSymbolic):
            bra = self.sl.s
            ket = self.sr.s

        elif isinstance(self.sl, HarmOscStateSymbolic):
            bra = self.sl.q
            ket = self.sr.q

        return ','.join([f"{'+'.join(bra)}",f"{'+'.join(ket)}"])

    def present(self):
        """
        Formatted printing of own attributes
        """
        print('   >> vibDiffTerm presents:')
        print('self.is_pert_wf_diff', self.is_pert_wf_diff,'\n')
        print('Freq diff term')

        if isinstance(self.sl, VibStateSymbolic):
            print('Bra state', self.sl.s)
            print('Ket state', self.sr.s)

        elif isinstance(self.sl, HarmOscStateSymbolic):
            print('Bra state', self.sl.q)
            print('Ket state', self.sr.q)
        if not self.is_pert_wf_diff:
            print('----')

    def h(self):
        """
        Hashing function
        """
        return hash( ( self.sl.h(), self.sr.h() ) )



class ResonanceCondition:
    """
    Resonance condition class

    Convention: Perturbing frequencies to be subtracted
    """

    def __init__(self, diff: VibDiffTerm, pf: list|tuple=[], id=None):
        """
        diff: VibDiffTerm instance: State energy level difference: States must here be HarmOscStateSymbolic or VibStateSymbolic
              - Several methods will only work with the states in HarmOscStateSymbolic form
              - However, having states as VibStateSymbolic instances is relevant in earlier stages of the overall term
              derivation process
        pf: List or tuple: Perturbing field frequency labels (their sum to be subtracted when evaluating)
        id: Optional integer id term for potential later handling of grouped
        resonance conditions in lineshape evaluation
        """

        # Energy difference
        if not(isinstance(diff, VibDiffTerm)):
            raise TypeError('The energy difference must be a VibDiffTerm instance')

        self.diff = diff

        if not(isinstance(pf, list) or isinstance(pf, tuple)):
            raise TypeError('Perturbing frequency labels must be list or tuple of strings or integers')
        for i in pf:
            if not (isinstance(i, str) or isinstance(i, int)):
                raise TypeError('Perturbing frequency labels must be list or tuple of strings or integers')

        self.pf = pf

        if id is not None:
            if not isinstance(id, int):
                raise TypeError('Optional identifier must, if specified, be an integer')

        self.id = id

    @classmethod
    def make_from_tuples(cls, left_state: tuple, right_state: tuple, pert_freqs: tuple):
        """
        res_cond_dict: Dictionary {'left': tuple, 'right': tuple, 'pert_freqs': tuple}
        """
        left_state = HarmOscStateSymbolic(list(left_state))
        right_state = HarmOscStateSymbolic(list(right_state))
        diff = VibDiffTerm(sl=left_state, sr=right_state)

        return cls(diff=diff, pf=pert_freqs)

    @property
    def pf_dict(self):
        return {i.strip('-'): -1 if '-' in i else 1 for i in self.pf}

    def __repr__(self):
        return f'ResCond(diff = {self.diff}, pf = {self.pf}, id = {self.id})'

    def present(self):
        """
        Formatted printing of own attributes
        """
        print('   >> ResonanceCondition presents:')

        print('Resonance condition states')
        self.diff.present()
        print('Resonance condition pert freqs', str(self.pf))
        print('----')

    def to_latex(self):
        pf = self.pf
        if all(isinstance(i, int) for i in self.pf):
            pf = [str(i) for i in self.pf]
        # reversing sign
        upd_pf_sign = ['-'+ax if '-' not in ax else '+'+ax.strip('-') for ax in pf]
        pf_string = ''.join(upd_pf_sign)
        return rf'(\omega_{{{self.diff.to_latex()}}} {pf_string})'

    def h(self) -> int:
        """
        Hashing function
        """
        return hash(( self.diff.h(), tuple(self.pf), self.id ))

    def uvCancels(self, cfs_uv: dict, tol: float=0.0) -> bool:
        """
        Do the UV/VIS parts of my perturbing frequencies cancel?

        cfs_uv: UV/VIS part of my perturbing frequencies
        tol: tolerance (default: 0.0)
        NOTE: Consequences if tolerance != 0.0 used are not yet supported/investigated
        """

        acc = 0.0

        for i in self.pf:
            sgn = (i > 0) - (i < 0)
            acc += sgn * cfs_uv[sgn*i]

        sgnacc = (acc > 0) - (acc < 0)

        return ((sgnacc * acc) <= tol)

    def netStateSign(self, instead_trimmed: VibDiffTerm=None) -> int:
        """
        Return 1 if evaluation of myself is a positive number, -1 if negative, or (special flag) -3 if indeterminate

        instead_trimmed (default None): Instead evaluate wrt. a trimmed resonance (trimmed according to
        other known information about equivalences)
        """

        # VibDiffTerm has its own type check where sl and sr must be same class, therefore checking one is sufficient here
        if not(isinstance(self.diff.sl, HarmOscStateSymbolic)):
            raise TypeError('The netStateSign method works only with a VibDiffTerm composed of HarmOscStateSymbolic instances')


        if instead_trimmed is None:

            q_bra_net = copy.deepcopy(self.diff.sl.q)
            q_ket_net = copy.deepcopy(self.diff.sr.q)

        else:

            q_bra_net = copy.deepcopy(instead_trimmed.sl.q)
            q_ket_net = copy.deepcopy(instead_trimmed.sr.q)

        len_bra = len(q_bra_net)
        len_ket = len(q_ket_net)

        term = False

        while not term:
            for i in q_bra_net:
                if i in q_ket_net:
                    q_bra_net.remove(i)
                    q_ket_net.remove(i)
            new_len_bra = len(q_bra_net)
            new_len_ket = len(q_ket_net)
            if (len_bra == new_len_bra) and (len_ket == new_len_ket):
                term = True
            len_bra = new_len_bra
            len_ket = new_len_ket

        if len_bra == 0:
            if len_ket > 0:
                return -1
            else:
                return 0
        else:
            # Special flag for indeterminate sign
            if len_ket > 0:
                return -3
            else:
                return 1

    def couldBeResonantWithFieldByRanges(self, pulse_freq_spans: dict) -> bool:
        """
        Unfinished/untested:
        Determine whether it's possible (return True) that this combination of field
        (as specified by frequency ranges for each pulse) and states might be resonant.
        Return False if this is definitely not possible.

        pulse_freq_spans: Dictionary {pulse #i: [lower range, upper range], ...}
        """

        # VibDiffTerm has its own type check where sl and sr must be same class, therefore checking one is sufficient here
        if not (isinstance(self.diff.sl, HarmOscStateSymbolic)):
            raise TypeError(
                'The netStateSign method works only with a VibDiffTerm composed of HarmOscStateSymbolic instances')

        pfr_curr = {}

        for i in self.pf:

            sgn = (i > 0) - (i < 0)

            if not(sgn*i in pulse_freq_spans):
                pfr_curr[i] = [float('-inf'), float('inf')]

            else:

                if not(i == sgn*i):
                    pfr_curr[i] = sorted([-1*pulse_freq_spans[sgn*i][0], -1*pulse_freq_spans[sgn*i][1]])

                else:
                    pfr_curr[i] = sorted([-1*j for j in pfr_curr[i]])

        pf_total_range = [sum([pfr_curr[i][0] for i in self.pf]), sum([pfr_curr[i][1] for i in self.pf])]

        if pf_total_range[0] < 0.0:

            if pf_total_range[1] < 0.0:
                pf_overall_sign = -1
                pf_closed = False

            elif pf_total_range[1] == 0.0:
                pf_overall_sign = -1
                pf_closed = True

            elif pf_total_range[1] > 0.0:
                return True

        if pf_total_range[0] == 0.0:

            if pf_total_range[1] == 0.0:
                pf_closed = True

            elif pf_total_range[1] > 0.0:
                pf_overall_sign = 1
                pf_closed = True

        if pf_total_range[0] > 0.0:
            pf_overall_sign = 1
            pf_closed = False

        sd_overall_sign = self.netStateSign()

        # Special flag for indeterminate sign
        if sd_overall_sign == -3:
            return True

        if sd_overall_sign == -1:

            if pf_overall_sign == 1:
                return True

            return False

        elif sd_overall_sign == 1:

            if pf_overall_sign == -1:
                return True

            return False

        elif sd_overall_sign == 0:

            if pf_closed:
                return True

            return False

    # FIXME: Functionality not general yet
    def couldBeResonantWithFieldByConditions(self, magn_conditions: list|tuple, given_prev_res=None):
        """
        Determine whether it's possible (return True) that this combination of field
        (as specified by magnitude conditions) and states might be resonant.
        Return False only if this is definitely not possible.

        Clarification: Returns True if answer is indeterminate or if arguments were not of the required form, or if
        the present instance's "pf" argument is not given in terms of (signed integer) pulse references
        (i.e. "If I can't find a definite 'no' I will return 'yes'" since it as far as I can tell resonance
        is still possible)

        Optionally, with the given_prev_res argument specified, determine the answer to the same question as above
        given that the resonance condition given by given_prev_res was satisfied.

        magn_conditions: List/tuple (here list) [[A = signed freq i, B = signed freq j, ...],
                               [C = signed freq k, ...],
                                ...], signifying that
                         A + B + ... > 0 (by a significant margin)
                         C + ... > 0 (ibid.)
                         ...

        A "significant margin" is here not unambiguous, but one useful definition can be "greater than the longest
        distance where the lineshape is visible around a lineshape"

        given_prev_res (default: None): ResonanceCondition instance telling (if any) the preceding
        resonance condition that was satisfied. Optional argument which will then adapt the functionining of this
        method as described at the beginning of this documentation string.:
        If the contents of given_prev_res for each attribute are a subset of the present instance,
        then these respective parts can be trimmed from the present instance and the resonance possibility determined
        (in the same way) with respect to the trimmed condition.

        This routine could be extended as follows:
        TODO: Can also combine with freq ranges (first rm according to conditions and then actual ranges for remaining?)
        TODO: For now just walking through conditions. Later, could apply all combinations of conditions
        to exhaust opportunities for eliminating all
        """

        # VibDiffTerm has its own type check where sl and sr must be same class, therefore checking one is sufficient here
        if not(isinstance(self.diff.sl, HarmOscStateSymbolic)):
            raise TypeError(
                'The netStateSign method works only with a VibDiffTerm composed of HarmOscStateSymbolic instances')

        # Catching if perturbing frequencies not specified as (signed integer) pulse references and returning True
        # (cannot rule out resonance)
        # This functionality currently not supported for other specifications of perturbing freqs. (e.g. axis labels)
        # This is not currently a hindrance to the overall functioning because this method is currently invoked
        # while perturbing freqs. are still specified in terms of pulse references
        for i in self.pf:
            if not isinstance(i, int):
                return True

        # Corresponding catch for magnitude conditions: A usable magnitude conditions set must be
        # a list of lists of (signed integer) pulse references; otherwise, return True (cannot rule out resonance)
        for i in magn_conditions:
            if not (isinstance(i, list) or isinstance(i, tuple)):
                return True
            for j in i:
                if not isinstance(j, int):
                    return True

        # If prev_res is not a ResonanceCondition instance, it cannot be used so return True (cannot rule out resonance)
        if given_prev_res is not None:
            if not isinstance(given_prev_res, ResonanceCondition):
                return True

        # Use conditions to trim blocks of definite sign
        # If all blocks are rmvd, see if results point to definite overall sign
        # If not all rmvd or not all definite sign, return as indeterminate
        pf_test = copy.deepcopy(self.pf)
        diff_test  = copy.deepcopy(self.diff)

        # For now simply trim last resonance from present
        if given_prev_res is not None:

            all_in = True

            for i in given_prev_res.diff.sl.q:
                if not(i in diff_test.sl.q):
                    all_in = False

            for i in given_prev_res.diff.sr.q:
                if not(i in diff_test.sr.q):
                    all_in = False

            for i in given_prev_res.pf:
                if not(i in pf_test):
                    all_in = False

            if all_in:

                for i in given_prev_res.diff.sl.q:
                    diff_test.sl.q.remove(i)

                for i in given_prev_res.diff.sr.q:
                    diff_test.sr.q.remove(i)

                for i in given_prev_res.pf:
                    pf_test.remove(i)


        # Start off sign as unset
        pf_overall_sign = None

        for i in magn_conditions:

            if len(i) > 0:

                rmd = False

                # Try one way (pulse signs as given), if match then rm and store sign, if no match then try other way
                match = True
                for j in i:
                    if match:
                        if j not in pf_test:
                            match = False

                if match:
                    for j in i:
                        pf_test.remove(j)
                    rmd = True

                    # Setting for first time. Note that this overall sign includes the convention
                    # that perturbing frequencies in a resonance condition are subtracted from the state energy
                    # level differences. Example:
                    # magn_conditions is [[-1, 2]] (i.e. -w1 + w2 > 0)
                    # pf is [-1, 2]
                    # Therefore, pf_overall_sign is -1 (subtracting something known to be positive)
                    if pf_overall_sign is None:
                        pf_overall_sign = -1

                    # Conflicting with earlier condition so return True (cannot rule out resonance)
                    elif pf_overall_sign == 1:
                        return True

                # If condition not recognized first way, try the other way
                # (opposite pulse signs, condition is now < 0 instead of > 0).
                if not rmd:

                    match = True
                    for j in i:
                        if match:
                            if -1 * j not in pf_test:
                                match = False
                    if match:
                        for j in i:
                            pf_test.remove(-1 * j)

                        # Setting for first time (opposite sign since other way)
                        if pf_overall_sign is None:
                            pf_overall_sign = 1

                        # Conflicting with earlier condition so return True
                        elif pf_overall_sign == -1:
                            return True

        # Reaching this point of the routine means that all (if any) magnitude conditions that were met
        # were found to point to the same sign (and we have removed the corresponding pulse references from pf_test
        # Therefore, pf_test contains all "residual" pulse references (if any) for which the magnitude conditions could
        # not be used to determine a sign. Under the assumption that unsigned frequency ranges are always positive
        # (i.e. a reference to w_i is always positive, while a reference to -w_i is always negative), then,
        # if these remaining pulse references all have the same sign, and if either a) no magnitude conditions were
        # applied or b) the overall sign as determined by the magnitude conditions corresponds to the sign of these
        # remaining pulses, then we can determine an overall sign for the full collection of perturbing frequencies.

        # Initialize "sign of residual pulse references" as unset
        res_sgn = None

        # Find out if all residual pulse references have the same sign and if so, which
        if (len(pf_test) > 0):
            for i in pf_test:
                res_sgn_i = (i < 0) - (i > 0)

                # Setting for the first time
                if res_sgn is None:
                    res_sgn = res_sgn_i

                # If the current sign does not match previous signs, return True (cannot rule out resonance)
                elif not(res_sgn_i == res_sgn):
                    return True

        # This means no perturbing frequencies: Should not be encountered in practice but included for completeness
        # Returning True (cannot rule out resonance) to be on the safe side even though resonance could
        # in fact then frequently be ruled out
        if res_sgn is None and pf_overall_sign is None:
            return True

        else:
            # If no magnitude conditions applied, set sign to residual sign
            if pf_overall_sign is None:
                pf_overall_sign = res_sgn

            else:
                # If magnitude conditions applied and a residual sign was determined, they need to match in order
                # for resonance to be possible to rule out
                if res_sgn is not None:
                    if not res_sgn == pf_overall_sign:
                        return True

        # Get sign of state energy level difference
        sd_overall_sign = self.netStateSign(instead_trimmed=diff_test)

        # Special flag for indeterminate sign
        if sd_overall_sign == -3:
            return True

        # State energy level difference sign and perturbing frequencies must be opposite in order for
        # resonance to be possible
        if sd_overall_sign == -1:
            if pf_overall_sign == 1:
                return True
            return False

        elif sd_overall_sign == 1:
            if pf_overall_sign == -1:
                return True
            return False

        # If state difference is zero but pert freq is significantly pos or neg, then return False
        elif sd_overall_sign == 0:
            if pf_overall_sign == 1:
                return False
            elif pf_overall_sign == -1:
                return False

        return True


class LineShape:
    """
    Lineshape class
    For now just outline, not presently used
    """

    def __init__(self, rcs: tuple[ResonanceCondition], evaluator=None):
        """
        rcs: Resonance conditions associated with this lineshape

        evaluator: Possible associated evaluator function
        """

        # Must be tuple of resonanceCondition instances
        self.rcs = rcs

        # Possible evaluator function?
        self.evaluator = evaluator

class PolProp:
    """
    Polarization property differentiated zero or more times w.r.t. geometrical displacement
    Also used more generally as energy derivative
    TODO: refactor names of attributes so they are understandable and clear
    """

    def __init__(self, ops: list[QOperator], dord: int = 0):
        """
        ops: List of QOperator instances: Electromagnetic field coupling operators

        dord: Integer: Order of geometric differentiation
        """

        if not isinstance(ops, list):
            raise TypeError('Operator argument must be a list of QOperator instances')

        if not (all([isinstance(i, QOperator) for i in ops])):
            raise TypeError('All operator arguments in PolProp must be QOperator instances')

        self.ops = ops

        if not (isinstance(dord, int)):
            raise TypeError('Differentiation order in PolProp must be integer')
        self.dord = dord

        # To be used as list of characters symbolizing indices of differentiation
        self.inds = None

    def __eq__(self, other):
        if isinstance(other, PolProp):
            return set(self.ops) == set(other.ops) and self.dord == other.dord and self.inds == other.inds
        return False

    def __hash__(self):
        return hash((tuple([hash(i) for i in self.ops]), self.dord, tuple(self.inds)))

    def setDerivOrder(self, dord):
        """
        Set order of geometric differentiation

        dord: Integer: Order of differentiation
        """
        if not (isinstance(dord, int)):
            raise TypeError('Differentiation order in PolProp must be integer')

        self.dord = dord

    def setInds(self, inds: list[str]):
        """
        Set (symbolic) indices of geometric differentiation

        inds: List of characters symbolizing indices of differentiation
        """

        if not len(inds) == self.dord:
            raise ValueError('Normal mode indices must have same length as differentiation order')

        if not isinstance(inds, list):
            raise TypeError('Indices must be given as list of characters')
        for i in inds:
            if not isinstance(i, str):
                raise TypeError('Indices must be given as list of characters')
            if not len(i) == 1:
                raise TypeError('Indices must be given as list of characters')

        self.inds = sorted(inds)

    def h(self) -> int:
        """
        Hashing function
        """

        return hash((tuple([i.h() for i in self.ops]), self.dord, tuple(self.inds)))

    def epochContained(self, epochs: list[list[int]], op_ind_omega: int) -> bool:
        """
        Boolean: Do all electromagnetic operators in this property belong to the same pulse epoch?

        epochs: List [[epoch 1 pulse index 1, epoch 1 pulse ind. 2], [epoch 2 pulse ind. 1, ...], ...]
        op_ind_omega: Label of omega (detected field) operator
        """

        if not isinstance(epochs, list):
            raise TypeError('Epochs must be given as list of lists of integer pulse references')

        if not (all([isinstance(i, list) for i in epochs])):
            raise TypeError('Epochs must be given as list of lists of integer pulse references')

        for i in epochs:
            if not (all([isinstance(j, int) for j in i])):
                raise TypeError('Epochs must be given as list of lists of integer pulse references')

        if not isinstance(op_ind_omega, int):
            raise TypeError('Index of omega operator must be given as integer')

        my_op_ids = [i.o for i in self.ops]

        # If omega is one of the operators, all of the others must belong to the last epoch:
        # (Epochs are given according to incident pulses and do not contain any reference to the emitted wave)
        if op_ind_omega in my_op_ids:
            for i in my_op_ids:
                if not (i == op_ind_omega):
                    if not i in epochs[len(epochs) - 1]:
                        return False

        # Otherwise, all operators must belong to the same epoch but it need not be the last
        else:

            required_epoch = None

            for i in range(len(epochs)):
                if my_op_ids[0] in epochs[i]:
                    required_epoch = i

            for i in my_op_ids:
                if not i in epochs[required_epoch]:
                    return False

        # If not ruled out by prev checks, then the ops. in this PolProp are indeed epoch contained
        return True

    def present(self):
        """
        Formatted printing of own attributes
        """
        print('   >> polProp presents:')
        print('Operators:', [i.o for i in self.ops])
        print('Geo differentiation order:', self.dord)
        if self.inds is not None:
            print('Normal mode differentiation indices:', self.inds)

        print('----')

    def __repr__(self):
        return f'PolProp(ops = {self.ops}, dord = {self.dord}, (inds = {self.inds}))'

    def to_latex(self):
        from ..wilson_utils import common_labels
        numalpha = {common_labels.op_omega_label_int: common_labels.op_omega_label_greek}
        for i in range(min(len(common_labels.op_labels_int), len(common_labels.op_labels_greek))):
            numalpha[common_labels.op_labels_int[i]] = common_labels.op_labels_greek[i]

        from ..wilson_utils.latex_rendering import prop_trivialname_latex

        curr_ops = tuple([numalpha[j.o] for j in self.ops])
        curr_diff_inds = tuple(self.inds)
        num_denom = prop_trivialname_latex(geo=curr_diff_inds, el=curr_ops)

        return rf'\frac{{{num_denom[0]}}}{{{num_denom[1]}}}'


class TransitionIntegral:
    """
    Transition integral class
    """

    def __init__(self, bra: VibStateSymbolic, ket: VibStateSymbolic, prop: PolProp):
        """
        bra and ket (each VibStateSymbolic): Resp. bra and ket states of integral
        prop (PolProp instance): Polarization property
        """

        if not (isinstance(prop, PolProp)):
            raise TypeError('transitionIntegral polarization property must be PolProp instance')

        self.prop = prop

        if not (isinstance(bra, VibStateSymbolic)):
            raise TypeError('transitionIntegral bra state must be VibStateSymbolic instance')

        self.bra = bra

        if not (isinstance(ket, VibStateSymbolic)):
            raise TypeError('transitionIntegral ket state must be VibStateSymbolic instance')
        self.ket = ket

    def setBra(self, new_bra: VibStateSymbolic):
        """
        Set a new_bra new bra state
        """

        if not (isinstance(new_bra, VibStateSymbolic)):
            raise TypeError('transitionIntegral bra state must be VibStateSymbolic instance')

        self.bra = new_bra

    def setKet(self, new_ket):
        """
        Set a new_ket new ket state
        """

        if not (isinstance(new_ket, VibStateSymbolic)):
            raise TypeError('transitionIntegral ket state must be VibStateSymbolic instance')

        self.ket = new_ket

    def present(self):
        """
        Formatted printing of own attributes
        """

        print('Bra', self.bra.s)
        print('Ket', self.ket.s)
        print('Prop:')
        self.prop.present()


