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

    def __init__(self, diff: VibDiffTerm, pf: list, id=None):
        """
        diff: VibDiffTerm instance: State energy level difference: States must here be HarmOscStateSymbolic or VibStateSymbolic
              - Several methods will only work with the states in HarmOscStateSymbolic form
              - However, having states as VibStateSymbolic instances is relevant in earlier stages of the overall term
              derivation process
        pf: List: Perturbing field frequency labels (their sum to be subtracted when evaluating)
        id: Optional integer id term for potential later handling of grouped
        resonance conditions in lineshape evaluation
        """

        # Energy difference
        if not(isinstance(diff, VibDiffTerm)):
            raise TypeError('The energy difference must be a VibDiffTerm instance')

        self.diff = diff

        if not(isinstance(pf, list)):
            raise TypeError('Perturbing frequency labels must be a list of strings or integers')
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

        # Removing matching quanta from bra and ket
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

        # Decide sign based on number of quanta left in bra and ket
        if len_bra == 0:

            if len_ket > 0:
                return -1
            # None left so zero sign
            else:
                return 0

        else:

            # Special flag for indeterminate sign
            if len_ket > 0:
                return -3
            else:
                return 1

    def could_be_resonant_by_conditions(self, mc_plus: list[list]|tuple[list], mc_minus: list[list]|tuple[list], given_prev_res = None) -> bool:
        """
        Determine whether it's possible (return True) that this combination of field
        (as specified by magnitude conditions) and states might be resonant.
        Return False only if this is definitely not possible.

        mc_plus: A list or tuple, containing lists of interaction indices whose frequency components sum is to be taken as positive
        by a margin deemed significant

        mc_minus: The analagous data as for mc_plus but referring to interaction indices whose freq sum is to be taken
        as negative

        given_prev_res (default: None): ResonanceCondition instance telling (if any) the preceding
        resonance condition that was satisfied. Optional argument which will then adapt the functioning of this method.
        """

        # Initialize test blocks to possibly trim if previous resonance condition is given (otherwise use directly)
        pf_test = copy.deepcopy(self.pf)
        diff_test  = copy.deepcopy(self.diff)

        # Trim last resonance from present
        if given_prev_res is not None:

            # All previous perturbing freq args should be in when this fn is used as intended, so remove them directly
            for i in given_prev_res.pf:
                pf_test.remove(i)

            # Determine which of the previous res. cond.'s freq. diff. quanta are also in the current
            # NOTE: Ignores anharmonicity effects
            # NOTE: Assumes each index is given max one time per side; may need revisit if this turns out not to hold
            # for the general term (currently not encountered)
            in_sl = []
            not_in_sl = []

            for i in given_prev_res.diff.sl.q:
                if not(i in diff_test.sl.q):
                    not_in_sl.append(i)
                else:
                    in_sl.append(i)

            in_sr = []
            not_in_sr = []

            for i in given_prev_res.diff.sr.q:
                if not(i in diff_test.sr.q):
                    not_in_sr.append(i)
                else:
                    in_sr.append(i)

            # Remove those quanta that were in
            for i in in_sl:
                diff_test.sl.q.remove(i)

            for i in in_sr:
                diff_test.sr.q.remove(i)

            # Add not_in "on credit" to other side
            for i in not_in_sl:
                diff_test.sr.q.append(i)

            for i in not_in_sr:
                diff_test.sl.q.append(i)

        # Get sign of state energy level difference
        sd_overall_sign = self.netStateSign(instead_trimmed=diff_test)

        # Special flag for indeterminate sign
        if sd_overall_sign == -3:
            return True

        # Attempt to find sign for perturbing freq. components combination

        # Start off sign as unset
        pf_overall_sign = None

        # As far as I can tell, there will always be just one indep var left in pf after trimming (and only one if this
        # is the first resonance condition). Moreover, a complete set of magnitude conditions created from the powerset
        # (as in FieldAnalysis construction) will also consider the powerset of interactions, so that any
        # valid combinations would be accessible here

        # Therefore a straightforward lookup should be sufficient

        # Case A: Found in mc_minus so pf sign is negative
        if sorted(list(pf_test)) in mc_minus:
            pf_overall_sign = -1

        # Case B: Found in mc_plus so pf sign is positive
        elif sorted(list(pf_test)) in mc_plus:
            pf_overall_sign = 1

        # Otherwise not found, therefore indeterminate so return True (resonance cannot be ruled out)
        else:
            return True

        # If this stage is reached, there is a definite state difference sign (or zero) and a definite perturbing freq.
        # combination sign, so analyze them w.r.t. each other

        # State energy level difference sign and perturbing frequencies must be same in order for
        # resonance to be possible
        if sd_overall_sign == -1:
            if pf_overall_sign == -1:
                return True
            return False

        elif sd_overall_sign == 1:
            if pf_overall_sign == 1:
                return True
            return False

        # If state difference is zero but pert freq is significantly pos or neg, then return False
        elif sd_overall_sign == 0:
            if pf_overall_sign == 1:
                return False
            elif pf_overall_sign == -1:
                return False

        # This line should ordinarily not be reached (no ruling could be made with the above analysis), so raise
        raise ValueError('A ruling could not be made on term resonance possibility')

class PolProp:
    """
    Polarization property differentiated zero or more times w.r.t. geometrical displacement
    Also used more generally as energy derivative
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


