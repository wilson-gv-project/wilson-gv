from fractions import Fraction
import copy

class QOperator:
    """
    Quantum-mechanical operator denoting an interaction with the field
    """

    def __init__(self, o: int, op_type: str=None, ax: tuple=None):
        """
        o: Operator label (integer)

        The next arguments are optional since derivations may be carried out without their specification and generating
        the specific terms arising from a choice of multipole expansion regime can be carried out at a later stage
        (avoiding repetitive derivations of similarly structured terms)

        op_type: String (default None): Optional specification of operator type
        ax: Tuple (default None): Optional specification of Cartesian axis composition of operator (for the
        electric dipole operator, this would be (3,), while for e.g. the electric quadrupole operator, this would
        be (3, 3))
        """

        if not(isinstance(o, int)):
            raise TypeError('All operator labels in qOperator must be integers')

        self.o = o

        self.op_type = op_type
        self.ax = ax

    def __repr__(self):
        return f'QOperator(o = {self.o}, op_type = {self.op_type}, ax = {self.ax})'

    def __hash__(self):
        return hash( ( self.o, self.op_type, self.ax ) )
    def __eq__(self, other):
        if isinstance(other, QOperator):
            return self.o == other.o and self.ax == other.ax and self.op_type == other.op_type
        return False
    
    def to_latex(self):
        return
    
    def setOperatorType(self, op_type: str, ax: tuple):
        """
        Set the operator type with associated axis argument. See __init__ for argument explanation
        """

        self.op_type = op_type
        self.ax = ax

    def permute(self, mask):
        """
        Permutation function: Change operator labels according to permutation mask
        Currently unimplemented/not needed
        """
        pass

    def h(self):
        """
        Hash function
        """

        return hash( ( self.o, self.op_type, self.ax ) )


class HarmOscStateSymbolic:
    """
    Symbol-described harmonic oscillator state class
    """

    def __init__(self, q: list):
        """
        q: list of normal mode index quanta
        Ground state: q = []
        One-quantum b: q = ['b']
        Three-quantum a,a,b: q = ['a', 'a', 'b']
        """

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

        # FIXME: May add test to verify that s is hashable because it needs to be
        self.s = s
        self.mbu = mbu
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


class PolProp:
    """
    Polarization property differentiated zero or more times w.r.t. geometrical displacement
    Also used more generally as energy derivative
    TODO: refactor names of attributes so they are understandable and clear
    """

    def __init__(self, ops: list[QOperator], dord: int=0):
        """
        ops: List of QOperator instances: Electromagnetic field coupling operators

        dord: Integer: Order of geometric differentiation
        """

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
        return hash(( tuple([i.h() for i in self.ops]), self.dord, tuple(self.inds) ) )

    def setDerivOrder(self, dord):
        """
        Set order of geometric differentiation

        dord: Integer: Order of differentiation
        """
        self.dord = dord

    def setInds(self, inds: list[str]):
        """
        Set (symbolic) indices of geometric differentiation

        inds: List of characters symbolizing indices of differentiation
        """

        if not len(inds) == self.dord:
            raise AssertionError('Normal mode indices must have same length as differentiation order')
        self.inds = sorted(inds)

    def h(self) -> int:
        """
        Hashing function
        """

        return hash(( tuple([i.h() for i in self.ops]), self.dord, tuple(self.inds) ) )

    def epochContained(self, epochs: list, op_ind_omega: int) -> bool:
        """
        Boolean: Do all electromagnetic operators in this property belong to the same pulse epoch?

        epochs: List [[epoch 1 pulse index 1, epoch 1 pulse ind. 2], [epoch 2 pulse ind. 1, ...], ...]
        op_ind_omega: Label of omega (detected field) operator
        """

        # FIXME: Dress epochs with interaction number instead of pulse number? Check if epochs made with pulse or interaction ids
        my_op_ids = [i.o for i in self.ops]

        # If omega is one of the operators, all of the others must belong to the last epoch
        if op_ind_omega in my_op_ids:
            for i in my_op_ids:
                if not(i == op_ind_omega):
                    if not i in epochs[len(epochs) - 1]:
                        return False

        # Otherwise, all operators must belong to the same epoch
        else:

            required_epoch = None

            for i in range(len(epochs)):
                if my_op_ids[0] in epochs[i]:
                    required_epoch = i

            for i in my_op_ids:
                if not i in epochs[required_epoch]:
                    return False

        # If not ruled out by prev checks, then the ops. in this rsp fn are indeed epoch contained
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

        self.is_pert_wf_diff = is_pert_wf_diff

    def __repr__(self):
        # return f'VibDiffTerm(sl = {self.sl}, sr = {self.sr}, is_pert_wf_diff = {self.is_pert_wf_diff})'
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

    def __init__(self, diff: VibDiffTerm, pf: list[str]=[], id=None):
        """
        diff: VibDiffTerm instance: State energy level difference
        pf: Perturbing field frequency labels (their sum to be subtracted when evaluating)
        id: Optional integer id term for potential later handling of grouped
        resonance conditions in lineshape evaluation
        """

        # Energy difference
        if not(isinstance(diff, VibDiffTerm)):
            raise TypeError('The energy difference must be a VibDiffTerm instance')

        self.diff = diff

        self.pf = pf
        self.id = id

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
        upd_pf_sign = ['-'+ax if '-' not in ax else '+'+ax.strip('-') for ax in self.pf]
        return rf'(\omega_{{{self.diff.to_latex()}}} {''.join(upd_pf_sign)})'

    def permute(self, mask: dict):
        """
        Permute frequency labels according to mask
        FIXME: May not currently be needed (previously permuted dummy labels)

        mask: Dictionary {label from: label to}
        """

        for i in range(len(self.pf)):
            self.pf[i] = mask[self.pf[i]]

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
    def couldBeResonantWithFieldByConditions(self, magn_conditions: list, prev_res=None):
        """
        Determine whether it's possible (return True) that this combination of field
        (as specified by magnitude conditions) and states might be resonant.
        Return False if this is definitely not possible.

        magn_conditions: List [[A = signed freq i, B = signed freq j], ...] signifying that
        B - A > 0

        prev_res (default: None): List of ResonanceCondition instances telling (if any) the preceding
        resonance condition to be satisfied

        This routing could be extended as follows:
        TODO: Can also combine with freq ranges (first rm according to conditions and then actual ranges for remaining?)
        TODO: For now just walk through conditions. Later, apply all combinations of conditions
        to exhaust opportunities for eliminating all
        """

        # Use conditions to trim blocks of definite sign
        # If all blocks are rmvd, see if results point to definite overall sign
        # If not all rmvd or not all definite sign, return as indeterminate
        pf_test = copy.deepcopy(self.pf)
        diff_test  = copy.deepcopy(self.diff)

        # For now simply trim last resonance from present
        if prev_res is not None:

            all_in = True
            any_in = False

            for i in prev_res.diff.sl.q:
                if not(i in diff_test.sl.q):
                    all_in = False

            for i in prev_res.diff.sr.q:
                if not(i in diff_test.sr.q):
                    all_in = False

            # This test possibly redundant
            for i in prev_res.pf:
                if not(i in pf_test):
                    all_in = False

            if all_in:

                for i in prev_res.diff.sl.q:
                    diff_test.sl.q.remove(i)

                for i in prev_res.diff.sr.q:
                    diff_test.sr.q.remove(i)

                for i in prev_res.pf:
                    pf_test.remove(i)


        # Start off sign as indeterminate
        pf_overall_sign = None

        for i in magn_conditions:

            rmd = False

            # Try one way, if match then rm and store sign, if no match then try other way
            match = True
            for j in i:
                if match:
                    if j not in pf_test:
                        match = False

            if match:
                for j in i:
                    pf_test.remove(j)
                rmd = True

                # Setting for first time
                if pf_overall_sign is None:
                    pf_overall_sign = -1

                # Conflicting with earlier condition so return
                elif pf_overall_sign == -1:
                    return True

            if not rmd:

                match = True
                for j in i:
                    if match:
                        if -1 * j not in pf_test:
                            match = False
                if match:
                    for j in i:
                        pf_test.remove(-1 * j)
                    rmd = True

                    # Setting for first time
                    if pf_overall_sign is None:
                        pf_overall_sign = 1

                    # Conflicting with earlier condition so return
                    elif pf_overall_sign == -1:
                        return True

        res_sgn = None
        if (len(pf_test) == 1):
            res_sgn = (pf_test[0] < 0) - (pf_test[0] > 0)

        if res_sgn is None and pf_overall_sign is None:
            return True

        else:
            if pf_overall_sign is None:
                pf_overall_sign = res_sgn
            else:
                if res_sgn is not None:
                    if not res_sgn == pf_overall_sign:
                        return True

        sd_overall_sign = self.netStateSign(instead_trimmed=diff_test)

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

        # If state difference is zero but pert freq is significantly pos or neg, then return False
        elif sd_overall_sign == 0:
            if pf_overall_sign == 1:
                return False
            elif pf_overall_sign == -1:
                return False

        return True


class VibPerturbedTerm:
    """
    Class to represent post-Hermite treatment term: Instances of this class are the main end result of a typical
    wilson-derive run.
    """

    def __init__(self, coeff: Fraction, props: list[PolProp], freqterms: list[VibDiffTerm], res: list[ResonanceCondition]):
        """
        A term consists of the following attributes:

        coeff: Coefficient of term (Fraction)

        props: Molecular properties: At the stage of derivation where the present class is relevant, all of these
        would involve at least one order of geometrical differentiation (list of PolProp instances)

        freqterms: Prefactors of inverse differences between vibrational energy levels as identified
        by the derivation (list of VibDiffTerm instances)

        res: Resonance conditions as identified by the derivation (list of ResonanceCondition instances)
        """

        if not (isinstance(coeff, Fraction) ):
            raise TypeError('VibPerturbedTerm coeff must be a Fraction instance')
        self.coeff = coeff

        if not (all([isinstance(i, PolProp) for i in props])):
            raise TypeError('All transition integral arguments in VibPerturbedTerm must be transitionIntegral instances')
        self.props = props

        if not (all([isinstance(i, VibDiffTerm) for i in freqterms])):
            raise TypeError('All frequency arguments in VibPerturbedTerm must be vibDiffTerm instances')
        self.freqterms = freqterms

        if not (all([isinstance(i, ResonanceCondition) for i in res])):
            raise TypeError('All resonance condition arguments in VibPerturbedTerm must be resonanceCondition instances')
        self.res = res

        # Boolean: Has this term been sorted according to the below sort method?
        self.was_sorted = False

        # Hash (currently indeterminate)
        self.hsh = None

    def __repr__(self):
        return f"VibPerturbedTerm(coeff = {self.coeff}, props = {self.props}, freqterms = {self.freqterms}, res = {self.res})"

    def tellNonSummSummIndices(self):

        summation_indices = []
        non_summation_indices = []

        for i in self.res:

            for j in i.diff.sl.q:
                if not j in non_summation_indices:
                    non_summation_indices.append(j)

            for j in i.diff.sr.q:
                if not j in non_summation_indices:
                    non_summation_indices.append(j)

        candidate_summation_indices = []

        for i in self.freqterms:

            for j in i.sl.q:
                if not j in candidate_summation_indices:
                    candidate_summation_indices.append(j)

            for j in i.sr.q:
                if not j in candidate_summation_indices:
                    candidate_summation_indices.append(j)

        for i in candidate_summation_indices:

            if not i in non_summation_indices:
                summation_indices.append(i)


        return non_summation_indices, summation_indices


    def nmRenameAndInternalResort(self, mask: dict):
        """
        Take a mask (dictionary of single-character key: value pairs) in an ordering to be replaced by the canonical normal mode
        index list: Example: mask is ['b': 'a', 'a': 'b', 'c': 'c'] -> Replace every reference to 'b' in self with 'a',
        replace every (original) 'a' with 'b' and leave 'c' unchanged
        """

        # In properties
        for i in range(len(self.props)):
            for j in range(len(self.props[i].inds)):
                self.props[i].inds[j] = mask[self.props[i].inds[j]]
            self.props[i].inds = sorted(self.props[i].inds)

        # In (non-resonance condition) frequency terms
        for i in range(len(self.freqterms)):
            for j in range(len(self.freqterms[i].sl.q)):
                self.freqterms[i].sl.q[j] = mask[self.freqterms[i].sl.q[j]]
            self.freqterms[i].sl.q = sorted(self.freqterms[i].sl.q)
            for j in range(len(self.freqterms[i].sr.q)):
                self.freqterms[i].sr.q[j] = mask[self.freqterms[i].sr.q[j]]
            self.freqterms[i].sr.q = sorted(self.freqterms[i].sr.q)

        # In resonance conditions
        for i in range(len(self.res)):
            for j in range(len(self.res[i].diff.sl.q)):
                self.res[i].diff.sl.q[j] = mask[self.res[i].diff.sl.q[j]]
            self.res[i].diff.sl.q = sorted(self.res[i].diff.sl.q)
            for j in range(len(self.res[i].diff.sr.q)):
                self.res[i].diff.sr.q[j] = mask[self.res[i].diff.sr.q[j]]
            self.res[i].diff.sr.q = sorted(self.res[i].diff.sr.q)

    def sort(self, nm_inds):
        """
        Sort and possibly rename indices in term to put in canonical term:
        - Sort resonance conditions in increasing order of number of perturbing frequencies
        - Rename normal mode indices according to encountered state labels in resonance conditions (currently leaving
        further indices in arbitrary order)
        - Sort operator references in properties in increasing numerolexical order
        - Sort properties among each order in increasing order of geometric differentiation
        - Then sort properties at same order of differentiation according to operator (numerolexical) ordering
        - Sort frequency difference terms internally to have greatest number of quanta in bra state
        - Sort terms tied wrt. previous sorting according to lexical order of state tuples
        - Sort freq diff terms wrt. each other according to lexical ordering of bras (sort tied terms by ket lex. ordering)

        nm_inds: List of canonically ordered normal mode indices (['a', 'b', ...])
        """

        # Sort resonance conditions in increasing order of number of perturbing frequencies
        self.res = sorted(self.res, key=lambda j:len(j.pf))

        # Make normal mode index sorting mask:
        # First gather according to encountered state labels in resonance conditions
        # Then get remaining indices as encountered in derivatives (FIXME: Likely degree of freedom)

        nm_labels = []

        for i in self.res:

            for j in i.diff.sl.q:
                if not j in nm_labels:
                    nm_labels.append(j)

            for j in i.diff.sr.q:
                if not j in nm_labels:
                    nm_labels.append(j)

        for i in self.props:
            for j in i.inds:
                if not j in nm_labels:
                    nm_labels.append(j)

        # If the label progression doesn't correspond to the canonical progression, then rename to make canonical
        # and re-sort (FIXME: Could be remaining sorting "slack" after this)
        if not (nm_labels == nm_inds[:len(nm_labels)]):
            nm_mask = {nm_labels[i]: nm_inds[i] for i in range(len(nm_labels))}
            self.nmRenameAndInternalResort(nm_mask)

        # Internal sorting: Operators in each property in increasing order
        for i in self.props:
            i.ops = sorted(i.ops, key=lambda j: j.o)

        # Inter-term sorting

        # First sort properties according to increasing order of differentiation
        self.props = sorted(self.props, key=lambda j: j.dord)

        # Then sort properties at same order of differentiation according to operators
        # NOTE: I believe this should take care of both number and lexical ordering

        m = 0
        dord_starts = [0]

        for i in range(len(self.props)):
            if self.props[i].dord > self.props[dord_starts[m]].dord:
                dord_starts.append(i)
                m += 1

        dord_starts.append(len(self.props) - 1)

        for i in range(len(dord_starts)):
            # No need to sort if at the last entry
            if not dord_starts[i] == len(self.props) - 1:
                # No need to sort a length 1 category
                if not dord_starts[i + 1] == dord_starts[i] + 1:
                    self.props[dord_starts[i]:dord_starts[i+1]] = sorted(self.props[dord_starts[i]:dord_starts[i+1]],
                                                                         key=lambda j: [k.o for k in j.ops])

        # Sort all freq term bras and kets

        # Internal sorting (assumes all individual nm tuples already sorted)

        # Make all freq diff terms have the greatest number of quanta in bra - flip and change sign if necessary
        for i in range(len(self.freqterms)):
            if len(self.freqterms[i].sl.q) < len(self.freqterms[i].sr.q):
                self.coeff *= Fraction(-1)
                tmp = copy.deepcopy(self.freqterms[i].sl)
                self.freqterms[i].sl = copy.deepcopy(self.freqterms[i].sr)
                self.freqterms[i].sr = tmp

            # Sort tied terms according to nm indices
            elif len(self.freqterms[i].sl.q) == len(self.freqterms[i].sr.q):
                if not (sorted([self.freqterms[i].sl.q, self.freqterms[i].sr.q]) ==
                               [self.freqterms[i].sl.q, self.freqterms[i].sr.q]):
                    self.coeff *= Fraction(-1)
                    tmp = copy.deepcopy(self.freqterms[i].sl)
                    self.freqterms[i].sl = copy.deepcopy(self.freqterms[i].sr)
                    self.freqterms[i].sr = tmp

        # Inter-term sorting

        # Sort freq diff terms first according to ket indices
        self.freqterms = sorted(self.freqterms, key=lambda j:j.sr.q)

        # Then sort tied freq diff terms according to bra indices
        ketq_starts = [0]

        m = 0

        for i in range(len(self.freqterms)):
            if not (self.freqterms[i].sr.q == self.freqterms[ketq_starts[m]].sr.q):
                ketq_starts.append(i)
                m += 1

        ketq_starts.append(len(self.freqterms))

        for i in range(len(ketq_starts)):
            if not ketq_starts[i] == len(self.freqterms):
                self.freqterms[ketq_starts[i]:ketq_starts[i + 1]] = \
                    sorted(self.freqterms[ketq_starts[i]:ketq_starts[i + 1]], key=lambda j: j.sl.q)

        # Mark term as having been sorted
        # WARNING: This flag doesn't update if any subsequent changes are made
        self.was_sorted = True

    def h(self, also_sort: bool=False, nm_inds: list=None) -> int:
        """
        Hashing function

        also_sort: Also sort term before hash is calculated?
        nm_inds: List of normal mode indices if sorting
        """

        if not(self.was_sorted) and not(also_sort):
            raise AssertionError('Term for which hash was requested has not been sorted')

        if also_sort:
            self.sort(nm_inds)

        # Getting hashes of constituent parts
        props_h = tuple([i.h() for i in self.props])
        ft_h = tuple([i.h() for i in self.freqterms])
        res_h = tuple([i.h() for i in self.res])

        # Combine constituent hashes for collective hash for this term
        self.hsh = hash((props_h, ft_h, res_h))

        return self.hsh

    def full_enhancement_possible(self, magn_conditions=None) -> bool:
        """
        Determine: Given the setup/requested frequency ranges, is it possible for this term to become resonant within these
        ranges with respect to all of its resonance conditions?

        FIXME: Currently only for sole harmonic oscillator state, may later need extension for variationally resolved states

        magn_conditions: List [[A = signed freq i, B = signed freq j], ...] signifying that
        B - A > 0
        """

        # Discernment by frequency ranges implementation not ready
        #
        #if field_freq_ranges is not None:
        #
        #    for i in self.res:
        #        if not(i.couldBeResonantWithFieldByRanges(field_freq_ranges)):
        #            return False
        #
        #    return True

        if magn_conditions is not None:

            prev_res = None
            for i in self.res:
                if not (i.couldBeResonantWithFieldByConditions(magn_conditions, prev_res=prev_res)):
                    return False
                if not (i.couldBeResonantWithFieldByConditions(magn_conditions, prev_res=None)):
                    return False
                prev_res = copy.deepcopy(i)

            return True

        # If no information given, return True to keep term
        else:
            return True

    def present(self):
        """
        Formatted printing of own attributes
        """
        print(' >> VibPerturbedTerm presents:')

        print('Coefficient:', self.coeff)
        print('Properties:')
        for i in self.props:
            i.present()
        print('Freqterms:')
        for i in self.freqterms:
            i.present()
        print('Resonances:')
        for i in self.res:
            i.present()

        print('\nHas attributes: coeff, freqterms, res, was_sorted, props, hsh'
              '\nHas methods: nmRenameAndResort, sort, h, full_enhancement_possible, present'
              '\nPresenting also elements of: self.props, self.freqterms, self.res')

    def present_better(self):
        """
        Formatted printing of own attributes
        """
        print('\n >> VibPerturbedTerm presents:')

        print('Coefficient:', self.coeff)
        print('Properties:')
        for i in self.props:
            print('    ', i)
        print('Freqterms:')
        for i in self.freqterms:
            print('    ', i)
        print('Resonances:')
        for i in self.res:
            print('    ', i)

        print('\nHas attributes: coeff, freqterms, res, was_sorted, props, hsh'
              '\nHas methods: nmRenameAndResort, sort, h, full_enhancement_possible, present'
              '\nPresenting also elements of: self.props, self.freqterms, self.res')

    def to_latex(self, part=None):
        """
        """
        res_conditions_denom = ''.join([rc.to_latex() for rc in self.res])
        if res_conditions_denom == '':
            res_conditions_str = ''
        else:
            res_conditions_str = rf'\frac{{1}}{{{res_conditions_denom}}}'

        coefficients_str = rf'\frac{{{self.coeff.numerator}}}{{{self.coeff.denominator}}}'
        properties_str = ''.join([p.to_latex() for p in self.props])
        
        freqterms_denom = ''.join([rf'\omega_{{{vd.to_latex()}}}' for vd in self.freqterms])
        if freqterms_denom == '':
            freqterms_str = ''
        else:
            freqterms_str = rf'\frac{{{1}}}{{{freqterms_denom}}}'

        if part is not None:
            if part=='res':
                return res_conditions_str
            elif part=='coeff':
                return coefficients_str
            elif part=='props':
                return properties_str
            elif part=='freqterms':
                return freqterms_str
        else:
            return coefficients_str + freqterms_str + properties_str + res_conditions_str


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


class VibContribTerm:
    """
    Class to represent a vibrational term in SOS expression up to but not including Hermite treatment
    """

    def __init__(self, coeff: Fraction, ints: list[TransitionIntegral], res: list[ResonanceCondition]):
        """
        A term consists of the following attributes:

        coeff: Coefficient of term (Fraction)

        ints: Transition integrals involving a bra state, one or more operators, and a ket state

        res: Resonance conditions as identified by the derivation (list of ResonanceCondition instances)
        """

        if not (isinstance(coeff, Fraction) ):
            raise TypeError('vibContribTerm coeff must be a Fraction instance')
        self.coeff = coeff

        if not (all([isinstance(i, TransitionIntegral) for i in ints])):
            raise TypeError('All transition integral arguments in VibContribTerm must be TransitionIntegral instances')
        self.ints = ints

        if not (all([isinstance(i, ResonanceCondition) for i in res])):
            raise TypeError('All resonance condition arguments in VibContribTerm must be ResonanceCondition instances')
        self.res = res

        # Initialize frequency difference terms in anticipation of possible addition
        # from mechanical anharmonicity handling
        self.freqdiff = []

    def addFreqTerm(self, new_term: VibDiffTerm):
        """
        Add (inverted) frequency difference term from mechanical anharmonicity handling
        """

        if not (isinstance(new_term, VibDiffTerm)):
            raise TypeError('VibContribTerm frequency difference terms must be VibDiffTerm instances')

        self.freqdiff.append(new_term)

    def dressWithPulseInteractions(self, int_seq: list[dict]):
        """
        Substitute interaction dummy indices in my resonance conditions with an earlier found specific pulse interaction sequence

        int_seq: list of dictionaries [{interaction #1 pulse label: sign}, {int. #2 pulse label: sign}, ...]
        """

        # This is confus, must be a better way to write this
        for i in self.res:
            for j in range(len(i.pf)):
                i.pf[j] = list(int_seq[j].keys())[0] * int_seq[j][list(int_seq[j].keys())[0]]


    def allUVCancels(self, cfs_uv: dict, tol: float=0.0) -> bool:
        """
        Boolean function: Do all UV/VIS carrier frequency components sum to zero
        (leaving only potentially vibrationally resonant values) for my resonance conditions?

        cfs_uv: Dictionary {pulse label: UV/VIS freq. component, ...}
        tol: Tolerance (allow a verdict of "cancels" if sum is within tolerance)
        """

        all_cancel = True

        for i in self.res:
            all_cancel = all_cancel and i.uvCancels(cfs_uv, tol)

        return all_cancel


    def allElRspEpochContained(self, epochs, op_ind_omega) -> bool:
        """
        Boolean function: Do all the individual operator tuples of my (electronic) response functions
        belong to pulses corresponding to the same epoch?

        epochs: List of epochs and which pulses (pulse IDs) are in each
        op_ind_omega: Which operator is the "omega" (interaction w/detected field) operator?
        """

        epoch_contained = True

        for i in self.ints:
            epoch_contained = epoch_contained and i.prop.epochContained(epochs, op_ind_omega)

        return epoch_contained

    def present(self):
        """
        Formatted printing of own attributes
        """

        print('Coefficient:', self.coeff)

        for i in self.ints:
            i.present()

        for i in self.res:
            i.present()

        for i in self.freqdiff:
            i.present()

        print('')

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



