from fractions import Fraction
import copy

# FIXME: Put these in a conceptually logical order

# Lineshape class
# Not sure whether to use this form or some other way
class lineShape:

    def __init__(self, rcs, evaluator=None):

        # Must be tuple of resonanceCondition instances
        self.rcs = rcs

        # Possible evaluator function?
        self.evaluator = evaluator

# To do:
# FIXME: Is this still valid?
# Some way to handle general coefficients (but could save for later bc only scalars if not doing "non-void" pop decay)

# Electromagnetic polarization property, differentiated or undifferentiated
class polProp:

    # Operators ops
    def __init__(self, ops, dord=0):

        if not (all([isinstance(i, qOperator) for i in ops])):
            raise TypeError('All operator arguments in polProp must be qOperator instances')
        self.ops = ops

        if not (isinstance(dord, int)):
            raise TypeError('Differentiation order in polProp must be integer')
        self.dord = dord

        self.inds = None

    def setDerivOrder(self, dord):
        self.dord = dord

    def setInds(self, inds):
        if not len(inds) == self.dord:
            raise AssertionError('Normal mode indices must have same length as differentiation order')
        self.inds = sorted(inds)

    # Dictionary with "from: to" entries
    def updInds(self, mask):
        for i in mask:
            self.inds[i] = mask[i]

    def h(self):
        return hash(( tuple([i.h() for i in self.ops]), self.dord, tuple(self.inds) ) )

    def epochContained(self, epochs, op_ind_omega):

        # FIXME: Possibly dress epochs with interaction number instead of pulse number? Check if epochs made with pulse or interaction ids

        my_op_ids = [i.o for i in self.ops]

        # If omega is one of the operators, all of the others must belong to the last epoch
        if op_ind_omega in my_op_ids:
            for i in my_op_ids:
                if not(i == op_ind_omega):
                    if not i in epochs[len(epochs) - 1]:
                        return False

        # Otherwise, all operators must belong to the same epoch
        else:
            for i in range(len(epochs)):
                if my_op_ids[0] in epochs[i]:
                    required_epoch = i

            for i in my_op_ids:
                if not i in epochs[required_epoch]:
                    return False

        # If not ruled out by prev checks, then the ops. in this rsp fn are indeed epoch contained
        return True


        # Otherwise, all operators must belong to same epoch

        pass

    def present(self):

        print('Operators:', [i.o for i in self.ops])
        print('Geo differentiation order:', self.dord)
        if self.inds is not None:
            print('Normal mode differentiation indices:', self.inds)


# Vibrational term in SOS expression up to but not including Hermite treatment
class vibContribTerm:

    def __init__(self, coeff, ints, res):

        if not (isinstance(coeff, Fraction) ):
            raise TypeError('vibContribTerm coeff must be a Fraction instance')
        self.coeff = coeff

        if not (all([isinstance(i, transitionIntegral) for i in ints])):
            raise TypeError('All transition integral arguments in vibContribTerm must be transitionIntegral instances')
        self.ints = ints

        if not (all([isinstance(i, resonanceCondition) for i in res])):
            raise TypeError('All resonance condition arguments in vibContribTerm must be resonanceCondition instances')
        self.res = res

        self.freqdiff = []

    # Add (inverted) frequency difference term from mechanical anharmonicity handling
    def addFreqTerm(self, new_term):

        if not (isinstance(new_term, vibDiffTerm)):
            raise TypeError('vibContribTerm frequency difference terms must be vibDiffTerm instances')

        self.freqdiff.append(new_term)

    # Substitute interaction dummy indices with actual pulse interactions
    def dressWithPulseInteractions(self, int_seq):
        for i in self.res:
            for j in range(len(i.pf)):
                i.pf[j] = list(int_seq[j].keys())[0] * int_seq[j][list(int_seq[j].keys())[0]]

    def allUVCancels(self, cfs_uv, tol=0.0):

        all_cancel = True

        for i in self.res:
            all_cancel = all_cancel and i.uvCancels(cfs_uv, tol)

        return all_cancel

    def allElRspEpochContained(self, epochs, op_ind_omega):

        epoch_contained = True

        for i in self.ints:
            epoch_contained = epoch_contained and i.prop.epochContained(epochs, op_ind_omega)

        return epoch_contained



    def present(self):

        print('Coefficient:', self.coeff)

        for i in self.ints:
            i.present()

        for i in self.res:
            i.present()

        for i in self.freqdiff:
            i.present()

        print('')

# Post-Hermite term
class vibPerturbedTerm:

    def __init__(self, coeff, props, freqterms, res):

        if not (isinstance(coeff, Fraction) ):
            raise TypeError('vibPerturbedTerm coeff must be a Fraction instance')
        self.coeff = coeff

        if not (all([isinstance(i, polProp) for i in props])):
            raise TypeError('All transition integral arguments in vibPerturbedTerm must be transitionIntegral instances')
        self.props = props

        if not (all([isinstance(i, vibDiffTerm) for i in freqterms])):
            raise TypeError('All frequency arguments in vibPerturbedTerm must be vibDiffTerm instances')
        self.freqterms = freqterms

        if not (all([isinstance(i, resonanceCondition) for i in res])):
            raise TypeError('All resonance condition arguments in vibPerturbedTerm must be resonanceCondition instances')
        self.res = res

        self.was_sorted = False
        self.hsh = None


    # FIXME: Make test for this
    def nmRenameAndResort(self, mask):

        # Rename according to mask in all props, freq diff, res conditions

        for i in range(len(self.props)):
            for j in range(len(self.props[i].inds)):
                self.props[i].inds[j] = mask[self.props[i].inds[j]]
            self.props[i].inds = sorted(self.props[i].inds)

        for i in range(len(self.freqterms)):
            for j in range(len(self.freqterms[i].sl.q)):
                self.freqterms[i].sl.q[j] = mask[self.freqterms[i].sl.q[j]]
            self.freqterms[i].sl.q = sorted(self.freqterms[i].sl.q)
            for j in range(len(self.freqterms[i].sr.q)):
                self.freqterms[i].sr.q[j] = mask[self.freqterms[i].sr.q[j]]
            self.freqterms[i].sr.q = sorted(self.freqterms[i].sr.q)

        for i in range(len(self.res)):
            for j in range(len(self.res[i].diff.sl.q)):
                self.res[i].diff.sl.q[j] = mask[self.res[i].diff.sl.q[j]]
            self.res[i].diff.sl.q = sorted(self.res[i].diff.sl.q)
            for j in range(len(self.res[i].diff.sr.q)):
                self.res[i].diff.sr.q[j] = mask[self.res[i].diff.sr.q[j]]
            self.res[i].diff.sr.q = sorted(self.res[i].diff.sr.q)

    def sort(self, nm_inds):

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

        # Then get mask to relabel mode diff indices to be in strictly increasing encountered nm index order
        nm_labels = []

        for i in self.props:
            for j in i.inds:
                if not j in nm_labels:
                    nm_labels.append(j)

        # If the label progression doesn't correspond to the canonical progression, then rename to make canonical
        # and re-sort (FIXME: I don't know if there is some sorting "slack" after this but I believe not)
        if not(nm_labels == nm_inds[:len(nm_labels)]):
            nm_mask = {nm_labels[i]: nm_inds[i] for i in range(len(nm_labels))}
            self.nmRenameAndResort(nm_mask)

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


        # Sort resonance conditions in increasing order of number of perturbing frequencies, which should be sufficient
        self.res = sorted(self.res, key=lambda j:len(j.pf))

        # WARNING: This flag doesn't update if any subsequent changes are made
        self.was_sorted = True

    def h(self, also_sort=False, nm_inds=None):

        if not(self.was_sorted) and not(also_sort):
            raise AssertionError('Term for which hash was requested has not been sorted')

        if also_sort:
            self.sort(nm_inds)

        props_h = tuple([i.h() for i in self.props])
        ft_h = tuple([i.h() for i in self.freqterms])
        res_h = tuple([i.h() for i in self.res])

        self.hsh = hash((props_h, ft_h, res_h))

        return self.hsh

    # FIXME: Current version only for harmonic wavefunctions, may later need extension for variationally resolved states
    def full_enhancement_possible(self, field_freq_ranges=None, magn_conditions=None):

        if field_freq_ranges is not None:

            for i in self.res:
                if not(i.couldBeResonantWithFieldByRanges(field_freq_ranges)):
                    return False

            return True

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

        print('Coefficient:', self.coeff)

        for i in self.props:
            i.present()

        for i in self.freqterms:
            i.present()

        for i in self.res:
            i.present()


class harmOscState:

    def __init__(self, q):

        # q: dict of normal mode index: quanta
        # Ground state: q = []
        # One-quantum b: q = ['b']
        # Three-quantum a,a,b: q = ['a', 'a', 'b']
        self.q = sorted(q)

    def h(self):
        return hash(tuple(self.q))

class vibState:

    # mbu: Must be unequal to state(s) given here
    def __init__(self, s, mbu=[], is_ground=False):

        # FIXME: May add test to verify that s is hashable because it needs to be
        self.s = s

        self.mbu = mbu
        self.is_ground = is_ground

    def mbuFulfilled(self, states_as_quanta):

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
        return hash( ( self.s, tuple([i.s for i in self.mbu]), self.isground ) )

class vibDiffTerm:

    def __init__(self, sl=None, sr=None):

        # Must be both vibState or both harmOscState instances
        if not (isinstance(sl, vibState) and isinstance(sr, vibState) or
                isinstance(sl, harmOscState) and isinstance(sr, harmOscState)):
            raise TypeError('Both sl and sr must be either both vibState instances or both harmOscState instances')

        self.sl = sl
        self.sr = sr

    def present(self):
        print('Freq diff term')

        if isinstance(self.sl, vibState):
            print('Bra state', self.sl.s)
            print('Ket state', self.sr.s)

        elif isinstance(self.sl, harmOscState):
            print('Bra state', self.sl.q)
            print('Ket state', self.sr.q)

    def h(self):
        return hash( ( self.sl.h(), self.sr.h() ) )


# Resonance condition
# Assumption: Perturbing frequencies subtracted
class resonanceCondition:

    def __init__(self, diff, pf=[], id=None):

        # Energy difference
        if not(isinstance(diff, vibDiffTerm)):
            raise TypeError('The energy difference must be a vibDiffTerm instance')

        self.diff = diff

        # Perturbing frequencies
        if not all(isinstance(i, int) for i in pf):
            raise TypeError('All perturbing frequencies must be represented by an integer index')

        self.pf = pf

        # Optional integer id term for potential later handling of grouped resonance conditions in lineshape evaluation
        self.id = id

    def present(self):

        print('Resonance condition states')
        self.diff.present()
        print('Resonance condition pert freqs', str(self.pf))

    # Permute axes according to mask (used for simultaneous freq and ax ref permutation)
    def permute(self, mask):

        for i in range(len(self.pf)):
            self.pf[i] = mask[self.pf[i]]

    def h(self):
        return hash(( self.diff.h(), tuple(self.pf), self.id ))

    # Determine if UV parts of carrier frequencies cancel
    # Default tolerance is 0.0
    # Consequences if tolerance != 0.0 used are not yet supported/investigated
    def uvCancels(self, cfs_uv, tol=0.0):
        acc = 0.0

        for i in self.pf:
            sgn = (i > 0) - (i < 0)
            acc += sgn * cfs_uv[sgn*i]

        sgnacc = (acc > 0) - (acc < 0)

        return ((sgnacc * acc) <= tol)

    def netStateSign(self, instead_trimmed=None):

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

    # Return True if it's possible that this combination of field and states might be resonant
    # Return False if this is definitely not possible
    # FIXME: This definitely needs verification/testing
    def couldBeResonantWithFieldByRanges(self, pulse_freq_spans):

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


    # Return True if it's possible that this combination of field and states might be resonant
    # Return False if this is definitely not possible
    # FIXME: Functionality not general yet
    def couldBeResonantWithFieldByConditions(self, magn_conditions, prev_res=None):

        # Use conditions to "rm" blocks of definite sign
        # If all blocks are rmvd, see if results point to definite overall sign
        # If not all rmvd or not all definite sign, return as indeterminate
        # TODO: Can also combine with freq ranges (first rm according to conditions and then actual ranges for remaining?)

        # TODO: For now just walk through conditions. Later, apply all combinations of conditions
        # to exhaust opportunities for eliminating all

        pf_test = copy.deepcopy(self.pf)
        diff_test  = copy.deepcopy(self.diff)

        # For now just simply trim last resonance from present
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

class qOperator:

    def __init__(self, o, ax):

        if not(isinstance(o, int)):
            raise TypeError('All operator labels in qOperator must be integers')

        self.o = o
        self.ax = ax

    def permute(self, mask):
        pass

    def h(self):
        return hash( ( self.o, self.ax ) )

class opDerivative:

    def __init__(self, op, order):
        self.op = op
        self.order = order

class transitionIntegral:

    def __init__(self, bra, ket, prop):

        if not (isinstance(prop, polProp)):
            raise TypeError('transitionIntegral polarization property must be polProp instance')

        self.prop = prop

        if not (isinstance(bra, vibState)):
            raise TypeError('transitionIntegral bra state must be vibState instance')

        self.bra = bra

        if not (isinstance(ket, vibState)):
            raise TypeError('transitionIntegral ket state must be vibState instance')
        self.ket = ket

    def setBra(self, new_bra):

        if not (isinstance(new_bra, vibState)):
            raise TypeError('transitionIntegral bra state must be vibState instance')

        self.bra = new_bra

    def setKet(self, new_ket):

        if not (isinstance(new_ket, vibState)):
            raise TypeError('transitionIntegral ket state must be vibState instance')

        self.ket = new_ket

    def present(self):

        print('Bra', self.bra.s)
        print('Ket', self.ket.s)
        print('Prop:')
        self.prop.present()




