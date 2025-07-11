import numpy as np

from wilson.spectrum.tools import convNu2Ene, combinations_with_permutations
from wilson.spectrum.spectrum_utils import MolProperty, AveragedProps, VibStatesDiff
from wilson.spectrum.spectrum_utils import get_allparts_indices, make_abc_dict, make_abc_tuple
from wilson.spectrum.spectrum_utils import abc_list, greek_list
from wilson.utils.tagger import tag
from wilson.debug import debugfunc, debug_deep

@tag('used in get_resonance_location_general for NO PRECALC')
def compute_vibdiff(vibdiff_type, idx):
    """
    vibdiff_types: (0,1), (1,1), (2,1)
    idx - one per 1 in vibdiff_type

    used in get_resonance_location_general for NO PRECALC
    """
    mn = []
    list_idx = list(idx)
    for vd_idx in vibdiff_type:
        if vd_idx == 0:
            mn.append(('zero',))
        else:
            tuple_str = []
            for i in range(vd_idx):
                tuple_str.append(str(list_idx.pop(i)))
            mn.append(tuple(tuple_str))

    return mn


class TermND:

    def __init__(self, term_id, expression):
        """
        Calculations using the expression.
        TermND object would have a dict-mathematical expression representation.

        expressions = {term_id: expression}
        expressions = {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': None,
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_QQ', ('a', 'b',), ('G',))),
                          'non_averaged_props': None,
                          'vibene_denom': ('a','b',),
                          'termB_pref': 1.,
                          'termA_pref': 1/24}

        So, what makes it work better (CPU and memory)?
        - precalculations:
            self.resonances_bank is made and saved for each ab combination (keys are unique resonances in terms);
            self.res_dict[res_formula[0]] collects valid resonance locations for given resonance type (res_formula[0])
            self.avrg_tensors_dict(s) containing all used averaged tensors
            self.prefac_2d containing harm vib ene prefactors
            self.comb_fac_dict[self.allterms_str[termID]] for terms, these would be 2d arrays over a and b combinations

        - use in-place addition to save memory
        - if factor is essentially zero, skip this addition
        - if current ab resonance location is invalid, skip this addition
        - use np.where and condition array to make addition (a product of factor and resonance)
        --------
        addition would be computed here? in this class? addition is for given ab! and given term!


        general expression:
                    AVG*GP*TP*HEP*(CFF)*RESCONDS*ODEN
            AVG      - avrg_props_expr
            GP       - prefactorA (e.g., 1/24 or 1/-48)
            TP       - prefactorB (e.g., -1/2 or 1/2, or 1.)
            HEP      - vibene_denom_expr (harmonic vib ene prefactor)
            NAVG     - non_avrg_props_expr
            RESCONDS - resonance conditions (list, will be multiplied together)
            ODEN - other denominators (e.g., vib ene levels denominators; will be added together)

        """
        self.term_id = term_id
        self.expression = expression

        # expression parts
        self.resonances_expr = expression['resonances']
        self.viblevelsdiff_expr = expression['vibenediff']
        if self.viblevelsdiff_expr is not None:
            self.term_label = 'MECH'
            self.F_vals = {}
        else:
            self.term_label = 'EL'
            self.viblevelsdiff_expr = []

        self.prefactorA = expression['termA_pref']
        self.prefactorB = expression['termB_pref']
        self.avrg_props_expr = expression['averaged_props']
        self.non_avrg_props_expr = expression['non_averaged_props']
        self.vibene_denom_expr = self.expression['vibene_denom']

        # collected avrg properties
        self.properties = [] #! unused later in class but used in TermsEvaluator.identify_to_precalculate()
        for p in self.avrg_props_expr:
            self.properties.append(MolProperty(name=p[0], cart_axes=p[2], nm_indices=p[1]))

        #! used in get_avrg_properties(); props together in one tuple; is a key for precalc dict
        self.nice_props = AveragedProps(self.properties)

        print('properties', self.properties)
        print('self.avrg_props_expr', self.avrg_props_expr, '\n')

        # collecting all vib ene diffs
        vibstates_diffs_collection = []
        # vib diffs with pert freqs
        for re in self.resonances_expr:
            l = re[0].split(',')
            ftuple = []
            for ll in l:
                if 'zero' not in ll.split('+'):
                    ftuple.append(len(set(ll.split('+'))))
                else:
                    ftuple.append(0)
            vibstates_diffs_collection.append((tuple(ftuple), True, re[1], re[0]))
        # vib diffs without pert freqs
        for vd in self.viblevelsdiff_expr:
            l = vd.split(',')
            ftuple = []
            for ll in l:
                if 'zero' not in ll.split('+'):
                    ftuple.append(len(set(ll.split('+'))))
                else:
                    ftuple.append(0)
            vibstates_diffs_collection.append((tuple(ftuple), False, None, vd))

        self.vibstatesdiff_objs = [VibStatesDiff(*i) for i in set(vibstates_diffs_collection)]

        from fractions import Fraction
        if isinstance(self.prefactorA, Fraction):
            self.prefactorA = float(self.prefactorA)

        # to be filled in after processing
        self.precalc_data = None

        self.collective_n_idx_rescond = None
        self.collective_n_idx_max = None

        # for current term
        allidx, res_idx = get_allparts_indices(self.expression)
        self.n_idx_rescond = res_idx
        self.n_idx_max = allidx
        self.collective_idx_counted = False


    def __repr__(self):
        s = f"\n{self.term_label} - {self.term_id}\n"
        for p in self.expression:
            s += f'\n    {p}'.ljust(25, ' ')+f'{self.expression[p]}'
        return s

    @tag('not_general', 'self.precalc_data is None')
    def for_ab(self, abc_comb):
        """
        making mn tuples for this term for ab combination
        #! not general; used when precalc_data is None
        """
        dict_id = {l: n for l,n in zip( abc_list[:len(abc_comb)], abc_comb)}
        dict_id['zero'] = 'zero'

        type12, type1 = self.resonances_expr
        m1_str, n1_str = type1[0].split(',')
        m1_tuple = tuple([str(dict_id[i]) for i in m1_str.split('+')])
        n1_tuple = tuple([str(dict_id[i]) for i in n1_str.split('+')])

        m12_str, n12_str = type12[0].split(',')

        m12_tuple = tuple(sorted([str(dict_id[i]) for i in m12_str.split('+')], key=int))
        n12_tuple = tuple(sorted([str(dict_id[i]) for i in n12_str.split('+')], key=int))
        return {'m1_tuple': m1_tuple, 'n1_tuple': n1_tuple, 'm12_tuple': m12_tuple, 'n12_tuple': n12_tuple}


    @tag('ok?')
    def load_calc_data(self, allstates: dict, harmonic_states: dict, properties_data,
                             mode_indices: np.ndarray|list, gammaCompsAll: np.ndarray|list):
        """
        Data for calculations


        """
        self.allstates = allstates
        self.allstates_Eh = {k: convNu2Ene(v) for k, v in self.allstates.items()}

        self.harmonic_states = harmonic_states
        self.harmonic_states_Eh = {k: convNu2Ene(v) for k, v in self.harmonic_states.items() if len(k)==1}

        from .spectrum_utils import dict2arraydict
        # changing format of storing states data
        self.states_arrays = dict2arraydict(self.allstates)
        self.states_arrays_Eh = dict2arraydict(self.allstates_Eh)
        self.harmonic_arrays = dict2arraydict(self.harmonic_states)
        self.harmonic_arrays_Eh = dict2arraydict(self.harmonic_states_Eh)

        self.allstates[('zero',)] = 0.
        self.allstates_Eh[("zero",)] = 0.0
        self.harmonic_states[('zero',)] = 0.
        self.harmonic_states_Eh[('zero',)] = 0.

        self.properties_data = properties_data
        self.mode_indices = mode_indices
        self.gammaCompsAll = gammaCompsAll


    @tag('general', 'restructure?')
    def get_resonance_location_general(self, abc_comb):
        """
        A resonance for this term for ab combination of modes

        if [-12][-1]:
                w1 = -mn_[-1]
                # w1 would be the axis in type_rc_mn = mn_[-x]
                        x = np.sign(type_rc[0]) * type_rc_mn where len(type_rc)==1 and e.g., type_rc_mn = mn_[-1]
                w2 = mn_[-12] + w1
        """
        # fixme: not quite general, fails with b and c indices, instead of a and b - ?
        # make dict with indices from rescond
        idx_str = {l: n for l,n in zip( abc_list[:len(abc_comb)], abc_comb)}

        sorted_vib_diffs = sorted([i for i in self.vibstatesdiff_objs if i.res_cond],
                                  key = lambda x: len(x.pf_type))

        axes_locs = []
        signes = []
        for vd in sorted_vib_diffs:
            indices_h = [k for i in vd.diff_str.split(',') for k in i.split('+') if k!='zero']
            idxs = tuple([idx_str[i] for i in indices_h])
            if not axes_locs:
                # fist identified axis
                if self.precalc_data is not None:
                    #! take care of units in precalc self.precalc_data['vibdiffs']
                    first_ax = self.precalc_data['vibdiffs'][tuple(sorted(vd.diff_type))][idxs]
                else:
                    mn_strs = compute_vibdiff(vd.diff_type, idxs)
                    first_ax = self.allstates_Eh[mn_strs[0]] - self.allstates_Eh[mn_strs[1]]
                axes_locs.append(first_ax * np.sign(vd.pf_type[0]))
                signes.append(np.sign(vd.pf_type[0]))

            else:
                next_sgn = np.sign(vd.pf_type[-1])
                if self.precalc_data is not None:
                    next_axis = self.precalc_data['vibdiffs'][tuple(sorted(vd.diff_type))][idxs]
                else:
                    mn_strs = compute_vibdiff(vd.diff_type, idxs)
                    next_axis = self.allstates_Eh[mn_strs[0]] - self.allstates_Eh[mn_strs[1]]

                # fixme: implicit minus here , also need to sum all
                prev = np.sum(np.array(axes_locs) * np.array(signes))
                axes_locs.append((prev + next_axis * next_sgn) * (-1) )
                signes.append(next_sgn)

        return [convNu2Ene(i, True) for i in axes_locs]


    @tag('general?')
    def get_res_factor(self, w1_rc, w2_rc, abc_comb, Gamma_rc,
                       condition=None, precalc=True):
        """
        A resonance factor for this term for ab combination of modes
        w1_rc, w2_rc - frequency arguments w1,w2 in reciprocal cm

        precalc - turns on or off precalculated resonance perturbing frequencies
        """
        if condition is None:
            condition = np.ones_like(w1_rc, dtype=bool)
        w1, w2 = convNu2Ene(w1_rc), convNu2Ene(w2_rc)

        d = {l: n for l,n in zip( abc_list[:len(abc_comb)], abc_comb)}
        Gamma_Eh = convNu2Ene(Gamma_rc)

        if self.precalc_data is not None and precalc:
            # fixme : no implied order!
            res_conds_vds = []
            res_conds_ax = []
            for vd in self.vibstatesdiff_objs:
                if vd.res_cond:
                    indices = tuple([d[i] for i in vd.diff_str.replace('+', ',').split(',') if i in d])
                    # units are Eh
                    vd_n = self.precalc_data['vibdiffs'][tuple(sorted(vd.diff_type))][indices]
                    if tuple(sorted(vd.diff_type)) != vd.diff_type:
                        vd_n *= -1
                    res_conds_vds.append(vd_n)
                    # units are cm-1
                    ax_n = self.precalc_data['res_conds'][tuple(np.array(vd.pf_type)*(-1.))]
                    res_conds_ax.append(ax_n)

            assert len(res_conds_vds) == len(res_conds_ax), 'Each resonance condition must have axes and vib diffs'

            RCs = [res_conds_vds[i]+convNu2Ene(res_conds_ax[i]) for i in range(len(res_conds_vds))]
            for i,rc in enumerate(RCs):
                if np.any(rc-1j*Gamma_Eh == 0):
                    print(i, rc)
                    raise ValueError("Division by zero detected!")

            from functools import reduce
            RC_prod = reduce(np.multiply, [i-1j*Gamma_Eh for i in RCs])
            if np.any(RC_prod == 0):
                raise ValueError("Division by zero detected!")
            r = np.where(condition, 1/RC_prod, 0.)

        else:
            #! not general

            # a, b = str(a), str(b)
            dict_mn_tuples = self.for_ab(abc_comb)

            vibdiff1 = self.allstates_Eh[dict_mn_tuples['m12_tuple']] - self.allstates_Eh[dict_mn_tuples['n12_tuple']]
            vibdiff2 = self.allstates_Eh[dict_mn_tuples['m1_tuple']] - self.allstates_Eh[dict_mn_tuples['n1_tuple']]

            if np.any((vibdiff1 + w1 - w2 -1j*Gamma_Eh) == 0):
                raise ValueError("Division by zero detected!")
            if np.any((vibdiff2 + w1 -1j*Gamma_Eh) == 0):
                raise ValueError("Division by zero detected!")
            r = np.where(condition, 1/(vibdiff1 + w1 - w2 -1j*Gamma_Eh)/(vibdiff2 + w1 -1j*Gamma_Eh), 0.)
        return r


    @tag('general')
    def get_properties_xyz(self, ABGD, abc_comb):
        """
        A step in calculation of averaged properties
        ABGD - alpha, beta, gamma, delta - so these are current choice of axes for greek indices
        """
        dict_id = {l: n for l,n in zip( abc_list[:len(abc_comb)], abc_comb)}

        # beta, alpha, delta, gamma ...
        # dict_ax_id = {'A': ABGD[0], 'B': ABGD[1], 'G': ABGD[2], 'D': ABGD[3]}
        dict_ax_id = {l: n for l,n in zip( greek_list[:len(ABGD)], ABGD)}

        propdict = {}
        for nn, p in enumerate(self.avrg_props_expr):
            # p is ('mu_QQ', ('a', 'b',), ('G',))
            # p is ('alpha_Q', ('c',), ('A', 'D'))
            indices = [dict_id[i] for i in p[1]] + [dict_ax_id[i] for i in p[2]]
            propdict[f'{nn}_'+p[0]] = self.properties_data[p[0]][*indices]

        return propdict

    @tag('general','naming!')
    def get_non_averaged_props(self, abc_comb):
        # dict_id = {'a': a, 'b': b, 'c': c}
        dict_id = {l: n for l,n in zip( abc_list[:len(abc_comb)], abc_comb)}

        if abc_comb not in self.non_avrg_props_expr[0][1]:
            idx = [dict_id[i] for i in self.non_avrg_props_expr[0][1]]
            self.F_vals[abc_comb] = self.properties_data['F_abc'][*idx] #! change names


    @tag('almost_general', 'make averg formula general')
    def get_avrg_properties(self, abc_comb, comps=False) -> float | tuple[float, dict]:
        """
        todo: maybe make a polarization choice which chooses then gammaCompsAll or smth
        fixme: not yet general; fix indices
        """
        # c is None if not given
        indices_dict = make_abc_dict(abc_comb)
        if self.precalc_data is None:
            components = {}
            total = 0.
            for ABGD in self.gammaCompsAll: #! is this general?
                props_dict = self.get_properties_xyz(ABGD, abc_comb)
                addition = np.prod(np.array([v for k,v in props_dict.items()])) # if 'mu' in k or 'alpha' in k
                total += addition
                if comps:
                    components[tuple(ABGD)] = (addition, props_dict)
            if abs(total)<1e-28:
                total = 0.

            # if type(total) == list or type(total) == np.ndarray:
            if isinstance(total, np.ndarray):
                print('OH NO, ndarray or list???')
            print(isinstance(total, np.ndarray))
            if comps:
                return total/15, components #! denominator should come with self.gammaCompsAll
            else:
                return total/15
        else:
            idxs = tuple([i for i in indices_dict.values() if i is not None])
            print(self.precalc_data['avrg_tensors'].keys())
            priv_names_tuple = tuple(sorted([p[0] for p in self.avrg_props_expr]))
            return self.precalc_data['avrg_tensors'][priv_names_tuple][idxs]


    @tag('almost_general')
    def get_factor_summed(self, ab_comb, comps=False, debugprint=False):
        """
        Sum of full factor over c index for given a,b
        ab_comb - rest of indices, index c is being summed over...
        remaining_length - number of indices to be summed over
        """

        components = {}
        remaining_length = self.n_idx_max - self.collective_n_idx_rescond # fixme? be careful ...

        ab_comb = tuple([i for i in ab_comb if i is not None])

        total = sum_over_suffixes(ab_comb,
                                  remaining_length,
                                  self.mode_indices,
                                  self.get_full_factor)

        if comps:
            return total, components
        else:
            return total


    @tag('almost_general', 'naming!')
    def get_full_factor(self, abc_comb, comps=False, debugprint=False):
        """
        product of: ene_factor, avrg_properties, (F_abc, viblevelsdiff)
        a, b, c=None

        #! not general yet
        """
        # a, b, c = abc_comb
        if debugprint:
            debugfunc('', f'get_full_factor called for {self.term_label} term')

        components = {}
        avrg_properties = self.get_avrg_properties(abc_comb, comps=comps)

        if avrg_properties==0:
            if comps:
                return 0., components
            else:
                return 0.

        ene_factor = self.get_ene_factor(abc_comb)
        product_all = ene_factor*avrg_properties # [0] if comps == True

        if debugprint:
            debugfunc(f'{ene_factor:.2e}', 'ene_factor')
            debugfunc(f'{avrg_properties:.2e}', 'avrg_properties_2')

        if comps:
            components['ene_factor'] = ene_factor
            components['avrg_properties'] = avrg_properties

        if self.viblevelsdiff_expr:

            self.get_non_averaged_props(abc_comb)

            if self.F_vals[abc_comb]==0:
                if comps:
                    return 0., components
                else:
                    return 0.

            vibdiff = self.get_viblevelsdiff(abc_comb)[0]
            if vibdiff==0:
                if comps:
                    return 0., components
                else:
                    return 0.

            product_all *= self.F_vals[abc_comb] * vibdiff

            if comps:
                components['F_abc'] = self.F_vals[abc_comb] #! change names
                components['viblevelsdiff'] = self.get_viblevelsdiff(abc_comb)[0]
            if debugprint:
                debugfunc(f'{self.F_vals[abc_comb]:.2e}', 'self.F_vals[(a,b,c)]') #! change names
                debugfunc(f'{self.get_viblevelsdiff(abc_comb)[0]:.2e}', 'self.get_viblevelsdiff(a, b, c)[0]')

        if comps:
            return product_all, components
        else:
            return product_all


    @tag('general', 'complete?')
    def get_ene_factor(self, abc_comb):
        """
        1/omega_a/omega_b/omega_c
        """

        d = {l: n for l,n in zip( abc_list[:len(abc_comb)], abc_comb)}
        modes = [i for i in d.values() if i is not None]

        if self.precalc_data is None:
            # collect freqs in Eh for given indices
            values = np.array([self.harmonic_states_Eh[(str(m),)] for m in modes])
            v = np.prod( values )

        else:
            tensor_label = self.vibene_denom_expr
            v = self.precalc_data['vibene_denoms'][tensor_label][tuple(modes)]

        if np.any(v == 0):
            raise ValueError("Division by zero detected!")
        return 1./v


    @tag('general')
    def get_viblevelsdiff(self, abc_comb):
        """
        1/omega_m,n + 1/omega_k,l
        a, b, c=None
        """
        d = {l: n for l,n in zip( abc_list[:len(abc_comb)], abc_comb)}

        if self.precalc_data is not None:
            vds = []
            for vd in self.vibstatesdiff_objs:
                if not vd.res_cond:
                    indices = tuple([d[i] for i in vd.diff_str.replace('+', ',').split(',') if i in d])
                    vd_n = self.precalc_data['vibdiffs'][tuple(sorted(vd.diff_type))][indices]
                    # opposite sign for reversed vib diff
                    if tuple(sorted(vd.diff_type)) != vd.diff_type:
                        vd_n *= -1
                    vds.append(vd_n)
                    if np.any(np.array(vd_n) == 0):
                        raise ValueError("Division by zero detected!")

            if np.any(np.array(vds) == 0):
                raise ValueError("Division by zero detected!")
            return np.sum(1./np.array(vds)), np.array(vds)

        else:
            raise NotImplementedError('get_viblevelsdiff not implemented without precalc data')

            a, b = str(a), str(b)
            dict_id = {'a': a, 'b': b, 'zero': 'zero'}
            if c is not None:
                c = str(c)
                dict_id['c'] = c

            total = []
            for e in self.viblevelsdiff_expr:
                m_str, n_str = e.split(',')

                l_m = [str(dict_id[i]) for i in m_str.split('+')]
                l_n = [str(dict_id[i]) for i in n_str.split('+')]

                if 'zero' not in l_m:
                    m_tuple = tuple(sorted(l_m, key=int))
                else:
                    m_tuple = tuple(l_m)
                if 'zero' not in l_n:
                    n_tuple = tuple(sorted(l_n, key=int))
                else:
                    n_tuple = tuple(l_n)

                total.append(self.allstates_Eh[m_tuple] - self.allstates_Eh[n_tuple])

            if np.any(np.array(total) == 0):
                raise ValueError("Division by zero detected!")

            total0 = 1./np.array(total)

            return np.sum(total0), np.array(total)


    @tag('general')
    def get_amplitudes(self, w1, w2, Gamma_rc, margin,
                       condition=None, collect_all=False, sel_abs=None,
                       debugprint=False):
        """
        gamma = prefnum * prefene * avrg * resonance

            ---->  New attributes after term.load_calc_data to term:
        {'harmonic_states', 'allstates_Eh',
            'allstates', 'harmonic_states_Eh',
            'properties_data', 'gammaCompsAll', 'mode_indices'}
        """
        result = 0.
        skipped = 0
        #! should get number of indices to sum over
        #! ab - are the indices from resonance conditions but ab should have others as None

        for ab in combinations_with_permutations(self.mode_indices, self.collective_n_idx_rescond):
            # full_abc = make_abc_tuple(ab, 3)
            if sel_abs is not None:
                if ab not in sel_abs:
                    skipped+=1
                    debug_deep(f'skipped {ab}', 'Term2D.get_intensity')
                    continue

            w1ab, w2ab = self.get_resonance_location_general(ab) #! this line isn't general

            resonance_is_ordered = w2ab > w1ab #! this line isn't general; should get from experiment?
            within_w1_window = (np.min(w1) + margin) <= w1ab <= (np.max(w1) - margin)
            within_w2_window = (np.min(w2) + margin) <= w2ab <= (np.max(w2) - margin)
            sufficient_margin_between = (w2ab - margin) > w1ab
            resonance_in_window = within_w1_window and within_w2_window and sufficient_margin_between

            if resonance_is_ordered and (collect_all or resonance_in_window):
                result += self.get_amplitudes_ab(ab, w1, w2, Gamma_rc,
                                                 condition=condition, debugprint=debugprint)[0]
            else:
                skipped += 1
                debug_deep(f'skipped later {ab}', 'Term2D.get_intensity')
                continue

        return result


    @tag('general')
    def get_amplitudes_ab(self, ab_comb, w1, w2, Gamma_rc,
                          condition=None, debugprint=False):
        """
        gamma = prefnum * prefene * avrg * resonance
        ab_comb are all except the ones to be summed over

        """

        # ab_comb are indices of res condition
        full_abc = make_abc_tuple(ab_comb, self.collective_n_idx_max)

        product_all = self.get_factor_summed(full_abc, comps=False,
                                             debugprint=debugprint)  # , components if comps==True

        components = {}

        if product_all==0.:
            return 0., components

        # ab_comb are indices of res conditions
        if isinstance(w1, float):
            resonance = self.get_res_factor(w1, w2, ab_comb, Gamma_rc, condition,
                                            precalc=False)

            debugfunc(f'{resonance:.2e}', 'resonance')
            debugfunc(f'{product_all:.2e}', 'product_all before prefA')
        else:
            resonance = self.get_res_factor(w1, w2, ab_comb, Gamma_rc, condition)

            debugfunc(f'{np.max(np.abs(resonance)):.2e}', 'resonance')
            debugfunc(f'{product_all:.2e}', 'product_all before prefA')

        product_all *= self.prefactorA * self.prefactorB
        result = product_all * resonance
        return result, components


    # fixme: unused; diagnostics?
    @tag('unused', 'diagnostics')
    def get_term_tree(self):
        """
        shows contributions/components
        """
        components = {}
        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            if self.term_label == 'EL':
                components[(a, b)] = self.get_full_factor(a, b)
            elif self.term_label == 'MECH':
                components[(a, b)] = self.get_factor_summed(a, b)
        return components


    # fixme: used in unused method; diagnostics?
    @tag('unused', 'diagnostics')
    def get_all_resonances(self, w2mw1=False):
        res = {}
        for ab in combinations_with_permutations(self.mode_indices, self.collective_n_idx_rescond):
            w1, w2 = self.get_resonance_location_general(ab)

            if w2mw1:
                res[ab] = (w1, w2-w1)
            else:
                res[ab] = (w1, w2)
        return res


    # fixme: unused
    @tag('unused')
    def fill_in_tensors(self, method, threshold=1e-18, use_threshold=False):
        """
        filling in 2d or 3d tensor with given method

        get_avrgprops_tensor(self, threshold=1e-18) - self.get_avrg_properties(a, b, c)[0]
        get_full_factor_tensor(self) - self.get_full_factor(a,b)[0]
        get_enefactor_tensor(self) - self.get_ene_factor(harm_modes_dict
        """
        t2 = np.zeros((max(self.mode_indices)+1, max(self.mode_indices)+1))
        t3 = np.zeros((max(self.mode_indices)+1, max(self.mode_indices)+1, max(self.mode_indices)+1))

        for ab in combinations_with_permutations(self.mode_indices, 2):
            a,b = ab
            if self.term_label == 'EL':
                t2[(a, b)] = method(a, b)
            else:
                for c in self.mode_indices:
                    t3[(a, b, c)] = method(a, b, c)

        all_zeros_t2 = not np.any(t2)

        if use_threshold:
            if all_zeros_t2:
                return np.where(np.abs(t3) > threshold, t3, 0.0)
            else:
                return np.where(np.abs(t3) > threshold, t2, 0.0)
        else:
            if all_zeros_t2:
                return t3
            else:
                return t2


    # fixme: unused; diagnostics?
    @tag('unused', 'diagnostics')
    def get_dotspectrum_df(self, Gamma_rc, margin, condition=None):
        """

        """
        import pandas as pd

        locations_dict = self.get_all_resonances(w2mw1=False)
        # print('locations_dict', locations_dict)
        from scipy.spatial import distance
        coords = np.array(list(locations_dict.keys()))
        distances = distance.cdist(coords, coords, 'euclidean')

        intensities_dict = {}

        for k in locations_dict:
            w1l, w2l = locations_dict[k]

            if w2l>w1l:
                intensities_dict[k] = self.get_amplitudes(w1l, w2l,
                                                          Gamma_rc, margin=0.,
                                                          condition=condition,
                                                          sel_abs=[k])
                # print('intensities_dict[k]', intensities_dict[k], k)

        data = {
            'ab': [(int(i[0]), int(i[1])) for i in list(intensities_dict.keys())],
            'intensity': [float(np.abs(i)) for i in list(intensities_dict.values())],
            'log10(abs(Intensity))': np.log10(np.abs(np.array(list(intensities_dict.values())))),
            'w1': [locations_dict[k][0] for k in intensities_dict.keys()],
            'w2': [locations_dict[k][1] for k in intensities_dict.keys()],
            'w2-w1': [locations_dict[k][1]-locations_dict[k][0] for k in intensities_dict.keys()]
        }

        data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}
        df = pd.DataFrame(data)
        df['term'] = self.term_id

        return df, distances



from itertools import product

def sum_over_suffixes(fixed_prefix, remaining_length, mode_indices, func):
    """
    Given a fixed_prefix like (0, 0) and remaining_length = 3,
    compute:
        sum(func(fixed_prefix + suffix))
    over all suffix ∈ product(mode_indices, repeat=remaining_length)
    """
    total = 0

    for suffix in product(mode_indices, repeat=remaining_length):
        full_input = fixed_prefix + suffix
        total += func(full_input)
    return total
