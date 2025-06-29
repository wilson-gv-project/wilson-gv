import numpy as np

from .tools import convNu2Ene, combinations_with_permutations
from .spectrum_utils import MolProperty, AveragedProps, VibStatesDiff, DoubleDict
from wilson.debug import debugfunc, debug_deep


def for_ab_for_vd(vd, indices_str):
    """
    make mn tuples for given vib diff

    indices_str = {'a': a, 'b': b, 'zero': 'zero'}
    """

    m12_str, n12_str = vd.diff_str.split(',')

    m12_tuple = tuple([indices_str[i] for i in m12_str.split('+')])
    n12_tuple = tuple([indices_str[i] for i in n12_str.split('+')])

    return m12_tuple, n12_tuple


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
            AVG - averaged props
            GP - general prefactor (e.g., 1/24 or 1/-48)
            TP - term prefactor (e.g., -1/2 or 1/2)
            HEP - harmonic vib ene prefactor
            (CFF) - optional CFF
            RESCONDS - resonance conditions (list, will be multiplied together)
            ODEN - other denominators (e.g., vib ene levels denominators; will be added together)

        """
        self.term_id = term_id
        self.expression = expression

        # expressions
        self.resonances_expr = expression['resonances']
        self.viblevelsdiff_expr = expression['vibenediff']
        if self.viblevelsdiff_expr is not None:
            self.term_label = 'MECH'
            self.F_vals = {}
        else:
            self.term_label = 'EL'
            self.viblevelsdiff_expr = []

        self.vibstates_all = []

        self.properties = []
        for p in expression['averaged_props'][:3]:
            self.properties.append(MolProperty(name=p[0], cart_axes=p[2], nm_indices=p[1]))

        self.property_simple_tuples = tuple([p.simple_tuple for p in self.properties])
        self.nice_props = AveragedProps(self.properties)

        # collecting vib ene diffs
        self.vibdiff_symbolic = []
        vibstates_diffs_collection = []
        # vib diffs with pert freqs
        for re in self.resonances_expr:
            self.vibdiff_symbolic.append(re[0])
            l = re[0].split(',')
            ftuple = []
            for ll in l:
                if 'zero' not in ll.split('+'):
                    ftuple.append(len(set(ll.split('+'))))
                else:
                    ftuple.append(0)
            vibstates_diffs_collection.append((tuple(ftuple), True, re[1], re[0]))
        # vib diffs withou pert freqs
        for vd in self.viblevelsdiff_expr:
            self.vibdiff_symbolic.append(vd)
            l = vd.split(',')
            ftuple = []
            for ll in l:
                if 'zero' not in ll.split('+'):
                    ftuple.append(len(set(ll.split('+'))))
                else:
                    ftuple.append(0)

            vibstates_diffs_collection.append((tuple(ftuple), False, None, vd))
        vibstates_diffs_collection = set(vibstates_diffs_collection)
        self.vibstatesdiff_objs = [VibStatesDiff(*i) for i in vibstates_diffs_collection]

        from fractions import Fraction
        if isinstance(self.expression['termA_pref'], Fraction):
            self.expression['termA_pref'] = float(self.expression['termA_pref'])

        self.precalc_data = None


    def __repr__(self):
        s = f'\n{self.term_label} - {self.term_id}\n'
        for p in self.expression:
            s += f'\n    {p}'.ljust(25, ' ')+f'{self.expression[p]}'
        return s


    def for_ab(self, a,b):
        """
        making mn tuples for this term for ab combination
        """
        a, b = str(a), str(b)

        dict_id = {'a': a, 'b': b, 'zero': 'zero'}

        type12, type1 = self.resonances_expr
        m1_str, n1_str = type1[0].split(',')
        m1_tuple = tuple([str(dict_id[i]) for i in m1_str.split('+')])
        n1_tuple = tuple([str(dict_id[i]) for i in n1_str.split('+')])

        m12_str, n12_str = type12[0].split(',')

        m12_tuple = tuple(sorted([str(dict_id[i]) for i in m12_str.split('+')], key=int))
        n12_tuple = tuple(sorted([str(dict_id[i]) for i in n12_str.split('+')], key=int))
        return {'m1_tuple': m1_tuple, 'n1_tuple': n1_tuple, 'm12_tuple': m12_tuple, 'n12_tuple': n12_tuple}

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


    def get_resonance_location(self, a, b):
        """
        A resonance for this term for ab combination of modes.

        Ad hoc implementation
        """
        a, b = str(a), str(b)

        dict_mn_tuples = self.for_ab(a, b)

        w1 = self.allstates[dict_mn_tuples['n1_tuple']] - self.allstates[dict_mn_tuples['m1_tuple']]
        w2 = self.allstates[dict_mn_tuples['m12_tuple']] - self.allstates[dict_mn_tuples['n12_tuple']] + w1

        return w1, w2


    def get_resonance_location_general(self, abc_comb):
        """
        A resonance for this term for ab combination of modes

        if [-12][-1]:
                w1 = -mn_[-1]
                # w1 would be the axis in type_rc_mn = mn_[-x]
                        x = np.sign(type_rc[0]) * type_rc_mn where len(type_rc)==1 and e.g., type_rc_mn = mn_[-1]
                w2 = mn_[-12] + w1
        """
        # fixme: not quite general, fails with b and c indices, instead of a and b
        # a, b = str(a), str(b)
        from tests import abc_list
        idx_str = {l: n for l,n in zip( abc_list[:len(abc_comb)], abc_comb)}
        # print('\nidx_str', idx_str)
        # idx_str = {'a': a, 'b': b, 'zero': 'zero'} # , 'c': c, 'd'

        if self.precalc_data is not None:
            sorted_vib_diffs = sorted([i for i in self.vibstatesdiff_objs if i.res_cond],
                                      key = lambda x: len(x.pf_type))

            axes_locs = []
            signes = []
            for vd in sorted_vib_diffs:
                indices_h = [k for i in vd.diff_str.split(',') for k in i.split('+') if k!='zero']
                # print(indices_h, vd)
                if not axes_locs:
                    # fist identified axis
                    idxs = tuple([idx_str[i] for i in indices_h])
                    # take care of units in precalc self.precalc_data['vibdiffs']
                    first_ax = self.precalc_data['vibdiffs'][tuple(sorted(vd.diff_type))][idxs]
                    axes_locs.append(first_ax * np.sign(vd.pf_type[0]))
                    signes.append(np.sign(vd.pf_type[0]))
                else:
                    idxs = tuple([idx_str[i] for i in indices_h])
                    next_sgn = np.sign(vd.pf_type[-1])
                    next_axis = self.precalc_data['vibdiffs'][tuple(sorted(vd.diff_type))][idxs]
                    # fixme: implicit minus here , also need to sum all
                    prev = np.sum(np.array(axes_locs) * np.array(signes))
                    axes_locs.append((prev + next_axis * next_sgn) * (-1) )
                    signes.append(next_sgn)
            # fixme: test against old way for the same data?
            return [convNu2Ene(i, True) for i in axes_locs]
        else:
            dict_mn_tuples = self.for_ab(idx_str['a'], idx_str['b'])
            w1 = self.allstates[dict_mn_tuples['n1_tuple']] - self.allstates[dict_mn_tuples['m1_tuple']]
            w2 = self.allstates[dict_mn_tuples['m12_tuple']] - self.allstates[dict_mn_tuples['n12_tuple']] + w1
            return w1, w2


    def get_res_factor(self, w1_rc, w2_rc, a, b, Gamma_rc, condition=None):
        """
        A resonance factor for this term for ab combination of modes
        w1_rc, w2_rc - frequency arguments w1,w2 in reciprocal cm
        """
        if condition is None:
            condition = np.ones_like(w1_rc, dtype=bool)

        w1, w2 = convNu2Ene(w1_rc), convNu2Ene(w2_rc)
        d = {'a': a, 'b': b}
        Gamma_Eh = convNu2Ene(Gamma_rc)

        if self.precalc_data is not None:
            # fixme : no implied order!
            res_conds_vds = []
            res_conds_ax = []
            for vd in self.vibstatesdiff_objs:
                if vd.res_cond:
                    indices = tuple([d[i] for i in vd.diff_str.replace('+', ',').split(',') if i in d])
                    vd_n = self.precalc_data['vibdiffs'][tuple(sorted(vd.diff_type))][indices]
                    if tuple(sorted(vd.diff_type)) != vd.diff_type:
                        vd_n *= -1
                    res_conds_vds.append(vd_n)
                    ax_n = self.precalc_data['res_conds'][tuple(np.array(vd.pf_type)*(-1.))]
                    res_conds_ax.append(ax_n)
            vibdiff1 = convNu2Ene(res_conds_vds[0]+res_conds_ax[0])
            vibdiff2 = convNu2Ene(res_conds_vds[1]+res_conds_ax[1])
            if np.any((vibdiff1-1j*Gamma_Eh) == 0):
                raise ValueError("Division by zero detected!")
            if np.any((vibdiff2-1j*Gamma_Eh) == 0):
                raise ValueError("Division by zero detected!")
            r = np.where(condition, 1/(vibdiff1-1j*Gamma_Eh)/(vibdiff2-1j*Gamma_Eh), 0.)

        else:
            a, b = str(a), str(b)
            dict_mn_tuples = self.for_ab(a, b)

            vibdiff1 = self.allstates_Eh[dict_mn_tuples['m12_tuple']] - self.allstates_Eh[dict_mn_tuples['n12_tuple']]
            vibdiff2 = self.allstates_Eh[dict_mn_tuples['m1_tuple']] - self.allstates_Eh[dict_mn_tuples['n1_tuple']]

            if np.any((vibdiff1 + w1 - w2 -1j*Gamma_Eh) == 0):
                raise ValueError("Division by zero detected!")
            if np.any((vibdiff2 + w1 -1j*Gamma_Eh) == 0):
                raise ValueError("Division by zero detected!")
            r = np.where(condition, 1/(vibdiff1 + w1 - w2 -1j*Gamma_Eh)/(vibdiff2 + w1 -1j*Gamma_Eh), 0.)
        return r


    def get_properties(self, ABGD, a, b, c=None):
        """
        A step in calculation of averaged properties
        ABGD - alpha, beta, gamma, delta - so these are current choice of axes for greek indices
        """
        dict_id = {'a': a, 'b': b, 'c': c}
        # beta, alpha, delta, gamma
        dict_ax_id = {'A': ABGD[0], 'B': ABGD[1], 'G': ABGD[2], 'D': ABGD[3]}
        propdict = {}
        for nn, p in enumerate(self.expression['averaged_props']):
            # p is ('mu_QQ', ('a', 'b',), ('G',))
            # p is ('alpha_Q', ('c',), ('A', 'D'))
            indices = [dict_id[i] for i in p[1]] + [dict_ax_id[i] for i in p[2]]
            propdict[f'{nn}_'+p[0]] = self.properties_data[p[0]][*indices]

        return propdict


    def get_non_averaged_props(self, a, b, c=None):
        dict_id = {'a': a, 'b': b, 'c': c}

        if (a,b,c) not in self.expression['non_averaged_props'][0][1]:
            idx = [dict_id[i] for i in self.expression['non_averaged_props'][0][1]]
            self.F_vals[(a,b,c)] = self.properties_data['F_abc'][*idx]


    def get_avrg_properties(self, a, b, c=None, comps=False):
        """
        todo: maybe make a polarization choice which chooses then gammaCompsAll or smth
        """
        if self.precalc_data is None:
            components = {}
            total = 0.
            for ABGD in self.gammaCompsAll:
                props_dict = self.get_properties(ABGD, a, b, c)
                # print(props_dict)
                addition = np.prod(np.array([v for k,v in props_dict.items() if 'mu' in k or 'alpha' in k]))
                total += addition
                if comps:
                    components[tuple(ABGD)] = (addition, props_dict)
            if abs(total)<1e-28:
                total = 0.
            if comps:
                return total/15, components
            else:
                return total/15
        else:

            if c is None:
                return self.precalc_data['avrg_tensors'][self.property_simple_tuples][a,b]
            else:
                return self.precalc_data['avrg_tensors'][self.property_simple_tuples][a,b,c]


    def get_factor_summed(self, a, b, comps=False, debugprint=False):
        """
        Sum of full factor over c index for given a,b
        """
        components = {}
        total = 0.
        for c in self.mode_indices:
            addition_2 = self.get_full_factor(a, b, c, comps, debugprint=debugprint)
            total += addition_2[0]
            if comps:
                components[c] = addition_2

        if comps:
            return total, components
        else:
            return total


    def get_full_factor(self, a, b, c=None, comps=False, debugprint=False):
        """
        product of: ene_factor, avrg_properties, (F_abc, viblevelsdiff)
        """
        if debugprint:
            debugfunc('', f'get_full_factor called for {self.term_label} term')

        components = {}
        avrg_properties_2 = self.get_avrg_properties(a, b, c) # todo: a single value for given abc
        if avrg_properties_2==0:
            if comps:
                return 0., components
            else:
                return 0.
        ene_factor = self.get_ene_factor(a, b, c) # todo: a single value for given abc
        product_all = ene_factor*avrg_properties_2 # [0] if comps == True
        if debugprint:
            debugfunc(f'{ene_factor:.2e}', 'ene_factor')
            debugfunc(f'{avrg_properties_2:.2e}', 'avrg_properties_2')

        if comps:
            components['ene_factor'] = ene_factor
            components['avrg_properties'] = avrg_properties_2

        if self.term_label=='MECH':

            vibdiff = self.get_viblevelsdiff(a, b, c)[0]
            if vibdiff==0:
                if comps:
                    return 0., components
                else:
                    return 0.

            self.get_non_averaged_props(a, b, c)

            if self.F_vals[(a,b,c)]==0:
                if comps:
                    return 0., components
                else:
                    return 0.
            product_all *= self.F_vals[(a,b,c)] * vibdiff

            if comps:
                components['F_abc'] = self.F_vals[(a,b,c)]
                components['viblevelsdiff'] = self.get_viblevelsdiff(a, b, c)[0]
            if debugprint:
                debugfunc(f'{self.F_vals[(a,b,c)]:.2e}', 'self.F_vals[(a,b,c)]')
                debugfunc(f'{self.get_viblevelsdiff(a, b, c)[0]:.2e}', 'self.get_viblevelsdiff(a, b, c)[0]')

        if comps:
            return product_all, components
        else:
            return product_all


    # fixme: unused
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


    # @staticmethod
    def get_ene_factor(self, a, b, c=None):
        """
        1/omega_a/omega_b/omega_c
        """
        modes = [a, b] if c is None else [a, b, c]
        if self.precalc_data is None:
            # collect freqs in Eh for given indices
            values = np.array([self.harmonic_states_Eh[(str(m),)] for m in modes])

            if np.any(np.prod( values ) == 0):
                raise ValueError("Division by zero detected!")
            # make inverse of product
            return 1./ np.prod( values )
        else:
            tensor_label = self.expression['vibene_denom']
            v = self.precalc_data['vibene_denoms'][tensor_label][tuple(modes)]
            if np.any(v == 0):
                raise ValueError("Division by zero detected!")
            return 1./v


    def get_viblevelsdiff(self, a, b, c=None):
        """
        1/omega_m,n + 1/omega_k,l
        """
        d = {'a': a, 'b': b, 'c': c}

        if self.precalc_data is not None:
            calc_tensors = [tuple(sorted(vd.diff_type)) for vd in self.vibstatesdiff_objs if not vd.res_cond]
            # for ct in calc_tensors:
            vds = []
            for vd in self.vibstatesdiff_objs:
                if not vd.res_cond:
                    indices = tuple([d[i] for i in vd.diff_str.replace('+', ',').split(',') if i in d])
                    vd_n = self.precalc_data['vibdiffs'][tuple(sorted(vd.diff_type))][indices]
                    if tuple(sorted(vd.diff_type)) != vd.diff_type:
                        vd_n *= -1
                    vds.append(vd_n)
                    if np.any(np.array(vd_n) == 0):
                        print('\n', vd)
                        print(vd_n)
                        print('indices', indices)
                        print('tuple(sorted(vd.diff_type))', tuple(sorted(vd.diff_type)))
                        print(self.precalc_data['vibdiffs'][tuple(sorted(vd.diff_type))])
                        raise ValueError("Division by zero detected!")
            if np.any(np.array(vds) == 0):
                raise ValueError("Division by zero detected!")
            return np.sum(1./np.array(vds)), np.array(vds)

        else:
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


    # fixme: unused
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


    # fixme: used in unused method
    def get_all_resonances(self, w2mw1=False):
        res = {}
        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            w1, w2 = self.get_resonance_location(a, b)

            if w2mw1:
                res[(a,b)] = (w1, w2-w1)
            else:
                res[(a,b)] = (w1, w2)
        return res


    def get_intensity(self, w1, w2, Gamma_rc, margin,
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

        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            if sel_abs is not None:
                if (a,b) not in sel_abs:
                    skipped+=1
                    debug_deep(f'skipped {(a, b)}', 'Term2D.get_intensity')
                    print('skipped', a, b)
                    continue

            # todo: remove these; it seems to work... but make a test??
            w1ab, w2ab = self.get_resonance_location_general(ab)

            resonance_is_ordered = w2ab > w1ab
            within_w1_window = (np.min(w1) + margin) <= w1ab <= (np.max(w1) - margin)
            within_w2_window = (np.min(w2) + margin) <= w2ab <= (np.max(w2) - margin)
            sufficient_margin_between = (w2ab - margin) > w1ab
            resonance_in_window = within_w1_window and within_w2_window and sufficient_margin_between

            if resonance_is_ordered and (collect_all or resonance_in_window):
                result += self.get_intensity_ab(a, b, w1, w2, Gamma_rc,
                                                condition=condition, debugprint=debugprint)[0]
            else:
                skipped += 1
                debug_deep(f'skipped later {(a,b)}', 'Term2D.get_intensity')
                continue

        return result

    def get_intensity_ab(self, a, b, w1, w2, Gamma_rc,
                      condition=None, debugprint=False):
        """
        gamma = prefnum * prefene * avrg * resonance
        """

        if self.term_label=='EL':
            # full_prefactor * resonance
            product_all, components= self.get_full_factor(a, b, comps=True, debugprint=debugprint) # , components if comps==True

        else:
            product_all, components = self.get_factor_summed(a, b, comps=True, debugprint=debugprint) # , components if comps==True

            components = {}
            # for k,v in components.items():
            #     if v[0]!=0.:
            #         shortcomponents[k] = {'full_product_abc':v[0], 'ene_factor':v[1]['ene_factor'],
            #                               'avrg_properties':v[1]['avrg_properties'][0],
            #                               'F_abc':v[1]['F_abc'],
            #                               'viblevelsdiff':v[1]['viblevelsdiff']}
        if product_all==0.:
            return 0., components

        resonance = self.get_res_factor(w1, w2, a, b, Gamma_rc, condition)

        if isinstance(w1, float):
            debugfunc(f'{resonance:.2e}', 'resonance')
            debugfunc(f'{product_all:.2e}', 'product_all before prefA')
        else:
            debugfunc(f'{np.max(np.abs(resonance)):.2e}', 'resonance')
            debugfunc(f'{product_all:.2e}', 'product_all before prefA')


        product_all *= self.expression['termA_pref']
        result = product_all * resonance
        return result, components

    # fixme: unused
    def get_vibdiff_tensor(self):

        viblevelsdiff_tensor = np.zeros((max(self.mode_indices)+1, max(self.mode_indices)+1, max(self.mode_indices)+1))
        viblevelsdiff_tensor2 = np.zeros((max(self.mode_indices)+1, max(self.mode_indices)+1, max(self.mode_indices)+1, 2))
        for abc in combinations_with_permutations(self.mode_indices, 3):
            a, b, c = abc
            viblevelsdiff_tensor[a, b, c] = self.get_viblevelsdiff(a, b, c)[0]
            viblevelsdiff_tensor2[a, b, c, :] = self.get_viblevelsdiff(a, b, c)[1]

        return viblevelsdiff_tensor, viblevelsdiff_tensor2


    # fixme: unused
    def get_dotspectrum_df(self, Gamma_rc, margin, condition=None):
        """

        """
        import pandas as pd

        locations_dict = self.get_all_resonances(w2mw1=False)
        from scipy.spatial import distance
        coords = np.array(list(locations_dict.keys()))
        distances = distance.cdist(coords, coords, 'euclidean')

        intensities_dict = {}

        for k in locations_dict:
            w1l, w2l = locations_dict[k]
            if w2l>w1l:
                intensities_dict[k] = self.get_intensity(w1l, w2l,
                                                         Gamma_rc, margin=margin,
                                                         condition=condition)

        data = {
            'ab': [(int(i[0]), int(i[1])) for i in list(intensities_dict.keys())],
            'intensity': [float(i) for i in list(intensities_dict.values())],
            'log10(Intensity)': np.log10(np.array(list(intensities_dict.values()))),
            'w1': [locations_dict[k][0] for k in intensities_dict.keys()],
            'w2': [locations_dict[k][1] for k in intensities_dict.keys()],
            'w2-w1': [locations_dict[k][1]-locations_dict[k][0] for k in intensities_dict.keys()]
        }

        data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}
        df = pd.DataFrame(data)
        df['term'] = self.term_id

        return df, distances