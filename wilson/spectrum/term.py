import numpy as np
from .tools import convNu2Ene, combinations_with_permutations
from .termeval_util_classes import MolProperty, AveragedProps, VibStatesDiff, DoubleDict
from wilson.debug import debugfunc, debug_deep


class Term2D:
    """
    Calculations using the expression.
        prefactor_num
        prefactor_ene
        property_1
        property_2
        property_3
        avrg_terms
        CFF (optional)

    {
    0: ((('a+b,a', 'zero,a'), None), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))), 1/24),
    1: ((('b,a', 'zero,a'), None), (('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))), 1/24)),
    2: ((('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc', 1.), -1/48.),
    3: ((('b,a', 'zero,a'), ('a+c,b', 'b+c,a')), (('mu_Q', ('a',)), ('alpha_Q', ('c',)), ('mu_Q', ('b',)), 'acb', 1.), -1/48.),
    4: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc', 0.5), -1/48.),
    5: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc', 0.5), -1/48.),
    6: ((('b,a', 'zero,a'), ('a,a+b', 'b,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc', -0.5), -1/48.),
    7: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc', -0.5, -1/48.))
    }

    """

    def __init__(self, term_id, expression):
        """
        expression[0] - resonance part
            expression[0][0] - actual resonances
            expression[0][1] - vib levels differences
        expression[1] - orientational average part
            expression[1][0] - property_tuple1: (property, mode indices)
            expression[1][1] - property_tuple2: (property, mode indices)
            expression[1][2] - property_tuple3: (property, mode indices)
            mech: expression[1][3] - CFF indices
            mech: expression[1][4] - term prefactor
        >expression[2] - overall el/mech prefactor (1/24. or -1/48.)

        So, what makes it work better (CPU and memory)? TODO based on Spectrum2D.intensity_both
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
        # print(self.nice_props)
        # self.property_tuple1, self.property_tuple2, self.property_tuple3 = self.property_tuples

        # collecting vib ene diffs
        self.vibdiff_symbolic = []
        vibstates_diffs_collection = []
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
        for vd in self.viblevelsdiff_expr:
            self.vibdiff_symbolic.append(vd)

            l = vd.split(',')
            ftuple = []
            for ll in l:
                if 'zero' not in ll.split('+'):
                    ftuple.append(len(set(ll.split('+'))))
                else:
                    ftuple.append(0)

            vibstates_diffs_collection.append((tuple(ftuple), False))
        vibstates_diffs_collection = set(vibstates_diffs_collection)
        self.vibstatesdiff_objs = [VibStatesDiff(*i) for i in vibstates_diffs_collection]

        self.part_prefactor = expression['termA_pref']

        # default numerical values of components -- ???? but they should be for given ab
        self.AVG = 1
        self.GP = 1
        self.TP = 1
        self.HEP = 1 # ab(c) indices
        self.CFF = 1 # abc indices
        self.RESCONDS = 1
        self.ODEN = 1

        if self.term_label != 'MECH':
            self.FacFull = self.AVG * self.GP * self.TP * self.HEP * self.CFF * self.ODEN
        else:
            self.FacFull = 1 # because should be summed over index c

        # addition = self.FacFull * self.RESCONDS
        #            self.FacFull = self.AVG * self.GP * self.TP * self.HEP * self.CFF * self.ODEN

        self.precalc_data = None


    def __repr__(self):
        s = f'{self.term_label} - {self.term_id}\n'
        for p in self.expression:
            s += f'\n    {p}'.ljust(25, ' ')+f'{self.expression[p]}'

        return s


    def for_ab(self, a,b):
        """

        """
        # print(a,b, '--------- a,b')
        a, b = str(a), str(b)

        dict_id = {'a': a, 'b': b, 'zero': 'zero'}

        type12, type1 = self.resonances_expr
        m1_str, n1_str = type1[0].split(',')
        # print('for ab - m1_str, n1_str', m1_str, n1_str)
        m1_tuple = tuple([str(dict_id[i]) for i in m1_str.split('+')])
        n1_tuple = tuple([str(dict_id[i]) for i in n1_str.split('+')])

        m12_str, n12_str = type12[0].split(',')
        # print('for ab - m12_str, n12_str', m12_str, n12_str)
        # print(m12_str.split('+'))
        # print([dict_id[i] for i in m12_str.split('+')], '\nyo')
        # print(dict_id)
        # print([i for i in m12_str.split('+')], '\nyo')

        m12_tuple = tuple(sorted([str(dict_id[i]) for i in m12_str.split('+')], key=int))
        n12_tuple = tuple(sorted([str(dict_id[i]) for i in n12_str.split('+')], key=int))
        return {'m1_tuple': m1_tuple, 'n1_tuple': n1_tuple, 'm12_tuple': m12_tuple, 'n12_tuple': n12_tuple}


    def for_ab_for_vd(self, vd, indices_str):
        # indices_str = {'a': a, 'b': b, 'zero': 'zero'}

        m12_str, n12_str = vd.diff_str.split(',')
        # print('m12_str', m12_str)
        # print('n12_str', n12_str)
        # print('1', [str(indices_str[i]) for i in m12_str.split('+')])
        # print('2', [str(indices_str[i]) for i in n12_str.split('+')])

        # m12_tuple = tuple(sorted([str(indices_str[i]) for i in m12_str.split('+')], key=int))
        # n12_tuple = tuple(sorted([str(indices_str[i]) for i in n12_str.split('+')], key=int))

        # m12_tuple = tuple(sorted([indices_str[i] for i in m12_str.split('+')], key=int))
        # n12_tuple = tuple(sorted([indices_str[i] for i in n12_str.split('+')], key=int))

        # m12_tuple = tuple(sorted([indices_str[i] for i in m12_str.split('+')]))
        # n12_tuple = tuple(sorted([indices_str[i] for i in n12_str.split('+')]))

        m12_tuple = tuple([indices_str[i] for i in m12_str.split('+')])
        n12_tuple = tuple([indices_str[i] for i in n12_str.split('+')])

        # print('indices_str', indices_str)
        # print('\nm12_tuple, n12_tuple', m12_tuple, n12_tuple)
        return m12_tuple, n12_tuple


    def load_calc_data(self, allstates: dict, harmonic_states: dict, properties_data,
                             mode_indices: np.ndarray|list, gammaCompsAll: np.ndarray|list):
        """
        Data for calculations


        """
        self.allstates = allstates
        self.allstates_Eh = {k: convNu2Ene(v) for k, v in self.allstates.items()}

        self.harmonic_states = harmonic_states
        self.harmonic_states_Eh = {k: convNu2Ene(v) for k, v in self.harmonic_states.items() if len(k)==1}

        from .termeval_util_classes import dict2arraydict


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


    # for given ab - good, modes_dict can be given elsewhere maybe, as a property for an instance
    def get_resonance_location(self, a, b):
        """
        A resonance for this term for ab combination of modes
        """
        a, b = str(a), str(b)

        dict_mn_tuples = self.for_ab(a, b)

        w1 = self.allstates[dict_mn_tuples['n1_tuple']] - self.allstates[dict_mn_tuples['m1_tuple']]
        w2 = self.allstates[dict_mn_tuples['m12_tuple']] - self.allstates[dict_mn_tuples['n12_tuple']] + w1

        return w1, w2


    # for given ab - good, modes_dict can be given elsewhere maybe, as a property for an instance
    def get_resonance_location_general(self, a, b): # spectralAxes????
        """
        A resonance for this term for ab combination of modes

        if [-12][-1]:
                w1 = -mn_[-1]
                # w1 would be the axis in type_rc_mn = mn_[-x]
                        x = np.sign(type_rc[0]) * type_rc_mn where len(type_rc)==1 and e.g., type_rc_mn = mn_[-1]
                w2 = mn_[-12] + w1
        """
        # a, b = str(a), str(b)
        idx_str = {'a': a, 'b': b, 'zero': 'zero'} # , 'c': c, 'd'

        if self.precalc_data is not None:
            sorted_vib_diffs = sorted([i for i in self.vibstatesdiff_objs if i.res_cond],
                                      key = lambda x: len(x.pf_type))

            axes_locs = []
            signes = []
            for vd in sorted_vib_diffs:
                indices_h = [k for i in vd.diff_str.split(',') for k in i.split('+') if k!='zero']

                if not axes_locs:
                    # fist identified axis
                    # q = self.for_ab_for_vd(vd, idx_str)
                    # print('aaaaa', q)
                    idxs = tuple([idx_str[i] for i in indices_h])
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
        # else:
            dict_mn_tuples = self.for_ab(idx_str['a'], idx_str['b'])
            # print(dict_mn_tuples)
            w1 = self.allstates[dict_mn_tuples['n1_tuple']] - self.allstates[dict_mn_tuples['m1_tuple']]
            w2 = self.allstates[dict_mn_tuples['m12_tuple']] - self.allstates[dict_mn_tuples['n12_tuple']] + w1
            print('w1, w2', w1, w2)
            return axes_locs
        else:
            dict_mn_tuples = self.for_ab(idx_str['a'], idx_str['b'])
            # print(dict_mn_tuples)
            w1 = self.allstates[dict_mn_tuples['n1_tuple']] - self.allstates[dict_mn_tuples['m1_tuple']]
            w2 = self.allstates[dict_mn_tuples['m12_tuple']] - self.allstates[dict_mn_tuples['n12_tuple']] + w1
            print('w1, w2', w1, w2)
            return w1, w2
        # return w1, w2


    # for given ab - good, modes_dict and others can be given elsewhere maybe, as a property for an instance?? or how
    def get_res_factor(self, w1_rc, w2_rc, a, b, Gamma_rc, condition=None):
        """
        A resonance factor for this term for ab combination of modes
        w1_rc, w2_rc - frequency arguments w1,w2 in reciprocal cm
        """
        if condition is None:
            condition = np.ones_like(w1_rc, dtype=bool)

        a, b = str(a), str(b)
        w1, w2 = convNu2Ene(w1_rc), convNu2Ene(w2_rc)

        dict_mn_tuples = self.for_ab(a, b)

        Gamma_Eh = convNu2Ene(Gamma_rc)
        r = np.where(condition,
                     1/(self.allstates_Eh[dict_mn_tuples['m12_tuple']] - self.allstates_Eh[dict_mn_tuples['n12_tuple']]
                        + w1 - w2 -1j*Gamma_Eh)/(self.allstates_Eh[dict_mn_tuples['m1_tuple']]
                                                 - self.allstates_Eh[dict_mn_tuples['n1_tuple']] + w1 -1j*Gamma_Eh), 0.)
        return r

    # for given ab(c) and ABGD - good, properties_data can be given elsewhere maybe, as a property for an instance?? or how;
    #                        ABGD - alpha, beta, gamma, delta - so these are current choice of axes for greek indices
    def get_properties(self, ABGD, a, b, c=None):
        dict_id = {'a': a, 'b': b, 'c': c}
        # beta, alpha, delta, gamma
        dict_ax_id = {'A': ABGD[0], 'B': ABGD[1], 'G': ABGD[2], 'D': ABGD[3]}
        propdict = {}
        for nn, p in enumerate(self.expression['averaged_props']):
            # p is ('mu_QQ', ('a', 'b',), ('G',))
            # p is ('alpha_Q', ('c',), ('A', 'D'))
            indices = [dict_id[i] for i in p[1]] + [dict_ax_id[i] for i in p[2]]
            propdict[f'{nn}_'+p[0]] = self.properties_data[p[0]][*indices]

        if self.term_label == 'MECH':
            # fixme
            if (a,b,c) not in self.expression['CFF'][1]: # fixme: keys don't exist
                idx = [dict_id[i] for i in self.expression['CFF'][1]] # fixme: key doesn't exist
                self.F_vals[(a,b,c)] = self.properties_data['F_abc'][*idx] # fixme: key doesn't exist
            propdict['F_abc'] = self.F_vals[(a,b,c)] # fixme: keys don't exist

        return propdict

    # for given ab - good, gammaCompsAll and properties_data can be given elsewhere maybe,
    #                  as a property for an instance?? or how
    #               maybe make a polarization choice which chooses then gammaCompsAll or smth

    def get_avrg_properties(self, a, b, c=None, comps=False):
        components = {}
        total = 0.
        for ABGD in self.gammaCompsAll:
            props_dict = self.get_properties(ABGD, a, b, c)
            addition = np.prod(np.array([v for k,v in props_dict.items() if 'mu' in k or 'alpha' in k]))
            total += addition
            if comps:
                components[tuple(ABGD)] = (addition, props_dict)

        if comps:
            return total/15, components
        else:
            return total/15


    def get_factor_summed(self, a, b, comps=False):
        """
        Sum of full factor over c index for given a,b
        """
        components = {}
        total = 0.
        for c in self.mode_indices:
            addition_2 = self.get_full_factor(a, b, c, comps)
            total += addition_2[0]
            if comps:
                components[c] = addition_2

        if comps:
            return total, components
        else:
            return total


    def get_full_factor(self, a, b, c=None, comps=False):
        """
        product of: ene_factor, avrg_properties, (F_abc, viblevelsdiff)
        """
        components = {}
        ene_factor = self.get_ene_factor(a, b, c)
        avrg_properties_2 = self.get_avrg_properties(a, b, c)
        product_all = ene_factor*avrg_properties_2 # [0] if comps == True

        if comps:
            components['ene_factor'] = ene_factor
            components['avrg_properties'] = avrg_properties_2

        # print(f'ene_factor {(a,b,c)}, {ene_factor:.3e}')
        # print(f'avrg_properties_2 {(a,b,c)}, {avrg_properties_2[0]:.3e}')

        if self.term_label=='MECH':
            product_all *= self.F_vals[(a,b,c)] * self.get_viblevelsdiff(a, b, c)[0]
            if comps:
                components['F_abc'] = self.F_vals[(a,b,c)]
                components['viblevelsdiff'] = self.get_viblevelsdiff(a, b, c)[0]

        if comps:
            return product_all, components
        else:
            return product_all


    # def get_full_factor_tensor(self):
    #     """
    #     Using self.get_full_factor
    #     """
    #     t2 = np.zeros((max(self.mode_indices)+1, max(self.mode_indices)+1))
    #     t3 = np.zeros((max(self.mode_indices)+1, max(self.mode_indices)+1, max(self.mode_indices)+1))
    #     for a in self.mode_indices:
    #         for b in self.mode_indices:
    #             if self.term_label == 'EL':
    #                 t2[(a, b)] = self.get_full_factor(a,b)[0]
    #             else:
    #                 for c in self.mode_indices:
    #                     t3[(a, b, c)] = self.get_full_factor(a,b,c)[0]
    #
    #     all_zeros_t2 = not np.any(t2)
    #     if all_zeros_t2:
    #         return t3
    #     else:
    #         return t2


    # def get_avrgprops_tensor(self, threshold=1e-18):
    #     """
    #     filling in 2d or 3d tensor with self.get_avrg_properties
    #     """
    #     t2 = np.zeros((max(self.mode_indices)+1, max(self.mode_indices)+1))
    #     t3 = np.zeros((max(self.mode_indices)+1, max(self.mode_indices)+1, max(self.mode_indices)+1))
    #     for a in self.mode_indices:
    #         for b in self.mode_indices:
    #             if self.term_label == 'EL':
    #                 t2[(a, b)] = self.get_avrg_properties(a, b)[0]
    #             else:
    #                 for c in self.mode_indices:
    #                     t3[(a, b, c)] = self.get_avrg_properties(a, b, c)[0]
    #
    #     all_zeros_t2 = not np.any(t2)
    #     if all_zeros_t2:
    #         return np.where(np.abs(t3)>threshold, t3, 0.)
    #     else:
    #         return np.where(np.abs(t3)>threshold, t2, 0.)


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

        # for a in self.mode_indices:
        #     for b in self.mode_indices:
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
        values = np.array([self.harmonic_states_Eh[(str(m),)] for m in modes])
        # values = np.array(list(self.harmonic_states_Eh.values()))

        return 1./ np.prod( values )


    def get_viblevelsdiff(self, a, b, c=None):
        """
        1/omega_m,n + 1/omega_k,l
        """
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
        total0 = 1./np.array(total)

        # if recip:
        #     return np.sum(total0)
        # else:
        #     return np.array(total)
        return np.sum(total0), np.array(total)


    # def get_enefactor_tensor(self):
    #
    #     t2 = np.zeros((max(mode_indices)+1, max(mode_indices)+1))
    #     t3 = np.zeros((max(mode_indices)+1, max(mode_indices)+1, max(mode_indices)+1))
    #     for a in mode_indices:
    #         for b in mode_indices:
    #             if self.term_label == 'EL':
    #                 t2[(a, b)] = self.get_ene_factor(harm_modes_dict, a, b)
    #             else:
    #                 for c in mode_indices:
    #                     t3[(a, b, c)] = self.get_ene_factor(harm_modes_dict, a, b, c)
    #
    #     all_zeros_t2 = not np.any(t2)
    #     if all_zeros_t2:
    #         return t3
    #     else:
    #         return t2


    # fixme: unused
    def get_term_tree(self):
        """
        shows contributions/components
        """
        components = {}
        # for a in self.mode_indices:
        #     for b in self.mode_indices:
        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            if self.term_label == 'EL':
                # components[((a,b), (self.get_resonance_location(modes_dict, a, b)))] = self.get_full_factor(gammaCompsAll, properties_data,
                #                                          modes_dict, a,b)
                components[(a, b)] = self.get_full_factor(a, b)
            elif self.term_label == 'MECH':
                # components[((a, b), (self.get_resonance_location(modes_dict, a, b)))] = self.get_factor_summed(gammaCompsAll, properties_data,
                #                                             modes_dict, mode_indices, a, b)
                components[(a, b)] = self.get_factor_summed(a, b)
        return components


    # fixme: used in unused method
    def get_all_resonances(self, w2mw1=False):
        res = {}
        # for a in self.mode_indices:
        #     for b in self.mode_indices:
        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            w1, w2 = self.get_resonance_location(a, b)

            if w2mw1:
                res[(a,b)] = (w1, w2-w1)
            else:
                res[(a,b)] = (w1, w2)
        return res


    def get_intensity(self, w1, w2, Gamma_rc, margin,
                      condition=None, collect_all=False, sel_abs=None):
        """
        gamma = prefnum * prefene * avrg * resonance

            ---->  New attributes after term.load_calc_data to term:
        {'harmonic_states', 'allstates_Eh',
            'allstates', 'harmonic_states_Eh',
            'properties_data', 'gammaCompsAll', 'mode_indices'}
        """
        result = 0.
        skipped = 0

        # self.layers = {}

        # for a in self.mode_indices:
        #     for b in self.mode_indices:
        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            if sel_abs is not None:
                if (a,b) not in sel_abs:
                    skipped+=1
                    debug_deep(f'skipped {(a, b)}', 'Term2D.get_intensity')
                    continue

            w1ab, w2ab = self.get_resonance_location(a, b)
            # check if resonance is in window w1,w2
            # if it's a single pair of w1,w2 -
            if w2ab>w1ab:
                if ((np.min(w1) + margin <= w1ab <= np.max(w1) - margin)
                        and (np.min(w2) + margin <= w2ab <= np.max(w2) - margin)
                        and (w2ab-margin)>w1ab) or collect_all:
                    result += self.get_intensity_ab(a, b, w1, w2, Gamma_rc,
                          condition=condition)[0]
                    # print('added')
                else:
                    skipped += 1
                    debug_deep(f'skipped later {(a,b)}', 'Term2D.get_intensity')
                    continue

        # print('skipped', skipped)
        # ab_combinations = list(combinations_with_permutations(self.mode_indices, 2))
        # print('ab_combinations', len(ab_combinations))
        return result

    def get_intensity_ab(self, a, b, w1, w2, Gamma_rc,
                      condition=None):
        """
        gamma = prefnum * prefene * avrg * resonance
        """
        # from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices
        # gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)

        # w1ab, w2ab = self.get_resonance_location(modes_dict, a, b)
        # if ((np.min(w1) + margin <= w1ab <= np.max(w1) - margin)
        #         and (np.min(w2) + margin <= w2ab <= np.max(w2) - margin)
        #         and (w2ab-margin)>w1ab):
        if self.term_label=='EL':
            # full_prefactor * resonance
            product_all, components= self.get_full_factor(a, b, comps=True) # , components if comps==True
            product_all /= 24.
            # print(f"a,b: {(a, b)}, factor: {product_all:.2e}")
            # result = (product_all*self.get_res_factor(w1, w2, a, b, Gamma_rc, condition))
            # return result, components

        else:
            product_all, components = self.get_factor_summed(a, b, comps=True) # , components if comps==True
            product_all /= -48.

            components = {}
            # for k,v in components.items():
            #     if v[0]!=0.:
            #         shortcomponents[k] = {'full_product_abc':v[0], 'ene_factor':v[1]['ene_factor'],
            #                               'avrg_properties':v[1]['avrg_properties'][0],
            #                               'F_abc':v[1]['F_abc'],
            #                               'viblevelsdiff':v[1]['viblevelsdiff']}

        result = (product_all*self.get_res_factor(w1, w2, a, b, Gamma_rc, condition))
        return result, components

    # fixme: unused
    def get_vibdiff_tensor(self):

        viblevelsdiff_tensor = np.zeros((max(self.mode_indices)+1, max(self.mode_indices)+1, max(self.mode_indices)+1))
        viblevelsdiff_tensor2 = np.zeros((max(self.mode_indices)+1, max(self.mode_indices)+1, max(self.mode_indices)+1, 2))
        # for a in self.mode_indices:
        #     for b in self.mode_indices:
        #         for c in self.mode_indices:
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
        # print(distances)

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