from typing import List
import string

import numpy as np
from dataclasses import dataclass, field
from collections import Counter
from .tools import convNu2Ene, combinations_with_permutations, DoubleDict
from wilson.debug import debugfunc, debug_deep

@dataclass
class MolProperty:
    """
    MolProperty(name='mu_Q', cart_axes=('B',), nm_indices=('a',)),
    MolProperty(name='mu_QQ', cart_axes=('G',), nm_indices=('a', 'b'))

    MolProperty(name='alpha_Q', cart_axes=('A', 'D'), nm_indices=('b',)),
    MolProperty(name='alpha_QQ', cart_axes=('A', 'D'), nm_indices=('a', 'b')),

    MolProperty(name='CFF', cart_axes=(), nm_indices=('a', 'b', 'c'))

    """
    name: str
    cart_axes: tuple[str]
    nm_indices: tuple[str]
    tensor: np.ndarray = 0.

    @property
    def simple_tuple(self):
        """
        len(self.cart_axes) - number of EL perturbations
        len(self.nm_indices) - number of normal coordinates derivatives
        """
        return tuple([len(self.cart_axes), len(self.nm_indices)])

    # def __repr__(self):
        # return rf'\{self.name.strip("_Q")}_{','.join(list(self.cart_axes))}__dQ{','.join(list(self.nm_indices))}'
        # return f'{self.simple_tuple}'


@dataclass
class VibState:
    quanta_dict: dict
    freq: float


@dataclass
class VibStatesDiff:
    diff_type: tuple

    def __eq__(self, other):
        if not isinstance(other, VibStatesDiff):
            return False
        return (
            self.diff_type == other.diff_type
        )

    def __hash__(self):
        # Since Counter doesn't have a hash, we convert it to a frozenset of items
        return hash(self.diff_type)

    def __repr__(self):
        return f'VibStatesDiff: {self.diff_type}'


@dataclass
class AveragedProps:
    props: List[MolProperty] = field(default_factory=list)

    @property
    def cart_axes(self):
        return tuple([p.cart_axes for p in self.props])

    @property
    def nm_indices(self):
        return tuple([p.nm_indices for p in self.props])

    # def __eq__(self, other):
    #     if not isinstance(other, AveragedProps):
    #         return False
    #     return (
    #         self.cart_axes == other.cart_axes and
    #         Counter(self.nm_indices) == Counter(other.nm_indices)
    #     )

    def __eq__(self, other):
        if not isinstance(other, AveragedProps):
            return False
        return (
            self.cart_axes == other.cart_axes and
            sorted(self.nm_indices) == sorted(other.nm_indices)
        )

    def __hash__(self):
        # Since Counter doesn't have a hash, we convert it to a frozenset of items
        return hash((self.cart_axes, frozenset(Counter(self.nm_indices).items())))

    def __repr__(self):
        return f'\nAveragedProps:\n   {self.cart_axes}\n   {self.nm_indices}\n'


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
        vibstates_diffs_collection = []
        for re in self.resonances_expr:
            l = re[0].split(',')
            ftuple = []
            for ll in l:
                if 'zero' not in ll.split('+'):
                    ftuple.append(len(set(ll.split('+'))))
                else:
                    ftuple.append(0)
            vibstates_diffs_collection.append(tuple(ftuple))
        for vd in self.viblevelsdiff_expr:
            l = vd.split(',')
            ftuple = []
            for ll in l:
                if 'zero' not in ll.split('+'):
                    ftuple.append(len(set(ll.split('+'))))
                else:
                    ftuple.append(0)

            vibstates_diffs_collection.append(tuple(ftuple))
        vibstates_diffs_collection = set(vibstates_diffs_collection)
        self.vibstatesdiff_objs = [VibStatesDiff(i) for i in vibstates_diffs_collection]

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


    def __repr__(self):
        return f'{self.term_label} - {self.term_id}'


    def for_ab(self, a,b):
        """

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
        self.allstates[('zero',)] = 0.

        self.allstates_Eh = {k: convNu2Ene(v) for k, v in self.allstates.items()}
        self.allstates_Eh[("zero",)] = 0.0

        self.harmonic_states = harmonic_states
        self.harmonic_states[('zero',)] = 0.

        self.harmonic_states_Eh = {k: convNu2Ene(v) for k, v in self.harmonic_states.items() if len(k)==1}
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
            if (a,b,c) not in self.expression['CFF'][1]:
                idx = [dict_id[i] for i in self.expression['CFF'][1]]
                self.F_vals[(a,b,c)] = self.properties_data['F_abc'][*idx]
            propdict['F_abc'] = self.F_vals[(a,b,c)]

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
            total


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


    def fill_in_tensors(self, method, threshold=1e-18, use_threshold=False):
        """
        filling in 2d or 3d tensor with given method

        get_avrgprops_tensor(self, threshold=1e-18) - self.get_avrg_properties(a, b, c)[0]
        get_full_factor_tensor(self) - self.get_full_factor(a,b)[0]
        get_enefactor_tensor(self) - self.get_ene_factor(harm_modes_dict
        """
        t2 = np.zeros((max(self.mode_indices)+1, max(self.mode_indices)+1))
        t3 = np.zeros((max(self.mode_indices)+1, max(self.mode_indices)+1, max(self.mode_indices)+1))

        for a in self.mode_indices:
            for b in self.mode_indices:
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

    def get_term_tree(self):

        components = {}
        for a in self.mode_indices:
            for b in self.mode_indices:
                if self.term_label == 'EL':
                    # components[((a,b), (self.get_resonance_location(modes_dict, a, b)))] = self.get_full_factor(gammaCompsAll, properties_data,
                    #                                          modes_dict, a,b)
                    components[(a, b)] = self.get_full_factor(a, b)
                elif self.term_label == 'MECH':
                    # components[((a, b), (self.get_resonance_location(modes_dict, a, b)))] = self.get_factor_summed(gammaCompsAll, properties_data,
                    #                                             modes_dict, mode_indices, a, b)
                    components[(a, b)] = self.get_factor_summed(a, b)
        return components


    def get_all_resonances(self, w2mw1=False):
        res = {}
        for a in self.mode_indices:
            for b in self.mode_indices:
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

        """
        result = 0.

        skipped = 0

        # self.layers = {}

        for a in self.mode_indices:
            for b in self.mode_indices:

                if sel_abs is not None:
                    if (a,b) not in sel_abs:
                        skipped+=1
                        print('skipped', (a, b))
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
                        print('skipped later', (a,b))
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
            result = (product_all*self.get_res_factor(w1, w2, a, b, Gamma_rc, condition))
            return result, components

        else:
            product_all, components = self.get_factor_summed(a, b, comps=True) # , components if comps==True
            product_all /= -48.

            shortcomponents = {}
            # for k,v in components.items():
            #     if v[0]!=0.:
            #         shortcomponents[k] = {'full_product_abc':v[0], 'ene_factor':v[1]['ene_factor'],
            #                               'avrg_properties':v[1]['avrg_properties'][0],
            #                               'F_abc':v[1]['F_abc'],
            #                               'viblevelsdiff':v[1]['viblevelsdiff']}
            result = (product_all*self.get_res_factor(w1, w2, a, b, Gamma_rc, condition))

            return result, shortcomponents

    def get_vibdiff_tensor(self):

        viblevelsdiff_tensor = np.zeros((max(self.mode_indices)+1, max(self.mode_indices)+1, max(self.mode_indices)+1))
        viblevelsdiff_tensor2 = np.zeros((max(self.mode_indices)+1, max(self.mode_indices)+1, max(self.mode_indices)+1, 2))
        for a in self.mode_indices:
            for b in self.mode_indices:
                for c in self.mode_indices:
                    viblevelsdiff_tensor[a, b, c] = self.get_viblevelsdiff(a, b, c)[0]
                    viblevelsdiff_tensor2[a, b, c, :] = self.get_viblevelsdiff(a, b, c)[1]

        return viblevelsdiff_tensor, viblevelsdiff_tensor2




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





@dataclass
class SpectrumExperimentSetup:
    Gamma_rc: float
    e_selected: list
    m_selected: list
    fundamentals_harmonic: dict
    fundamentals: dict
    all_states: dict
    mode_indices: list | np.ndarray
    vib_levels_harmonic: bool
    maxYX: float


class TermStorage:
    """Stores and manages multiple Term2D objects."""
    def __init__(self):
        self.terms = {}
        self.terms_amplitudes = {}  # Dictionary: term_id -> Term2D object

    # def add_term(self, term: Term2D):
    #     """Adds a Term2D object to storage."""
    #     if term.term_id in self.terms:
    #         print(f"Warning: Overwriting existing term {term.term_id}")
    #     self.terms[term.term_id] = term
    #
    # def calculate_all(self, settings):
    #     """Runs calculations for all stored terms."""
    #     for k, term in self.terms.items():
    #         self.terms_amplitudes[k] = term.get_intensity(settings.w1, settings.w2,
    #                                                       settings.properties_data,
    #                                                       settings.modes_dict,
    #                                                       settings.mode_indices,
    #                                                       settings.Gamma_rc,
    #                                                       settings.margin,
    #                                                       condition=settings.condition)
    #
    # def get_amplitude(self, term_id):
    #     """Retrieves the calculated result for a specific term."""
    #     return self.terms_amplitudes[term_id] if term_id in self.terms else None
    #
    # def filter_terms(self, condition_fn):
    #     """Filters terms based on a given function."""
    #     return {tid: term for tid, term in self.terms.items() if condition_fn(term)}


    def __repr__(self):
        return f"TermStorage({len(self.terms)} terms)"


class TermsEvaluator:
    """
    Takes a list(?)/collection of Term2D(?) objs and performs evaluation of amplitudes

    """
    def __init__(self, terms: list[Term2D]):
        """
        Need to have precalculated:
            resonances locations and their ab combinations - to filter some out according to spectral window

            avrg_tensors_dict[termID][a, b]
            prefac_2d[a, b]
            comb_fac_dict[self.allterms_str[termID]][a, b] - mech terms

        Computations:
            resonance types for given ab combination - to be reused in terms (types so far: a+b,a; b,a) -- in place but saved
            amplitudes += factor * resonance product

        """
        # self.terms = terms
        self.terms = {t.term_id: t for t in terms}


    def identify_to_precalculate(self):
        """
        go through terms; identify parts for precalculation

        only abc dependent:
            orientational averages - (1)
            vibene_denoms - (2)
            resonances pfs - (3)
            resonances w_mn - (4)

            composite, so later - ab_factors - summed over c; separate from indices in res.conds.


        maybe do all separately?? in case of selective precalculation or none of it

        """
        # (1) IDENTIFYING unique resonance conditions
        unique_res_conds = []
        for t in self.terms.values():
            for i in t.resonances_expr:
                unique_res_conds.append(i)
        self.unique_res_conds = list(set(unique_res_conds))

        # (2.1) IDENTIFYING unique avrg tensors - looking for unique sets on normal mode indices
        # number of these indices = number of dimensions of avrg tensor
        # now: collect relevant part
        self.seq_tuples = DoubleDict()
        for t in self.terms.values():
            # self.seq_tuples.add(t.property_simple_tuples, t.term_id)
            self.seq_tuples.add(k=t.property_simple_tuples, v=t)

        self.unique_avrg_tensors_props = set(self.seq_tuples.kv.keys())
        self.unique_avrg_tensors_tID = [self.seq_tuples.kv[k].term_id for k in self.unique_avrg_tensors_props]
        # print('kv', self.seq_tuples.kv)
        # print('vk', self.seq_tuples.vk)

        # (2.2) IDENTIFYING unique normal mode indices for avrg tensors, to know dimensionality
        # self.uni_nm_idx = []
        self.unique_avrg_tensors_all = {}
        for t in self.terms.values():
            nms = []
            if t.property_simple_tuples not in self.unique_avrg_tensors_all:
                self.unique_avrg_tensors_all[t.property_simple_tuples]=[]
            for p in t.properties:
                for s in p.nm_indices:
                    nms.append(s)
            self.unique_avrg_tensors_all[t.property_simple_tuples].append(len(set(nms)))
        for k in self.unique_avrg_tensors_all:
            self.unique_avrg_tensors_all[k] = max(self.unique_avrg_tensors_all[k])
        # print('self.unique_avrg_tensors_all', self.unique_avrg_tensors_all)

        self.props_cart_ax = [[p.cart_axes for p in t.properties] for t in self.terms.values()]
        # print('self.props_cart_ax', self.props_cart_ax)

        # (3) IDENTIFYING 1/omega_a/omega_b, 1/omega_a/omega_b/omega_c terms - to make nD tensors of products
        # in precalc_vibene_denoms()
        self.unique_vibene_denoms = set([t.expression['vibene_denom'] for t in self.terms.values()])

        # self.uProps = [AveragedProps(t.properties) for t in self.terms]

        # (4) IDENTIFYING vib states diffs types
        self.mn_types = [i for t in self.terms.values() for i in t.vibstatesdiff_objs]


    def precalc_vibene_denoms(self, freqs):
        """
        requires:
            freqs data; self.unique_vibene_denoms
        """
        inv_freqs = 1 / freqs
        stored = {}

        for nm_idxs in self.unique_vibene_denoms:
            stored[nm_idxs] = outer_product_einsum(inv_freqs, len(nm_idxs))

        return stored

    def precalc_avrg_tensors(self, Nnmodes, data, avrg_terms):
        """
        ((1, 1), (2, 1), (1, 2)) - avrg tensor coding - mu_Q, alpha_Q, mu_QQ
        ((1, 1), (2, 2), (1, 1)) - mu_Q, alpha_QQ, mu_Q
        ((1, 1), (2, 1), (1, 1)) - mu_Q, alpha_Q, mu_Q

        requires:
            self.unique_avrg_tensors_tID; self.seq_tuples; self.unique_avrg_tensors_all;
            self.terms[tID] so it's a dict;
        """
        terms_for_avrg_tensors = self.unique_avrg_tensors_tID
        storage_tensors = {}

        for tID in terms_for_avrg_tensors:
            simple_prop_tuple = self.seq_tuples.vk[self.terms[tID]]
            nm_indices = self.terms[tID].nice_props.nm_indices
            cart_indices = self.terms[tID].nice_props.cart_axes
            # print('nm_indices', nm_indices)
            # print('cart_indices', cart_indices)

            num_dims = self.unique_avrg_tensors_all[simple_prop_tuple]
            shape = (Nnmodes,) * num_dims

            avrg_tensor = np.zeros(shape)
            abcde_combs = combinations_with_permutations(range(Nnmodes), num_dims)
            for abcde_comb in abcde_combs:
                total = 0.

                names = list(string.ascii_lowercase)
                var_names = names[:len(abcde_comb)]
                variables = {var: val for var, val in zip(var_names, abcde_comb)}
                debugfunc(f'---variables {variables}', tag='')

                for comps in avrg_terms:
                    alpha, beta, gamma, delta = comps
                    greek_dict = {'A': alpha, 'B': beta, 'G': gamma, 'D': delta}
                    debugfunc(f'alpha, beta, gamma, delta {alpha, beta, gamma, delta}',
                              tag='')
                    product = 1.
                    for i, pp in enumerate(simple_prop_tuple):
                        input_tuple = (pp, cart_indices[i], nm_indices[i])
                        prop_key, idxs_key = get_data_keys(input_tuple, variables, greek_dict)
                        debug_deep(f'idxs_key {idxs_key}', tag='')

                        product *= data[prop_key][idxs_key]
                        debugfunc(f'prop_key {prop_key}, idxs_key {idxs_key}, value {data[prop_key][idxs_key]}',
                                  tag='')
                    total += product
                avrg_tensor[abcde_comb] = total / 15.

            storage_tensors[tID] = avrg_tensor
        return storage_tensors


    def precalc_vibdiffs(self, states, Nnmodes):
        """
        states - dict of state_idx_label_tuple(?) : frequency (cm-1)


        types of vib diffs: (0, 1), (1, 0), (2, 1), (1, 2), (2, 0), (0, 2)...

        """
        res = {}
        dims = [sum(list(t.diff_type)) for t in set(self.mn_types)]
        print('set(dims)', set(dims))

        for d in set(dims):
            shape = (Nnmodes,) * d
            res[d] = np.zeros(shape)

        # ApBmA[a, b] = ApB[a, b] - A[b] = A[a] + B[b] - A[b]

        """
        >>> B-A
        array([[0.2, 0.4, 0.6],
               [3.8, 4. , 4.2],
               [7.4, 7.6, 7.8]])
        >>> B
        array([[ 1.2,  2.4,  3.6],
               [ 4.8,  6. ,  7.2],
               [ 8.4,  9.6, 10.8]])
        >>> A
        array([1., 2., 3.])

        >>> B[0,1]-A[1] == (B-A)[0,1]
        np.True_
        >>> B[1,0]-A[0] == (B-A)[1,0] - B contains (a+b) states B[a,b]  
        np.True_
        """

        return res




    def precalc_res_conds(self, axes_dict, freqs):
        """
        requires:
            self.unique_res_conds
        """
        result_pfs = {}
        result_mns = {}

        unique_pert_freq_arrangements = set([i[1] for i in self.unique_res_conds])
        unique_w_m7n = set([i[0] for i in self.unique_res_conds])
        implicit_minus_one = -1

        # set up terms to add together
        uq_pert_freq_arrays = [implicit_minus_one*np.array(i) for i in unique_pert_freq_arrangements]

        # all_axes = set([abs(i) for j in unique_pert_freq_arrangements for i in j]) # should help to set up axes_dict
        # print('all_axes', all_axes)
        # axes_dict = {k:None for k in all_axes} # todo: how to somewhat automatically fill in this dict? input for now

        print('uq_pert_freq_arrays', uq_pert_freq_arrays)

        # collect types of pert freqs arrangements
        for pfs in uq_pert_freq_arrays:
            pfs = [int(i) for i in pfs]
            result_pfs[tuple(pfs)] = 0.

            for pf in pfs:
                result_pfs[tuple(pfs)] += axes_dict[abs(pf)] * np.sign(pf)

        states_mn = []
        for mn in unique_w_m7n:
            m,n = mn.split(',')
            states_mn.append(m)
            states_mn.append(n)

            # result_mns[mn] = 0.
        print('states_mn', set(states_mn))

        return result_pfs#, result_mns


    def precalculate(self, alldata):
        """
        requires identified parts for precalculation and external data

        """
        freqs, Nnmodes, data, avrg_terms = alldata # todo: set this up better

        self.precalc_vibene_denoms(freqs)
        self.precalc_avrg_tensors(Nnmodes, data, avrg_terms)
        # self.precalc_res_conds()



def get_data_keys(input_tuple, variables, greek_dict):
    """
    tuple_input = ((1, 1), ('B',), ('a',))

    input_tuples = [
        ((1, 1), ('B',), ('a',)),
        ((2, 1), ('A', 'D'), ('a',)),
        ((2, 2), ('A', 'D'), ('a', 'b'))
    ]
    """
    prop_der_key, second_part, third_part = input_tuple

    third_part = tuple([variables[v] for v in third_part])
    second_part = tuple([greek_dict[l] for l in second_part])
    # Combine third_part and second_part to make the second-level index
    idxs_key = tuple(third_part) + tuple(second_part)

    return prop_der_key, idxs_key


def outer_product_einsum(arr, n):
    # for n=3 -> 'i,j,k->ijk'
    indices = ','.join([chr(ord('i') + j) for j in range(n)]) + '->' + ''.join([chr(ord('i') + j) for j in range(n)])
    arrays = [arr] * n

    return np.einsum(indices, *arrays)


def calc_vibene_diff_mn(vibene_data, m, n, symbolic=False):
    """
    w_{m,n}^{whatever}

    if symbolic:
        precalculates an nDarray for keeping all possible indices values
    else:
        vibene_data is a dict or smth, containing keys or attributes m and n
    """

    if symbolic:
        return None

    else:
        return vibene_data[m] - vibene_data[n]


def get_resonance_location(resonances_expr, modes_dict, a, b):
    """
    A resonance for this term for ab combination of modes
    """
    a, b = str(a), str(b)
    modes_dict[('zero',)] = 0.
    dict_id = {'a': a, 'b': b, 'zero': 'zero'}
    type12, type1 = resonances_expr
    m1_str, n1_str = type1.split(',')

    m1_tuple = tuple([str(dict_id[i]) for i in m1_str.split('+')])
    n1_tuple = tuple([str(dict_id[i]) for i in n1_str.split('+')])
    w1 = modes_dict[n1_tuple] - modes_dict[m1_tuple]

    m12_str, n12_str = type12.split(',')
    m12_tuple = tuple(sorted([str(dict_id[i]) for i in m12_str.split('+')], key=int))
    n12_tuple = tuple(sorted([str(dict_id[i]) for i in n12_str.split('+')], key=int))
    w2 = modes_dict[m12_tuple] - modes_dict[n12_tuple] + w1

    return w1, w2


def get_all_resonances(resonances_expr, modes_dict, mode_indices, w2mw1=False):
    res = {}
    for a in mode_indices:
        for b in mode_indices:
            w1, w2 = get_resonance_location(resonances_expr, modes_dict, a, b)

            if w2mw1:
                res[(a,b)] = (w1, w2-w1)
            else:
                res[(a,b)] = (w1, w2)
    return res