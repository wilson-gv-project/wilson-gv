import string

import numpy as np
from .tools import convNu2Ene, combinations_with_permutations
from .spectrum_utils import DoubleDict
from wilson.utils import pairwise_differences, coolprint
from wilson.debug import debugfunc, debug_deep
from wilson.spectrum.termND import TermND


class TermsEvaluator:
    """
    Takes a list(?)/collection of Term2D(?) objs and performs evaluation of amplitudes

    """
    def __init__(self, terms: list[TermND]):
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
        self.terms = {t.term_id: t for t in terms}

    def __repr__(self):
        return f'\nTHIS IS TermsEvaluator with {len(self.terms)} terms\n'


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

            {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
                          'vibenediff': None,
                          'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_QQ', ('a', 'b',), ('G',))),
                          'non_averaged_props': None,
                          'vibene_denom': ('a','b',),
                          'termB_pref': 1.,
                          'termA_pref': 1/24}
        """
        # IDENTIFY INDICES
        collect_n_idx_rescond = []
        collect_n_idx_max = []
        for tid, t in self.terms.items():
            collect_n_idx_max.append(t.collective_n_idx_max)
            collect_n_idx_rescond.append(t.collective_n_idx_rescond)
        # print('collect_n_idx_rescond', collect_n_idx_rescond)
        for tid, t in self.terms.items():
            t.collective_n_idx_max = max(collect_n_idx_max)
            t.collective_n_idx_rescond = max(collect_n_idx_rescond)
            # print(tid, t.collective_n_idx_max, t.collective_n_idx_rescond)
            # print('self.collective_n_idx_rescond', t.collective_n_idx_rescond)
            t.collective_idx_counted = True

        # (1) IDENTIFYING unique resonance conditions
        unique_res_conds = []
        for t in self.terms.values():
            for i in t.resonances_expr:
                unique_res_conds.append(i)
        self.unique_res_conds = list(set(unique_res_conds))
        # self.unique_res_conds = [('b,a', (-1, 2)), ('a+b,a', (-1, 2)), ('zero,a', (-1,))]
        idx_res_conds_inter1 = [i[0].split(',') for i in self.unique_res_conds]
        idx_res_conds_inter2 = [j.split('+') for i in idx_res_conds_inter1 for j in i]
        idx_res_conds = set([j for i in idx_res_conds_inter2 for j in i if j!='zero'])

        # counting indices in resonance conditions
        # for t in self.terms.values():
        #     t.collective_n_idx_rescond = len(idx_res_conds)


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

        # FIXME: NOT IN USE
        # self.props_cart_ax = [[p.cart_axes for p in t.properties] for t in self.terms.values()]
        # print('self.props_cart_ax', self.props_cart_ax)

        # (3) IDENTIFYING 1/omega_a/omega_b, 1/omega_a/omega_b/omega_c terms - to make nD tensors of products
        # in precalc_vibene_denoms()
        self.unique_vibene_denoms = set([t.expression['vibene_denom'] for t in self.terms.values()])

        # self.uProps = [AveragedProps(t.properties) for t in self.terms]

        # (4) IDENTIFYING vib states diffs types - all
        self.mn_types = [i for t in self.terms.values() for i in t.vibstatesdiff_objs]

        # (5) IDENTIFYING numbers of quanta in state
        # FIXME: NOT IN USE
        # self.quanta_numbers = set([i for v in set(self.mn_types) for i in v.diff_type if i > 0])

        coolprint('To precalculate some quantities, you need to provide some data:\n')
        coolprint(r'For [dodger_blue2]vibene_denoms[/dodger_blue2]: [deep_pink3]qstates_harm dict\[q]')
        coolprint(r'For [dodger_blue2]avrg_tensors[/dodger_blue2]: Nnmodes int, data data\[prop_key]\[idxs_key], avrg_terms')
        coolprint('For [dodger_blue2]precalc_res_conds[/dodger_blue2]: [dark_goldenrod]axes_dict {1: x_mesh,..}[/dark_goldenrod]')
        coolprint(r'For [dodger_blue2]precalc_vibdiffs[/dodger_blue2]: [deep_pink3]qstates_choice dict\[q]')
        coolprint('\nOnly [dark_goldenrod]axes_dict[/dark_goldenrod] relates to spectrum pixels.')
        coolprint('And [medium_purple1]qstates_choice, qstates_harm, Nnmodes[/medium_purple1] are related to the states.')
        coolprint('And [medium_purple1]data, avrg_terms[/medium_purple1] are related to molecular properties.')


    def precalc_vibene_denoms(self, qstates_Eh):
        """
        requires:
            freqs data; self.unique_vibene_denoms
        """
        freqs = qstates_Eh[1]
        stored = {}
        for nm_idxs in self.unique_vibene_denoms:
            stored[nm_idxs] = outer_product_einsum(freqs, len(nm_idxs))
        # print('\nPRECALC VIB ENE DENOMS')
        # print(freqs)
        # print('self.unique_vibene_denoms')
        # print(self.unique_vibene_denoms)
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

            num_dims = self.unique_avrg_tensors_all[simple_prop_tuple]
            shape = (Nnmodes,) * num_dims

            avrg_tensor = np.zeros(shape)
            abcde_combs = combinations_with_permutations(range(Nnmodes), num_dims)
            # print('abcde_combs: ', list(abcde_combs))

            for abcde_comb in abcde_combs:
                total = 0.

                names = list(string.ascii_lowercase)
                var_names = names[:len(abcde_comb)]
                variables = {var: val for var, val in zip(var_names, abcde_comb)}
                debug_deep(f'---variables {variables}', tag='')

                # print('\nvar_names', var_names)
                # print('variables', variables)

                for comps in avrg_terms:
                    # todo: generalize greek indices
                    alpha, beta, gamma, delta = comps
                    greek_dict = {'A': alpha, 'B': beta, 'G': gamma, 'D': delta}
                    debug_deep(f'alpha, beta, gamma, delta {alpha, beta, gamma, delta}',
                              tag='')
                    product = 1.
                    for i, pp in enumerate(simple_prop_tuple):
                        input_tuple = (pp, cart_indices[i], nm_indices[i])
                        prop_key, idxs_key = get_data_keys(input_tuple, variables, greek_dict)
                        # if prop_key == (2,1):
                        #     print('input_tuple', input_tuple)
                        #     print('\nprop_key', prop_key)
                        #     print('idxs_key', idxs_key)
                        debug_deep(f'idxs_key {idxs_key}', tag='')

                        product *= data[prop_key][idxs_key]
                        debug_deep(f'prop_key {prop_key}, idxs_key {idxs_key}, value {data[prop_key][idxs_key]}',
                                  tag='')
                        # print('product =', product)
                    total += product
                # print('total: ', total)
                if abs(total)<1e-28:
                    total = 0.
                else:
                    total /= 15.
                avrg_tensor[abcde_comb] = total


            storage_tensors[simple_prop_tuple] = avrg_tensor
        # print('storage_tensors: \n', storage_tensors)
        return storage_tensors


    def precalc_vibdiffs(self, qstates_Eh):
        """
        states - dict of state_idx_label_tuple(?) : frequency (Eh)

        types of vib diffs: (0, 1), (1, 0), (2, 1), (1, 2), (2, 0), (0, 2)...

        states = {0: 0.,
                  1: np.zeros(Nnmodes),
                  2: np.zeros((Nnmodes, Nnmodes)),
                  3: np.zeros((Nnmodes, Nnmodes, Nnmodes),}
        """
        res = {}
        for d in self.mn_types:
            sort_d = sorted(d.diff_type)

            diff = pairwise_differences(qstates_Eh[sort_d[0]], qstates_Eh[sort_d[1]])
            res[tuple(sort_d)] = diff
            
        # ApBmA[a, b] = ApB[a, b] - A[b] = A[a] + B[b] - A[b]
        # from 2d array subtract 1d array => from each row subtract this 1d array
        # then it means: quant2[0,1] is quant2[a,b] - quant1[a] = diff2_1[b,a]

        # state1, state2, state2-state1
        # harmonic state is given by index of NM
        # complex state has a composition, and will have a new label

        return res




    def precalc_res_conds(self, axes_dict):
        """
        requires:
            self.unique_res_conds

        axes_dict - ??? {1: , 2: , 3: ....} pf labels: points array/meshgrid
        """
        result_pfs = {}
        # result_mns = {}

        unique_pert_freq_arrangements = set([i[1] for i in self.unique_res_conds])
        unique_w_m7n = set([i[0] for i in self.unique_res_conds])
        implicit_minus_one = -1

        # set up terms to add together
        uq_pert_freq_arrays = [implicit_minus_one * np.array(i) for i in unique_pert_freq_arrangements]

        # all_axes = set([abs(i) for j in unique_pert_freq_arrangements for i in j]) # should help to set up axes_dict
        # print('all_axes', all_axes)
        # axes_dict = {k:None for k in all_axes} # todo: how to somewhat automatically fill in this dict? input for now

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

        return result_pfs


    def precalculate(self, alldata):
        """
        requires identified parts for precalculation and external data

        alldata is DataForPrecalc
        Nnmodes, data, avrg_terms, axes_dict, qstates = alldata
        """

        Nnmodes = alldata.Nnmodes
        props_data = alldata.props_data
        avrg_terms = alldata.avrg_terms
        axes_dict = alldata.axes_dict
        qstates_Eh = alldata.states_arrays_Eh
        qstates_harm_Eh = alldata.harmonic_arrays_Eh

        a = self.precalc_vibene_denoms(qstates_harm_Eh) # what are freqs?
        b = self.precalc_avrg_tensors(Nnmodes, props_data, avrg_terms)
        c = self.precalc_res_conds(axes_dict) # fixme: not used now in the calculations
        d = self.precalc_vibdiffs(qstates_Eh)
        dictionary = {'vibene_denoms': a,
                      'avrg_tensors': b,
                      'res_conds': c,
                      'vibdiffs': d}

        return dictionary


    def compute_intensity(self, w1, w2, Gamma_rc=3.8,margin=0.):

        tot = 0.
        for tID in self.terms:
            tot += self.terms[tID].get_amplitudes(w1, w2, Gamma_rc, margin)


def get_data_keys(input_tuple, variables, greek_dict):
    """
    tuple_input = ((1, 1), ('B',), ('a',))

    input_tuples = [
        ((1, 1), ('B',), ('a',)),
        ((2, 1), ('A', 'D'), ('a',)),
        ((2, 2), ('A', 'D'), ('a', 'b'))
    ]
    in term:
    (('mu_Q', ('a',), ('B',)),
     ('alpha_Q', ('b',), ('A', 'D')),
     ('mu_Q', ('c',), ('G',)))
    """
    prop_der_key, second_part, third_part = input_tuple
    # print('\nget_data_keys()')
    # print(prop_der_key, second_part, third_part)

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
    for ab in combinations_with_permutations(mode_indices, 2):
        a, b = ab
        w1, w2 = get_resonance_location(resonances_expr, modes_dict, a, b)

        if w2mw1:
            res[(a,b)] = (w1, w2-w1)
        else:
            res[(a,b)] = (w1, w2)
    return res