import time
from collections.abc import Iterable
from typing import Callable

import numpy as np

from spectrum2d import get_iso_f


from parsing.parseCFOUR_forWilson import CFOURdataParser
from parsing.parseGaussian_forWilson import GaussianDataParser
from parsing.parser_template import MockParser

def convWnum2Freq(reciprocal_cm: float | np.ndarray) -> float | np.ndarray:
    """Convert wavenumber in cm-1 to frequency (1/s)"""
    from scipy import constants
    hartree2J = constants.physical_constants['hartree-joule relationship'][0]
    return reciprocal_cm * (100 * constants.h * constants.c / hartree2J)

class SpectrumEVV:
    """
    SpectrumEVV class
    Attributes:
        w1, w2 - np.arrays of frequencies
        w1_mesh, w2_mesh - grid of frequencies w1 and w2
        shape2d - shape of the grid
    """
    def __init__(self, w1: np.array, w2: np.array,
                 input_data_info: dict, vib_levels_harmonic: bool = True):

        if type(w1)==list or type(w2)==list:
            w1, w2 = np.array(w1), np.array(w2)

        # Define the grid of spectrum (pixels)
        self.w1_mesh, self.w2_mesh = np.meshgrid(w1, w2, indexing='ij')
        self.shape2d = self.w1_mesh.shape
        self.input_data_info = input_data_info
        self.Gamma = None

        # fixme not needed
        self.id = f'w1{min(w1)}_{max(w1)}w2{min(w2)}_{max(w2)}'

        self.gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)

        # margin for higher diagonal, to not show/compute data to close to the diagonal
        self.diagonal_margin = 10.

        self.vib_levels_harmonic = vib_levels_harmonic
        print(f'\nUsed vibrational energy levels are harmonic? - {self.vib_levels_harmonic}')

        self.saved_mech = {}
        self.saved_el = {}

    def load_data(self, parser):
        """Loading the data from a parser object/DataVault
            with the sources given to it"""
        # fixme - make it more flexible, give an option to
        # if self.input_data_info['source'] == 'cfour':
        #     parserObj = CFOURdataParser(self.input_data_info)
        # elif self.input_data_info['source'] == 'gaussian':
        #     parserObj = GaussianDataParser(self.input_data_info)
        # else:
        #     print('datasource not implemented')
        #     parserObj = MockParser({})
        parserObj = parser(self.input_data_info)
        parserObj.getData()

        self.fundamentals = parserObj.fundamentals_anharmonic_str
        self.fundamentals_harmonic = parserObj.fundamentals_harmonic_str
        self.all_states = parserObj.anharmonic_states
        self.all_states_harmonic = parserObj.harmonic_states

        ddata = [parserObj.dipole_first_derivatives,
                 parserObj.dipole_second_derivatives,
                 parserObj.polarizability_first_derivatives,
                 parserObj.polarizability_second_derivatives,
                 parserObj.cubic_force_constants]
        self.deriv_data = dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], ddata))


    def conversion2InternalUnits(self):
        """
        rs - reciprocal second
        """
        self.all_states_harmonic_rs = {k: convWnum2Freq(v) for k, v in self.all_states_harmonic.items()}
        self.all_states_rs = {k: convWnum2Freq(v) for k, v in self.all_states.items()}
        self.w1_rs_mesh, self.w2_rs_mesh = convWnum2Freq(self.w1_mesh), convWnum2Freq(self.w2_mesh)
        self.diagonal_margin_rs = convWnum2Freq(self.diagonal_margin)


    def addTerms(self, electrical_terms_selection: list, mechanical_terms_selection: list):
        """Creating functions for computing the expressions for mechanical and electrical anharmonicities"""

        self.conversion2InternalUnits() # need now at least for diag margin_rs

        # fixme - common precalculated terms

        # Terms in expressions
        electrical_terms_str = [(('a+b,a', 'zero,a'), None),
                                (('b,a', 'zero,a'), None)]

        # electrical_terms_str = [('a+b,a', 'zero,a'),
        #                         ('b,a', 'zero,a')]

        mechanical_terms_str = [(('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')),
                              (('c,a', 'zero,a'), ('a+b,c', 'b+c,a')),
                              (('a+b,a', 'zero,a'), ('a,a+b', 'b,zero')),
                              (('b,a', 'zero,a'), ('b,a+b', 'a,zero')),
                              (('b,a', 'zero,a'), ('a,a+b', 'b,zero')),
                              (('b,a', 'zero,a'), ('b,a+b', 'a,zero'))]

        # derivatives:
        # 1. mu_Q, mu QQ, alpha_Q - electric dipole (1st and 2nd derivatives), polarizability (1st der.)
        # 2. mu_Q, alpha_QQ - electric dipole (1st der.), polarizability (2nd der.)
        electric_avrg_str = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))],
                           [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))] ]

        # mu_Q, alpha_Q - for all 6 terms
        mechanical_avrg_str = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                             [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                             [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc'],
                             [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc'],
                             [('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc'],
                             [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc']]

        factors = [1., 1., 0.5, 0.5, -0.5, -0.5]

        # fixme: unify the format for the terms tuples
        # fixme : margin input
        self.e_selection, self.m_selection = electrical_terms_selection, mechanical_terms_selection
        self.mech_factors = [factors[i] for i in mechanical_terms_selection]

        self.electrical_terms = [electrical_terms_str[i] for i in self.e_selection]
        self.mechanical_terms = [mechanical_terms_str[i] for i in self.m_selection]

        self.electric_avrg = [electric_avrg_str[i] for i in self.e_selection]
        self.mechanical_avrg = [mechanical_avrg_str[i] for i in self.m_selection]
        # here the functions of 2 frequencies
        self.e_funcs = [generate_resonances_functions(i[0], i[1], margin_rs=self.diagonal_margin_rs) for i in self.electrical_terms]
        self.m_funcs = [generate_resonances_functions(i[0], i[1], margin_rs=self.diagonal_margin_rs) for i in self.mechanical_terms]

        # self.e_funcs = [generate_resonances_functions(i, margin=self.diagonal_margin) for i in self.electrical_terms]
        # self.m_funcs = [generate_resonances_functions(*i) for i in self.mechanical_terms]

        nmodes = len(self.fundamentals)
        self.combofuns = [dict(zip(self.e_funcs, self.electric_avrg)),
                          dict(zip(self.m_funcs, self.mechanical_avrg))]

        # setting up the combinations of states for the terms
        self.coords_ab = get_abc_indices(2, len(self.fundamentals)) if self.electrical_terms is not None else []
        self.coords_abc = get_abc_indices(3, len(self.fundamentals)) if self.mechanical_terms is not None else []

        if self.electrical_terms is not None:
            self.el_avrg_tensors = [avrg_abc_tensor(ea, self.deriv_data, self.gammaCompsAll) for ea in self.electric_avrg]
        else:
            self.el_avrg_tensors = []

        if self.mechanical_terms is not None:
            self.mech_avrg_tensors = [avrg_abc_tensor(ma, self.deriv_data, self.gammaCompsAll) for ma in self.mechanical_avrg]
        else:
            self.mech_avrg_tensors = []

        self.combofuns_tensors = [dict(zip(self.e_funcs, self.el_avrg_tensors)),
                                  dict(zip(self.m_funcs, self.mech_avrg_tensors))]

        w_ab = np.zeros((nmodes, nmodes))
        w_abc = np.zeros((nmodes, nmodes, nmodes))
        for state in self.all_states:
            if len(state) == 2:
                w_ab[int(state[0]), int(state[1])] = self.all_states[state]
                w_ab[int(state[1]), int(state[0])] = self.all_states[state]
            elif len(state) == 3:
                w_abc[int(state[0]), int(state[1]), int(state[2])] = self.all_states[state]
                w_abc[int(state[0]), int(state[2]), int(state[1])] = self.all_states[state]
                w_abc[int(state[1]), int(state[0]), int(state[2])] = self.all_states[state]
                w_abc[int(state[1]), int(state[2]), int(state[0])] = self.all_states[state]
                w_abc[int(state[2]), int(state[0]), int(state[1])] = self.all_states[state]
                w_abc[int(state[2]), int(state[1]), int(state[0])] = self.all_states[state]

        self.w_abc = convWnum2Freq(w_abc)
        self.w_ab = convWnum2Freq(w_ab) # for omega_{a+b} frequencies
        w = convWnum2Freq(np.array([v for k,v in self.fundamentals.items()]))
        vib_ene_levels_harmonic = convWnum2Freq(np.array([v for k,v in self.fundamentals_harmonic.items()]))

        self.prefac_2d = np.outer(vib_ene_levels_harmonic, vib_ene_levels_harmonic)
        self.prefac_3d = (vib_ene_levels_harmonic[:, np.newaxis, np.newaxis] *
                          vib_ene_levels_harmonic[np.newaxis, :, np.newaxis] *
                          vib_ene_levels_harmonic[np.newaxis, np.newaxis, :])

    # def gamma_mn(self, Gamma: float, a: int, b: int, c: int = False) -> np.ndarray:
    #     if self.vib_levels_harmonic:
    #         vib_ene_levels = self.all_states_harmonic
    #     else:
    #         vib_ene_levels = self.all_states
    #
    #     # if 'c' is not provided, compute electrical anharmonicity
    #     if type(c) == bool:
    #         total_sum_el = 0
    #         prefac_el = self.prefac_2d.T[a, b]
    #         for el_func, elavrg in self.combofuns_tensors[0].items():
    #             resonance = el_func(vib_ene_levels, self.w1_mesh, self.w2_mesh,
    #                                 Gamma, (a, b))
    #             total_sum_el += elavrg[a, b] * resonance / prefac_el
    #         return total_sum_el / 24.
    #
    #     else:
    #         total_sum_mech = 0
    #         prefac_mech = self.prefac_3d.T[a, b, c]
    #         for index, (mech_func, mechavrg) in enumerate(self.combofuns_tensors[1].items()):
    #             mechavrgF = list(self.combofuns[1].items())[index][1]
    #             abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
    #             ijk_indx = tuple([abc[j] for j in mechavrgF[-1]])
    #             F = self.deriv_data['F_abc'][ijk_indx]
    #             resonance2 = mech_func(vib_ene_levels, self.w1_mesh, self.w2_mesh, Gamma, (a, b, c))
    #
    #             total_sum_mech += self.mech_factors[index] / prefac_mech * mechavrg[a, b, c] * F * resonance2
    #         return -total_sum_mech / 48.

    def setSpectrumSettings(self, Gamma_rc):
        """Settings to be set before computing the intensities"""
        self.Gamma = convWnum2Freq(Gamma_rc)

    def get_total_gamma_sum_el(self, a: int, b: int) -> np.ndarray:
        """
        """
        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic_rs
        else:
            vib_ene_levels = self.all_states_rs
        print('a, b', a, b)
        total_sum_el = 0
        prefac_el = self.prefac_2d.T[a, b]
        for index, (el_func, elavrg) in enumerate(self.combofuns_tensors[0].items()):
            resonance = el_func(vib_ene_levels, self.w1_rs_mesh, self.w2_rs_mesh,
                                self.Gamma, (a, b))
            print('elavrg[a, b]', elavrg[a, b], a, b)
            total_sum_el += elavrg[a, b] * resonance / prefac_el
        return total_sum_el / 24.

    def get_total_gamma_sum_mech(self, a: int, b: int, c: int) -> np.ndarray:
        """
        """
        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic
        else:
            vib_ene_levels = self.all_states

        total_sum_mech = 0
        prefac_mech = self.prefac_3d.T[a, b, c]
        for index, (mech_func, mechavrg) in enumerate(self.combofuns_tensors[1].items()):
            if index not in self.saved_mech:
                self.saved_mech[index] = {}
            mechavrgF = list(self.combofuns[1].items())[index][1]
            abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
            ijk_indx = tuple([abc[j] for j in mechavrgF[-1]])
            F = self.deriv_data['F_abc'][ijk_indx]
            resonance2 = mech_func(vib_ene_levels, self.w1_mesh, self.w2_mesh, self.Gamma, (a, b, c))

            addition = self.mech_factors[index] / prefac_mech * mechavrg[a, b, c] * F * resonance2
            is_equal = np.allclose(np.abs(addition),
                                   np.abs(np.full(addition.shape, -0. + 0.j, dtype=complex)))
            if not is_equal:
                self.saved_mech[index][tuple([a, b, c])] = (addition*(-1./48.), mechavrg[a, b, c], F, resonance2)
                total_sum_mech += addition
        return -total_sum_mech / 48.


    def intensity_electrical(self) -> (np.ndarray, dict):
        start_time = time.time()

        Qab_contrib_dict = {}

        elall = np.zeros(self.shape2d, dtype='complex128')
        for ind, i in enumerate(self.coords_ab):
            contrib_ab = self.get_total_gamma_sum_el(i[0], i[1])
            Qab_contrib_dict[tuple(i)] = contrib_ab
            elall += contrib_ab
            if ind % 100 == 0:
                print(f'{ind}/{len(self.coords_ab)} modes combinations -- {ind*100/len(self.coords_ab)}%')
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time -| electrical: {execution_time} seconds")
        print('Electrical anharmonicities are calculated')

        return elall, Qab_contrib_dict

    def intensity_mechanical(self) -> (np.ndarray, dict):
        """Gama_rc - broadening factor in reciprocal centimeters (cm-1)"""

        start_time = time.time()

        Qabc_contrib_dict = {}

        mechall = np.zeros(self.shape2d, dtype='complex128')
        for ind, i in enumerate(self.coords_abc):
            contrib_abc = self.get_total_gamma_sum_mech(i[0], i[1], i[2])
            Qabc_contrib_dict[tuple(i)] = contrib_abc
            mechall += contrib_abc
            if ind % 1000 == 0:
                print(f'{ind}/{len(self.coords_abc)} modes combinations -- {ind*100/len(self.coords_abc)}%')

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nExecution time - mechanical: {execution_time} seconds")
        print('Mechanical anharmonicities are calculated')

        return mechall, Qabc_contrib_dict


def get_abc_indices(number_ofIndices: int, number_ofFundamentals: int) -> Iterable:
    """
    modes a, b, (c) - combinations of them in pairs (electric anharmonicity) or triplets (mechanical anharmonicity)
    :param number_ofIndices:
    :param number_ofFundamentals:
    :return:
    """
    return np.indices([number_ofFundamentals]*number_ofIndices).reshape(number_ofIndices, -1).T

def get_AlphaBetaGammaDelta_indices(num_f: int) -> np.ndarray:
    """
    pol_g = orientationalaveraging.get_iso_f(num_f)
    pol_g is a list of lists of 2 lists where the second one is empty
          but first one contains the lists of interest

    :param num_f:
    :return: array_of_4greekIndices - an array of arrays of 4 greek indices for second hyperpolarizability :
             [alpha, beta, gamma, delta]
    """
    pol_g = get_iso_f(num_f)
    array_of_4greekIndices = np.array([pol[0] for pol in pol_g], dtype='object').reshape(-1, num_f)
    return array_of_4greekIndices

def avrg_abc_tensor(formula: list[tuple[str, tuple[str]]],
                    data: dict[str:np.ndarray], gammaCompsAll: np.array) -> np.ndarray:
    """
    Calculate the averaging tensor for a given formula.
    Indices of the tensor are normal coordinates (NC) indices,
    and the shape of the tensor depends on the nature of the term that is being calculated.
    Shape of the averaging tensor for electrical anharmonicity terms is (n_NC, n_NC)
    Shape of the averaging tensor for mechanical anharmonicity terms is (n_NC, n_NC, n_NC)

    :param formula:
    :param data:
    :param gammaCompsAll:
    :return:
    """
    nmodes = data['mu_Q'].shape[0]

    if type(formula[-1]) == str:
        # True for mechanical anharmonicity terms
        formula = formula[:-1]

    # specific case of the gamma_1,0 first term
    if [i[0] for i in formula] == ['mu_Q', 'alpha_Q', 'mu_QQ']:
        avrg_tensor = np.zeros((nmodes, nmodes))
        for a in range(nmodes):
            for b in range(nmodes):
                total = 0.
                for comps in gammaCompsAll:
                    alpha, beta, gamma, delta = comps
                    total += data['mu_Q'][a, beta] * data['alpha_Q'][b, alpha, delta] * data['mu_QQ'][a, b, gamma]
                avrg_tensor[a, b] = total/15.
        return avrg_tensor

    # specific case of the gamma_1,0 second term
    elif [i[0] for i in formula] == ['mu_Q', 'alpha_QQ', 'mu_Q']:
        avrg_tensor = np.zeros((nmodes, nmodes))
        for a in range(nmodes):
            for b in range(nmodes):
                total = 0.
                for comps in gammaCompsAll:
                    alpha, beta, gamma, delta = comps
                    total += data['mu_Q'][a, beta] * data['alpha_QQ'][a, b, alpha, delta] * data['mu_Q'][b, gamma]
                avrg_tensor[a, b] = total/15.
        return avrg_tensor

    # all terms of gamma_0,1 have this structure of averaging part
    elif [i[0] for i in formula] == ['mu_Q', 'alpha_Q', 'mu_Q']:
        avrg_tensor = np.zeros((nmodes, nmodes, nmodes))
        # this part is changing for different terms
        modes_letters = [i[1] for i in formula]
        for a in range(nmodes):
            for b in range(nmodes):
                for c in range(nmodes):
                    abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
                    i1, i2, i3 = [abc[j[0]] for j in modes_letters]
                    total = 0.
                    for comps in gammaCompsAll:
                        alpha, beta, gamma, delta = comps
                        total += data['mu_Q'][i1, beta] * data['alpha_Q'][i2, alpha, delta] * data['mu_Q'][i3, gamma]
                    avrg_tensor[a, b, c] = total/15.
        return avrg_tensor

# function generator
def generate_resonances_functions(subscripts, fermi=None, margin_rs=convWnum2Freq(10.)) -> Callable:
    """
    rs - reciprocal second
    """
    m1n1m2n2 = [i.split(',') for i in subscripts]
    if fermi is not None:
        fermi = [i.split(',') for i in fermi]

    def function(w_all_rs: dict, w1_rs: np.ndarray, w2_rs: np.ndarray, Gamma_rs: float,
                 abctuple: tuple[int, int] | tuple[int, int, int],
                 m1n1m2n2: list = m1n1m2n2, fermi: list = fermi) -> np.ndarray:
        # fixme - lorenzian shape cutoff

        letters = ['a', 'b', 'c', 'zero'] if len(abctuple) == 3 else ['a', 'b', 'zero']
        dictabc = dict(zip(letters, abctuple + tuple(['zero'])))
        w_all_rs[('zero',)] = 0.

        wm1 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[0][0].split('+')], key=int))
        wn1 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[0][1].split('+')], key=int))# if len(m1n1m2n2[0][1].split('+')) > 1 else tuple([m1n1m2n2[0][1]])

        if 'zero' not in m1n1m2n2[1][0].split('+'):
            wm2 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[1][0].split('+')], key=int))
        else:
            wm2 = tuple([m1n1m2n2[1][0]])

        wn2 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[1][1].split('+')], key=int))

        if fermi is None:

            f1 = w_all_rs[wm2] - w_all_rs[wn2] + w1_rs - 1j * Gamma_rs
            # test np.where

            return np.where(w2_rs-margin_rs > w1_rs, 1 / (w_all_rs[wm1] - w_all_rs[wn1]
                                                 + w1_rs - w2_rs
                                                 - 1j * Gamma_rs) / f1, 0.)
            # return np.where(w2-margin > w1, 1 / (w_all[wm1] - w_all[wn1] + convWnum2Freq(w1) - convWnum2Freq(w2)
            #                                      - 1j * Gamma) / (w_all[wm2] - w_all[wn2]
            #                                                       + convWnum2Freq(w1) - 1j * Gamma), 0.)
        else:
            w_fr11 = tuple(sorted([str(dictabc[i]) for i in fermi[0][0].split('+')], key=int))
            if 'zero' not in fermi[0][1].split('+'):
                w_fr21 = tuple(sorted([str(dictabc[i]) for i in fermi[0][1].split('+')], key=int))
            else:
                w_fr21 = tuple([fermi[0][1]])

            w_fr12 = tuple(sorted([str(dictabc[i]) for i in fermi[1][0].split('+')], key=int))
            if 'zero' not in fermi[1][1].split('+'):
                w_fr22 = tuple(sorted([str(dictabc[i]) for i in fermi[1][1].split('+')], key=int))
            else:
                w_fr22 = tuple([fermi[1][1]])
            # fixme : get resonance conditions once for both, then sumfrac only for mech
            t1 = w_all_rs[wm1] - w_all_rs[wn1] + w1_rs - w2_rs - 1j * Gamma_rs
            t2 = w_all_rs[wm2] - w_all_rs[wn2] + w1_rs - 1j * Gamma_rs
            t3 = convWnum2Freq(w_all_rs[w_fr11] - w_all_rs[w_fr21])
            t4 = convWnum2Freq(w_all_rs[w_fr12] - w_all_rs[w_fr22])

            sumfrac = (1 / t3 + 1 / t4)

            return (1 / t1 / t2) * sumfrac

    return function
