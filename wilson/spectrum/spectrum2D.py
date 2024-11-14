import time
from typing import Callable

import numpy as np
from scipy import constants

from .averaging import get_iso_f


class Spectrum2D:
    """
    Contains settings for current EVV derivations.
    Will hold all the necessary data and the terms for the evaluation of the intensities;
    from this general data more specifics can be extracted (data_analysis module, ...)

    input_data_info is a dictionary:

    result = {'source': 'cfour', 'type': 'out', 'files': c4_files_dict, 'source': 'cfour', 'type': 'out'}
    c4_files_dict = {'mol_code': mol_code, 'method': method, 'basis': basis,
                                   'out': c4_out(anharmonic equil.),
                                   'cubic': c4_cubic',
                                   'dipolexyz': c4_dipolexyz (location/name w/o xyz part),
                                   'polar': pkl_polar,
                                   'out_anharm_final': c4_out,
                                   'polar_pkl': pkl_polar}

    result = {'source': 'cfour', 'type': 'out', 'files': c4_files_dict, 'source': 'gaussian', 'type': 'log'}
        a helper can be used - DataVault.make_DatainputDict with specific choice of molecule calculation
    """

    def __init__(self, w1: np.array, w2: np.array):
        """
        TODO: remove w1 and w2 from init here; clean up init
        """
        if type(w1)==list or type(w2)==list:
            w1, w2 = np.array(w1), np.array(w2)

        # Define the grid of spectrum (pixels)
        self.w1_mesh, self.w2_mesh = np.meshgrid(w1, w2, indexing='ij')
        self.shape2d = self.w1_mesh.shape
        self.Gamma = None
        self.diagonal_margin_rc = None

        self.gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)

        self.saved_mech = {}
        self.saved_el = {}

        self.deriv_data = None
        self.corrected_levels = None


    def getDerivedTermsEVV(self):
        """
        Currently available for selection EVV terms
        """
        self.__resonancesTypes = [(-1, 2), (-1,)]
        self.__axes = {1: self.w1_mesh_Eh, 2: self.w2_mesh_Eh}
        self.w1w2Condition = self.__axes[2] - self.diagonal_margin_Eh > self.__axes[1]

        # Terms in the expressions
        self.__electrical_terms_str = [(('a+b,a', 'zero,a'), None),
                                       (('b,a', 'zero,a'), None)]

        self.__mechanical_terms_str = [ (('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')),
                                        (('c,a', 'zero,a'), ('a+b,c', 'b+c,a')),
                                        (('a+b,a', 'zero,a'), ('a,a+b', 'b,zero')),
                                        (('b,a', 'zero,a'), ('b,a+b', 'a,zero')),
                                        (('b,a', 'zero,a'), ('a,a+b', 'b,zero')),
                                        (('b,a', 'zero,a'), ('b,a+b', 'a,zero'))]

        # derivatives:
        # 1. mu_Q, mu QQ, alpha_Q - electric dipole (1st and 2nd derivatives), polarizability (1st der.)
        # 2. mu_Q, alpha_QQ - electric dipole (1st der.), polarizability (2nd der.)
        self.__electric_avrg_str = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))],
                                    [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))] ]

        # mu_Q, alpha_Q - for all 6 terms
        self.__mechanical_avrg_str = [ [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                                       [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                                       [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc'],
                                       [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc'],
                                       [('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc'],
                                       [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc']]

        self.__factors = [1., 1., 0.5, 0.5, -0.5, -0.5]


    def load_data(self, parserObj):
        """Loading the data from a parser object/DataVault
            with the sources given to it"""
        # TODO - make it more flexible, give an option to supply files
        # parserObj = parser(self.input_data_info)
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

        cff_cm_1 = parserObj.cubic_cm_1
        qff_cm_1 = parserObj.quartic_cm_1
        rot_c, cor_c = parserObj.rotational_constant, parserObj.coriolis_constant
        from .vpt2 import anharm_corr_energiesVPT2
        # corrected_levels = funds, over2q, combo2q, over3q, combo3q
        self.corrected_levels = anharm_corr_energiesVPT2(list(self.fundamentals_harmonic.values()),
                                                         cff_cm_1, qff_cm_1, rot_c, cor_c,
                                                         'Anharmonic: VPT2')

        self.all_states_corr = {}
        for i in range(len(self.fundamentals)):
            self.all_states_corr[tuple(str(i))] = self.corrected_levels[0][i]

            for j in range(i+1):
                if i==j:
                    self.all_states_corr[tuple([str(i), str(i)])] = self.corrected_levels[1][i]
                else:
                    self.all_states_corr[tuple(sorted([str(i), str(j)]))] = self.corrected_levels[2][i, j]

                for k in range(len(self.fundamentals)):
                    if i==j==k:
                        self.all_states_corr[tuple([str(i), str(i), str(i)])] = self.corrected_levels[3][i]
                    else:
                        key = tuple(sorted([str(i), str(j), str(k)]))
                        if key not in self.all_states_corr:
                            if self.corrected_levels[4][i, j, k]!=0.:
                                self.all_states_corr[tuple(sorted([str(i), str(j), str(k)]))] = self.corrected_levels[4][i, j, k]

        # self.all_states_corr = {tuple(str(i -7) for i in k): v for k, v in anharm_states_dict.items()}

    def setSpectrumSettings(self, Gamma_rc: float, diag_margin_rc: float = 10., vib_levels_harmonic: bool =True):
        """Settings to be set before computing the intensities.
        rc - reciprocal centimeter.

        vib_levels_harmonic - weather to use harmonic levels for resonance terms
                (useful for the investigations of Fermi resonances? or other)
        """
        self.Gamma = convNu2Ene(Gamma_rc)
        # margin for higher diagonal, to not show/compute data to close to the diagonal
        self.diagonal_margin_rc = diag_margin_rc
        self.conversion2InternalUnits()
        self.vib_levels_harmonic = vib_levels_harmonic
        print(f'\nUsed vibrational energy levels are harmonic? - {self.vib_levels_harmonic}')

    def conversion2InternalUnits(self):
        """
        Eh - Hartree unit
        convNu2Ene converts from wavenumber to Hartree
        """
        self.all_states_harmonic_Eh = {k: convNu2Ene(v) for k, v in self.all_states_harmonic.items()}
        self.all_states_Eh = {k: convNu2Ene(v) for k, v in self.all_states.items()}
        self.w1_mesh_Eh, self.w2_mesh_Eh = convNu2Ene(self.w1_mesh), convNu2Ene(self.w2_mesh)
        self.diagonal_margin_Eh = convNu2Ene(self.diagonal_margin_rc)

    def addTerms(self, electrical_terms_selection: list, mechanical_terms_selection: list):
        """Creating functions for computing the expressions for mechanical and electrical anharmonicities.
            Different functions because of the difference in terms.

        The terms available for selection are set with self.getDerivedTermsEVV() and are currently for EVV experiment
        """

        # setting up terms available for selection (all EVV terms now)
        self.getDerivedTermsEVV()

        # now used in the analysis
        self.e_selected, self.m_selected = electrical_terms_selection, mechanical_terms_selection
        # fraction factors for mechanical anh. terms
        self.mech_factors = [self.__factors[i] for i in mechanical_terms_selection]

        self.electrical_terms = [self.__electrical_terms_str[i] for i in electrical_terms_selection]
        self.mechanical_terms = [self.__mechanical_terms_str[i] for i in mechanical_terms_selection]

        self.electric_avrg = [self.__electric_avrg_str[i] for i in electrical_terms_selection]
        self.mechanical_avrg = [self.__mechanical_avrg_str[i] for i in mechanical_terms_selection]


        # TODO: identification of unique contributions; future precalculation
        # lines from 159 to 167 will be used for this purpose later
        electric_avrg_tuples = [tuple(self.__electric_avrg_str[i]) for i in electrical_terms_selection]
        mechanical_avrg_tuples = [tuple(self.__mechanical_avrg_str[i][:-1]) for i in mechanical_terms_selection]
        # a combined list
        combFreqDiff = ([(self.__electrical_terms_str[i])[1] for i in electrical_terms_selection]
                      + [(self.__mechanical_terms_str[i])[1] for i in mechanical_terms_selection])
        self.__collectionFreqDiff = set([j for i in [x for x in combFreqDiff if x is not None] for j in i])
        self.__collectionFreqRes = set([(self.__electrical_terms_str[i])[0] for i in electrical_terms_selection]
                                     + [(self.__mechanical_terms_str[i])[0] for i in mechanical_terms_selection])
        self.__collectionAveraging = set(electric_avrg_tuples + mechanical_avrg_tuples)

        # here the functions of 2 frequencies
        e_funcs = [self.generate_resonances_functions(i[0], i[1]) for i in self.electrical_terms]
        m_funcs = [self.generate_resonances_functions(i[0], i[1]) for i in self.mechanical_terms]

        # precalculated here
        self.el_avrg_tensors = [avrg_abc_tensor(self.electric_avrg[i], self.deriv_data, self.gammaCompsAll)
                                                                            for i in range(len(self.electric_avrg))]
        self.mech_avrg_tensors = [avrg_abc_tensor(self.mechanical_avrg[i], self.deriv_data, self.gammaCompsAll)
                                                                            for i in range(len(self.mechanical_avrg))]
        # print(self.el_avrg_tensors)
        # print(self.mech_avrg_tensors)
        # this mapping is used in the evaluation methods
        self.combofuns_tensors = [dict(zip(e_funcs, self.el_avrg_tensors)),
                                  dict(zip(m_funcs, zip(self.mech_avrg_tensors, self.mechanical_avrg)))]

        nmodes = len(self.fundamentals)
        # setting up the combinations of states for the terms
        self.coords_ab = np.indices([nmodes] * 2).reshape(2, -1).T if self.electrical_terms else []
        self.coords_abc = np.indices([nmodes] * 3).reshape(3, -1).T if self.mechanical_terms else []


    def precalculateParts(self):
        """
        Precalculate some parts:
            factors (1/wa/wb/wc);
            resonance terms (wmn[-1,2], wmn[-1]);
            diff terms (wmn)
        """
        vib_ene_levels_harmonic = convNu2Ene(np.array([v for k,v in self.fundamentals_harmonic.items()]))

        self.prefac_2d = np.outer(vib_ene_levels_harmonic, vib_ene_levels_harmonic)
        self.prefac_3d = (vib_ene_levels_harmonic[:, np.newaxis, np.newaxis] *
                          vib_ene_levels_harmonic[np.newaxis, :, np.newaxis] *
                          vib_ene_levels_harmonic[np.newaxis, np.newaxis, :])
        # [-1, 2] and [-1] types of terms: w1-w2 or w1, without w_{m,n}
        # these 2d arrays will be added to combinations of wm and wn when looped over combinations of a, b, (c)
        self.resonances = {}
        for typelist in self.__resonancesTypes:
            self.resonances[typelist] = (-1) * sum([np.sign(ix)*np.where(self.w1w2Condition,
                                                                         self.__axes[abs(ix)], 0) for ix in typelist])


    def get_total_gamma_sum_el(self, a: int, b: int) -> np.ndarray:
        """
        Computes \gamma^{[1,0]} for given combination of modes

        Future: get_total_gamma_sum_el unified with get_total_gamma_sum_mech
        so there would be only one get_gamma function without if ... else conditions
        """
        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic_Eh
        else:
            vib_ene_levels = self.all_states_Eh
        total_sum_el = 0
        prefac_el = self.prefac_2d.T[a, b]                                              # a number

        for index, (el_func, elavrg) in enumerate(self.combofuns_tensors[0].items()):
            # resonance computed on the grid; could be precalculated with keys in self.__collectionFreqRes (later)
            resonance = el_func(vib_ene_levels, self.resonances, (a, b))                # a 2D np.array
            total_sum_el += elavrg[a, b] * resonance / prefac_el                        # elavrg[a, b] is a number

        return total_sum_el / 24.

    def get_total_gamma_sum_mech(self, a: int, b: int, c: int) -> np.ndarray:
        """
        Computes \gamma^{[0,1]} for given combination of modes

        Future: get_total_gamma_sum_el unified with get_total_gamma_sum_mech
        so there would be only one get_gamma function without if ... else conditions
        """
        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic_Eh
        else:
            vib_ene_levels = self.all_states_Eh

        total_sum_mech = 0
        prefac_mech = self.prefac_3d.T[a, b, c]
        for index, (mech_func, mechavrg_pair) in enumerate(self.combofuns_tensors[1].items()):

            mechavrgF = mechavrg_pair[1]                                            # a string of indices for F (cff)
            abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
            ijk_indx = tuple([abc[j] for j in mechavrgF[-1]])
            F = self.deriv_data['F_abc'][ijk_indx]                                  # a number, from F tensor
            # resonance2 is a product of resonances and freq. difference term
            resonance2 = mech_func(vib_ene_levels, self.resonances, (a, b, c))      # a 2D np.array (Nomega1, Nomega2)
            mechavrg = mechavrg_pair[0]                                             # a 2D np.array (nmodes, nmodes)

            addition = self.mech_factors[index] / prefac_mech * mechavrg[a, b, c] * F * resonance2
            total_sum_mech += addition

        return total_sum_mech / (-48.)

    def intensity_electrical(self) -> (np.ndarray, dict):
        """
        Looping over a,b combinations - full sum of \gamma^{[1,0]}
        """
        start_time = time.time()

        Qab_contrib_dict = {}

        elall = np.zeros(self.shape2d, dtype='complex128')
        for ind, i in enumerate(self.coords_ab):
            contrib_ab = self.get_total_gamma_sum_el(i[0], i[1])
            # saving contribution of each pair of normal modes - may be organized in other way or just taken out
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
        """
        Looping over a,b,c combinations - full sum of \gamma^{[0,1]}
        """

        start_time = time.time()

        Qabc_contrib_dict = {}

        mechall = np.zeros(self.shape2d, dtype='complex128')
        for ind, i in enumerate(self.coords_abc):
            contrib_abc = self.get_total_gamma_sum_mech(i[0], i[1], i[2])
            # saving contribution of each triple of normal modes - may be organized in other way or just taken out
            Qabc_contrib_dict[tuple(i)] = contrib_abc
            mechall += contrib_abc
            if ind % 1000 == 0:
                print(f'{ind}/{len(self.coords_abc)} modes combinations -- {ind*100/len(self.coords_abc)}%')

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nExecution time - mechanical: {execution_time} seconds")
        print('Mechanical anharmonicities are calculated')

        return mechall, Qabc_contrib_dict

    def generate_resonances_functions(self, subscripts, freqDiff=None) -> Callable:
        """
        Generates a python function for a term given by a formula (subscripts and freqDiff);
                varied argument of that function is abctuple (used in the loop over combinations of modes).
        subscripts - a tuple of strings from the formula; subscripts of omega energy levels in the resonance part;
                        e.g., ('a+b,a', 'zero,a')
        freqDiff - a tuple of strings from the formula; subscripts of omega energy levels in the freq. difference part;
                        e.g., ('a+b+c,0', 'c,a+b'); not None for mech. anharm.
        """

        m1n1m2n2 = [i.split(',') for i in subscripts]
        if freqDiff is not None:
            freqDiff = [i.split(',') for i in freqDiff]

        def function(allLevels_Eh: dict, w_res_dict: dict[str:np.ndarray],
                     abctuple: tuple[int, int] | tuple[int, int, int],
                     m1n1m2n2: list = m1n1m2n2, freqDiff: list = freqDiff,
                     Gamma_rs: float = self.Gamma, filter=self.w1w2Condition) -> np.ndarray:
            """
            allLevels_Eh collects all vibrational energy levels in Hartree; e.g., [('1', '2')] - combination mode
            w_res_dict contains [-1, 2] and [-1] 2d arrays (in s-1)
            abctuple is a tuple of normal mode indices for which current iteration is evaluating resonance term
            """
            # todo: lorentzian shape cutoff

            letters = ['a', 'b', 'c', 'zero'] if len(abctuple) == 3 else ['a', 'b', 'zero']
            dictabc = dict(zip(letters, abctuple + tuple(['zero'])))
            allLevels_Eh[('zero',)] = 0.

            wm1 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[0][0].split('+')], key=int))
            wn1 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[0][1].split('+')], key=int))
            # fixme ?
            # if len(m1n1m2n2[0][1].split('+')) > 1 else tuple([m1n1m2n2[0][1]])

            if 'zero' not in m1n1m2n2[1][0].split('+'):
                wm2 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[1][0].split('+')], key=int))
            else:
                wm2 = tuple([m1n1m2n2[1][0]])

            wn2 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[1][1].split('+')], key=int))

            t1 = allLevels_Eh[wm1] - allLevels_Eh[wn1] + w_res_dict[(-1, 2)] - 1j * Gamma_rs
            t2 = allLevels_Eh[wm2] - allLevels_Eh[wn2] + w_res_dict[(-1,)] - 1j * Gamma_rs

            if freqDiff is None:
                sumfrac = 1.

            else:
                w_fr11 = tuple(sorted([str(dictabc[i]) for i in freqDiff[0][0].split('+')], key=int))
                if 'zero' not in freqDiff[0][1].split('+'):
                    w_fr21 = tuple(sorted([str(dictabc[i]) for i in freqDiff[0][1].split('+')], key=int))
                else:
                    w_fr21 = tuple([freqDiff[0][1]])

                w_fr12 = tuple(sorted([str(dictabc[i]) for i in freqDiff[1][0].split('+')], key=int))
                if 'zero' not in freqDiff[1][1].split('+'):
                    w_fr22 = tuple(sorted([str(dictabc[i]) for i in freqDiff[1][1].split('+')], key=int))
                else:
                    w_fr22 = tuple([freqDiff[1][1]])

                t3 = allLevels_Eh[w_fr11] - allLevels_Eh[w_fr21]
                t4 = allLevels_Eh[w_fr12] - allLevels_Eh[w_fr22]

                sumfrac = (1 / t3 + 1 / t4)

            product = t1 * t2
            return  np.where(filter, sumfrac / product, 0)

        return function


def convNu2Ene(reciprocal_cm: float | np.ndarray) -> float | np.ndarray:
    """Convert wavenumber (cm-1) to energy (Hartree)"""
    hartree2J = constants.physical_constants['hartree-joule relationship'][0]
    return reciprocal_cm * (100 * constants.h * constants.c / hartree2J)


def get_AlphaBetaGammaDelta_indices(num_f: int) -> np.ndarray:
    """
    Now is set for the EVV experiment and for ZZZZ polarization.

    pol_g is a list of lists of 2 lists where the second one is empty
          but first one contains the lists of interest

    :param num_f: number of pulses
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
        formula = formula[:-1]      # removes the Fabc string, to deal only with the averaging part

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


def main_program(spectrumObj: Spectrum2D, dictInputs: dict):
    """
    Is here as a recipe for the sequence of methods to be called (most are not optional now; fixme later)
    """
    # init
    # loading derivatives and vib. energies data
    spectrumObj.load_data(dictInputs['parserObject'])
    # some parameters for the rendered spectrum
    spectrumObj.setSpectrumSettings(Gamma_rc=10., diag_margin_rc=10., vib_levels_harmonic=False)
    # spectrumObj.conversion2InternalUnits() # need now at least for diag margin_rs in addTerms();
    #                                           happens in setSpectrumSettings()
    spectrumObj.addTerms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculateParts()
    # now can do intensity calculation for init(w1, w2, gamma)

class EvalTerm:

    def __init__(self, fraction=None, prefac=None, resonances=None, freqDiff=None, avrg=None, cff=None):
        self.fraction = fraction
        self.prefac = prefac    # 'ab' or 'abc'
        self.avrg = avrg    # [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))]
        #                     [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',))]
        self.resonances = resonances
        self.freqDiff = freqDiff
        self.cff = cff

    def calcAveraging(self, data: dict[str:np.ndarray], gammaCompsAll: np.array) -> np.ndarray:
        # todo: to be continued later
        """
        Calculate the averaging tensor for a given formula.
        Indices of the tensor are normal coordinates (NC) indices,
        and the shape of the tensor depends on the nature of the term that is being calculated.
        Shape of the averaging tensor for electrical anharmonicity terms is (n_NC, n_NC)
        Shape of the averaging tensor for mechanical anharmonicity terms is (n_NC, n_NC, n_NC)

        data dict should contain 'alpha_Q', 'alpha_QQ', 'mu_Q', 'mu_QQ'

        """


        nmodes = data['mu_Q'].shape[0]

        # specific case of the gamma_1,0 first term
        if [i[0] for i in self.avrg] == ['mu_Q', 'alpha_Q', 'mu_QQ']:
            avrg_tensor = np.zeros((nmodes, nmodes))
            for a in range(nmodes):
                for b in range(nmodes):
                    total = 0.
                    for comps in gammaCompsAll:
                        alpha, beta, gamma, delta = comps
                        total += data['mu_Q'][a, beta] * data['alpha_Q'][b, alpha, delta] * data['mu_QQ'][a, b, gamma]
                    avrg_tensor[a, b] = total / 15.
            return avrg_tensor

        # specific case of the gamma_1,0 second term
        elif [i[0] for i in self.avrg] == ['mu_Q', 'alpha_QQ', 'mu_Q']:
            avrg_tensor = np.zeros((nmodes, nmodes))
            for a in range(nmodes):
                for b in range(nmodes):
                    total = 0.
                    for comps in gammaCompsAll:
                        alpha, beta, gamma, delta = comps
                        total += data['mu_Q'][a, beta] * data['alpha_QQ'][a, b, alpha, delta] * data['mu_Q'][b, gamma]
                    avrg_tensor[a, b] = total / 15.
            return avrg_tensor

        # all terms of gamma_0,1 have this structure of averaging part
        elif [i[0] for i in self.avrg] == ['mu_Q', 'alpha_Q', 'mu_Q']:
            avrg_tensor = np.zeros((nmodes, nmodes, nmodes))
            # this part is changing for different terms
            modes_letters = [i[1] for i in self.avrg]
            for a in range(nmodes):
                for b in range(nmodes):
                    for c in range(nmodes):
                        abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
                        i1, i2, i3 = [abc[j[0]] for j in modes_letters]
                        total = 0.
                        for comps in gammaCompsAll:
                            alpha, beta, gamma, delta = comps
                            total += data['mu_Q'][i1, beta] * data['alpha_Q'][i2, alpha, delta] * data['mu_Q'][
                                i3, gamma]
                        avrg_tensor[a, b, c] = total / 15.
            return avrg_tensor
