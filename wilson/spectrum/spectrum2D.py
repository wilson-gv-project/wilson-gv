import copy
import time
from datetime import timedelta
from typing import Callable

import numpy as np
import pandas as pd

from .averaging import get_AlphaBetaGammaDelta_indices
from .tools import convNu2Ene, avrg_abc_tensor


import itertools
def combinations_with_permutations(iterable, k):
    return (comb for comb in itertools.product(iterable, repeat=k))

numcombperm = lambda n, k: n**k


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

    def __init__(self, w1=None, w2=None):
        """
        TODO: remove w1 and w2 from init here; clean up init
        """
        if w2 is None:
            w2 = []
        if w1 is None:
            w1 = []
        if type(w1)==list or type(w2)==list:
            self.w1, self.w2 = np.array(w1), np.array(w2)
        else:
            self.w1, self.w2 = w1, w2
        # define the grid of spectrum (pixels)
        self.w1_mesh, self.w2_mesh = np.meshgrid(w1, w2, indexing='ij')
        self.shape2d = self.w1_mesh.shape

        # initialized final spectrum pixels
        self.intensities_grid = np.zeros(self.shape2d, dtype='complex64')

        self.Gamma = None
        self.Gamma_rc = None
        self.diagonal_margin_rc = None

        self.gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)

        self.saved_mech = {}
        self.saved_el = {}

        self.deriv_data = None
        self.corrected_levels = None

        self.mechab = False

    def getDerivedTermsEVV(self):
        """
        Currently available for selection EVV terms
        """
        self.resonancesTypes = [(-1, 2), (-1,)]
        self.axes = {1: self.w1_mesh_Eh, 2: self.w2_mesh_Eh}
        self.w1w2Condition = self.axes[2] - self.diagonal_margin_Eh > self.axes[1]
        # self.w1w2Condition = np.ones(self.w1_mesh_Eh.shape, dtype=bool)

        # Terms in the expressions
        self.electrical_terms_str = [(('a+b,a', 'zero,a'), None),
                                       (('b,a', 'zero,a'), None)]

        self.mechanical_terms_str = [ (('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')),
                                        (('b,a', 'zero,a'), ('a+c,b', 'b+c,a')),
                                        (('a+b,a', 'zero,a'), ('a,a+b', 'b,zero')),
                                        (('b,a', 'zero,a'), ('b,a+b', 'a,zero')),
                                        (('b,a', 'zero,a'), ('a,a+b', 'b,zero')),
                                        (('b,a', 'zero,a'), ('b,a+b', 'a,zero'))]

        # derivatives:
        # 1. mu_Q, mu QQ, alpha_Q - electric dipole (1st and 2nd derivatives), polarizability (1st der.)
        # 2. mu_Q, alpha_QQ - electric dipole (1st der.), polarizability (2nd der.)
        self.electric_avrg_str = [(('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))),
                                    (('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))) ]

        # mu_Q, alpha_Q - for all 6 terms
        self.mechanical_avrg_str = [   (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc', 1.),
                                       (('mu_Q', ('a',)), ('alpha_Q', ('c',)), ('mu_Q', ('b',)), 'acb', 1.),
                                       (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc', 0.5),
                                       (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc', 0.5),
                                       (('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc', -0.5),
                                       (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc', -0.5)]


    def load_data(self, parserObj, vpt2=False, vpt2settings=None):
        """
        Loading the data from a parser object/DataVault
            with the sources given to it

        anharmonic_type options:
            'Anharmonic: VPT2'                                                              - don't do_res, don't do_var
            'Anharmonic: DVPT2' = 'Anharmonic: Freq DVPT2, Int VPT2'
                    = 'Anharmonic: DVPT2, w/ 1-1 checks'                                    - do_res, don't do_var
            'Anharmonic: Freq GVPT2, Int DVPT2' = 'Anharmonic: Freq GVPT2, Int DVPT2, w/ 1-1 checks'
                    = 'Anharmonic: Freq GVPT2, Int DVPT2, w/ 1-1 checks and forced removal' - do_res, do_var
        """
        # TODO - make it more flexible, give an option to supply files
        # parserObj = parser(self.input_data_info)

        if vpt2settings is None:
            vpt2settings = {'anharmonic_type': 'Anharmonic: VPT2'}
            #

        parserObj.getData()
        self.parserObj = parserObj

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
        # 'mu_Q',  'mu_QQ',  'alpha_Q', 'alpha_QQ', 'F_abc'
        # (6, 3)  (6, 6, 3)  (6, 3, 3) (6, 6, 3, 3) (6, 6, 6)

        if vpt2:
            if parserObj.DD11 or parserObj.DD13 or parserObj.DD22:
                print("Warning: found Darling-Dennison resonances_args in data:")
                print(f"DD 1-1: {parserObj.DD11}")
                print(f"DD 2-2: {parserObj.DD22}")
                print(f"DD 1-3: {parserObj.DD13}")

            cff_cm_1 = parserObj.cubic_cm_1
            qff_cm_1 = parserObj.quartic_cm_1
            rot_c, cor_c = parserObj.rotational_constant, parserObj.coriolis_constant
            # print('\n', rot_c.shape, cor_c.shape)
            # exit()
            from .vpt2 import anharm_corr_energiesVPT2
            # corrected_levels = funds, over2q, combo2q, over3q, combo3q
            self.corrected_levels = anharm_corr_energiesVPT2(list(self.fundamentals_harmonic.values()),
                                                             cff_cm_1, qff_cm_1, rot_c, cor_c,
                                                             vpt2settings['anharmonic_type'])
            self.all_states_corr = {}
            for i in range(len(self.fundamentals)):
                self.all_states_corr[(str(i),)] = self.corrected_levels[0][i]
                # if i==10:
                #     print(i, tuple(str(i)), (str(i),))
                #     print(self.all_states_corr[(str(i),)])
                #     print(self.all_states_corr[('10',)])
                for j in range(i+1):
                    if i==j:
                        self.all_states_corr[tuple([str(i), str(i)])] = self.corrected_levels[1][i]
                    else:
                        self.all_states_corr[tuple([str(el) for el in sorted([i, j])])] = self.corrected_levels[2][i, j]

                    for k in range(len(self.fundamentals)):
                        if i==j==k:
                            self.all_states_corr[tuple([str(i), str(i), str(i)])] = self.corrected_levels[3][i]
                        else:
                            key = tuple([str(el) for el in sorted([i, j, k])])
                            if key not in self.all_states_corr:
                                if self.corrected_levels[4][i, j, k]!=0.:
                                    self.all_states_corr[tuple([str(el) for el in sorted([i, j, k])])] = self.corrected_levels[4][i, j, k]

            self.all_states = self.all_states_corr

    def setSpectrumSettings(self, Gamma_rc: float, diag_margin_rc: float = 10., vib_levels_harmonic: bool =True):
        """Settings to be set before computing the intensities.
        rc - reciprocal centimeter.

        vib_levels_harmonic - weather to use harmonic levels for resonance terms
                (useful for the investigations of Fermi resonances_args? or other)
        """
        self.Gamma_rc = Gamma_rc
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
        self.mech_factors = [self.mechanical_avrg_str[i][-1] for i in mechanical_terms_selection]

        self.electrical_terms = [self.electrical_terms_str[i] for i in electrical_terms_selection]
        self.mechanical_terms = [self.mechanical_terms_str[i] for i in mechanical_terms_selection]

        self.electric_avrg = [self.electric_avrg_str[i] for i in electrical_terms_selection]
        self.mechanical_avrg = [self.mechanical_avrg_str[i] for i in mechanical_terms_selection]


        # TODO: identification of unique contributions; future pre-calculation
        electric_avrg_tuples = [tuple(self.electric_avrg_str[i]) for i in electrical_terms_selection]
        mechanical_avrg_tuples = [tuple(self.mechanical_avrg_str[i][:-1]) for i in mechanical_terms_selection]
        # a combined list
        combFreqDiff = ([(self.electrical_terms_str[i])[1] for i in electrical_terms_selection]
                      + [(self.mechanical_terms_str[i])[1] for i in mechanical_terms_selection])
        self.collectionFreqDiff = set([j for i in [x for x in combFreqDiff if x is not None] for j in i])
        self.collectionFreqRes = set([(self.electrical_terms_str[i])[0] for i in electrical_terms_selection]
                                     + [(self.mechanical_terms_str[i])[0] for i in mechanical_terms_selection])
        self.collectionAveraging = set(electric_avrg_tuples + mechanical_avrg_tuples)

        # here the functions of 2 frequencies
        self.e_funcs = [self.generate_resonances_functions(i[0], i[1]) for i in self.electrical_terms]
        self.m_funcs = [self.generate_resonances_functions(i[0], i[1]) for i in self.mechanical_terms]

        # precalculated here
        self.el_avrg_tensors = [avrg_abc_tensor(self.electric_avrg[i], self.deriv_data, self.gammaCompsAll)
                                                                            for i in range(len(self.electric_avrg))]
        self.mech_avrg_tensors = [avrg_abc_tensor(self.mechanical_avrg[i], self.deriv_data, self.gammaCompsAll)
                                                                            for i in range(len(self.mechanical_avrg))]

        # precalculated here #TODO new!
        self.el_avrg_tensorsD = {tuple(i): avrg_abc_tensor(i, self.deriv_data, self.gammaCompsAll)
                                  for i in self.electric_avrg}
        self.mech_avrg_tensorsD = {tuple(i): avrg_abc_tensor(i, self.deriv_data, self.gammaCompsAll)
                                    for i in self.mechanical_avrg}
        self.el_terms = [(self.electrical_terms_str[i], self.electric_avrg_str[i]) for i in electrical_terms_selection]
        self.mech_terms = [(self.mechanical_terms_str[i], self.mechanical_avrg_str[i]) for i in mechanical_terms_selection]

        # this mapping is used in the evaluation methods
        self.combofuns_tensors = [dict(zip(self.e_funcs, self.el_avrg_tensors)),
                                  dict(zip(self.m_funcs, zip(self.mech_avrg_tensors, self.mechanical_avrg)))]

        self.nmodes = len(self.fundamentals)
        # setting up the combinations of states for the terms
        self.coords_ab = np.indices([self.nmodes] * 2).reshape(2, -1).T if self.electrical_terms else []
        self.coords_abc = np.indices([self.nmodes] * 3).reshape(3, -1).T if self.mechanical_terms else []

    def precalculateParts(self, *,
                          list2exclude=None,
                          preview=False,
                          screenmodeswindow=True):
        """
        Precalculate some parts:
            factors (1/wa/wb/wc);
            resonance terms (wmn[-1,2], wmn[-1]);
            diff terms (wmn)
        """
        if list2exclude is None:
            list2exclude = []

        self.screenmodeswindow = screenmodeswindow

        vib_ene_levels_harmonic = convNu2Ene(np.array([v for k,v in self.fundamentals_harmonic.items()]))

        self.prefac_2d = np.outer(vib_ene_levels_harmonic, vib_ene_levels_harmonic)
        self.prefac_3d = (vib_ene_levels_harmonic[:, np.newaxis, np.newaxis] *
                          vib_ene_levels_harmonic[np.newaxis, :, np.newaxis] *
                          vib_ene_levels_harmonic[np.newaxis, np.newaxis, :])
        # [-1, 2] and [-1] types of terms: w1-w2 or w1, without w_{m,n}
        # these 2d arrays will be added to combinations of wm and wn when looped over combinations of a, b, (c)
        self.resonances_args = {}
        # fixme: computes all (2) now
        for typelist in self.resonancesTypes:
            self.resonances_args[typelist] = (-1) * sum([np.sign(ix) * np.where(self.w1w2Condition,
                                                                                self.axes[abs(ix)], 0) for ix in typelist]) - 1j * self.Gamma
        # selection of vibrational energy levels
        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic_Eh
            vib_ene_levels_rc = copy.deepcopy(self.all_states_harmonic)
        else:
            vib_ene_levels = self.all_states_Eh
            vib_ene_levels_rc = copy.deepcopy(self.all_states)

        self.nmodes = len(self.fundamentals)
        self.nmodes_original = len(self.fundamentals)

        self.list2exclude = list2exclude
        if list2exclude:
            self.mode_indices = [i for i in np.arange(self.nmodes) if i not in list2exclude]
            self.nmodes -= len(list2exclude)
        else:
            self.mode_indices = [i for i in np.arange(self.nmodes)]

        w_apbbma = np.zeros((self.nmodes, self.nmodes))
        w_bma = np.zeros((self.nmodes, self.nmodes))
        for a in range(self.nmodes):
            for b in range(self.nmodes):
                w_apbbma[a, b] = vib_ene_levels[tuple([str(el) for el in sorted([a, b])])] - vib_ene_levels[tuple([str(a)])]
                w_bma[a, b] = vib_ene_levels[tuple([str(b)])] - vib_ene_levels[tuple([str(a)])]

        za = np.array([-vib_ene_levels[tuple([str(k)])] for k in range(self.nmodes)])
        self.w_mn_dict = {'a+b,a': w_apbbma, 'b,a': w_bma, 'c,a': w_bma,
                          'zero,a': np.tile(za, self.nmodes).reshape(self.nmodes, -1).T}

        self.res_dict = {}

        w_apbbma_rc = np.zeros((self.nmodes, self.nmodes))
        w_bma_rc = np.zeros((self.nmodes, self.nmodes))
        for a in range(self.nmodes):
            for b in range(self.nmodes):
                w_apbbma_rc[a, b] = vib_ene_levels_rc[tuple([str(el) for el in sorted([a, b])])] - vib_ene_levels_rc[
                    tuple([str(a)])]
                w_bma_rc[a, b] = vib_ene_levels_rc[tuple([str(b)])] - vib_ene_levels_rc[tuple([str(a)])]

        za_rc = np.array([-vib_ene_levels_rc[tuple([str(k)])] for k in range(self.nmodes)])
        za_rc = np.tile(za_rc, self.nmodes).reshape(self.nmodes, -1).T

        self.res_dict[('a+b,a', 'zero,a')] = []
        self.res_dict[('b,a', 'zero,a')] = []

        if self.screenmodeswindow:
            self.inwindow = {}
            mw1, Mw1 = self.w1_mesh.min(), self.w1_mesh.max()
            mw2, Mw2 = self.w2_mesh.min(), self.w2_mesh.max()

            self.inwindow[('a+b,a', 'zero,a')] = {}
            self.inwindow[('b,a', 'zero,a')] = {}

        for a in range(self.nmodes):
            for b in range(self.nmodes):
                if a in self.list2exclude or b in self.list2exclude:
                    continue
                self.res_dict[('a+b,a', 'zero,a')].append(((-za_rc[a, b],
                                                           w_apbbma_rc[a, b] - za_rc[a, b]), (a, b)))
                self.res_dict[('b,a', 'zero,a')].append(((-za_rc[a, b],
                                                         w_bma_rc[a, b] - za_rc[a, b]), (a, b)))
                if self.screenmodeswindow:
                    margin = 50.
                    if not (mw1+margin <= -za_rc[a, b] <= Mw1-margin) and not (mw2+margin <= w_apbbma_rc[a, b] - za_rc[a, b] <= Mw2-margin):
                        self.inwindow[('a+b,a', 'zero,a')][(a,b)] = (-za_rc[a, b], w_apbbma_rc[a, b] - za_rc[a, b])
                    if not (mw1+margin <= -za_rc[a, b] <= Mw1-margin) and not (mw2+margin <= w_bma_rc[a, b] - za_rc[a, b] <= Mw2-margin):
                        self.inwindow[('a+b,a', 'zero,a')][(a,b)] = (-za_rc[a, b], w_bma_rc[a, b] - za_rc[a, b])
        # print(self.inwindow) if self.screenmodeswindow else None
        # print(mw1, Mw1, mw2, Mw2)
        # exit()

        fromdiagonal = {}
        fromdiagonal[('a+b,a', 'zero,a')] = set(
            [i for i in self.res_dict[('a+b,a', 'zero,a')] if i[0][0] >= i[0][1]])
        fromdiagonal[('b,a', 'zero,a')] = set(
            [i for i in self.res_dict[('b,a', 'zero,a')] if i[0][0] >= i[0][1]])

        self.res_dict[('a+b,a', 'zero,a')] = set(
            [i for i in self.res_dict[('a+b,a', 'zero,a')] if i[0][0] < i[0][1] - self.diagonal_margin_rc])
        self.res_dict[('b,a', 'zero,a')] = set(
            [i for i in self.res_dict[('b,a', 'zero,a')] if i[0][0] < i[0][1] - self.diagonal_margin_rc])


        self.comb_fac_dict = {}
        for old_key in self.m_selected:
            newkey = tuple([self.mechanical_terms_str[old_key],
                                self.mechanical_avrg_str[old_key]])
            self.comb_fac_dict[newkey] = np.zeros((self.nmodes_original, self.nmodes_original))

        # for ab in self.coords_ab:
        for ab in combinations_with_permutations(self.mode_indices, 2):

            a, b = ab
            for key in self.m_selected:
                newkey = tuple([self.mechanical_terms_str[key],
                                self.mechanical_avrg_str[key]])
                self.comb_fac_dict[newkey][a,b] = self.get_gamma_mech(a, b, factor=True)[key]


        if preview:

            import matplotlib.pyplot as plt
            plt.figure(figsize=(18, 16))
            x, y = zip(*(set([i[0] for i in self.res_dict[('a+b,a', 'zero,a')]])|set([i[0] for i in self.res_dict[('b,a', 'zero,a')]])))
            x1, y2 = zip(*(fromdiagonal[('a+b,a', 'zero,a')]|fromdiagonal[('b,a', 'zero,a')]))
            plt.scatter(x, y)
            plt.scatter(x1, y2, color='r')
            plt.plot(x,x, label=f'X=Y, diag margin {self.diagonal_margin_rc} cm-1', color='g')
            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            x_ticks = np.arange(x_min - (x_min % 100), x_max + 100, 100)
            y_ticks = np.arange(y_min - (y_min % 100), y_max + 100, 100)
            plt.xticks(x_ticks)
            plt.yticks(y_ticks)
            plt.legend()
            plt.savefig(self.parserObj.molecule+'_resloc.svg', format='svg')
            exit()

        # from .. import analysis
        # dfs4terms_el, dfs4terms_mech = analysis.get_resonances_DF(self, rec_cm=True,
        #                                                           vib_levels_harmonic=self.vib_levels_harmonic)
        # if dfs4terms_el:
        #     self.resonancesDFel = pd.concat(dfs4terms_el, ignore_index=True).query('w_2>w_1 & avrg_g>1e-30')
        # else:
        #     self.resonancesDFel = pd.DataFrame()
        #
        # if dfs4terms_mech:
        #     self.resonancesDFmech = pd.concat(dfs4terms_mech, ignore_index=True).query('(avrg_g>1e-30 & F_abc != 0.) & w_2>w_1')
        # else:
        #     self.resonancesDFmech = pd.DataFrame()

    def locateOnBigGrid(self, seed, radius):

        stepX = self.w1[1]-self.w1[0]
        stepY = self.w2[1]-self.w2[0]

        # indices of grid point closest to resonance point
        closestXind = round((seed[0]-np.min(self.w1))/stepX)
        closestYind = round((seed[1]-np.min(self.w2))/stepY)

        # number of steps to reach radius distance (rounded)
        radiusIndX = round(radius/stepX)
        radiusIndY = round(radius/stepY)

        # start-end indices for smaller grid fo X
        endIndX = closestXind+radiusIndX
        strIndX = closestXind-radiusIndX

        # start-end indices for smaller grid fo Y
        endIndY = closestYind + radiusIndY
        strIndY = closestYind - radiusIndY

        # corrections for grid boundaries
        if endIndX>self.w1.shape[0]:
            endIndX = self.w1.shape[0]-1
        if endIndY>self.w2.shape[0]:
            endIndY = self.w2.shape[0]-1
        if strIndX<0:
            strIndX = 0
        if strIndY<0:
            strIndY = 0

        return strIndX, endIndX + 1, strIndY, endIndY + 1

    def findAllGrids(self, radius_rc):
        st = time.time()

        allRes = list(self.res_dict[('a+b,a', 'zero,a')] | self.res_dict[('b,a', 'zero,a')])
        resGridsDict = {}

        w1grid_Ha = self.w1_mesh_Eh
        w2grid_Ha = self.w2_mesh_Eh
        w1grid = self.w1_mesh
        w2grid = self.w2_mesh

        for seed in allRes:
            x1, x2, y1, y2 = self.locateOnBigGrid(seed, radius_rc)
            cutout_w1 = w1grid[x1:x2+1, y1:y2+1]
            cutout_w2 = w2grid[x1:x2+1, y1:y2+1]

            cutout_w1_Ha = w1grid_Ha[x1:x2+1, y1:y2+1]
            cutout_w2_Ha = w2grid_Ha[x1:x2+1, y1:y2+1]

            axes = {1: cutout_w1_Ha, 2: cutout_w2_Ha}
            resonances = {}
            resonancesTypes = [(-1, 2), (-1,)]
            for typelist in resonancesTypes:
                resonances[typelist] = ((-1) * sum([np.sign(ix) * axes[abs(ix)] for ix in typelist])
                                                - 1j * self.Gamma)

            resGridsDict[tuple([seed, (x1, x2+1, y1, y2+1)])] = (cutout_w1, cutout_w2,
                                                                 cutout_w1_Ha, cutout_w2_Ha,
                                                                 resonances)
        elapsed_time = time.time() - st
        elapsed_timedelta = timedelta(seconds=elapsed_time)
        formatted_time = str(elapsed_timedelta)
        print('findAllGrids in:', formatted_time)
        return resGridsDict


    def get_total_gamma_sum_el(self, a: int, b: int, selectionCond: np.ndarray) -> np.ndarray:
        """
        Computes \\gamma^{[1,0]} for given combination of modes

        Future: get_total_gamma_sum_el unified with get_total_gamma_sum_mech
        so there would be only one get_gamma function without if ... else conditions
        """

        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic_Eh
        else:
            vib_ene_levels = self.all_states_Eh

        prefac_el = self.prefac_2d.T[a, b]         # a number

        # going through el. terms
        for index, (el_func, elavrg) in enumerate(self.combofuns_tensors[0].items()):
            if self.screenmodeswindow:
                # checking if resonance point is inside the full window; getting with the resonance key
                if (a,b) in self.inwindow[self.electrical_terms[index][0]]:
                    continue

            # resonance computed on the grid; could be precalculated with keys in self.collectionFreqRes (later???)
            resonance = el_func(allLevels_Eh=vib_ene_levels, w_res_dict=self.resonances_args, abctuple=(a, b),
                                w1w2Condition=(self.w1w2Condition & selectionCond))       # a 2D np.array
            self.intensities_grid += elavrg[a, b] * resonance / prefac_el / 24.              # elavrg[a, b] is a number

        return self.intensities_grid

    def get_total_gamma_sum_mech(self, a: int, b: int, c: int, selectionCond: np.ndarray) -> np.ndarray:
        """
        Computes \\gamma^{[0,1]} for given combination of modes

        Future: get_total_gamma_sum_el unified with get_total_gamma_sum_mech
        so there would be only one get_gamma function without if ... else conditions
        """
        # import time
        # st = time.time()

        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic_Eh
        else:
            vib_ene_levels = self.all_states_Eh

        # total_sum_mech = 0
        prefac_mech = self.prefac_3d.T[a, b, c]
        for index, (mech_func, mechavrg_pair) in enumerate(self.combofuns_tensors[1].items()):

            mechavrgF = mechavrg_pair[1]                                            # a string of indices for F (cff)
            abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
            ijk_indx = tuple([abc[j] for j in mechavrgF[-2]])
            F = self.deriv_data['F_abc'][ijk_indx]                                  # a number, from F tensor
            # resonance2 is a product of resonances_args and freq. difference term
            resonance2 = mech_func(allLevels_Eh=vib_ene_levels, w_res_dict=self.resonances_args, abctuple=(a, b, c),
                                   w1w2Condition=(self.w1w2Condition & selectionCond))      # a 2D np.array (Nomega1, Nomega2)
            mechavrg = mechavrg_pair[0]                                             # a 2D np.array (nmodes, nmodes)

            self.intensities_grid += self.mech_factors[index] / prefac_mech * mechavrg[a, b, c] * F * resonance2 / (-48.)
            # self.intensities_grid += addition

        # result = total_sum_mech / (-48.)

        # elapsed_time = time.time() - st
        # print('Compute time for one ab(c) in get_total_gamma_sum_mech:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        return self.intensities_grid

    def get_gamma_mech(self, a: int, b: int, selectionCond: np.ndarray = None, factor=False) -> dict:
        """

        """

        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic_Eh
        else:
            vib_ene_levels = self.all_states_Eh

        factors = {}
        resonances = {}

        # for index, (mech_func, mechavrg_pair) in enumerate(self.combofuns_tensors[1].items()):
        for idx, mechterm in enumerate(self.mechanical_terms):

            if factor:
                fac = 0.
                termavrg = self.mechanical_avrg[idx]

                for c in self.mode_indices:
                    prefac_mech = self.prefac_3d.T[a, b, c]
                    # mechavrg = mechavrg_pair[0]                            # a 2D np.array (nmodes, nmodes)
                    mechavrg = self.mech_avrg_tensors[idx]
                    # mechavrgF = mechavrg_pair[1]                           # a string of indices for F (cff)
                    # mechavrgF = self.mechanical_avrg[idx]
                    abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
                    ijk_indx = tuple([abc[j] for j in termavrg[-2]])
                    F = self.deriv_data['F_abc'][ijk_indx]                 # a number, from F tensor

                    freqDiff = [i.split(',') for i in mechterm[1]]
                    letters = ['a', 'b', 'c', 'zero']
                    dictabc = dict(zip(letters, (a, b, c) + tuple(['zero'])))

                    if ('zero',) not in vib_ene_levels:
                        vib_ene_levels[('zero',)] = 0.

                    w_fr11 = tuple(sorted([str(dictabc[i]) for i in freqDiff[0][0].split('+')], key=int))
                    if 'zero' not in freqDiff[0][1]:
                        w_fr21 = tuple(sorted([str(dictabc[i]) for i in freqDiff[0][1].split('+')], key=int))
                    else:
                        w_fr21 = tuple([freqDiff[0][1]])
                    w_fr12 = tuple(sorted([str(dictabc[i]) for i in freqDiff[1][0].split('+')], key=int))
                    if 'zero' not in freqDiff[1][1]:
                        w_fr22 = tuple(sorted([str(dictabc[i]) for i in freqDiff[1][1].split('+')], key=int))
                    else:
                        w_fr22 = tuple([freqDiff[1][1]])

                    t3 = vib_ene_levels[w_fr11] - vib_ene_levels[w_fr21]
                    t4 = vib_ene_levels[w_fr12] - vib_ene_levels[w_fr22]
                    sumfrac = (1 / t3 + 1 / t4)

                    fac += termavrg[-1] * sumfrac / prefac_mech * mechavrg[a, b, c] * F / (-48.)
                    # ref_combfac[c] = termavrg[-1] / prefac_mech * mechavrg[a, b, c] * F / (-48.)
                factors[idx] = fac
            else:
                if self.screenmodeswindow:
                    if (a, b) in self.inwindow[mechterm[0]]:
                        continue

                self.mechab = True
                # a 2D np.array (Nomega1, Nomega2)
                resonances[idx] = self.m_funcs[idx](allLevels_Eh=vib_ene_levels,
                                                    w_res_dict=self.resonances_args, abctuple=(a, b),
                                                    w1w2Condition=(self.w1w2Condition & selectionCond))

        if factor:
            return factors
        else:
            return resonances

    # old method
    def intensity_electrical(self, selectionCond: np.ndarray = None) -> (np.ndarray, dict):
        """
        Looping over a,b combinations - full sum of \\gamma^{[1,0]}
        """
        import time
        st = time.time()

        if selectionCond is None:
            selectionCond = ~np.zeros(self.w1w2Condition.shape, dtype=bool)

        count = 0

        for ind, i in enumerate(self.coords_ab):
        # for ab in combinations_with_permutations(self.mode_indices, 2):

            import time
            st_ab = time.time()

            self.intensities_grid = self.get_total_gamma_sum_el(i[0], i[1], selectionCond)

            if ind % 10 == 0:
                print(f'{ind}/{len(self.coords_ab)} modes combinations -- {ind*100/len(self.coords_ab)}%; time passed: {time.strftime("%H:%M:%S", time.gmtime(time.time() - st_ab))}')

        elapsed_time = time.time() - st
        print('Compute time of looping over ab combinations in intensity_electrical:',
              time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        print('Electrical anharmonicities are calculated. Count:', count)

        return self.intensities_grid

    # old method with a wasteful loop
    def intensity_mechanical(self, selectionCond: np.ndarray = None) -> (np.ndarray, dict):
        """
        Looping over a,b,c combinations - full sum of \\gamma^{[0,1]}
        """
        import time
        st = time.time()

        if selectionCond is None:
            selectionCond = ~np.zeros(self.w1w2Condition.shape, dtype=bool)

        if not self.electrical_terms:
            self.intensities_grid = np.zeros(self.shape2d, dtype='complex64')

        count = 0

        for ind, i in enumerate(self.coords_abc):
            import time
            st_abc = time.time()

            self.intensities_grid = self.get_total_gamma_sum_mech(i[0], i[1], i[2], selectionCond)

            if ind % 10 == 0:
                print(f'{ind}/{len(self.coords_abc)} modes combinations -- {ind*100/len(self.coords_abc)}%; time passed: {time.strftime("%H:%M:%S", time.gmtime(time.time() - st_abc))}')

        elapsed_time = time.time() - st
        print('Compute time of looping over abc combinations in intensity_mechanical:',
              time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        print('Mechanical anharmonicities are calculated. Count:', count)

        return self.intensities_grid

    def intensity_both(self, selectionCond: np.ndarray = None) -> np.ndarray:
        """
        Looping over a,b combinations - full sum of \\gamma^{[1,0]}
        """
        import time
        st = time.time()

        if selectionCond is None:
            selectionCond = ~np.zeros(self.w1w2Condition.shape, dtype=bool)

        count = 0
        skipped = 0
        numberofcombs = numcombperm(len(self.mode_indices), 2)

        # for ind, i in enumerate(self.coords_ab):
        for ab in combinations_with_permutations(self.mode_indices, 2):

            import time
            st_ab = time.time()
            a,b = ab
            # filter out the modes that are to be excluded - fixme? - to be more careful with derivatives
            if a in self.list2exclude or b in self.list2exclude:
                skipped+=1
                continue

            count+=1

            # electrical terms added; self.intensities_grid += for a,b inside (for each term)
            self.intensities_grid = self.get_total_gamma_sum_el(a, b, selectionCond)

            # mechanical terms here
            # identifying the resonances_args for selected terms
            resonances = self.get_gamma_mech(a, b, selectionCond, factor=False)
            for term in resonances:
                # self.intensities_grid += self.comb_facs[term]*resonances_args[term]
                self.intensities_grid += self.comb_fac_dict[(self.mechanical_terms[term],
                                                             self.mechanical_avrg[term])][a,b]*resonances[term]
                # self.intensities_grid += self.comb_fac_dict[term][a,b]*resonances_args[term]
                # print('checking', a, b, self.comb_fac_dict[(self.mechanical_terms[term],
                #                                       self.mechanical_avrg[term])][a,b],
                #       self.comb_facs[term],
                #       self.comb_fac_dict[(self.mechanical_terms[term],
                #                                       self.mechanical_avrg[term])][a,b]==self.comb_facs[term])

            if count % 10 == 0:
                print(f'{count}/{numberofcombs} modes combinations -- {count*100/numberofcombs}%; '
                      f'time passed: {time.strftime("%H:%M:%S", time.gmtime(time.time() - st_ab))}')

        elapsed_time = time.time() - st
        print('Compute time of looping over abc combinations in intensity_mechanical:',
              time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        print('skipped', skipped)
        return self.intensities_grid

    def generate_resonances_functions(self, subscripts, freqDiff=None) -> Callable:
        """
        Generates a python function for a term given by a formula (subscripts and freqDiff);
                varied argument of that function is abctuple (used in the loop over combinations of modes).
        subscripts - a tuple of strings from the formula; subscripts of omega energy levels in the resonance part;
                        e.g., ('a+b,a', 'zero,a')
        freqDiff - a tuple of strings from the formula; subscripts of omega energy levels in the freq. difference part;
                        e.g., ('a+b+c,0', 'c,a+b'); not None for mech. anharm.
        """
        # superscripts isn't formally passed down but it is used there??
        m1n1m2n2 = [i.split(',') for i in subscripts]
        if freqDiff is not None:
            freqDiff = [i.split(',') for i in freqDiff]

        # @profile
        def function(allLevels_Eh: dict, w_res_dict: dict[str:np.ndarray],
                     abctuple: tuple[int, int] | tuple[int, int, int],
                     w1w2Condition: np.ndarray[bool],
                     freqDiff: list = freqDiff) -> np.ndarray:
            """
            allLevels_Eh_c collects all vibrational energy levels in Hartree; e.g., [('1', '2')] - combination mode
            w_res_dict contains [-1, 2] and [-1] 2d arrays (in s-1)
            abctuple is a tuple of normal mode indices for which current iteration is evaluating resonance term
            """
            # todo: lorentzian shape cutoff

            letters = ['a', 'b', 'c', 'zero'] if len(abctuple) == 3 else ['a', 'b', 'zero']
            dictabc = dict(zip(letters, abctuple + tuple(['zero'])))
            allLevels_Eh_c = copy.deepcopy(allLevels_Eh)
            if ('zero',) not in allLevels_Eh_c:
                allLevels_Eh_c[('zero',)] = 0.

            if 'c' not in subscripts[0]:
                index_wmn = (abctuple[0], abctuple[1])
                # resonancesDF = pd.concat(self.dfs4terms_el, ignore_index=True)
                # resonance_w1 = resonancesDF[(resonancesDF['a'] == abctuple[0])
                                            # & (resonancesDF['b'] == abctuple[1])]['w_1']
                # resonance_w2 = resonancesDF[(resonancesDF['a'] == abctuple[0])
                                            # & (resonancesDF['b'] == abctuple[1])]['w_2']
            else:
                index_wmn = (abctuple[0], abctuple[2])

            t1 = self.w_mn_dict[subscripts[0]][index_wmn] + w_res_dict[(-1, 2)]  # - 1j * Gamma_hartree

            t2 = self.w_mn_dict[subscripts[1]][abctuple[0], abctuple[1]] + w_res_dict[(-1,)] #- 1j * Gamma_hartree

            if freqDiff is None:
                sumfrac = 1.

            else:
                if not self.mechab:
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

                    t3 = allLevels_Eh_c[w_fr11] - allLevels_Eh_c[w_fr21]
                    t4 = allLevels_Eh_c[w_fr12] - allLevels_Eh_c[w_fr22]

                    sumfrac = (1 / t3 + 1 / t4)

                else:
                    sumfrac = 1.

            product = t1 * t2

            result = np.where(w1w2Condition, sumfrac / product, 0.)

            return  result

        return function



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


