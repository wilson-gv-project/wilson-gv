import copy
import time
from datetime import timedelta
from typing import Callable

import numpy as np

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

        self.resonances_bank = {}

        self.Gamma = None
        self.Gamma_rc = None
        self.diagonal_margin_rc = None

        self.gammaCompsAll = get_AlphaBetaGammaDelta_indices(num_f=4)

        self.deriv_data = None
        self.corrected_levels = None

        self.mechab = False


    def getDerivedTermsEVV(self):
        """
        Currently available for selection EVV terms
        """
        # Terms in the expressions
        # derivatives:
        # 1. mu_Q, mu QQ, alpha_Q - electric dipole (1st and 2nd derivatives), polarizability (1st der.)
        # 2. mu_Q, alpha_QQ - electric dipole (1st der.), polarizability (2nd der.)
        # mu_Q, alpha_Q - for all 6 terms
        self.allterms_str = {0: ((('a+b,a', 'zero,a'), None), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',)))),
                             1: ((('b,a', 'zero,a'), None), (('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',)))),
                             2: ((('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc', 1.)),
                             3: ((('b,a', 'zero,a'), ('a+c,b', 'b+c,a')), (('mu_Q', ('a',)), ('alpha_Q', ('c',)), ('mu_Q', ('b',)), 'acb', 1.)),
                             4: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc', 0.5)),
                             5: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc', 0.5)),
                             6: ((('b,a', 'zero,a'), ('a,a+b', 'b,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc', -0.5)),
                             7: ((('b,a', 'zero,a'), ('b,a+b', 'a,zero')), (('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc', -0.5))}


    def load_data(self, parserObj, vpt2=False, vpt2settings=None):
        """
        Loading the data from a parser object/DataVault
            with the sources given to it

        anharmonic_type options:
            'VPT2'   - don't do_res, don't do_var
            'DVPT2'  - do_res, don't do_var
            'GVPT2'  - do_res, do_var

        default is vpt2settings = {'anharmonic_type': 'VPT2'}
        """
        # TODO - make it more flexible, give an option to supply files
        # parserObj = parser(self.input_data_info)

        if vpt2settings is None:
            vpt2settings = {'anharmonic_type': 'VPT2'}

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
        # (6, 3)  (6, 6, 3)  (6, 3, 3) (6, 6, 3, 3) (6, 6, 6) if nmodes = 6

        if vpt2:
            if parserObj.DD11 or parserObj.DD13 or parserObj.DD22:
                print("Warning: found Darling-Dennison resonances_args in data:")
                print(f"DD 1-1: {parserObj.DD11}")
                print(f"DD 2-2: {parserObj.DD22}")
                print(f"DD 1-3: {parserObj.DD13}")

            one = {i: self.all_states[i] for i in self.all_states if len(i) == 1}
            two = {i: self.all_states[i] for i in self.all_states if len(i) == 2}
            print('\nOriginal anharm corrected:')
            print(dict(sorted(one.items())))
            print(dict(sorted(two.items())), '\n')

            cff_cm_1 = parserObj.cubic_cm_1
            qff_cm_1 = parserObj.quartic_cm_1
            rot_c, cor_c = parserObj.rotational_constant, parserObj.coriolis_constant

            from .vpt2 import anharm_corr_energiesVPT2
            # corrected_levels : funds, over2q, combo2q, over3q, combo3q
            self.corrected_levels = anharm_corr_energiesVPT2(list(self.fundamentals_harmonic.values()),
                                                             cff_cm_1, qff_cm_1, rot_c, cor_c,
                                                             vpt2settings['anharmonic_type'])
            self.all_states_corr = {}
            for i in range(len(self.fundamentals)):
                self.all_states_corr[(str(i),)] = self.corrected_levels[0][i]

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

            self.all_states = copy.deepcopy(self.all_states_corr)
            one = {i: self.all_states[i] for i in self.all_states if len(i) == 1}
            two = {i: self.all_states[i] for i in self.all_states if len(i) == 2}
            print('\nOriginal anharm corrected:')
            print(dict(sorted(one.items())))
            print(dict(sorted(two.items())), '\n')

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
        self.selection = electrical_terms_selection + mechanical_terms_selection


        self.avrg_tensors_dict = {i: avrg_abc_tensor(self.allterms_str[i][1], self.deriv_data, self.gammaCompsAll)
                                      for i in self.selection}
        self.allfunc_dict = {i: self.generate_resonances_functions(self.allterms_str[i][0][0], self.allterms_str[i][0][1]) for i in self.selection}
        self.nmodes = len(self.fundamentals)


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
        st0 = time.time()

        if list2exclude is None:
            list2exclude = []

        # used in get_gamma_el
        self.screenmodeswindow = screenmodeswindow

        vib_ene_levels_harmonic = convNu2Ene(np.array([v for k,v in self.fundamentals_harmonic.items()]))

        self.prefac_2d = np.outer(vib_ene_levels_harmonic, vib_ene_levels_harmonic)
        self.prefac_3d = (vib_ene_levels_harmonic[:, np.newaxis, np.newaxis] *
                          vib_ene_levels_harmonic[np.newaxis, :, np.newaxis] *
                          vib_ene_levels_harmonic[np.newaxis, np.newaxis, :])

        self.resonancesTypes = [(-1, 2), (-1,)]
        self.axes = {1: self.w1_mesh_Eh, 2: self.w2_mesh_Eh}
        self.w1w2Condition = self.axes[2] - self.diagonal_margin_Eh > self.axes[1]
        # self.w1w2Condition = np.ones(self.w1_mesh_Eh.shape, dtype=bool)

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

        w_apbbma = np.zeros((self.nmodes_original, self.nmodes_original))
        w_bma = np.zeros((self.nmodes_original, self.nmodes_original))
        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            w_apbbma[a, b] = vib_ene_levels[tuple([str(el) for el in sorted([a, b])])] - vib_ene_levels[tuple([str(a)])]
            w_bma[a, b] = vib_ene_levels[tuple([str(b)])] - vib_ene_levels[tuple([str(a)])]

        za = np.array([-vib_ene_levels[tuple([str(k)])] for k in range(self.nmodes_original)])
        self.w_mn_dict = {'a+b,a': w_apbbma, 'b,a': w_bma, 'c,a': w_bma,
                          'zero,a': np.tile(za, self.nmodes_original).reshape(self.nmodes_original, -1).T}
        self.res_dict = {}

        w_apbbma_rc = np.zeros((self.nmodes_original, self.nmodes_original))
        w_bma_rc = np.zeros((self.nmodes_original, self.nmodes_original))
        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            w_apbbma_rc[a, b] = vib_ene_levels_rc[tuple([str(el) for el in sorted([a, b])])] - vib_ene_levels_rc[
                tuple([str(a)])]
            w_bma_rc[a, b] = vib_ene_levels_rc[tuple([str(b)])] - vib_ene_levels_rc[tuple([str(a)])]

        za_rc = np.array([-vib_ene_levels_rc[tuple([str(k)])] for k in range(self.nmodes_original)])
        za_rc = np.tile(za_rc, self.nmodes_original).reshape(self.nmodes_original, -1).T

        self.res_dict[('a+b,a', 'zero,a')] = []
        self.res_dict[('b,a', 'zero,a')] = []

        self.inwindow = {}
        self.inwindow[('a+b,a', 'zero,a')] = {}
        self.inwindow[('b,a', 'zero,a')] = {}
        # check if resonances are withing the big spectrum window (w1_mesh, w2_mesh)
        # will collect those outside the window (with margin)
        if self.screenmodeswindow:
            mw1, Mw1 = self.w1_mesh.min(), self.w1_mesh.max()
            mw2, Mw2 = self.w2_mesh.min(), self.w2_mesh.max()

        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            self.res_dict[('a+b,a', 'zero,a')].append(((-za_rc[a, b],
                                                       w_apbbma_rc[a, b] - za_rc[a, b]), (a, b)))
            self.res_dict[('b,a', 'zero,a')].append(((-za_rc[a, b],
                                                     w_bma_rc[a, b] - za_rc[a, b]), (a, b)))

            # will collect those outside the window (with margin)
            if self.screenmodeswindow:
                margin = 50.
                if not (mw1+margin < -za_rc[a, b] < Mw1-margin) and not (mw2+margin < w_apbbma_rc[a, b] - za_rc[a, b] < Mw2-margin):
                    self.inwindow[('a+b,a', 'zero,a')][(a,b)] = (-za_rc[a, b], w_apbbma_rc[a, b] - za_rc[a, b])
                if not (mw1+margin < -za_rc[a, b] < Mw1-margin) and not (mw2+margin < w_bma_rc[a, b] - za_rc[a, b] < Mw2-margin):
                    self.inwindow[('a+b,a', 'zero,a')][(a,b)] = (-za_rc[a, b], w_bma_rc[a, b] - za_rc[a, b])

        # collect resonances below diagonal (without margin)
        fromdiagonal = {('a+b,a', 'zero,a'): set(
                            [i for i in self.res_dict[('a+b,a', 'zero,a')] if i[0][0] >= i[0][1]]),
                        ('b,a', 'zero,a'): set(
                            [i for i in self.res_dict[('b,a', 'zero,a')] if i[0][0] >= i[0][1]])}
        # omega2>omega1 condition
        self.res_dict[('a+b,a', 'zero,a')] = set(
            [i for i in self.res_dict[('a+b,a', 'zero,a')] if i[0][0] < i[0][1] - self.diagonal_margin_rc])
        self.res_dict[('b,a', 'zero,a')] = set(
            [i for i in self.res_dict[('b,a', 'zero,a')] if i[0][0] < i[0][1] - self.diagonal_margin_rc])

        st = time.time()
        # setting up a dict for combined mech factors - for each selected mech term
        self.comb_fac_dict = {}
        for key in self.m_selected:
            self.comb_fac_dict[self.allterms_str[key]] = np.zeros((self.nmodes_original, self.nmodes_original))

        # computing combined mech factors - summed over c for each a,b
        for ab in combinations_with_permutations(self.mode_indices, 2):
            a, b = ab
            for key in self.m_selected:
                self.comb_fac_dict[self.allterms_str[key]][a,b] = self.compute_mech_factors(a, b)[key]
        # print(self.comb_fac_dict)
        elapsed_time = time.time() - st
        print('self.comb_fac_dict collected:',
              time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

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

        elapsed_time = time.time() - st0
        print('Precalculate full:',
              time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


    def locateOnBigGrid(self, seed, radius):
        """
        Find corner points of the small square
        """
        stepX = self.w1[1]-self.w1[0]
        stepY = self.w2[1]-self.w2[0]

        # indices of grid point closest to resonance point
        closestXind = round((seed[0][0]-np.min(self.w1))/stepX)
        closestYind = round((seed[0][1]-np.min(self.w2))/stepY)

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
        """
        Find all small squares around each resonance point
        """
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


    def compute_mech_factors(self, a: int, b: int):
        """
        Precalculate prefactor of mechanical terms - summation over c for each a,b
        """
        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic_Eh
        else:
            vib_ene_levels = self.all_states_Eh

        factors = {}
        for m_idx in self.m_selected:

            fac = 0.
            mechterm, termavrg = self.allterms_str[m_idx]
            for c in self.mode_indices:
                prefac_mech = self.prefac_3d[a, b, c]
                mechavrg = self.avrg_tensors_dict[m_idx]
                abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
                ijk_indx = tuple([abc[j] for j in termavrg[-2]])
                F = self.deriv_data['F_abc'][ijk_indx]

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
            factors[m_idx] = fac

        return factors


    def intensity_both(self, selectionCond: np.ndarray = None) -> np.ndarray:
        """
        Collects all the contributions to intensity.
        Loop over (a,b) modes combinations.
        """
        import time
        st = time.time()

        if selectionCond is None:
            selectionCond = np.ones(self.w1w2Condition.shape, dtype=bool)
        condition = (self.w1w2Condition & selectionCond)

        if self.vib_levels_harmonic:
            vib_ene_levels = self.all_states_harmonic_Eh
        else:
            vib_ene_levels = self.all_states_Eh

        count = 0
        numberofcombs = numcombperm(len(self.mode_indices), 2)

        for ab in combinations_with_permutations(self.mode_indices, 2):
            import time
            st_ab = time.time()
            a,b = ab
            count+=1

            self.resonances_bank = {}

            for termID in self.selection:
                res_formula, avrg_formula = self.allterms_str[termID]

                if res_formula[-1] is None:
                    factor = self.avrg_tensors_dict[termID][a, b] / self.prefac_2d[a, b] / 24.
                else:
                    factor = self.comb_fac_dict[self.allterms_str[termID]][a, b]

                if factor==0.:
                    continue

                if res_formula[0] not in self.resonances_bank:
                    self.resonances_bank[res_formula[0]] = self.allfunc_dict[termID](allLevels_Eh=vib_ene_levels,
                                                                                     w_res_dict=self.resonances_args,
                                                                                     abctuple=(a, b),
                                                                                     w1w2Condition=selectionCond)
                # self.intensities_grid += elavrg[a, b] * resonance / prefac_el / 24.
                self.intensities_grid += np.where(condition,
                                                  factor * self.resonances_bank[res_formula[0]], 0.)

            if count % 10 == 0:
                print(f'{count}/{numberofcombs} modes combinations -- {count*100/numberofcombs}%; '
                      f'time passed: {time.strftime("%H:%M:%S", time.gmtime(time.time() - st_ab))}',
                      f'time passed since start: {time.strftime("%H:%M:%S", time.gmtime(time.time() - st))}')

        elapsed_time = time.time() - st
        print('Compute time of looping over abc combinations in intensity_mechanical:',
              time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

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
            else:
                index_wmn = (abctuple[0], abctuple[2])

            t1 = self.w_mn_dict[subscripts[0]][index_wmn] + w_res_dict[(-1, 2)]  # - 1j * Gamma_hartree

            t2 = self.w_mn_dict[subscripts[1]][abctuple[0], abctuple[1]] + w_res_dict[(-1,)] #- 1j * Gamma_hartree

            if freqDiff is None:
                sumfrac = 1.

            else:
                if self.mechab:
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
                    # self.mechab = False

                else:
                    sumfrac = 1.

            # product = t1 * t2

            result = np.where(w1w2Condition, sumfrac / (t1 * t2), 0.)

            return  result

        return function

