#####################################################################################
##                                                                                 ##
##          File contains main code for 2DIR spectrum generation (images)          ##
##                                                                                 ##
#####################################################################################

import time

import numpy as np
np.set_printoptions(linewidth=100000)

from .callbacks2DIR import CFOURdata, VeloxChemdata, LSDaltondata, GaussianData

# todo 1: numerical differentiation for missing orders
# todo 2: harmonic frequencies - SpectroscPy or VeloxChem
# todo 3: anharmonic corrections to frequencies (cubic, quartic) - SpectroscPy
# todo 4: cartesian to normal mode basis transformation - SpectroscPy
# todo 5: orientational averaging
# todo 6: rendering

"""
propsData - cart2norm transformed tensors
avrgT - averaged tensor for alpha, beta, gamma, delta: float

gamma_abc = prefactor * sum_abc
sum_abc = prefac_abc * sum_of_terms
term_in_sum = avrgT * resonances (* fermi)

gamma_abc = prefactor * sum_of_terms_abc
term_in_sum = prefac_abc * avrgT * resonances (* fermi)
"""

# Terms in expressions
electrical_terms = [('a+b,a', 'zero,a'), ('b,a', 'zero,a') ]

# derivatives:
# 1. mu_Q, mu QQ, alpha_Q - electric dipole (1st and 2nd derivatives), polarizability (1st der.)
# 2. mu_Q, alpha_QQ - electric dipole (1st der.), polarizability (2nd der.)
electric_avrg = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))],
                 [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))] ]

mechanical_terms = [[('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')],
                    [('c,a', 'zero,a'), ('a+b,c', 'b+c,a')],
                    [('a+b,a', 'zero,a'), ('a,a+b', 'b,zero')],
                    [('b,a', 'zero,a'), ('b,a+b', 'a,zero')],
                    [('b,a', 'zero,a'), ('a,a+b', 'b,zero')],
                    [('b,a', 'zero,a'), ('b,a+b', 'a,zero')] ]

# derivatives:
# mu_Q, alpha_Q - for all 6 terms
mechanical_avrg = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc'] ]

def picks(pool, listofinds):
    return [pool[i] for i in listofinds]


def rec_cm2rec_s(cm_m1):
    from scipy import constants
    hartree2J = constants.physical_constants['hartree-joule relationship'][0]
    return cm_m1 * (100*constants.h*constants.c/hartree2J)

class SpectrumEVV:
    """
    SpectrumEVV class
    Attributes:
        w1, w2 - np.arrays of of frequencies
        w1_mesh, w2_mesh - grid of frequencies w1 and w2
        shape2d - shape of the grid
        fermirm

    """
    def __init__(self, w1, w2, data):

        # defines the grid of spectrum (pixels)
        self.w1_mesh, self.w2_mesh = np.meshgrid(w1, w2, indexing='ij')
        self.w1, self.w2 = np.array(w1), np.array(w2)
        self.shape2d = self.w1_mesh.shape
        self.data = data # dictionary with data source and type - inputs

        cfuncs = {'cfour': CFOURdata(data), 'vlx': VeloxChemdata(data),
                  'openrsp': LSDaltondata(data), 'gaussian': GaussianData(data)}
        self.callbacks = cfuncs[data['source']]

        got_funds = self.callbacks.getFundamentals()

        # dictionary; keys from 0 to (3Natoms-6)
        self.fundamentals = {str(k):v for k,v in got_funds[0].items()}
        self.fundamentals_harmonic = {str(k):v for k,v in got_funds[1].items()}

        # for non-zero fermi terms
        self.fermirm = 0.0001

        # margin for higher diagonal
        self.margin = 10.

        parsed_states = self.callbacks.getAllStates()

        self.all_states = {tuple(str(i) for i in k): v for k, v in parsed_states[0].items()}
        self.all_states_harm = {tuple(str(i) for i in k): v for k, v in parsed_states[1].items()}

        self.id = f'w1{min(self.w1)}_{max(self.w1)}w2{min(self.w2)}_{max(self.w2)}'

        self.deriv_data = self.getDerivs()

        self.gammaCompsAll = getting_abcgreek4avrg(num_f=4)


    # setting up the expressions for mechanical and electrical anharmonicities
    def addTerms(self, electrical_terms, mechanical_terms, el_avrg, mech_avrg):
        if electrical_terms is None and mechanical_terms is None and el_avrg is None and mech_avrg is None:
            # Terms in expressions
            electrical_terms_r = [('a+b,a', 'zero,a'), ('b,a', 'zero,a')]

            # derivatives:
            # 1. mu_Q, mu QQ, alpha_Q - electric dipole (1st and 2nd derivatives), polarizability (1st der.)
            # 2. mu_Q, alpha_QQ - electric dipole (1st der.), polarizability (2nd der.)
            electric_avrg_r = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))],
                             [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))]]

            mechanical_terms_r = [[('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')],
                                [('c,a', 'zero,a'), ('a+b,c', 'b+c,a')],
                                [('a+b,a', 'zero,a'), ('a,a+b', 'b,zero')],
                                [('b,a', 'zero,a'), ('b,a+b', 'a,zero')],
                                [('b,a', 'zero,a'), ('a,a+b', 'b,zero')],
                                [('b,a', 'zero,a'), ('b,a+b', 'a,zero')]]

            # derivatives:
            # mu_Q, alpha_Q - for all 6 terms
            mechanical_avrg_r = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                               [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                               [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc'],
                               [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc'],
                               [('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc'],
                               [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc']]

            ee, mm = [0, 1], [0, 1, 2, 3, 4, 5]
            electrical_terms, mechanical_terms, el_avrg, mech_avrg = picks(electrical_terms_r, ee), picks(mechanical_terms_r, mm), picks(electric_avrg_r, ee), picks(mechanical_avrg_r, mm)

        # here the functions of 2 frequencies
        self.electr_funs = [w_mn_prod(i, margin=self.margin) for i in electrical_terms]
        self.mech_funs = [w_mn_prod(*i) for i in mechanical_terms]
        self.electric_avrg = el_avrg
        self.mechanical_avrg = mech_avrg

        # pairing the terms with averaging in those terms
        self.combofuns = [dict(zip(self.electr_funs, self.electric_avrg)),
                          dict(zip(self.mech_funs, self.mechanical_avrg))]

        # setting up the combinations of states for the terms
        self.coords_ab = get_abc(2, len(self.fundamentals)) if electrical_terms is not None else []
        self.coords_abc = get_abc(3, len(self.fundamentals)) if mechanical_terms is not None else []


    # derivs from rsp_tensor file + MOLECULE.INP # fixme: new way is to run PyOpenrsp
    #  (mu_Q, mu_QQ, alpha_Q, alpha_QQ, F_abc)
    def getDerivs(self):

        if self.data['source'] == 'mock':
            # FIXME : the simplest model data (verification of 2dir implementation)

            aa = len(self.fundamentals)
            data = [np.zeros(i) for i in [(aa, 3), (aa, aa, 3), (aa, 3, 3), (aa, aa, 3, 3), (aa, aa, aa)]]
            K = -.1
            data[0][:, 2].fill(K)
            # print('(aa, 3)', data[0], '\nfs')
            data[1][:, :, 2].fill(K)
            # print('(aa, aa, 3)', data[1], '\nfs')

            data[2][:, 2, :].fill(K)
            data[2][:, :, 2].fill(K)
            # print('(aa, 3, 3)', data[2], '\nfs')

            data[3][:, :, 2, :].fill(K)
            data[3][:, :, :, 2].fill(K)

            data[-1].fill(K)

            return dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], data))

        if self.data['source'] == 'openrsp':

            self.callbacks.getTensors()

            # here transformation from cart to nm basis is happening
            dimlessFile = self.data['dimensionless']
            if dimlessFile is None:
                print('>>>>>   Using non-dimensionless normal coordinates')
            else:
                print('>>>>>   Using dimensionless normal coordinates')

            self.callbacks.tensors2NMbasis(dimlessFile)
            prOperators = dict(zip([tuple(['GEO', 'EL']), tuple(['GEO', 'GEO', 'EL']),
                                     tuple(['GEO', 'EL', 'EL']), tuple(['GEO', 'GEO', 'EL', 'EL']),
                                     tuple(['GEO', 'GEO', 'GEO'])],
                                   ['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc']))
            # quit()
            # print(props_list[0], props_list[0].hasTensor)

            finaldict = {}
            for pt in self.callbacks.props:
                ops = pt.operator
                finaldict[prOperators[tuple(ops)]] = pt.tensor

            # mu_Q, mu_QQ, alpha_Q, alpha_QQ, F_abc

            return finaldict

        elif self.data['source'] == 'pyorsp':
            # run 2dir pyopenrsp calculation and get necessary tensors

            from mock2D.frompyopenrsp import pyrsp_2dir

            return dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], pyrsp_2dir.props_list))

        elif self.data['source'] == 'cfour':
            # data is a list of np.arrays 'mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'
            firstder, secder = self.callbacks.getDipDers()
            data = [firstder, secder]
            # print('firstder, secder', firstder.shape, secder.shape)

            polder = self.callbacks.getPolarDers()
            data.append(polder[0])
            data.append(polder[1])
            # print('polder1, polder2', polder[0].shape, polder[1].shape)

            cubicmat = self.callbacks.getCFF()
            data.append(cubicmat)
            # print('cubicmat', cubicmat.shape)

            allpropsdict = dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], data))

            return allpropsdict

        elif self.data['source'] == 'gaussian':
            from scipy import constants
            # to go from amu to au mass unit (m_e)
            amc_au = constants.physical_constants['atomic mass constant'][0] / \
                     constants.physical_constants['atomic unit of mass'][0]

            # data is a list of np.arrays 'mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'
            firstder, secder = self.callbacks.getDipDers()
            data = [firstder / np.sqrt(amc_au), secder / amc_au]

            polder = self.callbacks.getPolarDers()
            data.append(polder[0] / np.sqrt(amc_au))
            data.append(polder[1] / amc_au)

            cubicmat = self.callbacks.getCFF()
            data.append(cubicmat / amc_au**1.5)

            allpropsdict = dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], data))

            return allpropsdict

        else:
            print("Invalid data source")

    def gamma_mn(self, Gamma, a, b, c=False):

        # if 'c' is not provided, compute electrical anharmonicity
        if type(c) == bool:

            total_sum_el = 0 #np.zeros(self.shape2d, dtype='complex128')

            # prefac_el = 1 / self.fundamentals_harmonic[str(a)] / self.fundamentals_harmonic[str(b)]
            prefac_el = 1 / rec_cm2rec_s(self.fundamentals_harmonic[str(a)]) / rec_cm2rec_s(self.fundamentals_harmonic[str(b)])
            collectionAvrg_el = {}
            # for el_func, elavrg in self.combofuns[0].items():
            for index, (el_func, elavrg) in enumerate(self.combofuns[0].items()):

                # average for given (a, b) for a given term
                # averg_el1 = avrg_abc(elavrg, self.deriv_data, [a, b], self.gammaCompsAll)
                if el_func in collectionAvrg_el:
                    averg_el1 = collectionAvrg_el[el_func]
                else:
                    collectionAvrg_el[el_func] = avrg_abc_tensor(elavrg, self.deriv_data, self.gammaCompsAll)
                    averg_el1 = collectionAvrg_el[el_func]

                # now it's a big tensor (a, b, (c)) for a given term
                # averg_el1 = avrg_abc(elavrg, self.deriv_data, self.gammaCompsAll)
                total_sum_el += prefac_el * averg_el1[a, b] * el_func(self.all_states, self.w1_mesh, self.w2_mesh,
                                                                Gamma, (a, b))

            return total_sum_el / 24.

        else:

            total_sum_mech = 0 #np.zeros(self.shape2d, dtype='complex128')

            # mechanical
            # prefac_mech = 1 / self.fundamentals_harmonic[str(a)] / self.fundamentals_harmonic[str(b)] / self.fundamentals_harmonic[str(c)]
            prefac_mech = 1 / rec_cm2rec_s(self.fundamentals_harmonic[str(a)]) / rec_cm2rec_s(self.fundamentals_harmonic[str(b)]) / rec_cm2rec_s(self.fundamentals_harmonic[str(c)])
            factors = {0: 1., 1: 1., 2: 0.5, 3: 0.5, 4: -0.5, 5: -0.5}
            collectionAvrg_mech = {}
            # for mech_func, mechavrg in self.combofuns[1].items():
            # for id, (mech_func, mechavrg) in enumerate(self.combofuns[1].items()):
            for index, (mech_func, mechavrg) in enumerate(self.combofuns[1].items()):

                # averg_mech1 = avrg_abc(mechavrg[:-1], self.deriv_data, [a, b, c], self.gammaCompsAll)
                if mech_func in collectionAvrg_mech:
                    averg_mech1 = collectionAvrg_mech[mech_func]
                else:
                    collectionAvrg_mech[mech_func] = avrg_abc_tensor(mechavrg[:-1], self.deriv_data, self.gammaCompsAll)
                    averg_mech1 = collectionAvrg_mech[mech_func]

                abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
                # print(mechavrg, c)
                # quit()
                indx = tuple([abc[j] for j in mechavrg[-1]])
                F = self.deriv_data['F_abc'][indx]

                total_sum_mech += factors[index] * prefac_mech * averg_mech1[a, b, c] * F * mech_func(self.all_states,
                                                                            self.w1_mesh, self.w2_mesh, Gamma,
                                                                            (a, b, c))

            return -total_sum_mech / 48.

    # def gamma_mn(self, Gamma, a, b, c=False):
    #     import numexpr as ne
    #     if type(c) == bool:
    #         total_sum_el = 0
    #         prefac_el = 1 / rec_cm2rec_s(self.fundamentals_harmonic[str(a)]) / rec_cm2rec_s(
    #             self.fundamentals_harmonic[str(b)])
    #         collectionAvrg_el = {}
    #         for index, (el_func, elavrg) in enumerate(self.combofuns[0].items()):
    #             if el_func in collectionAvrg_el:
    #                 averg_el1 = collectionAvrg_el[el_func]
    #             else:
    #                 collectionAvrg_el[el_func] = avrg_abc_tensor(elavrg, self.deriv_data, self.gammaCompsAll)
    #                 averg_el1 = collectionAvrg_el[el_func]
    #
    #             func_result = el_func(self.all_states, self.w1_mesh, self.w2_mesh, Gamma, (a, b))
    #
    #             # Debugging: print shapes
    #             print("Shapes - averg_el1:", averg_el1.shape, "func_result:", func_result.shape)
    #
    #             # Retrieve the value from averg_el1 for indices a, b
    #             value_ab = averg_el1[a, b]
    #
    #             # Using numexpr for the computation
    #             # Ensure func_result can be combined with a scalar value_ab
    #             total_sum_el = ne.evaluate("total_sum_el + prefac_el * value_ab * func_result")
    #
    #         return total_sum_el / 24.
    #
    #     else:
    #         total_sum_mech = 0
    #         prefac_mech = 1 / rec_cm2rec_s(self.fundamentals_harmonic[str(a)]) / rec_cm2rec_s(self.fundamentals_harmonic[str(b)]) / rec_cm2rec_s(self.fundamentals_harmonic[str(c)])
    #         factors = {0: 1., 1: 1., 2: 0.5, 3: 0.5, 4: -0.5, 5: -0.5}
    #         collectionAvrg_mech = {}
    #
    #         for index, (mech_func, mechavrg) in enumerate(self.combofuns[1].items()):
    #             if mech_func in collectionAvrg_mech:
    #                 averg_mech1 = collectionAvrg_mech[mech_func]
    #             else:
    #                 collectionAvrg_mech[mech_func] = avrg_abc_tensor(mechavrg[:-1], self.deriv_data, self.gammaCompsAll)
    #                 averg_mech1 = collectionAvrg_mech[mech_func]
    #
    #             abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
    #             indx = tuple([abc[j] for j in mechavrg[-1]])
    #             F = self.deriv_data['F_abc'][indx]
    #
    #             # Retrieve the specific value from averg_mech1 using indices a, b, c
    #             value_abc = averg_mech1[a, b, c]
    #
    #             # Calculate the contribution from this mechanical function
    #             contribution = factors[index] * prefac_mech * value_abc * F * mech_func(self.all_states, self.w1_mesh, self.w2_mesh, Gamma, (a, b, c))
    #             total_sum_mech += contribution
    #
    #         return -total_sum_mech / 48.

    def gamma_mn_tensors(self, Gamma, el=True, mech=True):
        nmodes = len(self.fundamentals)
        wstart = rec_cm2rec_s(np.array(list(self.fundamentals_harmonic.values()))) # shape (6,)

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
        w_abc = rec_cm2rec_s(w_abc)
        w_ab = rec_cm2rec_s(w_ab) # for omega_{a+b} frequencies
        w_a = rec_cm2rec_s(np.array(list(self.fundamentals.values()))) # for omega_a frequencies

        total_sum_el = 0.
        total_sum_mech = 0.

        if el:
            start_time1 = time.time()
            # Expand dimensions of w_a to shape (i, i, 1, 1) for broadcasting
            w_a_expanded = w_a[:, np.newaxis, np.newaxis, np.newaxis]
            w_b_expanded = w_a[np.newaxis, :, np.newaxis, np.newaxis]
            w_ab_expanded = w_ab[:, :, np.newaxis, np.newaxis]

            w1_expanded = self.w1_mesh[np.newaxis, np.newaxis, :, :]
            w2_expanded = self.w2_mesh[np.newaxis, np.newaxis, :, :]

            pref_Tab = 1. / np.einsum('i,j->ij', wstart, wstart) # shape (6, 6)
            pref_Tab_extended = pref_Tab[:, :, np.newaxis, np.newaxis]  # shape (6, 6, Nx, Ny)

            # for omega_{a+b} - omega_a + omega_1 - omega_2 - iGamma - shape (6, 6, Nx, Ny)
            # w_a_extended = w_a[:, np.newaxis, np.newaxis, np.newaxis]  # shape (6, 1, 1, 1)
            # w_ab_extended = w_ab[:, :, np.newaxis, np.newaxis]  # shape (6, 6, 1, 1)
            # w_abc_extended = w_abc[:, :, :, np.newaxis, np.newaxis]  # shape (6, 6, 6, 1, 1)
            # for omega_a in omega_{0,a}
            # w_a_prime_extended = w_a_prime[:, :, np.newaxis, np.newaxis]  # shape (6, 6, 1, 1)

            # for omega_{a+b} - omega_a + omega_1 - omega_2 - iGamma - shape (6, 6, Nx, Ny)
            # w1_extended_ab = self.w1_mesh[np.newaxis, np.newaxis, :, :]  # shape (1, 1, Nx, Ny)
            # w2_extended_ab = self.w2_mesh[np.newaxis, np.newaxis, :, :]  # shape (1, 1, Nx, Ny)
            #
            # w1_extended_abc = self.w1_mesh[np.newaxis, np.newaxis, np.newaxis, :, :]  # shape (1, 1, Nx, Ny)
            # w2_extended_abc = self.w2_mesh[np.newaxis, np.newaxis, np.newaxis, :, :]  # shape (1, 1, Nx, Ny)

            #
            result_11 = 1. / (w_ab_expanded - w_a_expanded + w1_expanded - w2_expanded - 1j * Gamma) # shape (6, 6, Nx, Ny)
            result_12 = 1. / ((-1) * w_a_expanded + w1_expanded - 1j * Gamma) # shape (6, 6, Nx, Ny)
            avg1 = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))]
            averg_el1 = avrg_abc_tensor(avg1, self.deriv_data, self.gammaCompsAll)

            result_21 = 1. / (w_b_expanded - w_a_expanded + w1_expanded - w2_expanded - 1j * Gamma)
            # result_22 = copy.deepcopy(result_12)
            result_22 = result_12
            avg2 = [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))]
            averg_el2 = avrg_abc_tensor(avg2, self.deriv_data, self.gammaCompsAll)
            print(result_21.shape, result_11.shape)
            sum_el1 = (result_11*result_12).sum(axis=(0, 1)) # resulting shape (Nx, Ny)
            sum_el2 = (result_21*result_22).sum(axis=(0, 1)) # resulting shape (Nx, Ny)
            # print(averg_el1)
            # print(sum_el1)
            print('=====================sum_el1')
            # print(sum_el1)
            print(np.nonzero(sum_el1))

            print('=====================sum_el2')
            # print(sum_el2)
            print(np.nonzero(sum_el2))

            total_sum_el += (averg_el1 * result_11 * result_12 + averg_el2 * result_21 * result_22) * pref_Tab_extended
            total_sum_el = total_sum_el.sum(axis=(0, 1))
            # print('total_sum_el += (averg_el1 * sum_el1 + averg_el2 * sum_el2) * pref_Tab_extended / 24.')
            # print shapes of the terms of total eum el
            # print(f'averg_el1 {averg_el1.shape}')
            # print(f'sum_el1 {sum_el1.shape}')
            # print(f'averg_el2 {averg_el2.shape}')
            # print(f'sum_el2 {sum_el2.shape}')
            # print(f'pref_Tab_extended {pref_Tab_extended.shape}')
            # print(f'result_11 {result_11.shape}')
            # print(f'result_12 {result_12.shape}')
            # print(f'total_sum_el {total_sum_el.shape}')
            # quit()


            # print(f'\nresult11 {result_11.shape}')
            # print(f'\nresult12 {result_12.shape}')
            # print(f'pref_Tab {pref_Tab.shape}')
            # # print(f'sum_result_el {sum_result_el.shape}')
            # print(f'w1 mesh {self.w1_mesh.shape}')
            # print(f'w2 mesh {self.w2_mesh.shape}')
            # print(len(self.w1), len(self.w2))
            # print(f'w_a_prime_extended {w_a_prime_extended.shape}')
            # print(f'w_a_prime_extended \n{w_a_prime_extended}')

            # r11 = result_11[0, 2, 0, 0]
            # check11 = 1./(w_ab[0, 2] - w_a[0] + self.w1_mesh[0, 0] - self.w2_mesh[0, 0] - 1j * Gamma)
            # print(f'r11 {r11}')
            # print(f'check11 {check11}')
            # print('-------')
            #
            # r12 = result_12[0, 2, 0, 0]
            # check12 = 1./((-1) * w_a_prime[0, 2] + self.w1_mesh[0, 0] - 1j * Gamma)
            # print(f'r12 {r12}')
            # print(f'check12 {check12}')
            #
            # r11 = result_11[2, 4, 0, 0]
            # check11 = 1./(w_ab[2, 4] - w_a[0] + self.w1_mesh[0, 0] - self.w2_mesh[0, 0] - 1j * Gamma)
            # print(f'r11 {r11}')
            # print(f'check11 {check11}')
            # print('-------')
            #
            # r12 = result_12[2, 4, 0, 0]
            # check12 = 1./((-1) * w_a_prime[2, 4] + self.w1_mesh[0, 0] - 1j * Gamma)
            # print(f'r12 {r12}')
            # print(f'check12 {check12}')

            # for el_func, elavrg in self.combofuns[0].items():
            #     averg_el1 = avrg_abc_tensor(elavrg, self.deriv_data, self.gammaCompsAll)
            #     print(f'averg_el1 {averg_el1.shape}')
            #     quit()
            #     term_result_el = pref_Tab * averg_el1 * sum_result_el
            #     print(f'term_result_el {term_result_el.shape}')
            #     total_sum_el += term_result_el
            # # return total_sum_el / 24.
            end_time1 = time.time()
            execution_time1 = end_time1 - start_time1
            print(f"Execution time - total_sum_el: {execution_time1} seconds")
        if mech:
            start_time2 = time.time()
            # Expand dimensions of w_a to shape (i, i, i, 1, 1) for broadcasting
            w_a_expanded = w_a[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
            w_b_expanded = w_a[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
            w_c_expanded = w_a[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

            w_ab_expanded = w_ab[:, :, np.newaxis, np.newaxis, np.newaxis]
            w_bc_expanded = w_ab[np.newaxis, :, :, np.newaxis, np.newaxis]
            w_abc_expanded = w_abc[:, :, :, np.newaxis, np.newaxis]

            w1_expanded = self.w1_mesh[np.newaxis, np.newaxis, np.newaxis, :, :]
            w2_expanded = self.w2_mesh[np.newaxis, np.newaxis, np.newaxis, :, :]

            pref_Tabc = 1. /np.einsum('i,j,k->ijk', wstart, wstart, wstart)
            pref_Tabc_extended = pref_Tabc[:, :, :, np.newaxis, np.newaxis]  # shape (6, 6, 6, Nx, Ny)

            # print('pref_Tabc', pref_Tabc.shape)

            result_31 = 1. / (w_ab_expanded - w_a_expanded + w1_expanded - w2_expanded - 1j * Gamma) # shape (6, 6, Nx, Ny)
            result_32 = 1. / ((-1) * w_a_expanded + w1_expanded - 1j * Gamma) # shape (6, 6, Nx, Ny)
            result_33 = 1. / w_abc_expanded
            result_34 = 1. / (w_c_expanded - w_ab_expanded)
            avg3 = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc']
            averg_mech3 = avrg_abc_tensor(avg3[:-1], self.deriv_data, self.gammaCompsAll)
            F_abc = self.deriv_data['F_abc']
            F_abc_extended = F_abc[:, :, :, np.newaxis, np.newaxis]  # shape (6, 6, Nx, Ny)

            result_41 = 1. / (w_c_expanded - w_a_expanded + w1_expanded - w2_expanded - 1j * Gamma)
            # result_42 = copy.deepcopy(result_32)
            result_42 = result_32
            result_43 = 1. / (w_ab_expanded - w_c_expanded)
            result_44 = 1. / (w_bc_expanded - w_a_expanded)
            # averg_mech4 = copy.deepcopy(averg_mech3)
            averg_mech4 = averg_mech3

            # result_51 = copy.deepcopy(result_31)
            result_51 = result_31

            # result_52 = copy.deepcopy(result_32)
            result_52 = result_32
            result_53 = 1. / (w_a_expanded - w_ab_expanded)
            result_54 = 1. / w_b_expanded
            avg5 = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc']
            averg_mech5 = avrg_abc_tensor(avg5[:-1], self.deriv_data, self.gammaCompsAll)
            F_ijj = np.zeros_like(F_abc)
            i, j = np.arange(F_abc.shape[0]), np.arange(F_abc.shape[0])
            I, J = np.meshgrid(i, j, indexing='ij')
            F_ijj[:, J, J] = F_abc[:, J, J]
            F_ijj_extended = F_ijj[:, :, :, np.newaxis, np.newaxis]  # shape (6, 6, Nx, Ny)

            # result_61 = copy.deepcopy(result_31)
            result_61 = result_31
            # result_62 = copy.deepcopy(result_32)
            result_62 = result_32
            result_63 =  1. / (w_b_expanded - w_ab_expanded)
            result_64 = 1. / w_a_expanded
            avg6 = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc']
            averg_mech6 = avrg_abc_tensor(avg6[:-1], self.deriv_data, self.gammaCompsAll)

            result_71 = 1. / (w_b_expanded - w_a_expanded + w1_expanded - w2_expanded - 1j * Gamma)
            # result_72 = copy.deepcopy(result_32)
            result_72 = result_32
            # result_73 = copy.deepcopy(result_53)
            result_73 = result_53
            # result_74 = copy.deepcopy(result_54)
            result_74 = result_54
            avg7 = [('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc']
            averg_mech7 = avrg_abc_tensor(avg7[:-1], self.deriv_data, self.gammaCompsAll)

            # result_81 = copy.deepcopy(result_71)
            result_81 = result_71
            # result_82 = copy.deepcopy(result_32)
            result_82 = result_32
            # result_83 = copy.deepcopy(result_63)
            result_83 = result_63
            # result_84 = copy.deepcopy(result_64)
            result_84 = result_64
            # averg_mech8 = copy.deepcopy(averg_mech6)
            averg_mech8 = averg_mech6

            sum_el3 = result_31*result_32*(result_33+result_34)*F_abc_extended*averg_mech3 # resulting shape (Nx, Ny)
            sum_el4 = result_41*result_42*(result_43+result_44)*F_abc_extended*averg_mech4 # resulting shape (Nx, Ny)
            sum_el5 = result_51*result_52*(result_53+result_54)*F_ijj_extended*averg_mech5 # resulting shape (Nx, Ny)
            sum_el6 = result_61*result_62*(result_63+result_64)*F_ijj_extended*averg_mech6 # resulting shape (Nx, Ny)
            sum_el7 = result_71*result_72*(result_73+result_74)*F_ijj_extended*averg_mech7 # resulting shape (Nx, Ny)
            sum_el8 = result_81*result_82*(result_83+result_84)*F_ijj_extended*averg_mech8 # resulting shape (Nx, Ny)
            total_sum_mech += -(sum_el3 + sum_el4 + 0.5*sum_el5 + 0.5*sum_el6 - 0.5*sum_el7 - 0.5*sum_el8) * pref_Tabc_extended
            total_sum_mech = 1./48. * total_sum_mech.sum(axis=(0, 1, 2))
            print('\n=====================sum_el3', sum_el3.shape)
            print(np.nonzero(sum_el3))
            print('\n=====================sum_el4', sum_el4.shape)
            print(np.nonzero(sum_el4))
            # print(f'\nresult_41 {result_41.shape}\n', np.where(np.isinf(result_41)))
            # print(f'result_41 {result_41.shape}\n', result_41[np.where(np.isinf(result_41))])
            # print(result_41)
            print('\n=====================sum_el5', sum_el5.shape)
            print(np.nonzero(sum_el5))
            print('\n=====================sum_el6', sum_el6.shape)
            print(np.nonzero(sum_el6))
            print('\n=====================sum_el7', sum_el7.shape)
            # print(sum_el7)
            print(np.nonzero(sum_el7))

            # print(f'averg_mech7 {averg_mech7.shape}\n', np.nonzero(averg_mech7))
            # print(f'F_ijj_extended {F_ijj_extended.shape}\n', np.nonzero(F_ijj_extended))
            # print(F_ijj_extended[np.nonzero(F_ijj_extended)])
            # print(f'\nresult_71 {result_71.shape}\n', np.where(np.isinf(result_71)))
            # print(f'result_71 {result_71.shape}\n', result_71[np.where(np.isinf(result_71))])
            # print(f'result_72 {result_72.shape}\n', np.where(np.isinf(result_72)))
            # print(f'result_73 {result_73.shape}\n', np.where(np.isinf(result_73)))
            # print(f'result_74 {result_74.shape}\n', np.where(np.isinf(result_74)))

            print('\n=====================sum_el8', sum_el8.shape)
            print(np.nonzero(sum_el8))

            # print(sum_el8)
            # print(f'averg_mech7 {averg_mech7.shape}\n', np.nonzero(averg_mech7))
            # print(f'F_ijj_extended {F_ijj_extended.shape}\n', np.nonzero(F_ijj_extended))
            # print(F_ijj_extended[np.nonzero(F_ijj_extended)])
            print(f'\nresult_81 {result_81.shape}\n', np.where(np.isinf(result_71)))
            print(f'result_82 {result_82.shape}\n', np.where(np.isinf(result_72)))
            print(f'result_83 {result_83.shape}\n', np.where(np.isinf(result_73)))
            print(f'result_84 {result_84.shape}\n', np.where(np.isinf(result_74)))

            end_time2 = time.time()
            execution_time2 = end_time2 - start_time2
            print(f"Execution time - total_sum_mech: {execution_time2} seconds")
            # quit()
        return total_sum_el+total_sum_mech


    def intensity(self, Gamma, savedict, el=True, mech=True, printdata=False):

        Qab, Qabc = self.coords_ab, self.coords_abc
        # print('Qab', Qab)
        # print('Qabc', Qabc)
        # quit()
        Z = 0
        Qab_contrib_dict = {}
        Qabc_contrib_dict = {}

        if el:
            start_time = time.time()
            elall = np.zeros(self.shape2d, dtype='complex128')
            for i in Qab:
                print('hello', i)
                contrib_ab = self.gamma_mn(Gamma, i[0], i[1])
                Qab_contrib_dict[tuple(i)] = contrib_ab
                elall += contrib_ab
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time - electrical: {execution_time} seconds")
            print('Electrical anharmonicities are calculated')
            Z += elall

        if mech:
            start_time = time.time()
            mechall = np.zeros(self.shape2d, dtype='complex128')
            for i in Qabc:
                contrib_abc = self.gamma_mn(Gamma, i[0], i[1], i[2])
                Qabc_contrib_dict[tuple(i)] = contrib_abc
                mechall += contrib_abc
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"\nExecution time - mechanical: {execution_time} seconds")
            print('Mechanical anharmonicities are calculated')
            Z += mechall

        if printdata:
            np.set_printoptions(linewidth=250, suppress=True, precision=10)
            np.set_printoptions(threshold=np.inf)

            import sys
            import io
            output = io.StringIO()
            sys.stdout = output
            print("\nGrid:")

            array_3d = np.array([self.w1_mesh,self.w2_mesh]).T
            print(array_3d.shape)
            alist = [np.array2string(i).splitlines() for i in array_3d]
            print('\n'.join(['\t'.join(k) for k in zip(*alist)]))
            np.set_printoptions(linewidth=250, suppress=False, precision=10)

            print("\nMechanical  :") if mech else None
            print(mechall) if mech else None
            print("\nElectrical  :") if el else None
            print(elall) if el else None

            print("\nMechanical np.log10(mechall) :") if mech else None
            print(np.log10(mechall)) if mech else None
            print("\nElectrical np.log10(elall)   :") if el else None
            print(np.log10(elall)) if el else None
            print("\nnp.log10(elall*mechall)      :") if el and mech else None
            print(np.log10(elall*mechall)) if el and mech else None

            # print('\nnp.log10(elall)/np.log10(mechall):')
            # print(np.log10(elall)/np.log10(mechall))
            # print(np.mean((np.log10(elall)/np.log10(mechall)).flatten()))
            # print("\nTotal (abs(mechall)+abs(elall))** 2:")
            # print((abs(mechall)+abs(elall))** 2)

            print("\nTotal abs(Z)** 2:")
            print(abs(Z)** 2)

            print("\nTotal np.log10(abs(Z)** 2):")
            print(np.log10(abs(Z)** 2))

            sys.stdout = sys.__stdout__
            array_str = output.getvalue()

            with open(f'./anharmonicities_Gamma{Gamma}_({self.w1[0]}_{self.w1[-1]}_{self.w1[1]-self.w1[0]})({self.w2[0]}_{self.w2[-1]}_{self.w2[1]-self.w2[0]}).txt', 'w') as f:
                import os
                f.write(f"""Generated with:
                getcwd:        {os.getcwd()}
                __file__:      {__file__}
                sys.argv:      {sys.argv[0]}\n\n""")
                f.write(array_str)

        key = self.id+f'_gamma{Gamma}'
        if key not in savedict:
            savedict[key] = {}

        if mech: savedict[key]['mechanical'] = mechall
        if el: savedict[key]['electrical'] = elall
        savedict[key]['Qab_contrib_dict'] = Qab_contrib_dict
        savedict[key]['Qabc_contrib_dict'] = Qabc_contrib_dict

        return Z, savedict

    def plot2D(self, figname, source, w1mw2=False, style='contour', Gamma=0.99):
        c0 = time.process_time()

        # plt.ion()
        # matplotlib.use('TkAgg')
        # matplotlib.use('Agg')
        # matplotlib.use('QtAgg')
        # matplotlib.rcParams['backend'] = 'QtAgg'
        import matplotlib.pyplot as plt
        from matplotlib import colors
        def custom_format_coord(x, y):
            return f'x = {x:.2f}\n  y = {y:.2f}'  # Separate x and y on different lines

        # PLOTTING
        if style == 'surface':
            ax = plt.figure(figsize=(10, 8)).add_subplot(projection='3d')
        else:
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 12)
        # Set the custom format using Axes.format_coord
        ax.format_coord = custom_format_coord
        # points
        Z = self.totInt(style, source, Gamma)
        # Z_positive = abs(Z) ** 2

        print('Z are calculated')

        if style == 'surface' or style == 'contour':
            X, Y = self.w1_mesh, self.w2_mesh

        else:
            # scatter plot
            X, Y = self.w1, self.w2
        ax.set_xlabel('w1', fontsize=18)

        if w1mw2:
            y = -(X - Y)
            ax.set_ylabel('w2-w1', fontsize=18)
            # xlim = 4.
            # ax.set_xlim([xlim, max(list(self.fundamentals.values()))+3.])
            # ylim = 4.
            # ax.set_ylim([ylim, max(y.flatten())])

        else:
            y = Y
            ax.set_ylabel('w2', fontsize=18)

        positions = np.vstack([X.ravel(), y.ravel()])
        # print('positions', positions)
        # print(len(positions[0]))

        if style == 'surface':
            ax.plot_surface(X, y, abs(Z) ** 2, cmap='brg')

        elif style == 'contour':
            # Define the number of levels you want
            num_levels = 25

            # Since the minimum value is 0.0, we need to start from a small positive number
            # The maximum value is the maximum of abs(Z) squared
            min_value = np.min(abs(Z)[abs(Z) > 0]) ** 2 if np.any(abs(Z) > 0) else 1e-30
            max_value = np.max(abs(Z) ** 2)

            # Generate logarithmically spaced levels between the min and max values
            levels = np.logspace(np.log10(min_value), np.log10(max_value), num_levels)

            # Create the contour plot with the specified levels
            cp = ax.contourf(X, y, abs(Z) ** 2, levels=levels, norm=colors.LogNorm(vmin=min_value, vmax=max_value),
                             cmap='ocean')

            # cp = ax.contourf(X, y, abs(Z) ** 2, 8, cmap='magma')
            # cp = ax.contour(X, y, abs(Z) ** 2, 8, cmap='magma')
            # cp = ax.scatter(X, y, color="green")
            print(X.size, y.size, 'X.y size')
            print(self.w1.size, self.w2.size)
            # brg, magma
            # cp = ax.scatter(X, y, c=abs(Z) ** 2, cmap='magma') # , norm=colors.LogNorm()
            # positions = np.vstack([X.ravel(), y.ravel()])
            # print('positions', positions)

        elif style == 'scatter':
            print(X.size, y.size, 'X.y size')
            print('just before ax.scatter')

            cp = ax.scatter(X, y, c=abs(Z) ** 2, norm=colors.LogNorm(), cmap='brg')

            # cp = ax.scatter(X, y, c=abs(Z) ** 2, norm=colors.LogNorm(), cmap='magma')
            # cp = ax.scatter(X, y, c=abs(Z) ** 2, cmap='brg')

            if len(positions[0]) < 100:
                for i in range(len(positions[0])):
                    # print(f'({X[i]}, {y[i]})')
                    ax.text(positions[0][i], positions[1][i],
                            f'({positions[0][i]}, {positions[1][i]})', fontsize=9)
                    # ax.annotate(f'({X[i]}, {y[i]})', (X[i], y[i]))

        cbar = fig.colorbar(cp, ax=ax)
        tick_values = np.logspace(np.log10(Z.min()), np.log10(Z.max()), num=5)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f"{tick:.2f}" for tick in tick_values])

        # xlabels = ['%i' % i for i in np.linspace(min(X.flatten()), max(X.flatten()), 25)]
        numticks = 20
        xlabels = ['%i' % i for i in np.linspace(min(X.flatten()), max(X.flatten()), numticks)]

        ax.set_xticklabels(xlabels, rotation=45)
        # if w1mw2:
        #     ax.set_xticks(np.linspace(xlim, max(list(self.fundamentals.values()))+3., 25))
        #     ax.set_yticks(np.linspace(ylim, max(y.flatten()), 45))
        # else:
        # ax.set_xticks(np.linspace(min(X.flatten()), max(X.flatten()), 25))
        ax.set_xticks(np.linspace(min(X.flatten()), max(X.flatten()), numticks))
        # ax.set_yticks(np.linspace(min(y.flatten()), max(y.flatten()), 45))
        ax.set_yticks(np.linspace(min(y.flatten()), max(y.flatten()), numticks))

        # lines
        # print('before if len(positions[0]) > 100 and w1mw2')
        up = 40.
        # if len(positions[0]) > 100 and w1mw2:
        if w1mw2:
            color = 'k'
        else:
            color = 'w'
        # vertical lines
        # for pp in list(self.fundamentals.values()):
        #     plt.plot((pp, pp), (min(y.flatten()), max(y.flatten())), 'r-', linewidth=0.3)
        #     if pp<2700.:
        #         ax.text(pp+15.0, min(y.flatten())+up, f'{pp}', fontsize=12, color=color)
        #         up+=185.
        #     else:
        #         ax.text(pp+25.0, max(y.flatten())-up, f'{pp}', fontsize=12, color=color)
        #         up += 185.
        # horizontal lines
        # if w1mw2:
        #     side = 83.
        #     for dd in list(self.fundamentals.values()):
        #         plt.plot((min(X.flatten()), max(X.flatten())), (dd, dd), 'k-', linewidth=0.3)
        #         ax.text(min(X.flatten()) + side, dd + 1.0, f'{dd}', fontsize=12, color='w')
        #         side+=150.
        #     print('just before plt.plot')

            if w1mw2:
                plt.plot((min(X.flatten()), max(X.flatten())), (0., 0.), 'y-', linewidth=0.8)
            # else:
                # plt.plot((min(X.flatten()), max(X.flatten())), (min(X.flatten()), max(X.flatten())), 'y-', linewidth=0.8)
            # print('before plt.tight_layout()')
        plt.tight_layout()
        # matplotlib.pyplot.show()
        # % matplot plt

        plt.grid(False)  # Turn off gridlines
        # plt.minorticks_off()  # Turn off minor ticks

        dpi_value = 600  # For example, 300 dpi is a good resolution for print quality
        plt.savefig(f'./pics/{figname}_{Gamma}_sp8.svg', dpi=dpi_value)

        c1 = time.process_time()
        print('plot2D time', c1-c0)


        import os
        # Calculate the squared absolute value of Z
        Z_squared_abs = abs(Z) ** 2

        # Create a meshgrid if you haven't already
        X_grid, y_grid = X, y

        # Print the current working directory
        print(f"Current working directory: {os.getcwd()}")

        # Prepare the data to be printed
        # Flatten the arrays and stack them column-wise
        data_to_print = np.column_stack((X_grid.flatten(), y_grid.flatten(), Z_squared_abs.flatten()))

        # Sort the data by X and then by y
        # np.lexsort() uses the last key as the primary sort key, so we pass y first and then X
        sorted_indices = np.lexsort((data_to_print[:, 1], data_to_print[:, 0]))
        sorted_data = data_to_print[sorted_indices]

        # Define the filenames
        meshgrid_filename = f'./pics/meshgrid_data_{figname}_{Gamma}_sp8.txt'
        z_squared_abs_filename = f'./pics/z_squared_abs_data_{figname}_{Gamma}_sp8.txt'

        # Open a file to write the tuple of 3 values
        try:
            with open(meshgrid_filename, 'w') as f:
                # Write the header
                f.write("X, y, abs(Z)^2\n")
                # Write the sorted data
                for row in sorted_data:
                    f.write(f"{row[0]}, {row[1]}, {row[2]}\n")
            print(f"Data has been printed to '{meshgrid_filename}'")
        except IOError as e:
            print(f"Error writing to file {meshgrid_filename}: {e}")

        # Open a separate file to write the array of only abs(Z) ** 2 values
        # Since we only care about Z values here, we can sort just the Z array
        sorted_Z = Z_squared_abs.flatten()[sorted_indices]

        try:
            with open(z_squared_abs_filename, 'w') as f:
                # Write the header
                f.write("abs(Z)^2\n")
                # Write the sorted data
                for value in sorted_Z:
                    f.write(f"{value}   {np.log10(value)}\n")
            print(f"Data has been printed to '{z_squared_abs_filename}'")
        except IOError as e:
            print(f"Error writing to file {z_squared_abs_filename}: {e}")

        # Calculate the order of magnitude of the Z values
        orders_of_magnitude = np.log10(sorted_Z)
        print(orders_of_magnitude)
        # Define the bins for the histogram
        # For example, if you want bins from -20 to 0 (inclusive) in steps of 1
        bins = np.arange(-35, -9, 1)  # Adjust the range and step as needed

        # Calculate the histogram
        hist, bin_edges = np.histogram(orders_of_magnitude, bins=bins)
        print(hist)
        tot = 0.
        # Print the number of occurrences for each order of magnitude
        for i in range(len(bins) - 1):
            tot += hist[i]*100/sum(hist)
            print(f"Order of magnitude range [{bins[i]}, {bins[i + 1]}): {hist[i]} occurrences, "
                  f"{hist[i]*100/sum(hist)} %, total now {tot}")
        return abs(Z) ** 2, fig

    def print2file(self, figname, w1mw2, Gamma, step):

        Z, savedict = self.intensity(Gamma, {})
        X_grid, y_grid = self.w1_mesh, self.w2_mesh
        if w1mw2:
            y = -(X_grid - y_grid)
        else:
            y = y_grid

        # Calculate the squared absolute value of Z
        Z_squared_abs = abs(Z) ** 2

        # Flatten the arrays and stack them column-wise
        data_to_print = np.column_stack((X_grid.flatten(), y.flatten(), Z_squared_abs.flatten()))

        # Sort the data by X and then by y
        sorted_indices = np.lexsort((data_to_print[:, 1], data_to_print[:, 0]))
        sorted_data = data_to_print[sorted_indices]

        # Define the filenames
        meshgrid_filename = f'./picsnew/meshgrid_data_{figname}_{Gamma}_step{step}.txt'
        # z_squared_abs_filename = f'./pics/z_squared_abs_data_{figname}_{Gamma}st8.txt'

        # Open a file to write the tuple of 3 values
        try:
            with open(meshgrid_filename, 'w') as f:
                # Write the header
                f.write(f"X, y, abs(Z)^2, 2d shape {Z_squared_abs.shape}\n")
                # Write the sorted data
                for row in sorted_data:
                    f.write(f"{row[0]}, {row[1]}, {row[2]}\n")
            print(f"Data has been printed to '{meshgrid_filename}'")
        except IOError as e:
            print(f"Error writing to file {meshgrid_filename}: {e}")

    def plot2Dmatplotlib(self, Z, w1mw2, name, dpi=500, contour_levels=100, log10=True):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        plt.rcParams['path.simplify'] = True
        plt.rcParams['agg.path.chunksize'] = 10000
        X, Y = self.w1_mesh, self.w2_mesh
        if w1mw2:
            y = -(X - Y)
            ystr = 'w2-w1'
            ystr_mesh = ystr+'_'
        else:
            y = Y
            ystr = 'w2'
            ystr_mesh = ystr+'_'
        Z_positive = abs(Z) ** 2
        if log10:
            df = {'w1_mesh': X, ystr_mesh: y, 'values_mesh': np.log10(Z_positive)}
        else:
            df = {'w1_mesh': X, ystr_mesh: y, 'values_mesh': Z_positive}
        plt.figure(figsize=(12, 11))

        start_time = time.time()
        cont = plt.contourf(df['w1_mesh'], df[ystr_mesh], df['values_mesh'], levels=contour_levels, cmap='viridis')
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time - plt.contourf: {execution_time} seconds")

        # This is the fix for the white lines between contour levels
        for c in cont.collections:
            c.set_edgecolor("face")
        plt.colorbar()  # Add a colorbar to show the Z scale
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        xs = df['w1_mesh'][0], df['w1_mesh'][-1]
        ys = df[ystr_mesh][0], df[ystr_mesh][-1]
        plt.title(f'plot2Dmatplotlib().\ndpi={dpi} contour_levels={contour_levels} x{xs[0][0]}..{xs[-1][-1]} y{ys[0][0]}..{ys[-1][-1]}\n{name}')

        start_time = time.time()
        plt.savefig(name, dpi=dpi, format='svg')
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time - plt.savefig: {execution_time} seconds")

    def plt_matshow_Skewed(self, Z, w1mw2, skew_factor, Gamma):
        print('in plt_matshow')
        import matplotlib.pyplot as plt

        X, Y = self.w1_mesh, self.w2_mesh
        if w1mw2:
            y = -(X - Y)
            ystr = 'w2-w1'
            ystr_mesh = ystr+'_'

        else:
            y = Y
            ystr = 'w2'
            ystr_mesh = ystr+'_'
        Z_positive = abs(Z) ** 2
        fig, ax = plt.subplots()
        # im = ax.imshow(np.log2(Z_positive))

        import matplotlib.colors as colors

        class SkewNormalize(colors.Normalize):
            def __init__(self, vmin=None, vmax=None, skew_factor=2, clip=False):
                self.skew_factor = skew_factor
                colors.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                normalized_value = super().__call__(value, clip)
                return np.minimum(normalized_value ** self.skew_factor, 1)

        # Use SkewNormalize for the normalization of the colorbar
        pcm = ax.imshow(np.log10(Z_positive), norm=SkewNormalize(vmin=np.log10(Z_positive).min(), vmax=np.log10(Z_positive).max(), skew_factor=skew_factor), interpolation='nearest')

        # Create colorbar
        cbar = fig.colorbar(pcm, ax=ax, extend='max')

        # Loop over data dimensions and create text annotations.
        for i in range(len(list(Y[:, 0]))):
            for j in range(len(list(X[0, :]))):
                text = ax.text(j, i, round(np.log10(Z_positive)[i, j], 3),
                               ha="center", va="center", color="w")

        ax.set_xticks(np.arange(len(list(X[0, :]))), labels=list(X[0, :]))
        ax.xaxis.tick_top()
        ax.set_yticks(np.arange(len(list(Y[:, 0]))), labels=list(Y[:, 0]))
        fig.set_size_inches(8, 12)
        ax.set_aspect('auto')
        # fig.colorbar(im)
        import os
        fig.suptitle(f'{os.getcwd()}')
        # Use LogNorm for the normalization of the colorbar
        # pcm = ax.imshow(Z_positive, norm=colors.LogNorm(vmin=Z_positive.min(), vmax=Z_positive.max()))
        # Create colorbar
        # cbar = fig.colorbar(pcm, ax=ax, extend='max')

        # fig.tight_layout()
        figfilename = f'./anharmonicities_Gamma{Gamma}_({self.w1[0]}_{self.w1[-1]}_{self.w1[1]-self.w1[0]})({self.w2[0]}_{self.w2[-1]}_{self.w2[1]-self.w2[0]}).svg'

        plt.savefig(figfilename, dpi=500)

        print('exiting plt_matshow')

    def plot2Dplotly(self, Z, w1mw2, Gamma, percent, step):
        import plotly.graph_objects as go
        import numpy as np

        if w1mw2:
            y = -(self.w1-self.w2)
            ystr = 'w1-w2'
            ystr_mesh = ystr #+ '_'

        else:
            y = self.w2
            ystr = 'w2'
            ystr_mesh = ystr #+ '_'

        # Z = self.totInt(style, source, Gamma)
        Z_positive = abs(Z) ** 2 #+10e-72
        maximum = max(np.log10(Z_positive.flatten()))
        minimum = min(np.log10(Z_positive.flatten()))

        # Assuming Z_positive is already defined and contains positive values
        log_Z = np.log10(Z_positive)

        # Step 1: Find the maximum value of the log-transformed array
        max_log_Z = np.max(log_Z)

        # Step 2: Calculate the target value
        target_value = max_log_Z - 2

        # Step 3: Flatten the array to 1D
        flattened_log_Z = log_Z.flatten()

        # Step 4: Count the number of elements greater than the target value
        count_above_target = np.sum(flattened_log_Z > target_value)

        # Step 5: Calculate the percentile rank from the top
        percentile_from_top = 100 * count_above_target / flattened_log_Z.size

        # print(f"The target value is: {target_value}")
        # print(f"The percentile rank of the target value from the top is: {percentile_from_top}")

        if percent is not None:
            custom_colorscale = [
                [0.0, 'rgb(67, 4, 82)'],  # Color for the bottom 95%
                [percent, 'rgb(67, 4, 82)'],  # Same color up to the 95th percentile
                [percent, 'rgb(234, 245, 20)'],  # Color change at the 95th percentile
                [1.0, 'rgb(234, 245, 20)']  # Color for the top 5%
            ]
        else:
            custom_colorscale = 'plasma'

        # print('-====----[:, 0]\n', np.log10(Z_positive)[:, 0], '\n')
        # print('-====----[0. :]\n', np.log10(Z_positive)[0, :], '\n')
        # print('-====----[0]\n', np.log10(Z_positive)[0], '\n')

        # Create the contour plot
        fig = go.Figure(data=
        go.Contour(
            z=np.log10(Z_positive),  # 2D array of Z values
            x=self.w1,  # Corresponding 1D array of X values
            y=y,  # Corresponding 1D array of Y values
            # colorscale='Viridis',  # Color scale
            dx=200.,
            dy=200.,
            colorscale=custom_colorscale,
            colorbar=dict(dtick=1),
            contours=dict(
                coloring='fill',
                showlabels=False,  # show labels on contours
                showlines=False,
            )
        )
        )

        fig.update_layout(
            title=f'Gamma = {Gamma}, top = {percent}, step = {step} cm-1',
            xaxis_title='X-axis',
            yaxis_title=ystr_mesh,
            height=700,
            xaxis=dict(
                # Set a fixed range for the x-axis if needed
                range=[min(self.w1), max(self.w1)],
                tickvals = np.arange(1000., 3000., 200.)

                # range = [1000, 3000]
        ),
            yaxis=dict(
                tickvals=np.arange(2000., 8400., 200.) if not w1mw2 else np.arange(-850., 7400., 200.)

                #         # Set a fixed range for the y-axis if needed
        #         # range=[min(self.w2), max(self.w2)] if not w1mw2 else [min(-(self.w1-self.w2)), max(-(self.w1-self.w2))]
        #         range = [min(self.w2), max(self.w2)] if not w1mw2 else [min(-(self.w1 - self.w2)), max(-(self.w1 - self.w2))]
        #
        ),
            margin=dict(l=40, r=40, t=40, b=40),  # Adjust margins to fit colorbar
            # scrollZoom=True  # Enable zoom on scroll
        )

        return fig

     #    # Convert the figure to an HTML div string
     #    plot_div = fig.to_html(full_html=False)
     #
     #    # HTML template
     #    html_template = """
     #    <!DOCTYPE html>
     #    <html lang="en">
     #    <head>
     #        <meta charset="UTF-8">
     #        <meta name="viewport" content="width=device-width, initial-scale=1.0">
     #        <title>Interactive Plot</title>
     #        <!-- Plotly.js -->
     #        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
     #    </head>
     #    <body>
     # <!--       <h1>My Interactive Plot</h1>  -->
     #        <div id="my-plot">
     #            {plot_div}
     #        </div>
     #    </body>
     #    </html>
     #    """
     #
     #    # Insert the plot_div into the template
     #    html_content = html_template.format(plot_div=plot_div)
     #
     #    # Save the HTML content to a file
     #    with open('plot.html', 'w') as f:
     #        f.write(html_content)

# Qab = [[0, 0], [0, 1]]
def get_abc(nloops, abcrange):
    # print('abcrange', abcrange)
    # print('nloops', nloops)
    stacklist = []
    for i in range(nloops):
        stacklist.append(np.arange(abcrange))

    return np.stack(np.meshgrid(*stacklist), axis=-1).reshape(-1, nloops)


# num_f = 4 -four-wave mixing
def getting_abcgreek4avrg(num_f):
    from mock2D.macroscopic import macroscopics
    # polarizations = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    # pol_laser = macroscopics.get_pol_laser(polarizations)
    # print('pol_laser\n', pol_laser)
    # pol_mat = macroscopics.get_iso_mat(num_f)
    # print('pol_mat\n', pol_mat)
    # pol_g = get_iso_f(spec_cfg.num_fields)

    pol_g = macroscopics.get_iso_f(num_f)
    new = np.array([pol[0] for pol in pol_g], dtype='object').reshape(-1, num_f)

    return new


# works with formula = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))]
def avrg_abc(formula, data, normalModes, gammaCompsAll):
    avrg = 0.

    if data=='ones':
        return 1.
    else:
        for gammaComps in gammaCompsAll:
        # fixme: for loop can be optimized?

            alpha, beta, gamma, delta = gammaComps
            abc = dict(zip(['a', 'b', 'c'], normalModes))

            # this is indexing for "formula" that has 3 elements, therefore 0, 1, 2
            abc_greek = {0: (beta,), 1: (alpha, delta,), 2: (gamma,)}

            tot = 1.

            # [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',))]
            # f - tuple ('mu_Q', ('a',))
            for i, f in enumerate(formula):
                # index for tensor component
                # print(f[1])
                # f[1] - normal modes - tuple ('a',),
                # abc dict is made from the input normalModes, e.g. [a, b] where a and b are indices of normal modes
                indx = tuple(abc[j] for j in f[1]) + abc_greek[i]
                # print(tuple(abc[j] for j in f[1]))
                # print(abc_greek[i])
                # f[0] - property name - 'mu_Q'
                # print(f[0], indx)
                # print(data[f[0]].shape, data[f[0]])
                # print(data[f[0]].T.shape, data[f[0]].T)
                tot *= data[f[0]][indx]
            avrg += tot

        return avrg / 15.

# and now vectorized avrg_abc
def avrg_abc_tensor(formula: list[tuple[str, tuple[str]]], data: dict[str:np.ndarray], gammaCompsAll: list[tuple[float]]):

    import copy
    nmodes = data['mu_Q'].shape[0]
    if type(formula[-1]) == str:
        formula = formula[:-1]
    nmodes_selections = nm_pattern(nmodes, sum([len(i[1]) for i in formula]), formula)
    if [i[0] for i in formula] == ['mu_Q', 'alpha_Q', 'mu_QQ']:
        # [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))]
        # mu_Q, alpha_Q, mu_QQ - (6, 6, 6, 6, 3, 3, 3, 3)
        mu_Q = copy.deepcopy(data['mu_Q']).reshape(nmodes, 1, 1, 1, 1, 3, 1, 1)
        alpha_Q = copy.deepcopy(data['alpha_Q']).reshape(1, nmodes, 1, 1, 3, 1, 1, 3)
        mu_QQ = copy.deepcopy(data['mu_QQ']).reshape(1, 1, nmodes, nmodes, 1, 1, 3, 1)

        bigT = mu_Q * alpha_Q * mu_QQ
        totB = np.zeros((nmodes, nmodes))

        for indices in nmodes_selections:
            tot = 0.
            # print('indices', indices)
            for gamma_indices in gammaCompsAll:
                # print([*indices, *gamma_indices])
                tot += bigT[*indices, *gamma_indices]

            totB[indices[0], indices[1]] = tot/15.
        # totB = totB[:, :, np.newaxis, np.newaxis]  # shape (6, 6, Nx, Ny)

        # print('totB\n', totB)
        return totB

    elif [i[0] for i in formula] == ['mu_Q', 'alpha_QQ', 'mu_Q']:
        # [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))]
        # mu_Q, alpha_QQ, mu_Q
        mu_Q = copy.deepcopy(data['mu_Q']).reshape(nmodes, 1, 1, 1, 1, 3, 1, 1)
        alpha_QQ = copy.deepcopy(data['alpha_QQ']).reshape(1, 1, nmodes, nmodes, 3, 1, 1, 3)
        mu_Qc = copy.deepcopy(data['mu_Q']).reshape(1, nmodes, 1, 1, 1, 1, 3, 1)
        # print('hey')
        bigT = mu_Q * alpha_QQ * mu_Qc
        totB = np.zeros((nmodes, nmodes))
        for indices in nmodes_selections:
            tot = 0.
            # print('indices', indices)
            for gamma_indices in gammaCompsAll:
                # print([*indices, *gamma_indices])
                tot += bigT[*indices, *gamma_indices]

            totB[indices[0], indices[2]] = tot/15.
        # totB = totB[:, :, np.newaxis, np.newaxis]  # shape (6, 6, Nx, Ny)
        # print('totB\n', totB)
        return totB

    elif [i[0] for i in formula] == ['mu_Q', 'alpha_Q', 'mu_Q']:
        # mu_Q, alpha_Q, mu_Q - (6, 6, 6, 3, 3, 3, 3)
        mu_Q = copy.deepcopy(data['mu_Q']).reshape(nmodes, 1, 1, 1, 3, 1, 1)
        alpha_Q = copy.deepcopy(data['alpha_Q']).reshape(1, nmodes, 1, 3, 1, 1, 3)
        mu_Qc = copy.deepcopy(data['mu_Q']).reshape(1, 1, nmodes, 1, 1, 3, 1)

        bigT = mu_Q * alpha_Q * mu_Qc
        totB = np.zeros((nmodes, nmodes, nmodes))
        # print(nmodes_selections)
        # quit()
        for indices in nmodes_selections:
            tot = 0.
            # print('indices', indices)
            for gamma_indices in gammaCompsAll:
                # print([*indices, *gamma_indices])
                tot += bigT[*indices, *gamma_indices]

            if [i[1] for i in formula] == [('a',), ('b',), ('a',)] or [i[1] for i in formula] == [('a',), ('b',), ('b',)]:
                totB[indices[0], indices[1], :] = tot / 15.

            elif [i[1] for i in formula] == [('a',), ('a',), ('b',)]:
                totB[indices[0], indices[2], :] = tot / 15.
        # totB = totB[:, :, :, np.newaxis, np.newaxis]  # shape (6, 6, Nx, Ny)

        # print('totB\n', totB)
        return totB


def nm_pattern(nmodes: int, numberofindices: int, formula: list[tuple[str, tuple[str]]]):
    """
    Generate a pattern of indices for a tensor of nmodes.
    :param numberofindices:
    :param formula:
    :param nmodes:
    :return:
    """
    # Define the shape of the tensor
    shape = tuple([nmodes for _ in range(numberofindices)])
    # print('>>>', shape)
    # Create the tensor using broadcasting
    tensor = np.indices(shape).transpose(*[i for i in range(1, numberofindices + 1)] + [0])

    flattened_tensor = tensor.reshape(-1, numberofindices)

    # Create an empty list to store the selected indices
    selected_indices = []

    # Iterate through the flattened tensor and select indices of the form (a, a, b, b)
    # [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))]
    if [i[1] for i in formula] == [('a',), ('b',), ('a', 'b',)]:
        for indices in flattened_tensor:
            if indices[0] == indices[2] and indices[1] == indices[3]:
                selected_indices.append(indices)

    elif [i[1] for i in formula] == [('a',), ('a', 'b',), ('b',)]:
        for indices in flattened_tensor:
            if indices[0] == indices[1] and indices[1] == indices[2]:
                selected_indices.append(indices)

    elif [i[1] for i in formula] == [('a',), ('b',), ('c',)]:
        selected_indices = flattened_tensor

    elif [i[1] for i in formula] == [('a',), ('b',), ('a',)]:
        # print(f'WARNING: this is not implemented yet - {formula}')
        for indices in flattened_tensor:
            if indices[0] == indices[2]:
                selected_indices.append(indices)

    elif [i[1] for i in formula] == [('a',), ('b',), ('b',)]:
        # print(f'WARNING: this is not implemented yet - {formula}')
        for indices in flattened_tensor:
            if indices[1] == indices[2]:
                selected_indices.append(indices)

    elif [i[1] for i in formula] == [('a',), ('a',), ('b',)]:
        # print(f'WARNING: this is not implemented yet - {formula}')
        for indices in flattened_tensor:
            if indices[0] == indices[1]:
                selected_indices.append(indices)

    selected_indices_array = np.array(selected_indices)
    # print('selected_indices_array', selected_indices_array)
    return selected_indices_array

# function generator
def w_mn_prod(subscripts, fermi=None, margin=10):
    m1n1m2n2 = [i.split(',') for i in subscripts]
    if fermi is not None:
        fermi = [i.split(',') for i in fermi]

    def function(w_all, w1, w2, Gamma, abctuple, m1n1m2n2=m1n1m2n2, fermi=fermi):

        letters = ['a', 'b', 'c', 'zero'] if len(abctuple) == 3 else ['a', 'b', 'zero']
        dictabc = dict(zip(letters, abctuple + tuple(['zero'])))
        w_all[('zero',)] = 0.

        wm1 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[0][0].split('+')]))
        wn1 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[0][1].split('+')]))
        wm2 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[1][0].split('+')]))
        wn2 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[1][1].split('+')]))

        if fermi is None:
            # print('w_all[wm1] - w_all[wn1] + w1 - w2', w_all[wm1], w_all[wn1], w1, w2)
            # print('w1, w2, margin', margin)
            # removes lower diagonal with margin 4
            # print(rec_cm2rec_s(w_all[wm1]) , rec_cm2rec_s(w_all[wn1]), rec_cm2rec_s(w1) , rec_cm2rec_s(w2) , 1j * Gamma)
            return np.where(w2-margin > w1, 1 / (rec_cm2rec_s(w_all[wm1]) - rec_cm2rec_s(w_all[wn1]) + rec_cm2rec_s(w1) - rec_cm2rec_s(w2) - 1j * Gamma) / (rec_cm2rec_s(w_all[wm2]) - rec_cm2rec_s(w_all[wn2]) + rec_cm2rec_s(w1) - 1j * Gamma), 0.)

        else:
            w_fr11 = tuple(sorted([str(dictabc[i]) for i in fermi[0][0].split('+')]))
            w_fr21 = tuple(sorted([str(dictabc[i]) for i in fermi[0][1].split('+')]))

            w_fr12 = tuple(sorted([str(dictabc[i]) for i in fermi[1][0].split('+')]))
            w_fr22 = tuple(sorted([str(dictabc[i]) for i in fermi[1][1].split('+')]))

            tail = 0.0
            t1 = rec_cm2rec_s(w_all[wm1]) - rec_cm2rec_s(w_all[wn1]) + rec_cm2rec_s(w1) - rec_cm2rec_s(w2) - 1j * Gamma
            t2 = rec_cm2rec_s(w_all[wm2]) - rec_cm2rec_s(w_all[wn2]) + rec_cm2rec_s(w1) - 1j * Gamma
            t3 = rec_cm2rec_s(w_all[w_fr11]) - rec_cm2rec_s(w_all[w_fr21])
            t4 = rec_cm2rec_s(w_all[w_fr12]) - rec_cm2rec_s(w_all[w_fr22])
            return (1 / t1 / t2) * (1 / t3 + 1 / t4)

    return function

def makehtml(name, fig):
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multiple Plots</title>
        <!-- Plotly.js -->
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            /* Simple grid layout with two columns taking up 45% each */
            .row {{
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-start; /* Align columns to the start of the row */
            }}
            .column {{
                flex: 0 0 40%; /* Do not grow or shrink, base size is 45% */
                padding: 5px;
                box-sizing: border-box;
                # margin-right: 7%; /* Right margin of 10% (adjust as needed) */
            }}
            /* Remove right margin for the last column */
            .column:last-child {{
                margin-right: 0;
            }}
        </style>
    </head>
    <body>
        <div class="row">
            {plot_divs}
        </div>
    </body>
    </html>
    """

    include_plotlyjs = 'cdn'  # Include Plotly.js only in the first plot
    plot_div = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
    plot_divs = f'<div class="column">{plot_div}</div>'

    # Insert the plot DIVs into the template
    html_content = html_template.format(plot_divs=plot_divs)

    # Save the HTML content to a file
    with open(name, 'w') as f:
        f.write(html_content)

def printT(tensor):
    import pandas as pd
    pd.set_option('display.float_format', '{:.10f}'.format)

    ndims = len(tensor.shape)

    # mu_Q
    if ndims == 2:
        column_names = ['x', 'y', 'z']
        row_names    = [f'{i}' for i in range(tensor.shape[0])]
        df = pd.DataFrame(tensor, columns=column_names)#, index=row_names)
        df.insert(0, "I", row_names, allow_duplicates=True)
        df.insert(1, "", ['|']*len(row_names), allow_duplicates=True)

        # print(df)
        print(df.to_string(index=False))

    elif ndims == 3:
        # F_abc
        if tensor.shape[0] == tensor.shape[1] == tensor.shape[2]:
            row_names = [f'K {i}' for i in range(tensor.shape[0])]
            indx = [f'{i}' for i in range(tensor.shape[1])]
            df = pd.DataFrame(tensor[0], columns=row_names)#, index=row_names)
            df.insert(0, "I", ['0']*len(row_names), allow_duplicates=True)
            df.insert(1, "J", indx, allow_duplicates=True)
            df.insert(2, "", ['|'] * len(row_names), allow_duplicates=True)

            for ii, k in enumerate(tensor[1:]):
                dfi = pd.DataFrame(k, columns=row_names)#, index=row_names)
                dfi.insert(0, "I", [f'{ii+1}']*len(row_names), allow_duplicates=True)
                dfi.insert(1, "J", indx, allow_duplicates=True)
                dfi.insert(2, "", ['|'] * len(row_names), allow_duplicates=True)

                df = pd.concat([df, dfi], ignore_index=True)

            n = len(indx)  # chunk row size
            list_df = [df[i:i + n] for i in range(0, df.shape[0], n)]

            for dframe in list_df:
                print(dframe.to_string(index=False))

        # mu_QQ
        elif tensor.shape[0] == tensor.shape[1] != tensor.shape[2]:
            row_names = ['x', 'y', 'z']
            indx = [f'{i}' for i in range(tensor.shape[1])]
            df = pd.DataFrame(tensor[0], columns=row_names)  # , index=row_names)
            df.insert(0, "I", ['0'] * len(indx), allow_duplicates=True)
            df.insert(1, "J", indx, allow_duplicates=True)
            df.insert(2, "", ['|'] * len(indx), allow_duplicates=True)

            for ii, k in enumerate(tensor[1:]):
                dfi = pd.DataFrame(k, columns=row_names)  # , index=row_names)
                dfi.insert(0, "I", [f'{ii + 1}'] * len(indx), allow_duplicates=True)
                dfi.insert(1, "J", indx, allow_duplicates=True)
                dfi.insert(2, "", ['|'] * len(indx), allow_duplicates=True)

                df = pd.concat([df, dfi], ignore_index=True)

            n = len(indx)  # chunk row size
            list_df = [df[i:i + n] for i in range(0, df.shape[0], n)]

            for dframe in list_df:
                print(dframe.to_string(index=False))

        # alpha_Q
        elif tensor.shape[0] != tensor.shape[1] == tensor.shape[2]:
            row_names = ['x', 'y', 'z']
            indx = [f'{i}' for i in range(tensor.shape[1])]
            df = pd.DataFrame(tensor[0], columns=row_names)  # , index=row_names)
            df.insert(0, "I", ['0'] * len(indx), allow_duplicates=True)
            df.insert(1, "", row_names, allow_duplicates=True)
            df.insert(2, "", ['|'] * len(indx), allow_duplicates=True)

            for ii, k in enumerate(tensor[1:]):
                dfi = pd.DataFrame(k, columns=row_names)  # , index=row_names)
                dfi.insert(0, "I", [f'{ii + 1}'] * len(indx), allow_duplicates=True)
                dfi.insert(1, "", row_names, allow_duplicates=True)
                dfi.insert(2, "", ['|'] * len(indx), allow_duplicates=True)

                df = pd.concat([df, dfi], ignore_index=True)

            n = len(indx)  # chunk row size
            list_df = [df[i:i + n] for i in range(0, df.shape[0], n)]

            for dframe in list_df:
                print(dframe.to_string(index=False))

    # alpha_QQ
    elif ndims == 4:
        listdf = []
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[0]):
                row_names = ['x', 'y', 'z']
                df = pd.DataFrame(tensor[i, j], columns=row_names)  # , index=row_names)
                df.insert(0, "I", [f'{i}'] * 3, allow_duplicates=True)
                df.insert(1, "J", [f'{j}'] * 3, allow_duplicates=True)
                df.insert(2, "", row_names, allow_duplicates=True)
                df.insert(3, "", ['|'] * 3, allow_duplicates=True)

                listdf.append(df)

        dfs = pd.concat(listdf, ignore_index=True)
        n = tensor.shape[2]  # chunk row size
        list_df = [dfs[i:i + n] for i in range(0, dfs.shape[0], n)]

        for dframe in list_df:
            print(dframe.to_string(index=False))

    else:
        print(f"Dimension of the property in not 2, 3 or 4, it's {ndims}")

def printed2DIRtensors(setup: SpectrumEVV):
    ders = setup.getDerivs()
    print('\nFundamental frequencies (anharmonic):', list(setup.fundamentals.values()))
    print('Fundamental frequencies (harmonic)  :', list(setup.fundamentals_harmonic.values()), '\n')

    print('All frequencies (anharmonic)  :', setup.all_states, '\n')
    # print('All frequencies (harmonic)    :', setup.all_states_harm, '\n')

    # for k in setup.fundamentals:
    #     print()
    for d in ders:
        print(d, ders[d].shape)#, '\n', ders[d])
        printT(ders[d])
        print('=========================================================\n')

def makeHTML(figures, w1mw2, step, toppercent):
    # HTML template with a three-column layout using inline CSS
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multiple Plots</title>
        <!-- Plotly.js -->
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            /* Simple grid layout with two columns taking up 45% each */
            .row {{
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-start; /* Align columns to the start of the row */
            }}
            .column {{
                flex: 0 0 40%; /* Do not grow or shrink, base size is 45% */
                padding: 5px;
                box-sizing: border-box;
                # margin-right: 7%; /* Right margin of 10% (adjust as needed) */
            }}
            /* Remove right margin for the last column */
            .column:last-child {{
                margin-right: 0;
            }}
        </style>
    </head>
    <body>
        <div class="row">
            {plot_divs}
        </div>
    </body>
    </html>
    """

    plot_divs = ""
    for i, fig in enumerate(figures):
        include_plotlyjs = 'cdn' if i == 0 else False  # Include Plotly.js only in the first plot
        plot_div = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
        plot_divs += f'<div class="column">{plot_div}</div>'

    # Insert the plot DIVs into the template
    html_content = html_template.format(plot_divs=plot_divs)

    # Save the HTML content to a file
    with open(f'./small_w1mw2{w1mw2}_step{step}_t{toppercent}_n.html', 'w') as f:
        f.write(html_content)

