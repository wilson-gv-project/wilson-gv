#####################################################################################
##                                                                                 ##
##          File contains main code for 2DIR spectrum generation (images)          ##
##                                                                                 ##
#####################################################################################

import time

import numpy as np
np.set_printoptions(linewidth=100000)

from .callbacks2DIR import CFOURdata, GaussianData

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

        cfuncs = {'cfour': CFOURdata(data), 'gaussian': GaussianData(data)}
        self.callbacks = cfuncs[data['source']]

        got_funds = self.callbacks.getFundamentals()

        # dictionary; keys from 0 to (3Natoms-6)
        self.fundamentals = {str(k):v for k,v in got_funds[0].items()}
        self.fundamentals_harmonic = {str(k):v for k,v in got_funds[1].items()}

        # margin for higher diagonal
        self.margin = 10.

        parsed_states = self.callbacks.getAllStates()

        self.all_states = {tuple(str(i) for i in k): v for k, v in parsed_states[0].items()}
        self.all_states_harm = {tuple(str(i) for i in k): v for k, v in parsed_states[1].items()}

        self.id = f'w1{min(self.w1)}_{max(self.w1)}w2{min(self.w2)}_{max(self.w2)}'

        self.deriv_data  = self.getDerivs()
        self.gammaCompsAll = getting_abcgreek4avrg(num_f=4)
        print('gammaCompsAll\n', len(self.gammaCompsAll), '\n',self.gammaCompsAll)


    # setting up the expressions for mechanical and electrical anharmonicities
    def addTerms(self, electrical_terms, mechanical_terms, el_avrg, mech_avrg):
        if electrical_terms is None and mechanical_terms is None and el_avrg is None and mech_avrg is None:
            # Terms in expressions
            electrical_terms_r = [('a+b,a', 'zero,a'), ('b,a', 'zero,a')]

            # derivatives:
            # 1. mu_Q, mu QQ, alpha_Q - electric dipole (1st and 2nd derivatives), polarizability (1st der.)
            # 2. mu_Q, alpha_QQ - electric dipole (1st der.), polarizability (2nd der.)
            electric_avrg_r = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))],
                              [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))]
                               ]

            mechanical_terms_r = [(('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')),
                                (('c,a', 'zero,a'), ('a+b,c', 'b+c,a')),
                                (('a+b,a', 'zero,a'), ('a,a+b', 'b,zero')),
                                (('b,a', 'zero,a'), ('b,a+b', 'a,zero')),
                                (('b,a', 'zero,a'), ('a,a+b', 'b,zero')),
                                (('b,a', 'zero,a'), ('b,a+b', 'a,zero'))]

            # derivatives:
            # mu_Q, alpha_Q - for all 6 terms
            mechanical_avrg_r = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                                [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                               [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc'],
                               [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc'],
                               [('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc'],
                               [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc']]

            ee, mm = [0, 1], [0, 1, 2, 3, 4, 5]
            self.electrical_terms, self.mechanical_terms, self.electric_avrg, self.mechanical_avrg = picks(electrical_terms_r, ee), picks(mechanical_terms_r, mm), picks(electric_avrg_r, ee), picks(mechanical_avrg_r, mm)

        # here the functions of 2 frequencies
        self.electr_funs = [w_mn_prod(i, margin=self.margin) for i in self.electrical_terms]
        self.mech_funs = [w_mn_prod(*i) for i in self.mechanical_terms]

        nmodes = len(self.fundamentals)
        self.combofuns = [dict(zip(self.electr_funs, self.electric_avrg)),
                          dict(zip(self.mech_funs, self.mechanical_avrg))]

        # setting up the combinations of states for the terms
        self.coords_ab = get_abc(2, len(self.fundamentals)) if self.electrical_terms is not None else []
        self.coords_abc = get_abc(3, len(self.fundamentals)) if self.mechanical_terms is not None else []

        if self.electrical_terms is not None:
            self.el_avrg_tensors = [avrg_abc_tensor_new(ea, self.deriv_data, self.gammaCompsAll) for ea in self.electric_avrg]
        else:
            self.el_avrg_tensors = []

        if self.mechanical_terms is not None:
            self.mech_avrg_tensors = [avrg_abc_tensor_new(ma, self.deriv_data, self.gammaCompsAll) for ma in self.mechanical_avrg]
        else:
            self.mech_avrg_tensors = []

        self.combofuns_tensors = [dict(zip(self.electr_funs, self.el_avrg_tensors)),
                                  dict(zip(self.mech_funs, self.mech_avrg_tensors))]

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
        self.w_abc = rec_cm2rec_s(w_abc)
        self.w_ab = rec_cm2rec_s(w_ab) # for omega_{a+b} frequencies
        w = rec_cm2rec_s(np.array([v for k,v in self.fundamentals.items()]))
        w_h = rec_cm2rec_s(np.array([v for k,v in self.fundamentals_harmonic.items()]))

        self.matrix_2d = np.outer(w_h, w_h)
        self.tensor_3d = w_h[:, np.newaxis, np.newaxis] * w_h[np.newaxis, :, np.newaxis] * w_h[np.newaxis, np.newaxis, :]

        for i, te in enumerate(self.el_avrg_tensors):
            print(f'el_avrg_tensors {self.electric_avrg[i]}\n', te)

        for k, tm in enumerate(self.mech_avrg_tensors):
            print(f'mech_avrg_tensors {self.mechanical_avrg[k]}\n', tm)

    def getDerivs(self):

        if self.data['source'] == 'cfour':
            # data is a list of np.arrays 'mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'
            firstder, secder = self.callbacks.getDipDers()
            data = [firstder, secder]

            polder = self.callbacks.getPolarDers()
            data.append(polder[0])
            data.append(polder[1])

            cubicmat = self.callbacks.getCFF()
            data.append(cubicmat)

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
            total_sum_el = 0
            prefac_el = self.matrix_2d.T[a, b]
            for index, (el_func, elavrg) in enumerate(self.combofuns_tensors[0].items()):
                total_sum_el += prefac_el * elavrg[a, b] * el_func(self.all_states, self.w1_mesh, self.w2_mesh,
                                                                Gamma, (a, b))
            return total_sum_el / 24.

        else:

            total_sum_mech = 0
            prefac_mech = self.tensor_3d.T[a, b, c]
            factors = [1., 1., 0.5, 0.5, -0.5, -0.5]
            for index, (mech_func, mechavrg) in enumerate(self.combofuns_tensors[1].items()):
                mechavrgF = list(self.combofuns[1].items())[index][1]
                abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
                indx = tuple([abc[j] for j in mechavrgF[-1]])
                F = self.deriv_data['F_abc'][indx]
                total_sum_mech += factors[index] * prefac_mech * mechavrg[a, b, c] * F * mech_func(self.all_states, self.w1_mesh, self.w2_mesh,
                                                                Gamma, (a, b, c))
            return -total_sum_mech / 48.

    def intensity(self, Gamma, savedict, el=True, mech=True, printdata=False):
        Qab, Qabc = self.coords_ab, self.coords_abc
        Z = 0
        Qab_contrib_dict = {}
        Qabc_contrib_dict = {}

        if el:
            start_time = time.time()
            elall = np.zeros(self.shape2d, dtype='complex128')
            for i in Qab:
                contrib_ab = self.gamma_mn(Gamma, i[0], i[1])
                Qab_contrib_dict[tuple(i)] = contrib_ab
                elall += contrib_ab
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time -| electrical: {execution_time} seconds")
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

    def plot2Dmatplotlib(self, Z, w1mw2, name, Gamma, dpi=500, contour_levels=100, log10=True, shift_scale=None):
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
            ystr_mesh = ystr + '_'
        else:
            y = Y
            ystr = 'w2'
            ystr_mesh = ystr + '_'
        Z_positive = abs(Z) ** 2
        if log10:
            Z_data = np.log10(Z_positive)
        else:
            Z_data = Z_positive

        df = {'w1_mesh': X, ystr_mesh: y, 'values_mesh': Z_data}

        plt.figure(figsize=(12, 11))

        if shift_scale is not None:
            import matplotlib.colors as colors

            class SkewNormalize(colors.Normalize):
                def __init__(self, vmin=None, vmax=None, skew_factor=2, clip=False):
                    self.skew_factor = skew_factor
                    colors.Normalize.__init__(self, vmin, vmax, clip)

                def __call__(self, value, clip=None):
                    normalized_value = super().__call__(value, clip)
                    return np.minimum(normalized_value ** self.skew_factor, 1)

            norm = SkewNormalize(vmin=np.log10(Z_positive).min(), vmax=np.log10(Z_positive).max(), skew_factor=shift_scale)
        else:
            norm = None

        start_time = time.time()
        cont = plt.contourf(df['w1_mesh'], df[ystr_mesh], df['values_mesh'], levels=contour_levels, cmap='viridis',
                            norm=norm)
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
        plt.title(
            f'plot2Dmatplotlib().\ndpi={dpi} contour_levels={contour_levels} x{xs[0][0]}..{xs[-1][-1]} y{ys[0][0]}..{ys[-1][-1]}\n{name}\nGamma={Gamma}')

        start_time = time.time()
        plt.savefig(name, dpi=dpi, format='svg')
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time - plt.savefig: {execution_time} seconds")

def get_abc(nloops, abcrange):
    stacklist = []
    for i in range(nloops):
        stacklist.append(np.arange(abcrange))
    return np.stack(np.meshgrid(*stacklist), axis=-1).reshape(-1, nloops)

def getting_abcgreek4avrg(num_f):
    from mock2D.macroscopic import macroscopics
    pol_g = macroscopics.get_iso_f(num_f)
    new = np.array([pol[0] for pol in pol_g], dtype='object').reshape(-1, num_f)
    return new

def avrg_abc(formula, data, normalModes, gammaCompsAll):
    avrg = 0.

    for gammaComps in gammaCompsAll:
        alpha, beta, gamma, delta = gammaComps
        abc = dict(zip(['a', 'b', 'c'], normalModes))
        # this is indexing for "formula" that has 3 elements, therefore 0, 1, 2
        abc_greek = {0: (beta,), 1: (alpha, delta,), 2: (gamma,)}
        tot = 1.
        for i, f in enumerate(formula):
            # abc dict is made from the input normalModes, e.g. [a, b] where a and b are indices of normal modes
            indx = tuple(abc[j] for j in f[1]) + abc_greek[i]
            tot *= data[f[0]][indx]
        avrg += tot
    return avrg / 15.

def avrg_abc_tensor_new(formula: list[tuple[str, tuple[str]]], data: dict[str:np.ndarray], gammaCompsAll: list[tuple[float]]):
    nmodes = data['mu_Q'].shape[0]
    if type(formula[-1]) == str:
        formula = formula[:-1]
    if [i[0] for i in formula] == ['mu_Q', 'alpha_Q', 'mu_QQ']:
        totB = np.zeros((nmodes, nmodes))
        for a in range(nmodes):
            for b in range(nmodes):
                total = 0.
                for comps in gammaCompsAll:
                    alpha, beta, gamma, delta = comps
                    total += data['mu_Q'][a, beta] * data['alpha_Q'][b, alpha, delta] * data['mu_QQ'][a, b, gamma]
                totB[a, b] = total/15.
        return totB

    elif [i[0] for i in formula] == ['mu_Q', 'alpha_QQ', 'mu_Q']:
        totB = np.zeros((nmodes, nmodes))
        for a in range(nmodes):
            for b in range(nmodes):
                total = 0.
                for comps in gammaCompsAll:
                    alpha, beta, gamma, delta = comps
                    total += data['mu_Q'][a, beta] * data['alpha_QQ'][a, b, alpha, delta] * data['mu_Q'][b, gamma]
                totB[a, b] = total/15.
        return totB

    elif [i[0] for i in formula] == ['mu_Q', 'alpha_Q', 'mu_Q']:
        totB = np.zeros((nmodes, nmodes, nmodes))
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
                    totB[a, b, c] = total/15.
        return totB

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

def printed2DIRtensors(setup: SpectrumEVV):
    ders = setup.getDerivs()
    print('\nFundamental frequencies (anharmonic):', list(setup.fundamentals.values()))
    print('Fundamental frequencies (harmonic)  :', list(setup.fundamentals_harmonic.values()), '\n')

    print('All frequencies (anharmonic)  :', setup.all_states, '\n')
    for d in ders:
        print(d, ders[d].shape)#, '\n', ders[d])
        printT(ders[d])
        print('=========================================================\n')
