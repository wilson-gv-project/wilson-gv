#####################################################################################
##                                                                                 ##
##          File contains main code for 2DIR spectrum generation (images)          ##
##                                                                                 ##
#####################################################################################

import time

import numpy as np
np.set_printoptions(linewidth=100000)

from src.wilson.retrievedata import CFOURdata, GaussianData

def picks(pool, list_of_indices):
    return [pool[i] for i in list_of_indices]

def rec_cm2rec_s(reciprocal_cm):
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
        fermirm
    """
    def __init__(self, w1, w2, input_data_info):

        # Define the grid of spectrum (pixels)
        self.w1_mesh, self.w2_mesh = np.meshgrid(w1, w2, indexing='ij')
        # axes as arrays
        self.w1, self.w2 = np.array(w1), np.array(w2)
        self.shape2d = self.w1_mesh.shape
        self.data_info = input_data_info # dictionary with input_data_info source and type - inputs
        # Get the appropriate functions to retrieve the data
        cfuncs = {'cfour': CFOURdata(input_data_info), 'gaussian': GaussianData(input_data_info)}
        self.callbacks = cfuncs[input_data_info['source']]

        got_funds = self.callbacks.getFundamentals()

        # dictionary; keys from 0 to (3Natoms-6)
        self.fundamentals = {str(k):v for k,v in got_funds[0].items()}
        self.fundamentals_harmonic = {str(k):v for k,v in got_funds[1].items()}

        # margin for higher diagonal
        self.diagonal_margin = 10.

        parsed_states = self.callbacks.getAllStates()

        self.all_states = {tuple(str(i) for i in k): v for k, v in parsed_states[0].items()}
        self.all_states_harmonic = {tuple(str(i) for i in k): v for k, v in parsed_states[1].items()}
        print('all states\n', self.all_states, '\n')
        print('all all_states_harmonic\n', self.all_states_harmonic, '\n')

        print(sorted(self.all_states_harmonic.values()))

        self.id = f'w1{min(self.w1)}_{max(self.w1)}w2{min(self.w2)}_{max(self.w2)}'

        self.deriv_data  = self.getDerivs()
        self.gammaCompsAll = getting_abcgreek4avrg(num_f=4)


    def addTerms(self, electrical_terms_selection, mechanical_terms_selection):
        """Creating functions for computing the expressions for mechanical and electrical anharmonicities"""
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

        ee, mm = electrical_terms_selection, mechanical_terms_selection
        self.electrical_terms, self.mechanical_terms = picks(electrical_terms_r, ee), picks(mechanical_terms_r, mm)
        self.electric_avrg, self.mechanical_avrg = picks(electric_avrg_r, ee), picks(mechanical_avrg_r, mm)
        # here the functions of 2 frequencies
        self.electr_funs = [w_mn_prod(i, margin=self.diagonal_margin) for i in self.electrical_terms]
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
        print('\nw_h')
        print(w_h)
        print('\n1./self.matrix_2d = 1./np.outer(w_h, w_h)')
        print(1./self.matrix_2d)
        self.tensor_3d = w_h[:, np.newaxis, np.newaxis] * w_h[np.newaxis, :, np.newaxis] * w_h[np.newaxis, np.newaxis, :]

        for i, te in enumerate(self.el_avrg_tensors):
            print(f'\nel_avrg_tensors {self.electric_avrg[i]}\n', te)

        for k, tm in enumerate(self.mech_avrg_tensors):
            print(f'mech_avrg_tensors {self.mechanical_avrg[k]}\n', tm)

    def getDerivs(self):

        if self.data_info['source'] == 'cfour':
            w_h = rec_cm2rec_s(np.array([v for k, v in self.fundamentals_harmonic.items()]))
            self.matrix_2d = np.outer(w_h, w_h)
            self.tensor_3d = w_h[:, np.newaxis, np.newaxis] * w_h[np.newaxis, :, np.newaxis] * w_h[np.newaxis,
                                                                                               np.newaxis, :]

            sqrtvec = 1./np.sqrt(w_h)
            sqrtmat = 1./np.sqrt(self.matrix_2d.T)
            sqrt3d = 1./np.sqrt(self.tensor_3d.T)

            # input_data_info is a list of np.arrays 'mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'
            firstder, secder = self.callbacks.getDipDers()

            firstder_mat = np.zeros_like(firstder)
            for i in range(len(sqrtvec)):
                for j in range(3):
                    firstder_mat[i, j] = firstder[i, j] / sqrtvec[i]

            secder_mat = np.zeros_like(secder)
            for i in range(len(sqrtvec)):
                for j in range(len(sqrtvec)):
                    for k in range(3):
                        secder_mat[i, j, k] = secder[i, j, k] / sqrtmat[i, j]

            data = [firstder_mat, secder_mat]

            polder = self.callbacks.getPolarDers()
            fdpol = np.zeros_like(polder[0])
            for i in range(len(sqrtvec)):
                for j in range(3):
                    for k in range(3):
                        fdpol[i, j, k] = polder[0][i, j, k] / sqrtvec[i]

            sdpol = np.zeros_like(polder[1])
            for i in range(len(sqrtvec)):
                for j in range(len(sqrtvec)):
                    with open('./secPolder', 'a') as file1:
                        file1.write(f'\n=============================={i} {j}\n{sqrtmat[i, j]}\n')
                        file1.writelines(str(polder[1][i, j, :, :]))

                    for k in range(3):
                        for l in range(3):
                            sdpol[i, j, k, l] = polder[1][i, j, k, l] / sqrtmat[i, j]

            data.append(fdpol)
            data.append(sdpol)

            from scipy import constants
            # to go from amu to au mass unit (m_e)
            amc_au = constants.physical_constants['atomic mass constant'][0] / \
                     constants.physical_constants['atomic unit of mass'][0]

            cubicmat = self.callbacks.getCFF()
            cubicmat /= amc_au**1.5
            data.append(cubicmat)
            allpropsdict = dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], data))

            return allpropsdict

        elif self.data_info['source'] == 'gaussian':
            from scipy import constants
            # to go from amu to au mass unit (m_e)
            amc_au = constants.physical_constants['atomic mass constant'][0] / \
                     constants.physical_constants['atomic unit of mass'][0]

            # input_data_info is a list of np.arrays 'mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'
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
            print("Invalid input_data_info source")

    def gamma_mn(self, Gamma, a, b, c=False):
        # if 'c' is not provided, compute electrical anharmonicity
        if type(c) == bool:
            total_sum_el = 0
            prefac_el = self.matrix_2d.T[a, b]
            for index, (el_func, elavrg) in enumerate(self.combofuns_tensors[0].items()):
                resonance = el_func(self.all_states_harmonic, self.w1_mesh, self.w2_mesh,
                                    Gamma, (a, b))
                if a==1 and b==4:
                    print(a, b)
                    print(resonance)
                    print(self.w1_mesh)
                    print(self.w2_mesh)
                    print(1./prefac_el, elavrg[a, b])
                    print("result\n", elavrg[a, b] * resonance / prefac_el/24.)
                total_sum_el += elavrg[a, b] * resonance / prefac_el
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
                resonance2 = mech_func(self.all_states_harmonic, self.w1_mesh, self.w2_mesh,
                                       Gamma, (a, b, c))
                with open('./resonance2', 'a') as file1:
                    file1.write(f'{mech_func}\n')
                    file1.writelines(str(resonance2)+'\n')
                total_sum_mech += factors[index] / prefac_mech * mechavrg[a, b, c] * F * resonance2
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

        key = self.id+f'_gamma{Gamma}'
        if key not in savedict:
            savedict[key] = {}

        if mech: savedict[key]['mechanical'] = mechall
        if el: savedict[key]['electrical'] = elall
        savedict[key]['Qab_contrib_dict'] = Qab_contrib_dict
        savedict[key]['Qabc_contrib_dict'] = Qabc_contrib_dict

        return Z, savedict

    def plot2Dmatplotlib(self, Z, w1mw2, nametuple, Gamma, el, mech, dpi=200, log10=True):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib

        matplotlib.use('Agg')
        plt.rcParams['path.simplify'] = True
        plt.rcParams['agg.path.chunksize'] = 10000
        plt.rcParams['axes.titlepad'] = 30

        font = {'family': 'normal',
                # 'weight': 'bold',
                'size': 18}
        matplotlib.rc('font', **font)

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
            Z_data = Z_positive
            # Z_data = np.log10(Z_positive)
        else:
            Z_data = Z_positive
            # Z_data = np.log10(Z_positive)

        df = {'w1_mesh': X, ystr_mesh: y, 'values_mesh': Z_data}
        print('\n>>>>>>   Z_data max', max(np.max(df['values_mesh'].flatten(), axis=0), np.max(df['values_mesh'].flatten(), axis=0)),
              f'\nmax in axes: {np.max(df['values_mesh'].flatten(), axis=0)}, {np.max(df['values_mesh'].flatten(), axis=0)}')

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)

        import matplotlib.colors as colors
        start_time = time.time()
        normnow = colors.LogNorm(vmin=1e3, vmax=1e8)

        dynrange = 1000 # stop plotting when lower than this (number times 10) dmax
        n_cont = 30
        dynrange_log = np.log10(dynrange)

        intensities = df['values_mesh']
        d_min = (1.0 / float(dynrange)) * intensities.max()

        dmax_dict = {(True, False): 48778401.3, (False, True): 29519537.48, (True, True): 48218929.9}
        d_max = dmax_dict[(el, mech)] # m, e, t 29519537.48  48778401.3  48218929.9

        dmax_log10 = float(int(np.log10(d_max)))
        print("dmax_log10", dmax_log10)
        levels_ticks = [10**(dmax_log10-i) for i in range(6)]
        print("levels_ticks", levels_ticks)

        levels = []
        for i in range(n_cont):
            levels.append(
                (d_max) * 10.0 ** (-1.0 * dynrange_log * ((float(n_cont - 1) - float(i)) / float(n_cont - 1))))

        cont = plt.contourf(df['w1_mesh'], df[ystr_mesh], df['values_mesh'], levels=levels, cmap='hot_r',
                            norm=normnow)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time - plt.contourf: {execution_time} seconds")

        # This is the fix for the white lines between contour levels
        for c in cont.collections:
            c.set_edgecolor("face")

        import matplotlib.ticker as ticker
        def fmt(x, pos):
            a, b = '{:.0e}'.format(x).split('e')
            b = int(b)
            return r'${} \times 10^{{{}}}$'.format(a, b)

        # colorbar = plt.colorbar(cont, ticks=levels_ticks, format='%s')
        colorbar = plt.colorbar(cont, ticks=levels_ticks, format=ticker.FuncFormatter(fmt))
        # https://stackoverflow.com/questions/25983218/scientific-notation-colorbar
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

        # colorbar.set_ticks(levels_ticks, format='%.4f')

        # plt.xlabel(r'$\omega_1$')
        # plt.ylabel(r'$\omega_2$')
        xs = df['w1_mesh'][0], df['w1_mesh'][-1]
        ys = df[ystr_mesh][0], df[ystr_mesh][-1]
        labeltypedict = {(True, False): r'electrical anharmonicity $|\gamma^{[1,0]}|^2$ only',
                         (False, True): r'mechanical anharmonicity $|\gamma^{[0,1]}|^2$ only',
                         (True, True): r'both $|\gamma^{[1,0]}+\gamma^{[0,1]}|^2$'}
        nicetitle = f'{nametuple[2]}'
        # plt.title(
        #     f'plot2Dmatplotlib().\ndpi={dpi}\nx{xs[0][0]}..{xs[-1][-1]} y{ys[0][0]}..{ys[-1][-1]}\n{nametuple[0]}\nGamma={Gamma}\n{nametuple[1]}\n{np.max(df['values_mesh'].flatten(), axis=0)} or {'{:.4e}'.format(np.max(df['values_mesh'].flatten(), axis=0))}')
        plt.title(nicetitle) # +'\n\n'+labeltypedict[(el, mech)]
        plt.tight_layout()
        start_time = time.time()
        plt.savefig(nametuple[0], dpi=dpi, format='svg')
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time - plt.savefig: {execution_time} seconds")

def get_abc(nloops, abcrange):
    stacklist = []
    for i in range(nloops):
        stacklist.append(np.arange(abcrange))
    return np.stack(np.meshgrid(*stacklist), axis=-1).reshape(-1, nloops)

def getting_abcgreek4avrg(num_f):
    from .. import macroscopics
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
def w_mn_prod(subscripts, fermi=None, margin=10.):
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

        if  abctuple == (1, 4):
            print(wm1, wn1)
            print('wm1-wn1', rec_cm2rec_s(w_all[wm1]) - rec_cm2rec_s(w_all[wn1]))
            print(wm2, wn2)
            print('wm2-wn2', rec_cm2rec_s(w_all[wm2]) - rec_cm2rec_s(w_all[wn2]))
            print('w1-w2', rec_cm2rec_s(w1) - rec_cm2rec_s(w2))
            print('rescod1\n', (rec_cm2rec_s(w_all[wm1]) - rec_cm2rec_s(w_all[wn1])
                                                 + rec_cm2rec_s(w1) - rec_cm2rec_s(w2) - 1j * Gamma) )
            print('rescond2\n', (rec_cm2rec_s(w_all[wm2]) - rec_cm2rec_s(w_all[wn2]) + rec_cm2rec_s(w1) - 1j * Gamma))
            print('1/rescod1\n',1./ (rec_cm2rec_s(w_all[wm1]) - rec_cm2rec_s(w_all[wn1])
                                                 + rec_cm2rec_s(w1) - rec_cm2rec_s(w2) - 1j * Gamma) )
            print('1/rescond2\n', 1./ (rec_cm2rec_s(w_all[wm2]) - rec_cm2rec_s(w_all[wn2]) + rec_cm2rec_s(w1) - 1j * Gamma))
        if fermi is None:
            return np.where(w2-margin > w1, 1 / (rec_cm2rec_s(w_all[wm1]) - rec_cm2rec_s(w_all[wn1])
                                                 + rec_cm2rec_s(w1) - rec_cm2rec_s(w2) - 1j * Gamma) / (rec_cm2rec_s(w_all[wm2]) - rec_cm2rec_s(w_all[wn2]) + rec_cm2rec_s(w1) - 1j * Gamma), 0.)

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

            sumfrac = (1 / t3 + 1 / t4)
            with open('./fermi', 'a') as file1:
                file1.write('\n==============================\n')
                file1.write(f'{abctuple}\n{w_fr11} {w_fr21} {w_fr12} {w_fr22}\n{fermi}\n')
                file1.writelines(str(t3)+'\n')
                file1.writelines(str(t4)+'\n')
                file1.writelines(str(sumfrac) + '\n')

            with open('./fermi_other', 'a') as file1:
                file1.write('\n==============================\n')
                file1.write(f'{abctuple}\n{m1n1m2n2}\n')
                file1.writelines(str(rec_cm2rec_s(w_all[wm1]) - rec_cm2rec_s(w_all[wn1]))+'\n')
                file1.writelines(str( rec_cm2rec_s(w_all[wm2]) - rec_cm2rec_s(w_all[wn2]) )+'\n')
                file1.writelines(str((1 / t1 / t2)) + '\n')

            return (1 / t1 / t2) * sumfrac

    return function

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

    for d in ders:
        print(d, ders[d].shape)#, '\n', ders[d])
        printT(ders[d])
        print('==================================\n')