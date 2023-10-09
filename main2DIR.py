#####################################################################################
##                                                                                 ##
##          File contains main code for 2DIR spectrum generation (images)          ##
##                                                                                 ##
#####################################################################################


import copy
import numpy as np

np.set_printoptions(linewidth=100000)

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


class SpectrumEVV:

    def __init__(self, w1, w2, fundamentals, Gamma, avrg_ones):

        # defines the grid of spectrum (pixels)
        self.w1, self.w2 = np.meshgrid(w1, w2)
        self.shape2d = self.w1.shape

        self.fundamentals = fundamentals

        # non-zero fermi terms
        self.fermirm = 0.0001

        elements = list(self.fundamentals.keys())
        pairs = [a + b for i, a in enumerate(elements) for b in elements[i:]]
        triples = [a + b + c for i, a in enumerate(elements) for j, b in enumerate(elements[i:]) for c in elements[j:]]

        # anharmonic correction
        corr = 1
        self.Delta = {k: corr*sum([self.fundamentals[i] for i in k]) for k in [*pairs, *triples]}
        self.all_states = copy.deepcopy(self.fundamentals)
        self.all_states.update(self.Delta)

        self.Gamma = Gamma
        self.avrg_ones = avrg_ones

    def addTerms(self, electrical_terms, mechanical_terms, el_avrg, mech_avrg):

        self.electr_funs = [w_mn_prod(i) for i in electrical_terms]
        self.mech_funs = [w_mn_prod(*i) for i in mechanical_terms]

        self.electric_avrg = el_avrg
        self.mechanical_avrg = mech_avrg

        self.combofuns = [dict(zip(self.electr_funs, self.electric_avrg)),
                          dict(zip(self.mech_funs, self.mechanical_avrg))]

        self.coords_ab = get_abc(2, len(self.fundamentals)) if electrical_terms is not None else []
        self.coords_abc = get_abc(3, len(self.fundamentals)) if mechanical_terms is not None else []

    # derivs from rsp_tensor file + MOLECULE.INP # fixme: new way is to run PyOpenrsp
    #  (mu_Q, mu_QQ, alpha_Q, alpha_QQ, F_abc)
    def getDerivs(self, source='files', molfile=None, rspfile=None):

        import openrsp_tensor_reader as orspReader

        if source == 'files' and molfile is not None:

            props_list, tens_list = orspReader.read_openrsp_tensor_file(rspfile)
            print(props_list[0], props_list[0].hasTensor)

            for i in range(len(props_list)):
                props_list[i].addTensor(tens_list[i])

            # mu_Q, mu_QQ, alpha_Q, alpha_QQ, F_abc
            transf_props_list = []

            # cartesian basis to normal mode  # todo 3 is here; after reading openrsp tensors
            for prop in props_list[:-1]:
                trsfMatrix = orspReader.get_transfMat_Scpy(molfile, rspfile)
                transformed = orspReader.cart2normal(prop, trsfMatrix)
                transf_props_list.append(transformed)

            return dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], transf_props_list))

        elif source == 'files' and molfile is None:
            # FIXME : the simplest model data (verification of 2dir implementation)

            aa = len(self.fundamentals)
            data = [np.zeros(i) for i in [(aa,3), (aa,aa,3), (aa,3,3), (aa,aa,3,3), (aa,aa,aa)]]

            data[0][:,2].fill(1.)
            data[1][:, :, 2].fill(1.)
            data[2][:, 2, :].fill(1.)
            data[2][:, :, 2].fill(1.)

            data[3][:, :, 2, :].fill(1.)
            data[3][:, :, :, 2].fill(1.)

            data[-1].fill(1.)

            return dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], data))

        elif source == 'pyorsp':
            # run 2dir pyopenrsp calculation and get necessary tensors
            import pyrsp_2dir
            poprsp = []
            # for i in pyrsp_2dir.props_list:
            #     orspReader.rspProperty()
            return dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], pyrsp_2dir.props_list))

        else:
            print("Invalid combination of arguments for getDerivs method")

    # gamma all for normal modes (a, b, c)
    def gamma_mn(self, a, b, c=False, molfile=None, rspfile=None):

        # components lists for averaging: terms of the sum
        gammaCompsAll = getting_abcgreek4avrg(num_f=4)

        # getting derivs
        data = self.getDerivs(molfile=molfile, rspfile=rspfile)

        # orientational average for prop tensors

        # do somewhere else?
        # self.addTerms()

        # if 'c' is not provided, compute electrical anharmonicity
        if type(c) == bool:

            total_sum_el = np.zeros(self.shape2d, dtype='complex128')
            prefac_el = 1 / self.fundamentals[str(a)] / self.fundamentals[str(b)]

            for elfun, elavrg in self.combofuns[0].items():
                # average for given (a, b) for a given term
                averg_el1 = avrg_abc(elavrg, data, [a, b], gammaCompsAll)

                total_sum_el += prefac_el * averg_el1 * elfun(self.all_states, self.w1, self.w2, self.Gamma, (a, b))

            return total_sum_el / 24

        else:

            total_sum_mech = np.zeros(self.shape2d, dtype='complex128')

            # mechanical
            prefac_mech = 1 / self.fundamentals[str(a)] / self.fundamentals[str(b)] / self.fundamentals[str(c)]

            for mechfun, mechavrg in self.combofuns[1].items():
                averg_mech1 = avrg_abc(mechavrg[:-1], data, [a, b, c], gammaCompsAll)
                abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
                # print('hiii', mechavrg)
                indx = tuple([abc[j] for j in mechavrg[-1]])
                # print(indx, mechavrg[-1], [a, b, c], 'F_abc', flush=True)
                # print(data['F_abc'], flush=True)
                F = data['F_abc'][indx]
                # print(F)
                total_sum_mech += prefac_mech * averg_mech1 * F * mechfun(self.all_states,
                                                                          self.w1, self.w2, self.Gamma, (a, b, c))

            return -total_sum_mech / 48.

    def totInt(self, Qab, Qabc):

        X, Y = self.w1, self.w2

        Z = np.zeros(self.shape2d, dtype='complex128')

        for i in Qab:
            Z += self.gamma_mn(i[0], i[1])

        for i in Qabc:
            Z += self.gamma_mn(i[0], i[1], i[2])

        return X, Y, Z

    def plot2D(self, figname, w1mw2=False, surface=False):
        import matplotlib.pyplot as plt
        # from matplotlib import cm, ticker, colors

        Qab, Qabc = self.coords_ab, self.coords_abc

        # PLOTTING
        if not surface:
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 8)
        else:
            ax = plt.figure(figsize=(10, 8)).add_subplot(projection='3d')

        X, Y, Z = self.totInt(Qab, Qabc)

        ax.set_xlabel('w1', fontsize=18)

        if w1mw2:
            y = (X - Y)
            ax.set_ylabel('-(w1-w2)', fontsize=18)

        else:
            y = Y
            ax.set_ylabel('w2', fontsize=18)

        if not surface:
            cp = ax.contourf(X, y, abs(Z) ** 2, 8, cmap='magma')
            fig.colorbar(cp)
        else:
            ax.plot_surface(X, y, abs(Z) ** 2, cmap='magma')

        ax.set_xticks(np.linspace(min(X.flatten()), max(X.flatten()), 10))
        ax.set_yticks(np.linspace(min(y.flatten()), max(y.flatten()), 10))
        plt.tight_layout()
        plt.show()

        plt.savefig(f'{figname}.png')
        return abs(Z) ** 2

    def plot2D_surface(self, Q_ab, Q_abc=[], w1mw2=False):
        import matplotlib.pyplot as plt

        X, Y, Z = self.totInt(Q_ab, Q_abc)

        ax = plt.figure(figsize=(10, 8)).add_subplot(projection='3d')

        ax.set_xlabel('w1', fontsize=18)

        if w1mw2:
            y = (X - Y)
            ax.set_ylabel('-(w1-w2)', fontsize=18)

        else:
            y = Y
            ax.set_ylabel('w2', fontsize=18)

        pp = ax.plot_surface(X, y, abs(Z) ** 2, cmap='magma')
        plt.tight_layout()
        plt.show()
        return pp


# Qab = get_abc(2, len(self.fundamentals))
# Qab = [[0, 0], [0, 1]]
def get_abc(nloops, abcrange):
    stacklist = []
    for i in range(nloops):
        stacklist.append(np.arange(abcrange))

    return np.stack(np.meshgrid(*stacklist), axis=-1).reshape(-1, nloops)


# num_f = 4 -four-wave mixing
def getting_abcgreek4avrg(num_f):
    import macroscopics
    pol_g = macroscopics.get_iso_f(num_f)
    new = np.array([pol[0] for pol in pol_g], dtype='object').reshape(-1, num_f)

    return new


# works with formula = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))]
def avrg_abc(formula, data, normalModes, gammaCompsAll):
    avrg = 0.

    for gammaComps in gammaCompsAll:

        alpha, beta, gamma, delta = gammaComps
        abc = dict(zip(['a', 'b', 'c'], normalModes))
        abc_greek = {0: (beta,), 1: (alpha, delta,), 2: (gamma,)}

        tot = 1.

        # [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',))]
        # f - tuple ('mu_Q', ('a',))
        for i, f in enumerate(formula):
            # index for tensor component
            # f[1] - tuple ('a',)
            indx = tuple(abc[j] for j in f[1]) + abc_greek[i]

            # f[0] - 'mu_Q'
            tot *= data[f[0]][indx]
        avrg += tot

    return avrg / 15


# t1el = lambda w, w1, w2, Gamma, a, b: \
#     1 / (w[''.join(sorted(str(a) + str(b)))] - w[str(a)] + w1 - w2 - 1j * Gamma) / (
#             0. - w[str(a)] + w1 - 1j * Gamma)
#
# t2el = lambda w, w1, w2, Gamma, a, b: \
#     1 / (w[str(b)] - w[str(a)] + w1 - w2 - 1j * Gamma) / (0. - w[str(a)] + w1 - 1j * Gamma)
#
# t1mech = lambda w, w1, w2, Gamma, a, b, c: \
#     1 / (w[''.join(sorted(str(a) + str(b)))] - w[str(a)] + w1 - w2 - 1j * Gamma) / (
#             0. - w[str(a)] + w1 - 1j * Gamma) * \
#     (1 / (w[''.join(sorted(str(a) + str(b) + str(c)))] - 0. + self.fermirm) \
#      + 1 / (w[str(c)] - w[''.join(sorted(str(a) + str(b)))] + self.fermirm))
#
# t2mech = lambda w, w1, w2, Gamma, a, b, c: \
#     1 / (w[str(c)] - w[str(a)] + w1 - w2 - 1j * Gamma) / (0. - w[str(a)] + w1 - 1j * Gamma) * \
#     (1 / (w[''.join(sorted(str(a) + str(b)))] - w[str(c)] + self.fermirm) \
#      + 1 / (w[''.join(sorted(str(a) + str(b)))] - w[str(a)] + self.fermirm))

# function generator
def w_mn_prod(subscripts, fermi=None):

    m1n1m2n2 = [i.split(',') for i in subscripts]
    if fermi is not None:
        fermi = [i.split(',') for i in fermi]

    def res1(w_all, w1, w2, Gamma, abctuple, m1n1m2n2=m1n1m2n2, fermi=fermi):
        letters = ['a', 'b', 'c', 'zero'] if len(abctuple) == 3 else ['a', 'b', 'zero']
        dictabc = dict(zip(letters, abctuple + tuple(['zero'])))
        w_all['zero'] = 0.

        wm1 = ''.join(sorted([str(dictabc[i]) for i in m1n1m2n2[0][0].split('+')]))
        wn1 = ''.join(sorted([str(dictabc[i]) for i in m1n1m2n2[0][1].split('+')]))
        wm2 = ''.join(sorted([str(dictabc[i]) for i in m1n1m2n2[1][0].split('+')]))
        wn2 = ''.join(sorted([str(dictabc[i]) for i in m1n1m2n2[1][1].split('+')]))

        if fermi is None:
            return 1 / (w_all[wm1] - w_all[wn1] + w1 - w2 - 1j * Gamma) / (w_all[wm2] - w_all[wn2] + w1 - 1j * Gamma)

        else:
            w_fr1 = ''.join(sorted([str(dictabc[i]) for i in fermi[0][0].split('+')]))
            w_fr2 = ''.join(sorted([str(dictabc[i]) for i in fermi[0][1].split('+')]))

            return (1 / (w_all[wm1] - w_all[wn1] + w1 - w2 - 1j * Gamma) / (
                    w_all[wm2] - w_all[wn2] + w1 - 1j * Gamma)) * (1 / (w_all[w_fr1] + 0.0001) + 1 / (w_all[w_fr2] + 0.0001))

    return res1
