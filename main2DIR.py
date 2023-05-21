import copy
import numpy as np

# np.set_printoptions(linewidth=100000)

# class DerivResonanceTerm:
#     """
#
#     """
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

    def __init__(self, w1, w2, fundamentals, Gamma):

        # defines the grid of spectrum (pixels)
        self.w1, self.w2 = np.meshgrid(w1, w2)
        self.shape2d = self.w1.shape

        self.fundamentals = fundamentals

        # non-zero fermi terms
        self.fermirm = 0.0001

        elements = list(self.fundamentals.keys())
        pairs = [a + b for i, a in enumerate(elements) for b in elements[i:]]
        triples = [a + b + c for i, a in enumerate(elements) for j, b in enumerate(elements[i:]) for c in elements[j:]]

        self.Delta = {k: sum([self.fundamentals[i] for i in k]) for k in [*pairs, *triples]}

        self.Gamma = Gamma

        self.t1el = None
        self.t2el = None
        self.t1mech = None
        self.t2mech = None
        self.t3mech = None
        self.t4mech = None

        self.avrg_t1el = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))]
        self.avrg_t2el = [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))]
        # electric = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))],
        #             [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))]]
        self.avrg_t1mech = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',))]
        self.avrg_t2mech = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',))]
        self.avrg_t3mech = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',))]
        self.avrg_t4mech = [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',))]
        # mechanical = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',))],
        #               [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',))],
        #               [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',))],
        #               [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',))]]

    def addTerms(self):
        # 'a+b,a', '00,a'
        self.t1el = lambda w, w1, w2, Gamma, a, b: \
            1 / (w[''.join(sorted(str(a) + str(b)))] - w[str(a)] + w1 - w2 - 1j * Gamma) / (
                    0. - w[str(a)] + w1 - 1j * Gamma)

        # 'b,a', '00,a'
        self.t2el = lambda w, w1, w2, Gamma, a, b: \
            1 / (w[str(b)] - w[str(a)] + w1 - w2 - 1j * Gamma) / (0. - w[str(a)] + w1 - 1j * Gamma)

        # 'a+b,a', '00,a'
        self.t1mech = lambda w, w1, w2, Gamma, a, b, c: \
            1 / (w[''.join(sorted(str(a) + str(b)))] - w[str(a)] + w1 - w2 - 1j * Gamma) / (
                    0. - w[str(a)] + w1 - 1j * Gamma) * \
            (1 / (w[''.join(sorted(str(a) + str(b) + str(c)))] - 0. + self.fermirm) \
             + 1 / (w[str(c)] - w[''.join(sorted(str(a) + str(b)))] + self.fermirm))

        self.t2mech = lambda w, w1, w2, Gamma, a, b, c: \
            1 / (w[str(c)] - w[str(a)] + w1 - w2 - 1j * Gamma) / (0. - w[str(a)] + w1 - 1j * Gamma) * \
            (1 / (w[''.join(sorted(str(a) + str(b)))] - w[str(c)] + self.fermirm) \
             + 1 / (w[''.join(sorted(str(a) + str(b)))] - w[str(a)] + self.fermirm))

        self.t3mech = lambda w, w1, w2, Gamma, a, b, c: \
            1 / (w[''.join(sorted(str(a) + str(b)))] - w[str(a)] + w1 - w2 - 1j * Gamma) / (
                    0. - w[str(a)] + w1 - 1j * Gamma) * \
            (1 / (w[str(a)] - w[''.join(sorted(str(a) + str(b)))] + self.fermirm) \
             + 1 / (w[str(b)] - 0. + self.fermirm))

        self.t4mech = lambda w, w1, w2, Gamma, a, b, c: \
            1 / (w[''.join(sorted(str(a) + str(b)))] - w[str(a)] + w1 - w2 - 1j * Gamma) / (
                    0. - w[str(a)] + w1 - 1j * Gamma) * \
            (1 / (w[str(b)] - w[''.join(sorted(str(a) + str(b)))] + self.fermirm) \
             + 1 / (w[str(a)] - 0. + self.fermirm))

    # derivs from rsp_tensor file + MOLECULE.INP
    def getDerivs(self, test):

        if test:

            import openrsp_tensor_reader as orspReader
            props_list, tens_list = orspReader.read_openrsp_tensor_file('rsp_tensor')

            for i in range(len(props_list)):
                props_list[i].addTensor(tens_list[i])

            # mu_Q, mu_QQ, alpha_Q, alpha_QQ, F_abc
            transf_props_list = []

            # cartesian basis to normal mode
            for prop in props_list[:-1]:
                transf = orspReader.cart2normal(prop, 'MOLECULE.INP', 'rsp_tensor')
                transf_props_list.append(transf)

            # mu_Q, mu_QQ, alpha_Q, alpha_QQ, F_abc = transf_props_list

            # return mu_Q, mu_QQ, alpha_Q, alpha_QQ, F_abc
            return transf_props_list

    # gamma all for normal modes (a, b, c)
    def gamma_mn(self, a, b, c=False):
        new_states = copy.deepcopy(self.fundamentals)
        new_states.update(self.Delta)

        # components lists for averaging: terms of the sum
        gammaCompsAll = getting_abcgreek4avrg(num_f=4)

        # orientational average for prop tensors
        q = self.getDerivs(test=True)

        # getting derivs
        data = dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], q))
        self.addTerms()

        # if 'c' is not provided, compute electrical anharmonicity
        if type(c) == bool:

            total_sum_el = np.zeros(self.shape2d, dtype='complex128')
            prefac_el = 1 / self.fundamentals[str(a)] / self.fundamentals[str(b)]

            # average for given (a, b, c) for a given term
            averg_el1 = avrg_abc(self.avrg_t1el, data, [a, b], gammaCompsAll)
            total_sum_el += prefac_el * averg_el1 * self.t1el(new_states, self.w1, self.w2, self.Gamma, a, b)
            total_sum_el += prefac_el * averg_el1 * w_mn_prod(('a+b,a', '00,a'))(w_all, self.w1, self.w2, Gamma, (a, b))

            averg_el2 = avrg_abc(self.avrg_t2el, data, [a, b], gammaCompsAll)
            total_sum_el += prefac_el * averg_el2 * self.t2el(new_states, self.w1, self.w2, self.Gamma, a, b)

            return total_sum_el / 24

        else:

            total_sum_mech = np.zeros(self.shape2d, dtype='complex128')

            # mechanical
            prefac_mech = 1 / self.fundamentals[str(a)] / self.fundamentals[str(b)] / self.fundamentals[str(c)]

            averg_mech1 = avrg_abc(self.avrg_t1mech, data, [a, b, c], gammaCompsAll)
            total_sum_mech += prefac_mech * averg_mech1 * self.t1mech(new_states, self.w1, self.w2, self.Gamma, a, b, c)

            averg_mech2 = avrg_abc(self.avrg_t1mech, data, [a, b, c], gammaCompsAll)
            total_sum_mech += prefac_mech * averg_mech2 * self.t2mech(new_states, self.w1, self.w2, self.Gamma, a, b, c)

            averg_mech3 = avrg_abc(self.avrg_t1mech, data, [a, b, c], gammaCompsAll)
            total_sum_mech += (1 / 2) * prefac_mech * averg_mech3 * self.t3mech(new_states, self.w1, self.w2,
                                                                                self.Gamma, a, b, c)

            averg_mech4 = avrg_abc(self.avrg_t1mech, data, [a, b, c], gammaCompsAll)
            total_sum_mech += (1 / 2) * prefac_mech * averg_mech4 * self.t4mech(new_states, self.w1, self.w2,
                                                                                self.Gamma,
                                                                                a, b, c)

            return -total_sum_mech / 48.

    def plot2D(self, coords_ab, coords_abc=[], w1mw2=False):
        import matplotlib.pyplot as plt
        # from matplotlib import cm, ticker, colors

        X, Y = self.w1, self.w2

        Z = np.zeros(self.shape2d, dtype='complex128')

        for i in coords_ab:
            # print(i, self.fundamentals[str(i[0])], self.fundamentals[str(i[1])])
            # added = self.gamma_mn(i[0], i[1])
            # x_index = np.where(self.w1 == self.fundamentals[str(i[0])])[0][0]
            # y_index = np.where(self.w2 == self.fundamentals[str(i[1])])[0][0]

            # print(added[y_index, x_index])
            # print(added[y_index+5, x_index+5])

            Z += self.gamma_mn(i[0], i[1])

        for i in coords_abc:
            Z += self.gamma_mn(i[0], i[1], i[2])

        # PLOTTING
        fig, ax = plt.subplots()
        fig.set_size_inches(11, 9)

        ax.set_xlabel('w1', fontsize=18)

        if w1mw2:
            y = (X - Y)
            ax.set_ylabel('-(w1-w2)', fontsize=18)

        else:
            y = Y
            ax.set_ylabel('w2', fontsize=18)

        cp = ax.contourf(X, y, abs(Z) ** 2, 8, cmap='magma')

        cb = fig.colorbar(cp)

        ax.set_xticks(np.linspace(min(X.flatten()), max(X.flatten()), 10))
        ax.set_yticks(np.linspace(min(y.flatten()), max(y.flatten()), 10))

        plt.show()
        return abs(Z) ** 2

    def plot2D_surface(self, coords_ab, coords_abc=[], w1mw2=False):
        import matplotlib.pyplot as plt

        X, Y = self.w1, self.w2
        a, b, c = np.arange(len(self.fundamentals)), np.arange(len(self.fundamentals)), np.arange(
            len(self.fundamentals))

        Z = np.zeros(self.shape2d, dtype='complex128')

        for i in coords_ab:
            # print(i, self.fundamentals[str(i[0])], self.fundamentals[str(i[1])])
            # added = self.gamma_mn(i[0], i[1])
            # x_index = np.where(self.w1 == self.fundamentals[str(i[0])])[0][0]
            # y_index = np.where(self.w2 == self.fundamentals[str(i[1])])[0][0]

            # print(added[y_index, x_index])
            # print(added[y_index+5, x_index+5])

            Z += self.gamma_mn(i[0], i[1])

        for i in coords_abc:
            Z += self.gamma_mn(i[0], i[1], i[2])

        ax = plt.figure(figsize=(11, 9)).add_subplot(projection='3d')

        ax.set_xlabel('w1', fontsize=18)

        if w1mw2:
            y = (X - Y)
            ax.set_ylabel('-(w1-w2)', fontsize=18)

        else:
            y = Y
            ax.set_ylabel('w2', fontsize=18)

        # ax.plot_surface(X, y, abs(Z)**2, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
        # ax.bar3d(X, y, Z, 5, 5, (abs(Z)**2).flatten())

        pp = ax.plot_surface(X, y, abs(Z) ** 2, cmap='magma')
        plt.tight_layout()

        plt.show()
        # return fig


# coords_ab = get_abc(2, len(self.fundamentals))
# coords_ab = [[0, 0], [0, 1]]
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


def w_mn_prod(subscripts, superscripts=None):
    m1, n1 = subscripts[0].split(',')
    m2, n2 = subscripts[1].split(',')

    def res1(w_all, w1, w2, Gamma, abctuple, m=(m1, m2), n=(n1, n2), superscr=superscripts):
        letters = ['a', 'b', 'c', '00'] if len(abctuple) == 3 else ['a', 'b', '00']
        dictabc = dict(zip(letters, abctuple + tuple(['00'])))
        w_all['00'] = 0.

        wm1 = ''.join([str(dictabc[i]) for i in m[0].split('+')])
        wn1 = ''.join([str(dictabc[i]) for i in n[0].split('+')])
        wm2 = ''.join([str(dictabc[i]) for i in m[1].split('+')])
        wn2 = ''.join([str(dictabc[i]) for i in n[1].split('+')])

        # results = []
        res = 0.
        # for one in superscr:
        #     if one == '-12':
        #         results.append(1 / (w_all[wm1] - w_all[wn1] + w1 - w2 - 1j * Gamma))
        #     elif one == '-1':
        #         results.append(1 / (w_all[wm2] - w_all[wn2] + w1 - 1j * Gamma))

        return 1 / (w_all[wm1] - w_all[wn1] + w1 - w2 - 1j * Gamma) / (w_all[wm2] - w_all[wn2] + w1 - 1j * Gamma)
        # return results

    return res1


# # testing template function for w_mn terms for the whole 2d spectrum tensor
w_all = {'0': 1., '1': 5., '00': 2., '01': 6., '11': 10.}
Gamma = 10 ** (-0.3)
w1, w2 = np.arange(1, 13, 0.5), np.arange(1, 13, 0.5)
w1grid, w2grid = np.meshgrid(w1, w2)

a = w_mn_prod(('a+b,a', '00,a'))(w_all, w1grid, w2grid, Gamma, (0, 1))
print(a)

fundamentals = {'0': 1., '1': 5.}

test2d = SpectrumEVV(w1, w2, fundamentals, Gamma)
# yy = test2d.gamma_mn(1, 0, 0)
# print(yy)

# print(test2d.plot2D([(0, 0), (0, 1), (1, 0), (1, 1)]))
# print(test2d)
