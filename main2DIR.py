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

    def __init__(self, w1, w2, fundamentals, Gamma, avrg_ones, Delta=None, margin=10):

        # defines the grid of spectrum (pixels)
        self.w1_mesh, self.w2_mesh = np.meshgrid(w1, w2)
        self.w1, self.w2 = np.array(w1), np.array(w2)
        self.shape2d = self.w1_mesh.shape

        self.fundamentals = fundamentals

        # non-zero fermi terms
        self.fermirm = 0.0001

        # margin for higher diagonal
        self.margin = margin

        elements = list(self.fundamentals.keys())
        pairs = [a + b for i, a in enumerate(elements) for b in elements[i:]]
        triples = [a + b + c for i, a in enumerate(elements) for j, b in enumerate(elements[i:]) for c in elements[j:]]

        # anharmonic correction
        corr = 1
        if Delta is None:
            self.Delta = {k: corr * sum([self.fundamentals[i] for i in k]) for k in [*pairs, *triples]}
        else:
            self.Delta = Delta
        # print('delta', self.Delta)
        self.all_states = copy.deepcopy(self.fundamentals)
        self.all_states.update(self.Delta)

        self.Gamma = Gamma
        self.avrg_ones = avrg_ones

    def addTerms(self, electrical_terms, mechanical_terms, el_avrg, mech_avrg):

        self.electr_funs = [w_mn_prod(i, margin=self.margin) for i in electrical_terms]
        print(electrical_terms, 'electrical_terms')

        self.mech_funs = [w_mn_prod(*i) for i in mechanical_terms]
        print(mechanical_terms, 'mechanical_terms')
        # for tt in mechanical_terms:
        #     w_mn_prod(*i, margin=self.margin)
        #     for i in mechanical_terms

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
        if self.avrg_ones:
            return 'ones'

        if source == 'files' and molfile is not None:

            props_list, tens_list = orspReader.read_openrsp_tensor_file(rspfile)
            # print(props_list[0], props_list[0].hasTensor)

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
    def gamma_mn(self, style, a, b, c=False, molfile=None, rspfile=None):

        # components lists for averaging: terms of the sum
        gammaCompsAll = getting_abcgreek4avrg(num_f=4)

        # getting derivs
        data = self.getDerivs(molfile=molfile, rspfile=rspfile)

        # orientational average for prop tensors

        # do somewhere else?
        # self.addTerms()

        if style == 'surface' or style == 'contour':
            shape = self.shape2d

        else:
            shape = (len(self.w1),)

        # if 'c' is not provided, compute electrical anharmonicity
        if type(c) == bool:

            total_sum_el = np.zeros(shape, dtype='complex128')
            prefac_el = 1 / self.fundamentals[str(a)] / self.fundamentals[str(b)]

            for el_func, elavrg in self.combofuns[0].items():
                # average for given (a, b) for a given term
                averg_el1 = avrg_abc(elavrg, data, [a, b], gammaCompsAll)
                # res1(w_all, w1, w2, Gamma, abctuple, m1n1m2n2=m1n1m2n2, fermi=fermi)

                if style == 'surface' or style == 'contour':
                    # print('type', type(self.w1_mesh))
                    # print((self.w2_mesh > self.w1_mesh).all())
                    total_sum_el += prefac_el * averg_el1 * el_func(self.all_states, self.w1_mesh, self.w2_mesh,
                                                                        self.Gamma, (a, b))
                else:
                    total_sum_el = []
                    for comp in range(len(self.w1)):
                        # for kk in
                        val = prefac_el * averg_el1 * el_func(self.all_states, self.w1[comp], self.w2[comp], self.Gamma,
                                                            (a, b))
                        total_sum_el.append(val)
                        # print(val, 'val', self.w1[comp], self.w2[comp], (a, b))
                        # print(prefac_el , averg_el1 , el_func(self.all_states, self.w1[comp], self.w2[comp], self.Gamma,
                        #                                     (a, b)), '\n')
                        # total_sum_el += prefac_el * averg_el1 * el_func(self.all_states, self.w1[comp], self.w2[comp], self.Gamma, (a, b))
                    total_sum_el = np.array(total_sum_el)

            return total_sum_el / 24.

        else:

            total_sum_mech = np.zeros(shape, dtype='complex128')

            # mechanical
            prefac_mech = 1 / self.fundamentals[str(a)] / self.fundamentals[str(b)] / self.fundamentals[str(c)]

            for mech_func, mechavrg in self.combofuns[1].items():
                averg_mech1 = avrg_abc(mechavrg[:-1], data, [a, b, c], gammaCompsAll)
                abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
                # print('hiii', mechavrg)
                indx = tuple([abc[j] for j in mechavrg[-1]])
                # print(indx, mechavrg[-1], [a, b, c], 'F_abc', flush=True)

                if data=='ones':
                    F = 1.
                else:
                    F = data['F_abc'][indx]
                # print(F)

                if style == 'surface' or style == 'contour':
                    total_sum_mech += prefac_mech * averg_mech1 * F * mech_func(self.all_states,
                                                                                self.w1_mesh, self.w2_mesh, self.Gamma,
                                                                                (a, b, c))
                else:
                    total_sum_mech = []
                    for comp in range(len(self.w1)):
                        val = prefac_mech * averg_mech1 * F * mech_func(self.all_states, self.w1[comp], self.w2[comp], self.Gamma,
                                                              (a, b, c))
                        total_sum_mech.append(val)
                    total_sum_mech = np.array(total_sum_mech)

            return -total_sum_mech / 48.

    def totInt(self, style):

        Qab, Qabc = self.coords_ab, self.coords_abc

        if style == 'surface' or style == 'contour':
            Z = np.zeros(self.shape2d, dtype='complex128')
        else:
            Z = np.zeros((len(self.w1),), dtype='complex128')

        # print('Z here?????', Z)

        Qab_contrib_dict = {}
        Qabc_contrib_dict = {}

        for i in Qab:
            # print(i)

            # print(self.gamma_mn(style, i[0], i[1]))
            contrib_ab = self.gamma_mn(style, i[0], i[1])
            # print('contrib_ab', contrib_ab, i[0], i[1])
            Qab_contrib_dict[tuple(i)] = contrib_ab
            Z += contrib_ab

        for i in Qabc:
            # print(i)
            # print('Z here again', Z)
            # print('gamma_mn', self.gamma_mn(style, i[0], i[1]))
            contrib_abc = self.gamma_mn(style, i[0], i[1], i[2])
            Qabc_contrib_dict[tuple(i)] = contrib_abc
            Z += contrib_abc
        # if style == 'scatter':
            # print('printing Qab_contrib_dict for scatter\n')
            # for x in Qab_contrib_dict: print(x, Qab_contrib_dict[x])
            # print('Qabc_contrib_dict\n', Qabc_contrib_dict)
        return Z

    def plot2D(self, figname, w1mw2=False, style='surface'):
        import time
        c0 = time.process_time()

        import matplotlib
        # plt.ion()
        # matplotlib.use('TkAgg')
        # matplotlib.use('Agg')
        # matplotlib.use('QtAgg')
        # matplotlib.rcParams['backend'] = 'QtAgg'
        import matplotlib.pyplot as plt
        from matplotlib import cm, ticker, colors
        def custom_format_coord(x, y):
            return f'x = {x:.2f}\n  y = {y:.2f}'  # Separate x and y on different lines



        # PLOTTING
        if style == 'surface':
            ax = plt.figure(figsize=(10, 8)).add_subplot(projection='3d')
        else:
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 8)
        # Set the custom format using Axes.format_coord
        ax.format_coord = custom_format_coord
        # points
        Z = self.totInt(style)
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
            print('just before ax.scatter')

            # cp = ax.contourf(X, y, abs(Z) ** 2, 8, cmap='magma')
            # cp = ax.contour(X, y, abs(Z) ** 2, 8, cmap='magma')
            # cp = ax.scatter(X, y, color="green")
            print(X.size, y.size, 'X.y size')
            print(self.w1.size, self.w2.size)
            cp = ax.scatter(X, y, c=abs(Z) ** 2, cmap='brg')
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

        fig.colorbar(cp)

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
        print('before if len(positions[0]) > 100 and w1mw2')
        if len(positions[0]) > 100 and w1mw2:
            for pp in list(self.fundamentals.values()):
                plt.plot((pp, pp), (min(y.flatten()), max(y.flatten())), 'k-', linewidth=0.3)
                ax.text(pp-2.0, min(y.flatten())+3., f'{pp}', fontsize=9)
            for dd in [17., 22., 32., 37.]:
                plt.plot((min(X.flatten()), max(X.flatten())), (dd, dd), 'k-', linewidth=0.3)
                ax.text(min(X.flatten()) + 3., dd + 1.0, f'{dd}', fontsize=9)
            print('just before plt.plot')

            if w1mw2:
                plt.plot((min(X.flatten()), max(X.flatten())), (0., 0.), 'r-', linewidth=0.8)
            else:
                plt.plot((min(X.flatten()), max(X.flatten())), (min(X.flatten()), max(X.flatten())), 'r-', linewidth=0.8)
            print('before plt.tight_layout()')
        plt.tight_layout()
        # matplotlib.pyplot.show()
        # % matplot plt

        # matplotlib.pyplot.savefig(f'./pics/{figname}.png')

        c1 = time.process_time()
        print('plot2D', c1-c0)
        return abs(Z) ** 2, fig


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

    if data=='ones':
        return 1.
    else:
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


# function generator
def w_mn_prod(subscripts, fermi=None, margin=10):
    m1n1m2n2 = [i.split(',') for i in subscripts]
    print(m1n1m2n2)
    if fermi is not None:
        fermi = [i.split(',') for i in fermi]

    def function(w_all, w1, w2, Gamma, abctuple, m1n1m2n2=m1n1m2n2, fermi=fermi):
        # print('type(w1)', type(w1))

        letters = ['a', 'b', 'c', 'zero'] if len(abctuple) == 3 else ['a', 'b', 'zero']
        dictabc = dict(zip(letters, abctuple + tuple(['zero'])))
        w_all['zero'] = 0.
        # print(m1n1m2n2)

        # .join(sorted([str(dictabc[i]) for i in m1n1m2n2[0][0].split('+')]))
        wm1 = ''.join(sorted([str(dictabc[i]) for i in m1n1m2n2[0][0].split('+')]))
        wn1 = ''.join(sorted([str(dictabc[i]) for i in m1n1m2n2[0][1].split('+')]))
        wm2 = ''.join(sorted([str(dictabc[i]) for i in m1n1m2n2[1][0].split('+')]))
        wn2 = ''.join(sorted([str(dictabc[i]) for i in m1n1m2n2[1][1].split('+')]))

        if fermi is None:
            # print('w_all[wm1] - w_all[wn1] + w1 - w2', w_all[wm1], w_all[wn1], w1, w2)
            print('w1, w2, margin', margin)
            # removes lower diagonal with margin 4
            return np.where(w2-margin > w1, 1 / (w_all[wm1] - w_all[wn1] + w1 - w2 - 1j * Gamma) / (w_all[wm2] - w_all[wn2] + w1 - 1j * Gamma), 0.)

        else:
            w_fr1 = ''.join(sorted([str(dictabc[i]) for i in fermi[0][0].split('+')]))
            w_fr2 = ''.join(sorted([str(dictabc[i]) for i in fermi[0][1].split('+')]))

            return (1 / (w_all[wm1] - w_all[wn1] + w1 - w2 - 1j * Gamma) / (
                    w_all[wm2] - w_all[wn2] + w1 - 1j * Gamma)) * (
                    1 / (w_all[w_fr1] + 0.0001) + 1 / (w_all[w_fr2] + 0.0001))

    return function
