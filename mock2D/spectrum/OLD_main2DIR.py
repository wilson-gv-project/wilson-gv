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

    def __init__(self, w1, w2, fundamentals, Delta=None, margin=10):

        # defines the grid of spectrum (pixels)
        self.w1_mesh, self.w2_mesh = np.meshgrid(w1, w2)
        self.w1, self.w2 = np.array(w1), np.array(w2)
        self.shape2d = self.w1_mesh.shape
        print('len(self.w1)', len(self.w1))
        print('len(self.w2)', len(self.w2))

        print('self.w1_mesh.shape/shape_2d', self.w1_mesh.shape)
        print('self.w2_mesh.shape', self.w2_mesh.shape)
        self.fundamentals = fundamentals

        # non-zero fermi terms
        self.fermirm = 0.0001

        # margin for higher diagonal
        self.margin = margin

        elements = list(self.fundamentals.keys())
        pairs = [(a, b) for i, a in enumerate(elements) for b in elements[i:]]
        triples = [(a, b, c) for i, a in enumerate(elements) for j, b in enumerate(elements[i:]) for c in elements[j:]]
        # print(pairs)
        # anharmonic correction
        corr = 1
        if Delta is None:
            self.Delta = {k: corr * sum([self.fundamentals[i] for i in k]) for k in [*pairs, *triples]}
        else:
            self.Delta = Delta
        # print('delta', self.Delta)
        self.all_states = copy.deepcopy(self.fundamentals)
        self.all_states.update(self.Delta)

        # Create a list of keys to iterate over
        keys = list(self.all_states.keys())

        # Iterate over the list of keys
        for a in keys:
            if type(a) != tuple:
                # Update the dictionary without modifying it during iteration
                self.all_states[(a,)] = self.all_states.pop(a)

        # print('self.all_states', self.all_states)
        # quit()

        # self.Gamma = Gamma
        self.id = f'w1{min(self.w1)}_{max(self.w1)}w2{min(self.w2)}_{max(self.w2)}'

    def addTerms(self, electrical_terms, mechanical_terms, el_avrg, mech_avrg):

        self.electr_funs = [w_mn_prod(i, margin=self.margin) for i in electrical_terms]
        # print(electrical_terms, 'electrical_terms')

        self.mech_funs = [w_mn_prod(*i) for i in mechanical_terms]
        # print(mechanical_terms, 'mechanical_terms')
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
    # source={'source': 'files', 'molfile': None, 'rspfile': None}
    def getDerivs(self, source):

        if source['source'] == 'mock':
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

        if source['source'] == 'openrsp':
            from mock2D.fromspectroscpy import openrsp_tensor_reader as orspReader
            molfile = source['molfile']
            rspfile = source['rspfile']

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

        elif source['source'] == 'pyorsp':
            # run 2dir pyopenrsp calculation and get necessary tensors

            from frompyopenrsp import pyrsp_2dir

            return dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], pyrsp_2dir.props_list))

        elif source['source'] == 'cfour':
            import pickle
            # Data

            # Load the data structure from the pickle file
            filename = '../scriptsHPC/cfourscripts/vibdata.pkl'
            with open(filename, 'rb') as f:
                loaded_data_with_metadata = pickle.load(f)

            # Extract the metadata and data
            metadata = loaded_data_with_metadata['metadata']
            data = loaded_data_with_metadata['data']

            combd = dict(
                zip([tuple([str((t[:3] + t[5:8])[i] - 7) for i in range(3) for _ in
                            range((t[:3] + t[5:8])[i + 3])]) for
                     t in data['modes']], data['anharmonic_frequencies']))

            Delta = {k: v for k, v in combd.items() if len(k) > 1}

            filename = '../scriptsHPC/cfourscripts/dipolexyz.pkl'

            # Load the dictionaries from the file
            with open(filename, 'rb') as file:
                dipx, dipy, dipz = pickle.load(file)

            # dipall = {'x': dipx, 'y': dipy, 'z': dipz}

            labels = sorted(list(set([t[0] for t in data['modes']])))

            dmulist = []

            dmudqdict = {}
            dmudqdqdict = {}
            dq = len(labels)
            dmudqdqdict['x'] = np.zeros((dq, dq))
            dmudqdqdict['y'] = np.zeros((dq, dq))
            dmudqdqdict['z'] = np.zeros((dq, dq))

            for l in labels:
                if type(l) == int:
                    print(l, 'eruiexx')
                    dmudqdict[l] = np.zeros(3)
                    if l in dipx:
                        print('in x')
                        dmudqdict[l][0] = dipx[l]
                    if l in dipy:
                        print('in y')
                        dmudqdict[l][1] = dipy[l]
                    if l in dipz:
                        print('in z')
                        dmudqdict[l][2] = dipz[l]
                    dmulist.append(dmudqdict[l])

            for l in dipx:
                if type(l) != int:
                    if len(l) == 2:
                        dmudqdqdict['x'][(l[0] - 7, l[1] - 7)] = dipx[l]

            for l in dipy:
                if type(l) != int:
                    if len(l) == 2:
                        dmudqdqdict['y'][(l[0] - 7, l[1] - 7)] = dipy[l]

            for l in dipz:
                if type(l) != int:
                    if len(l) == 2:
                        dmudqdqdict['z'][(l[0] - 7, l[1] - 7)] = dipz[l]

            dmuarray = np.array(dmulist)

            # data is a list of np.arrays 'mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'
            data = [dmuarray, np.array([dmudqdqdict['x'], dmudqdqdict['y'], dmudqdqdict['z']]).T]
            # data = [np.zeros(i) for i in [(aa, 3), (aa, aa, 3), (aa, 3, 3), (aa, aa, 3, 3), (aa, aa, aa)]]

            # Specify the filename from which to load the dictionaries
            filename = '../scriptsHPC/cfourscripts/polarders.pkl'

            # Load the dictionaries from the file
            with open(filename, 'rb') as file:
                firstder, secder = pickle.load(file)

            polder = []
            for p in firstder:
                polder.append(firstder[p])
            data.append(np.array(polder))

            matrix = np.zeros((6, 6, 3, 3))
            indices_to_insert = list(secder.keys())
            # print(indices_to_insert)
            # Insert the matrices at the specified indices
            for index, mat in zip(indices_to_insert, list(secder.values())):
                i, j = index
                matrix[i - 7, j - 7] = mat
            data.append(matrix)

            # Specify the filename from which to load the dictionaries
            filename = '../scriptsHPC/cfourscripts/cubicarray.pkl'

            # Load the dictionaries from the file
            with open(filename, 'rb') as file:
                cubic = pickle.load(file)

            # cubicmat = np.zeros((6, 6, 6))
            # for e in cubic:
            #     # print((int(e[0]), int(e[1]), int(e[2])))
            #     els = [int(e[0]) - 7, int(e[1]) - 7, int(e[2]) - 7]
            #     import itertools
            #     permutations = list(itertools.permutations(els))
            #     for p in permutations:
            #         cubicmat[p] = e[3]
            cubicpickle = '../scriptsHPC/cfourscripts/cubicarray.pkl'
            picklefilevib = '../scriptsHPC/cfourscripts/vibdata.pkl'
            from mock2D.testmain_realdata import cubicpost
            cubicmat = cubicpost(picklefilevib, cubicpickle)
            data.append(cubicmat)

            return dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], data))

        else:
            print("Invalid data source")

    # def getDerivs(self, source='files', molfile=None, rspfile=None):
    #     from testmain_realdata import dmuarray, dmudqdqdict, firstder, secder, cubicmat
    #     # aa = len(funds)
    #     # data is a list of np.arrays 'mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'
    #     data = []
    #     # data = [np.zeros(i) for i in [(aa, 3), (aa, aa, 3), (aa, 3, 3), (aa, aa, 3, 3), (aa, aa, aa)]]
    #
    #     # data[0] = dmuarray
    #     data.append(dmuarray)
    #     # print('(aa, 3)', data[0], '\nfs')
    #     data.append(np.array([dmudqdqdict['x'], dmudqdqdict['y'], dmudqdqdict['z']]).T)
    #
    #     # print('(aa, aa, 3)', data[1], '\nfs')
    #
    #     polder = []
    #     for p in firstder:
    #         polder.append(firstder[p])
    #
    #     data.append(np.array(polder))
    #
    #     # print('(aa, 3, 3)', data[2], '\nfs')
    #
    #     matrix = np.zeros((6, 6, 3, 3))
    #     indices_to_insert = list(secder.keys())
    #     # print(indices_to_insert)
    #     # Insert the matrices at the specified indices
    #     for index, mat in zip(indices_to_insert, list(secder.values())):
    #         i, j = index
    #         matrix[i - 7, j - 7] = mat
    #
    #     # data[3] = matrix
    #     data.append(matrix)
    #     # data[4] = cubicmat
    #     data.append(cubicmat)
    #
    #     return dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], data))

    # gamma all for normal modes (a, b, c)
    def gamma_mn(self, style, source, Gamma, a, b, c=False):
        # print('.........', a, b, c)
        # components lists for averaging: terms of the sum
        gammaCompsAll = getting_abcgreek4avrg(num_f=4)
        # print('gammaCompsAll', gammaCompsAll)

        # getting derivs
        data = self.getDerivs(source=source)

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
                                                                        Gamma, (a, b))
                else:
                    total_sum_el = []
                    for comp in range(len(self.w1)):
                        # for kk in
                        val = prefac_el * averg_el1 * el_func(self.all_states, self.w1[comp], self.w2[comp], Gamma,
                                                            (a, b))
                        total_sum_el.append(val)
                        # print(val, 'val', self.w1[comp], self.w2[comp], (a, b))
                        # print(prefac_el , averg_el1 , el_func(self.all_states, self.w1[comp], self.w2[comp], Gamma,
                        #                                     (a, b)), '\n')
                        # total_sum_el += prefac_el * averg_el1 * el_func(self.all_states, self.w1[comp], self.w2[comp], Gamma, (a, b))
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
                                                                                self.w1_mesh, self.w2_mesh, Gamma,
                                                                                (a, b, c))
                else:
                    total_sum_mech = []
                    for comp in range(len(self.w1)):
                        val = prefac_mech * averg_mech1 * F * mech_func(self.all_states, self.w1[comp], self.w2[comp], Gamma,
                                                              (a, b, c))
                        total_sum_mech.append(val)
                    total_sum_mech = np.array(total_sum_mech)

            return -total_sum_mech / 48.

    def totInt(self, style, source, Gamma, savedict):

        Qab, Qabc = self.coords_ab, self.coords_abc
        # print('Qab', Qab)
        # print('Qabc', Qabc)
        # if style == 'surface' or style == 'contour':
        Z = np.zeros(self.shape2d, dtype='complex128')
        # else:
        #     Z = np.zeros((len(self.w1),), dtype='complex128')

        # print('Z here?????', Z)

        Qab_contrib_dict = {}
        Qabc_contrib_dict = {}

        elall = np.zeros(self.shape2d, dtype='complex128')
        for i in Qab:
            # print(i, '======deees===')

            # print(self.gamma_mn(style, i[0], i[1]))
            contrib_ab = self.gamma_mn(style, source, Gamma, i[0], i[1])
            # print(contrib_ab, '999999')
            # print('contrib_ab', contrib_ab, i[0], i[1])
            Qab_contrib_dict[tuple(i)] = contrib_ab
            elall += contrib_ab

        mechall = np.zeros(self.shape2d, dtype='complex128')
        for i in Qabc:
            # print(i)
            # print('Z here again', Z)
            # print('gamma_mn', self.gamma_mn(style, i[0], i[1]))
            contrib_abc = self.gamma_mn(style, source, Gamma, i[0], i[1], i[2])
            Qabc_contrib_dict[tuple(i)] = contrib_abc
            mechall += contrib_abc

        Z += mechall+elall
        with open(f'./picsnew/anharmonicities_Gamma{Gamma}.txt', 'w') as f:
            f.write(f'Mechanical mechall*100/Z: \n{mechall*100/Z}\n')
            f.write(f'Electrical elall*100/Z: \n{elall*100/Z}\n')

        key = self.id+f'_gamma{Gamma}'
        if key not in savedict:
            savedict[key] = {}  # Initialize with an empty dictionary or appropriate structure

        # Now you can safely assign the value to the nested key
        savedict[key]['mechanical'] = mechall
        savedict[key]['mechanical'] = mechall
        savedict[key]['electrical'] = elall
        savedict[key]['Qab_contrib_dict'] = Qab_contrib_dict
        savedict[key]['Qabc_contrib_dict'] = Qabc_contrib_dict

            # f.write(str(Qab_contrib_dict))
            # f.write(str(Qabc_contrib_dict))

        # print(f'Mechanical mechall*100/Z: \n{mechall*100/Z}')
        # print(f'Electrical elall*100/Z: \n{elall*100/Z}')
        # if style == 'scatter':
            # print('printing Qab_contrib_dict for scatter\n')
            # for x in Qab_contrib_dict: print(x, Qab_contrib_dict[x])
            # print('Qabc_contrib_dict\n', Qabc_contrib_dict)

        return Z, savedict

    def plot2D(self, figname, source, w1mw2=False, style='contour', Gamma=0.99):
        import time
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

    def print2file(self, figname, w1mw2, style, source, Gamma, step):
        Z, savedict = self.totInt(style, source, Gamma, {})
        X_grid, y_grid = self.w1_mesh, self.w2_mesh
        if w1mw2:
            y = -(X_grid - y_grid)
        else:
            y = y_grid

        # Calculate the squared absolute value of Z
        Z_squared_abs = abs(Z) ** 2

        # Create a meshgrid if you haven't already

        # Print the current working directory
        # print(f"Current working directory: {os.getcwd()}")

        # Prepare the data to be printed
        # Flatten the arrays and stack them column-wise
        data_to_print = np.column_stack((X_grid.flatten(), y.flatten(), Z_squared_abs.flatten()))

        # Sort the data by X and then by y
        # np.lexsort() uses the last key as the primary sort key, so we pass y first and then X
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


    # def plot2Dplotly(self, figname, source, w1mw2=False, style='contour', Gamma=0.99):
    #     import plotly.graph_objects as go
    #     import numpy as np
    #
    #     # Assuming self.totInt, self.w1_mesh, self.w2_mesh, self.w1, self.w2, and self.fundamentals are defined
    #     Z = self.totInt(style, source, Gamma)
    #     Z_positive = np.abs(Z) ** 2
    #
    #     X, Y = self.w1_mesh, self.w2_mesh
    #
    #     if w1mw2:
    #         y = -(X - Y)
    #     else:
    #         y = Y
    #
    #     # Define the minimum and maximum values for the color scale
    #     min_value = np.min(Z_positive[Z_positive > 0]) if np.any(Z_positive > 0) else 1e-30
    #     max_value = np.max(Z_positive)
    #
    #     # Define the color scale
    #     color_scale = 'haline'  # or any other color scale
    #
    #     # Create the contour plot
    #     fig = go.Figure(data=[go.Contour(
    #         z=Z_positive,
    #         x=X,
    #         y=y,
    #         colorscale=color_scale,
    #         colorbar=dict(
    #             title='Intensity',
    #             tickvals=[min_value, max_value],
    #             ticktext=[f'{min_value:.2e}', f'{max_value:.2e}'],
    #         ),
    #         contours=dict(
    #             coloring='fill',
    #             showlabels=False,  # show labels on contours
    #             showlines=False,
    #         ),
    #         hoverinfo='x+y+z',  # show x, y, and z values when hovering
    #         autocontour=True,  # automatically determine contour levels
    #     )])
    #
    #     # Update layout for x-axis and y-axis
    #     fig.update_layout(
    #         xaxis=dict(
    #             title='w1',
    #             ticks='outside',
    #             tickwidth=2,
    #             tickcolor='crimson',
    #             ticklen=10,
    #             showgrid=True,
    #             gridcolor='LightPink',
    #             gridwidth=1,
    #         ),
    #         yaxis=dict(
    #             title='w2' if not w1mw2 else 'w2-w1',
    #             ticks='outside',
    #             tickwidth=2,
    #             tickcolor='crimson',
    #             ticklen=10,
    #             showgrid=True,
    #             gridcolor='LightPink',
    #             gridwidth=1,
    #         )
    #     )
    #
    #     # To set a logarithmic color scale, we need to adjust the z-axis to a log scale
    #     fig.update_traces(contours_coloring='fill', z=np.log10(Z_positive))
    #
    #     # Update color bar to reflect log scale
    #     fig.update_layout(coloraxis_colorbar=dict(
    #         title='Log Intensity',
    #         tickvals=np.log10([min_value, max_value]),
    #         ticktext=[f'{min_value:.2e}', f'{max_value:.2e}'],
    #     ))
    #
    #     # Convert the figure to an HTML div string
    #     plot_div = fig.to_html(full_html=False)
    #
    #     # Now you can embed `plot_div` into your webpage's HTML where you want the plot to appear.
    #     # For example, you can write it to an HTML file:
    #     with open('plot.html', 'w') as f:
    #         f.write(plot_div)

    # def plot2Dplotly(self, figname, source, w1mw2=False, style='scatter', threshold=0):
    #     print('\nplot2Dplotly')
    #     # if style=='scatter':
    #     #     import numpy as np
    #     #     import plotly.graph_objects as go
    #     #
    #     #     Z = self.totInt(style, source)
    #     #     X, Y = self.w1_mesh, self.w2_mesh
    #     #     if w1mw2:
    #     #         y = -(X - Y)
    #     #     else:
    #     #         y = Y
    #     #     # Assuming X, y, and Z are defined and have the same meaning as in the first snippet
    #     #     Z_squared_abs = abs(Z) ** 2
    #     #
    #     #     print('-----------')
    #     #     print(X, X.shape[0]*X.shape[1])
    #     #     print(y)
    #     #
    #     #     # Flatten the X and Y arrays
    #     #     X_flat = X.flatten()
    #     #     Y_flat = y.flatten()
    #     #     print('---------')
    #     #     print(X_flat, X_flat.shape)
    #     #     print(Y_flat)
    #     #
    #     #     # Flatten the Z array and sort it along with X and Y
    #     #     data_to_print = np.column_stack((X_flat, Y_flat, Z_squared_abs.flatten()))
    #     #     sorted_indices = np.lexsort((data_to_print[:, 1], data_to_print[:, 0]))
    #     #     sorted_data = data_to_print[sorted_indices]
    #     #
    #     #     # Extract the sorted Z values and take the logarithm base 10
    #     #     sorted_Z = Z_squared_abs.flatten()[sorted_indices]
    #     #     Z_positive_flat = np.log10(sorted_Z)
    #     #     print(max(Z_positive_flat), min(Z_positive_flat), Z_positive_flat)
    #     #     print(max(Z_squared_abs.flatten()), np.log10(max(Z_squared_abs.flatten())), Z_squared_abs.flatten())
    #     #     bins = np.arange(-45, -9, 1)  # Adjust the range and step as needed
    #     #
    #     #     meshgrid_filename = f'./pics/meshgrid_data_{figname}_{Gamma}_sp8_new.txt'
    #     #     with open(meshgrid_filename, 'w') as f:
    #     #         # Write the header
    #     #         f.write("X, y, abs(Z)^2\n")
    #     #         # Write the sorted data
    #     #         for row in sorted_data:
    #     #             f.write(f"{row[0]}, {row[1]}, {row[2]}\n")
    #     #     print(f"Data has been printed to '{meshgrid_filename}'")
    #     #
    #     #     z_squared_abs_filename = f'./pics/z_squared_abs_data_{figname}_{Gamma}_sp8_new.txt'
    #     #     with open(z_squared_abs_filename, 'w') as f:
    #     #         # Write the header
    #     #         f.write("abs(Z)^2\n")
    #     #         # Write the sorted data
    #     #         for value in sorted_Z:
    #     #             f.write(f"{value}   {np.log10(value)}\n")
    #     #     print(f"Data has been printed to '{z_squared_abs_filename}'")
    #     #
    #     #     # Calculate the histogram
    #     #     hist, bin_edges = np.histogram(Z_positive_flat, bins=bins)
    #     #     print(hist)
    #     #     tot = 0.
    #     #     # Print the number of occurrences for each order of magnitude
    #     #     for i in range(len(bins) - 1):
    #     #         tot += hist[i] * 100 / sum(hist)
    #     #         print(f"Order of magnitude range [{bins[i]}, {bins[i + 1]}): {hist[i]} occurrences, "
    #     #               f"{hist[i] * 100 / sum(hist)} %, total now {tot}")
    #     #     # Define a threshold for filtering
    #     #     # threshold = threshold  # Define your threshold value here
    #     #
    #     #     # Create a mask for values above the threshold
    #     #     mask = Z_positive_flat > threshold
    #     #
    #     #     # Apply the mask to the flattened arrays
    #     #     X_filtered = X_flat[mask]
    #     #     Y_filtered = Y_flat[mask]
    #     #     Z_positive_filtered = Z_positive_flat[mask]
    #     #     print(X_filtered)
    #     #     print(Y_filtered)
    #     #     print(Z_positive_filtered)
    #     #     # Define the color scale for the plot
    #     #     color_scale = 'haline'  # or any other color scale
    #     #
    #     #     # Create the scatter plot using the filtered data
    #     #     fig = go.Figure(data=[go.Scatter(
    #     #         x=X_filtered,
    #     #         y=Y_filtered,  # Use y_filtered directly if you don't need to transform it
    #     #         mode='markers',
    #     #         marker=dict(
    #     #             size=10,
    #     #             color=Z_positive_filtered,  # Use the filtered Z values for coloring
    #     #             colorscale=color_scale,
    #     #             colorbar=dict(
    #     #                 title='Intensity'
    #     #             ),
    #     #             cmin=np.min(Z_positive_filtered),
    #     #             cmax=np.max(Z_positive_filtered)
    #     #         )
    #     #     )])
    #     #     fig.update_layout(
    #     #         xaxis_title='X',
    #     #         yaxis_title='y'
    #     #     )
    #
    #     if style=='scatter':
    #         import plotly.graph_objects as go
    #         import numpy as np
    #
    #         # Assuming self.totInt, self.w1_mesh, self.w2_mesh, self.w1, self.w2, and self.fundamentals are defined
    #         Z = self.totInt(style, source, Gamma)
    #         Z_positive = np.abs(Z) ** 2
    #         print('___________________\n', max(Z_positive.flatten()))
    #         X, Y = self.w1_mesh, self.w2_mesh
    #
    #         if w1mw2:
    #             y = -(X - Y)
    #         else:
    #             y = Y
    #
    #         # Sample points for the scatter plot
    #         # You can adjust the number of points by changing the step size in np.arange
    #         step_size = 20  # for example, take every 10th point
    #         x_sampled = X[::step_size, ::step_size].flatten()
    #         y_sampled = y[::step_size, ::step_size].flatten()
    #         z_sampled = np.log10(Z_positive[::step_size, ::step_size].flatten())
    #
    #         # Define the threshold value for Z
    #         threshold_value = 5. * 10 ** -14  # This is the actual value, not the log
    #
    #         # Create a mask for values above the threshold
    #         mask = Z > threshold_value
    #
    #         # Apply the mask to X, Y, and Z to get the points above the threshold
    #         x_filtered = X[mask]
    #         y_filtered = Y[mask]
    #         z_filtered = Z_positive[mask]
    #
    #         # Now you can print the length and the values
    #         print(len(z_filtered.flatten()))
    #         np.set_printoptions(linewidth=250, precision=10)
    #         print(list(z_filtered)[:20])
    #         # print(tuple(zip(x_filtered, y_filtered, z_filtered)))
    #
    #         # Create the scatter plot
    #         fig = go.Figure(data=[go.Scatter(
    #             x=x_filtered.flatten(),
    #             y=y_filtered.flatten(),
    #             mode='markers',
    #             marker=dict(
    #                 size=5,
    #                 color=z_filtered.flatten(),  # set color equal to a variable
    #                 colorscale='haline',  # choose a colorscale
    #                 colorbar=dict(title='Log Intensity'),
    #                 showscale=True
    #             ),
    #             hovertemplate='x: %{x:.4f} , y: %{y:.4f} , z: %{np.log10(marker.color):.5f}<extra></extra>',
    #         )])
    #
    #         # Update layout for x-axis and y-axis
    #         fig.update_layout(
    #             xaxis=dict(
    #                 title='w1',
    #                 ticks='outside',
    #                 tickwidth=2,
    #                 tickcolor='crimson',
    #                 ticklen=10,
    #                 showgrid=True,
    #                 gridcolor='LightPink',
    #                 gridwidth=1,
    #             ),
    #             yaxis=dict(
    #                 title='w2' if not w1mw2 else 'w2-w1',
    #                 ticks='outside',
    #                 tickwidth=2,
    #                 tickcolor='crimson',
    #                 ticklen=10,
    #                 showgrid=True,
    #                 gridcolor='LightPink',
    #                 gridwidth=1,
    #             )
    #         )
    #
    #         # Convert the figure to an HTML div string
    #         plot_div = fig.to_html(full_html=False)
    #
    #         # Now you can embed `plot_div` into your webpage's HTML where you want the plot to appear.
    #         # For example, you can write it to an HTML file:
    #         with open('scatter_plot.html', 'w') as f:
    #             f.write(plot_div)
    #
    #     elif style=='contour':
    #         import plotly.graph_objects as go
    #         import numpy as np
    #
    #         # Assuming self.totInt, self.w1_mesh, self.w2_mesh, self.w1, self.w2, and self.fundamentals are defined
    #         Z = self.totInt(style, source, Gamma)
    #         Z_positive = np.abs(Z) ** 2
    #
    #         X, Y = self.w1_mesh, self.w2_mesh
    #
    #         if w1mw2:
    #             y = -(X - Y)
    #         else:
    #             y = Y
    #
    #         # Define the minimum and maximum values for the color scale
    #         min_value = np.min(Z_positive[Z_positive > 0]) if np.any(Z_positive > 0) else 1e-30
    #         max_value = np.max(Z_positive)
    #
    #         # Define the color scale
    #         color_scale = 'haline'  # or any other color scale
    #
    #         # Create the contour plot
    #         fig = go.Figure(data=[go.Contour(
    #             z=np.log10(Z_positive),
    #             x=X,
    #             y=y,
    #             colorscale=color_scale,
    #             colorbar=dict(
    #                 title='Intensity',
    #                 tickvals=[min_value, max_value],
    #                 ticktext=[f'{min_value:.2e}', f'{max_value:.2e}'],
    #             ),
    #             contours=dict(
    #                 coloring='fill',
    #                 showlabels=False,  # show labels on contours
    #                 showlines=False,
    #             ),
    #             hovertemplate='x: %{x:.4f} , y: %{y:.4f} , z: %{z:.5f}<extra></extra>',
    #             # hoverinfo='x+y+z',  # show x, y, and z values when hovering
    #             autocontour=True,  # automatically determine contour levels
    #         )])
    #         # fig.update_traces(mode="markers+lines", hovertemplate=None)
    #
    #         # Update layout for x-axis and y-axis
    #         fig.update_layout(
    #             xaxis=dict(
    #                 title='w1',
    #                 ticks='outside',
    #                 tickwidth=2,
    #                 tickcolor='crimson',
    #                 ticklen=10,
    #                 showgrid=True,
    #                 gridcolor='LightPink',
    #                 gridwidth=1,
    #             ),
    #             yaxis=dict(
    #                 title='w2' if not w1mw2 else 'w2-w1',
    #                 ticks='outside',
    #                 tickwidth=2,
    #                 tickcolor='crimson',
    #                 ticklen=10,
    #                 showgrid=True,
    #                 gridcolor='LightPink',
    #                 gridwidth=1,
    #             )
    #         )
    #
    #         # To set a logarithmic color scale, we need to adjust the z-axis to a log scale
    #         fig.update_traces(contours_coloring='fill', z=np.log10(Z_positive))
    #
    #         # Update color bar to reflect log scale
    #         # fig.update_layout(coloraxis_colorbar=dict(
    #         #     title='Log Intensity',
    #             # tickvals=np.log10([min_value, max_value]),
    #             # ticktext=[f'{min_value:.2e}', f'{max_value:.2e}'],
    #         # ))
    #
    #     # Convert the figure to an HTML div string
    #     plot_div = fig.to_html(full_html=False)
    #
    #     # Now you can embed `plot_div` into your webpage's HTML where you want the plot to appear.
    #     # For example, you can write it to an HTML file:
    #     with open('plot.html', 'w') as f:
    #         f.write(plot_div)
    #
    #     # orders_of_magnitude = np.log10(sorted_Z)
    #     # print(orders_of_magnitude)
    #     # # Define the bins for the histogram
    #     # # For example, if you want bins from -20 to 0 (inclusive) in steps of 1
    #     # bins = np.arange(-35, -9, 1)  # Adjust the range and step as needed
    #     #
    #     # # Calculate the histogram
    #     # hist, bin_edges = np.histogram(orders_of_magnitude, bins=bins)
    #     # print(hist)
    #     # tot = 0.
    #     # # Print the number of occurrences for each order of magnitude
    #     # for i in range(len(bins) - 1):
    #     #     tot += hist[i]*100/sum(hist)
    #     #     print(f"Order of magnitude range [{bins[i]}, {bins[i + 1]}): {hist[i]} occurrences, "
    #     #           f"{hist[i]*100/sum(hist)} %, total now {tot}")


    def plot2Dseabornheatmap(self, source, w1mw2, style, Gamma):
        import pandas as pd

        X, Y = self.w1_mesh, self.w2_mesh
        if w1mw2:
            y = -(X - Y)
            ystr = 'w2-w1'
            # ax.set_ylabel('w2-w1', fontsize=18)
            # xlim = 4.
            # ax.set_xlim([xlim, max(list(self.fundamentals.values()))+3.])
            # ylim = 4.
            # ax.set_ylim([ylim, max(y.flatten())])
        else:
            y = Y
            ystr = 'w2'
            # ax.set_ylabel('w2', fontsize=18)

        Z = self.totInt(style, source, Gamma)
        Z_positive = abs(Z) ** 2

        # load the sample data
        df = pd.DataFrame({'w1': X.flatten(),
                           ystr: y.flatten(),
                           'value': Z_positive.flatten(),
                           'valuelog10': np.log10(Z_positive).flatten()})
        # print(df)
        # print(df['w1'].shape, df[ystr].shape, df['value'].shape)
        df.to_csv(r'pandas.txt', header=None, index=None, sep=' ', mode='a')
        quit()
        # pivot the dataframe from long to wide form
        result = df.pivot(index='w1', columns=ystr, values='value')

        sns.heatmap(result, annot=False, fmt="g", cmap='viridis')
        # plt.show()
        plt.savefig("seaborn_heatmap.svg", format='svg', dpi=500)

    def plot2Dmatplotlib(self, source, w1mw2, style, Gamma):
        import matplotlib.pyplot as plt
        import numpy as np

        X, Y = self.w1_mesh, self.w2_mesh
        if w1mw2:
            y = -(X - Y)
            ystr = 'w2-w1'
            ystr_mesh = ystr+'_'

        else:
            y = Y
            ystr = 'w2'
            ystr_mesh = ystr+'_'

        Z = self.totInt(style, source, Gamma)
        Z_positive = abs(Z) ** 2

        # load the sample data
        df = {
            # 'w1': X.flatten(),
            #                ystr: y.flatten(),
            #                'value': Z_positive.flatten(),
            #                'valuelog10': np.log10(Z_positive).flatten(),
                           'w1_mesh': X,
                           ystr_mesh: y,
                           'valueslog10_mesh': np.log10(Z_positive)
                           }
        # Create the contour plot
        plt.contourf(df['w1_mesh'], df[ystr_mesh], df['valueslog10_mesh'], levels=100, cmap='viridis')  # Filled contour plot
        plt.colorbar()  # Add a colorbar to show the Z scale
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Contour plot with X, Y, and Z data')
        # plt.show()
        plt.savefig(f'./pics/simplecontour.svg', dpi=500)


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
        Z_positive = abs(Z) ** 2 +10e-72
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

        print('-====----[:, 0]\n', np.log10(Z_positive)[:, 0], '\n')
        print('-====----[0. :]\n', np.log10(Z_positive)[0, :], '\n')
        print('-====----[0]\n', np.log10(Z_positive)[0], '\n')

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
                # quit()
                tot *= data[f[0]][indx]
            avrg += tot

        return avrg / 15


# function generator
def w_mn_prod(subscripts, fermi=None, margin=10):
    m1n1m2n2 = [i.split(',') for i in subscripts]
    # print(m1n1m2n2, 'm1n1m2n2')
    if fermi is not None:
        fermi = [i.split(',') for i in fermi]

    def function(w_all, w1, w2, Gamma, abctuple, m1n1m2n2=m1n1m2n2, fermi=fermi):
        # print('type(w1)', type(w1))

        letters = ['a', 'b', 'c', 'zero'] if len(abctuple) == 3 else ['a', 'b', 'zero']
        dictabc = dict(zip(letters, abctuple + tuple(['zero'])))
        w_all[('zero',)] = 0.
        # print(m1n1m2n2)

        # .join(sorted([str(dictabc[i]) for i in m1n1m2n2[0][0].split('+')]))
        # wm1 = ''.join(sorted([str(dictabc[i]) for i in m1n1m2n2[0][0].split('+')]))
        # wn1 = ''.join(sorted([str(dictabc[i]) for i in m1n1m2n2[0][1].split('+')]))
        # wm2 = ''.join(sorted([str(dictabc[i]) for i in m1n1m2n2[1][0].split('+')]))
        # wn2 = ''.join(sorted([str(dictabc[i]) for i in m1n1m2n2[1][1].split('+')]))

        wm1 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[0][0].split('+')]))
        wn1 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[0][1].split('+')]))
        wm2 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[1][0].split('+')]))
        wn2 = tuple(sorted([str(dictabc[i]) for i in m1n1m2n2[1][1].split('+')]))

        if fermi is None:
            # print('w_all[wm1] - w_all[wn1] + w1 - w2', w_all[wm1], w_all[wn1], w1, w2)
            # print('w1, w2, margin', margin)
            # removes lower diagonal with margin 4
            return np.where(w2-margin > w1, 1 / (w_all[wm1] - w_all[wn1] + w1 - w2 - 1j * Gamma) / (w_all[wm2] - w_all[wn2] + w1 - 1j * Gamma), 0.)

        else:
            w_fr1 = tuple(sorted([str(dictabc[i]) for i in fermi[0][0].split('+')]))
            w_fr2 = tuple(sorted([str(dictabc[i]) for i in fermi[0][1].split('+')]))

            return (1 / (w_all[wm1] - w_all[wn1] + w1 - w2 - 1j * Gamma) / (
                    w_all[wm2] - w_all[wn2] + w1 - 1j * Gamma)) * (
                    1 / (w_all[w_fr1] + 0.0001) + 1 / (w_all[w_fr2] + 0.0001))

    return function
