#####################################################################################
##                                                                                 ##
##          File contains main code for 2DIR spectrum generation (images)          ##
##                                                                                 ##
#####################################################################################


import numpy as np
np.set_printoptions(linewidth=100000)

from .callbacks2DIR import CFOURdata, VeloxChemdata, LSDaltondata

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
        self.w1_mesh, self.w2_mesh = np.meshgrid(w1, w2)
        self.w1, self.w2 = np.array(w1), np.array(w2)
        self.shape2d = self.w1_mesh.shape
        self.data = data

        cfuncs = {'cfour': CFOURdata(data), 'vlx': VeloxChemdata(data), 'openrsp': LSDaltondata(data)}
        self.callbacks = cfuncs[data['source']]

        # dictionary; keys from 0 to (3Natoms-6)
        self.fundamentals = {str(k):v for k,v in self.callbacks.getFundamentals().items()}

        # for non-zero fermi terms
        self.fermirm = 0.0001

        # margin for higher diagonal
        self.margin = 10.

        self.all_states = {tuple(str(i) for i in k): v for k, v in self.callbacks.getAllStates().items()}

        self.id = f'w1{min(self.w1)}_{max(self.w1)}w2{min(self.w2)}_{max(self.w2)}'

    # setting up the expressions for mechanical and electrical anharmonicities
    def addTerms(self, electrical_terms, mechanical_terms, el_avrg, mech_avrg):

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
            self.callbacks.tensors2NMbasis()
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

            from frompyopenrsp import pyrsp_2dir

            return dict(zip(['mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'], pyrsp_2dir.props_list))

        elif self.data['source'] == 'cfour':
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

        else:
            print("Invalid data source")

    def gamma_mn(self, Gamma, a, b, c=False):
        # components lists for averaging: terms of the sum
        gammaCompsAll = getting_abcgreek4avrg(num_f=4)

        # getting derivs
        derdata = self.getDerivs()

        shape = self.shape2d

        # if 'c' is not provided, compute electrical anharmonicity
        if type(c) == bool:

            total_sum_el = np.zeros(shape, dtype='complex128')
            prefac_el = 1 / self.fundamentals[str(a)] / self.fundamentals[str(b)]

            for el_func, elavrg in self.combofuns[0].items():
                # average for given (a, b) for a given term
                averg_el1 = avrg_abc(elavrg, derdata, [a, b], gammaCompsAll)

                total_sum_el += prefac_el * averg_el1 * el_func(self.all_states, self.w1_mesh, self.w2_mesh,
                                                                Gamma, (a, b))

            return total_sum_el / 24.

        else:

            total_sum_mech = np.zeros(shape, dtype='complex128')

            # mechanical
            prefac_mech = 1 / self.fundamentals[str(a)] / self.fundamentals[str(b)] / self.fundamentals[str(c)]

            for mech_func, mechavrg in self.combofuns[1].items():
                averg_mech1 = avrg_abc(mechavrg[:-1], derdata, [a, b, c], gammaCompsAll)
                abc = dict(zip(['a', 'b', 'c'], [a, b, c]))
                indx = tuple([abc[j] for j in mechavrg[-1]])

                F = derdata['F_abc'][indx]

                total_sum_mech += prefac_mech * averg_mech1 * F * mech_func(self.all_states,
                                                                            self.w1_mesh, self.w2_mesh, Gamma,
                                                                            (a, b, c))

            return -total_sum_mech / 48.

    def intensity(self, Gamma, savedict):

        Qab, Qabc = self.coords_ab, self.coords_abc

        Z = np.zeros(self.shape2d, dtype='complex128')

        Qab_contrib_dict = {}
        Qabc_contrib_dict = {}

        elall = np.zeros(self.shape2d, dtype='complex128')
        for i in Qab:
            contrib_ab = self.gamma_mn(Gamma, i[0], i[1])
            Qab_contrib_dict[tuple(i)] = contrib_ab
            elall += contrib_ab

        mechall = np.zeros(self.shape2d, dtype='complex128')
        for i in Qabc:
            contrib_abc = self.gamma_mn(Gamma, i[0], i[1], i[2])
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
    from src.macroscopic import macroscopics
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
    print('Fundamental frequencies:', list(setup.fundamentals.values()), '\n')
    # for k in setup.fundamentals:
    #     print()
    for d in ders:
        print(d, ders[d].shape)#, '\n', ders[d])
        printT(ders[d])
        print('=========================================================\n')
