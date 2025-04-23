import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class SpectrumFigure:

    def __init__(self, sec_hypol_data, computedSpectrum,
                 w1_mesh, w2_mesh, settings):

        # figure XYZ data
        self.gamma_data = sec_hypol_data
        self.intensities = abs(sec_hypol_data) ** 2

        min_positive = 1e-6
        self.intensities[self.intensities <= 0] = min_positive

        self.X = w1_mesh
        self.Y = w2_mesh
        if settings['maxYX'] is None:
            try:
                # self.maxX, self.maxY, self.maxYX =  computedSpectrum.maxX, computedSpectrum.maxY, computedSpectrum.maxYX
                self.maxYX = computedSpectrum.maxYX
            except AttributeError:
                # self.maxX, self.maxY, self.maxYX = None, None, 3400.
                self.maxYX = 3400.
        else:
            # self.maxX, self.maxY, self.maxYX = settings['maxX'], settings['maxY'], settings['maxYX']
            self.maxYX = settings['maxYX']

        # defaults
        self.settings = {'omega1_minus_omega2': False, 'log10': True,
                         'font_dict': {'size': 20}, 'dpi': 200, 'figsize': (12, 12),
                         'norm_max': None, 'norm_min': None,
                         'levels': None, 'level_ticks': None,
                         'num_color_levels': None,
                         'Gamma_rc': computedSpectrum.Gamma_rc,
                         'electrical': computedSpectrum.e_selected, 'mechanical': computedSpectrum.m_selected}
        self.settings.update(settings)

        if self.settings['omega1_minus_omega2']:
            self.Y = -(self.X - self.Y)

        # figure settings
        self.figsize = self.settings['figsize']
        self.dpi = self.settings['dpi']
        self.font_dict = self.settings['font_dict']

        el, mech = self.settings['electrical'], self.settings['mechanical']

        # dynamic range max - for setting up the norm and colorbar ticks
        if 'dmax_dict' in self.settings:
            self.d_max = self.settings['dmax_dict'][(el, mech)]
        else:
            self.d_max = self.intensities.max()
        self.settings['d_max'] = self.d_max
        if 'norm_max' not in self.settings:
            self.settings['norm_max'] = self.intensities.max()
        if 'norm_min' not in self.settings:
            self.settings['norm_min'] = self.intensities.min()


    def update_settings(self, settings: dict):
        self.settings.update(settings)

    def normalize(self, norm_intensity:float):
        """
        norm_intensity_log10: float - max intensity (gamma abs square) of reference
        norm_intensity_log10: float - log10 of max intensity of reference
        """
        # should be done on log10
        self.intensities /= norm_intensity

        min_positive = 1e-6
        self.intensities[self.intensities <= 0] = min_positive

        el, mech = self.settings['electrical'], self.settings['mechanical']
        # dynamic range max - for setting up the norm and colorbar ticks
        self.d_max = self.intensities.max()
        # self.settings['d_max'] = self.d_max
        if 'norm_max' not in self.settings:
            self.settings['norm_max'] = self.intensities.max()
        if 'norm_min' not in self.settings:
            self.settings['norm_min'] = self.intensities.min()



    def plot2Dmatplotlib(self, nametuple: tuple, text_under_the_figure: str = '',
                         normalized=None, log10=False,
                         diagonal=False, to_save=True, textbox=False):

        if to_save:
            matplotlib.use('Agg')

        plt.rcParams['path.simplify'] = True
        plt.rcParams['agg.path.chunksize'] = 10000
        plt.rcParams['axes.titlepad'] = 30
        matplotlib.rc('font', **self.font_dict)

        fig, ax = plt.subplots(figsize=self.figsize)
        fig.subplots_adjust(left=0.1, right=0.9, top=1.05, bottom=0.15)

        import matplotlib.colors as colors

        dynamic_range = self.settings['dynamic_range_n']
        num_color_levels = self.settings['num_color_levels']
        dynrange_log = np.log10(dynamic_range)
        # d_max - max intensity
        dmax_log10 = float(int(np.log10(self.d_max)))
        print('d_max', self.d_max, )
        print('dmax_log10', dmax_log10)
        num_level_ticks = self.settings['num_level_ticks']

        if log10:
            l10 = np.log10(self.intensities)
            if normalized == '01':
                minimum_intensity = np.min(l10) if np.min(l10)!=0. else 0.
                intensity_plot = (l10 - minimum_intensity) / (np.max(l10) - minimum_intensity)
            elif type(normalized) == tuple:
                minimum_intensity = np.log10(normalized[0]) if normalized[0]!=0. else 0.
                intensity_plot = (l10 - minimum_intensity) / (np.log10(normalized[1]) - minimum_intensity)

            else:
                intensity_plot = l10
                colorbar_norm = colors.LogNorm(vmax=np.ceil(dmax_log10) + 0.2, vmin=self.settings['norm_min'])

            self.maxhere = np.max(l10)
            self.minhere = np.min(l10)
        else:
            intensity_plot = self.intensities
            # colorbar_norm = colors.LogNorm(vmax=self.settings['norm_max'], vmin=self.settings['norm_min'])
            colorbar_norm = colors.LogNorm(vmax=10 ** np.ceil(np.log10(self.d_max)), vmin=self.settings['norm_min'])


        # levels settings
        if log10:

            if normalized=='01':

                min_dynrange = np.log10(self.d_max) - dynrange_log
                dt = - (np.log10(self.d_max) - min_dynrange) / num_level_ticks

                levels_before_norm = np.sort(np.arange(np.log10(self.d_max), min_dynrange, round(dt,4)))
                self.levels = levels_before_norm/np.log10(self.d_max)
                self.levels_ticks = self.levels

                levels_nums = [10**i for i in levels_before_norm]
                levels_nums_str = [f'{tick:.2e}' for tick in levels_nums]

            elif type(normalized) == tuple:

                min_dynrange = np.log10(normalized[1]) - dynrange_log
                dt = - (np.log10(normalized[1]) - min_dynrange) / num_level_ticks
                print(np.log10(normalized[1]), min_dynrange, round(dt,4))
                levels_before_norm = np.sort(np.arange(np.log10(normalized[1]), min_dynrange, round(dt,4)))

                self.levels = np.sort(np.arange(np.log10(normalized[1]),
                                                min_dynrange, round(dt, 4)))/np.log10(normalized[1])
                self.levels_ticks = self.levels

                levels_nums = [10**i for i in levels_before_norm]
                # print(levels_nums)
                levels_nums_str = [f'{tick:.2e}' for tick in levels_nums]

            else:

                if self.settings['levels_ticks'] is None:
                    min_dynrange = (round(np.log10(self.d_max), 1)) - dynrange_log
                    dt = - ((round(np.log10(self.d_max), 1))-min_dynrange) / num_level_ticks
                    self.levels = np.sort(np.arange(round(np.log10(self.d_max), 1), min_dynrange, round(dt,1)))

                else:
                    self.levels = self.settings['levels']

                if self.settings['levels_ticks'] is None:
                    # contour regions
                    self.levels_ticks = self.levels
                else:
                    self.levels_ticks = self.settings['levels_ticks']
            print('min_dynrange', min_dynrange)
            print('dynrange_log', dynrange_log)
            print('dynamic_range', dynamic_range)
        else:

            # self.intensities are in original value, not log10
            if self.settings['levels_ticks'] is None:
                # contour regions
                self.levels_ticks = [10**(dmax_log10+1-i) for i in range(num_level_ticks)]
                self.levels_ticks = np.array(sorted(self.levels_ticks))

            else:
                self.levels_ticks = self.settings['levels_ticks']

            if self.settings['levels'] is None:
                if num_color_levels is None:
                    num_color_levels = len(self.levels_ticks)
                    print('Using default num_color_levels = len(levels_ticks)')

                self.levels = [self.d_max * 10.0 ** (-1.0 * dynrange_log *
                                                (float(num_color_levels - 1 - i) / (num_color_levels - 1)))
                          for i in range(num_color_levels)]

            else:
                self.levels = self.settings['levels']

        if self.settings['w1mw2']:
            y = -(self.X - self.Y)
            ax.set_ylabel(r'$(\omega_2-\omega_1)/2\pi c, \text{cm}^{-1}$', fontsize=25, labelpad=21.)
            # ax.set_ylabel(r'(\\omega_2-\\omega_1)/2\pi c, \\text{cm}^{-1}', fontsize=18)
        else:
            y = self.Y
            # ax.set_ylabel(r'$\\omega_2/2\pi c, \\text{cm}^{-1}$', fontsize=18)
            ax.set_ylabel(r'$\omega_2/2\pi c, \text{cm}^{-1}$', fontsize=25, labelpad=21.)
        ax.set_xlabel(r'$\omega_1/2\pi c, \text{cm}^{-1}$', fontsize=25, labelpad=21.)
        nicetitle = f'{nametuple[2]}'
        plt.title(nicetitle)

        cmap = plt.get_cmap('hot_r').copy()
        cmap.set_extremes(over=self.settings['saturation_color'])
        cont = ax.contourf(self.X, y, intensity_plot,
                            levels=self.levels, cmap=cmap  #'hot_r'
                           # , norm=colorbar_norm
                            , extend='max'
                           )
        # print('self.levels', self.levels)
        # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        # print(intensity_plot)
        if diagonal:
            plt.plot(self.X[:, 0], self.X[:, 0], color='red', linestyle='--', label='x = y')

        if self.settings['w1mw2']:
            # x_limits = ax.get_xlim()
            if 'minY' in self.settings:
                ax.set_ylim(self.settings['minY'], self.maxYX)
            else:
                ax.set_ylim(0, self.maxYX)

        # This is the fix for the white lines between contour levels
        for c in cont.collections:
            c.set_edgecolor("face")

        # formatting of colorbar tick labels
        import matplotlib.ticker as ticker
        def fmt(x, pos):
            a, b = '{:.0e}'.format(x).split('e')
            b = int(b)
            return r'${} \times 10^{{{}}}$'.format(a, b)

        if log10:
            from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=2.1)

            # colorbar = ax.colorbar(cont, aspect=65, shrink=0.9,
            #                         ticks=self.levels_ticks)
            colorbar = plt.colorbar(cont, cax=cax,
                                   ticks=self.levels_ticks)
            colorbar.set_ticks(self.levels_ticks)
            colorbar.set_ticklabels([f'{tick:.4f}' for tick in self.levels_ticks])

            for tick, label in zip(self.levels_ticks, levels_nums_str):
                colorbar.ax.text(-2.5, tick, label, ha='left', va='center')
        else:

            # https://stackoverflow.com/questions/25983218/scientific-notation-colorbar
            colorbar = plt.colorbar(cont, aspect=65, shrink=0.9,
                                    ticks=self.levels_ticks, format=ticker.FuncFormatter(fmt))

        if textbox:
            bbox_args = dict(boxstyle="round,pad=0.8", edgecolor='black', facecolor='lightgray')
            ax.annotate(text_under_the_figure, xy=(0.05, -0.10), xycoords='axes fraction',
                        ha="left", va="top", bbox=bbox_args, fontsize=12)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.7)

        ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)

        if to_save:
            plt.savefig(nametuple[0], dpi=self.dpi, format='svg')
        return fig


    def plot2Damplitudes(self, nametuple: tuple, text_under_the_figure: str = '',
                         normalized=None, log10=False,
                         diagonal=False, to_save=True, textbox=False):

        if to_save:
            matplotlib.use('Agg')

        plt.rcParams['path.simplify'] = True
        plt.rcParams['agg.path.chunksize'] = 10000
        plt.rcParams['axes.titlepad'] = 30
        matplotlib.rc('font', **self.font_dict)

        fig, ax = plt.subplots(figsize=self.figsize)
        fig.subplots_adjust(left=0.1, right=0.9, top=1.05, bottom=0.15)

        import matplotlib.colors as colors

        dynamic_range = self.settings['dynamic_range_n']
        num_color_levels = self.settings['num_color_levels']
        dynrange_log = np.log10(dynamic_range)
        # d_max - max intensity
        dmax_log10 = float(int(np.log10(self.d_max)))
        print('d_max', self.d_max, )
        print('dmax_log10', dmax_log10)
        num_level_ticks = self.settings['num_level_ticks']

        if log10:
            l10 = np.log10(self.gamma_data)
            if normalized == '01':
                minimum_intensity = np.min(np.abs(l10)) if np.min(np.abs(l10))!=0. else 0.
                intensity_plot = (l10 - minimum_intensity) / (np.max(l10) - minimum_intensity)
            elif type(normalized) == tuple:
                minimum_intensity = np.log10(normalized[0]) if normalized[0]!=0. else 0.
                intensity_plot = (l10 - minimum_intensity) / (np.log10(normalized[1]) - minimum_intensity)

            else:
                intensity_plot = l10
                colorbar_norm = colors.LogNorm(vmax=np.ceil(dmax_log10) + 0.2, vmin=self.settings['norm_min'])

            self.maxhere = np.max(np.abs(l10))
            self.minhere = np.min(np.abs(l10))
        else:
            intensity_plot = self.gamma_data
            # colorbar_norm = colors.LogNorm(vmax=self.settings['norm_max'], vmin=self.settings['norm_min'])
            colorbar_norm = colors.LogNorm(vmax=10 ** np.ceil(np.log10(self.d_max)), vmin=self.settings['norm_min'])


        # levels settings
        if log10:

            if normalized=='01':

                min_dynrange = np.log10(self.d_max) - dynrange_log
                dt = - (np.log10(self.d_max) - min_dynrange) / num_level_ticks

                levels_before_norm = np.sort(np.arange(np.log10(self.d_max), min_dynrange, round(dt,4)))
                self.levels = levels_before_norm/np.log10(self.d_max)
                self.levels_ticks = self.levels

                levels_nums = [10**i for i in levels_before_norm]
                levels_nums_str = [f'{tick:.2e}' for tick in levels_nums]

            elif type(normalized) == tuple:

                min_dynrange = np.log10(normalized[1]) - dynrange_log
                dt = - (np.log10(normalized[1]) - min_dynrange) / num_level_ticks
                print(np.log10(normalized[1]), min_dynrange, round(dt,4))
                levels_before_norm = np.sort(np.arange(np.log10(normalized[1]), min_dynrange, round(dt,4)))

                self.levels = np.sort(np.arange(np.log10(normalized[1]),
                                                min_dynrange, round(dt, 4)))/np.log10(normalized[1])
                self.levels_ticks = self.levels

                levels_nums = [10**i for i in levels_before_norm]
                # print(levels_nums)
                levels_nums_str = [f'{tick:.2e}' for tick in levels_nums]

            else:

                if self.settings['levels_ticks'] is None:
                    min_dynrange = (round(np.log10(self.d_max), 1)) - dynrange_log
                    dt = - ((round(np.log10(self.d_max), 1))-min_dynrange) / num_level_ticks
                    self.levels = np.sort(np.arange(round(np.log10(self.d_max), 1), min_dynrange, round(dt,1)))

                else:
                    self.levels = self.settings['levels']

                if self.settings['levels_ticks'] is None:
                    # contour regions
                    self.levels_ticks = self.levels
                else:
                    self.levels_ticks = self.settings['levels_ticks']
            print('min_dynrange', min_dynrange)
            print('dynrange_log', dynrange_log)
            print('dynamic_range', dynamic_range)
        else:

            # self.intensities are in original value, not log10
            if self.settings['levels_ticks'] is None:
                # contour regions
                self.levels_ticks = [10**(dmax_log10+1-i) for i in range(num_level_ticks)]
                self.levels_ticks = np.array(sorted(self.levels_ticks))

            else:
                self.levels_ticks = self.settings['levels_ticks']

            if self.settings['levels'] is None:
                if num_color_levels is None:
                    num_color_levels = len(self.levels_ticks)
                    print('Using default num_color_levels = len(levels_ticks)')

                self.levels = [self.d_max * 10.0 ** (-1.0 * dynrange_log *
                                                (float(num_color_levels - 1 - i) / (num_color_levels - 1)))
                          for i in range(num_color_levels)]

            else:
                self.levels = self.settings['levels']

        if self.settings['w1mw2']:
            y = -(self.X - self.Y)
            ax.set_ylabel(r'$(\omega_2-\omega_1)/2\pi c, \text{cm}^{-1}$', fontsize=25, labelpad=21.)
            # ax.set_ylabel(r'(\\omega_2-\\omega_1)/2\pi c, \\text{cm}^{-1}', fontsize=18)
        else:
            y = self.Y
            # ax.set_ylabel(r'$\\omega_2/2\pi c, \\text{cm}^{-1}$', fontsize=18)
            ax.set_ylabel(r'$\omega_2/2\pi c, \text{cm}^{-1}$', fontsize=25, labelpad=21.)
        ax.set_xlabel(r'$\omega_1/2\pi c, \text{cm}^{-1}$', fontsize=25, labelpad=21.)
        nicetitle = f'{nametuple[2]}'
        plt.title(nicetitle)

        cont = ax.contourf(self.X, y, intensity_plot,
                            levels=self.levels, cmap='hot_r'
                           # , norm=colorbar_norm
                           )
        # print('self.levels', self.levels)
        # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        # print(intensity_plot)
        if diagonal:
            plt.plot(self.X[:, 0], self.X[:, 0], color='red', linestyle='--', label='x = y')

        if self.settings['w1mw2']:
            # x_limits = ax.get_xlim()
            if 'minY' in self.settings:
                ax.set_ylim(self.settings['minY'], self.maxYX + 300.)
            else:
                ax.set_ylim(0, self.maxYX+300.)

        # This is the fix for the white lines between contour levels
        for c in cont.collections:
            c.set_edgecolor("face")

        # formatting of colorbar tick labels
        import matplotlib.ticker as ticker
        def fmt(x, pos):
            a, b = '{:.0e}'.format(x).split('e')
            b = int(b)
            return r'${} \times 10^{{{}}}$'.format(a, b)

        if log10:
            from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=2.1)

            # colorbar = ax.colorbar(cont, aspect=65, shrink=0.9,
            #                         ticks=self.levels_ticks)
            colorbar = plt.colorbar(cont, cax=cax,
                                   ticks=self.levels_ticks)
            colorbar.set_ticks(self.levels_ticks)
            colorbar.set_ticklabels([f'{tick:.4f}' for tick in self.levels_ticks])

            for tick, label in zip(self.levels_ticks, levels_nums_str):
                colorbar.ax.text(-2.5, tick, label, ha='left', va='center')
        else:

            # https://stackoverflow.com/questions/25983218/scientific-notation-colorbar
            colorbar = plt.colorbar(cont, aspect=65, shrink=0.9,
                                    ticks=self.levels_ticks, format=ticker.FuncFormatter(fmt))

        if textbox:
            bbox_args = dict(boxstyle="round,pad=0.8", edgecolor='black', facecolor='lightgray')
            ax.annotate(text_under_the_figure, xy=(0.05, -0.10), xycoords='axes fraction',
                        ha="left", va="top", bbox=bbox_args, fontsize=12)
        print('HELLO???')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.7)

        ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)

        if to_save:
            plt.savefig(nametuple[0], dpi=self.dpi, format='svg')
        return fig