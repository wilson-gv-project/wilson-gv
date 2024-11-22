import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class SpectrumFigure:

    def __init__(self, sec_hypol_data, computedSpectrum, w1_mesh, w2_mesh, settings):

        # figure XYZ data
        self.gamma_data = sec_hypol_data
        self.intensities = abs(sec_hypol_data) ** 2

        min_positive = 1e-6
        self.intensities[self.intensities <= 0] = min_positive

        self.X = w1_mesh
        self.Y = w2_mesh

        # defaults
        self.settings = {'omega1_minus_omega2': False, 'log10': True,
                         'font_dict': {'size': 18}, 'dpi': 200, 'figsize': (12, 12),
                         'norm_max': None, 'norm_min': None,
                         'levels': None, 'level_ticks': None,
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
            print('\nself.intensities.max()==np.max(self.intensities.flatten(), axis=0):',
                  self.intensities.max()==np.max(self.intensities.flatten(), axis=0), '{:.4e}'.format(self.intensities.max()))
            self.d_max = self.intensities.max()
        self.settings['d_max'] = self.d_max
        if 'norm_max' not in self.settings:
            self.settings['norm_max'] = self.intensities.max()
        if 'norm_min' not in self.settings:
            self.settings['norm_min'] = self.intensities.min()


    def update_settings(self, settings: dict):
        self.settings.update(settings)


    def plot2Dmatplotlib(self, nametuple: tuple, text_under_the_figure: str = '', diagonal=False, to_save=True):

        if to_save:
            matplotlib.use('Agg')
        plt.rcParams['path.simplify'] = True
        plt.rcParams['agg.path.chunksize'] = 10000
        plt.rcParams['axes.titlepad'] = 30
        matplotlib.rc('font', **self.font_dict)

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1, 1, 1)

        import matplotlib.colors as colors

        dynamic_range = self.settings['dynamic_range_n']
        num_color_levels = self.settings['num_color_levels']
        dynrange_log = np.log10(dynamic_range)
        # d_max - max intensity
        dmax_log10 = float(int(np.log10(self.d_max)))

        num_level_ticks = self.settings['num_level_ticks']

        if self.settings['levels_ticks'] is None:
            # contour regions
            levels_ticks = [10**(dmax_log10-i) for i in range(num_level_ticks)]
        else:
            levels_ticks = self.settings['levels_ticks']

        if self.settings['levels_ticks'] is None:
            levels = []
            for i in range(num_color_levels):
                levels.append(self.d_max * 10.0 ** (-1.0 * dynrange_log * (float(num_color_levels - 1 - i) / (num_color_levels - 1))))
        else:
            levels = self.settings['levels']
        print('levels\n', levels)

        # range for color on the color bar
        colorbar_norm = colors.LogNorm(vmax=self.settings['norm_max'], vmin=self.settings['norm_min'])
        cont = plt.contourf(self.X, self.Y, self.intensities,
                            levels=levels, cmap='hot_r',
                            norm=colorbar_norm)
        if diagonal:
            plt.plot(self.X[:, 0], self.X[:, 0], color='red', linestyle='--', label='x = y')

        # This is the fix for the white lines between contour levels
        for c in cont.collections:
            c.set_edgecolor("face")

        # formatting of colorbar tick labels
        import matplotlib.ticker as ticker
        def fmt(x, pos):
            a, b = '{:.0e}'.format(x).split('e')
            b = int(b)
            return r'${} \times 10^{{{}}}$'.format(a, b)

        # https://stackoverflow.com/questions/25983218/scientific-notation-colorbar
        colorbar = plt.colorbar(cont, ticks=levels_ticks, format=ticker.FuncFormatter(fmt))

        # plt.xlabel(r'$\omega_1$')
        # plt.ylabel(r'$\omega_2$')
        xs = self.X[0], self.X[-1]
        ys = self.Y[0], self.Y[-1]

        title_type_dict = {(True, False): r'electrical anharmonicity $|\gamma^{[1,0]}|^2$ only',
                           (False, True): r'mechanical anharmonicity $|\gamma^{[0,1]}|^2$ only',
                           (True, True): r'both $|\gamma^{[1,0]}+\gamma^{[0,1]}|^2$'}

        nicetitle = f'{nametuple[2]}'
        plt.title(nicetitle)
        bbox_args = dict(boxstyle="round,pad=0.8", edgecolor='black', facecolor='lightgray')
        ax.annotate(text_under_the_figure, xy=(0.05, -0.11), xycoords='axes fraction',
                    ha="left", va="top", bbox=bbox_args, fontsize=12)
        plt.tight_layout()
        if to_save:
            plt.savefig(nametuple[0], dpi=self.dpi, format='svg')

        # import shutil
        # shutil.copy2(nametuple[0], '/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/svgs/'+nametuple[0])
        return fig