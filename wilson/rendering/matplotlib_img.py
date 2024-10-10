import numpy as np

class SpectrumFigure:

    def __init__(self, sec_hypol_data, w1_mesh, w2_mesh, settings):

        # figure XYZ data
        self.gamma_data = sec_hypol_data
        self.intensities = abs(sec_hypol_data) ** 2
        self.X = w1_mesh
        self.Y = w2_mesh

        self.settings = {'omega1_minus_omega2': False, 'log10': True,
                         'font_dict': {'size': 18}, 'dpi': 200,
                         'figsize': (12, 12)}
        self.settings.update(settings)

        if self.settings['omega1_minus_omega2']:
            self.Y = -(self.X - self.Y)

        # figure settings
        self.figsize = self.settings['figsize']
        self.dpi = self.settings['dpi']
        self.font_dict = self.settings['font_dict'] # font = {'size': 18}
        # self.settings['norm_min'] = 1e3
        # self.settings['norm_max'] = 1e8

        el, mech = self.settings['electrical'], self.settings['mechanical']

        # dynamic range max - for setting up the norm and colorbar ticks
        if 'dmax_dict' in self.settings:
            self.d_max = self.settings['dmax_dict'][(el, mech)]
        else:
            print('\nself.intensities.max()==np.max(self.intensities.flatten(), axis=0):',
                  self.intensities.max()==np.max(self.intensities.flatten(), axis=0), '{:.4e}'.format(self.intensities.max()))
            self.d_max = self.intensities.max()
        self.settings['d_max'] = self.d_max
        # dmax_dict = {(True, False): 48778401.3, (False, True): 29519537.48, (True, True): 48218929.9}
        # d_max = dmax_dict[(el_bool, mech_bool)] # m, e, t 29519537.48  48778401.3  48218929.9

    def update_settings(self, settings: dict):
        self.settings.update(settings)

    def plot2Dmatplotlib(self, nametuple: tuple, text_under_the_figure: str = '', diagonal=False, to_save=True):
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib
        if to_save:
            matplotlib.use('Agg')
        plt.rcParams['path.simplify'] = True
        plt.rcParams['agg.path.chunksize'] = 10000
        plt.rcParams['axes.titlepad'] = 30
        matplotlib.rc('font', **self.font_dict)

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1, 1, 1)

        import matplotlib.colors as colors
        colorbar_norm = colors.LogNorm(vmin=self.settings['norm_min'], vmax=self.settings['norm_max'])

        num_count = self.settings['dynamic_range_n']
        dynamic_range = num_count*10 # stop plotting when lower than this (number times 10) dmax

        dynrange_log = np.log10(dynamic_range)
        d_min = (1.0 / float(dynamic_range)) * self.intensities.max()
        dmax_log10 = float(int(np.log10(self.d_max)))

        num_level_ticks = 6
        levels_ticks = [10**(dmax_log10-i) for i in range(num_level_ticks)]
        levels = []
        for i in range(num_count):
            levels.append(self.d_max * 10.0 ** (-1.0 * dynrange_log * (float(num_count - 1 - i) / (num_count - 1))))

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