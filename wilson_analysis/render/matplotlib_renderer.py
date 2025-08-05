import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.ticker as ticker
from .base_template import BaseRenderer
from wilson_main.abstractions import RenderingInfo, EvaluationInfo, SpectralGrid

import logging
logger = logging.getLogger("wilson."+__name__)
# wilson.wilson_main.abstractions
namelogger = f'"wilson."+__name__: {"wilson."+__name__}'
logger.info(namelogger)


class MatplotlibRenderer(BaseRenderer):
    """
    spec_grid = ws.main.abstractions.SpectralGrid({1: axis1, 2: axis2}, range_style='uniform',
                                                    start=start, end=end, spacer=spacer)

    evi = {'dynrange': 500, 'Gamma': 4.7, 'diag_margin': 5., 'maxmax': None}
    rndi = {'num_level_ticks': 15}
    eval_setup = ws.main.abstractions.SpecEvalSetup(grid=spec_grid, ev_info=evi, rnd_info=rndi)

    Input:
    self.spec, self.system, self.exp, self.diagn, self.name, self.spec_eval_setup

    main input: 
        axes: x1, x2, x3, ...
        projection: 1d, 2d, 3d
        dynamic_range
    figure styling


class RenderingInfo:
	projection: str = '2d'
	maxPeak: float = None
	dynrange: float = 100
	num_level_ticks: int = 12
	metadat: dict = field(default_factory=lambda: dict())
	figsize: tuple = (10, 13)
	font_dict: dict = {'size': 20}
	to_save: bool = False

class EvaluationInfo:
	freq_variables: dict
	Gamma: float
	Gamma_unit: str
	fixed_variables: dict = field(default_factory=lambda: dict())

    """
    def __init__(self, *args, spec_grid: SpectralGrid, 
                 rnd_info = None, ev_info = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.spec_grid = spec_grid

        if rnd_info is None:
            # default_rnd_info_class
            rnd_info = RenderingInfo()

        if ev_info is None:
            # default_rnd_info_class
            ev_info = EvaluationInfo()

        self.rnd_info = rnd_info
        self.ev_info = ev_info

        # self.spec_grid.axes is - 'x': axis1({'w1': 1}), 'y': axis2({'w1': 1, 'w2': -1})
        assert len(self.spec_grid.axes) == int(self.rnd_info.projection[0]), 'Provided spectral axes do not correspond to the projection choice'



    def create_figure(self):
        """
        Minimal figure setup using matplotlib.

        Args:
            figsize (tuple): Size of the figure in inches.
            projection (str, optional): e.g., '3d' or 'polar'.
            to_save (bool): Use 'Agg' backend to allow saving without display.

        Returns:
            tuple: (fig, ax)
        """
        if self.rnd_info.to_save:
            matplotlib.use('Agg')

        if self.rnd_info.font_dict:
            matplotlib.rc('font', **self.rnd_info.font_dict)

        subplot_kw = {'projection': self.rnd_info.projection} if self.rnd_info.projection else {}
        fig, ax = plt.subplots(figsize=self.rnd_info.figsize, subplot_kw=subplot_kw)
        return fig, ax

    def prepare_axes_data(self) -> dict:
        """
        variables = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4} - dict[str, np.ndarray]
        plot_axes = {
            'x': 'x2',
            'y': 'x3 - x2',
            'z': 'values'  # or even something like 'np.sin(values)'
        } - dict[str, str]

        - need to fix unused variables ('x1', 'x4' here)
        - need to extract appropriate shapes from meshgrids

        """
        # 'w1': mesh, 'w2': mesh; variables
        freq_vars = self.ev_info.freq_variables
        
        # 'x': meshsum, 'y': meshsum; plot_axes
        xy_axes = {}
        for i in self.spec_grid.axes:
            xy_axes[i] = sum([freq_vars[k]*v for k,v in self.spec_grid.axes[i].freq_vars.items()])

        return xy_axes

    def prepare_contour_levels(self):
        """
        For contourf plots

        set values to:
            self.tick_values
            self.tick_labels
            self.tick_norm_positions
        """

        d_max = np.max(self.intensities)
        d_min = d_max / self.rnd_info.dynrange

        if self.rnd_info.log10:
            log_min = np.log10(d_min)
            log_max = np.log10(d_max)

            # evenly spaced log10 levels
            log_ticks = np.linspace(log_min, log_max, self.rnd_info.num_level_ticks)
            self.tick_values = np.power(10, log_ticks)

            # Format labels nicely (LaTeX-style strings)
            self.tick_labels = [f"${val:.1e}$".replace('e', r'\times 10^{') + '}$' for val in self.tick_values]

            # Normalize log-scale values to [0, 1]
            self.tick_norm_positions = (log_ticks - log_min) / (log_max - log_min)

        else:
            self.tick_values = np.linspace(d_min, d_max, self.num_levels)
            # Format labels nicely (LaTeX-style strings)
            self.tick_labels = [f"${val:.1e}$".replace('e', r'\times 10^{') + '}$' for val in self.tick_values]

            # Normalize log-scale values to [0, 1]
            self.tick_norm_positions = (self.tick_values - d_min) / (d_max - d_min)


    def do_log10_data(self, specAxes_dict: dict, log10: list = None) -> dict:
        """
        log10 of given axes data
        log10: dict : {axesID_1:None, axesID_2:None, ...}
        axesID_1, axesID_2 - self.specAxes dict keys
        """
        if log10 is None:
            self.log10 = {}
        
        for intsID in self.log10:
            intensities = specAxes_dict[intsID]
            intensities_new = np.zeros_like(intensities)
            valid = intensities > 0
            intensities_new[valid] = np.log10(intensities[valid])
            assert np.all(np.isfinite(intensities_new)), "Log10 produced NaN or Inf!"
            self.log10[intsID] = intensities_new
        
        return self.log10

    def do_normalize_data(self, specAxes_dict: dict, normalize: list = None) -> dict:

        if normalize is None:
            self.normalize = {}
        
        for intsID in self.normalize:
            min_val = np.min(specAxes_dict[intsID]) if np.min(specAxes_dict[intsID]) != 0. else 0.
            self.normalize[intsID] = (specAxes_dict[intsID] - min_val) / (np.max(specAxes_dict[intsID]) - min_val)
        
        return self.normalize

    def plot_contours(self, fig: plt.Figure, ax: plt.Axes, axes_dict: dict):
        """
        needs 3 dimensions
        """
        
        assert 'x' in axes_dict and 'y' in axes_dict, 'Did not recieve `x` and `y` axes, cannot make a contour plot'
        assert len(axes_dict) < 3, 'Recieved more than 2 axes for plot. Contour plot needs `x` and `y` axes'

        X, Y = axes_dict['x'], axes_dict['x']
        Z = self.intensities

        cmap = plt.get_cmap('hot_r').copy()
        cmap.set_extremes(over='#FF00FF')
        cont = ax.contourf(
            X, Y, Z,
            levels=self.tick_values,
            cmap=cmap,
            extend='max'
        )

        ax.set_xlabel(r'$\omega_1/2\pi c, \text{cm}^{-1}$', fontsize=25)
        ax.set_ylabel(r'$(\omega_2 - \omega_1)/2\pi c, \text{cm}^{-1}$', fontsize=25)
        ax.set_title(self.rnd_info.title)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=2.1)
        cbar = fig.colorbar(cont, cax=cax, aspect=65, shrink=0.9,
                            ticks=self.tick_values, format=ticker.FuncFormatter(self._fmt_tick))
        
        cbar.set_ticklabels([f'{tick:.2f}' for tick in self.tick_values])
        for tick, label in zip(self.tick_values, self.tick_labels):
            cbar.ax.text(-2.0, tick, label, ha='left', va='center')


    def finalize(self, ax: plt.Axes, filename):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
        if self.rnd_info.to_save:
            plt.savefig(filename, dpi=250, format='svg')


    def _format_level(self, val):
        if val >= 1000 or val < 0.01:
            return f"{val:.0e}"
        elif val >= 1:
            return f"{val:.0f}"
        else:
            return f"{val:.2f}"

    def _fmt_tick(self, x, pos=''):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)
