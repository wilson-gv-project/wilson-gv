import wilson_main.abstractions as wm_abst
from abc import ABC, abstractmethod
import numpy as np

class BaseRenderer(ABC):
    """
1. choice of projection - how to present spectrum data: 2d or 3d or 1d ... (figure dimension setup)
2. assignment of data to axes
3. dynamic range settings (for contour - colorbar, for 1d - threshold on y values?)
. style:
    a. axes titles
    b. axes ticks
    c. axes ticks labels
    d. axes labels
    e. title if any
. finalize figure styling: tight layout, etc...
. Save figure (where, filename)

    spec_grid = ws.main.abstractions.SpectralGrid({1: axis1, 2: axis2}, range_style='uniform',
                                                    start=start, end=end, spacer=spacer)

    evi = {'dynrange': 500, 'Gamma': 4.7, 'diag_margin': 5., 'maxmax': None}
    rndi = {'num_level_ticks': 15}
    eval_setup = ws.main.abstractions.SpecEvalSetup(grid=spec_grid, ev_info=evi, rnd_info=rndi)
    """
    def __init__(self, specAxes=None, projection=None, dynrange_n=None):
        """
        dynrange_n: 
            max / dynrange_n ==> min
            1e8 / 1000       ==> 1e5
        """
        self.specAxes = None
        self.projection = None
        self.dynrange_n = None


        self.num_levels = None
        self.title = None

        self.tick_values = None
        self.tick_labels = None
        self.tick_norm_positions = None


    def render(self, spec_eval_setup: wm_abst.SpecEvalSetup, contourAxes: dict, filename: str):
        
        self.specAxes = spec_eval_setup.grid.axes        
        assert self.specAxes is not None, 'Set specAxes value'
        
        Zval_axID = contourAxes['z']
        self.prepare_contour_levels(self.specAxes[Zval_axID])
        
        fig, ax = self.create_figure()

        self.plot_contours(fig, ax)
        self.finalize(fig, ax, filename)

    @abstractmethod
    def prepare_axes_data(self,
                            variables: dict[str, np.ndarray],
                            values: np.ndarray, # self.spec from WilsonSim
                            axes_def: dict[str, str],       # 'x', 'y', 'z' expressions
                            fixed_axes: dict[str, float] = None,  # x1=0.5, x4=1.0
                            interpolate: bool = True)  -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        variables = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4} - dict[str, np.ndarray]
        axes_def = {
            'x': 'x2',
            'y': 'x3 - x2',
            'z': 'values'  # or even something like 'np.sin(values)'
        } - dict[str, str]

        - need to fix unused variables ('x1', 'x4' here)
        - need to extract appropriate shapes from meshgrids
        """
        pass

    @abstractmethod
    def prepare_contour_levels(self):
        pass

    @abstractmethod
    def create_figure(self):
        pass

    @abstractmethod
    def plot_contours(self, fig, ax):
        pass

    @abstractmethod
    def finalize(self, fig, ax, filename):
        pass
