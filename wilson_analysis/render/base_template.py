from abc import ABC, abstractmethod

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
    def __init__(self, intensities = None):
        """
        intensities - final values for Z axis of the figure

        dynrange_n: 
            max / dynrange_n ==> min
            1e8 / 1000       ==> 1e5
        """
        self.intensities = intensities


    def render(self, filename: str):
        """
        pipeline of rendering:
            from data prep to daved figure

        after prepare_contour_levels():
            self.tick_values
            self.tick_labels
            self.tick_norm_positions

        
        """
        # pre-processing
        axes_dict = self.prepare_axes_data()
        self.prepare_contour_levels()
        # fig prep
        fig, ax = self.create_figure()
        # plotting
        self.plot_contours(fig, ax, axes_dict)
        # final styling of figure and saving
        self.finalize(fig, ax, filename)


    @abstractmethod
    def prepare_axes_data(self)  -> dict:
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
