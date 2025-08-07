"""
Use in WilsonSimulation:
self.rendering = renderer(self.spec, self.system, self.exp, self.diagn, self.name, self.spec_eval_setup)
self.rendering, self.diagn = renderer(self.spec, self.system, self.exp, self.diagn, self.name, self.spec_eval_setup)

Input:
self.spec, self.system, self.exp, self.diagn, self.name, self.spec_eval_setup; 

Steps: 
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
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from enum import Enum


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wilson_main.abstractions import SpectralGrid, EvaluationInfo, RenderingInfo, SpecEvalSetup, SimContext


import logging
logger = logging.getLogger("wilson."+__name__)

class NormalizationType(Enum):
    """
    # LOG_RATIO: log10(x)/log10(max)
    # Shows relative order of magnitude
    # Example: 0.5 means halfway between min and max in log scale

    # DECIBEL: 10 * log10(x/max)
    # Standard intensity scale in spectroscopy
    # 0 dB = max, -20 dB = 1/100 of max
    https://www.montana.edu/rmaher/eele417_fl14/decibel_scale_eele417.pdf
    https://en.wikipedia.org/wiki/Decibel
    https://www.animations.physics.unsw.edu.au/jw/dB.htm
    
    # PERCENTAGE: (x/max) * 100
    # Linear scale percentage
    # Direct proportion to maximum

    # LOG_SCALE: (log10(x) - log10(min))/(log10(max) - log10(min))
    # Normalized position in log space
    # 0 = minimum, 1 = maximum
    """
    LOG_RATIO = "log_ratio"
    DECIBEL = "db"
    PERCENTAGE = "percent"
    LOG_SCALE = "log_scale"


@dataclass
class PlotConfig:
    """Configuration for plot styling"""
    figsize: Tuple[int, int] = (35, 45)
    label_fontsize: int = 25
    font_dict: Dict[str, Any] = field(default_factory=lambda: {'size': 20})
    colormap: str = 'magma'  # Better contrast colormap
    saturation_color: str = '#FF00FF'
    dpi: int = 250
    tick_step: float = 200.0  # Step size for both axes ticks
    equal_aspect: bool = True  # Force equal aspect ratio for axes
    no_data_color: str = '#E0E0E0'  # Light gray
    below_range_color: str = '#F8F8F8'  # Very light gray
    data_edge_color: str = 'black'
    data_edge_width: float = 0.75
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    colorbar_main_label: str = "Intensity"
    colorbar_padding: float = 0.01  # Padding between colorbar and plot
    show_top_ticks: bool = False
    show_right_ticks: bool = False
    x_tick_rotation: float = 45  # Add this line for configurable rotation
    colormap_spacing: str = None  # Options: "log", "power", "linear"
    colormap_power: float = 0.5    # For power-law spacing; Adjust this value to change color distribution

class LevelCalculator:
    """
    Handles calculation of contour levels and normalization
    
    organizes the logic for computing levels and labels
    based on dynamic range, number of levels, and normalization type.
    """

    @staticmethod
    def compute_levels(d_max: float, dynamic_range: float, num_levels: int, 
                  ref_max: Optional[float] = None, 
                  colormap_spacing: str = None, 
                  colormap_power: float = 0.5) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
        """Calculate levels for contours and colorbar ticks"""
        d_min = d_max / dynamic_range
        log_min = np.log10(d_min)
        log_max = np.log10(d_max)

        if colormap_spacing == "log":        
            # Linear spacing in log scale
            log_space = np.linspace(log_min, log_max, num_levels)
            # back to linear scale
            level_values = np.power(10, log_space)

        elif colormap_spacing == "linear":
            # Linear spacing in original scale
            level_values = np.linspace(d_min, d_max, num_levels)
            # calculate corresponding log space points
            log_space = np.log10(level_values)
    
        elif colormap_spacing == "power":
            # Power-law spacing for more uniform color distribution
            power_space = np.power(np.linspace(0, 1, num_levels), colormap_power)
            log_space = log_min + (log_max - log_min) * power_space
            # back to linear scale
            level_values = np.power(10, log_space)
    
        else:  # default to log spacing
            log_space = np.linspace(log_min, log_max, num_levels)
            level_values = np.power(10, log_space)

        # Format original value labels
        level_labels = [f"${val:.1e}$" for val in level_values]
        
        # Calculate normalized values in log space
        if ref_max is None:
            ref_max = d_max
    
        # Normalize in log space to preserve logarithmic spacing
        norm_positions = (log_space - log_min) / (log_max - log_min)
        norm_labels = [f"{val:.2f}" for val in norm_positions]
        
        logger.debug(f"Computed levels: {level_values}, labels: {level_labels}, "
                     f"normalized positions: {norm_positions}, normalized labels: {norm_labels}")
        
        return level_values, level_labels, norm_positions, norm_labels


class SpectrumRenderer(ABC):
    """
    Abstract base class for spectrum rendering
    
    PlotConfig instance would normally be stored in the RenderingInfo instance
    but can be provided as config parameter
    """
    
    def __init__(self, 
                 spec_data: np.ndarray | dict,
                 spec_grid: "SpectralGrid" = None,
                 ev_info: "EvaluationInfo" = None,
                 rnd_info: "RenderingInfo" = None, 
                 do_diagn: bool = False,
                 config: PlotConfig = PlotConfig()):

        self.spec_data = spec_data
        self.rnd_info = rnd_info
        self.ev_info = ev_info
        self.spec_grid = spec_grid

        # TODO not used currently
        self.do_diagn = do_diagn
        
        self.config = self.rnd_info.style_config if rnd_info else config
        self.level_calc = LevelCalculator()
        self.intensities = None

        self.Xdata = None
        self.Ydata = None


    @abstractmethod
    def initialize_plot(self) -> Any:
        """Initialize plotting surface"""
        pass
    
    @abstractmethod
    def create_contour(self, plot_obj: Any, levels: np.ndarray, data: np.ndarray) -> Any:
        """Create contour plot"""
        pass
    
    @abstractmethod
    def setup_axes(self, plot_obj: Any) -> Any:
        """Configure axes, ticks and labels"""
        pass

    @abstractmethod
    def add_colorbar(self, plot_obj: Any, levels: np.ndarray, labels: List[str]) -> None:
        """Add colorbar to plot"""
        pass
    
    @abstractmethod
    def finalize(self, plot_obj) -> None:
        """some finishing styling touches (positioning and resizing)"""
        pass

    @abstractmethod
    def save_plot(self, plot_obj: Any, filename: str) -> None:
        """Save plot to file"""
        pass

    def prep_data(self, spec_data_operations: str) -> np.ndarray:
        """
        Prepare data for contour plotting.
        """
        if self.intensities is None:
            if spec_data_operations == 'abs()**2':
                self.intensities = np.abs(self.spec_data) ** 2
            elif spec_data_operations == 'abs':
                self.intensities = np.abs(self.spec_data)
            elif spec_data_operations == 'real':
                self.intensities = np.real(self.spec_data)
            elif spec_data_operations == 'imag':
                self.intensities = np.imag(self.spec_data)
            else:
                raise ValueError(f"Unsupported spec_data_operations: {spec_data_operations}")

        if self.spec_grid is not None:
            # 'w1': mesh, 'w2': mesh; variables
            freq_vars = self.ev_info.freq_variables

            # 'x': meshsum, 'y': meshsum; plot_axes
            xy_axes = {}

            for i in self.spec_grid.axes:
                xy_axes[i] = sum([freq_vars[k]*v for k,v in self.spec_grid.axes[i].freq_vars.items()])

            self.Xdata = xy_axes.get('x', None)
            self.Ydata = xy_axes.get('y', None)
            self.Zdata = xy_axes.get('z', None) # 3D plot with 3 spectral axes

    
    def render(self, filename: str) -> None:
        """Main rendering pipeline"""

        # prepare data for contour plotting with spec_data_operations and spec_grid.axes
        self.prep_data(spec_data_operations=self.rnd_info.spec_data_operations)

        # Calculate levels with both original and normalized scales
        levels, labels, norm_positions, norm_labels = self.level_calc.compute_levels(
            np.max(self.intensities),
            self.rnd_info.dynamic_range,
            self.rnd_info.num_levels,
            ref_max=self.rnd_info.reference_max,
            colormap_spacing=self.config.colormap_spacing,
            colormap_power=self.config.colormap_power
        )

        fig, ax = self.initialize_plot()
        fig, ax, contour = self.create_contour(plot_obj=(fig, ax), levels=levels, data=self.intensities)
        fig, ax = self.setup_axes(plot_obj=(fig, ax))
        fig, ax, cbar = self.add_colorbar(plot_obj=(fig, ax, contour), levels=levels, labels=labels)
        
        self.finalize(plot_obj=(fig, ax, cbar))
        self.save_plot(plot_obj=(fig, ax, cbar), filename=filename)

        return fig, ax, contour, cbar


class MatplotlibRenderer(SpectrumRenderer):
    """Matplotlib implementation of spectrum renderer"""
    
    def initialize_plot(self) -> Tuple[plt.Figure, plt.Axes]:
        matplotlib.use('Agg')
        plt.rcParams['path.simplify'] = True
        plt.rcParams['agg.path.chunksize'] = 10000
        
        # Apply font settings
        matplotlib.rc('font', **self.config.font_dict)
        
        # Create figure and axes with more appropriate margins
        fig = plt.figure(figsize=self.config.figsize)
        # Add axes with specific margins to ensure content fits
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.75])  # [left, bottom, width, height]
        
        return fig, ax
    

    def create_contour(self, 
                       plot_obj: Tuple[plt.Figure, plt.Axes], 
                       levels: np.ndarray, 
                       data: np.ndarray) -> Tuple[plt.Figure, plt.Axes, Any]:
        """
        2D contour plot.

        data parameter here refers to the Z-axis data for contour plotting, the signal magnitude
        """

        fig, ax = plot_obj
        
        # Create masked arrays
        no_data_mask = np.isnan(data)
        d_min = np.max(data) / self.rnd_info.dynamic_range
        below_range_mask = (~no_data_mask) & (data < d_min)
        
        # Setup base colormap
        cmap = plt.get_cmap(self.config.colormap).copy()
        cmap.set_over(self.config.saturation_color)
        
        # Fill no-data and below-range regions
        ax.contourf(self.Xdata, self.Ydata,
                   no_data_mask,
                   levels=[0, 0.5, 1],
                   colors=[self.config.no_data_color])
        
        ax.contourf(self.Xdata, self.Ydata,
                   below_range_mask,
                   levels=[0, 0.5, 1],
                   colors=[self.config.below_range_color])
        
        # Create logarithmic normalization for color mapping
        norm = matplotlib.colors.LogNorm(vmin=levels[0], vmax=levels[-1])
        
        # Plot main data with normalized colors
        contour = ax.contourf(self.Xdata, self.Ydata, 
                           data,
                           levels=levels,
                           norm=norm,  # Add normalization
                           cmap=cmap,
                           extend='max')
        
        # Single clean edge line
        ax.contour(self.Xdata, self.Ydata,
                  ~no_data_mask,
                  levels=[0.5],
                  colors=[self.config.data_edge_color],
                  linewidths=[self.config.data_edge_width])
        
        return fig, ax, contour


    def setup_axes(self, plot_obj):
        fig, ax = plot_obj

        # Set up axes labels
        label_fontsize = self.config.label_fontsize if hasattr(self.config, 'label_fontsize') else 25

        xlabel_str = spectral_axis_to_label(self.spec_grid.axes.get('x').freq_vars) if self.spec_grid else r'$default x /2\pi c, \text{cm}^{-1}$'
        ylabel_str = spectral_axis_to_label(self.spec_grid.axes.get('y').freq_vars) if self.spec_grid else r'$default y /2\pi c, \text{cm}^{-1}$'

        # labelpad - distance from axis to label
        ax.set_xlabel(xlabel_str, fontsize=label_fontsize, labelpad=65.) 
        ax.set_ylabel(ylabel_str, fontsize=label_fontsize, labelpad=65.)


        # Simple aspect ratio setting
        if self.config.equal_aspect:
            ax.set_aspect('equal', adjustable='box')
        
        # Generate tick positions
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Set regular ticks
        ax.set_xticks(np.arange(
            np.ceil(x_min / self.config.tick_step) * self.config.tick_step,
            np.floor(x_max / self.config.tick_step) * self.config.tick_step + self.config.tick_step,
            self.config.tick_step
        ))
        
        ax.set_yticks(np.arange(
            np.ceil(y_min / self.config.tick_step) * self.config.tick_step,
            np.floor(y_max / self.config.tick_step) * self.config.tick_step + self.config.tick_step,
            self.config.tick_step
        ))
        
        # Set grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis limits
        ax.set_xlim(
            self.config.x_min if self.config.x_min is not None else np.min(self.Xdata),
            self.config.x_max if self.config.x_max is not None else np.max(self.Xdata)
        )
        ax.set_ylim(
            self.config.y_min,
            self.config.y_max if self.config.y_max is not None else np.max(self.Ydata)
        )
        
        # After setting up the main axes ticks and labels
        if self.config.show_right_ticks:
            # Show ticks and labels on both sides of y-axis
            ax.yaxis.set_ticks_position('both')
            # Keep labels on both sides
            ax.tick_params(labelleft=True, labelright=True)
        
        if self.config.show_top_ticks:
            # Show ticks and labels on both sides of x-axis
            ax.xaxis.set_ticks_position('both')
            # Keep labels on both sides
            rotation=self.config.x_tick_rotation
            ax.tick_params(labelbottom=True, labeltop=True, axis='x', rotation=rotation)
        
        # Set axis limits
        ax.set_xlim(
            self.config.x_min if self.config.x_min is not None else np.min(self.Xdata),
            self.config.x_max if self.config.x_max is not None else np.max(self.Xdata)
        )
        ax.set_ylim(
            self.config.y_min,
            self.config.y_max if self.config.y_max is not None else np.max(self.Ydata)
        )
        
        # After setting ticks, rotate x-axis tick labels
        ax.tick_params(axis='x', rotation=self.config.x_tick_rotation)
        # https://stackoverflow.com/questions/2969867/how-do-i-add-space-between-the-ticklabels-and-the-axes
        
        return fig, ax
    
    def add_colorbar(self, plot_obj: Tuple[plt.Figure, plt.Axes, Any], 
                    levels: np.ndarray, labels: List[str]) -> Tuple[plt.Figure, plt.Axes, Any]:
        """
        https://pythonmatplotlibtips.blogspot.com/2019/07/draw-two-axis-to-one-colorbar.html
        """
        fig, ax, contour = plot_obj

        # Create colorbar and manually align it to the plot's height
        cbar = fig.colorbar(contour, ax=ax)

        ax1 = cbar.ax
        ax1.set_aspect('auto')

        fig.canvas.draw()  # Ensure layout is updated
        pos = cbar.ax.get_position()

        # Create and set up normalized (left) axis
        ax2 = ax1.twinx()
        ax2.set_position(pos)
        
        # Calculate normalized positions based on selected normalization type
        if self.rnd_info.intensity_normalization_type == NormalizationType.LOG_RATIO:
            norm_positions = np.log10(levels)/np.log10(levels[-1])
            norm_format = "{x:.3f}"
            norm_label = "Log Ratio"
        elif self.rnd_info.intensity_normalization_type == NormalizationType.DECIBEL:
            norm_positions = 10 * np.log10(levels/levels[-1])
            norm_format = "{x:.1f} dB"
            norm_label = "Intensity (dB)"
        elif self.rnd_info.intensity_normalization_type == NormalizationType.PERCENTAGE:
            norm_positions = (levels/levels[-1]) * 100
            norm_format = "{x:.1f}%"
            norm_label = "Relative Intensity (%)"
        else:  # LOG_SCALE
            norm_positions = (np.log10(levels) - np.log10(levels[0]))/(np.log10(levels[-1]) - np.log10(levels[0]))
            norm_format = "{x:.2f}"
            norm_label = "Log-scale Normalized"
        
        logger.debug(f"Normalized positions ({self.rnd_info.intensity_normalization_type.value}): {norm_positions}") #z
        
        # Set up normalized axis limits and ticks
        ax2.set_ylim(min(norm_positions), max(norm_positions))
        ax2.set_yticks(norm_positions)
        ax2.set_yticklabels([norm_format.format(x=x) for x in norm_positions])
        ax2.yaxis.set_ticks_position('left')
        ax2.yaxis.set_label_position('left')
        
        # Move colorbar position slightly to the right
        pos.x0 += 0.06
        pos.x1 += 0.06
        
        # Set up main (right) axis
        ax1.set_position(pos)
        ax1.yaxis.set_ticks_position('right')
        ax1.yaxis.set_label_position('right')
        ax1.set_yticks(levels)
        ax1.set_yticklabels(labels)
        ax1.set_ylabel(self.config.colorbar_main_label,
                       rotation=90,
                       labelpad=48, # distance from axis to label
                       fontsize=self.config.label_fontsize if hasattr(self.config, 'label_fontsize') else 25)
        
        # Set normalized axis label
        ax2.set_ylabel(norm_label,
                       rotation=90,
                       labelpad=48, # distance from axis to label
                       fontsize=self.config.label_fontsize if hasattr(self.config, 'label_fontsize') else 25)
        
        # Adjust spacing between axes
        cbar.ax.spines['right'].set_position(('outward', 0))
        ax2.spines['left'].set_position(('outward', 0))
        return fig, ax, cbar

    def finalize(self, plot_obj: Tuple[plt.Figure, plt.Axes, Any]) -> None:
        """
        Finalize plot styling.
        This method can be overridden in subclasses for additional styling.
        """
        
        fig, ax, cbar = plot_obj
        fig.canvas.draw()  # Make sure the figure layout is updated

        # Get the main axis position (where the actual plot is)
        ax_pos = ax.get_position()

        # Example: shrink or stretch colorbar to match ax height
        cbar_pos = cbar.ax.get_position()
        cbar.ax.set_position([
            cbar_pos.x0+self.config.colorbar_padding,       # x-position (keep same or adjust)
            ax_pos.y0,         # align bottom of colorbar to ax
            cbar_pos.width,    # keep same width
            ax_pos.height      # match ax height
        ])

    def save_plot(self, plot_obj: Tuple[plt.Figure, plt.Axes, Any], filename: str) -> None:
        fig, ax, _ = plot_obj
        
        # No need to set aspect here anymore as it's handled in create_contour
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.savefig(filename, bbox_inches='tight',
                    dpi=self.config.dpi, format=filename.split('.')[-1])
        plt.close(fig)


def render_spectrum(context: 'SimContext') -> None:
    
    """
    High-level function to render spectrum with specified backend
 -------------
    spec_data is not the final Z values of spectrum (intensities)
        it is amplitudes here, complex values, but they get to be transformed
        using spec_data_operations attribute of RenderingInfo instance

    context = {'spec': self.spec, 'system': self.system, 
                'exp': self.exp, 'diagn': self.diagn, 
                'name': self.name, 
                'spec_eval_setup': self.spec_eval_setup,
                'do_diagn': True}
    """
    spec_data = context.spec
    spec_eval_setup = context.spec_eval_setup
    filename = context.filename
    backend = context.backend

    # TODO not used currently
    do_diagn = context.do_diagn
    diagn = {}

    if backend == 'matplotlib':
        renderer_class=MatplotlibRenderer
    else:
        raise NotImplementedError('Only matplotlib backend is currently supported')
    
    plot_config = spec_eval_setup.rnd_info.style_config

    renderer = renderer_class(spec_data=spec_data, 
                              spec_grid=spec_eval_setup.grid,
                              ev_info=spec_eval_setup.ev_info, 
                              rnd_info=spec_eval_setup.rnd_info, 
                              config=plot_config, do_diagn=do_diagn)
    fig, ax, contour, cbar = renderer.render(filename)
    
    if do_diagn:
        return tuple([fig, ax, contour, cbar]), diagn
    else:
        return tuple([fig, ax, contour, cbar])


def spectral_axis_to_label(axis_dict: dict, divide_by_2pic: bool = True) -> str:
    """
    Utility function.
    Making labels for axes using SpectralAxis.freq_vars dict
    
    axis_dict = SpectralAxis.freq_vars
    """
    terms = []
    for key, coeff in sorted(axis_dict.items(), key=lambda x: x[0]):
        if coeff == 0:
            continue
        sign = '+' if coeff > 0 else '-'
        abs_coeff = abs(coeff)

        if abs_coeff == 1:
            term = f"{sign} \\omega_{{{key[1:]}}}"  # remove 'w' from 'w1'
        else:
            term = f"{sign} {abs_coeff}\\omega_{{{key[1:]}}}"
        terms.append(term)

    if not terms:
        expr = "0"
    else:
        expr = " ".join(terms)
        if expr.startswith('+ '):
            expr = expr[2:]  # remove leading '+ ' for aesthetics

    # Wrap in parentheses if there are multiple terms
    if len(terms) > 1:
        expr = f"({expr})"

    if divide_by_2pic:
        expr = rf"${expr}/2\pi c, \text{{cm}}^{{-1}}$"
    else:
        expr = rf"${expr}, \text{{cm}}^{{-1}}$"

    return expr
