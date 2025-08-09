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
from enum import Enum


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wilson_main.abstractions import SpectralGrid, EvaluationInfo, RenderingInfo


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
    other_colors: bool = True
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
            elif spec_data_operations == 'none':
                self.intensities = self.spec_data
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

