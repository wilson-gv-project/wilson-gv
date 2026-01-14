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
    from ...wilson_main.spectrum_abstractions import EvaluationInfo, RenderingInfo


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
    colormap_spacing: str = "log"  # Options: "log", "linear"

    def __post_init__(self):
        if not isinstance(self.figsize, tuple):
            raise TypeError("figsize needs to be given as a tuple")
        if any(i < 0 for i in self.figsize):
            raise ValueError("Negative figsize was provided")
        if self.tick_step < 0:
            raise ValueError("Negative tick_step was provided")
        
        if self.x_min is not None:
            if not isinstance(float(self.x_min), float):
                raise TypeError("x_min needs to be given a float")
            self.x_min = float(self.x_min)
        if self.y_min is not None:
            if not isinstance(float(self.y_min), float):
                raise TypeError("y_min needs to be given a float")
            self.y_min = float(self.y_min)
        
        if self.x_max is not None:
            if not isinstance(float(self.x_max), float):
                raise TypeError("x_max needs to be given a float")
            self.x_max = float(self.x_max)
        if self.y_max is not None:
            if not isinstance(float(self.y_max), float):
                raise TypeError("y_max needs to be given a float")
            self.y_max = float(self.y_max)


class LevelCalculator:
    """
    Handles calculation of contour levels and normalization
    
    organizes the logic for computing levels and labels
    based on dynamic range, number of levels, and normalization type.
    """

    @staticmethod
    def compute_levels(intensities: float, dynamic_range: float, 
                       nlevels: int, colormap_spacing: str = None) -> Tuple[np.ndarray, List[str]]:
        """Calculate levels for contours and colorbar ticks"""
        d_max = np.max(intensities)
        d_min = d_max / dynamic_range
        
        if colormap_spacing == "log":
            level_values = np.logspace(np.log10(d_min), np.log10(d_max), nlevels)
        
        elif colormap_spacing == "linear":
            level_values = np.linspace(d_min, d_max, nlevels)
        else:
            raise ValueError('Choose log or linear colormap_spacing')
        level_labels = [f"${val:.1e}$" for val in level_values]

        return level_values, level_labels


class SpectrumRenderer(ABC):
    """
    Abstract base class for spectrum rendering
    
    PlotConfig instance would normally be stored in the RenderingInfo instance
    but can be provided as config parameter

    spec_data - amplitudes data from evaluation procedure
    """
    
    def __init__(self, 
                 spec_data: np.ndarray | dict,
                 spec_grid: dict,
                 ev_info: "EvaluationInfo",
                 rnd_info: "RenderingInfo", 
                 do_diagn: bool = False):

        self.spec_data = spec_data
        self.rnd_info = rnd_info
        self.ev_info = ev_info
        self.spec_grid = spec_grid

        # TODO not used currently
        self.do_diagn = do_diagn
        
        self.config = self.rnd_info.style_config
        self.level_calc = LevelCalculator()
        self.intensities = None

        self.Xdata = None
        self.Ydata = None
        self.Zdata = None

    def _validate_inputs(self):
        """
        data and settings should not contradict:
        - appropriate number of axes
        """

        return

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
        
        if self.spec_grid is None:
            raise ValueError('This SpectrumRenderer.spec_grid is None')        

        self.xyz_labels = {'x': None, 'y': None, 'z': None}
        for i, o_k in enumerate(list(self.spec_grid.keys())):
            self.xyz_labels[list(self.xyz_labels.keys())[i]] = o_k
        
        if len(self.spec_grid)==3:
            self.Xdata, self.Ydata, self.Zdata = list(self.spec_grid.values())
        elif len(self.spec_grid)==2:
            self.Xdata, self.Ydata = list(self.spec_grid.values())

    def validate_inputs(self):
        if not isinstance(self.spec_data, np.ndarray):
            raise TypeError("spec_data should be a np.ndarray")
        if self.spec_data.size == 0:
            raise ValueError("spec_data array should not be empty")
        if not isinstance(self.spec_grid, dict):
            raise TypeError("spec_grid should be a dictionary with X,Y,(Z) data")

        for key, val in self.spec_grid.items():
            if not isinstance(val, np.ndarray):
                raise TypeError(f"spec_grid[{key!r}] is not a np.ndarray")
            if val.size == 0:
                raise ValueError(f"spec_grid[{key!r}] is an empty array")


    def validate_data_2d(self):
        if self.Xdata is None:
            raise ValueError("Xdata was not set")
        if self.Ydata is None:
            raise ValueError("Ydata was not set")

        if self.intensities.ndim != 2:
            raise ValueError("intensities in renderer must be a 2D array")

        if self.Xdata.shape != self.intensities.shape or self.Ydata.shape != self.intensities.shape:
            raise ValueError("X,Y and intensities data do not match in shape:\n"
                             f"  x.shape = {self.Xdata.shape}\n"
                             f"  y.shape = {self.Ydata.shape}\n"
                             f"  z.shape = {self.intensities.shape}\n"
                             )

    def render(self, filename: str) -> None:
        """Main rendering pipeline"""
        self.validate_inputs()

        # prepare data for contour plotting with spec_data_operations and spec_grid.axes
        self.prep_data(spec_data_operations=self.rnd_info.spec_data_operations)
        self.validate_data_2d()

        log10 = True if self.rnd_info.intensity_normalization_type is not None else False
        # Calculate levels with both original and normalized scales
        levels, labels = self.level_calc.compute_levels(
            intensities=self.intensities,
            dynamic_range=self.ev_info.dynamic_range,
            nlevels=self.rnd_info.nlevels,
            colormap_spacing=self.config.colormap_spacing
        )
        self.levels = levels
        self.labels = labels

        fig, ax = self.initialize_plot()
        fig, ax, contour = self.create_contour(plot_obj=(fig, ax), levels=levels, data=self.intensities)
        fig, ax = self.setup_axes(plot_obj=(fig, ax))
        fig, ax, cbar = self.add_colorbar(plot_obj=(fig, ax, contour), levels=levels, labels=labels)
        
        self.finalize(plot_obj=(fig, ax, cbar))
        self.save_plot(plot_obj=(fig, ax, cbar), filename=filename)

        return fig, ax, contour, cbar

