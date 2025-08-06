from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from enum import Enum

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
class SpectrumData:
    """Container for spectrum data and metadata"""
    intensities: np.ndarray
    w1: np.ndarray  # omega 1 frequencies
    w2: np.ndarray  # omega 2 frequencies
    title: str = ""
    dynamic_range: float = 1e3
    num_levels: int = 10
    reference_max: Optional[float] = None  # For normalization reference

@dataclass
class PlotConfig:
    """Configuration for plot styling"""
    figsize: Tuple[int, int] = (35, 45)
    label_fontsize: int = 25
    font_dict: Dict[str, Any] = field(default_factory=lambda: {'size': 20})
    colormap: str = 'magma'  # Better contrast colormap
    saturation_color: str = '#FF00FF'
    dpi: int = 250
    max_frequency: float = 3000.0
    min_frequency: Optional[float] = None
    tick_step: float = 200.0  # Step size for both axes ticks
    equal_aspect: bool = True  # Force equal aspect ratio for axes
    no_data_color: str = '#E0E0E0'  # Light gray
    below_range_color: str = '#F8F8F8'  # Very light gray
    data_edge_color: str = 'black'
    data_edge_style: str = '--'  # Line style for data boundary
    data_edge_width: float = 0.75
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    colorbar_main_label: str = "Intensity"
    colorbar_norm_label: str = "Normalized"
    normalization_type: NormalizationType = NormalizationType.LOG_RATIO  # Add this line
    show_top_ticks: bool = False
    show_right_ticks: bool = False
    x_tick_rotation: float = 45  # Add this line for configurable rotation
    colormap_spacing: str = None  # Options: "log", "power", "linear"
    colormap_power: float = 0.5    # For power-law spacing

class LevelCalculator:
    """Handles calculation of contour levels and normalization"""
    
    @staticmethod
    def normalize_intensities(intensities: np.ndarray) -> np.ndarray:
        """Normalize intensities to [0,1] range"""
        return intensities / np.max(intensities)
    
    @staticmethod
    def compute_levels(d_max: float, dynamic_range: float, num_levels: int, 
                  ref_max: Optional[float] = None, colormap_spacing: str = None) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
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
            alpha = 0.5  # Adjust this value to change color distribution
            power_space = np.power(np.linspace(0, 1, num_levels), alpha)
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
    """Abstract base class for spectrum rendering"""
    
    def __init__(self, data: SpectrumData, config: PlotConfig):
        self.data = data
        self.config = config
        self.level_calc = LevelCalculator()
    
    @abstractmethod
    def initialize_plot(self) -> Any:
        """Initialize plotting surface"""
        pass
    
    @abstractmethod
    def create_contour(self, plot_obj: Any, levels: np.ndarray, normalized_data: np.ndarray) -> Any:
        """Create contour plot"""
        pass
    
    @abstractmethod
    def add_colorbar(self, plot_obj: Any, levels: np.ndarray, labels: List[str]) -> None:
        """Add colorbar to plot"""
        pass
    
    @abstractmethod
    def save_plot(self, plot_obj: Any, filename: str) -> None:
        """Save plot to file"""
        pass
    
    def render(self, filename: str) -> None:
        """Main rendering pipeline"""
        # Calculate levels with both original and normalized scales
        levels, labels, norm_positions, norm_labels = self.level_calc.compute_levels(
            np.max(self.data.intensities),
            self.data.dynamic_range,
            self.data.num_levels,
            ref_max=self.data.reference_max,
            colormap_spacing=self.config.colormap_spacing
        )
        
        # Create and save plot using original data
        plot_obj = self.initialize_plot()
        plot_obj = self.create_contour(plot_obj, levels, self.data.intensities)
        self.add_colorbar(plot_obj, levels, labels)
        self.save_plot(plot_obj, filename)

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
    
    def create_contour(self, plot_obj: Tuple[plt.Figure, plt.Axes], 
                      levels: np.ndarray, 
                      data: np.ndarray) -> Tuple[plt.Figure, plt.Axes, Any]:
        fig, ax = plot_obj
        
        # Create masked arrays
        no_data_mask = np.isnan(data)
        d_min = np.max(data) / self.data.dynamic_range
        below_range_mask = (~no_data_mask) & (data < d_min)
        
        # Setup base colormap
        cmap = plt.get_cmap(self.config.colormap).copy()
        cmap.set_over(self.config.saturation_color)
        
        # Fill no-data and below-range regions
        ax.contourf(self.data.w1, -(self.data.w1 - self.data.w2),
                   no_data_mask,
                   levels=[0, 0.5, 1],
                   colors=[self.config.no_data_color])
        
        ax.contourf(self.data.w1, -(self.data.w1 - self.data.w2),
                   below_range_mask,
                   levels=[0, 0.5, 1],
                   colors=[self.config.below_range_color])
        
        # Create logarithmic normalization for color mapping
        norm = matplotlib.colors.LogNorm(vmin=levels[0], vmax=levels[-1])
        
        # Plot main data with normalized colors
        contour = ax.contourf(self.data.w1, -(self.data.w1 - self.data.w2), 
                           data,
                           levels=levels,
                           norm=norm,  # Add normalization
                           cmap=cmap,
                           extend='max')
        
        # Single clean edge line
        ax.contour(self.data.w1, -(self.data.w1 - self.data.w2),
                  ~no_data_mask,
                  levels=[0.5],
                  colors=[self.config.data_edge_color],
                  linewidths=[self.config.data_edge_width])
        
        # Set up axes labels
        label_fontsize = self.config.label_fontsize if hasattr(self.config, 'label_fontsize') else 25
        ax.set_xlabel(r'$\omega_1/2\pi c, \text{cm}^{-1}$', 
                     fontsize=label_fontsize, labelpad=65.) # labelpad - distance from axis to label
        ax.set_ylabel(r'$(\omega_2-\omega_1)/2\pi c, \text{cm}^{-1}$', 
                     fontsize=label_fontsize, labelpad=65.) # labelpad - distance from axis to label
        
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
            self.config.x_min if self.config.x_min is not None else np.min(self.data.w1),
            self.config.x_max if self.config.x_max is not None else np.max(self.data.w1)
        )
        ax.set_ylim(
            self.config.y_min,
            self.config.y_max if self.config.y_max is not None else np.max(-(self.data.w1 - self.data.w2))
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
            self.config.x_min if self.config.x_min is not None else np.min(self.data.w1),
            self.config.x_max if self.config.x_max is not None else np.max(self.data.w1)
        )
        ax.set_ylim(
            self.config.y_min,
            self.config.y_max if self.config.y_max is not None else np.max(-(self.data.w1 - self.data.w2))
        )
        
        # After setting ticks, rotate x-axis tick labels
        ax.tick_params(axis='x', rotation=self.config.x_tick_rotation)
        # https://stackoverflow.com/questions/2969867/how-do-i-add-space-between-the-ticklabels-and-the-axes
        
        return fig, ax, contour
    
    def add_colorbar(self, plot_obj: Tuple[plt.Figure, plt.Axes, Any], 
                    levels: np.ndarray, labels: List[str]) -> None:
        """
        https://pythonmatplotlibtips.blogspot.com/2019/07/draw-two-axis-to-one-colorbar.html
        """
        fig, ax, contour = plot_obj
        
        # Create main colorbar with some padding
        cbar = fig.colorbar(contour, ax=ax, pad=0.08)
        
        # Get colorbar axes and adjust position
        pos = cbar.ax.get_position()
        ax1 = cbar.ax
        ax1.set_aspect('auto')
        
        # Create and set up normalized (left) axis
        ax2 = ax1.twinx()
        ax2.set_position(pos)
        
        # Calculate normalized positions based on selected normalization type
        if self.config.normalization_type == NormalizationType.LOG_RATIO:
            norm_positions = np.log10(levels)/np.log10(levels[-1])
            norm_format = "{x:.3f}"
            norm_label = "Log Ratio"
        elif self.config.normalization_type == NormalizationType.DECIBEL:
            norm_positions = 10 * np.log10(levels/levels[-1])
            norm_format = "{x:.1f} dB"
            norm_label = "Intensity (dB)"
        elif self.config.normalization_type == NormalizationType.PERCENTAGE:
            norm_positions = (levels/levels[-1]) * 100
            norm_format = "{x:.1f}%"
            norm_label = "Relative Intensity (%)"
        else:  # LOG_SCALE
            norm_positions = (np.log10(levels) - np.log10(levels[0]))/(np.log10(levels[-1]) - np.log10(levels[0]))
            norm_format = "{x:.2f}"
            norm_label = "Log-scale Normalized"
        
        logger.debug(f"Normalized positions ({self.config.normalization_type.value}): {norm_positions}")
        
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
                       fontsize=self.config.font_dict.get('size', 20))
        
        # Set normalized axis label
        ax2.set_ylabel(norm_label,
                       rotation=90,
                       labelpad=48, # distance from axis to label
                       fontsize=self.config.font_dict.get('size', 20))
        
        # Adjust spacing between axes
        cbar.ax.spines['right'].set_position(('outward', 0))
        ax2.spines['left'].set_position(('outward', 0))
        

    def save_plot(self, plot_obj: Tuple[plt.Figure, plt.Axes, Any], filename: str) -> None:
        fig, ax, contour = plot_obj
        
        # No need to set aspect here anymore as it's handled in create_contour
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.savefig(filename, dpi=self.config.dpi, format='svg')
        plt.close(fig)

# Example usage:
def render_spectrum(intensities: np.ndarray, 
                   w1: np.ndarray,
                   w2: np.ndarray,
                   filename: str,
                   plort_config: Optional[PlotConfig] = None,
                   renderer_class=MatplotlibRenderer,
                   **kwargs) -> None:
    """
    High-level function to render spectrum with specified backend
    """
    data = SpectrumData(
        intensities=intensities,
        w1=w1,
        w2=w2,
        **kwargs
    )
    
    renderer = renderer_class(data, plort_config)
    renderer.render(filename)