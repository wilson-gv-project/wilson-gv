from typing import Tuple, List, Any
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from .spectrum_renderer import SpectrumRenderer 

import logging
logger = logging.getLogger("wilson."+__name__)

class MatplotlibRenderer(SpectrumRenderer):
    """Matplotlib implementation of spectrum renderer"""
    
    def initialize_plot(self) -> Tuple[plt.Figure, plt.Axes]:
        matplotlib.use('Agg')
        plt.rcParams['path.simplify'] = True
        plt.rcParams['agg.path.chunksize'] = 10000
        
        matplotlib.rc('font', **self.config.font_dict)
        
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
        no_data_mask, below_range_mask = self._create_data_masks(data)

        if self.config.other_colors:
            # Fill below-range regions
            ax.contourf(self.Xdata, self.Ydata,
                    below_range_mask,
                    levels=[0, 0.5, 1],
                    colors=[self.config.below_range_color])

        # Setup base colormap
        cmap = plt.get_cmap(self.config.colormap).copy()
        cmap.set_over(self.config.saturation_color)
                
        # Create logarithmic normalization for color mapping
        norm = matplotlib.colors.LogNorm(vmin=levels[0], vmax=levels[-1])

        # Plot main data with normalized colors
        contour = ax.contourf(self.Xdata, self.Ydata, 
                           data,
                           levels=levels,
                           norm=norm,  # Add normalization
                           cmap=cmap,
                           extend='max')

        # Replace contourf with pcolormesh for the mask
        # Use a masked array so that 'False' values are completely transparent
        masked_no_data = np.ma.masked_where(~no_data_mask, no_data_mask)
        ax.contourf(self.Xdata, self.Ydata,
                masked_no_data,
                levels=[0, 0.5, 1],
                colors=[self.config.no_data_color])

        # Hide contour linestroke on pyplot.contourf to get only fills
        # https://stackoverflow.com/questions/8263769/hide-contour-linestroke-on-pyplot-contourf-to-get-only-fills
        contour.set_edgecolor("face")

        # if self.config.other_colors:   
        #     # Single clean edge line
        #     ax.contour(self.Xdata, self.Ydata,
        #             ~no_data_mask,
        #             levels=[0.5],
        #             colors=[self.config.data_edge_color],
        #             linewidths=[self.config.data_edge_width])

        return fig, ax, contour


    def setup_axes(self, plot_obj):
        fig, ax = plot_obj

        # Set up axes labels
        label_fontsize = self.config.label_fontsize if hasattr(self.config, 'label_fontsize') else 25

        xlabel_str = self.xyz_labels.get('x', r'$default x /2\pi c, \text{cm}^{-1}$')
        ylabel_str = self.xyz_labels.get('y', r'$default y /2\pi c, \text{cm}^{-1}$')

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
                     levels: np.ndarray, labels: List[str]):
        fig, ax, contour = plot_obj
        cbar = fig.colorbar(contour, ax=ax)

        # right axis — absolute intensity (positions handled by colorbar)
        ax1 = cbar.ax
        ax1.set_aspect('auto')
        fig.canvas.draw()
        pos = cbar.ax.get_position()

        # left axis — fractional, positioned via log-stretch
        ax2 = ax1.twinx()
        ax2.set_position(pos)

        is_normalized = self.rnd_info.reference_max is not None

        # Right axis: always absolute intensity.
        if is_normalized:
            ref = self.rnd_info.reference_max
            right_labels = [f"${val * ref:.1e}$" for val in levels]
        else:
            right_labels = labels  # already absolute, formatted in compute_levels
        right_ticks = levels
        right_label_text = self.config.colorbar_main_label

        # Left axis: uniformly-spaced positions (log-stretched to [0, 1]) so ticks
        # align visually with the right axis. Labels differ by mode.
        left_ticks = (np.log10(levels) - np.log10(levels[0])) / \
                    (np.log10(levels[-1]) - np.log10(levels[0]))
        if is_normalized:
            left_labels = [f"{val:.3f}" for val in levels]
            left_label_text = "Normalized (a.u.)"
        else:
            left_labels = [f"{val / levels[-1]:.3f}" for val in levels]
            left_label_text = "Normalized to self (a.u.)"

        # Left axis (ax2)
        ax2.set_ylim(min(left_ticks), max(left_ticks))
        ax2.set_yticks(left_ticks)
        ax2.set_yticklabels(left_labels)
        ax2.yaxis.set_ticks_position('left')
        ax2.yaxis.set_label_position('left')
        ax2.set_ylabel(left_label_text, rotation=90, labelpad=48,
                    fontsize=getattr(self.config, 'label_fontsize', 25))

        # Right axis (ax1) — shift slightly to make room
        pos.x0 += 0.06
        pos.x1 += 0.06
        ax1.set_position(pos)
        ax1.yaxis.set_ticks_position('right')
        ax1.yaxis.set_label_position('right')
        ax1.set_yticks(right_ticks)
        ax1.set_yticklabels(right_labels)
        ax1.set_ylabel(right_label_text, rotation=90, labelpad=48,
                    fontsize=getattr(self.config, 'label_fontsize', 25))

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

        if self.rnd_info.style_config.axes_limits is not None:
            ax.set_xlim(*self.rnd_info.style_config.axes_limits['x'])
            ax.set_ylim(*self.rnd_info.style_config.axes_limits['y'])

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

