import numpy as np
import matplotlib.pyplot as plt
import matplotlib

"""
d_max / dynrange = d_min

should be in log10 and normalized
"""


def set_figure(figsize, font_dict, to_save=True):
    """
    setting up the figure
    """
    # prep for saving, if saving
    if to_save:
        matplotlib.use('Agg')

    # prep plt
    plt.rcParams['path.simplify'] = True
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.rcParams['axes.titlepad'] = 30
    matplotlib.rc('font', **font_dict)

    # set up a figure, axes
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.1, right=0.9, top=1.05, bottom=0.15)
    return fig, ax


# def set_dynrange(dynamic_range, num_color_levels, num_level_ticks):
def set_dynrange(d_max, dynamic_range, log10):
    # dynamic range, dynamic range max, number of levels
    if log10:
        dynrange_log = np.log10(dynamic_range)
        dmax_log10 = float(int(np.log10(d_max)))

        return dynrange_log, dmax_log10
    else:
        return dynamic_range, d_max


def prep_intensity_log10(intensities, normalized):
    """
    normalization and log10
    """
    # log10 of intensities
    # l10 = np.where(intensities!=0.,np.log10(intensities),0.)
    l10 = np.zeros_like(intensities)
    valid = intensities > 0
    l10[valid] = np.log10(intensities[valid])
    assert np.all(np.isfinite(l10)), "log10 produced NaN or Inf!"

    if normalized == '01':
        minimum_intensity = np.min(l10) if np.min(l10)!=0. else 0.
        intensity_plot = (l10 - minimum_intensity) / (np.max(l10) - minimum_intensity)

    else:
        intensity_plot = l10

    return intensity_plot

def prep_colorbar(intensities, normalized, norm_min):
    """
    setting up the colorbar with d_max
    """
    import matplotlib.colors as colors
    d_max = intensities.max()

    # d_max - max intensity
    dmax_log10 = float(int(np.log10(d_max)))
    if normalized!='01' and type(normalized) != tuple:
        colorbar_norm = colors.LogNorm(vmax=np.ceil(dmax_log10) + 0.2, vmin=norm_min)


# formatting of colorbar tick labels
import matplotlib.ticker as ticker
def fmt(x, pos=''):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def prep_levels(d_max, dynamic_range, num_level_ticks):

    dynrange_log = np.log10(dynamic_range)
    min_dynrange = np.log10(d_max) - dynrange_log
    dt = - (np.log10(d_max) - min_dynrange) / num_level_ticks

    if d_max > 1:
        levels_before_norm = np.arange(np.log10(d_max), min_dynrange, round(dt, 4))
    else:
        levels_before_norm = np.arange(np.log10(d_max), min_dynrange, round(dt, 4))

    levels_ticks = np.sort(levels_before_norm/np.max(levels_before_norm))
    levels_ticks = np.sort(levels_ticks)

    levels_nums = np.sort([float(10**i) for i in levels_before_norm])
    levels_nums_str = [fmt(tick) for tick in levels_nums]

    return levels_nums, levels_ticks, levels_nums_str


def prep_levels_clean(d_max: float, dynamic_range: float, num_level_ticks: int):
    """
    Prepares numerical levels and their normalized counterparts for plotting,
    such as for colorbars in contour plots.

    Parameters:
    ----------
    d_max : float
        Maximum data value to be visualized (e.g., highest density).
    dynamic_range : float
        The ratio between the maximum and minimum value shown (e.g., 1e3).
    num_level_ticks : int
        Number of ticks/levels to show between d_max and d_max/dynamic_range.

    Returns:
    -------
    levels_nums : np.ndarray
        Actual values for each contour level (linear scale).
    levels_ticks : np.ndarray
        Normalized log-scale values between 0 and 1.
    levels_nums_str : list[str]
        Formatted string labels (e.g., '1e5') for the levels.
    """

    # Compute min value (log scale) based on dynamic range
    log_dmax = np.log10(d_max)
    log_min = log_dmax - np.log10(dynamic_range)

    # Step size in log space (negative step to go from d_max downward)
    dt = - (log_dmax - log_min) / num_level_ticks

    # Compute raw log10-levels
    levels_log = np.arange(log_dmax, log_min, dt)

    # Convert log-levels back to actual values
    levels_nums = np.power(10, levels_log)

    # Normalize log values to [0, 1] range (used in colormaps)
    levels_ticks = (levels_log - log_min) / (log_dmax - log_min)

    # Format labels (e.g., '1e5') – you can replace `fmt` with your formatter
    levels_nums_str = [f"{val:.0e}" for val in levels_nums]

    return levels_nums, levels_ticks, levels_nums_str


def compute_log_colorbar_levels(d_max: float, dynamic_range: float, num_levels: int):
    """
    Generate tick values, formatted labels, and normalized positions for a colorbar with logarithmic scaling.

    Parameters
    ----------
    d_max : float
        Maximum data value for the colorbar.
    dynamic_range : float
        Ratio between d_max and minimum value to display. E.g., 1e3 means d_min = d_max / 1e3.
    num_levels : int
        Number of levels/ticks to generate between d_min and d_max (inclusive).

    Returns
    -------
    tick_values : np.ndarray
        The actual tick values (e.g., [1e5, 1e6, ..., 1e8]).
    tick_labels : list of str
        Tick labels formatted in LaTeX style (e.g., ["1.0x10⁵", ...]).
    tick_norm_positions : np.ndarray
        Tick positions normalized to [0, 1] in log10 scale.
    """
    # Calculate min value from dynamic range
    d_min = d_max / dynamic_range
    log_min = np.log10(d_min)
    log_max = np.log10(d_max)

    # evenly spaced log10 levels
    log_ticks = np.linspace(log_min, log_max, num_levels)
    tick_values = np.power(10, log_ticks)

    # Format labels nicely (LaTeX-style strings)
    tick_labels = [f"${val:.1e}$".replace('e', r'\times 10^{') + '}$' for val in tick_values]

    # Normalize log-scale values to [0, 1]
    tick_norm_positions = (log_ticks - log_min) / (log_max - log_min)

    return tick_values, tick_labels, tick_norm_positions


def set_xyz(X, Y, intensity_plot, fig, ax, w1mw2, nicetitle,
            levels, saturation_color, levels_ticks, levels_nums_str,
            maxYX, minY=None):

    if w1mw2:
        y = -(X - Y)
        ax.set_ylabel(r'$(\omega_2-\omega_1)/2\pi c, \text{cm}^{-1}$', fontsize=25, labelpad=21.)
    else:
        y = Y
        ax.set_ylabel(r'$\omega_2/2\pi c, \text{cm}^{-1}$', fontsize=25, labelpad=21.)
    ax.set_xlabel(r'$\omega_1/2\pi c, \text{cm}^{-1}$', fontsize=25, labelpad=21.)

    plt.title(nicetitle)

    # contours colors
    cmap = plt.get_cmap('hot_r').copy()
    cmap.set_extremes(over=saturation_color)

    # contour plot
    cont = ax.contourf(X, y, intensity_plot,
                       levels=levels, cmap=cmap  #'hot_r'
                       # , norm=colorbar_norm
                       , extend='max'
                       )

    # limits of Y axis
    if w1mw2:
        # x_limits = ax.get_xlim()
        if minY is not None:
            ax.set_ylim(minY, maxYX)
        else:
            ax.set_ylim(0, maxYX)

    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=2.1)

    colorbar = plt.colorbar(cont, cax=cax, aspect=65, shrink=0.9,
                            ticks=levels_ticks, format=ticker.FuncFormatter(fmt))
    # colorbar = plt.colorbar(cont, cax=cax,
    #                         ticks=levels_ticks, format=ticker.FuncFormatter(fmt))
    colorbar.set_ticks(levels_ticks)
    colorbar.set_ticklabels([f'{tick:.2f}' for tick in levels_ticks])

    for tick, label in zip(levels_ticks, levels_nums_str):
        colorbar.ax.text(-2.0, tick, label, ha='left', va='center')

    return fig, ax


def finilize_ax(ax, filename, dpi=250, to_save=True):
    import matplotlib.ticker as ticker

    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)

    if to_save:
        plt.savefig(filename, dpi=dpi, format='svg')


def render_spectrum_with_debug(intensities, w1m, w2m, filename, nicetitle='yes'):
    """
    Helper function to render the spectrum figure with debugging.
    """
    print(f"Rendering spectrum: {filename}")
    print(f"Intensity data stats - Min: {np.min(intensities)}, Max: {np.max(intensities)}, Mean: {np.mean(intensities)}")
    # Normalize intensities for rendering
    normalized_intensities = intensities / np.max(intensities)
    print(f"Normalized intensity stats - Min: {np.min(normalized_intensities)}, Max: {np.max(normalized_intensities)}")
    fig, ax = set_figure(figsize=(40, 60), font_dict={'size': 20}, to_save=True)
    levels_nums, levels_ticks, levels_nums_str = prep_levels(
        d_max=np.max(normalized_intensities),
        dynamic_range=100,
        num_level_ticks=10
    )
    print(f"Levels: {levels_nums}")
    print(f"Ticks: {levels_ticks}")
    intensity_plot = prep_intensity_log10(normalized_intensities, normalized='01')
    set_xyz(
        w1m, w2m, intensity_plot, fig, ax,
        w1mw2=True, nicetitle=nicetitle,
        levels=levels_ticks, saturation_color='#FF00FF',
        levels_ticks=levels_ticks,
        levels_nums_str=levels_nums_str,
        maxYX=3000., minY=None
    )
    finilize_ax(ax, filename=filename, dpi=250, to_save=True)


def render_spectrum(intensities, w1m, w2m, filename, dynamic_range, num_level_ticks=10, nicetitle='yes'):
    """
    Helper function to render the spectrum figure.
    """
    # from rich import print
    from ... wilson_intensities.utils.utils import coolprint

    coolprint('1. Setting figure...')
    fig, ax = set_figure(figsize=(35, 45), font_dict={'size': 20}, to_save=True)

    coolprint('2. Setting levels, level ticks and level labels...')
    levels_nums, levels_ticks, levels_nums_str = prep_levels(
        d_max=np.max(intensities),
        dynamic_range=dynamic_range,
        num_level_ticks=num_level_ticks
    )
    assert all(upper > lower for upper, lower in zip(levels_nums[1:], levels_nums[:-1])), "Invalid contour"

    np.set_printoptions(precision=4,suppress=False)

    # print('\nlevels_nums', np.array(levels_nums))
    print('\nlevels_nums_str', levels_nums_str)
    print('levels_ticks', np.array(levels_ticks), '\n')

    coolprint('3. Log10 of intensity and normalization...')
    intensity_plot = prep_intensity_log10(intensities, normalized='01')
    print('intensity_plot', intensity_plot)

    hist, bin_edges = np.histogram(intensity_plot, bins=10)
    print('\nintensity_plot log10 from render_spectrum:')
    print("Histogram counts:", hist)
    print("Bin edges:", bin_edges, '\n')

    coolprint('4. Prepare XYZ, make contourf and colorbar... '
              'Using previously configured intensity, levels_ticks, levels_nums_str...')
    fig, ax = set_xyz(
        X=w1m, Y=w2m, intensity_plot=intensity_plot, 
        fig=fig, ax=ax,
        w1mw2=True, nicetitle=nicetitle,
        levels=levels_ticks, saturation_color='#FF00FF',
        levels_ticks=levels_ticks,
        levels_nums_str=levels_nums_str,
        maxYX=3000., minY=None
    )

    coolprint('5. Finish axes settings...')
    finilize_ax(ax, filename=filename, dpi=250, to_save=True)
