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

    levels_before_norm = np.sort(np.arange(np.log10(d_max), min_dynrange, round(dt,4)))
    levels_ticks = levels_before_norm/np.log10(d_max)

    levels_nums = [10**i for i in levels_before_norm]
    # levels_nums_str = [f'{tick:.2e}' for tick in levels_nums]
    levels_nums_str = [fmt(tick) for tick in levels_nums]

    # levels = [d_max * 10.0 ** (-1.0 * dynrange_log *
    #                                      (float(num_color_levels - 1 - i) / (num_color_levels - 1)))
    #                for i in range(num_color_levels)]

    return levels_nums, levels_ticks, levels_nums_str


def set_xyz(X, Y, intensity_plot, fig, ax, w1mw2, nicetitle,
            levels, saturation_color, levels_ticks, levels_nums_str,
            maxYX, minY=None):

    if w1mw2:
        y = -(X - Y)
        ax.set_ylabel(r'$(\omega_2-\omega_1)/2\pi c, \text{cm}^{-1}$', fontsize=25, labelpad=21.)
        # ax.set_ylabel(r'(\\omega_2-\\omega_1)/2\pi c, \\text{cm}^{-1}', fontsize=18)
    else:
        y = Y
        # ax.set_ylabel(r'$\\omega_2/2\pi c, \\text{cm}^{-1}$', fontsize=18)
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
