"""
1D and 2D spectra:
    plot_spectrum(intensity, grid, ...)   any intensity array on a grid
    plot_rsp_func(spec, ...)              the new evaluator's result (EvalRspResult, refac_rsp_eval):
                                          plot_spectrum of |spec.amplitude|^2 on spec.mesh
    plot_feature_boxes(ax, spec)          the outline of each feature's box on a plot of spec
Both show intensity / top, from 1 / dynrange to 1 on a log scale; top: the largest intensity, or reference_max
(spectra on one shared scale).
    1D  a line; y axis log
    2D  filled contours, the key steps of MatplotlibRenderer without its setup objects (RenderingInfo,
        EvaluationInfo, PlotConfig): nlevels log-spaced levels; magma; below the lowest level: the axes'
        background; above the top level: over_color; colorbar ticks as fractions of the top, 2 significant
        digits
"""
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import EvalRspResult


def plot_spectrum(intensity: np.ndarray, grid: dict[str, np.ndarray], *,
                  dynrange: float,
                  nlevels: int = 12,
                  reference_max: float | None = None,
                  ax: 'Axes | None' = None,
                  axis_labels: dict[str, str] | None = None,
                  cmap: str = 'magma',
                  below_color: str = '#F8F8F8',
                  over_color: str = '#FF00FF'):
    """
    A 1D or 2D intensity array on its grid: first axis of grid along x, second (2D) along y.
        fig, ax, contour, cbar = plot_spectrum(np.abs(amplitude)**2, {'A': A, 'B': B}, dynrange=500.)
        fig, ax, line, _ = plot_spectrum(intensity_a, {'A': a_points}, dynrange=500.)
        fig.savefig('spectrum.png')
    intensity      1D or 2D, real, e.g. abs(amplitude)**2, in au. Values <= 0 count as below the lowest level.
    grid           one entry per dimension of intensity, each a full array of the grid points with intensity's
                   shape: 1D the points themselves; 2D e.g. np.meshgrid(a_points, b_points, indexing='ij'),
                   EvalRspResult.mesh or the old spec_eval_setup.grid
    dynrange       the plot's dynamic range, top / lowest intensity shown, e.g. 500. For a spectrum from
                   spectrum_on_grid, use the box rule's dynrange: with a larger one the plot shows where the boxes
                   cut the tails.
    reference_max  au; None: each plot on its own scale. A number: intensity / reference_max, so several spectra
                   share one scale; above the reference: 1D above 1, 2D over_color.
    ax             plot into this axes; None: a new figure
    axis_labels    e.g. {'A': r'$\\omega_1/2\\pi c$, cm$^{-1}$'}; default: the axis name with cm-1
    nlevels, cmap, below_color, over_color: 2D only
    returns (fig, ax, contour, colorbar) for 2D, (fig, ax, line, None) for 1D; e.g. ax.set_yscale('linear')
    for a 1D plot on a linear scale.
    """
    if np.iscomplexobj(intensity):
        raise TypeError('intensity must be real, e.g. abs(amplitude)**2 (plot_rsp_func takes the amplitudes)')
    if intensity.ndim not in (1, 2) or len(grid) != intensity.ndim:
        raise ValueError(f'plot_spectrum plots 1D and 2D spectra, one grid axis per dimension: got intensity of '
                         f'shape {intensity.shape} and the axes {list(grid)}')
    for name, points in grid.items():
        if np.shape(points) != intensity.shape:
            raise ValueError(f'grid[{name!r}] has the shape {np.shape(points)}, intensity {intensity.shape}: '
                             f'each axis needs a full array of the grid points')
    if not dynrange > 1.:
        raise ValueError(f'dynrange must be > 1, got {dynrange}')

    top = reference_max if reference_max is not None else intensity.max()
    if not top > 0.:
        raise ValueError('nothing to plot: no intensity above 0')
    fraction = intensity / top
    if reference_max is None:
        scale_label = f'intensity / max   (max = {top:.2e} au)'
    else:
        scale_label = f'intensity / reference   (reference = {reference_max:.2e} au)'

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5) if intensity.ndim == 1 else (8, 7))
    fig = ax.figure
    labels = axis_labels or {}
    names = list(grid)
    ax.set_xlabel(labels.get(names[0], f'{names[0]}, cm$^{{-1}}$'))
    ax.grid(True, linestyle='--', alpha=0.7)

    if intensity.ndim == 1:
        points = grid[names[0]]
        (line,) = ax.plot(points, fraction)
        ax.set_yscale('log')
        ax.set_ylim(1. / dynrange, 1.2 * max(1., fraction.max()))
        ax.set_xlim(points.min(), points.max())
        ax.set_ylabel(scale_label)
        return fig, ax, line, None

    levels = np.logspace(-np.log10(dynrange), 0., nlevels)          # fractions of the top: 1 / dynrange .. 1
    colormap = plt.get_cmap(cmap).with_extremes(over=over_color)
    contour = ax.contourf(grid[names[0]], grid[names[1]], fraction, levels=levels,
                          norm=LogNorm(levels[0], levels[-1]), cmap=colormap, extend='max')
    contour.set_edgecolor('face')           # no lines between the filled levels
    ax.set_facecolor(below_color)           # contourf leaves everything below levels[0] empty
    ax.set_ylabel(labels.get(names[1], f'{names[1]}, cm$^{{-1}}$'))
    ax.set_aspect('equal', adjustable='box')

    cbar = fig.colorbar(contour, ax=ax, ticks=levels)
    # 2 significant digits, '#' keeps the trailing zeros: 1.0, 0.10, 0.00010 (.3f gives 0.000 below 0.0005)
    cbar.ax.set_yticklabels([f'{v:#.2g}' for v in levels])
    cbar.set_label(scale_label)
    return fig, ax, contour, cbar


def plot_rsp_func(spec: 'EvalRspResult', *, dynrange: float, boxes: bool = False, **options):
    """
    The new evaluator's spectrum (spectrum_on_grid), 1D or 2D: plot_spectrum of |spec.amplitude|^2 on
    spec.mesh.
        fig, ax, contour, cbar = plot_rsp_func(spec, dynrange=500.)
    boxes    also outline each feature's box (plot_feature_boxes, default look)
    options  the keyword arguments of plot_spectrum, e.g. reference_max, ax, axis_labels
    """
    out = plot_spectrum(np.abs(spec.amplitude) ** 2, spec.mesh, dynrange=dynrange, **options)
    if boxes:
        plot_feature_boxes(out[1], spec)
    return out


def plot_feature_boxes(ax: 'Axes', spec: 'EvalRspResult', *,
                       full_color: str = '#00BFFF',
                       contrib_color: str = '#7FFF00',
                       linewidth: float = 0.8,
                       peaks: bool = True):
    """
    Outlines each feature's box on a plot of spec. With spectrum_on_grid's default (cut_tails=False)
    the box only decided that the feature is kept; with cut_tails=True the box is the part of the
    grid where the feature was evaluated, and its tail beyond is cut off.
        fig, ax, _, _ = plot_rsp_func(spec, dynrange=500.)
        plot_feature_boxes(ax, spec)
    2D: one rectangle per box. 1D: a vertical line at each box edge.
        full_features (peak inside the window)   solid lines, full_color
        contrib_features (peak outside)          dashed lines, contrib_color
    peaks: also mark each peak, 2D with a cross, 1D with a tick at the bottom.
    A flat feature's box is the whole window, so it gets no outline. The axes limits stay as they
    are: of a box larger than the plot, only the part inside shows.
    """
    names = list(spec.axis_points)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    window = spec.spectral_window
    for features, color, style in ((window.full_features, full_color, '-'),
                                   (window.contrib_features, contrib_color, '--')):
        boxed = [f for f in features if not f.is_flat and f.feat_box is not None]
        if not boxed:
            continue
        x = names[0]
        if len(names) == 1:
            for f in boxed:
                for edge in f.feat_box.bounds[x]:  # type: ignore
                    ax.axvline(edge, color=color, linestyle=style, linewidth=linewidth)
            if peaks:
                # x in data units, y as a fraction of the axes height
                ax.plot([f.location[x] for f in boxed], [0.] * len(boxed), '|', color=color,
                        markersize=10, transform=ax.get_xaxis_transform(), clip_on=False)
            continue
        y = names[1]
        rects = [Rectangle((f.feat_box.bounds[x][0], f.feat_box.bounds[y][0]),  # type: ignore
                           f.feat_box.bounds[x][1] - f.feat_box.bounds[x][0],  # type: ignore
                           f.feat_box.bounds[y][1] - f.feat_box.bounds[y][0])  # type: ignore
                 for f in boxed]
        ax.add_collection(PatchCollection(rects, facecolor='none', edgecolor=color,
                                          linestyle=style, linewidth=linewidth))
        if peaks:
            ax.plot([f.location[x] for f in boxed], [f.location[y] for f in boxed], 'x',
                    color=color, markersize=4)
    # the boxes must not widen the view
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
