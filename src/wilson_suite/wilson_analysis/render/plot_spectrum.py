"""
1D and 2D spectra:
    plot_spectrum(intensity, grid, ...)   any intensity array on a grid
    plot_rsp_func(spec, ...)              the new evaluator's result (EvalRspResult, refac_rsp_eval):
                                          plot_spectrum of |amplitude|^2 on spec.grid_mesh
Both show intensity / top, from 1 / dynrange to 1 on a log scale; top: the largest intensity, or reference_max
(spectra on one shared scale).
    1D  a line; y axis log
    2D  filled contours, the key steps of MatplotlibRenderer without its setup objects (RenderingInfo,
        EvaluationInfo, PlotConfig): nlevels log-spaced levels; magma; below the lowest level: the axes'
        background; above the top level: over_color; colorbar ticks as fractions of the top
"""
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

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
                   EvalRspResult.grid_mesh or the old spec_eval_setup.grid
    dynrange       the plot's dynamic range, top / lowest intensity shown, e.g. 500. For a spectrum from
                   draw_spectrum, use the box rule's dynrange: with a larger one the plot shows where the boxes
                   cut the tails.
    reference_max  au; None: each plot on its own scale. A number: intensity / reference_max, so several spectra
                   share one scale; above the reference: 1D above 1, 2D over_color.
    ax             draw into this axes; None: a new figure
    axis_labels    e.g. {'A': r'$\\omega_1/2\\pi c$, cm$^{-1}$'}; default: the axis name with cm-1
    nlevels, cmap, below_color, over_color: 2D only
    returns (fig, ax, contour, colorbar) for 2D, (fig, ax, line, None) for 1D; e.g. ax.set_yscale('linear')
    for a 1D plot on a linear scale.
    """
    if np.iscomplexobj(intensity):
        raise TypeError('intensity must be real, e.g. abs(amplitude)**2 (plot_rsp_func takes the amplitudes)')
    if intensity.ndim not in (1, 2) or len(grid) != intensity.ndim:
        raise ValueError(f'plot_spectrum draws 1D and 2D spectra, one grid axis per dimension: got intensity of '
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
    cbar.ax.set_yticklabels([f'{v:.3f}' for v in levels])
    cbar.set_label(scale_label)
    return fig, ax, contour, cbar


def plot_rsp_func(spec: 'EvalRspResult', *, dynrange: float, **options):
    """
    The new evaluator's spectrum (draw_spectrum), 1D or 2D: plot_spectrum of |spec.grid_result|^2 on
    spec.grid_mesh.
        fig, ax, contour, cbar = plot_rsp_func(spec, dynrange=500.)
    options: the keyword arguments of plot_spectrum, e.g. reference_max, ax, axis_labels
    """
    return plot_spectrum(np.abs(spec.grid_result) ** 2, spec.grid_mesh, dynrange=dynrange, **options)
