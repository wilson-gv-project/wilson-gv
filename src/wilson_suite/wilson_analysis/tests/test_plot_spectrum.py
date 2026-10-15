"""
render/plot_spectrum.py: plot_spectrum (any 1D or 2D intensity array on a grid) and plot_rsp_func (the new
evaluator's result, EvalRspResult). The toy spectrum: one Lorentzian peak, width 3 cm-1, on 81 (A) x 41 (B) points;
1D: along A only.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from wilson_suite.wilson_analysis.render.plot_spectrum import plot_rsp_func, plot_spectrum
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import EvalRspResult


def toy_axes(a_range=(-20., 20.), b_range=(-20., 20.)) -> dict[str, np.ndarray]:
    return {'A': np.linspace(*a_range, 81), 'B': np.linspace(*b_range, 41)}


def toy_amplitude(axes: dict[str, np.ndarray], peak=(0., 0.), amplitude=10.) -> np.ndarray:
    """|amplitude|^2 = amplitude**2 at `peak`; array dimension 0 is A, 1 is B (if there)"""
    mesh = np.meshgrid(*axes.values(), indexing='ij')
    return amplitude / np.prod([1 - 1j * (m - p) / 3. for m, p in zip(mesh, peak)], axis=0)


def toy_spectrum(a_range=(-20., 20.), peak=(0., 0.), amplitude=10.) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(intensity, grid) as plot_spectrum takes them, 2D"""
    axes = toy_axes(a_range)
    grid = dict(zip(axes, np.meshgrid(*axes.values(), indexing='ij')))
    return np.abs(toy_amplitude(axes, peak, amplitude)) ** 2, grid


def toy_spectrum_1d(peak=0., amplitude=10.) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(intensity, grid) as plot_spectrum takes them, 1D along A"""
    axes = {'A': toy_axes()['A']}
    return np.abs(toy_amplitude(axes, (peak,), amplitude)) ** 2, axes


def toy_result(axes: dict[str, np.ndarray], peak=(0., 0.)) -> EvalRspResult:
    """only the grid fields plot_rsp_func reads: grid_axes, grid_result (and grid_mesh from them)"""
    return EvalRspResult(feature_result=None, grid_axes=axes, grid_result=toy_amplitude(axes, peak),  # type: ignore
                         spectral_window=None)  # type: ignore


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


## plot_spectrum, 2D ------------------------------------------------------------

def test_2d_levels_are_fractions_of_the_max_log_spaced_from_1_over_dynrange_to_1():
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=500., nlevels=12)

    np.testing.assert_allclose(np.asarray(contour.levels), np.logspace(np.log10(1. / 500.), 0., 12))


def test_2d_reference_max_puts_the_top_level_at_the_reference_and_intensities_above_it_saturate():
    """intensity / 50: the peak (100) is at 2, above the top level 1"""
    _, _, contour, cbar = plot_spectrum(*toy_spectrum(), dynrange=100., nlevels=3, reference_max=50.)

    np.testing.assert_allclose(np.asarray(contour.levels), [0.01, 0.1, 1.])
    assert contour.extend == 'max'
    assert cbar.ax.get_ylabel() == 'intensity / reference   (reference = 5.00e+01 au)'


def test_2d_colorbar_ticks_are_fractions_of_the_max():
    _, _, _, cbar = plot_spectrum(*toy_spectrum(), dynrange=100., nlevels=3)

    assert [t.get_text() for t in cbar.ax.get_yticklabels()] == ['0.010', '0.100', '1.000']
    assert cbar.ax.get_ylabel() == 'intensity / max   (max = 1.00e+02 au)'


def test_2d_first_axis_along_x_second_along_y():
    """A from 0 to 40, B from -20 to 20, peak at A = 30, B = -10: the top band sits there"""
    _, ax, contour, _ = plot_spectrum(*toy_spectrum(a_range=(0., 40.), peak=(30., -10.)), dynrange=100., nlevels=3)

    assert ax.get_xlim() == (0., 40.) and ax.get_ylim() == (-20., 20.)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ('A, cm$^{-1}$', 'B, cm$^{-1}$')
    top_band = contour.get_paths()[-2]        # between the two highest levels ([-1]: above the top, extend='max')
    np.testing.assert_allclose(np.asarray(top_band.vertices).mean(axis=0), (30., -10.), atol=1.)


def test_2d_draws_into_the_given_axes_with_the_given_labels():
    fig, ax = plt.subplots()

    fig_out, ax_out, _, _ = plot_spectrum(*toy_spectrum(), dynrange=100., ax=ax, axis_labels={'A': 'w1'})

    assert (fig_out, ax_out) == (fig, ax)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ('w1', 'B, cm$^{-1}$')


## plot_spectrum, 1D ------------------------------------------------------------

def test_1d_is_a_line_of_intensity_over_max_on_a_log_scale_from_1_over_dynrange():
    intensity, grid = toy_spectrum_1d()

    _, ax, line, colorbar = plot_spectrum(intensity, grid, dynrange=500.)

    np.testing.assert_allclose(np.asarray(line.get_xdata()), grid['A'])
    np.testing.assert_allclose(np.asarray(line.get_ydata()), intensity / 100.)
    assert ax.get_yscale() == 'log'
    np.testing.assert_allclose(ax.get_ylim(), (1. / 500., 1.2))
    assert ax.get_xlim() == (-20., 20.)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ('A, cm$^{-1}$', 'intensity / max   (max = 1.00e+02 au)')
    assert colorbar is None


def test_1d_reference_max_shows_the_intensities_above_it():
    """intensity / 50: the peak at 2; the y axis reaches above it"""
    _, ax, line, _ = plot_spectrum(*toy_spectrum_1d(), dynrange=100., reference_max=50.)

    assert np.max(np.asarray(line.get_ydata())) == pytest.approx(2.)
    np.testing.assert_allclose(ax.get_ylim(), (0.01, 2.4))
    assert ax.get_ylabel() == 'intensity / reference   (reference = 5.00e+01 au)'


## plot_spectrum: input checks ----------------------------------------------------

def test_raises_for_complex_intensity():
    _, grid = toy_spectrum()

    with pytest.raises(TypeError, match='intensity must be real'):
        plot_spectrum(toy_amplitude(toy_axes()), grid, dynrange=100.)


def test_raises_for_3d():
    cube = np.ones((2, 3, 4))
    grid = dict(zip('ABC', np.meshgrid(np.arange(2.), np.arange(3.), np.arange(4.), indexing='ij')))

    with pytest.raises(ValueError, match=r'draws 1D and 2D spectra.*shape \(2, 3, 4\)'):
        plot_spectrum(cube, grid, dynrange=100.)


def test_raises_for_one_grid_axis_with_2d_intensity():
    intensity, grid = toy_spectrum()

    with pytest.raises(ValueError, match=r"one grid axis per dimension: got intensity of shape \(81, 41\) and the "
                                         r"axes \['A'\]"):
        plot_spectrum(intensity, {'A': grid['A']}, dynrange=100.)


def test_raises_for_grid_axes_that_are_not_full_arrays():
    """1D points per axis (grid_axes) instead of the full arrays (grid_mesh)"""
    intensity, _ = toy_spectrum()

    with pytest.raises(ValueError, match=r"grid\['A'\] has the shape \(81,\), intensity \(81, 41\)"):
        plot_spectrum(intensity, toy_axes(), dynrange=100.)


@pytest.mark.parametrize('dynrange', [1., 0.5])
def test_raises_for_a_dynrange_of_1_or_less(dynrange):
    with pytest.raises(ValueError, match='dynrange must be > 1'):
        plot_spectrum(*toy_spectrum(), dynrange=dynrange)


@pytest.mark.parametrize('spectrum', [toy_spectrum, toy_spectrum_1d], ids=['2D', '1D'])
def test_raises_when_no_intensity_is_above_0(spectrum):
    with pytest.raises(ValueError, match='no intensity above 0'):
        plot_spectrum(*spectrum(amplitude=0.), dynrange=100.)


## plot_rsp_func --------------------------------------------------------------

def test_plot_rsp_func_plots_the_intensity_of_the_amplitudes_on_the_grid_mesh():
    """the same plot as plot_spectrum of |grid_result|^2 on grid_mesh"""
    spec = toy_result(toy_axes(), peak=(10., -5.))

    _, ax, contour, _ = plot_rsp_func(spec, dynrange=100., nlevels=3)
    _, ax_direct, direct, _ = plot_spectrum(np.abs(spec.grid_result) ** 2, spec.grid_mesh, dynrange=100., nlevels=3)

    np.testing.assert_allclose(np.asarray(contour.levels), np.asarray(direct.levels))
    np.testing.assert_allclose(np.asarray(contour.get_paths()[-2].vertices),
                               np.asarray(direct.get_paths()[-2].vertices))
    assert (ax.get_xlabel(), ax.get_ylabel()) == (ax_direct.get_xlabel(), ax_direct.get_ylabel())


def test_plot_rsp_func_plots_a_1d_result():
    """one axis: grid_mesh holds the points themselves"""
    spec = toy_result({'A': toy_axes()['A']}, peak=(5.,))

    _, _, line, colorbar = plot_rsp_func(spec, dynrange=100.)

    intensity = np.abs(spec.grid_result) ** 2
    np.testing.assert_allclose(np.asarray(line.get_ydata()), intensity / intensity.max())
    assert colorbar is None


def test_plot_rsp_func_passes_the_options_on():
    _, ax = plt.subplots()

    _, ax_out, contour, _ = plot_rsp_func(toy_result(toy_axes()), dynrange=100., nlevels=3, reference_max=50., ax=ax)

    assert ax_out is ax
    np.testing.assert_allclose(np.asarray(contour.levels), [0.01, 0.1, 1.])
