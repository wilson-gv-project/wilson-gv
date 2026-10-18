"""
render/plot_spectrum.py: plot_spectrum (any 1D or 2D intensity array on a grid) and plot_rsp_func (the new
evaluator's result, EvalRspResult). The toy spectrum: one Lorentzian peak, width 3 cm-1, on 81 (A) x 41 (B) points;
1D: along A only.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

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

# autouse=True fixture: runs for every test in this module
# yield: This splits the fixture into setup and teardown phases. 
#   Any code before yield runs before the test, 
#   and any code after yield runs immediately after the test finishes (even if the test fails).
@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all') # closes all currently open figure windows


## plot_spectrum, 2D: levels ----------------------------------------------------

@pytest.mark.parametrize('dynrange, nlevels, levels', [
    (10., 2, [0.1, 1.]),                                        # only the two ends
    (1000., 4, [0.001, 0.01, 0.1, 1.]),                         # one level per factor 10
    (100., 5, [0.01, 0.1 ** 1.5, 0.1, 0.1 ** 0.5, 1.]),         # one level per factor sqrt(10)
    (16., 5, [1. / 16., 1. / 8., 1. / 4., 1. / 2., 1.]),        # one level per factor 2
])
def test_2d_levels_are_log_spaced_fractions_of_the_max_from_1_over_dynrange_to_1(dynrange, nlevels, levels):
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=dynrange, nlevels=nlevels)

    np.testing.assert_allclose(np.asarray(contour.levels), levels)


def test_2d_neighbouring_levels_differ_by_one_factor():
    """dynrange 500 over 11 steps: each level is 500**(1/11) = 1.76 times the one below"""
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=500., nlevels=12)

    levels = np.asarray(contour.levels)
    assert len(levels) == 12
    assert (levels[0], levels[-1]) == (pytest.approx(0.002), pytest.approx(1.))
    np.testing.assert_allclose(levels[1:] / levels[:-1], 500. ** (1. / 11.))


def test_2d_default_is_12_levels():
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=500.)

    assert len(contour.levels) == 12


def test_2d_colours_are_evenly_spaced_on_the_log_scale():
    """levels 0.01, 0.0316, 0.1, 0.316, 1: the colour scale goes 0, 1/4, 1/2, 3/4, 1"""
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=100., nlevels=5)

    assert (contour.norm.vmin, contour.norm.vmax) == (pytest.approx(0.01), pytest.approx(1.))
    np.testing.assert_allclose(contour.norm(np.asarray(contour.levels)), [0., 0.25, 0.5, 0.75, 1.])


def test_2d_the_lowest_level_lies_where_the_intensity_is_max_over_dynrange():
    """
    dynrange 10, levels 0.1 and 1. Intensity / max = 1 / ((1 + (A/3)^2) (1 + (B/3)^2)); along B = 0 it is 0.1 at
    A = ±9 (1 / (1 + 3^2) = 1/10), along A = 0 at B = ±9. The filled area reaches out to ±9 in both directions.
    """
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=10., nlevels=2)

    lowest_band = np.asarray(contour.get_paths()[0].vertices)
    np.testing.assert_allclose(lowest_band.min(axis=0), (-9., -9.), atol=0.01)
    np.testing.assert_allclose(lowest_band.max(axis=0), (9., 9.), atol=0.01)


def test_2d_below_the_lowest_level_is_the_background_colour():
    _, ax, _, _ = plot_spectrum(*toy_spectrum(), dynrange=10., below_color='#00FF00')

    assert ax.get_facecolor() == to_rgba('#00FF00')


def test_2d_without_reference_max_nothing_is_above_the_top_level():
    """the max is the top level 1 itself: it sits in the top band; the band above 1 (over_color) is empty"""
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=10., nlevels=2)

    assert len(contour.get_paths()[-1].vertices) == 0
    assert len(contour.get_paths()[-2].vertices) > 0


def test_2d_reference_max_below_the_peak_fills_over_color_where_the_intensity_is_above_the_reference():
    """
    peak 100, reference 50: intensity / reference = 2 / ((1 + (A/3)^2) (1 + (B/3)^2)) is above 1 out to A = ±3 along
    B = 0 (2 / (1 + 1^2) = 1), and out to B = ±3 along A = 0
    """
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=10., nlevels=2, reference_max=50., over_color='#00FF00')

    above_top = np.asarray(contour.get_paths()[-1].vertices)
    np.testing.assert_allclose(above_top.min(axis=0), (-3., -3.), atol=0.01)
    np.testing.assert_allclose(above_top.max(axis=0), (3., 3.), atol=0.01)
    assert tuple(contour.cmap.get_over()) == to_rgba('#00FF00')


## plot_spectrum, 2D ------------------------------------------------------------

def test_2d_reference_max_puts_the_top_level_at_the_reference_and_intensities_above_it_saturate():
    """intensity / 50: the peak (100) is at 2, above the top level 1"""
    _, _, contour, cbar = plot_spectrum(*toy_spectrum(), dynrange=100., nlevels=3, reference_max=50.)

    np.testing.assert_allclose(np.asarray(contour.levels), [0.01, 0.1, 1.])
    assert contour.extend == 'max'
    assert cbar.ax.get_ylabel() == 'intensity / reference   (reference = 5.00e+01 au)'


def test_2d_colorbar_ticks_are_fractions_of_the_max():
    _, _, _, cbar = plot_spectrum(*toy_spectrum(), dynrange=100., nlevels=3)

    assert [t.get_text() for t in cbar.ax.get_yticklabels()] == ['0.010', '0.10', '1.0']
    assert cbar.ax.get_ylabel() == 'intensity / max   (max = 1.00e+02 au)'


def test_2d_colorbar_ticks_of_a_large_dynrange_stay_nonzero():
    """dynrange 10^4: the lowest tick is 0.0001 (a fixed 3 decimals would print 0.000)"""
    _, _, _, cbar = plot_spectrum(*toy_spectrum(), dynrange=1e4, nlevels=3)

    assert [t.get_text() for t in cbar.ax.get_yticklabels()] == ['0.00010', '0.010', '1.0']


def test_2d_first_axis_along_x_second_along_y():
    """A from 0 to 40, B from -20 to 20, peak at A = 30, B = -10: the top band sits there"""
    _, ax, contour, _ = plot_spectrum(*toy_spectrum(a_range=(0., 40.), peak=(30., -10.)), dynrange=100., nlevels=3)

    assert ax.get_xlim() == (0., 40.) and ax.get_ylim() == (-20., 20.)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ('A, cm$^{-1}$', 'B, cm$^{-1}$')
    top_band = contour.get_paths()[-2]        # between the two highest levels ([-1]: above the top, extend='max')
    np.testing.assert_allclose(np.asarray(top_band.vertices).mean(axis=0), (30., -10.), atol=1.)


def test_2d_plots_into_the_given_axes_with_the_given_labels():
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


@pytest.mark.parametrize('dynrange', [10., 500.])
def test_1d_y_axis_starts_at_the_lowest_2d_level(dynrange):
    _, ax, _, _ = plot_spectrum(*toy_spectrum_1d(), dynrange=dynrange)
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=dynrange)

    assert ax.get_ylim()[0] == pytest.approx(contour.levels[0])


## plot_spectrum: normalisation, intensity / top ---------------------------------

def test_2d_plots_intensity_over_max():
    """peak 100: the plotted values reach 1 at the peak"""
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=10.)

    assert contour.zmax == pytest.approx(1.)


def test_2d_spectra_of_different_heights_give_the_same_plot():
    """
    peaks 1 and 4096: the same levels and bands; only the colorbar label shows the max. 4096 = 2^12: intensity / max
    is then the same to the last digit (with 10000 a point can land just above a level in one plot and just below it
    in the other, and the band outlines get a different number of points)
    """
    _, _, low, low_cbar = plot_spectrum(*toy_spectrum(amplitude=1.), dynrange=100., nlevels=3)
    _, _, high, high_cbar = plot_spectrum(*toy_spectrum(amplitude=64.), dynrange=100., nlevels=3)

    np.testing.assert_allclose(np.asarray(low.levels), np.asarray(high.levels))
    for low_band, high_band in zip(low.get_paths(), high.get_paths()):
        np.testing.assert_allclose(np.asarray(low_band.vertices), np.asarray(high_band.vertices))
    assert low_cbar.ax.get_ylabel() == 'intensity / max   (max = 1.00e+00 au)'
    assert high_cbar.ax.get_ylabel() == 'intensity / max   (max = 4.10e+03 au)'


def test_2d_reference_max_equal_to_the_max_gives_the_same_plot_as_no_reference_max():
    _, _, own, _ = plot_spectrum(*toy_spectrum(), dynrange=100., nlevels=3)
    _, _, shared, _ = plot_spectrum(*toy_spectrum(), dynrange=100., nlevels=3, reference_max=100.)

    np.testing.assert_allclose(np.asarray(own.levels), np.asarray(shared.levels))
    for own_band, shared_band in zip(own.get_paths(), shared.get_paths()):
        np.testing.assert_allclose(np.asarray(own_band.vertices), np.asarray(shared_band.vertices))


def test_2d_reference_max_keeps_a_lower_spectrum_lower():
    """
    peak 25, reference 100: the peak is at 0.25, inside the lowest band (0.1 to 0.32) of the levels 0.1, 0.32, 1;
    the band from 0.32 to 1 and the band above 1 are empty
    """
    _, _, contour, _ = plot_spectrum(*toy_spectrum(amplitude=5.), dynrange=10., nlevels=3, reference_max=100.)

    assert contour.zmax == pytest.approx(0.25)
    assert [len(band.vertices) > 0 for band in contour.get_paths()] == [True, False, False]


def test_1d_values_are_intensity_over_max():
    """intensity / max = 1 / (1 + (A/3)^2): 1 at A = 0, 1/2 at A = 3 (the half width), 1/10 at A = 9"""
    _, _, line, _ = plot_spectrum(*toy_spectrum_1d(), dynrange=100.)

    a, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    np.testing.assert_allclose(np.interp([0., 3., 9.], a, y), [1., 0.5, 0.1])


def test_1d_spectra_of_different_heights_give_the_same_line():
    """peaks 1 and 10000: the same values; only the y label shows the max"""
    _, low_ax, low, _ = plot_spectrum(*toy_spectrum_1d(amplitude=1.), dynrange=100.)
    _, high_ax, high, _ = plot_spectrum(*toy_spectrum_1d(amplitude=100.), dynrange=100.)

    np.testing.assert_allclose(np.asarray(low.get_ydata()), np.asarray(high.get_ydata()))
    assert low_ax.get_ylabel() == 'intensity / max   (max = 1.00e+00 au)'
    assert high_ax.get_ylabel() == 'intensity / max   (max = 1.00e+04 au)'


def test_1d_values_are_intensity_over_reference_max():
    """peak 100, reference 50: intensity / reference = 2 / (1 + (A/3)^2): 2 at A = 0, 1 at A = 3, 1/5 at A = 9"""
    _, _, line, _ = plot_spectrum(*toy_spectrum_1d(), dynrange=100., reference_max=50.)

    a, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    np.testing.assert_allclose(np.interp([0., 3., 9.], a, y), [2., 1., 0.2])


def test_1d_reference_max_keeps_a_lower_spectrum_lower():
    """peak 25, reference 100: the line tops out at 0.25; the y axis still reaches 1.2 (1.2 times the reference)"""
    _, ax, line, _ = plot_spectrum(*toy_spectrum_1d(amplitude=5.), dynrange=10., reference_max=100.)

    assert np.max(np.asarray(line.get_ydata())) == pytest.approx(0.25)
    np.testing.assert_allclose(ax.get_ylim(), (0.1, 1.2))


## plot_spectrum: input checks ----------------------------------------------------

def test_raises_for_complex_intensity():
    _, grid = toy_spectrum()

    with pytest.raises(TypeError, match='intensity must be real'):
        plot_spectrum(toy_amplitude(toy_axes()), grid, dynrange=100.)


def test_raises_for_3d():
    cube = np.ones((2, 3, 4))
    grid = dict(zip('ABC', np.meshgrid(np.arange(2.), np.arange(3.), np.arange(4.), indexing='ij')))

    with pytest.raises(ValueError, match=r'plots 1D and 2D spectra.*shape \(2, 3, 4\)'):
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
