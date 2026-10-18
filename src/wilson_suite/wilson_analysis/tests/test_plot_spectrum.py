"""
Tests for render/plot_spectrum.py.
    plot_spectrum(intensity, grid, ...)   plots a 1D or 2D intensity array on a grid
    plot_rsp_func(spec, ...)              plots an EvalRspResult (the new evaluator's result)

The toy spectrum
    One peak at (A0, B0). Default peak height: 100 au (atomic units).
    2D: 81 points along A, 41 along B, both from -20 to 20 cm-1. 1D: along A only.
    intensity / max = 1 / (1 + ((A - A0)/3)^2) / (1 + ((B - B0)/3)^2)
    Example, along A with B = B0:
        A - A0 = 0   ->  1
        A - A0 = 3   ->  1/2
        A - A0 = 9   ->  1/10
    The formula is simple, so each test can work out its expected numbers by hand.

What plot_spectrum shows
    1. It divides the intensity by a "top" value: the max, or reference_max if given.
    2. It shows values from 1 / dynrange up to 1, on a log scale. Example: dynrange 100 shows 0.01 to 1.
    3. 1D: one line. The y axis goes from 1 / dynrange to 1.2. If the line goes above 1, the axis goes to 1.2 times
       the highest value.
    4. 2D: filled contours. Levels: nlevels values from 1 / dynrange to 1, evenly spaced on a log scale.
       Example: dynrange 100, nlevels 3 gives 0.01, 0.1, 1.
       A band is the area between two neighbouring levels.
       Above the top level 1: one extra band, coloured over_color.
       Below the lowest level: nothing drawn. The background colour (below_color) shows.

How the tests read a 2D plot
    contour = the object that matplotlib's contourf returns.
    contour.levels           the levels
    contour.get_paths()[k]   the outline of band k. An empty band has no points (vertices).
                                 [0]   the lowest band
                                 [-2]  the top band, up to 1
                                 [-1]  the band above 1
    contour.zmax             the largest plotted value
    contour.norm             turns a value into a place on the colour scale: 0 at the lowest level, 1 at the top
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from wilson_suite.wilson_analysis.render.plot_spectrum import plot_rsp_func, plot_spectrum
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import EvalRspResult


def toy_axes(a_range=(-20., 20.), b_range=(-20., 20.)) -> dict[str, np.ndarray]:
    """
    The grid points along each axis, in cm-1. A: 81 points over a_range. B: 41 points over b_range.
    Same form as EvalRspResult.grid_axes.
    """
    return {'A': np.linspace(*a_range, 81), 'B': np.linspace(*b_range, 41)}


def toy_amplitude(axes: dict[str, np.ndarray], peak=(0., 0.), amplitude=10.) -> np.ndarray:
    """
    The complex amplitude of the toy peak, at every grid point:
        amplitude / (1 - i (A - A0)/3) / (1 - i (B - B0)/3)
    |result|^2 is the toy intensity. At the peak, |result|^2 = amplitude**2.
    Array dimension 0 is A. Dimension 1 is B (2D only).
    """
    mesh = np.meshgrid(*axes.values(), indexing='ij')
    return amplitude / np.prod([1 - 1j * (m - p) / 3. for m, p in zip(mesh, peak)], axis=0)


def toy_spectrum(a_range=(-20., 20.), peak=(0., 0.), amplitude=10.) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    The 2D toy spectrum, as plot_spectrum takes it: (intensity, grid).
    intensity: shape (81, 41).
    grid: one (81, 41) array per axis, like EvalRspResult.grid_mesh.
    a_range moves the A axis. B always runs from -20 to 20.
    """
    axes = toy_axes(a_range)
    grid = dict(zip(axes, np.meshgrid(*axes.values(), indexing='ij')))
    return np.abs(toy_amplitude(axes, peak, amplitude)) ** 2, grid


def toy_spectrum_1d(peak=0., amplitude=10.) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    The 1D toy spectrum along A, as plot_spectrum takes it: (intensity, grid).
    grid: {'A': the 81 A points}.
    """
    axes = {'A': toy_axes()['A']}
    return np.abs(toy_amplitude(axes, (peak,), amplitude)) ** 2, axes


def toy_result(axes: dict[str, np.ndarray], peak=(0., 0.)) -> EvalRspResult:
    """
    An EvalRspResult with only the two fields plot_rsp_func reads: grid_axes and grid_result.
    grid_result: the toy amplitude, peak height 100. The other fields are None.
    plot_rsp_func also uses grid_mesh. EvalRspResult builds grid_mesh from grid_axes.
    """
    return EvalRspResult(feature_result=None, grid_axes=axes, grid_result=toy_amplitude(axes, peak),  # type: ignore
                         spectral_window=None)  # type: ignore


@pytest.fixture(autouse=True)
def close_figures():
    """
    Closes all figures after each test. Otherwise the figures of all tests stay open (matplotlib warns after 20).
    autouse=True: pytest runs this fixture for every test in this module.
    yield: code before yield runs before the test. Code after yield runs after the test, even if the test fails.
    """
    yield
    plt.close('all')


## plot_spectrum, 2D: levels, colours, bands --------------------------------------

@pytest.mark.parametrize('dynrange, nlevels, levels', [
    (10., 2, [0.1, 1.]),                                        # only the two ends
    (1000., 4, [0.001, 0.01, 0.1, 1.]),                         # one level per factor 10
    (100., 5, [0.01, 0.1 ** 1.5, 0.1, 0.1 ** 0.5, 1.]),         # one level per factor sqrt(10)
    (16., 5, [1. / 16., 1. / 8., 1. / 4., 1. / 2., 1.]),        # one level per factor 2
])
def test_2d_levels_are_log_spaced_fractions_of_the_max_from_1_over_dynrange_to_1(dynrange, nlevels, levels):
    """
    Levels go from 1 / dynrange to 1, evenly spaced on a log scale.
    Example: dynrange 1000, nlevels 4 gives 0.001, 0.01, 0.1, 1.
    """
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=dynrange, nlevels=nlevels)

    np.testing.assert_allclose(np.asarray(contour.levels), levels)


def test_2d_neighbouring_levels_differ_by_one_factor():
    """
    dynrange 500, 12 levels: from 1/500 = 0.002 to 1, in 11 steps.
    Each level is the one below times the same factor: 500**(1/11) = 1.76.
    """
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=500., nlevels=12)

    levels = np.asarray(contour.levels)
    assert len(levels) == 12
    assert (levels[0], levels[-1]) == (pytest.approx(0.002), pytest.approx(1.))
    np.testing.assert_allclose(levels[1:] / levels[:-1], 500. ** (1. / 11.))


def test_2d_default_is_12_levels():
    """No nlevels given: 12 levels."""
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=500.)

    assert len(contour.levels) == 12


def test_2d_colours_are_evenly_spaced_on_the_log_scale():
    """
    Each band gets the same step in colour.
    dynrange 100, 5 levels: 0.01, 0.0316, 0.1, 0.316, 1.
    The colour scale is log, from 0.01 to 1. The levels sit at 0, 1/4, 1/2, 3/4, 1 on the colour scale.
    """
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=100., nlevels=5)

    assert (contour.norm.vmin, contour.norm.vmax) == (pytest.approx(0.01), pytest.approx(1.))
    np.testing.assert_allclose(contour.norm(np.asarray(contour.levels)), [0., 0.25, 0.5, 0.75, 1.])


def test_2d_the_lowest_level_lies_where_the_intensity_is_max_over_dynrange():
    """
    The lowest level marks where intensity / max = 1 / dynrange.
    dynrange 10, 2 levels: 0.1 and 1. So the lowest band holds the area where intensity / max is above 0.1.
    Along B = 0: intensity / max = 1 / (1 + (A/3)^2) = 0.1 at A = ±9. The same for B, along A = 0.
    So the lowest band reaches from -9 to 9 in A and in B.
    """
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=10., nlevels=2)

    lowest_band = np.asarray(contour.get_paths()[0].vertices)
    np.testing.assert_allclose(lowest_band.min(axis=0), (-9., -9.), atol=0.01)
    np.testing.assert_allclose(lowest_band.max(axis=0), (9., 9.), atol=0.01)


def test_2d_below_the_lowest_level_is_the_background_colour():
    """Below the lowest level, contourf draws nothing. The axes' background colour (below_color) shows there."""
    _, ax, _, _ = plot_spectrum(*toy_spectrum(), dynrange=10., below_color='#00FF00')

    assert ax.get_facecolor() == to_rgba('#00FF00')


def test_2d_without_reference_max_nothing_is_above_the_top_level():
    """
    No reference_max: the values are intensity / max, so the highest value is exactly 1.
    The band above 1 (get_paths()[-1]) is empty.
    The top band (get_paths()[-2]) holds the peak.
    """
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=10., nlevels=2)

    assert len(contour.get_paths()[-1].vertices) == 0
    assert len(contour.get_paths()[-2].vertices) > 0


def test_2d_reference_max_below_the_peak_fills_over_color_where_the_intensity_is_above_the_reference():
    """
    Peak 100, reference_max 50. The values are intensity / 50, so the peak sits at 2.
    Values above 1 go in the band above 1 (get_paths()[-1]), coloured over_color.
    Along B = 0: intensity / 50 = 2 / (1 + (A/3)^2) = 1 at A = ±3. The same for B, along A = 0.
    So the band above 1 reaches from -3 to 3 in A and in B.
    """
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=10., nlevels=2, reference_max=50., over_color='#00FF00')

    above_top = np.asarray(contour.get_paths()[-1].vertices)
    np.testing.assert_allclose(above_top.min(axis=0), (-3., -3.), atol=0.01)
    np.testing.assert_allclose(above_top.max(axis=0), (3., 3.), atol=0.01)
    assert tuple(contour.cmap.get_over()) == to_rgba('#00FF00')


## plot_spectrum, 2D: colorbar, axes ----------------------------------------------

def test_2d_reference_max_puts_the_top_level_at_the_reference_and_intensities_above_it_saturate():
    """
    Peak 100, reference_max 50. The values are intensity / 50, so the peak sits at 2.
    The levels stay 0.01, 0.1, 1. Level 1 is the reference.
    extend='max': contourf draws values above 1 too, in over_color.
    The colorbar label shows the reference.
    """
    _, _, contour, cbar = plot_spectrum(*toy_spectrum(), dynrange=100., nlevels=3, reference_max=50.)

    np.testing.assert_allclose(np.asarray(contour.levels), [0.01, 0.1, 1.])
    assert contour.extend == 'max'
    assert cbar.ax.get_ylabel() == 'intensity / reference   (reference = 5.00e+01 au)'


def test_2d_colorbar_ticks_are_fractions_of_the_max():
    """
    One colorbar tick per level, with 2 significant digits: 0.010, 0.10, 1.0.
    The colorbar label shows the max in au.
    """
    _, _, _, cbar = plot_spectrum(*toy_spectrum(), dynrange=100., nlevels=3)

    assert [t.get_text() for t in cbar.ax.get_yticklabels()] == ['0.010', '0.10', '1.0']
    assert cbar.ax.get_ylabel() == 'intensity / max   (max = 1.00e+02 au)'


def test_2d_colorbar_ticks_of_a_large_dynrange_stay_nonzero():
    """
    dynrange 10^4: the lowest level is 0.0001. Its tick shows 0.00010.
    A fixed 3 decimals would show 0.000.
    """
    _, _, _, cbar = plot_spectrum(*toy_spectrum(), dynrange=1e4, nlevels=3)

    assert [t.get_text() for t in cbar.ax.get_yticklabels()] == ['0.00010', '0.010', '1.0']


def test_2d_first_axis_along_x_second_along_y():
    """
    The first grid axis (A) goes along x. The second (B) goes along y.
    A runs from 0 to 40, B from -20 to 20. Swapped axes would give other limits.
    The peak sits at A = 30, B = -10. The top band (get_paths()[-2]) must be centred there.
    If the data were flipped (A along y), the top band would sit elsewhere.
    """
    _, ax, contour, _ = plot_spectrum(*toy_spectrum(a_range=(0., 40.), peak=(30., -10.)), dynrange=100., nlevels=3)

    assert ax.get_xlim() == (0., 40.) and ax.get_ylim() == (-20., 20.)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ('A, cm$^{-1}$', 'B, cm$^{-1}$')
    top_band = contour.get_paths()[-2]        # between the two highest levels ([-1]: above the top, extend='max')
    np.testing.assert_allclose(np.asarray(top_band.vertices).mean(axis=0), (30., -10.), atol=1.)


def test_2d_plots_into_the_given_axes_with_the_given_labels():
    """
    ax given: plot_spectrum draws into ax and returns ax and its figure.
    axis_labels sets the label of A to 'w1'. B keeps the default label.
    """
    fig, ax = plt.subplots()

    fig_out, ax_out, _, _ = plot_spectrum(*toy_spectrum(), dynrange=100., ax=ax, axis_labels={'A': 'w1'})

    assert (fig_out, ax_out) == (fig, ax)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ('w1', 'B, cm$^{-1}$')


## plot_spectrum, 1D ------------------------------------------------------------

def test_1d_is_a_line_of_intensity_over_max_on_a_log_scale_from_1_over_dynrange():
    """
    1D: one line. x: the A points. y: intensity / max (max = 100).
    The y axis is log, from 1/500 to 1.2. The 1.2 leaves room above the peak at 1.
    The x axis spans the A points exactly.
    No colorbar: the fourth return value is None.
    """
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
    """
    Peak 100, reference_max 50. The line shows intensity / 50, so the peak sits at 2.
    The y axis goes up to 1.2 times the peak: 2.4. So the part of the line above 1 stays visible.
    """
    _, ax, line, _ = plot_spectrum(*toy_spectrum_1d(), dynrange=100., reference_max=50.)

    assert np.max(np.asarray(line.get_ydata())) == pytest.approx(2.)
    np.testing.assert_allclose(ax.get_ylim(), (0.01, 2.4))
    assert ax.get_ylabel() == 'intensity / reference   (reference = 5.00e+01 au)'


@pytest.mark.parametrize('dynrange', [10., 500.])
def test_1d_y_axis_starts_at_the_lowest_2d_level(dynrange):
    """
    The 1D y axis starts at 1 / dynrange. The lowest 2D level is also 1 / dynrange.
    So 1D and 2D plots of one spectrum show the same range of values.
    """
    _, ax, _, _ = plot_spectrum(*toy_spectrum_1d(), dynrange=dynrange)
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=dynrange)

    assert ax.get_ylim()[0] == pytest.approx(contour.levels[0])


## plot_spectrum: normalisation, intensity / top ---------------------------------

def test_2d_plots_intensity_over_max():
    """Peak 100. The plot shows intensity / 100, so the highest value is 1."""
    _, _, contour, _ = plot_spectrum(*toy_spectrum(), dynrange=10.)

    assert contour.zmax == pytest.approx(1.)


def test_2d_spectra_of_different_heights_give_the_same_plot():
    """
    Two toy peaks: height 1 and height 4096. After dividing by the max, both plots are the same.
    Only the colorbar label differs. The label shows each max.
    Why 4096 = 2^12: dividing by a power of 2 is exact. So both plots get the same values, to the last digit.
    With 10000, a value could land just above a level in one plot and just below the level in the other.
    """
    _, _, low, low_cbar = plot_spectrum(*toy_spectrum(amplitude=1.), dynrange=100., nlevels=3)
    _, _, high, high_cbar = plot_spectrum(*toy_spectrum(amplitude=64.), dynrange=100., nlevels=3)

    np.testing.assert_allclose(np.asarray(low.levels), np.asarray(high.levels))
    for low_band, high_band in zip(low.get_paths(), high.get_paths()):
        np.testing.assert_allclose(np.asarray(low_band.vertices), np.asarray(high_band.vertices))
    assert low_cbar.ax.get_ylabel() == 'intensity / max   (max = 1.00e+00 au)'
    assert high_cbar.ax.get_ylabel() == 'intensity / max   (max = 4.10e+03 au)'


def test_2d_reference_max_equal_to_the_max_gives_the_same_plot_as_no_reference_max():
    """
    reference_max 100 equals the peak. So intensity / reference equals intensity / max.
    The plot is the same as the plot without reference_max.
    """
    _, _, own, _ = plot_spectrum(*toy_spectrum(), dynrange=100., nlevels=3)
    _, _, shared, _ = plot_spectrum(*toy_spectrum(), dynrange=100., nlevels=3, reference_max=100.)

    np.testing.assert_allclose(np.asarray(own.levels), np.asarray(shared.levels))
    for own_band, shared_band in zip(own.get_paths(), shared.get_paths()):
        np.testing.assert_allclose(np.asarray(own_band.vertices), np.asarray(shared_band.vertices))


def test_2d_reference_max_keeps_a_lower_spectrum_lower():
    """
    With reference_max, a weaker spectrum must stay low. reference_max must not stretch the spectrum up to 1.
    Peak 25, reference_max 100. The highest value is 25 / 100 = 0.25.
    Levels: 0.1, 0.32, 1. The value 0.25 lies in the lowest band (0.1 to 0.32).
    The two bands above (0.32 to 1, and above 1) are empty.
    """
    _, _, contour, _ = plot_spectrum(*toy_spectrum(amplitude=5.), dynrange=10., nlevels=3, reference_max=100.)

    assert contour.zmax == pytest.approx(0.25)
    assert [len(band.vertices) > 0 for band in contour.get_paths()] == [True, False, False]


def test_1d_values_are_intensity_over_max():
    """
    The line shows intensity / max = 1 / (1 + (A/3)^2).
    The test reads the line at 3 points and compares with the formula:
        A = 0  ->  1
        A = 3  ->  1/2
        A = 9  ->  1/10
    """
    _, _, line, _ = plot_spectrum(*toy_spectrum_1d(), dynrange=100.)

    a, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    np.testing.assert_allclose(np.interp([0., 3., 9.], a, y), [1., 0.5, 0.1])


def test_1d_spectra_of_different_heights_give_the_same_line():
    """
    Two toy peaks: height 1 and height 10000. After dividing by the max, both lines are the same.
    Only the y label differs. The label shows each max.
    """
    _, low_ax, low, _ = plot_spectrum(*toy_spectrum_1d(amplitude=1.), dynrange=100.)
    _, high_ax, high, _ = plot_spectrum(*toy_spectrum_1d(amplitude=100.), dynrange=100.)

    np.testing.assert_allclose(np.asarray(low.get_ydata()), np.asarray(high.get_ydata()))
    assert low_ax.get_ylabel() == 'intensity / max   (max = 1.00e+00 au)'
    assert high_ax.get_ylabel() == 'intensity / max   (max = 1.00e+04 au)'


def test_1d_values_are_intensity_over_reference_max():
    """
    Peak 100, reference_max 50. The line shows intensity / 50 = 2 / (1 + (A/3)^2).
    The test reads the line at 3 points and compares with the formula:
        A = 0  ->  2
        A = 3  ->  1
        A = 9  ->  1/5
    """
    _, _, line, _ = plot_spectrum(*toy_spectrum_1d(), dynrange=100., reference_max=50.)

    a, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    np.testing.assert_allclose(np.interp([0., 3., 9.], a, y), [2., 1., 0.2])


def test_1d_reference_max_keeps_a_lower_spectrum_lower():
    """
    Peak 25, reference_max 100. The line tops out at 25 / 100 = 0.25.
    The y axis still goes up to 1.2, the same as for a spectrum that reaches the reference.
    """
    _, ax, line, _ = plot_spectrum(*toy_spectrum_1d(amplitude=5.), dynrange=10., reference_max=100.)

    assert np.max(np.asarray(line.get_ydata())) == pytest.approx(0.25)
    np.testing.assert_allclose(ax.get_ylim(), (0.1, 1.2))


## plot_spectrum: input checks ----------------------------------------------------

def test_raises_for_complex_intensity():
    """
    Complex input (the amplitude, not |amplitude|^2) raises TypeError.
    For amplitudes, use plot_rsp_func.
    """
    _, grid = toy_spectrum()

    with pytest.raises(TypeError, match='intensity must be real'):
        plot_spectrum(toy_amplitude(toy_axes()), grid, dynrange=100.)


def test_raises_for_3d():
    """3D intensity raises ValueError. plot_spectrum plots 1D and 2D only."""
    cube = np.ones((2, 3, 4))
    grid = dict(zip('ABC', np.meshgrid(np.arange(2.), np.arange(3.), np.arange(4.), indexing='ij')))

    with pytest.raises(ValueError, match=r'plots 1D and 2D spectra.*shape \(2, 3, 4\)'):
        plot_spectrum(cube, grid, dynrange=100.)


def test_raises_for_one_grid_axis_with_2d_intensity():
    """2D intensity with only one grid axis (A) raises ValueError. 2D needs two grid axes."""
    intensity, grid = toy_spectrum()

    with pytest.raises(ValueError, match=r"one grid axis per dimension: got intensity of shape \(81, 41\) and the "
                                         r"axes \['A'\]"):
        plot_spectrum(intensity, {'A': grid['A']}, dynrange=100.)


def test_raises_for_grid_axes_that_are_not_full_arrays():
    """
    The grid must hold one full array per axis, of the intensity's shape (81, 41), like grid_mesh.
    Here the grid holds the 1D points instead, shapes (81,) and (41,), like grid_axes. This raises ValueError.
    """
    intensity, _ = toy_spectrum()

    with pytest.raises(ValueError, match=r"grid\['A'\] has the shape \(81,\), intensity \(81, 41\)"):
        plot_spectrum(intensity, toy_axes(), dynrange=100.)


@pytest.mark.parametrize('dynrange', [1., 0.5])
def test_raises_for_a_dynrange_of_1_or_less(dynrange):
    """
    dynrange must be above 1. The lowest level is 1 / dynrange.
    dynrange 1 puts the lowest level at the top (1). dynrange 0.5 puts the lowest level above the top (2).
    """
    with pytest.raises(ValueError, match='dynrange must be > 1'):
        plot_spectrum(*toy_spectrum(), dynrange=dynrange)


@pytest.mark.parametrize('spectrum', [toy_spectrum, toy_spectrum_1d], ids=['2D', '1D'])
def test_raises_when_no_intensity_is_above_0(spectrum):
    """All intensities 0: the max is 0. plot_spectrum cannot divide by 0, so it raises ValueError."""
    with pytest.raises(ValueError, match='no intensity above 0'):
        plot_spectrum(*spectrum(amplitude=0.), dynrange=100.)


## plot_rsp_func --------------------------------------------------------------

def test_plot_rsp_func_plots_the_intensity_of_the_amplitudes_on_the_grid_mesh():
    """
    plot_rsp_func(spec) must give the same plot as plot_spectrum(|grid_result|^2, grid_mesh).
    The test compares the levels, the top band (get_paths()[-2]) and the axis labels.
    The peak sits off centre (A = 10, B = -5). So swapped axes would move the top band, and the test would fail.
    """
    spec = toy_result(toy_axes(), peak=(10., -5.))

    _, ax, contour, _ = plot_rsp_func(spec, dynrange=100., nlevels=3)
    _, ax_direct, direct, _ = plot_spectrum(np.abs(spec.grid_result) ** 2, spec.grid_mesh, dynrange=100., nlevels=3)

    np.testing.assert_allclose(np.asarray(contour.levels), np.asarray(direct.levels))
    np.testing.assert_allclose(np.asarray(contour.get_paths()[-2].vertices),
                               np.asarray(direct.get_paths()[-2].vertices))
    assert (ax.get_xlabel(), ax.get_ylabel()) == (ax_direct.get_xlabel(), ax_direct.get_ylabel())


def test_plot_rsp_func_plots_a_1d_result():
    """A result with one axis (A). The line must show |grid_result|^2 / max. No colorbar."""
    spec = toy_result({'A': toy_axes()['A']}, peak=(5.,))

    _, _, line, colorbar = plot_rsp_func(spec, dynrange=100.)

    intensity = np.abs(spec.grid_result) ** 2
    np.testing.assert_allclose(np.asarray(line.get_ydata()), intensity / intensity.max())
    assert colorbar is None


def test_plot_rsp_func_passes_the_options_on():
    """
    plot_rsp_func passes keyword options on to plot_spectrum. The test checks 3 of them:
        ax              the plot lands in the given ax
        nlevels 3       the levels are 0.01, 0.1, 1
        reference_max   the colorbar label shows the reference, 50
    """
    _, ax = plt.subplots()

    _, ax_out, contour, cbar = plot_rsp_func(toy_result(toy_axes()), dynrange=100., nlevels=3, reference_max=50., ax=ax)

    assert ax_out is ax
    np.testing.assert_allclose(np.asarray(contour.levels), [0.01, 0.1, 1.])
    assert cbar.ax.get_ylabel() == 'intensity / reference   (reference = 5.00e+01 au)'
