import numpy as np
from wilson_suite.wilson_analysis.render.matplotlib_renderer import MatplotlibRenderer
from wilson_suite.wilson_main.spectrum_abstractions import EvaluationInfo, RenderingInfo
import pytest
from wilson_suite.wilson_utils.paths import SUITE_ROOT
import os

import logging
logger = logging.getLogger(__name__)

def tests_MatplotlibRenderer():

    spec_data = []
    spec_grid = {'x': np.array([]), 'y': np.array([])}
    ev_info = EvaluationInfo()
    rnd_info = RenderingInfo()

    with pytest.raises(TypeError) as error:
        MatplotlibRenderer(spec_data=spec_data, 
                                    spec_grid=spec_grid, 
                                    ev_info=ev_info, rnd_info=rnd_info, 
                                    do_diagn=True).render('f0.svg')
    assert str(error.value) == "spec_data should be a np.ndarray"

    spec_data = np.array([])
    with pytest.raises(ValueError) as error:
        MatplotlibRenderer(spec_data=spec_data, 
                                    spec_grid=spec_grid, 
                                    ev_info=ev_info, rnd_info=rnd_info, 
                                    do_diagn=True).render('f0.svg')
    assert str(error.value) == "spec_data array should not be empty"


    spec_data = np.array([[1000, 826, 826, 826, 826],
                  [826, 826, 826, 826, 826],
                  [826, 826, 826, 826, 826],
                  [826, 826, 826, 826, 826],
                  [826, 826, 826, 827,   1]])
    
    with pytest.raises(ValueError) as error:
        MatplotlibRenderer(spec_data=spec_data, 
                        spec_grid=spec_grid, 
                        ev_info=ev_info, rnd_info=rnd_info, 
                        do_diagn=True).render('f0.svg')
    assert str(error.value) == "spec_grid['x'] is an empty array"

    spec_grid = {'x': np.array([1, 2, 3]), 'y': np.array([])}
    with pytest.raises(ValueError) as error:
        MatplotlibRenderer(spec_data=spec_data, 
                        spec_grid=spec_grid, 
                        ev_info=ev_info, rnd_info=rnd_info, 
                        do_diagn=True).render('f0.svg')
    assert str(error.value) == "spec_grid['y'] is an empty array"

    spec_grid = {'x': np.array([1, 2, 3]), 'y': np.array([4, 5, 6])}
    with pytest.raises(ValueError) as error:
        MatplotlibRenderer(spec_data=spec_data, 
                        spec_grid=spec_grid, 
                        ev_info=ev_info, rnd_info=rnd_info, 
                        do_diagn=True).render('f0.svg')
    assert "X,Y and intensities data do not match in shape:" in str(error.value)
    
    start, stop, n_values = -4, 4, 9
    x_vals = np.linspace(start, stop, n_values)
    y_vals = np.linspace(start, stop, n_values)
    X, Y = np.meshgrid(x_vals, y_vals,indexing='xy')
    spec = -(X**2 + Y**2)


    spec_grid = {'x': X, 'y': Y}
    rnd_info = RenderingInfo(intensity_normalization_type=None, 
                             spec_data_operations='none')
    rnd_info.style_config.colormap_spacing = 'linear'

    with pytest.raises(ValueError) as error:
        MatplotlibRenderer(spec_data=spec, 
                    spec_grid=spec_grid, 
                    ev_info=ev_info, rnd_info=rnd_info, 
                    do_diagn=True).render('f0.svg')
    assert "Logarithmic colormap requested, but data contains no positive values" in str(error.value)


def test_render_returns():
    start, stop, n_values = -4, 4, 9
    x_vals = np.linspace(start, stop, n_values)
    y_vals = np.linspace(start, stop, n_values)
    X, Y = np.meshgrid(x_vals, y_vals,indexing='xy')
    spec = -(X**2 + Y**2)

    spec_grid = {'xlabel': X, 'y_lbl': Y}
    rnd_info = RenderingInfo(intensity_normalization_type=None, 
                             spec_data_operations='none')
    rnd_info.spec_data_operations = 'abs()**2'
    ev_info = EvaluationInfo(dynamic_range=1000)
    
    fig, ax, contour, cbar = MatplotlibRenderer(spec_data=spec, 
                                                spec_grid=spec_grid, 
                                                ev_info=ev_info, rnd_info=rnd_info, 
                                                do_diagn=True).render(SUITE_ROOT+'/wilson_suite/wilson_analysis/tests/f0.svg')
    import matplotlib
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)

    # contourf returns QuadContourSet
    assert isinstance(contour, matplotlib.contour.QuadContourSet)

    # colorbar
    assert isinstance(cbar, matplotlib.colorbar.Colorbar)

    assert ax.figure is fig
    assert ax in fig.axes

    assert ax.get_xlabel() == "xlabel"
    assert ax.get_ylabel() == "y_lbl"

    xmin, xmax = ax.get_xlim()
    assert xmin < xmax
    
    assert contour.levels is not None
    assert len(contour.levels) == 12
    assert np.all(np.diff(contour.levels) > 0) # ascending order

    from matplotlib.testing.compare import compare_images
    # returns None when images are considered the same (within tolerance)
    # returns a dict when images differ too much
    diff = compare_images(
        SUITE_ROOT+'/wilson_suite/wilson_analysis/tests/f0.svg',
        SUITE_ROOT+'/wilson_suite/wilson_analysis/tests/f_ref.svg',
        tol=2.0  # allow small numerical differences
    )
    assert diff is None, diff # if diff is not None, show diff as the error message
    os.remove(SUITE_ROOT+'/wilson_suite/wilson_analysis/tests/f0.svg')
    os.remove(SUITE_ROOT+'/wilson_suite/wilson_analysis/tests/f0_svg.png')
    os.remove(SUITE_ROOT+'/wilson_suite/wilson_analysis/tests/f_ref_svg.png')
