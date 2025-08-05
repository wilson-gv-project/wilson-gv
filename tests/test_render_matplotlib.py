"""
MatplotlibRenderer unit tests

Variables and objects outside of a test function body
are fixtures used in tests
"""
from wilson_analysis.render.matplotlib_renderer import MatplotlibRenderer
from wilson_main import abstractions as wm_abst
import numpy as np
import matplotlib.pyplot as plt
import pytest

axis1 = wm_abst.SpectralAxis({'w1': 1})
axis2 = wm_abst.SpectralAxis({'w1': 1, 'w2': -1}) 

start = {'x': 1, 'y': 4}
end = {'x': 4, 'y': 7}
spacer = {'x': 1., 'y': 1.}

spec_grid = wm_abst.SpectralGrid({'x': axis1, 'y': axis2}, range_style='uniform',
                                 start=start, end=end, spacer=spacer)

start = {'w1': 1, 'w2': 4}
end = {'w1': 4, 'w2': 7}
spacer = {'w1': 1., 'w2': 1.}

eval_vars = {'w1': wm_abst.EvaluationVariable(range_style='uniform', start=start['w1'], end=end['w1'], spacer=spacer['w1']).range,
             'w2': wm_abst.EvaluationVariable(range_style='uniform', start=start['w2'], end=end['w2'], spacer=spacer['w2']).range}
meshgrids = np.meshgrid(*eval_vars.values(), indexing='ij')

eval_vars_meshgrids = {}
for i, key in enumerate(eval_vars.keys()):
    eval_vars_meshgrids[key] = meshgrids[i]


def test_eval_vars_meshgrids():
    ref_eval_vars_meshgrids = {'w1': np.array([[1., 1., 1.],
                                               [2., 2., 2.],
                                               [3., 3., 3.]]), 
                            'w2': np.array([[4., 5., 6.],
                                            [4., 5., 6.],
                                            [4., 5., 6.]])}
    for key in ref_eval_vars_meshgrids:
        assert key in eval_vars_meshgrids, f"Key {key} is missing in eval_vars_meshgrids"
        assert np.array_equal(ref_eval_vars_meshgrids[key], eval_vars_meshgrids[key]), \
            f"Arrays for key {key} do not match"
    

evi = wm_abst.EvaluationInfo(**{'freq_variables': eval_vars_meshgrids,
                                'Gamma': 4.7, 'Gamma_unit': 'cm-1'})
rndi = wm_abst.RenderingInfo(**{'dynrange': 500, 'num_level_ticks': 15, 'maxPeak': None,
                                'projection': '2d', 'to_save': True, 
                                'title': "Test Title"})
eval_setup = wm_abst.SpecEvalSetup(grid=spec_grid, ev_info=evi, rnd_info=rndi)

intensities=np.array([[1, 2], [11, 21]])
renderer_matplt = MatplotlibRenderer(intensities=intensities, spec_grid=spec_grid, rnd_info=rndi, ev_info=evi)


def test_MatplotlibRenderer_prepare_axes_data():
    axes_data = renderer_matplt.prepare_axes_data()
    xref = np.array([[1., 1., 1.],
                     [2., 2., 2.],
                     [3., 3., 3.]])
    yref = np.array([[-3., -4., -5.],
                     [-2., -3., -4.],
                     [-1., -2., -3.]])
    assert np.array_equal(axes_data['x'], xref)
    assert np.array_equal(axes_data['y'], yref)


def test_MatplotlibRenderer_prepare_contour_levels():
    renderer_matplt.prepare_contour_levels()
    ref_tick_values = np.array([ 0.042 ,  0.0655,  0.1021,  0.1591,  0.248 ,  0.3865,  
                                0.6025,  0.9391,  1.4639,  2.2819,  3.557 ,  5.5446,  
                                8.6428, 13.4721, 21.    ])
    assert np.allclose(renderer_matplt.tick_values, ref_tick_values, atol=0.001)
    ref_tick_labels = ['$4.2\\times 10^{-02$}$', '$6.5\\times 10^{-02$}$', '$1.0\\times 10^{-01$}$', 
                       '$1.6\\times 10^{-01$}$', '$2.5\\times 10^{-01$}$', '$3.9\\times 10^{-01$}$', 
                       '$6.0\\times 10^{-01$}$', '$9.4\\times 10^{-01$}$', '$1.5\\times 10^{+00$}$', 
                       '$2.3\\times 10^{+00$}$', '$3.6\\times 10^{+00$}$', '$5.5\\times 10^{+00$}$', 
                       '$8.6\\times 10^{+00$}$', '$1.3\\times 10^{+01$}$', '$2.1\\times 10^{+01$}$']
    assert renderer_matplt.tick_labels == ref_tick_labels

    ref_pos = np.array([0. , 0.0714, 0.1428, 0.2143, 0.2857, 0.3571, 0.4286, 0.5, 0.5714, 
                        0.6428, 0.7143, 0.7857, 0.8571, 0.9286 , 1. ])
    assert np.allclose(renderer_matplt.tick_norm_positions, ref_pos, atol=0.0001)

def test_plot_contours_missing_keys():
    renderer_matplt.prepare_contour_levels()
    fig, ax = plt.subplots()
    axes_dict = {'z': np.array([[1, 2], [3, 4]])}
    with pytest.raises(AssertionError, match="Did not recieve `x` and `y` axes"):
        renderer_matplt.plot_contours(fig, ax, axes_dict)

def test_plot_contours_more_keys():
    renderer_matplt.prepare_contour_levels()
    fig, ax = plt.subplots()
    axes_dict = {'z': np.array([[1, 2], [3, 4]]), 
                 'x': np.array([[1, 2], [3, 4]]), 
                 'y': np.array([[1, 2], [3, 4]])}
    with pytest.raises(AssertionError, match="Recieved more than 2 axes for plot. Contour plot needs `x` and `y` axes"):
        renderer_matplt.plot_contours(fig, ax, axes_dict)

def test_plot_contours():
    renderer_matplt.prepare_contour_levels()
    fig, ax = plt.subplots()
    axes_dict = {'x': np.array([[1, 2], [3, 4]]),
                 'y': np.array([[1, 2], [3, 4]])}
    renderer_matplt.plot_contours(fig, ax, axes_dict)
    
    assert ax.get_xlabel() == r'$\omega_1/2\pi c, \text{cm}^{-1}$'
    assert ax.get_ylabel() == r'$(\omega_2 - \omega_1)/2\pi c, \text{cm}^{-1}$'
    assert ax.get_title() == "Test Title"
    assert len(fig.axes) == 2

def test_MatplotlibRenderer_finalize():
    renderer_matplt.prepare_contour_levels()
    fig, ax = plt.subplots()
    axes_dict = {'x': np.array([[1, 2], [3, 4]]),
                 'y': np.array([[1, 2], [3, 4]])}
    renderer_matplt.plot_contours(fig, ax, axes_dict)
    # colorbar is the second axis in the figure
    cbar = fig.axes[1]
    assert np.allclose(cbar.get_yticks().tolist(), renderer_matplt.tick_values)
    assert [label.get_text() for label in cbar.get_yticklabels()] == [f"{tick:.2f}" for tick in renderer_matplt.tick_values]