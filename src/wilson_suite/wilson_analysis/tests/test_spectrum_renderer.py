from dataclasses import dataclass
from wilson_suite.wilson_analysis.render.spectrum_renderer import SpectrumRenderer, LevelCalculator, compute_masks
from wilson_suite.wilson_analysis.render.render_utils import NormalizationType, PlotConfig
import pytest
import numpy as np

# mock subclass that implements the abstract methods
class ConcreteSpectrumRenderer(SpectrumRenderer):
    def add_colorbar(self):
        pass
    def create_contour(self):
        pass
    def finalize():
        pass
    def initialize_plot():
        pass
    def save_plot():
        pass
    def setup_axes():
        pass

class IncompleteRenderer(SpectrumRenderer):
    pass

@dataclass
class MockRndInfo:
    style_config: str = ''
    spec_data_operations: str = ''

def test_SpectrumRenderer():
    # test1
    with pytest.raises(TypeError) as excinfo:
        SpectrumRenderer()
    assert "Can't instantiate abstract class SpectrumRenderer" in str(excinfo.value)

    # test2
    with pytest.raises(TypeError) as excinfo:
        IncompleteRenderer()
    assert "without an implementation for abstract methods" in str(excinfo.value)
    
    # test3 - No arguments provided
    try:
        renderer = ConcreteSpectrumRenderer()
        assert renderer.spec_data is None
        assert renderer.spec_grid is None
        assert renderer.ev_info is None
        assert renderer.rnd_info is None

    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")
    # test4
    with pytest.raises(TypeError) as excinfo:
        renderer = ConcreteSpectrumRenderer(spec_data='smth', spec_grid='smth', ev_info='smth', rnd_info=MockRndInfo(), do_diagn='smth')
        renderer.render()
    assert "SpectrumRenderer.render() missing 1 required positional argument" in str(excinfo.value)
    
    # test5
    with pytest.raises(TypeError) as excinfo:
        renderer = ConcreteSpectrumRenderer(spec_data='smth', spec_grid='smth', ev_info='smth', rnd_info=MockRndInfo(), do_diagn='smth')
        renderer.render('f')
    assert "spec_data should be a np.ndarray" in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        renderer = ConcreteSpectrumRenderer(spec_data=np.array([]), spec_grid='smth', ev_info='smth', rnd_info=MockRndInfo(), do_diagn='smth')
        renderer.render('f')
    assert "spec_data array should not be empty" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        renderer = ConcreteSpectrumRenderer(spec_data=np.array([2]), spec_grid='smth', ev_info='smth', rnd_info=MockRndInfo(), do_diagn='smth')
        renderer.render('f')
    assert "spec_grid should be a dictionary with X,Y,(Z) data" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        renderer = ConcreteSpectrumRenderer(spec_data=np.array([2]), spec_grid={}, ev_info='smth', rnd_info=MockRndInfo(), do_diagn='smth')
        renderer.render('f')
    assert "ev_info should be an instance of a class EvaluationInfo" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        import wilson_suite.wilson_main.spectrum_abstractions as specabst
        renderer = ConcreteSpectrumRenderer(spec_data=np.array([2]), spec_grid={}, 
                                            ev_info=specabst.EvaluationInfo(), 
                                            rnd_info=MockRndInfo(), 
                                            do_diagn='smth')
        renderer.render('f')
    assert "Unsupported spec_data_operations" in str(excinfo.value)

def test_LevelCalculator():
    dmax = 1000.
    intensities = np.array([[1000, 826, 826, 826, 826],
                  [826, 826, 826, 826, 826],
                  [826, 826, 826, 826, 826],
                  [826, 826, 826, 826, 826],
                  [826, 826, 826, 827,   1]])
    dyn_range = 10
    nlevels=5

    # original values, linear spacing
    levels, labels = LevelCalculator().compute_levels(intensities=intensities,
                                                      dynamic_range=dyn_range,
                                                      nlevels=nlevels,
                                                      colormap_spacing='linear')
    assert np.allclose(levels, np.array([ 100.,  325.,  550.,  775., 1000.]))
    assert labels == ['$1.0e+02$', '$3.2e+02$', '$5.5e+02$', '$7.8e+02$', '$1.0e+03$']
    
    # log10 values
    levels, labels = LevelCalculator().compute_levels(intensities=np.log10(intensities),
                                                      dynamic_range=dyn_range,
                                                      nlevels=nlevels,
                                                      colormap_spacing='linear')
    assert np.allclose(levels, np.array([0.3, 0.975, 1.65 , 2.325, 3.]))
    assert labels == ['$3.0e-01$', '$9.8e-01$', '$1.7e+00$', '$2.3e+00$', '$3.0e+00$']

    levels, labels = LevelCalculator().compute_levels(intensities=intensities,
                                                      dynamic_range=dyn_range,
                                                      nlevels=nlevels,
                                                      colormap_spacing='log')
    # linspace of powers of 10
    lvls = np.linspace(np.log10(dmax/dyn_range), np.log10(dmax), nlevels)
    assert np.allclose(lvls, np.array([2.,  2.25, 2.5,  2.75, 3. ]))

    assert np.allclose(levels, np.power(10, lvls))
    assert np.allclose(levels, np.array([ 100.,  177.827941, 316.22776602, 562.34132519, 1000. ]))
    

def test_NormalizationType():
    print('\nNormalizationType.LOG_SCALE', NormalizationType.LOG_SCALE, '\n')
    for normtype in NormalizationType:
        print(normtype)

    print('\n', list(NormalizationType))

    # Convert string to Enum
    method_name = "log_ratio"
    method = NormalizationType(method_name)
    print(method)

    with pytest.raises(ValueError) as excinfo:
        method_name = "smth"
        method = NormalizationType(method_name)
        print(method)
    assert 'is not a valid NormalizationType' in str(excinfo.value)


def test_PlotConfig():
    assert PlotConfig() == PlotConfig(figsize=(35,45),
                                      label_fontsize=25,
                                      font_dict={'size': 20},
                                      colormap='magma',
                                      saturation_color='#FF00FF',
                                      dpi=250,
                                      tick_step=200,
                                      equal_aspect=True,
                                      other_colors=True,
                                      no_data_color='#E0E0E0',
                                      below_range_color='#F8F8F8',
                                      data_edge_color='black',
                                      data_edge_width=0.75,
                                      x_min=None,x_max=None,y_min=None,y_max=None,
                                      colorbar_main_label='Intensity',
                                      colorbar_padding=0.01,
                                      show_top_ticks=False,
                                      show_right_ticks=False,
                                      x_tick_rotation=45,
                                      colormap_spacing='log')

def test_below_range_mask_logic():
    data = np.array([
        [0.0, 0.5],
        [10.0, np.nan],
    ])
    no_data, below = compute_masks(data=data, dynamic_range=10)

    assert no_data[1, 1]
    assert below[0, 0]
    assert not below[1, 0]
