from dataclasses import dataclass
from wilson_suite.wilson_analysis.render.spectrum_renderer import SpectrumRenderer, LevelCalculator, NormalizationType, PlotConfig
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
    
    # test3
    with pytest.raises(TypeError) as excinfo:
        ConcreteSpectrumRenderer()
    assert "SpectrumRenderer.__init__() missing 4 required positional arguments" in str(excinfo.value)

    # test4
    with pytest.raises(TypeError) as excinfo:
        renderer = ConcreteSpectrumRenderer(spec_data='smth', spec_grid='smth', ev_info='smth', rnd_info=MockRndInfo(), do_diagn='smth')
        renderer.render()
    assert "SpectrumRenderer.render() missing 1 required positional argument" in str(excinfo.value)
    
    # test5
    with pytest.raises(ValueError) as excinfo:
        renderer = ConcreteSpectrumRenderer(spec_data='smth', spec_grid='smth', ev_info='smth', rnd_info=MockRndInfo(), do_diagn='smth')
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
    PlotConfig()
    pass