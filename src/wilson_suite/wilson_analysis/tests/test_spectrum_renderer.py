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
    dyn_range = 10
    log10 = False


    levels, labels, _, _ = LevelCalculator().compute_levels(d_max=dmax,
                                                            dynamic_range=dyn_range,
                                                            nlevels=5,
                                                            log10=log10)

    assert np.allclose(levels, np.array([ 10., 32.5,  55., 77.5, 100.]))
    assert labels == ['$1.0e+01$', '$3.2e+01$', '$5.5e+01$', '$7.8e+01$', '$1.0e+02$']
    
    log10 = True
    cm_spacing = 'linear'
    levels, labels, _, _ = LevelCalculator().compute_levels(d_max=dmax,
                                                            dynamic_range=dyn_range,
                                                            nlevels=5,
                                                            log10=log10,
                                                            colormap_spacing=cm_spacing)
    print()
    print(levels)
    print(labels)

    log10 = True
    cm_spacing = 'power'
    cm_power = 0.1
    levels, labels, _, _ = LevelCalculator().compute_levels(d_max=dmax,
                                                            dynamic_range=dyn_range,
                                                            nlevels=5,
                                                            log10=log10,
                                                            colormap_spacing=cm_spacing,
                                                            colormap_power=cm_power)



def test_NormalizationType():
    print('\nNormalizationType.LOG_SCALE', NormalizationType.LOG_SCALE, '\n')
    for normtype in NormalizationType:
        print(normtype)

    print('\n', list(NormalizationType))

    # Convert string to Enum
    method_name = "log_ratio"
    method = NormalizationType(method_name)
    print(method)

    # method_name = "smth"
    # method = NormalizationType(method_name)
    # print(method)

def test_PlotConfig():
    PlotConfig()
    pass