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
    log10 = False # original values, linear spacing
    nlevels=5

    levels, labels, norm_positions, norm_labels = LevelCalculator().compute_levels(d_max=dmax,
                                                            dynamic_range=dyn_range,
                                                            nlevels=nlevels,
                                                            log10=log10)
    assert np.allclose(levels, np.array([ 100.,  325.,  550.,  775., 1000.]))
    assert labels == ['$1.0e+02$', '$3.2e+02$', '$5.5e+02$', '$7.8e+02$', '$1.0e+03$']
    assert norm_positions is None
    assert norm_labels is None
    
    log10 = True # log10 values
    cm_spacing = 'linear' # linear spacing of original level vals
    levels, labels, norm_positions, norm_labels = LevelCalculator().compute_levels(d_max=dmax,
                                                            dynamic_range=dyn_range,
                                                            nlevels=nlevels,
                                                            log10=log10,
                                                            colormap_spacing=cm_spacing)
    assert np.allclose(levels, np.array([ 100.,  325.,  550.,  775., 1000.]))
    assert labels == ['$1.0e+02$', '$3.2e+02$', '$5.5e+02$', '$7.8e+02$', '$1.0e+03$']

    ref_norm_log10_lin = (np.log10(levels) - np.log10(dmax/dyn_range)) / (np.log10(dmax) - np.log10(dmax/dyn_range))
    assert np.allclose(ref_norm_log10_lin, np.array([0. , 0.51188336, 0.74036269, 0.8893017 , 1.]))
    assert np.allclose(norm_positions, np.array([0. , 0.51188336, 0.74036269, 0.8893017 , 1.]))
    assert np.allclose(ref_norm_log10_lin, norm_positions)
    
    
    log10 = True
    cm_spacing = 'log' # Linear spacing in log scale with values back to linear scale
    levels, labels, norm_positions, norm_labels = LevelCalculator().compute_levels(d_max=dmax,
                                                            dynamic_range=dyn_range,
                                                            nlevels=nlevels,
                                                            log10=log10,
                                                            colormap_spacing=cm_spacing)
    # linspace of powers of 10
    lvls = np.linspace(np.log10(dmax/dyn_range), np.log10(dmax), nlevels)
    assert np.allclose(lvls, np.array([2.,  2.25, 2.5,  2.75, 3. ]))
    assert np.allclose(levels, np.power(10, lvls))
    assert np.allclose(levels, np.array([ 100.,  177.827941, 316.22776602, 562.34132519, 1000. ]))
    
    ref_norm_log10_log = (np.log10(levels) - np.log10(dmax/dyn_range)) / (np.log10(dmax) - np.log10(dmax/dyn_range))
    ref = (np.log10(levels) - np.log10(np.min(levels))) / (np.log10(np.max(levels)) - np.log10(np.min(levels)))
    print(ref)
    print(np.log10(levels))
    print(np.log10(levels) - np.log10(np.min(levels)))
    assert np.allclose(ref_norm_log10_log, np.array([0., 0.25, 0.5 , 0.75, 1. ]))
    assert np.allclose(norm_positions, ref_norm_log10_log)
    assert norm_labels == ['0.00', '0.25', '0.50', '0.75', '1.00']


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