from dataclasses import dataclass
from wilson_suite.wilson_analysis.render.spectrum_renderer import SpectrumRenderer
import pytest

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
    pass

def test_NormalizationType():
    pass

def test_PlotConfig():
    pass