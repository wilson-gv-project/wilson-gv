from pathlib import Path
import pytest
from unittest.mock import MagicMock
from matplotlib.figure import Figure
from wilson_analysis.render.base_template import BaseRenderer  

class MockRenderer(BaseRenderer):
    def prepare_axes_data(self) -> dict:
        return {'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9]}
    def prepare_contour_levels(self):
        self.tick_values = [0, 1, 2]
        self.tick_labels = ['Low', 'Medium', 'High']
        self.tick_norm_positions = [0.0, 0.5, 1.0]
    def create_figure(self):
        fig = MagicMock(spec=Figure)
        ax = MagicMock()
        return fig, ax
    def plot_contours(self, fig, ax, axes_dict):
        pass  # Mock implementation for testing
    def finalize(self, fig, ax, filename):
        # Simulate saving the figure
        fig.savefig(filename)

@pytest.fixture
def renderer():
    """Fixture to provide a mock renderer instance."""
    return MockRenderer(intensities=[1, 2, 3])

def test_prepare_axes_data(renderer: MockRenderer):
    axes_data = renderer.prepare_axes_data()
    assert axes_data['x'] == [1, 2, 3]
    assert axes_data['y'] == [4, 5, 6]
    assert axes_data['z'] == [7, 8, 9]
def test_prepare_contour_levels(renderer: MockRenderer):
    renderer.prepare_contour_levels()
    assert renderer.tick_values == [0, 1, 2]
    assert renderer.tick_labels == ['Low', 'Medium', 'High']
    assert renderer.tick_norm_positions == [0.0, 0.5, 1.0]
def test_create_figure(renderer: MockRenderer):
    fig, ax = renderer.create_figure()
    assert isinstance(fig, MagicMock)
    assert isinstance(ax, MagicMock)
def test_render_pipeline(renderer: MockRenderer):
    # Mock methods to ensure they are called
    renderer.prepare_axes_data = MagicMock(return_value={'x': [1], 'y': [2], 'z': [3]})
    renderer.prepare_contour_levels = MagicMock()
    renderer.create_figure = MagicMock(return_value=(MagicMock(), MagicMock()))
    renderer.plot_contours = MagicMock()
    renderer.finalize = MagicMock()
    # Call the render method
    renderer.render("test_output.png")
    # Assert that all methods in the pipeline are called
    renderer.prepare_axes_data.assert_called_once()
    renderer.prepare_contour_levels.assert_called_once()
    renderer.create_figure.assert_called_once()
    renderer.plot_contours.assert_called_once()
    renderer.finalize.assert_called_once()
def test_finalize_creates_file(renderer: MockRenderer, tmp_path: Path):
    # Create a temporary file path
    filename = tmp_path / "test_output.png"
    # Use a real matplotlib Figure object
    fig = Figure()
    ax = fig.add_subplot(111)
    renderer.finalize(fig, ax, str(filename))
    assert filename.exists()