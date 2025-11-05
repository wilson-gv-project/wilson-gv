from ...amplitudes.term_parts import SpectralWindow, RectangularWindow
from dataclasses import dataclass
import numpy as np
import pytest

def test_cannot_instantiate_abstract_class():
    """SpectralWindow cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Cannot instantiate abstract class"):
        SpectralWindow((10,))


def test_cannot_instantiate_direct_subclass_without_implementation():
    """Direct subclass without generate() implementation cannot be instantiated."""
    @dataclass
    class IncompleteWindow(SpectralWindow):
        pass
    
    with pytest.raises(TypeError):
        IncompleteWindow((10,))


def test_rectangular_1d_window():
    """Test 1D rectangular window."""
    window = RectangularWindow((64,))
    result = window.generate()
    
    assert result.shape == (64,)
    assert result.ndim == 1
    assert np.all(result == 1.0)
    assert result.dtype == np.float64


def test_rectangular_2d_window():
    """Test 2D rectangular window."""
    window = RectangularWindow((32, 48))
    result = window.generate()
    print('\n', result)

    assert result.shape == (32, 48)
    assert result.ndim == 2
    assert np.all(result == 1.0)
    assert window.ndim == 2


def test_rectangular_3d_window():
    """Test 3D rectangular window."""
    window = RectangularWindow((16, 16, 16))
    result = window.generate()
    
    assert result.shape == (16, 16, 16)
    assert result.ndim == 3
    assert np.all(result == 1.0)
    assert window.ndim == 3


def test_callable_interface():
    """Test that window can be called like a function."""
    window = RectangularWindow((10, 10))
    result1 = window.generate()
    result2 = window()
    print()
    print(result1)
    print(result2)

    assert np.array_equal(result1, result2)


def test_asymmetric_shape():
    """Test window with asymmetric shape."""
    window = RectangularWindow((10, 20, 30))
    result = window.generate()
    
    assert result.shape == (10, 20, 30)
    assert np.all(result == 1.0)


def test_single_element_dimensions():
    """Test window with single element in some dimensions."""
    window = RectangularWindow((1, 10, 1))
    result = window.generate()
    
    assert result.shape == (1, 10, 1)
    assert np.all(result == 1.0)


def test_ndim_property():
    """Test ndim property returns correct dimensionality."""
    assert RectangularWindow((10,)).ndim == 1
    assert RectangularWindow((10, 10)).ndim == 2
    assert RectangularWindow((10, 10, 10)).ndim == 3


def test_multiple_generate_calls_consistent():
    """Test that multiple calls to generate() produce consistent results."""
    window = RectangularWindow((20, 20))
    result1 = window.generate()
    result2 = window.generate()
    
    assert np.array_equal(result1, result2)
    assert result1 is not result2  # Should be different objects


def test_default_bounds():
    """Test default bounds are set correctly."""
    window = RectangularWindow((10, 20))
    
    assert window.bounds == ((0.0, 10.0), (0.0, 20.0))


def test_custom_bounds():
    """Test custom bounds."""
    window = RectangularWindow((10, 20), bounds=((-5.0, 5.0), (0.0, 100.0)))
    
    assert window.bounds == ((-5.0, 5.0), (0.0, 100.0))


def test_bounds_dimensionality_mismatch():
    """Test that mismatched bounds dimensionality raises error."""
    with pytest.raises(ValueError, match="bounds dimensionality.*must match"):
        RectangularWindow((10, 20), bounds=((-5.0, 5.0),))


def test_grid_1d():
    """Test grid generation for 1D window."""
    window = RectangularWindow((5,), bounds=((0.0, 10.0),))
    grids = window.meshgrids()
    
    assert len(grids) == 1
    assert grids[0].shape == (5,)
    np.testing.assert_array_almost_equal(grids[0], [0.0, 2.0, 4.0, 6.0, 8.0])


def test_grid_2d():
    """Test grid generation for 2D window."""
    window = RectangularWindow((3, 4), bounds=((-1.0, 2.0), (0.0, 4.0)))
    grids = window.meshgrids()
    
    assert len(grids) == 2
    assert grids[0].shape == (3, 4)
    assert grids[1].shape == (3, 4)


def test_grid_flat_1d():
    """Test flattened grid for 1D window."""
    window = RectangularWindow((5,), bounds=((0.0, 5.0),))
    flat_grid = window.grid_flat()
    
    assert flat_grid.shape == (5, 1)
    np.testing.assert_array_almost_equal(flat_grid[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])


def test_grid_flat_2d():
    """Test flattened grid for 2D window."""
    window = RectangularWindow((2, 3), bounds=((0.0, 2.0), (0.0, 3.0)))
    flat_grid = window.grid_flat()
    
    assert flat_grid.shape == (6, 2)
    assert flat_grid.shape[0] == 2 * 3  # Total points


def test_contains_inside():
    """Test points inside bounds."""
    window = RectangularWindow((10, 10), bounds=((0.0, 10.0), (0.0, 10.0)))
    
    points = np.array([[5.0, 5.0], [1.0, 9.0], [0.0, 0.0]])
    result = window.contains(points)
    
    assert np.all(result == [True, True, True])


def test_contains_outside():
    """Test points outside bounds."""
    window = RectangularWindow((10, 10), bounds=((0.0, 10.0), (0.0, 10.0)))
    
    points = np.array([[10.0, 5.0], [-1.0, 5.0], [5.0, 10.0]])
    result = window.contains(points)
    
    assert np.all(result == [False, False, False])


def test_contains_mixed():
    """Test mix of inside and outside points."""
    window = RectangularWindow((10, 10), bounds=((0.0, 10.0), (0.0, 10.0)))
    
    points = np.array([[5.0, 5.0], [15.0, 5.0], [5.0, -1.0]])
    result = window.contains(points)
    
    assert np.all(result == [True, False, False])


def test_contains_wrong_dimensionality():
    """Test contains with wrong number of dimensions."""
    window = RectangularWindow((10, 10))
    
    with pytest.raises(ValueError, match="must have.*coordinates"):
        window.contains(np.array([[1.0, 2.0, 3.0]]))


def test_extent():
    """Test extent calculation."""
    window = RectangularWindow((10, 20), bounds=((-5.0, 5.0), (0.0, 100.0)))
    
    assert window.extent == (10.0, 100.0)


def test_volume_1d():
    """Test volume for 1D window (length)."""
    window = RectangularWindow((10,), bounds=((0.0, 10.0),))
    
    assert window.volume == 10.0


def test_volume_2d():
    """Test volume for 2D window (area)."""
    window = RectangularWindow((10, 20), bounds=((0.0, 5.0), (0.0, 10.0)))
    
    assert window.volume == 50.0


def test_volume_3d():
    """Test volume for 3D window."""
    window = RectangularWindow((10, 10, 10), bounds=((0.0, 2.0), (0.0, 3.0), (0.0, 4.0)))
    
    assert window.volume == 24.0


def test_center():
    """Test center calculation."""
    window = RectangularWindow((10, 20), bounds=((-5.0, 5.0), (0.0, 100.0)))
    
    assert window.center == (0.0, 50.0)