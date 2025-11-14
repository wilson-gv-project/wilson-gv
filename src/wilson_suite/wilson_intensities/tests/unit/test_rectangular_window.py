# import pytest
# import numpy as np
# from ...amplitudes.spectrum_composition import SpectralWindow
# from wilson_suite.wilson_utils.common_labels import cap_alpha_labels
# from ...amplitudes.term_parts import RectangularWindow


# def test_init_default_labels():
#     """Default labels come from cap_alpha_labels and match shape length."""
#     w = RectangularWindow(shape=(3, 4, 2))
#     assert w.labels == tuple(cap_alpha_labels[:3])
#     assert len(w.labels) == 3
#     assert all(label in w._label_to_index for label in w.labels)
#     assert w._label_to_index['A'] == 0
#     assert w._label_to_index['C'] == 2


# def test_init_custom_labels():
#     """Custom labels override default and must match shape length."""
#     labels = ('X', 'Y', 'Z')
#     w = RectangularWindow(shape=(3, 4, 2), labels=labels)
#     assert w.labels == labels
#     assert w._label_to_index['Z'] == 2

#     # Mismatched label count → ValueError
#     with pytest.raises(ValueError):
#         RectangularWindow(shape=(3, 4), labels=('X', 'Y', 'Z'))


# def test_too_many_dimensions_for_labels():
#     """Error if more dimensions than available predefined labels."""
#     shape = tuple(range(1, len(cap_alpha_labels) + 2))  # one more than available
#     with pytest.raises(ValueError):
#         RectangularWindow(shape=shape)


# def test_axis_index_mapping():
#     """Axis index can be accessed by label or number."""
#     w = RectangularWindow(shape=(5, 5))
#     assert w.axis_index('A') == 0
#     assert w.axis_index('B') == 1
#     assert w.axis_index(0) == 0
#     assert w.axis_index(1) == 1


# def test_axis_bounds_and_extent():
#     """Axis bounds and extents computed correctly."""
#     shape = (4, 6)
#     w = RectangularWindow(shape=shape)
#     b0 = w.axis_bounds('A')
#     b1 = w.axis_bounds('B')
#     print()
#     print(b0, b1)
#     b1 = w.axis_bounds(1)

#     assert b0 == (0.0, float(shape[0]))
#     assert b1 == (0.0, float(shape[1]))
#     assert w.axis_extent('A') == shape[0]
#     assert w.axis_extent('B') == shape[1]


# def test_axis_coords_generation():
#     """Coordinates are evenly spaced between min and max."""
#     w = RectangularWindow(shape=(4,), labels=('A',))
#     coords = w.axis_coords('A')
#     expected = np.linspace(0, 4, 4, endpoint=False)
#     np.testing.assert_allclose(coords, expected)


# def test_getitem_behavior():
#     """__getitem__ returns coordinate arrays for one or multiple labels."""
#     w = RectangularWindow(shape=(3, 2, 4))
#     coord_A = w['A']
#     coord_B, coord_C = w['B', 'C']

#     print()
#     print(coord_A)
#     print(coord_B)
#     print(w)

#     assert isinstance(coord_A, np.ndarray)
#     assert coord_A.shape == (3,)
#     assert isinstance(coord_B, np.ndarray)
#     assert isinstance(coord_C, np.ndarray)

#     # Invalid key type
#     with pytest.raises(TypeError):
#         _ = w[123]


# def test_generate_and_call():
#     """generate() and __call__() should return all-ones arrays."""
#     w = RectangularWindow(shape=(2, 3))
#     arr = w.generate()
#     assert np.all(arr == 1)
#     assert arr.shape == (2, 3)

#     # __call__ should behave identically
#     arr2 = w()
#     np.testing.assert_array_equal(arr, arr2)


# def test_grid_and_grid_flat():
#     """grid() and grid_flat() produce consistent coordinate grids."""
#     w = RectangularWindow(shape=(2, 3))
#     grids = w.meshgrids()
#     print()
#     print('grids\n', grids)
#     assert len(grids) == 2
#     assert grids[0].shape == (2, 3)
#     assert grids[1].shape == (2, 3)

#     flat = w.grid_flat()
#     print('flat\n', flat)
#     assert flat.shape == (2 * 3, 2)
#     np.testing.assert_allclose(flat[:, 0].reshape(2, 3), grids[0])
#     np.testing.assert_allclose(flat[:, 1].reshape(2, 3), grids[1])


# def test_contains_points():
#     """contains() correctly identifies points inside bounds."""
#     w = RectangularWindow(shape=(4, 4))
#     inside_points = np.array([[1, 1], [0.5, 3.9]])
#     outside_points = np.array([[4.1, 2], [-0.1, 0]])
#     mask_inside = w.contains(inside_points)
#     mask_outside = w.contains(outside_points)
#     print()
#     print(w)
#     print(mask_inside)
#     print(mask_outside)

#     assert np.all(mask_inside)
#     assert not np.any(mask_outside)

#     # Wrong shape
#     with pytest.raises(ValueError):
#         w.contains(np.array([[1, 2, 3]]))


# def test_extent_volume_center():
#     """extent, volume, and center properties are correct."""
#     bounds = ((0.0, 2.0), (1.0, 3.0))
#     w = RectangularWindow(shape=(2, 2), bounds=bounds)
#     assert np.allclose(w.extent, (2.0, 2.0))
#     assert np.isclose(w.volume, 4.0)
#     assert np.allclose(w.center, (1.0, 2.0))
