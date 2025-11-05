"""
Claude
"""
import pytest
from ...amplitudes.term_parts import GeometricObject

def test_point_creation():
    point = GeometricObject({'A': 1864.0, 'B': 900.0})
    assert point.is_point()
    assert not point.is_line()
    assert point.dimensionality == 0
    assert point['A'] == 1864.0
    assert point['B'] == 900.0
    assert point.dims == ('A', 'B')
    assert point.values == (1864.0, 900.0)

def test_line_creation():
    line = GeometricObject({'A': 1864.0, 'B': 'all'})
    assert line.is_line()
    assert not line.is_point()
    assert line.dimensionality == 1

def test_plane_creation():
    plane = GeometricObject({'A': 'all', 'B': 'all', 'C': 1200.0})
    assert plane.is_plane()
    assert plane.dimensionality == 2
    assert plane['C'] == 1200.0

def test_hashable():
    point1 = GeometricObject({'A': 1864.0, 'B': 900.0})
    point2 = GeometricObject({'A': 1864.0, 'B': 900.0})
    point3 = GeometricObject({'A': 1864.0, 'B': 'all'})
    
    # Test set operations
    unique_objects = {point1, point2, point3}
    assert len(unique_objects) == 2
    
    # Test dict operations
    object_map = {point1: "point", point3: "line"}
    assert len(object_map) == 2
    assert object_map[point2] == "point"  # point2 equals point1
    assert hash(point1) == hash(point2)
    assert hash(point1) != hash(point3)

def test_ordering_invariance():
    obj1 = GeometricObject({'A': 1.0, 'B': 2.0})
    obj2 = GeometricObject({'B': 2.0, 'A': 1.0})
    assert obj1 == obj2
    assert hash(obj1) == hash(obj2)

def test_invalid_access():
    point = GeometricObject({'A': 1.0, 'B': 2.0})
    with pytest.raises(KeyError):
        _ = point['C']