"""
Cahtgpt

Utilities for testing
"""
import ast
import inspect
from functools import wraps
from typing import Callable

def has_asserts(func: Callable) -> bool:
    """
    Suplementary function to check for asserts
    """
    source = inspect.getsource(func)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            return True
    return False

def require_asserts(test_func: Callable):
    """Decorator to ensure the test function contains assert statements."""
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        if not has_asserts(test_func):
            print(f"\n\033[33m⚠️  WARNING: {test_func.__name__} has no assert statements!\033[0m")
        return test_func(*args, **kwargs)
    return wrapper
