"""
Tag decorator for functions/classes.

All tagged parts can be listed
"""
# from functools import wraps
from typing import Callable

TAG_REGISTRY = {}

def tag(*tags: tuple[str]) -> Callable[[Callable], Callable]:
    """
    Tags decorator for functions and methods
    """
    def decorator(obj: Callable) -> Callable:
        TAG_REGISTRY[obj.__name__] = tags
        return obj
    return decorator
