# from functools import wraps

TAG_REGISTRY = {}

def tag(*tags):
    """
    Tags decorator for functions and methods
    """
    def decorator(obj):
        TAG_REGISTRY[obj.__name__] = tags
        return obj
    return decorator
