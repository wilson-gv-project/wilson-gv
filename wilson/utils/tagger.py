# tagger.py
# from functools import wraps

TAG_REGISTRY = {}

def tag(*tags):
    def decorator(obj):
        TAG_REGISTRY[obj.__name__] = tags
        return obj
    return decorator
