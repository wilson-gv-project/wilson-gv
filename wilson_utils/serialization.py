import json
from dataclasses import is_dataclass, asdict
import numpy as np

def find_non_json_safe(obj, path=""):
    """
    Recursively find non-JSON-serializable elements in an object.

    chatgpt
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{path}[{repr(key)}]"
            if isinstance(key, (tuple, list, set)):  # JSON keys must be str, int, float, bool, or None
                print(f"❌ Non-JSON-safe key at {key_path}: {key}")
            find_non_json_safe(value, key_path)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            find_non_json_safe(item, f"{path}[{index}]")
    elif isinstance(obj, tuple):
        print(f"❌ Non-JSON-safe value (tuple) at {path}: {obj}")
        for index, item in enumerate(obj):
            find_non_json_safe(item, f"{path}[{index}]")
    elif isinstance(obj, set):
        print(f"❌ Non-JSON-safe value (set) at {path}: {obj}")
        for item in obj:
            find_non_json_safe(item, f"{path}[set_item]")
    else:
        try:
            json.dumps(obj)  # Attempt to serialize the object
        except TypeError:
            print(f"❌ Non-JSON-safe value at {path}: {obj}")

def check_if_jsonsafe(obj):
    """
    Check if this object is JSON-serializable.
    For dataclasses, uses asdict().
    """
    try:
        if is_dataclass(obj):
            dict_obj = asdict(obj)
        elif hasattr(obj, "__dict__"):
            dict_obj = obj.__dict__
        else:
            dict_obj = obj

        json.dumps(dict_obj)
        print("✅ JSON-safe")
        return True
    except TypeError as e:
        print("🔍 Offending object:", dict_obj)
        print("❌ Not JSON-safe:", e)
        # find_non_json_safe(dict_obj)
        return False
    except Exception as e:
        print("⚠️ Other error:", e)
        return False


def ndarray_to_dict(arr, serial=True):
    """
    Converts a numpy array into a dict headed by tuple of indices of values. 

    If serial then tuples are turned into strings (tuples can't be keys for JSON)
    """
    if serial:
        return {
            str(idx): arr[idx].item() if hasattr(arr[idx], 'item') else arr[idx]
            for idx in np.ndindex(arr.shape)
        }
    else:
        return {idx: arr[idx].item() for idx in np.ndindex(arr.shape)}