import json
from dataclasses import is_dataclass, asdict
import numpy as np
import pickle

import logging
logger = logging.getLogger("wilson."+__name__)

def find_non_json_safe(obj, path=""):
    """
    Recursively find non-JSON-serializable elements in an object.

    chatgpt
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{path}[{repr(key)}]"
            if isinstance(key, (tuple, list, set)):  # JSON keys must be str, int, float, bool, or None
                logger.warning(f"❌ Non-JSON-safe key at {key_path}: {key}")
            find_non_json_safe(value, key_path)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            find_non_json_safe(item, f"{path}[{index}]")
    elif isinstance(obj, tuple):
        logger.warning(f"❌ Non-JSON-safe value (tuple) at {path}: {obj}")
        for index, item in enumerate(obj):
            find_non_json_safe(item, f"{path}[{index}]")
    elif isinstance(obj, set):
        logger.warning(f"❌ Non-JSON-safe value (set) at {path}: {obj}")
        for item in obj:
            find_non_json_safe(item, f"{path}[set_item]")
    else:
        try:
            json.dumps(obj)  # Attempt to serialize the object
        except TypeError:
            logger.warning(f"❌ Non-JSON-safe value at {path}: {obj}")

def check_if_jsonsafe(obj):
    """
    Check if this object is JSON-serializable.
    For dataclasses, uses asdict().
    """
    try:
        if is_dataclass(obj):
            dict_obj = asdict(obj)
        # check if to_dict() method is implemented
        elif hasattr(obj, "to_dict"):
            dict_obj = obj.to_dict()
        # as the last resort, use __dict__
        elif hasattr(obj, "__dict__"):
            dict_obj = obj.__dict__
        # should be fine then
        else:
            dict_obj = obj

        json.dumps(dict_obj)
        logger.info("✅ JSON-safe")
        return True
    except TypeError as e:
        logger.warning(f"🔍 Offending object: {dict_obj}")
        logger.error(f"❌ Not JSON-safe: {e}")
        # for more infor here can do: find_non_json_safe(dict_obj)
        return False
    except Exception as e:
        logger.error(f"⚠️ Other error: {e}")
        return False


def ndarray_to_dict(arr: np.ndarray, serial=True):
    """
    Converts a numpy array into a dict headed by tuple of indices of values. 

    If serial then tuples are turned into strings (tuples can't be keys for JSON)
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("ndarray_to_dict() accepts only numpy arrays.")

    if serial:
        return {
            str(idx): arr[idx].item() if hasattr(arr[idx], 'item') else arr[idx]
            for idx in np.ndindex(arr.shape)
        }
    else:
        return {idx: arr[idx].item() for idx in np.ndindex(arr.shape)}
    

def pickle_this_to(obj, filenamepkl=str, save_to: str = ''):
    import os
    filepath = os.path.abspath(os.path.join(save_to, filenamepkl))

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

    return

def unpickle_smth_from(filenamepkl: str, load_from: str = ''):
    import os
    filepath = os.path.abspath(os.path.join(load_from, filenamepkl))

    with open(filepath, "rb") as f:
        loaded_obj = pickle.load(f)

    return loaded_obj