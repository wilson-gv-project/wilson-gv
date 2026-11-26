from ...wilson_utils.serialization import pickle_this_to, unpickle_smth_from
from dataclasses import dataclass


@dataclass
class Mock1:
    a: int
    b: float

@dataclass
class Mock2:
    a: int
    b: dict

def test_pickle_unpickle():

    obj1 = Mock1(2, 3.4)
    
    obj21 = Mock2(1, {0.2: 'str'})
    obj22 = Mock2(1, {'0.2': 'str'})
    obj23 = Mock2(1, {(0,1): 'str'})
    
    dict1 = {2: 'fd', '23': 1.2}
    dict2 = {('d', 2): 'fd', '23': 1.2}
    list1 = [3, 4, 2, 'frf']
    list2 = [3, 4, 2]
    objects = [obj1, obj21, obj22, obj23, dict1, dict2, list1, list2]
    types = [Mock1, Mock2, Mock2, Mock2, dict, dict, list, list]
    import os
    import tempfile
    
    # a temporary file path
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, obj in enumerate(objects):
            filepath = "test_obj.pkl"
            fullpath = os.path.join(tmpdir, filepath)

            pickle_this_to(obj, filenamepkl=filepath, save_to=tmpdir)

            assert os.path.exists(fullpath)

            loaded = unpickle_smth_from(filenamepkl=filepath, load_from=tmpdir)

            assert isinstance(loaded, types[i])
            assert loaded == obj
