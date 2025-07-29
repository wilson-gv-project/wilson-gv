from wilson_utils.serialization import find_non_json_safe, check_if_jsonsafe, ndarray_to_dict, pickle_this_to, unpickle_smth_from
import numpy as np
from dataclasses import dataclass


def test_ndarray_to_dict():
    arr1 = np.array([[0.096, 0.030, 0.179],
                     [0.755, 0.657, 0.191],
                     [0.318, 0.859, 0.258]])
    strkeys1 = ndarray_to_dict(arr1, serial=True)
    assert strkeys1 == {'(0, 0)': 0.096, '(0, 1)': 0.03, '(0, 2)': 0.179, 
                        '(1, 0)': 0.755, '(1, 1)': 0.657, '(1, 2)': 0.191, 
                        '(2, 0)': 0.318, '(2, 1)': 0.859, '(2, 2)': 0.258}
    
    tuplekeys1 = ndarray_to_dict(arr1, serial=False)
    assert tuplekeys1 == {(0, 0): 0.096, (0, 1): 0.03, (0, 2): 0.179, 
                          (1, 0): 0.755, (1, 1): 0.657, (1, 2): 0.191, 
                          (2, 0): 0.318, (2, 1): 0.859, (2, 2): 0.258}
    
    arr2 = np.array([[0.91, 0.21, 0.39]])
    strkeys2 = ndarray_to_dict(arr2, serial=True)
    tuplekeys2 = ndarray_to_dict(arr2, serial=False)

    assert strkeys2 == {'(0, 0)': 0.91, '(0, 1)': 0.21, '(0, 2)': 0.39}
    assert tuplekeys2 == {(0, 0): 0.91, (0, 1): 0.21, (0, 2): 0.39}


def test_check_if_jsonsafe():
    @dataclass
    class Mock:
        a: int
        b: float

    assert check_if_jsonsafe(Mock(1, 0.2))

    @dataclass
    class Mock:
        a: int
        b: dict

    assert check_if_jsonsafe(Mock(1, {0.2: 'str'}))
    assert check_if_jsonsafe(Mock(1, {'0.2': 'str'}))
    assert not check_if_jsonsafe(Mock(1, {(0,1): 'str'}))

    dict1 = {2: 'fd', '23': 1.2}
    assert check_if_jsonsafe(dict1)
    dict2 = {('d', 2): 'fd', '23': 1.2}
    assert not check_if_jsonsafe(dict2)

    list1 = [3, 4, 2, 'frf']
    assert check_if_jsonsafe(list1)
    list2 = [3, 4, 2]
    assert check_if_jsonsafe(list2)


def test_find_non_json_safe():
    """
    Function find_non_json_safe is not used much yet.

    Will add more use cases when in use.
    """
    nestedobj1 = [(4,2), (4,3), (-1,5)]
    r = find_non_json_safe(nestedobj1)
    
    print(r)


def test_get_utils_root():
    import os
    from wilson_utils.paths import UTILS_ROOT

    assert os.path.isabs(UTILS_ROOT)
    assert os.path.exists(UTILS_ROOT)
    assert os.path.basename(UTILS_ROOT) == "wilson_utils"
    
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
