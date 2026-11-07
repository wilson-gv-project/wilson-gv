"""
data_obtainer groups the requested names by their DataOriginInfo, calls CQCParse's parse_from_source once per
origin, and merges the results. CQCParse tests its own readers; the tests here cover what the obtainer adds, and one
load of a real file. Which names the old workflow asks for (WilsonSimulation.requestData) is tested in
wilson_main/tests/test_workflow_abstractions.py.
"""
from importlib.resources import files

import pytest

from wilson_suite.wilson_main.abstractions import DataOriginInfo
from wilson_suite.wilson_system.data_obtainer import wilson_data_obtainer

data_dir = files('wilson_suite').joinpath('data_for_tests')
WATER = DataOriginInfo(source_type='gaussian', base_file_loc=data_dir / 'g16_h2o_HF_STO3G.out')


def test_getting_data():
    """
    A real file, water HF/STO-3G from Gaussian. Checks what wilson relies on, not CQCParse's full list of keys: every
    requested name comes back, cff_rc and qff_rc come with cff and qff (the VPT2 reads them), and each tensor has one
    axis per mode index.
    """
    names = ['cff', 'anharmonic_states', 'nc_sqrt_eigval', 'dipgrad', 'B', 'polgrad', 'coriolis', 'polhess', 'qff',
             'diphess']
    data = wilson_data_obtainer(dict.fromkeys(names, WATER))

    assert set(names) | {'cff_rc', 'qff_rc'} <= set(data)
    assert all(data[k] is not None for k in names)
    n = len(data['nc_sqrt_eigval'])
    assert n == 3
    shapes = {'dipgrad': (n, 3), 'diphess': (n, n, 3), 'polgrad': (n, 3, 3), 'polhess': (n, n, 3, 3),
              'cff': (n, n, n), 'cff_rc': (n, n, n), 'qff': (n, n, n, n), 'qff_rc': (n, n, n, n),
              'coriolis': (3, n, n)}
    assert {k: data[k].shape for k in shapes} == shapes


@pytest.fixture
def calls(monkeypatch):
    """parse_from_source replaced by a fake: reads no file, records the names of each call, returns text as values."""
    made = []

    def fake(requested_data, **origin):
        made.append(list(requested_data))
        return {name: f"{name} from {origin['base_file_loc']}" for name in requested_data}

    monkeypatch.setattr('CQCParse.parsing.parse_from_source', fake)
    return made


def test_one_call_per_origin(calls):
    """Names of equal DataOriginInfo go to parse_from_source together; each name keeps the value from its own call."""
    g16 = DataOriginInfo(source_type='gaussian', base_file_loc='g16.out')
    same_as_g16 = DataOriginInfo(source_type='gaussian', base_file_loc='g16.out')
    c4 = DataOriginInfo(source_type='cfour', base_file_loc='c4_folder')

    data = wilson_data_obtainer({'cff': g16, 'dipgrad': c4, 'qff': same_as_g16})

    assert calls == [['cff', 'qff'], ['dipgrad']]
    assert data == {'cff': 'cff from g16.out', 'qff': 'qff from g16.out', 'dipgrad': 'dipgrad from c4_folder'}


def test_unknown_source_type(calls):
    """A source type the obtainer does not know stops with an error and never reaches parse_from_source."""
    unknown = DataOriginInfo(source_type='orca')

    with pytest.raises(ValueError):
        wilson_data_obtainer({'cff': unknown})
    assert not calls
