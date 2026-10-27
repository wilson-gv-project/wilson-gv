"""
wilson_data_obtainer groups the requested names by their DataOriginInfo, calls CQCParse's parse_from_source once per
origin, and merges the results. CQCParse tests its own readers; the tests here cover what the obtainer adds, and one
load of a real file. Which names the old workflow asks for (WilsonSimulation.requestData) is tested in
wilson_main/tests/test_workflow_abstractions.py.
"""
import dataclasses
import inspect
from importlib.resources import files

import pytest

from wilson_suite.wilson_main.abstractions import DataOriginInfo
from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer

data_dir = files('wilson_suite').joinpath('data_for_tests')
WATER = DataOriginInfo(source_type='gaussian', base_file_loc=data_dir / 'g16_h2o_HF_STO3G.out')

G16 = DataOriginInfo(source_type='gaussian', base_file_loc='g16.out')
C4 = DataOriginInfo(source_type='cfour', base_file_loc='c4_folder')


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
    """parse_from_source replaced by a fake that records its calls and gives each name its origin's file."""
    made = []

    def fake(requested_data, **origin):
        made.append((list(requested_data), origin))
        return {name: origin['base_file_loc'] for name in requested_data}

    monkeypatch.setattr('CQCParse.parsing.parse_from_source', fake)
    return made


def test_one_call_per_origin(calls):
    """The names of one origin go to parse_from_source together; an equal DataOriginInfo is the same origin."""
    same_as_g16 = DataOriginInfo(source_type='gaussian', base_file_loc='g16.out')
    data = wilson_data_obtainer({'cff': G16, 'dipgrad': C4, 'qff': same_as_g16})

    assert [names for names, _ in calls] == [['cff', 'qff'], ['dipgrad']]
    assert data == {'cff': 'g16.out', 'qff': 'g16.out', 'dipgrad': 'c4_folder'}


def test_origin_fields_are_the_arguments(calls):
    """Every DataOriginInfo field goes to parse_from_source as the argument of the same name."""
    wilson_data_obtainer({'cff': G16})

    assert calls[0][1] == dataclasses.asdict(G16)


def test_parse_from_source_takes_every_origin_field():
    """A DataOriginInfo field that parse_from_source does not take would make every load fail."""
    from CQCParse.parsing import parse_from_source

    fields = {f.name for f in dataclasses.fields(DataOriginInfo)}
    assert fields <= set(inspect.signature(parse_from_source).parameters)


@pytest.mark.parametrize('requested', [['cff'], {'cff': 'g16.out'}])
def test_request_maps_names_to_origins(requested):
    with pytest.raises(TypeError):
        wilson_data_obtainer(requested)


@pytest.mark.parametrize('source_type, error', [('wilson', NotImplementedError), ('orca', ValueError)])
def test_source_types(calls, source_type, error):
    """Only Gaussian and CFOUR go to CQCParse: 'wilson' is not written yet, anything else is unknown."""
    with pytest.raises(error):
        wilson_data_obtainer({'cff': DataOriginInfo(source_type=source_type)})
    assert not calls
