"""
data_obtainer groups the requested names by their DataOriginInfo, calls CQCParse's parse_from_source once per
origin, and merges the results. CQCParse tests its own readers; the tests here cover what the obtainer adds, some
on real files. Which names the old workflow asks for (WilsonSimulation.requestData) is tested in
wilson_main/tests/test_workflow_abstractions.py.
"""
from importlib.resources import files

import pytest

from wilson_suite.wilson_main.abstractions import DataOriginInfo
from wilson_suite.wilson_system.data_obtainer import wilson_data_obtainer

data_dir = files('wilson_suite').joinpath('data_for_tests')
WATER = DataOriginInfo(source_type='gaussian', base_file_loc=str(data_dir / 'g16_h2o_HF_STO3G.out'))
FORMALDEHYDE_G16 = DataOriginInfo(source_type='gaussian',
                                  base_file_loc=str(data_dir / 'g16_formaldehyde_B3LYPcc_pVQZ.out'))
FORMALDEHYDE_C4 = DataOriginInfo(source_type='cfour', base_file_loc=str(data_dir / 'c4_formaldehyde_HFcc_pVQZ'))


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


@pytest.mark.parametrize('g16_first', [True, False])
def test_merge_of_two_origins(g16_first):
    """
    The merge of the reads of two origins. One molecule, formaldehyde, from two programs:
    Gaussian at B3LYP/cc-pVQZ and CFOUR at HF/cc-pVQZ. The two methods give different
    values, so a value shows which read it comes from.

    A read returns more names than requested:
        nc_sqrt_eigval   also harmonic_states
        cff              also cff_rc
        any name         also reindex_modes
    The request: nc_sqrt_eigval and cff from Gaussian, harmonic_states from CFOUR. So both
    reads return harmonic_states: CFOUR as requested, Gaussian as an extra.
    What the result holds:
        harmonic_states  the CFOUR value: it was requested from CFOUR, so the Gaussian
                         extra does not replace it
        cff_rc           the Gaussian value: only the Gaussian read returns it, and no
                         origin was asked for it
        reindex_modes    left out: both reads return it. The result holds one value per
                         name, and the reads give two: True from Gaussian (the read
                         renumbers its modes by rising frequency), False from CFOUR (its
                         modes are in that order already). So the value only repeats the
                         source type of its read.
    The reads run in the order of the request; both orders give the same result.
    """
    from_g16 = {'nc_sqrt_eigval': FORMALDEHYDE_G16, 'cff': FORMALDEHYDE_G16}
    from_c4 = {'harmonic_states': FORMALDEHYDE_C4}
    request = {**from_g16, **from_c4} if g16_first else {**from_c4, **from_g16}

    data = wilson_data_obtainer(request)
    g16, c4 = wilson_data_obtainer(from_g16), wilson_data_obtainer(from_c4)    # each origin alone

    assert set(data) == {'nc_sqrt_eigval', 'cff', 'cff_rc', 'harmonic_states'}
    assert data['harmonic_states'] == c4['harmonic_states'] != g16['harmonic_states']


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
    """A source type the obtainer does not know stops with an error that names it, before any file is read."""
    known = DataOriginInfo(source_type='gaussian', base_file_loc='g16.out')
    unknown = DataOriginInfo(source_type='orca')

    with pytest.raises(ValueError, match="'orca'"):
        wilson_data_obtainer({'cff': known, 'dipgrad': unknown})
    assert not calls
