"""
Term-level evaluation of the 14 EVV terms, against evv_reference.json (written by make_evv_reference.py).
For every pair (term, fixed resonance labels): the coefficient and the resonance location,
and which pairs are zero. Formaldehyde, water and CO; anharmonic and harmonic states.
Features and the spectrum are not checked here.

A deliberate change of numbers: rerun make_evv_reference.py and check the diff.
"""

import json

import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval.tests import make_evv_reference as ref
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.stored_terms import load_setup

REFERENCE = json.loads(ref.REFERENCE_FILE.read_text())
CASES = [(b['molecule'], b['states']) for b in REFERENCE['results']]

# same code, same data: only float rounding may differ between machines.
# Coefficients need abs=0: pytest.approx's default abs=1e-12 would allow up to 6e-5 relative on the smallest
# stored coefficient (1.6e-8). Locations (cm-1, some 0.) keep a small abs.
REL = 1e-12


def _by_pair(entries: list[dict]) -> dict:
    """{(term, params): entry}"""
    return {(e['term'], tuple(sorted(e['params'].items()))): e for e in entries}


@pytest.fixture(scope='module')
def compiled():
    return ref.compiled_evv_terms()


@pytest.fixture(scope='module', params=CASES, ids=[f'{m}-{s}' for m, s in CASES])
def stored_and_now(request, compiled) -> tuple[dict, dict]:
    """(stored block, block computed now) for one molecule and one states choice"""
    molecule, states = request.param
    stored = next(b for b in REFERENCE['results'] if (b['molecule'], b['states']) == request.param)
    return stored, ref.compute_block(compiled, molecule, states)


def test_reference_file_has_the_settings_of_the_script():
    assert REFERENCE['settings'] == {'terms_file': ref.TERMS_FILE, 'axes': ref.AXES,
                                     'laser_pol': list(ref.LASER_POL), 'rank': ref.RANK}
    assert sorted(CASES) == sorted((m, s) for m in ref.DATA_FILES for s in ref.STATES_CHOICES)


def test_terms_file_has_the_axes_and_laser_pol_of_the_script():
    """evv.json records its translation: A = [1], B = [-1, 2] as {label: [pulse ids of each variable]}"""
    setup = load_setup('evv')

    assert setup.axes == {label: [(pulse_id,) for pulse_id in ids] for label, ids in ref.AXES.items()}
    assert setup.laser_pol == ref.LASER_POL
    assert len(setup.terms) == 14


def test_formaldehyde_has_504_pairs_340_rows_164_zero():
    for states in ref.STATES_CHOICES:
        (block,) = [b for b in REFERENCE['results'] if (b['molecule'], b['states']) == ('formaldehyde', states)]
        assert block['counts'] == {'rows': 340, 'zero': 164}


def test_same_pairs_are_rows_and_zero(stored_and_now):
    stored, now = stored_and_now

    assert now['n_modes'] == stored['n_modes']
    assert now['counts'] == stored['counts']
    for kind in ('rows', 'zero'):
        assert _by_pair(now[kind]).keys() == _by_pair(stored[kind]).keys(), kind


def test_coefficients_match_the_reference(stored_and_now):
    stored, now = stored_and_now

    now_by_pair = _by_pair(now['rows'])
    wrong = [(pair, e['coeff'], now_by_pair[pair]['coeff'])
             for pair, e in _by_pair(stored['rows']).items()
             if pair in now_by_pair and now_by_pair[pair]['coeff'] != pytest.approx(e['coeff'], rel=REL, abs=0)]
    assert wrong == [], '(pair, stored, now)'


def test_locations_match_the_reference(stored_and_now):
    stored, now = stored_and_now

    now_by_pair = _by_pair(now['rows'])
    wrong = [(pair, e['location'], now_by_pair[pair]['location'])
             for pair, e in _by_pair(stored['rows']).items()
             if pair in now_by_pair and now_by_pair[pair]['location'] != pytest.approx(e['location'], rel=REL, abs=1e-9)]
    assert wrong == [], '(pair, stored, now), cm-1'
