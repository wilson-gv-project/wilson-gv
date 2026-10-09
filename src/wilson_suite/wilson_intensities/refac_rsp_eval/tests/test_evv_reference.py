"""
Term-level evaluation of the 14 EVV terms, against evv_reference.json (written by make_evv_reference.py).
For every pair (term, fixed resonance labels): the coefficient and the resonance location,
and which pairs are zero. Formaldehyde, water and CO; anharmonic and harmonic states.
Features and the spectrum are not checked here (C3).

C2, checked 2026-10-07: these numbers equal the old pipeline's (relative difference below 1e-12), for all
3 molecules, GVPT2 (anharmonic states) and harmonic regime. Old side: EvaluationWorkflow set up as in
test_full_integration_EVV_axes. Formaldehyde, GVPT2:
    - 2664 full label sets, factor by factor (NON_AVRG, AVRG, VIBDIFF_TERMS, VIBENE_DENOM):
      old evaluate_single_index_dict = new evaluate_full_index_dict
    - 504 coefficients per (term, fixed labels): old evaluate_term_coeffs = new evaluate_term_coeff_sumover
    - 504 locations: old resonances.solve_LSE_motif = new solve_LSE_motif
On these 14 terms the old and new pipelines do not differ at all.
They differ only for motifs none of the 14 terms has:
    - one condition, on B only: the old code raises IndexError, the new one gives the location (test_evaluate.py).
    - conditions that contradict each other: the old code prints and returns None; the new compile_terms raises
      (B7, test_plan.py).
    - one condition on A + B (a line, not a point): the old code raises ValueError; the new compile_terms raises
      (B7, test_plan.py).

A deliberate change of numbers: rerun make_evv_reference.py, check the diff, name the change above.
"""

import json

import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval.tests import make_evv_reference as ref

REFERENCE = json.loads(ref.REFERENCE_FILE.read_text())
CASES = [(b['molecule'], b['states']) for b in REFERENCE['results']]

# same code, same data: only float rounding may differ between machines
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
             if pair in now_by_pair and now_by_pair[pair]['coeff'] != pytest.approx(e['coeff'], rel=REL)]
    assert wrong == [], '(pair, stored, now)'


def test_locations_match_the_reference(stored_and_now):
    stored, now = stored_and_now

    now_by_pair = _by_pair(now['rows'])
    wrong = [(pair, e['location'], now_by_pair[pair]['location'])
             for pair, e in _by_pair(stored['rows']).items()
             if pair in now_by_pair and now_by_pair[pair]['location'] != pytest.approx(e['location'], rel=REL, abs=1e-9)]
    assert wrong == [], '(pair, stored, now), cm-1'
