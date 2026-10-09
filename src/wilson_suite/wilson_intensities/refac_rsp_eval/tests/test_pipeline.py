"""
pipeline.py — compute_features end to end on the toy molecule (helpers.py, conftest.py),
then load_molsys_data + compute_features on real data (14 EVV terms + formaldehyde).

The steps have their own tests (build_contributions in test_evaluate.py, features_from_rows in
test_features.py). Here: the steps are joined right, and FeatureResult keeps the table as its source.

The toy term fixes a, b and sums over c. <polgrad>[1] = 0 (pre_a1_zero), so index sets with a=1
have coeff 0: per term, 2 rows (a=0) at 2 different locations and 2 zero pairs (a=1).
"""

import dataclasses
import json
from collections import defaultdict
from dataclasses import dataclass, replace

import pytest

from wilson_suite.wilson_intensities.amplitudes.averaging import (
    getGeneralPolarizationAveragingExpression,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    build_contributions,
    evaluate_term_coeff_sumover,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    ContributionTable,
    SpectralFeature,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import (
    FeatureResult,
    averaging_rank,
    compute_features,
    compute_features_from_terms,
    load_molsys_data,
    make_polarization_linear_comb,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    PropsCollection,
    ResonanceMotif,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests import (
    make_evv_reference as ref,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
    E0,
    E00,
    E01,
    E1,
    MOTIF_AB,
    PS_00,
    PS_01,
    PS_10,
    PS_11,
    ab_term,
    polprop,
    toy_term,
)
from wilson_suite.wilson_utils.builders import make_SpectralAxisSet


def two_terms_one_motif():
    """Same motif; frac_factor 0.5 and 2., so every coefficient of term 1 is 4x that of term 0."""
    term = ab_term()
    return [term, replace(term, frac_factor=2.)]


## compute_features ---------------------------------------------------------

def test_compute_features_one_feature_per_location_holding_both_terms(molsys, pre_a1_zero):
    result = compute_features(two_terms_one_motif(), molsys, precalculated_data=pre_a1_zero)

    f00, f01 = result.features
    assert f00.location.as_dict() == pytest.approx({'A': E00 - E0, 'B': 0.})
    assert f01.location.as_dict() == pytest.approx({'A': E01 - E0, 'B': E1 - E0})
    for f, ps in ((f00, PS_00), (f01, PS_01)):
        assert f.motif == ResonanceMotif.from_tuples(MOTIF_AB)
        assert f.param_sets == (ps,)
        assert f.term_ids == (0, 1)


def test_compute_features_amplitude_adds_both_terms(molsys, pre_a1_zero):
    terms = two_terms_one_motif()

    result = compute_features(terms, molsys, precalculated_data=pre_a1_zero)

    for f, idx in zip(result.features, ({'a': 0, 'b': 0}, {'a': 0, 'b': 1})):
        single, _ = evaluate_term_coeff_sumover(terms[0], idx, molsys, precalculated_data=pre_a1_zero)
        assert single != 0.
        # 1x (term 0) + 4x (term 1) -- 0.5 and 2.0 frac_factor, so 5x single
        assert f.amplitude_coeff == pytest.approx(5 * single)


def test_compute_features_reports_zero_pairs(molsys, pre_a1_zero):
    result = compute_features(two_terms_one_motif(), molsys, precalculated_data=pre_a1_zero)

    assert result.zero == [(0, PS_10), (0, PS_11), (1, PS_10), (1, PS_11)]


def test_compute_features_gives_every_feature_the_lineshape_parameter(molsys, pre_a1_zero):
    result = compute_features([ab_term()], molsys, precalculated_data=pre_a1_zero, lineshape_parameter=5.)

    assert len(result.features) == 2
    assert all(f.lineshape_parameter == 5. for f in result.features)
    # no boxes yet: they come later, from dress_these_with_boxes
    assert all(f.feat_box is None for f in result.features)


## FeatureResult ------------------------------------------------------------

def test_feature_result_table_is_what_build_contributions_returns(molsys, pre_a1_zero):
    terms = two_terms_one_motif()
    
    # compute_features calls build_contributions, so this is a bit redundant, but it checks that the table is preserved
    result = compute_features(terms, molsys, precalculated_data=pre_a1_zero)
    table, _ = build_contributions(terms, molsys, precalculated_data=pre_a1_zero)

    assert list(result.table) == list(table)


def test_feature_result_features_hold_every_row_of_the_table_once(molsys, pre_a1_zero):
    result = compute_features(two_terms_one_motif(), molsys, precalculated_data=pre_a1_zero)

    rows_in_features = [r for f in result.features for r in f.rows]

    assert len(rows_in_features) == len(result.table) == 4
    assert set(rows_in_features) == set(result.table)


def test_feature_result_builds_features_once(molsys, pre_a1_zero):
    result = compute_features([ab_term()], molsys, precalculated_data=pre_a1_zero)

    assert result.features is result.features


def test_feature_result_is_frozen(molsys, pre_a1_zero):
    result = compute_features([ab_term()], molsys, precalculated_data=pre_a1_zero)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.table = ContributionTable([])  # type: ignore


## polarization recipe: averaging_rank, make_polarization_linear_comb ----------
# The rank counts the Cartesian slots (op.o) of a term's averaged part.
# toy_term(): <polgrad> with slots 0, 1 -> rank 2.

def term_with_slots(*ops: tuple[int, ...]) -> CompiledTerm:
    """toy_term() whose averaged part has one property per ops tuple"""
    return replace(toy_term(), avrg_props=PropsCollection([polprop(ops=o, inds='a') for o in ops]))


RANK_4_TERM = term_with_slots((0, 3), (1,), (2,))      # like EVV term 0: polhess[0,3] dipgrad[1] dipgrad[2]


def test_averaging_rank_counts_the_slots_of_the_averaged_part():
    assert averaging_rank([toy_term()]) == 2
    assert averaging_rank([RANK_4_TERM]) == 4


def test_averaging_rank_of_terms_with_the_same_slots():
    assert averaging_rank([toy_term(), replace(toy_term(), frac_factor=2.)]) == 2


def test_averaging_rank_raises_if_terms_differ():
    """compute_features uses one recipe for all terms: rank 2 and rank 4 cannot share it"""
    with pytest.raises(ValueError, match='terms differ'):
        averaging_rank([toy_term(), RANK_4_TERM])


def test_averaging_rank_raises_if_a_slot_is_missing():
    """slots 0, 2: the recipe puts slot i at position i, and a rank-2 recipe has no position 2"""
    with pytest.raises(ValueError, match=r'0 \.\. 1'):
        averaging_rank([term_with_slots((0, 2))])


def test_averaging_rank_raises_without_terms():
    with pytest.raises(ValueError, match='no terms'):
        averaging_rank([])


@pytest.mark.parametrize('term, rank, laser_pol', [(toy_term(), 2, (1.,)), (RANK_4_TERM, 4, (1., 0., 1.))],
                         ids=['rank 2', 'rank 4'])
def test_make_polarization_linear_comb_uses_the_rank_of_the_terms(term, rank, laser_pol):
    expected = getGeneralPolarizationAveragingExpression(rank=rank, laser_pol=laser_pol)

    assert make_polarization_linear_comb([term], laser_pol) == expected


@pytest.mark.parametrize('laser_pol', [(1., 1.), (1., 1., 1., 1.)], ids=['too short', 'too long'])
def test_make_polarization_linear_comb_raises_if_laser_pol_does_not_fit_the_rank(laser_pol):
    """rank 4 needs 3 numbers. getGeneralPolarizationAveragingExpression alone ignores a 4th number without an error."""
    with pytest.raises(ValueError, match='rank 4 needs laser_pol with 3 numbers'):
        make_polarization_linear_comb([RANK_4_TERM], laser_pol)


def test_make_polarization_linear_comb_raises_for_rank_above_6():
    with pytest.raises(ValueError, match='rank 2 to 6'):
        make_polarization_linear_comb([term_with_slots(tuple(range(7)))], (1.,))


## real data: 14 EVV terms + formaldehyde -------------------------------------
# load_molsys_data -> compute_features on the real data file; the settings and the stored rows are
# the ones of evv_reference.json (make_evv_reference.py).
# The 14 terms have 2 motifs: terms 0, 2-7 and terms 1, 8-13. Both fix a and b: 6 modes -> 36 (a, b)
# pairs per motif. Every pair has its own location, so here one feature = one motif at one (a, b),
# holding that pair's rows from all terms of the motif.

EVV_MOTIF_TERMS = ((0, 2, 3, 4, 5, 6, 7), (1, 8, 9, 10, 11, 12, 13))
ALL_AB = {(a, b) for a in range(6) for b in range(6)}


@dataclass(frozen=True)
class EvvRun:
    states: str
    compiled: list[CompiledTerm]
    result: FeatureResult
    stored: dict            # the evv_reference.json block for formaldehyde and these states


@pytest.fixture(scope='module', params=ref.STATES_CHOICES)
def evv_formaldehyde(request) -> EvvRun:
    compiled = ref.compiled_evv_terms()
    molsys = load_molsys_data(compiled, ref.data_origin('formaldehyde'), states_choice=request.param)
    polarization = make_polarization_linear_comb(compiled, ref.LASER_POL)

    stored = next(b for b in json.loads(ref.REFERENCE_FILE.read_text())['results']
                  if (b['molecule'], b['states']) == ('formaldehyde', request.param))
    return EvvRun(request.param, compiled, compute_features(compiled, molsys, polarization), stored)


def _ab(feature: SpectralFeature) -> tuple[int, int]:
    """(a, b) of a feature that holds one index set"""
    (ps,) = feature.param_sets
    return ps['a'], ps['b']


def _motif_terms(feature: SpectralFeature) -> tuple[int, ...]:
    """the EVV_MOTIF_TERMS group that the feature's terms belong to"""
    (group,) = [g for g in EVV_MOTIF_TERMS if set(feature.term_ids) <= set(g)]
    return group


def test_real_data_the_14_terms_have_2_motifs(evv_formaldehyde):
    compiled = evv_formaldehyde.compiled

    motifs = dict.fromkeys(t.cmp_resmotf for t in compiled)
    assert tuple(tuple(i for i, t in enumerate(compiled) if t.cmp_resmotf == m) for m in motifs) == EVV_MOTIF_TERMS


def test_real_data_averaging_rank_is_the_one_stored_in_the_reference_file(evv_formaldehyde):
    assert averaging_rank(evv_formaldehyde.compiled) == ref.RANK == 4


def test_real_data_504_pairs_340_rows_164_zero(evv_formaldehyde):
    result = evv_formaldehyde.result

    assert (len(result.table), len(result.zero)) == (340, 164)


def test_real_data_68_features_36_and_32_per_motif(evv_formaldehyde):
    """Motif of terms 1, 8-13 has no feature at (3, 4), (3, 5), (4, 3), (5, 3): all its 7 terms give coeff 0 there."""
    features = evv_formaldehyde.result.features

    ab_per_motif = defaultdict(set)
    for f in features:
        ab_per_motif[_motif_terms(f)].add(_ab(f))

    assert len(features) == 68
    assert ab_per_motif[EVV_MOTIF_TERMS[0]] == ALL_AB
    assert ALL_AB - ab_per_motif[EVV_MOTIF_TERMS[1]] == {(3, 4), (3, 5), (4, 3), (5, 3)}


def test_real_data_each_feature_holds_the_stored_rows_of_its_motif_and_index_set(evv_formaldehyde):
    """features_from_rows groups by (motif, location); the expected groups here come from (motif, params)"""
    compiled, result, stored = evv_formaldehyde.compiled, evv_formaldehyde.result, evv_formaldehyde.stored

    expected = defaultdict(set)
    for r in stored['rows']:
        expected[(compiled[r['term']].cmp_resmotf, (r['params']['a'], r['params']['b']))].add(r['term'])

    assert {(f.motif, _ab(f)): set(f.term_ids) for f in result.features} == dict(expected)
    assert sum(len(f.rows) for f in result.features) == len(stored['rows'])     # each term once per feature


def test_real_data_amplitude_is_the_sum_of_the_stored_coefficients(evv_formaldehyde):
    result, stored = evv_formaldehyde.result, evv_formaldehyde.stored
    stored_coeff = {(r['term'], (r['params']['a'], r['params']['b'])): r['coeff'] for r in stored['rows']}

    for f in result.features:
        expected = sum(stored_coeff[(t, _ab(f))] for t in f.term_ids)
        assert f.amplitude_coeff == pytest.approx(expected, rel=1e-12), f


# states choice: location in cm-1, amplitude in au
STRONGEST = {'anharmonic': ({'A': 2682.765, 'B': 0.}, -5.0513e-4),
             'harmonic':   ({'A': 2933.526, 'B': 0.}, -4.7346e-4)}


def test_real_data_strongest_feature(evv_formaldehyde):
    """
    a = b = 4, motif of terms 0, 2-7; terms 3 and 6 add to term 0, the other 4 terms give coeff 0.
    A = E(1_4) - E(0): Gaussian mode 5 in the .out file, 2682.765 (anharmonic fundamental) or
    2933.526 (harmonic) cm-1. B = E(1_b) - E(1_a) = 0, since a = b.
    """
    location, amplitude = STRONGEST[evv_formaldehyde.states]

    strongest = max(evv_formaldehyde.result.features, key=lambda f: abs(f.amplitude_coeff))

    assert _ab(strongest) == (4, 4)
    assert strongest.term_ids == (0, 3, 6)
    assert strongest.location.as_dict() == pytest.approx(location)
    assert strongest.amplitude_coeff == pytest.approx(amplitude, rel=1e-4)


## compute_features_from_terms: the one top function ---------------------------
# Same settings as evv_reference.json, formaldehyde, anharmonic states. The evv_formaldehyde fixture
# runs the same steps one by one, so equal results mean the top function passes every setting to
# the right step. One run (top_run) serves all tests but the reverse-order one.

def from_terms(terms, **kwargs) -> FeatureResult:
    return compute_features_from_terms(terms, axes=make_SpectralAxisSet(ref.AXES),  # type: ignore
                                       data_origin=ref.data_origin('formaldehyde'),
                                       states_choice='anharmonic', laser_pol=ref.LASER_POL, **kwargs)


@dataclass(frozen=True)
class TopRun:
    terms: list             # the input terms, after the run
    terms_before: list      # their to_str, before the run
    result: FeatureResult


@pytest.fixture(scope='module')
def top_run() -> TopRun:
    terms = ref.evv_terms()
    before = [t.to_str() for t in terms]
    return TopRun(terms, before, from_terms(terms, lineshape_parameter=4.7))


@pytest.mark.parametrize('evv_formaldehyde', ['anharmonic'], indirect=True)
def test_compute_features_from_terms_equals_the_steps_one_by_one(top_run, evv_formaldehyde):
    result, by_steps = top_run.result, evv_formaldehyde.result

    assert list(result.table) == list(by_steps.table)
    assert result.zero == by_steps.zero


def test_compute_features_from_terms_term_id_is_the_index_in_the_input_list(top_run):
    """terms in reverse order: the row of term i comes back with term_id 13 - i, same params and coeff"""
    backward = from_terms(ref.evv_terms()[::-1])

    last = len(top_run.terms) - 1
    assert ({(last - r.term_id, r.params, r.coeff) for r in backward.table}
            == {(r.term_id, r.params, r.coeff) for r in top_run.result.table})


def test_compute_features_from_terms_leaves_the_input_terms_as_they_are(top_run):
    """the translation to axes works on copies: the caller's terms keep their pulse IDs"""
    assert [t.to_str() for t in top_run.terms] == top_run.terms_before


def test_compute_features_from_terms_gives_every_feature_the_lineshape_parameter(top_run):
    assert all(f.lineshape_parameter == 4.7 for f in top_run.result.features)
