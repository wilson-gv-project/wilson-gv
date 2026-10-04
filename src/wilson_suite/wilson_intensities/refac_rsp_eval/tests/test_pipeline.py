"""
pipeline.py — compute_features end to end on the toy molecule (helpers.py, conftest.py).

The steps have their own tests (build_contributions in test_evaluate.py, features_from_rows in
test_features.py). Here: the steps are joined right, and FeatureResult keeps the table as its source.

The toy term fixes a, b and sums over c. <polgrad>[1] = 0 (pre_a1_zero), so index sets with a=1
have coeff 0: per term, 2 rows (a=0) at 2 different locations and 2 zero pairs (a=1).
"""

import dataclasses
from dataclasses import replace

import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    build_contributions,
    evaluate_term_coeff_sumover,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import ContributionTable
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import compute_features
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import ResonanceMotif
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
    E0,
    E00,
    E01,
    E1,
    MOTIF_AB,
    MOTIF_B_TWICE,
    PS_00,
    PS_01,
    PS_10,
    PS_11,
    ab_term,
)


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
    assert result.failed == []


def test_compute_features_without_a_single_point_gives_failed_pairs_and_no_features(molsys, pre_a1_zero):
    """MOTIF_B_TWICE asks w_B = -E_a and w_B = E_b - E_a at once: no point for any a=0 pair."""
    result = compute_features([ab_term(MOTIF_B_TWICE)], molsys, precalculated_data=pre_a1_zero)

    assert result.features == []
    assert [(term_id, ps) for term_id, ps, _ in result.failed] == [(0, PS_00), (0, PS_01)]


def test_compute_features_gives_every_feature_the_lineshape_parameter(molsys, pre_a1_zero):
    result = compute_features([ab_term()], molsys, precalculated_data=pre_a1_zero, lineshape_parameter=5.)

    assert len(result.features) == 2
    assert all(f.lineshape_parameter == 5. and f.feat_box is not None for f in result.features)
    # boxes are built from the lineshape parameter when feature is initialized, so they should be present
    assert all(f.feat_box is not None for f in result.features)


## FeatureResult ------------------------------------------------------------

def test_feature_result_table_is_what_build_contributions_returns(molsys, pre_a1_zero):
    terms = two_terms_one_motif()
    
    # compute_features calls build_contributions, so this is a bit redundant, but it checks that the table is preserved
    result = compute_features(terms, molsys, precalculated_data=pre_a1_zero)
    table, _, _ = build_contributions(terms, molsys, precalculated_data=pre_a1_zero)

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
