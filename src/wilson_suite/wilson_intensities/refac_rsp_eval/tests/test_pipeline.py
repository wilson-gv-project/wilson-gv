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

import numpy as np
import pytest

import wilson_suite.wilson_intensities.refac_rsp_eval.pipeline as pipeline_mod
from wilson_suite.wilson_intensities.amplitudes.averaging import (
    getGeneralPolarizationAveragingExpression,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    build_contributions,
    draw_all,
    evaluate_term_coeff_sumover,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    ContributionTable,
    SpectralFeature,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import Box
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import (
    FeatureResult,
    averaging_rank,
    compute_features,
    compute_features_from_terms,
    draw_spectrum,
    evaluate_response_terms,
    load_molsys_data,
    make_polarization_linear_comb,
    prop_names,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    ParameterSet,
    PropsCollection,
    ResLocPoint,
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
    row,
    toy_term,
)
from wilson_suite.wilson_system.system_data import MolecularProperty, MolPropsCollection
from wilson_suite.wilson_utils.builders import make_SpectralAxisSet


def two_terms_one_motif():
    """Same motif; frac_factor 0.5 and 2., so every coefficient of term 1 is 4x that of term 0."""
    term = ab_term()
    return [term, replace(term, frac_factor=2.)]


## compute_features ---------------------------------------------------------

def test_compute_features_one_feature_per_location_holding_both_terms(molsys, pre_a1_zero):
    result = compute_features(two_terms_one_motif(), molsys, precalculated_data=pre_a1_zero, states_choice='harmonic')

    f00, f01 = result.features
    assert f00.location.as_dict() == pytest.approx({'A': E00 - E0, 'B': 0.})
    assert f01.location.as_dict() == pytest.approx({'A': E01 - E0, 'B': E1 - E0})
    for f, ps in ((f00, PS_00), (f01, PS_01)):
        assert f.motif == ResonanceMotif.from_tuples(MOTIF_AB)
        assert f.param_sets == (ps,)
        assert f.term_ids == (0, 1)


def test_compute_features_amplitude_adds_both_terms(molsys, pre_a1_zero):
    terms = two_terms_one_motif()

    result = compute_features(terms, molsys, precalculated_data=pre_a1_zero, states_choice='harmonic')

    for f, idx in zip(result.features, ({'a': 0, 'b': 0}, {'a': 0, 'b': 1})):
        single, _ = evaluate_term_coeff_sumover(terms[0], idx, molsys, precalculated_data=pre_a1_zero,
                                                states_choice='harmonic')
        assert single != 0.
        # 1x (term 0) + 4x (term 1) -- 0.5 and 2.0 frac_factor, so 5x single
        assert f.amplitude_coeff == pytest.approx(5 * single)


def test_compute_features_reports_zero_pairs(molsys, pre_a1_zero):
    result = compute_features(two_terms_one_motif(), molsys, precalculated_data=pre_a1_zero, states_choice='harmonic')

    assert result.zero == [(0, PS_10), (0, PS_11), (1, PS_10), (1, PS_11)]


def test_compute_features_gives_every_feature_the_propagator_param(molsys, pre_a1_zero):
    result = compute_features([ab_term()], molsys, precalculated_data=pre_a1_zero, propagator_param=5.,
                              states_choice='harmonic')

    assert len(result.features) == 2
    assert all(f.propagator_param == 5. for f in result.features)
    # no boxes yet: they come later, from dress_these_with_boxes
    assert all(f.feat_box is None for f in result.features)


def test_compute_features_gives_a_propagator_param_function_the_conditions(molsys, pre_a1_zero):
    """
    The function gets the pair of states of each condition, from the result's states. ab_term (MOTIF_AB), a=0:
    conditions <ab - a> and <b - a>, so with width 5 + 1 % of |E|:
        b=0:  E00 - E0 = 950,   E0 - E0 = 0    ->  14.5,  5 cm-1
        b=1:  E01 - E0 = 1600,  E1 - E0 = 500  ->  21,   10 cm-1
    """
    width = lambda vd, freqs: 5. + 0.01 * abs(vd.energy_difference())

    result = compute_features([ab_term()], molsys, precalculated_data=pre_a1_zero, propagator_param=width,
                              states_choice='harmonic')

    assert all(f.propagator_param is width for f in result.features)
    assert [f.peak_widths() for f in result.features] == [pytest.approx((14.5, 5.)), pytest.approx((21., 10.))]


def test_feature_result_made_by_hand_gives_a_propagator_param_function_no_conditions():
    """No states to take them from: the feature has no conditions, and its box and peak height raise."""
    result = FeatureResult(ContributionTable([row(0.3, a=0)]), zero=[], propagator_param=lambda vd, freqs: 5.)

    [f] = result.features

    assert f.conditions is None
    with pytest.raises(ValueError, match='no conditions'):
        f.get_intensity()


## FeatureResult ------------------------------------------------------------

def test_feature_result_table_is_what_build_contributions_returns(molsys, pre_a1_zero):
    terms = two_terms_one_motif()
    
    # compute_features calls build_contributions, so this is a bit redundant, but it checks that the table is preserved
    result = compute_features(terms, molsys, precalculated_data=pre_a1_zero, states_choice='harmonic')
    table, _ = build_contributions(terms, molsys, precalculated_data=pre_a1_zero, states_choice='harmonic')

    assert list(result.table) == list(table)


def test_feature_result_features_hold_every_row_of_the_table_once(molsys, pre_a1_zero):
    result = compute_features(two_terms_one_motif(), molsys, precalculated_data=pre_a1_zero, states_choice='harmonic')

    rows_in_features = [r for f in result.features for r in f.rows]

    assert len(rows_in_features) == len(result.table) == 4
    assert set(rows_in_features) == set(result.table)


def test_feature_result_builds_features_once(molsys, pre_a1_zero):
    result = compute_features([ab_term()], molsys, precalculated_data=pre_a1_zero, states_choice='harmonic')

    assert result.features is result.features


def test_feature_result_is_frozen(molsys, pre_a1_zero):
    result = compute_features([ab_term()], molsys, precalculated_data=pre_a1_zero, states_choice='harmonic')

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.table = ContributionTable([])  # type: ignore


def leaf_value(term: CompiledTerm, contribs: dict) -> float:
    """
    One leaf's part of the row coefficient, from its factors, as evaluate_full_index_dict multiplies them.
    A leaf that stopped early (average or non-averaged part 0) has fewer factors and gives 0.
    """
    if 'VIBENE_DENOM' not in contribs:
        return 0.
    return term.frac_factor * (contribs['NON_AVRG'] * contribs['AVRG'] * contribs['VIBDIFF_TERMS']
                               * contribs['VIBENE_DENOM'])


def test_feature_result_leaves_of_a_row_are_its_summed_index_sets(molsys, pre_a1_zero):
    """ab_term fixes a, b and sums c over 2 modes: 2 leaves per row, the same as evaluate_term_coeff_sumover gives"""
    terms = two_terms_one_motif()
    result = compute_features(terms, molsys, precalculated_data=pre_a1_zero, states_choice='harmonic')

    for r in result.table:
        leaves = result.leaves(r)

        _, expected = evaluate_term_coeff_sumover(terms[r.term_id], r.params.to_dict(), molsys,
                                                  precalculated_data=pre_a1_zero, states_choice='harmonic')
        assert set(leaves) == {ParameterSet({'a': r.params['a'], 'b': r.params['b'], 'c': c}) for c in (0, 1)}
        assert leaves == expected


def test_feature_result_leaves_add_up_to_the_row_coefficient(molsys, pre_a1_zero):
    terms = two_terms_one_motif()
    result = compute_features(terms, molsys, precalculated_data=pre_a1_zero, states_choice='harmonic')

    for r in result.table:
        total = sum(leaf_value(terms[r.term_id], contribs) for contribs in result.leaves(r).values())
        assert total == pytest.approx(r.coeff, rel=1e-12, abs=0)


def test_feature_result_leaves_use_the_polarization_of_the_result(molsys):
    """
    On the fly, no precalculated data: the leaves need the result's own recipe (without it, the average raises).
    ab_term's <polgrad>[a] has slots 0, 1: rank 2. polgrad of mode 1 is 0, so a = 1 stops at the average,
    as with pre_a1_zero: the toy states have no '1,1'.
    """
    terms = [ab_term()]
    polgrad = np.zeros((2, 3, 3))
    polgrad[0] = np.arange(9.).reshape(3, 3) + 1.
    otf_molsys = replace(molsys, mol_props=MolPropsCollection(
        [*molsys.mol_props, MolecularProperty(trivial_name='polgrad', vals=polgrad, extra_data={})]))
    recipe = getGeneralPolarizationAveragingExpression(rank=2, laser_pol=(1.,))

    result = compute_features(terms, otf_molsys, recipe, states_choice='harmonic')

    assert result.polarization_linear_comb is recipe
    assert len(result.table) == 2                       # a = 0, b = 0 and 1
    for r in result.table:
        total = sum(leaf_value(terms[0], contribs) for contribs in result.leaves(r).values())
        assert total == pytest.approx(r.coeff, rel=1e-12, abs=0) != 0.


def test_feature_result_keeps_its_inputs_without_copying(molsys, pre_a1_zero):
    terms = two_terms_one_motif()

    result = compute_features(terms, molsys, precalculated_data=pre_a1_zero, states_choice='harmonic')

    assert result.compiled == tuple(terms) and all(a is b for a, b in zip(result.compiled, terms))
    assert result.molsys is molsys
    assert result.precalculated_data is pre_a1_zero
    assert result.polarization_linear_comb is None


def test_feature_result_inputs_do_not_count_for_equality():
    """two results with the same rows are equal, whatever they were computed from"""
    table = ContributionTable([row(0.3, a=0)])

    assert FeatureResult(table, zero=[]) == FeatureResult(table, zero=[], compiled=(toy_term(),))


def test_feature_result_made_by_hand_has_no_leaves():
    r = row(0.3, a=0)

    with pytest.raises(ValueError, match='no terms or data'):
        FeatureResult(ContributionTable([r]), zero=[]).leaves(r)


def two_rows_at(a_cm: float, c0: float, c1: float):
    """Terms 0 and 1 at one index set and one location: one feature with two rows."""
    loc = ResLocPoint({'A': a_cm})
    return [row(c0, term_id=0, location=loc, a=0), row(c1, term_id=1, location=loc, a=0)]


def test_feature_result_cancelling_lists_the_features_whose_rows_cancel():
    """
    A = 0:  0.5 - 0.5    net_fraction 0        exact
    A = 1:  0.5 - 0.495  net_fraction 0.005    nearly
    A = 2:  0.3 + 0.5    net_fraction 1        no cancellation
    """
    table = ContributionTable(two_rows_at(0., 0.5, -0.5) + two_rows_at(1., 0.5, -0.495) + two_rows_at(2., 0.3, 0.5))
    result = FeatureResult(table, zero=[])

    assert [f.location['A'] for f in result.cancelling(0.)] == [0.]
    assert [f.location['A'] for f in result.cancelling(0.01)] == [0., 1.]


def test_feature_result_cancelling_leaves_out_zero_pairs():
    """A term with coefficient 0 is in `zero`, not a row: no feature, so no cancellation. One row never cancels."""
    result = FeatureResult(ContributionTable([row(0.3, a=0)]), zero=[(1, PS_00), (2, PS_01)])

    assert result.cancelling(0.5) == []


@pytest.mark.parametrize('rel_tol', [-0.1, 1., 2.])
def test_feature_result_cancelling_needs_rel_tol_from_0_to_below_1(rel_tol):
    with pytest.raises(ValueError, match='rel_tol'):
        FeatureResult(ContributionTable([]), zero=[]).cancelling(rel_tol)


## prop_names: what MolSystemData.load reads ---------------------------------------

def test_prop_names_every_property_of_the_terms_each_once():
    """toy_term: <polgrad>[a], cff[a, b, c]; a second term with dipgrad[a] and cff again"""
    with_dipgrad = replace(toy_term(), avrg_props=PropsCollection([polprop(ops=(0,), inds='a')]))

    assert prop_names([toy_term(), with_dipgrad, toy_term()]) == ['polgrad', 'cff', 'dipgrad']


def test_load_molsys_data_props_are_the_prop_names(monkeypatch):
    """one property per name, no matter how many terms use it; filled from the data dict"""
    data = {'polgrad': np.ones((2, 3, 3)), 'cff': np.ones((2, 2, 2)), 'nc_sqrt_eigval': {0: 1., 1: 2.}}
    monkeypatch.setattr(pipeline_mod, 'wilson_data_obtainer', lambda requested_data_dict: data)

    molsys = load_molsys_data([toy_term(), toy_term()], data_origin=None)  # type: ignore

    assert molsys.mol_props.names() == ['polgrad', 'cff']
    assert molsys.data_filled


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
    molsys = load_molsys_data(compiled, ref.data_origin('formaldehyde'))
    polarization = make_polarization_linear_comb(compiled, ref.LASER_POL)

    stored = next(b for b in json.loads(ref.REFERENCE_FILE.read_text())['results']
                  if (b['molecule'], b['states']) == ('formaldehyde', request.param))
    return EvvRun(request.param, compiled,
                  compute_features(compiled, molsys, polarization, states_choice=request.param), stored)


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
        assert f.amplitude_coeff == pytest.approx(expected, rel=1e-12, abs=0), f   # abs=0: amplitudes are ~1e-8 .. 1e-4


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


def test_real_data_leaves_add_up_to_every_row_coefficient(evv_formaldehyde):
    """
    All 340 rows, on the fly (einsum average) with the result's recipe. One leaf per summed index set:
    6 modes ** number of summation labels (term 0: none, 1 leaf; a term summing c: 6 leaves).
    """
    compiled, result = evv_formaldehyde.compiled, evv_formaldehyde.result

    for r in result.table:
        term = compiled[r.term_id]
        leaves = result.leaves(r)

        assert len(leaves) == 6 ** len(term.idx_summ)
        total = sum(leaf_value(term, contribs) for contribs in leaves.values())
        assert total == pytest.approx(r.coeff, rel=1e-12, abs=0), r


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
    return TopRun(terms, before, from_terms(terms, propagator_param=4.7))


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


def test_compute_features_from_terms_gives_every_feature_the_propagator_param(top_run):
    assert all(f.propagator_param == 4.7 for f in top_run.result.features)


def test_compute_features_from_terms_keeps_what_drawing_and_leaves_need(top_run):
    """the compiled terms, the data with its states (draw_window), the recipe: one row's leaves add up to its coeff"""
    result = top_run.result

    assert len(result.compiled) == len(top_run.terms)  # type: ignore
    assert result.states_choice == 'anharmonic'
    assert result.states is result.molsys.states['anharmonic']  # type: ignore
    assert result.polarization_linear_comb is not None

    r = max(result.table, key=lambda r: abs(r.coeff))
    term = result.compiled[r.term_id]  # type: ignore
    assert sum(leaf_value(term, c) for c in result.leaves(r).values()) == pytest.approx(r.coeff, rel=1e-12, abs=0)


## evaluate_response_terms: features and the spectrum on a grid ----------------
# The caller loads the data and makes the recipe once (from the compiled terms, as evv_formaldehyde does);
# evaluate_response_terms then takes the terms as derive writes them (pulse IDs). Formaldehyde, anharmonic.
# Strongest feature: A = 2682.765, B = 0 (test_real_data_strongest_feature). Its neighbours: A = 2726.8, B = 0
# and A = 2682.8, B = 44 (and A = 2726.8, B = -44).

WINDOW_AROUND_STRONGEST = Box({'A': (2632.765, 2732.765), 'B': (-50., 50.)})   # 100 x 100 cm-1
STEPS_1_CM = {'A': 101, 'B': 101}                                               # 1 cm-1 steps, both edges included


@pytest.fixture(scope='module')
def formaldehyde_data() -> tuple:
    """(molsys_data, orientation-average recipe): loaded once, used by every run below"""
    compiled = ref.compiled_evv_terms()
    return (load_molsys_data(compiled, ref.data_origin('formaldehyde')),
            make_polarization_linear_comb(compiled, ref.LASER_POL))


def evaluate_formaldehyde(formaldehyde_data, propagator_param, **kwargs):
    molsys, recipe = formaldehyde_data
    return evaluate_response_terms(terms=ref.evv_terms(), 
                                   axes=make_SpectralAxisSet(ref.AXES), 
                                   molsys_data=molsys, 
                                   states_choice='anharmonic',  # type: ignore
                                   orient_avrg_recipe=recipe, 
                                   propagator_param=propagator_param, **kwargs)


def test_evaluate_response_terms_draws_the_spectrum_with_a_width_function(formaldehyde_data, top_run):
    """
    The whole use: terms + data + recipe + width + window -> features and the spectrum on a grid.
    Width: 3 cm-1 plus 0.2 % of the condition's |E|. The strongest feature's conditions have E = 2682.765 (on A)
    and E = 0 (on B, since a = b), so its widths are 3 + 5.366 = 8.366 and 3 cm-1.
    """
    width = lambda vd, freqs: 3. + 0.002 * abs(vd.energy_difference())

    result = evaluate_formaldehyde(formaldehyde_data, width,
                                   window=WINDOW_AROUND_STRONGEST, dim_sizes=STEPS_1_CM, dynrange=1001.)

    # features: the same rows as compute_features_from_terms; every feature has the width function
    feature_result = result.feature_result
    assert list(feature_result.table) == list(top_run.result.table)
    assert len(feature_result.features) == 68
    assert all(f.propagator_param is width for f in feature_result.features)
    strongest = max(feature_result.features, key=lambda f: abs(f.amplitude_coeff))  # type: ignore
    assert strongest.location.as_dict() == pytest.approx({'A': 2682.765, 'B': 0.})
    assert strongest.peak_widths() == pytest.approx((3. + 0.002 * 2682.765, 3.))

    # the grid: 101 points per axis, from edge to edge of the window
    assert list(result.grid_axes) == ['A', 'B']  # type: ignore
    for ax, (mn, mx) in WINDOW_AROUND_STRONGEST.bounds.items():
        points = result.grid_axes[ax]  # type: ignore
        assert (points[0], points[-1], len(points)) == pytest.approx((mn, mx, 101))
    assert result.grid_result.shape == (101, 101)  # type: ignore
    intensity = np.abs(result.grid_result) ** 2

    # at the strongest feature: its own peak height (get_intensity), plus about 2 % from its neighbours
    i = np.argmin(np.abs(result.grid_axes['A'] - 2682.765))  # type: ignore
    j = np.argmin(np.abs(result.grid_axes['B']))  # type: ignore
    assert intensity[i, j] == pytest.approx(strongest.get_intensity(), rel=0.05)

    # drawn box by box (draw_window) vs every feature on the whole grid (draw_all): measured 0.3 % of the top
    reference = np.abs(draw_all(feature_result.features, feature_result.states, result.grid_axes)) ** 2  # type: ignore
    assert np.max(np.abs(intensity - reference)) < 0.01 * reference.max()


def test_evaluate_response_terms_width_function_of_a_constant_draws_as_the_number(formaldehyde_data):
    """propagator_param 5. and lambda vd, freqs: 5. give the same spectrum"""
    number = evaluate_formaldehyde(formaldehyde_data, 5., window=WINDOW_AROUND_STRONGEST, dim_sizes=STEPS_1_CM)
    function = evaluate_formaldehyde(formaldehyde_data, lambda vd, freqs: 5.,
                                     window=WINDOW_AROUND_STRONGEST, dim_sizes=STEPS_1_CM)

    np.testing.assert_allclose(function.grid_result, number.grid_result)  # type: ignore


def test_evaluate_response_terms_without_a_window_gives_the_features_only(formaldehyde_data):
    result = evaluate_formaldehyde(formaldehyde_data, 5.)

    assert len(result.feature_result.features) == 68
    assert result.grid_axes is None and result.grid_result is None
    assert result.grid_mesh is None and result.spectral_window is None


def test_evaluate_response_terms_window_needs_dim_sizes(formaldehyde_data):
    with pytest.raises(ValueError, match='dim_sizes'):
        evaluate_formaldehyde(formaldehyde_data, 5., window=WINDOW_AROUND_STRONGEST)


def test_evaluate_response_terms_grid_mesh_holds_every_grid_point(formaldehyde_data):
    """grid_mesh['A'][i, j] = grid_axes['A'][i], grid_mesh['B'][i, j] = grid_axes['B'][j]: the form the renderer reads"""
    result = evaluate_formaldehyde(formaldehyde_data, 5., window=WINDOW_AROUND_STRONGEST, dim_sizes=STEPS_1_CM)
    mesh, points = result.grid_mesh, result.grid_axes

    assert list(mesh) == ['A', 'B']  # type: ignore
    assert mesh['A'].shape == mesh['B'].shape == result.grid_result.shape == (101, 101)  # type: ignore
    for k in (0, 7, 100):
        np.testing.assert_array_equal(mesh['A'][:, k], points['A'])  # type: ignore
        np.testing.assert_array_equal(mesh['B'][k, :], points['B'])  # type: ignore


def test_evaluate_response_terms_keeps_the_window_as_drawn(formaldehyde_data):
    """
    The window's 4 features (example_evaluate_response_terms.py lists them), each with its box:
    5 cm-1, dynrange 1000, margin 0.1 -> 5 * sqrt(999) * 1.1 = 173.8 cm-1 on each side. No feature outside
    reaches in. The 4 boxes overlap, so draw_window drew them as one domain.
    """
    result = evaluate_formaldehyde(formaldehyde_data, 5., window=WINDOW_AROUND_STRONGEST, dim_sizes=STEPS_1_CM)
    window = result.spectral_window

    assert window.box is WINDOW_AROUND_STRONGEST  # type: ignore
    assert len(window.full_features) == 4 and window.contrib_features == []  # type: ignore
    for f in window.full_features:  # type: ignore
        mn, mx = f.feat_box.bounds['A']  # type: ignore
        assert (mx - mn) / 2 == pytest.approx(5. * 999 ** 0.5 * 1.1)
    [domain] = window.find_clusters_by_featboxes()  # type: ignore
    assert len(domain.full_features + domain.contrib_features) == 4


## draw_spectrum: drawing again without computing the rows again --------------

def test_draw_spectrum_with_a_new_width_reuses_the_rows(formaldehyde_data):
    """replace(..., propagator_param=8.) keeps the table (no new rows) and draws what a new run with 8 cm-1 draws"""
    first = evaluate_formaldehyde(formaldehyde_data, 5., window=WINDOW_AROUND_STRONGEST, dim_sizes=STEPS_1_CM)

    wider = draw_spectrum(replace(first.feature_result, propagator_param=8.), WINDOW_AROUND_STRONGEST, STEPS_1_CM)

    assert wider.feature_result.table is first.feature_result.table
    assert all(f.propagator_param == 8. for f in wider.feature_result.features)
    new_run = evaluate_formaldehyde(formaldehyde_data, 8., window=WINDOW_AROUND_STRONGEST, dim_sizes=STEPS_1_CM)
    np.testing.assert_allclose(wider.grid_result, new_run.grid_result)  # type: ignore


def test_draw_spectrum_on_another_window_reuses_the_features(formaldehyde_data):
    """Same features, another window and grid: what a new run on that window draws"""
    first = evaluate_formaldehyde(formaldehyde_data, 5.)
    other, sizes = Box({'A': (2700., 2750.), 'B': (-60., 0.)}), {'A': 51, 'B': 61}

    again = draw_spectrum(first.feature_result, other, sizes)

    assert again.feature_result is first.feature_result
    new_run = evaluate_formaldehyde(formaldehyde_data, 5., window=other, dim_sizes=sizes)
    np.testing.assert_allclose(again.grid_result, new_run.grid_result)  # type: ignore
