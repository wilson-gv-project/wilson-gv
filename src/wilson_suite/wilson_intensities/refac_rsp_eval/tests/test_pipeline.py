"""
pipeline.py — compute_features end to end on the toy molecule (helpers.py, conftest.py),
then MolSystemData.load + compute_features on real data (14 EVV terms + formaldehyde).

The steps have their own tests (build_contributions in test_evaluate.py, features_from_rows in
test_features.py). Here: the steps are joined right, and FeatureResult keeps the table as its source.

ab_term fixes a, b and sums over c. <polgrad>[1] = 0 (molsys_a1_zero with TOY_RECIPE), so index sets
with a=1 have coeff 0: per term, 2 rows (a=0) at 2 different locations and 2 zero pairs (a=1).
"""

import dataclasses
import json
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import partial

import numpy as np
import pytest

import wilson_suite.wilson_intensities.refac_rsp_eval.pipeline as pipeline_mod
from wilson_suite.wilson_derive.abstractions import PolProp
from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_intensities.amplitudes.averaging import (
    getGeneralPolarizationAveragingExpression,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.compat import term_to_str
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    all_features_on_grid,
    build_contributions,
    evaluate_term_coeff_sumover,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    ContributionTable,
    SpectralFeature,
    SpectralWindow,
    amplitude_half_widths,
    intensity_half_widths,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import Box
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import (
    FeatureResult,
    averaging_rank,
    check_one_axis_set,
    compute_features,
    evaluate_response_terms,
    make_polarization_linear_comb,
    prop_names,
    spectrum_on_grid,
    terms_by_axis_set,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    ParameterSet,
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
    MOTIF_A,
    MOTIF_AB,
    PS_00,
    PS_01,
    PS_10,
    PS_11,
    TOY_RECIPE,
    ab_term,
    flat_term,
    get_box_extent,
    polprop,
    row,
    toy_term,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.stored_terms import TERMS_DIR
from wilson_suite.wilson_system.system_data import MolecularProperty, MolPropsCollection, MolSystemData


def two_terms_one_motif():
    """Same motif; frac_factor 0.5 and 2., so every coefficient of term 1 is 4x that of term 0."""
    term = ab_term()
    return [term, replace(term, frac_factor=2.)]


## compute_features ---------------------------------------------------------

def test_compute_features_one_feature_per_location_holding_both_terms(molsys_a1_zero):
    result = compute_features(two_terms_one_motif(), molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    f00, f01 = result.features
    assert f00.location.as_dict() == pytest.approx({'A': E0, 'B': E00 - E0})
    assert f01.location.as_dict() == pytest.approx({'A': E0, 'B': E01 - E0})
    for f, ps in ((f00, PS_00), (f01, PS_01)):
        assert f.motif == ResonanceMotif.from_tuples(MOTIF_AB)
        assert f.param_sets == (ps,)
        assert f.term_ids == (0, 1)


def test_compute_features_amplitude_adds_both_terms(molsys_a1_zero):
    terms = two_terms_one_motif()

    result = compute_features(terms, molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    for f, idx in zip(result.features, ({'a': 0, 'b': 0}, {'a': 0, 'b': 1})):
        single, _ = evaluate_term_coeff_sumover(terms[0], idx, molsys_a1_zero, TOY_RECIPE,
                                                states_choice='anharmonic')
        assert single != 0.
        # 1x (term 0) + 4x (term 1) -- 0.5 and 2.0 frac_factor, so 5x single
        assert f.amplitude_coeff == pytest.approx(5 * single)


def test_compute_features_reports_zero_pairs(molsys_a1_zero):
    result = compute_features(two_terms_one_motif(), molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    assert result.zero == [(0, PS_10), (0, PS_11), (1, PS_10), (1, PS_11)]


def test_compute_features_gives_every_feature_the_propagator_param(molsys_a1_zero):
    result = compute_features([ab_term()], molsys_a1_zero, TOY_RECIPE, propagator_param=5.,
                              states_choice='anharmonic')

    assert len(result.features) == 2
    assert all(f.propagator_param == 5. for f in result.features)
    # no boxes yet: they come later, from SpectralWindow.from_features
    assert all(f.feat_box is None for f in result.features)


def test_compute_features_gives_a_propagator_param_function_the_conditions(molsys_a1_zero):
    """
    The function gets the pair of states of each condition, from the result's states. ab_term (MOTIF_AB), a=0:
    conditions <0 - a> and <ab - a>, so with width 5 + 1 % of |E|:
        b=0:  0 - E0 = -1000,  E00 - E0 = 950   ->  15, 14.5 cm-1
        b=1:  0 - E0 = -1000,  E01 - E0 = 1400  ->  15, 19 cm-1
    """
    width = lambda vd, freqs: 5. + 0.01 * abs(vd.energy_difference())

    result = compute_features([ab_term()], molsys_a1_zero, TOY_RECIPE, propagator_param=width,
                              states_choice='anharmonic')

    assert all(f.propagator_param is width for f in result.features)
    assert [f.peak_widths() for f in result.features] == [pytest.approx((15., 14.5)), pytest.approx((15., 19.))]


## one set of resonance axes per evaluation ----------------------
# MOTIF_AB resonates on {A, B}, MOTIF_A on {A}, MOTIF_B on {B}.

MOTIF_B = (((('a',), ()), ('B',)),)           # B = E_a - E_0


def test_compute_features_raises_for_terms_on_two_axis_sets(molsys_a1_zero):
    """
    {A, B} and {A}: compiling takes both (each fixes one point on its own axes); the grid would not
    ("Expected SpectralFeature with axes ('A', 'B'), got ('A',)"). The message lists the term ids per set.
    """
    terms = [ab_term(MOTIF_AB), ab_term(MOTIF_A), ab_term(MOTIF_AB)]

    with pytest.raises(ValueError, match=re.escape("term ids by set: {('A', 'B'): [0, 2], ('A',): [1]}")):
        compute_features(terms, molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')


def test_compute_features_raises_for_terms_on_A_only_and_on_B_only(molsys_a1_zero):
    """{A} and {B} are different sets: in an (A, B) plane the first is a stripe along B, the second along A."""
    with pytest.raises(ValueError, match='2 sets of axes'):
        compute_features([ab_term(MOTIF_A), ab_term(MOTIF_B)], molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')


def test_check_one_axis_set_does_not_count_a_term_without_conditions():
    """A term without conditions has no axes: it is not a set of its own."""
    check_one_axis_set([ab_term(MOTIF_AB), flat_term()])


def test_compute_features_term_without_conditions_is_one_flat_feature(molsys_a1_zero):
    """flat_term in the same call as ab_term: ab_term's 2 features, and 1 flat feature with flat_term's row."""
    result = compute_features([ab_term(), flat_term()], molsys_a1_zero, TOY_RECIPE, 5., states_choice='anharmonic')

    flat = [f for f in result.features if f.is_flat]
    assert len(result.features) == 3
    assert [f.term_ids for f in flat] == [(1,)]


def test_spectrum_on_grid_adds_a_term_without_conditions_on_every_grid_point(molsys_a1_zero):
    """
    ab_term alone, then with flat_term in the same call: the two grid results differ by the flat feature's
    coefficient on every grid point. Boxes of 20 cm-1 for both, so the peaks are cut alike.
    """
    window, dim_sizes = Box({'A': (1500., 1700.), 'B': (300., 700.)}), {'A': 101, 'B': 101}
    boxes = lambda window, features: [20.] * len(features)
    alone = compute_features([ab_term()], molsys_a1_zero, TOY_RECIPE, 5., states_choice='anharmonic')
    with_flat = compute_features([ab_term(), flat_term()], molsys_a1_zero, TOY_RECIPE, 5., states_choice='anharmonic')
    [flat] = [f for f in with_flat.features if f.is_flat]

    spec = spectrum_on_grid(with_flat, window, dim_sizes, box_rule=boxes)

    assert flat.amplitude_coeff != 0.
    alone_on_grid = spectrum_on_grid(alone, window, dim_sizes, box_rule=boxes).amplitude
    np.testing.assert_allclose(spec.amplitude, alone_on_grid + flat.amplitude_coeff)


def test_feature_result_made_by_hand_gives_a_propagator_param_function_no_conditions():
    """No states to take them from: the feature has no conditions, and its box and peak height raise."""
    result = FeatureResult(ContributionTable([row(0.3, a=0)]), zero=[], propagator_param=lambda vd, freqs: 5.)

    [f] = result.features

    assert f.conditions is None
    with pytest.raises(ValueError, match='no conditions'):
        f.get_intensity()


## FeatureResult ------------------------------------------------------------

def test_feature_result_table_is_what_build_contributions_returns(molsys_a1_zero):
    terms = two_terms_one_motif()
    
    # compute_features calls build_contributions, so this is a bit redundant, but it checks that the table is preserved
    result = compute_features(terms, molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')
    table, _ = build_contributions(terms, molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    assert list(result.table) == list(table)


def test_feature_result_features_hold_every_row_of_the_table_once(molsys_a1_zero):
    result = compute_features(two_terms_one_motif(), molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    rows_in_features = [r for f in result.features for r in f.rows]

    assert len(rows_in_features) == len(result.table) == 4
    assert set(rows_in_features) == set(result.table)


def test_feature_result_builds_features_once(molsys_a1_zero):
    result = compute_features([ab_term()], molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    assert result.features is result.features


def test_feature_result_is_frozen(molsys_a1_zero):
    result = compute_features([ab_term()], molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.table = ContributionTable([])  # type: ignore


def index_set_value(term: CompiledTerm, contribs: dict) -> float:
    """
    One index set's part of the row coefficient, from its factors, as evaluate_full_index_dict multiplies them.
    An index set that stopped early (average or non-averaged part 0) has fewer factors and gives 0.
    """
    if 'VIBENE_DENOM' not in contribs:
        return 0.
    return term.frac_factor * (contribs['NON_AVRG'] * contribs['AVRG'] * contribs['VIBDIFF_TERMS']
                               * contribs['VIBENE_DENOM'])


def test_feature_result_factors_by_index_set_of_a_row_are_its_summed_index_sets(molsys_a1_zero):
    """ab_term fixes a, b and sums c over 2 modes: 2 index sets per row, the same as evaluate_term_coeff_sumover gives"""
    terms = two_terms_one_motif()
    result = compute_features(terms, molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    for r in result.table:
        factors = result.factors_by_index_set(r)

        _, expected = evaluate_term_coeff_sumover(terms[r.term_id], r.params.to_dict(), molsys_a1_zero,
                                                  TOY_RECIPE, states_choice='anharmonic')
        assert set(factors) == {ParameterSet({'a': r.params['a'], 'b': r.params['b'], 'c': c}) for c in (0, 1)}
        assert factors == expected


def test_feature_result_factors_by_index_set_add_up_to_the_row_coefficient(molsys_a1_zero):
    terms = two_terms_one_motif()
    result = compute_features(terms, molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    for r in result.table:
        total = sum(index_set_value(terms[r.term_id], contribs) for contribs in result.factors_by_index_set(r).values())
        assert total == pytest.approx(r.coeff, rel=1e-12, abs=0)


def test_feature_result_factors_by_index_set_use_the_polarization_of_the_result(molsys):
    """
    factors_by_index_set needs the result's own recipe (without it, the average raises), here a real one: rank 2 from
    getGeneralPolarizationAveragingExpression. ab_term's <polgrad>[a] has slots 0, 1: rank 2. polgrad of mode 1
    is 0, so a = 1 stops at the average, as with molsys_a1_zero: the toy states have no '1,1'.
    """
    terms = [ab_term()]
    polgrad = np.zeros((2, 3, 3))
    polgrad[0] = np.arange(9.).reshape(3, 3) + 1.
    otf_molsys = replace(molsys, mol_props=MolPropsCollection(
        [*molsys.mol_props, MolecularProperty(trivial_name='polgrad', vals=polgrad, extra_data={})]))
    recipe = getGeneralPolarizationAveragingExpression(rank=2, laser_pol=(1.,))

    result = compute_features(terms, otf_molsys, recipe, states_choice='anharmonic')

    assert result.polarization_linear_comb is recipe
    assert len(result.table) == 2                       # a = 0, b = 0 and 1
    for r in result.table:
        total = sum(index_set_value(terms[0], contribs) for contribs in result.factors_by_index_set(r).values())
        assert total == pytest.approx(r.coeff, rel=1e-12, abs=0) != 0.


def test_feature_result_keeps_its_inputs_without_copying(molsys_a1_zero):
    terms = two_terms_one_motif()

    result = compute_features(terms, molsys_a1_zero, TOY_RECIPE, states_choice='anharmonic')

    assert result.compiled == tuple(terms) and all(a is b for a, b in zip(result.compiled, terms)) # type: ignore
    assert result.molsys is molsys_a1_zero
    assert result.polarization_linear_comb is TOY_RECIPE


def test_feature_result_inputs_do_not_count_for_equality():
    """two results with the same rows are equal, whatever they were computed from"""
    table = ContributionTable([row(0.3, a=0)])

    assert FeatureResult(table, zero=[]) == FeatureResult(table, zero=[], compiled=(toy_term(),))


def test_feature_result_made_by_hand_has_no_factors_by_index_set():
    r = row(0.3, a=0)

    with pytest.raises(ValueError, match='no terms or data'):
        FeatureResult(ContributionTable([r]), zero=[]).factors_by_index_set(r)


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


## toy terms with props only: what prop_names, averaging_rank, make_polarization_linear_comb read ----------
# Only the props are read (plan.split_props), so a toy term needs no denominators and no resonance conditions.

def pulse_term(*props: PolProp) -> VibPerturbedTerm:
    """a term with props only: no conditions, so neither pulse IDs nor axes"""
    return VibPerturbedTerm(Fraction(1, 2), list(props), [], [])


def toy_pulse_term() -> VibPerturbedTerm:
    """the props of toy_term(): <polgrad>[a] (slots 0, 1), cff[a, b, c]"""
    return pulse_term(polprop(ops=(0, 1), inds='a'), polprop(inds='abc'))


## prop_names: what MolSystemData.load reads ---------------------------------------

def test_prop_names_every_property_of_the_terms_each_once():
    """toy_pulse_term: <polgrad>[a], cff[a, b, c]; a second term with dipgrad[a] and cff again"""
    with_dipgrad = pulse_term(polprop(ops=(0,), inds='a'), polprop(inds='abc'))

    assert prop_names([toy_pulse_term(), with_dipgrad, toy_pulse_term()]) == ['polgrad', 'cff', 'dipgrad']


def test_prop_names_lists_the_averaged_props_of_a_term_first():
    """as CompiledTerm.prop_names: cff written first, still after <polgrad>"""
    assert prop_names([pulse_term(polprop(inds='abc'), polprop(ops=(0, 1), inds='a'))]) == ['polgrad', 'cff']


def test_prop_names_of_the_terms_in_pulse_ids_are_those_of_the_compiled_terms():
    """the 14 EVV terms: the axis translation leaves the props alone"""
    from_compiled = list(dict.fromkeys(name for t in ref.compiled_evv_terms() for name in t.prop_names))

    assert prop_names(ref.evv_terms()) == from_compiled


def test_molsys_data_loaded_with_the_prop_names_has_one_property_per_name():
    """the caller's load (pipeline docstring): one property per name, no matter how many terms use it; filled from
    the data dict"""
    data = {'polgrad': np.ones((2, 3, 3)), 'cff': np.ones((2, 2, 2)), 'nc_sqrt_eigval': {0: 1., 1: 2.},
            'harmonic_states': {('0',): 1., ('1',): 2.}}

    molsys = MolSystemData.load(None, prop_names([toy_pulse_term(), toy_pulse_term()]),  # type: ignore
                                obtainer=lambda request: data)

    assert molsys.mol_props.names() == ['polgrad', 'cff']
    assert molsys.data_filled


## polarization recipe: averaging_rank, make_polarization_linear_comb ----------
# The rank counts the Cartesian slots (op.o) of a term's averaged part.
# toy_pulse_term(): <polgrad> with slots 0, 1 -> rank 2.

def term_with_slots(*ops: tuple[int, ...]) -> VibPerturbedTerm:
    """one averaged property per ops tuple, and cff[a, b, c] (no slots)"""
    return pulse_term(*(polprop(ops=o, inds='a') for o in ops), polprop(inds='abc'))


RANK_4_TERM = term_with_slots((0, 3), (1,), (2,))      # like EVV term 0: polhess[0,3] dipgrad[1] dipgrad[2]


def test_averaging_rank_counts_the_slots_of_the_averaged_part():
    assert averaging_rank([toy_pulse_term()]) == 2
    assert averaging_rank([RANK_4_TERM]) == 4


def test_averaging_rank_of_terms_with_the_same_slots():
    """slots 0, 1 in one property or in two: the same slots"""
    assert averaging_rank([toy_pulse_term(), term_with_slots((1,), (0,))]) == 2


def test_averaging_rank_raises_if_terms_differ():
    """compute_features uses one recipe for all terms: rank 2 and rank 4 cannot share it"""
    with pytest.raises(ValueError, match='terms differ'):
        averaging_rank([toy_pulse_term(), RANK_4_TERM])


def test_averaging_rank_raises_if_a_slot_is_missing():
    """slots 0, 2: the recipe puts slot i at position i, and a rank-2 recipe has no position 2"""
    with pytest.raises(ValueError, match=r'0 \.\. 1'):
        averaging_rank([term_with_slots((0, 2))])


def test_averaging_rank_raises_without_terms():
    with pytest.raises(ValueError, match='no terms'):
        averaging_rank([])


@pytest.mark.parametrize('term, rank, laser_pol', [(toy_pulse_term(), 2, (1.,)), (RANK_4_TERM, 4, (1., 0., 1.))],
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


def test_make_polarization_linear_comb_makes_the_recipe_once_per_rank_and_laser_pol(monkeypatch):
    """the same rank and laser_pol again (other terms, a list instead of a tuple): no new recipe; another laser_pol: a new one"""
    made = []

    def counting(**kwargs):
        made.append(kwargs)
        return getGeneralPolarizationAveragingExpression(**kwargs)
    monkeypatch.setattr(pipeline_mod, 'getGeneralPolarizationAveragingExpression', counting)
    pipeline_mod._averaging_recipe.cache_clear()

    first = make_polarization_linear_comb([RANK_4_TERM], (1., 0., 1.))
    again = make_polarization_linear_comb([RANK_4_TERM, RANK_4_TERM], [1., 0., 1.])
    other = make_polarization_linear_comb([RANK_4_TERM], (1., 1., 1.))

    assert made == [{'rank': 4, 'laser_pol': (1., 0., 1.)}, {'rank': 4, 'laser_pol': (1., 1., 1.)}]
    assert again == first != other


def test_make_polarization_linear_comb_gives_each_call_its_own_copy():
    """changing one result changes neither the kept recipe nor the next result"""
    first = make_polarization_linear_comb([RANK_4_TERM], (1., 1., 1.))
    first.clear()

    assert make_polarization_linear_comb([RANK_4_TERM], (1., 1., 1.)) == \
        getGeneralPolarizationAveragingExpression(rank=4, laser_pol=(1., 1., 1.))


## real data: 14 EVV terms + formaldehyde -------------------------------------
# MolSystemData.load -> compute_features on the real data file; the settings and the stored rows are
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
    terms, compiled = ref.evv_terms(), ref.compiled_evv_terms()
    molsys = MolSystemData.load(ref.data_origin('formaldehyde'), prop_names(terms))
    polarization = make_polarization_linear_comb(terms, ref.LASER_POL)

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


def test_real_data_averaging_rank_is_the_one_stored_in_the_reference_file():
    assert averaging_rank(ref.evv_terms()) == ref.RANK == 4


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


def test_real_data_factors_by_index_set_add_up_to_every_row_coefficient(evv_formaldehyde):
    """
    All 340 rows, on the fly (einsum average) with the result's recipe. One entry per summed index set:
    6 modes ** number of summation labels (term 0: none, 1 entry; a term summing c: 6 entries).
    """
    compiled, result = evv_formaldehyde.compiled, evv_formaldehyde.result

    for r in result.table:
        term = compiled[r.term_id]
        factors = result.factors_by_index_set(r)

        assert len(factors) == 6 ** len(term.idx_summ)
        total = sum(index_set_value(term, contribs) for contribs in factors.values())
        assert total == pytest.approx(r.coeff, rel=1e-12, abs=0), r


## evaluate_response_terms: the top function -----------------------------------
# Same settings as evv_reference.json, formaldehyde, anharmonic states. The data is loaded once
# (formaldehyde_molsys), for the stored EVV terms (already in axes). The evv_formaldehyde fixture
# runs the same steps one by one, so equal results mean evaluate_response_terms passes every setting to
# the right step. One run (top_run) serves most tests.

@pytest.fixture(scope='module')
def formaldehyde_molsys() -> MolSystemData:
    """loaded once, used by every run below"""
    return MolSystemData.load(ref.data_origin('formaldehyde'), prop_names(ref.evv_terms()))


def evaluate_formaldehyde(molsys, propagator_param, terms=None, **kwargs) -> FeatureResult:
    return evaluate_response_terms(ref.evv_terms() if terms is None else terms,
                                   molsys_data=molsys,
                                   states_choice='anharmonic',
                                   laser_pol=ref.LASER_POL,
                                   propagator_param=propagator_param, **kwargs)


@dataclass(frozen=True)
class TopRun:
    terms: list             # the input terms, after the run
    terms_before: list      # their to_str, before the run
    result: FeatureResult


@pytest.fixture(scope='module')
def top_run(formaldehyde_molsys) -> TopRun:
    terms = ref.evv_terms()
    before = [term_to_str(t) for t in terms]
    return TopRun(terms, before, evaluate_formaldehyde(formaldehyde_molsys, 4.7, terms))


@pytest.mark.parametrize('evv_formaldehyde', ['anharmonic'], indirect=True)
def test_evaluate_response_terms_equals_the_steps_one_by_one(top_run, evv_formaldehyde):
    result, by_steps = top_run.result, evv_formaldehyde.result

    assert list(result.table) == list(by_steps.table)
    assert result.zero == by_steps.zero


def test_evaluate_response_terms_term_id_is_the_index_in_the_input_list(top_run, formaldehyde_molsys):
    """terms in reverse order: the row of term i comes back with term_id 13 - i, same params and coeff"""
    backward = evaluate_formaldehyde(formaldehyde_molsys, 4.7, ref.evv_terms()[::-1])

    last = len(top_run.terms) - 1
    assert ({(last - r.term_id, r.params, r.coeff) for r in backward.table}
            == {(r.term_id, r.params, r.coeff) for r in top_run.result.table})


def test_evaluate_response_terms_leaves_the_input_terms_as_they_are(top_run):
    """the caller's terms come back unchanged (never change derive's objects)"""
    assert [term_to_str(t) for t in top_run.terms] == top_run.terms_before


def test_evaluate_response_terms_gives_every_feature_the_propagator_param(top_run):
    assert all(f.propagator_param == 4.7 for f in top_run.result.features)


def test_evaluate_response_terms_keeps_what_the_grid_and_factors_by_index_set_need(top_run, formaldehyde_molsys):
    """
    the compiled terms, the data with its states (boxed_features_on_grid), the recipe: one row's index sets add up to
    its coeff
    """
    result = top_run.result

    assert len(result.compiled) == len(top_run.terms)  # type: ignore
    assert result.molsys is formaldehyde_molsys
    assert result.states_choice == 'anharmonic'
    assert result.states is formaldehyde_molsys.states['anharmonic']
    assert result.polarization_linear_comb == make_polarization_linear_comb(top_run.terms, ref.LASER_POL)

    r = max(result.table, key=lambda r: abs(r.coeff))
    term = result.compiled[r.term_id]  # type: ignore
    assert sum(index_set_value(term, c) for c in result.factors_by_index_set(r).values()) == pytest.approx(r.coeff, rel=1e-12, abs=0)


def test_terms_by_axis_set_the_14_evv_terms_are_one_set():
    assert terms_by_axis_set(ref.evv_terms()) == {('A', 'B'): list(range(14))}


def evv_terms_with_term_0_on_A_only() -> list:
    """the 14 EVV terms; term 0 keeps only its first condition (pulse 1, -A in these axes): its set is {A}"""
    terms = ref.evv_terms()
    terms[0].res = terms[0].res[:1]
    return terms


def test_terms_by_axis_set_groups_the_term_ids_in_the_chosen_axes():
    groups = terms_by_axis_set(evv_terms_with_term_0_on_A_only())

    assert groups == {('A',): [0], ('A', 'B'): list(range(1, 14))}


def test_evaluate_response_terms_raises_for_two_axis_sets_and_runs_per_set(formaldehyde_molsys, top_run):
    """
    The mix raises before any row is computed. Split by terms_by_axis_set, the {A, B} set runs: its rows are those of
    terms 1-13 in top_run, with ids[row.term_id] the index in the full list.
    """
    terms = evv_terms_with_term_0_on_A_only()
    with pytest.raises(ValueError, match='2 sets of axes'):
        evaluate_formaldehyde(formaldehyde_molsys, 4.7, terms)

    ids = terms_by_axis_set(terms)[('A', 'B')]
    result = evaluate_formaldehyde(formaldehyde_molsys, 4.7, [terms[i] for i in ids])

    assert ({(ids[r.term_id], r.params, r.coeff) for r in result.table}
            == {(r.term_id, r.params, r.coeff) for r in top_run.result.table if r.term_id != 0})


def test_evaluate_response_terms_takes_every_setting_by_keyword(formaldehyde_molsys):
    """positional settings raise: a swapped pair, e.g. states_choice and laser_pol, cannot slip through"""
    with pytest.raises(TypeError, match='positional'):
        evaluate_response_terms(ref.evv_terms(), formaldehyde_molsys, 'anharmonic', ref.LASER_POL, 5.)  # type: ignore


def evv_terms_in_pulse_ids() -> list[VibPerturbedTerm]:
    """the 14 EVV terms as derive writes them, before the translation: e.g. pf [-1, 2] where evv.json has ['B']"""
    return VibPerturbedTerm.load_many_from_json(str(TERMS_DIR / 'evv_in_pulse_ids.json'))


def test_evaluate_response_terms_raises_on_terms_in_pulse_ids(formaldehyde_molsys):
    with pytest.raises(ValueError, match='pulse IDs'):
        evaluate_formaldehyde(formaldehyde_molsys, 4.7, evv_terms_in_pulse_ids())


def test_terms_by_axis_set_raises_on_terms_in_pulse_ids():
    with pytest.raises(ValueError, match='pulse IDs'):
        terms_by_axis_set(evv_terms_in_pulse_ids())


def test_evaluate_response_terms_excluded_mode_is_left_out_of_rows_and_sums(formaldehyde_molsys, top_run):
    """
    exclude_modes (5,), formaldehyde, 6 modes: 5 x 5 (a, b) pairs per term instead of 6 x 6, none with mode 5.
    Every row's coefficient is the sum of the full run's index sets without mode 5 (top_run: no exclusion), in every
    term. That includes the terms whose averaged part has no c (2, 4, 5, 7, 9-12): their sums lose mode 5 too, so
    for term 2 the coefficients change.
    """
    result = evaluate_formaldehyde(formaldehyde_molsys, 4.7, exclude_modes=(5,))

    pairs = [(r.term_id, r.params) for r in result.table] + result.zero
    assert len(pairs) == 14 * 5 * 5
    assert all(5 not in params.indices() for _, params in pairs)
    assert result.exclude_modes == (5,)

    full = {(r.term_id, r.params): r for r in top_run.result.table}
    changed_in_term_2 = 0
    for r in result.table:
        term = result.compiled[r.term_id]  # type: ignore
        assert all(5 not in index_set.indices() for index_set in result.factors_by_index_set(r))
        full_factors = top_run.result.factors_by_index_set(full[(r.term_id, r.params)])
        without_5 = sum(index_set_value(term, c) for index_set, c in full_factors.items() if 5 not in index_set.indices())
        assert r.coeff == pytest.approx(without_5, rel=1e-12, abs=0)
        changed_in_term_2 += r.term_id == 2 and r.coeff != pytest.approx(full[(2, r.params)].coeff)
    assert changed_in_term_2 > 0


## spectrum_on_grid: the features on a grid -------------------------------------
# Strongest feature: A = 2682.765, B = 0 (test_real_data_strongest_feature). Its neighbours: A = 2726.8, B = 0
# and A = 2682.8, B = 44 (and A = 2726.8, B = -44).

WINDOW_AROUND_STRONGEST = Box({'A': (2632.765, 2732.765), 'B': (-50., 50.)})   # 100 x 100 cm-1
STEPS_1_CM = {'A': 101, 'B': 101}                                               # 1 cm-1 steps, both edges included
AMPLITUDE_BOXES = partial(amplitude_half_widths, dynrange=1001.)                # the box rule, unless a test says


def test_spectrum_on_grid_with_a_width_function(formaldehyde_molsys, top_run):
    """
    The whole use: terms + data + width -> features (evaluate_response_terms) -> the spectrum on a grid.
    Width: 3 cm-1 plus 0.2 % of the condition's |E|. The strongest feature's conditions have E = 2682.765 (on A)
    and E = 0 (on B, since a = b), so its widths are 3 + 5.366 = 8.366 and 3 cm-1.
    """
    width = lambda vd, freqs: 3. + 0.002 * abs(vd.energy_difference())

    result = evaluate_formaldehyde(formaldehyde_molsys, width)
    spec = spectrum_on_grid(result, WINDOW_AROUND_STRONGEST, STEPS_1_CM, box_rule=AMPLITUDE_BOXES)

    # features: the same rows as top_run; every feature has the width function
    assert spec.feature_result is result
    assert list(result.table) == list(top_run.result.table)
    assert len(result.features) == 68
    assert all(f.propagator_param is width for f in result.features)
    strongest = max(result.features, key=lambda f: abs(f.amplitude_coeff))  # type: ignore
    assert strongest.location.as_dict() == pytest.approx({'A': 2682.765, 'B': 0.})
    assert strongest.peak_widths() == pytest.approx((3. + 0.002 * 2682.765, 3.))

    # the grid: 101 points per axis, from edge to edge of the window
    assert list(spec.axis_points) == ['A', 'B']
    for ax, (mn, mx) in WINDOW_AROUND_STRONGEST.bounds.items():
        points = spec.axis_points[ax]
        assert (points[0], points[-1], len(points)) == pytest.approx((mn, mx, 101))
    assert spec.amplitude.shape == (101, 101)
    intensity = np.abs(spec.amplitude) ** 2

    # at the strongest feature: its own peak height (get_intensity), plus about 2 % from its neighbours
    i = np.argmin(np.abs(spec.axis_points['A'] - 2682.765))
    j = np.argmin(np.abs(spec.axis_points['B']))
    assert intensity[i, j] == pytest.approx(strongest.get_intensity(), rel=0.05)

    # the kept features on the whole grid (the default, cut_tails=False) vs every feature on the whole grid
    # (all_features_on_grid): at most eps = 0.1 floors (floor: max / dynrange)
    reference = np.abs(all_features_on_grid(result.features, result.states, spec.axis_points)) ** 2
    assert np.max(np.abs(intensity - reference)) < 0.1 * reference.max() / 1001.


def test_spectrum_on_grid_width_function_of_a_constant_gives_the_same_as_the_number(formaldehyde_molsys):
    """propagator_param 5. and lambda vd, freqs: 5. give the same spectrum"""
    number = spectrum_on_grid(evaluate_formaldehyde(formaldehyde_molsys, 5.), WINDOW_AROUND_STRONGEST, STEPS_1_CM,
                           box_rule=AMPLITUDE_BOXES)
    function = spectrum_on_grid(evaluate_formaldehyde(formaldehyde_molsys, lambda vd, freqs: 5.),
                             WINDOW_AROUND_STRONGEST, STEPS_1_CM, box_rule=AMPLITUDE_BOXES)

    np.testing.assert_allclose(function.amplitude, number.amplitude)


def test_spectrum_on_grid_mesh_holds_every_grid_point(top_run):
    """mesh['A'][i, j] = axis_points['A'][i], mesh['B'][i, j] = axis_points['B'][j]: the form the renderer reads"""
    spec = spectrum_on_grid(top_run.result, WINDOW_AROUND_STRONGEST, STEPS_1_CM, box_rule=AMPLITUDE_BOXES)
    mesh, points = spec.mesh, spec.axis_points

    assert list(mesh) == ['A', 'B']
    assert mesh['A'].shape == mesh['B'].shape == spec.amplitude.shape == (101, 101)
    for k in (0, 7, 100):
        np.testing.assert_array_equal(mesh['A'][:, k], points['A'])
        np.testing.assert_array_equal(mesh['B'][k, :], points['B'])


def test_spectrum_on_grid_keeps_the_window_it_used(formaldehyde_molsys):
    """
    The window as used: each feature with its amplitude box (amplitude_half_widths, dynrange 1001). The 4 features
    inside are strong enough for boxes far wider than the window (29000 to 100000 cm-1); 32 of the 64 outside
    reach in, as contrib_features.
    """
    result = evaluate_formaldehyde(formaldehyde_molsys, 5.)

    spec = spectrum_on_grid(result, WINDOW_AROUND_STRONGEST, STEPS_1_CM, box_rule=AMPLITUDE_BOXES)

    window = spec.spectral_window
    expected = SpectralWindow.from_features(WINDOW_AROUND_STRONGEST, result.features,
                                            amplitude_half_widths(WINDOW_AROUND_STRONGEST, result.features, dynrange=1001.))
    assert window.box is WINDOW_AROUND_STRONGEST
    assert (len(window.full_features), len(window.contrib_features)) == (4, 32)
    for used, made in zip(window.full_features + window.contrib_features,
                           expected.full_features + expected.contrib_features, strict=True):
        assert used == made and used.feat_box == made.feat_box
    assert all(WINDOW_AROUND_STRONGEST.bounds['A'][1] - WINDOW_AROUND_STRONGEST.bounds['A'][0]
               < get_box_extent(f) for f in window.full_features)


def test_spectrum_on_grid_uses_the_boxes_of_its_box_rule(top_run):
    """
    The rule gets the window and every feature; the boxes on the grid are its half-widths.
    A function of our own: 7 cm-1 for every feature.
    """
    calls = []

    def seven_cm(window, features):
        calls.append((window, list(features)))
        return [7.] * len(features)

    spec = spectrum_on_grid(top_run.result, WINDOW_AROUND_STRONGEST, STEPS_1_CM, box_rule=seven_cm)

    assert calls == [(WINDOW_AROUND_STRONGEST, top_run.result.features)]
    kept = spec.spectral_window.full_features + spec.spectral_window.contrib_features
    assert kept and all(get_box_extent(f) == pytest.approx(7.) for f in kept)


def test_spectrum_on_grid_takes_the_knobs_through_the_box_rule(top_run):
    """
    partial(amplitude_half_widths, **knobs): the boxes on the grid are its half-widths with the same settings.
    min_gammas 2 (gamma 4.7): the smallest box is 9.4 cm-1.
    """
    features = top_run.result.features
    knobs = dict(dynrange=1001., eps=0.3, min_gammas=2., a_max=10. * max(f.get_intensity() for f in features) ** 0.5)

    spec = spectrum_on_grid(top_run.result, WINDOW_AROUND_STRONGEST, STEPS_1_CM,
                         box_rule=partial(amplitude_half_widths, **knobs))

    kept = spec.spectral_window.full_features + spec.spectral_window.contrib_features
    expected = amplitude_half_widths(WINDOW_AROUND_STRONGEST, features, **knobs)
    assert kept and all(get_box_extent(f) == pytest.approx(expected[features.index(f)]) for f in kept)
    assert min(expected) == pytest.approx(9.4)


def test_spectrum_on_grid_with_the_intensity_rule(top_run):
    """
    intensity_half_widths, unscaled, dynrange 101: every box 4.7 * sqrt(100) * 1.1 = 51.7 cm-1. With cut_tails=True
    it cuts more than the amplitude rule: against all_features_on_grid, 2.6 floors off (floor: max / 101);
    amplitude boxes with the same dynrange: 0.03 floors.
    """
    spec = spectrum_on_grid(top_run.result, WINDOW_AROUND_STRONGEST, STEPS_1_CM,
                         box_rule=partial(intensity_half_widths, dynrange=101.), cut_tails=True)

    kept = spec.spectral_window.full_features + spec.spectral_window.contrib_features
    assert kept and all(get_box_extent(f) == pytest.approx(51.7) for f in kept)
    intensity = np.abs(spec.amplitude) ** 2
    reference = np.abs(all_features_on_grid(top_run.result.features, top_run.result.states, spec.axis_points)) ** 2
    assert np.max(np.abs(intensity - reference)) > reference.max() / 101.


def test_spectrum_on_grid_without_cut_puts_each_kept_feature_on_the_whole_grid(top_run):
    """
    cut_tails=False (default): the boxes only choose the features (the same ones as with cuts), and each
    kept feature is on the whole grid: all_features_on_grid of the kept features. With cut_tails=True the
    contrib features' tails end at their boxes, so the two spectra differ.
    """
    cut = spectrum_on_grid(top_run.result, WINDOW_AROUND_STRONGEST, STEPS_1_CM, box_rule=AMPLITUDE_BOXES,
                           cut_tails=True)

    uncut = spectrum_on_grid(top_run.result, WINDOW_AROUND_STRONGEST, STEPS_1_CM, box_rule=AMPLITUDE_BOXES)

    kept = uncut.spectral_window.full_features + uncut.spectral_window.contrib_features
    assert kept == cut.spectral_window.full_features + cut.spectral_window.contrib_features
    np.testing.assert_allclose(uncut.amplitude, all_features_on_grid(kept, top_run.result.states, uncut.axis_points))
    assert not np.allclose(uncut.amplitude, cut.amplitude)


def test_spectrum_on_grid_without_cut_keeping_every_feature_is_the_uncut_spectrum(top_run):
    """A box rule that keeps every feature (half-width 1e6 cm-1), no cut: every feature on every grid point."""
    keep_all = lambda window, features: [1e6] * len(features)

    spec = spectrum_on_grid(top_run.result, WINDOW_AROUND_STRONGEST, STEPS_1_CM, box_rule=keep_all, cut_tails=False)

    np.testing.assert_allclose(spec.amplitude,
                               all_features_on_grid(top_run.result.features, top_run.result.states, spec.axis_points))


def test_spectrum_on_grid_with_a_new_width_reuses_the_rows(formaldehyde_molsys):
    """replace(..., propagator_param=8.) keeps the table (no new rows) and gives what a new run with 8 cm-1 gives"""
    first = evaluate_formaldehyde(formaldehyde_molsys, 5.)

    wider = spectrum_on_grid(replace(first, propagator_param=8.), WINDOW_AROUND_STRONGEST, STEPS_1_CM,
                          box_rule=AMPLITUDE_BOXES)

    assert wider.feature_result.table is first.table
    assert all(f.propagator_param == 8. for f in wider.feature_result.features)
    new_run = spectrum_on_grid(evaluate_formaldehyde(formaldehyde_molsys, 8.), WINDOW_AROUND_STRONGEST, STEPS_1_CM,
                            box_rule=AMPLITUDE_BOXES)
    np.testing.assert_allclose(wider.amplitude, new_run.amplitude)


def test_spectrum_on_grid_on_another_window_reuses_the_features(formaldehyde_molsys):
    """Same features, another window and grid: what a new run on that window gives"""
    result = evaluate_formaldehyde(formaldehyde_molsys, 5.)
    other, sizes = Box({'A': (2700., 2750.), 'B': (-60., 0.)}), {'A': 51, 'B': 61}

    again = spectrum_on_grid(result, other, sizes, box_rule=AMPLITUDE_BOXES)

    assert again.feature_result is result
    new_run = spectrum_on_grid(evaluate_formaldehyde(formaldehyde_molsys, 5.), other, sizes, box_rule=AMPLITUDE_BOXES)
    np.testing.assert_allclose(again.amplitude, new_run.amplitude)


def test_spectrum_on_grid_takes_the_box_rule_by_keyword(top_run):
    with pytest.raises(TypeError, match='positional'):
        spectrum_on_grid(top_run.result, WINDOW_AROUND_STRONGEST, STEPS_1_CM, AMPLITUDE_BOXES)  # type: ignore


def test_spectrum_on_grid_needs_a_box_rule(top_run):
    """no default: both rules need the plot's dynrange, and the plot decides what it shows, so the caller passes it"""
    with pytest.raises(TypeError, match='box_rule'):
        spectrum_on_grid(top_run.result, WINDOW_AROUND_STRONGEST, STEPS_1_CM)  # type: ignore
