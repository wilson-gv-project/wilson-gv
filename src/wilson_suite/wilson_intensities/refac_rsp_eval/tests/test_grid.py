"""
grid.py — boxes, spectral features, windows and box clustering. Runs with zero molecular data.

Tests marked xfail(strict=True) pin known bugs: they start passing (and so fail the run) once the bug is fixed,
which is the reminder to drop the marker.
----


**Bugs the tests found in `grid.py`**
1. [grid.py:939](src/wilson_suite/wilson_intensities/refac_rsp_eval/grid.py#L939): `RectangularDomain.from_features` passes the features themselves to `Box.union`, which expects boxes. The call always crashes, so `SpectralWindow.find_clusters_by_featboxes` crashes too. The old copy in `amplitudes/spectrum_composition.py:818` has the fix: `Box.union([f.feat_box for f in feats])`.
2. [grid.py:442](src/wilson_suite/wilson_intensities/refac_rsp_eval/grid.py#L442): `SpectralFeature.union` does not pass `lineshape_parameter` to the merged feature. The merged feature ends up with no box.
3. [grid.py:651](src/wilson_suite/wilson_intensities/refac_rsp_eval/grid.py#L651): `dress_these_with_boxes` removes a weak feature with `list.remove`. `list.remove` deletes the first *equal* feature, and feature equality ignores amplitude. Example: a strong and a weak feature at the same spot with the same terms. The strong one gets removed and the weak one stays.
4. [grid.py:404](src/wilson_suite/wilson_intensities/refac_rsp_eval/grid.py#L404): `normalize_coeffs_to_max` raises "max_feat_coeff is None" when any feature has amplitude 0. The message is wrong.
5. [grid.py:163](src/wilson_suite/wilson_intensities/refac_rsp_eval/grid.py#L163): `Box.contains` always crashes. The code is marked UNUSED.
6. [grid.py:708](src/wilson_suite/wilson_intensities/refac_rsp_eval/grid.py#L708): `apply_magn_cond_filter` returns an empty list for an unknown condition instead of raising an error. The code has a FIXME for this.

**What you need to do**
1. Decide about `sample_grid`. For the range 0 to 10 with 10 points, the grid runs 0, 1.1, …, 9.9, so the last point is 9.9, not 10. The current test only checks that all points stay inside the range. If the grid should end at 10, tell me and I'll tighten the test.
2. Fix the bugs above when you are ready, and delete each fixed test's `xfail` line.
3. `test_eval_avrg_per_indexdict` in `test_evaluate.py` still fails. The test is your `assert False` placeholder.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval.grid import (
    Box,
    RectangularDomain,
    SpectralFeature,
    SpectralWindow,
    TermParametersChoice,
    compute_box_adjacency,
    connected_components_from_adjacency,
    features_to_clusters,
    points_to_bounds,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ParameterSet,
    ResLocPoint,
    ResonanceMotif,
)

CM_TO_AU = 4.556335e-6  # 1 cm-1 in Hartree


def tpc(term_ids: tuple = (0,), **params) -> TermParametersChoice:
    """One term group with a single parameter choice, e.g. tpc(a=0, b=1)."""
    return TermParametersChoice(res_motif=ResonanceMotif(()),
                                states_parameters=(ParameterSet(params),),
                                term_ids=term_ids)


def feat(gamma: float | None = 1.0, amp: float | None = 1.0, terms: tuple | None = None,
         **coords: float) -> SpectralFeature:
    """A feature at the given coordinates, e.g. feat(A=100.). gamma in cm-1; its box is location +- gamma."""
    return SpectralFeature(location=ResLocPoint(coords),
                           term_contributions=(tpc(a=0),) if terms is None else terms,
                           lineshape_parameter=gamma,
                           amplitude_coeff=amp)


def box_halfwidth(f: SpectralFeature, axis: str = 'A') -> float:
    mn, mx = f.feat_box.bounds[axis] # type: ignore
    return (mx - mn) / 2


## points_to_bounds ---------------------------------------------------------

def test_points_to_bounds_pads_every_axis_by_halfwidth():
    assert points_to_bounds([{'A': 10., 'B': 20.}], halfwidth=2.) == [{'A': (8., 12.), 'B': (18., 22.)}]


## Box: construction --------------------------------------------------------

def test_box_sorts_axes_by_name():
    box = Box({'B': (0., 1.), 'A': (2., 3.)})

    assert box.axes == ('A', 'B')
    assert list(box.bounds) == ['A', 'B']
    assert box.ndim == 2


@pytest.mark.parametrize('bad', [[0., 1.], (0., 1., 2.), (0.,)])
def test_box_rejects_bounds_that_are_not_a_pair(bad):
    with pytest.raises(ValueError, match=r'expected \(min, max\)'):
        Box({'A': bad})


def test_box_rejects_non_numeric_bounds():
    with pytest.raises(TypeError):
        Box({'A': ('0', 1.)})


def test_box_rejects_min_above_max():
    with pytest.raises(ValueError, match='min'):
        Box({'A': (2., 1.)})


def test_box_equality_and_hash_ignore_input_order():
    b1 = Box({'A': (0., 1.), 'B': (2., 3.)})
    b2 = Box({'B': (2., 3.), 'A': (0., 1.)})

    assert b1 == b2
    assert hash(b1) == hash(b2)


## Box: operations ----------------------------------------------------------

def test_box_expand_returns_new_box_and_pads_only_given_axes():
    box = Box({'A': (0., 1.), 'B': (0., 1.)})

    bigger = box.expand({'A': 0.5})

    assert bigger.bounds == {'A': (-0.5, 1.5), 'B': (0., 1.)}
    assert box.bounds == {'A': (0., 1.), 'B': (0., 1.)}


def test_box_expand_inplace_changes_the_box():
    box = Box({'A': (0., 1.)})

    assert box.expand({'A': 1.}, inplace=True) is box
    assert box.bounds == {'A': (-1., 2.)}


def test_box_intersect_returns_overlap_on_shared_axes():
    b1 = Box({'A': (0., 2.), 'B': (0., 2.)})

    assert b1.intersect(Box({'A': (1., 3.), 'B': (-1., 1.)})) == Box({'A': (1., 2.), 'B': (0., 1.)})
    assert b1.intersect(Box({'A': (1., 3.)})) == Box({'A': (1., 2.)})


@pytest.mark.parametrize('other', [
    Box({'A': (5., 6.)}),  # apart
    Box({'A': (1., 2.)}),  # touching
    Box({'B': (0., 1.)}),  # no shared axis
])
def test_box_intersect_returns_none_without_overlap(other):
    assert Box({'A': (0., 1.)}).intersect(other) is None


def test_box_union_spans_all_boxes_and_all_axes():
    union = Box.union([Box({'A': (0., 1.)}), Box({'A': (5., 6.), 'B': (2., 3.)})])

    assert union == Box({'A': (0., 6.), 'B': (2., 3.)})


def test_box_contains_box():
    outer = Box({'A': (0., 10.)})

    assert outer.contains_box(Box({'A': (2., 3.)}))
    assert outer.contains_box(outer)
    assert not outer.contains_box(Box({'A': (9., 11.)}))


def test_box_overlaps_excludes_touching_edges():
    box = Box({'A': (0., 1.)})

    assert box.overlaps(Box({'A': (0.5, 2.)}))
    assert not box.overlaps(Box({'A': (1., 2.)}))
    assert not box.overlaps(Box({'A': (3., 4.)}))


def test_box_contains_points():
    box = Box({'A': (0., 1.), 'B': (0., 1.)})
    np.testing.assert_array_equal(box.contains(np.array([[0.5, 0.5], [2., 0.5]])), [True, False])


## Box: relation to features ------------------------------------------------

@pytest.mark.parametrize('a, inside', [(5., True), (0., True), (10., True), (10.1, False), (-1., False)])
def test_box_contains_feature_by_location_includes_edges(a, inside):
    assert Box({'A': (0., 10.)}).contains_feature(feat(A=a)) == inside


def test_box_contains_feature_by_location_needs_same_dimensionality():
    with pytest.raises(ValueError, match='coords'):
        Box({'A': (0., 10.)}).contains_feature(feat(A=1., B=1.))


def test_box_contains_feature_by_box_checks_feature_box_overlap():
    box = Box({'A': (0., 10.)})

    assert box.contains_feature(feat(A=10.5, gamma=1.), mode='box')
    assert not box.contains_feature(feat(A=11., gamma=1.), mode='box')  # boxes only touch


def test_box_contains_feature_by_box_needs_a_feature_box():
    with pytest.raises(ValueError, match='box'):
        Box({'A': (0., 10.)}).contains_feature(feat(A=5., gamma=None), mode='box')


def test_box_contains_feature_rejects_unknown_mode():
    with pytest.raises(ValueError, match='modes'):
        Box({'A': (0., 10.)}).contains_feature(feat(A=5.), mode='circle')


@pytest.mark.parametrize('a, contributing', [
    (5., False),    # inside -> a full feature, not a contributing one
    (11., True),    # outside, within 2*gamma
    (12., True),    # exactly 2*gamma away
    (12.5, False),  # beyond 2*gamma
    (-2., True),
])
def test_box_contributing_feature_is_outside_but_within_two_gamma(a, contributing):
    assert Box({'A': (0., 10.)}).contributing_feature(feat(A=a, gamma=1.)) == contributing


def test_box_contributing_feature_needs_lineshape_parameter():
    with pytest.raises(ValueError, match='lineshape_parameter'):
        Box({'A': (0., 10.)}).contributing_feature(feat(A=11., gamma=None))


## TermParametersChoice -----------------------------------------------------

def test_tpc_equality_uses_term_ids_and_parameters_not_motif():
    other_motif = ResonanceMotif.from_tuples((((('a',), ()), ('A',)),))
    t1 = tpc(term_ids=(1,), a=0)
    t2 = TermParametersChoice(other_motif, (ParameterSet({'a': 0}),), (1,))

    assert t1 == t2
    assert hash(t1) == hash(t2)
    assert t1 != tpc(term_ids=(2,), a=0)
    assert t1 != tpc(term_ids=(1,), a=1)


def test_tpc_orders_by_term_count_then_term_ids_then_parameters():
    assert tpc(term_ids=(9,), a=5) < tpc(term_ids=(0, 1), a=0)
    assert tpc(term_ids=(0,), a=5) < tpc(term_ids=(1,), a=0)
    assert tpc(term_ids=(0,), a=0) < tpc(term_ids=(0,), a=1)


def test_tpc_sort_parameters_returns_sorted_copy():
    unsorted = TermParametersChoice(ResonanceMotif(()), (ParameterSet({'a': 2}), ParameterSet({'a': 0})), (0,))

    result = unsorted.sort_parameters()

    assert result.states_parameters == (ParameterSet({'a': 0}), ParameterSet({'a': 2}))
    assert unsorted.states_parameters[0] == ParameterSet({'a': 2})


def test_check_states_parameters_returns_the_single_shared_choice():
    result = TermParametersChoice.check_states_parameters((tpc(term_ids=(0,), a=0, b=1),
                                                           tpc(term_ids=(1,), a=0, b=1)))

    assert result == {'a': 0, 'b': 1, 'zero': 'zero'}


def test_check_states_parameters_returns_none_for_no_choices():
    assert TermParametersChoice.check_states_parameters(()) is None


def test_check_states_parameters_rejects_different_choices():
    with pytest.raises(ValueError, match='2 different'):
        TermParametersChoice.check_states_parameters((tpc(a=0), tpc(a=1)))


## SpectralFeature: basics --------------------------------------------------

def test_feature_gets_box_of_halfwidth_gamma():
    assert feat(A=100., B=50., gamma=2.).feat_box == Box({'A': (98., 102.), 'B': (48., 52.)})


def test_feature_without_gamma_has_no_box():
    assert feat(A=100., gamma=None).feat_box is None


def test_feature_equality_ignores_amplitude():
    assert feat(A=1., amp=1.) == feat(A=1., amp=5.)
    assert feat(A=1.) != feat(A=2.)
    assert feat(A=1., gamma=1.) != feat(A=1., gamma=2.)
    assert feat(A=1.) != feat(A=1., terms=(tpc(a=1),))


def test_feature_orders_by_absolute_amplitude():
    assert feat(A=0., amp=1.) < feat(A=0., amp=-2.)
    assert not feat(A=0., amp=-3.) < feat(A=0., amp=2.)


def test_feature_order_needs_amplitudes():
    with pytest.raises(ValueError, match='amplitude_coeff'):
        sorted([feat(A=0., amp=None), feat(A=0., amp=1.)])


def test_feature_param_set_comes_from_terms_without_zero():
    assert feat(A=0., terms=(tpc(a=0, b=1),)).param_set == {'a': 0, 'b': 1}


def test_feature_param_set_is_none_without_terms():
    assert feat(A=0., terms=()).param_set is None


def test_feature_param_set_setter_overrides_terms():
    f = feat(A=0., terms=(tpc(a=0),))

    f.param_set = {'a': 7}

    assert f.param_set == {'a': 7}


def test_feature_anharm_contributions_sums_per_anharmonicity():
    term_map = {'t1': SimpleNamespace(anharmonicity='mech'),
                't2': SimpleNamespace(anharmonicity='mech'),
                't3': SimpleNamespace(anharmonicity='el')}
    f = feat(A=0.)
    f.term_contrib_by_id = {'t1': (1.,), 't2': (2.,), 't3': (5.,)}

    assert f.anharm_contributions(term_map) == {'mech': 3., 'el': 5.}


def test_feature_anharm_contributions_is_empty_without_term_contributions():
    assert feat(A=0.).anharm_contributions({}) == {}


def test_feature_get_res_motifs_lists_one_motif_per_term_group():
    assert feat(A=0., terms=(tpc(a=0), tpc(a=1))).get_res_motifs() == [ResonanceMotif(()), ResonanceMotif(())]
    assert feat(A=0., terms=()).get_res_motifs() == []


## SpectralFeature: list helpers --------------------------------------------

def test_sort_by_params_orders_by_first_term_group():
    f_high = feat(A=0., terms=(tpc(a=1),))
    f_low = feat(A=1., terms=(tpc(a=0),))

    assert SpectralFeature.sort_by_params([f_high, f_low]) == [f_low, f_high]


def test_normalize_coeffs_divides_by_largest_absolute_amplitude():
    feats = [feat(A=0., amp=2.), feat(A=1., amp=-4.)]

    result = SpectralFeature.normalize_coeffs_to_max(feats)

    assert [f.amplitude_coeff for f in result] == [0.5, -1.]
    assert [f.amplitude_coeff for f in feats] == [2., -4.]


def test_normalize_coeffs_uses_external_max():
    result = SpectralFeature.normalize_coeffs_to_max([feat(A=0., amp=2.)], external_max=-8.)

    assert result[0].amplitude_coeff == 0.25


def test_normalize_coeffs_rejects_missing_amplitude():
    with pytest.raises(ValueError, match='no amplitude_coeff'):
        SpectralFeature.normalize_coeffs_to_max([feat(A=0., amp=None)], external_max=1.)


def test_normalize_coeffs_keeps_zero_amplitude():
    result = SpectralFeature.normalize_coeffs_to_max([feat(A=0., amp=2.), feat(A=1., amp=0.)])

    assert result[1].amplitude_coeff == 0.


def test_get_feats_with_params_matches_first_term_group():
    f0 = feat(A=0., terms=(tpc(a=0, b=1),))
    f1 = feat(A=1., terms=(tpc(a=1, b=1),))

    assert SpectralFeature.get_feats_with_params([f0, f1], {'a': 0, 'b': 1}) == [f0]


def test_share_location():
    assert SpectralFeature.share_location([feat(A=1., amp=1.), feat(A=1., amp=2.)])
    assert not SpectralFeature.share_location([feat(A=1.), feat(A=2.)])
    with pytest.raises(ValueError):
        SpectralFeature.share_location([feat(A=1.)])


## SpectralFeature: union ---------------------------------------------------

def test_feature_union_adds_amplitudes_and_joins_terms():
    f1 = feat(A=1., amp=1., terms=(tpc(term_ids=(0,), a=0),))
    f2 = feat(A=1., amp=2., terms=(tpc(term_ids=(1,), a=0),))

    merged = f1.union(f2)

    assert merged.location == f1.location
    assert merged.amplitude_coeff == 3.
    assert merged.term_contributions == f1.term_contributions + f2.term_contributions


@pytest.mark.parametrize('other', [feat(A=2.), feat(A=1., gamma=3.)])
def test_feature_union_needs_same_location_and_gamma(other):
    with pytest.raises(ValueError, match='Union'):
        feat(A=1.).union(other)


def test_feature_union_needs_amplitudes():
    with pytest.raises(ValueError, match='amplitude_coeff'):
        feat(A=1., amp=None).union(feat(A=1.))


def test_feature_union_keeps_lineshape_parameter():
    merged = feat(A=1., gamma=2.).union(feat(A=1., gamma=2.))

    assert merged.lineshape_parameter == 2.
    assert merged.feat_box is not None


## SpectralFeature: filter_to_spec_window -----------------------------------

def test_filter_to_spec_window_splits_full_and_contributing_features():
    window = SpectralWindow(Box({'A': (0., 10.)}))
    inside, near, far = feat(A=5.), feat(A=11.), feat(A=50.)

    result = SpectralFeature.filter_to_spec_window([inside, near, far], window)

    assert result.full_features == [inside]
    assert result.contrib_features == [near]
    assert result.full_features[0].feat_type == 'full'
    assert result.contrib_features[0].feat_type == 'contributing'
    assert window.full_features == [] and window.contrib_features == []


## SpectralFeature: intensity -----------------------------------------------

@pytest.mark.parametrize('coords, gamma, expected', [
    ({'A': 100.}, 5., 4. / (5. * CM_TO_AU) ** 2),             # gamma > 1e-5 is cm-1 and gets converted
    ({'A': 100., 'B': 200.}, 5., 4. / (5. * CM_TO_AU) ** 4),  # one 1/gamma factor per axis
    ({'A': 100.}, 1e-6, 4. / 1e-6 ** 2),                      # gamma <= 1e-5 is taken as au
])
def test_intensity_is_abs_squared_of_amplitude_over_gamma_power(coords, gamma, expected):
    assert feat(amp=2., gamma=gamma, **coords).get_intensity() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize('amp, gamma', [(None, 1.), (1., None)])
def test_intensity_needs_amplitude_and_gamma(amp, gamma):
    with pytest.raises(ValueError):
        feat(A=0., amp=amp, gamma=gamma).get_intensity()


def test_intensity_supports_only_abs_squared():
    with pytest.raises(NotImplementedError):
        feat(A=0.).get_intensity('real()')


def test_max_intensity_feat_by_intensity_or_by_amplitude():
    narrow = feat(A=0., amp=2., gamma=1.)  # intensity ~ 2**2 / 1**2 = 4
    broad = feat(A=1., amp=3., gamma=2.)   # intensity ~ 3**2 / 2**2 = 2.25

    assert SpectralFeature.get_max_intensity_feat([broad, narrow]) is narrow
    assert SpectralFeature.get_max_intensity_feat([broad, narrow], intensity_expr=None) is broad


def test_max_intensity_feat_rejects_empty_list():
    with pytest.raises(ValueError):
        SpectralFeature.get_max_intensity_feat([])


## SpectralFeature: dress_these_with_boxes ----------------------------------
# A Lorentzian drops to 1/dynrange of its peak at distance gamma*sqrt(dynrange - 1).
# dynrange = 101 -> 10*gamma ; dynrange = 26 -> 5*gamma

def test_dress_box_halfwidth_is_gamma_times_sqrt_dynrange_minus_one():
    f = feat(A=100., gamma=5.)
    top = f.get_intensity()

    [dressed] = SpectralFeature.dress_these_with_boxes([f], top, top / 101, box_range_safety_margin=0.)

    assert dressed.feat_box.bounds['A'] == pytest.approx((50., 150.)) # type: ignore


def test_dress_safety_margin_widens_the_box():
    f = feat(A=0., gamma=5.)
    top = f.get_intensity()

    [dressed] = SpectralFeature.dress_these_with_boxes([f], top, top / 101, box_range_safety_margin=0.1)

    assert box_halfwidth(dressed) == pytest.approx(55.)


def test_dress_drops_features_below_min_intensity():
    strong, weak = feat(A=0., amp=1.), feat(A=50., amp=1e-3)
    top = strong.get_intensity()

    assert SpectralFeature.dress_these_with_boxes([strong, weak], top, top / 100) == [strong]


def test_dress_minimum_padding_keeps_weak_features_and_sets_a_floor():
    strong, weak = feat(A=0., amp=1., gamma=5.), feat(A=500., amp=1e-3, gamma=5.)
    top = strong.get_intensity()

    result = SpectralFeature.dress_these_with_boxes([strong, weak], top, top / 101,
                                                    box_range_safety_margin=0.,
                                                    scale_wrt_max_intensity=True,
                                                    minimum_box_padding=20.)

    assert [box_halfwidth(f) for f in result] == pytest.approx([50., 20.])


def test_dress_scaled_boxes_shrink_for_weaker_features():
    # the feature is at half of max_intensity: dynrange 52 is scaled to 26
    f = feat(A=0., gamma=5.)
    top = 2 * f.get_intensity()

    [scaled] = SpectralFeature.dress_these_with_boxes([f], top, top / 52, box_range_safety_margin=0.,
                                                      scale_wrt_max_intensity=True)
    [unscaled] = SpectralFeature.dress_these_with_boxes([f], top, top / 52, box_range_safety_margin=0.)

    assert box_halfwidth(scaled) == pytest.approx(25.)
    assert box_halfwidth(unscaled) == pytest.approx(5. * 51 ** 0.5)


def test_dress_lineshape_parameter_overrides_feature_gamma():
    f = feat(A=0., gamma=5.)
    top = f.get_intensity()

    [dressed] = SpectralFeature.dress_these_with_boxes([f], top, top / 101, lineshape_parameter=2.,
                                                       box_range_safety_margin=0.)

    assert dressed.lineshape_parameter == 2.
    assert box_halfwidth(dressed) == pytest.approx(20.)


def test_dress_rejects_feature_above_max_intensity():
    f = feat(A=0.)
    top = f.get_intensity()

    with pytest.raises(ValueError, match='higher intensity'):
        SpectralFeature.dress_these_with_boxes([f], top / 2, top / 200)


def test_dress_does_not_change_input_features():
    f = feat(A=0., gamma=5.)
    top = f.get_intensity()

    SpectralFeature.dress_these_with_boxes([f], top, top / 101)

    assert f.feat_box == Box({'A': (-5., 5.)})


@pytest.mark.xfail(strict=True, reason='list.remove() drops the first *equal* feature, and equality ignores amplitude')
def test_dress_drops_the_weak_one_of_two_equal_features():
    strong, weak = feat(A=0., amp=1.), feat(A=0., amp=1e-3)  # same location, gamma and terms
    top = strong.get_intensity()

    result = SpectralFeature.dress_these_with_boxes([strong, weak], max_intensity=top, min_intensity=(top / 100))

    assert [f.amplitude_coeff for f in result] == [1.]


## SpectralFeature: apply_magn_cond_filter ----------------------------------

def test_magn_cond_filter_none_returns_input():
    feats = [feat(A=0., B=-1.)]

    assert SpectralFeature.apply_magn_cond_filter(feats, None, 0.) is feats # type: ignore


def test_magn_cond_filter_keeps_B_above_margin():
    feats = [feat(A=0., B=-1.), feat(A=0., B=0.5), feat(A=0., B=2.)]

    assert SpectralFeature.apply_magn_cond_filter(feats, (('B',),), magn_conditions_margin=1.) == [feats[2]]


def test_magn_cond_filter_keeps_B_minus_A_above_margin():
    feats = [feat(A=5., B=3.), feat(A=5., B=7.)]

    assert SpectralFeature.apply_magn_cond_filter(feats, (('-A', 'B'),), magn_conditions_margin=0.) == [feats[1]]


def test_magn_cond_filter_rejects_unknown_condition():
    with pytest.raises(ValueError):
        SpectralFeature.apply_magn_cond_filter([feat(A=0., B=1.)], (('A',),), 0.)


## Box clustering -----------------------------------------------------------

def test_adjacency_is_symmetric_with_false_diagonal():
    boxes = [Box({'A': (0., 2.)}), Box({'A': (1., 3.)}), Box({'A': (10., 11.)})]

    np.testing.assert_array_equal(compute_box_adjacency(boxes), [[False, True, False],
                                                                 [True, False, False],
                                                                 [False, False, False]])


def test_adjacency_counts_touching_boxes_only_when_inclusive():
    boxes = [Box({'A': (0., 1.)}), Box({'A': (1., 2.)})]

    assert compute_box_adjacency(boxes, touch_inclusive=True)[0, 1]
    assert not compute_box_adjacency(boxes, touch_inclusive=False)[0, 1]


def test_adjacency_needs_overlap_on_every_checked_axis():
    boxes = [Box({'A': (0., 2.), 'B': (0., 1.)}), Box({'A': (1., 3.), 'B': (5., 6.)})]

    assert not compute_box_adjacency(boxes)[0, 1]
    assert compute_box_adjacency(boxes, axis_order=('A',))[0, 1]


def test_adjacency_of_no_boxes_is_empty():
    assert compute_box_adjacency([]).shape == (0, 0)


def test_connected_components_follow_chains():
    # a-b and b-c are linked, d is alone
    adjacency = np.zeros((4, 4), dtype=bool)
    adjacency[0, 1] = adjacency[1, 0] = True
    adjacency[1, 2] = adjacency[2, 1] = True

    clusters = connected_components_from_adjacency(adjacency, ['a', 'b', 'c', 'd'])

    assert {k: sorted(v) for k, v in clusters.items()} == {0: ['a', 'b', 'c'], 1: ['d']}


def test_features_to_clusters_groups_by_overlapping_boxes():
    # boxes: (-1, 1), (0.5, 2.5), (9, 11)
    f0, f1, f2 = feat(A=0.), feat(A=1.5), feat(A=10.)

    assert list(features_to_clusters([f0, f1, f2]).values()) == [[f0, f1], [f2]]


## SpectralWindow -----------------------------------------------------------

def test_window_bounds_ndim_and_widths():
    window = SpectralWindow(Box({'B': (0., 4.), 'A': (10., 20.)}))

    assert window.bounds == {'A': (10., 20.), 'B': (0., 4.)}
    assert window.ndim == 2
    assert window.widths == {'A': 10., 'B': 4.}


def test_sample_grid_shapes_and_ij_indexing():
    window = SpectralWindow(Box({'A': (0., 10.), 'B': (100., 200.)}))

    axes, grid = window.sample_grid({'A': 5, 'B': 3})

    assert (len(axes['A']), len(axes['B'])) == (5, 3)
    assert grid['A'].shape == grid['B'].shape == (5, 3)
    np.testing.assert_array_equal(grid['A'][:, 0], axes['A'])
    np.testing.assert_array_equal(grid['B'][0, :], axes['B'])


def test_sample_grid_is_evenly_spaced_from_min_and_stays_inside_box():
    axes, _ = SpectralWindow(Box({'A': (0., 10.)})).sample_grid({'A': 10})

    assert axes['A'][0] == 0.
    assert axes['A'][-1] <= 10.
    np.testing.assert_allclose(np.diff(axes['A']), np.diff(axes['A'])[0])


def test_sample_grid_needs_one_size_per_axis():
    with pytest.raises(ValueError, match='dimensionality'):
        SpectralWindow(Box({'A': (0., 10.)})).sample_grid({'A': 5, 'B': 5})


def test_window_dress_with_featboxes_uses_strongest_full_feature():
    strong = feat(A=5., amp=1., gamma=5.)
    weak = feat(A=6., amp=1e-3, gamma=5.)
    near = feat(A=12., amp=0.5, gamma=5.)
    window = SpectralWindow(Box({'A': (0., 10.)}), full_features=[strong, weak], contrib_features=[near])

    dressed = window.dress_with_featboxes(dynrange=101.)

    assert dressed.box == window.box
    assert dressed.full_features == [strong]
    assert dressed.contrib_features == [near]
    assert box_halfwidth(dressed.full_features[0]) == pytest.approx(5. * 10. * 1.1)


def test_window_find_clusters_by_featboxes():
    f0, f1, f2 = feat(A=0.), feat(A=1.5), feat(A=10.)
    window = SpectralWindow(Box({'A': (-5., 15.)}), full_features=[f0, f1, f2])

    domains = window.find_clusters_by_featboxes()

    assert [d.box for d in domains] == [Box({'A': (-1., 2.5)}), Box({'A': (9., 11.)})]


## RectangularDomain --------------------------------------------------------

def test_domain_from_features_spans_feature_boxes():
    f0, f1 = feat(A=0.), feat(A=5.)

    domain = RectangularDomain.from_features([f0, f1])

    assert domain.box == Box({'A': (-1., 6.)})
    assert domain.full_features == [f0, f1]


def test_domain_from_features_needs_feature_boxes():
    with pytest.raises(ValueError, match='feat_box'):
        RectangularDomain.from_features([feat(A=0., gamma=None)])
