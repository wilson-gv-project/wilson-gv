"""
features.py — term groups, spectral features, windows and domains. Runs with zero molecular data.

Tests marked xfail(strict=True) pin known bugs: they start passing (and so fail the run) once the bug is fixed,
which is the reminder to drop the marker.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    RectangularDomain,
    SpectralFeature,
    SpectralWindow,
    TermParametersChoice,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import (
    Box,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ParameterSet,
    ResonanceMotif,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
    AXES,
    NDIMS,
    box_halfwidth,
    coords,
    cube,
    feat,
    tpc,
)

CM_TO_AU = 4.556335e-6  # 1 cm-1 in Hartree
OTHER_MOTIF = ResonanceMotif.from_tuples((((('a',), ()), ('A',)),))  # tpc() uses the empty motif


## TermParametersChoice -----------------------------------------------------

def test_tpc_equality_and_hash_use_motif_term_ids_and_parameters():
    t1 = tpc(term_ids=(1,), a=0)

    assert t1 == tpc(term_ids=(1,), a=0)
    assert hash(t1) == hash(tpc(term_ids=(1,), a=0))
    assert t1 != TermParametersChoice(OTHER_MOTIF, (ParameterSet({'a': 0}),), (1,))
    assert t1 != tpc(term_ids=(2,), a=0)
    assert t1 != tpc(term_ids=(1,), a=1)


def test_tpc_is_not_equal_or_ordered_with_other_types():
    assert tpc() != (0,)
    assert tpc().__lt__((0,)) is NotImplemented


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

def test_feature_without_gamma_has_no_box():
    assert feat(A=100., gamma=None).feat_box is None


def test_feature_equality_ignores_amplitude():
    assert feat(A=1., amp=1.) == feat(A=1., amp=5.)
    assert feat(A=1.) != feat(A=2.)
    assert feat(A=1., gamma=1.) != feat(A=1., gamma=2.)
    assert feat(A=1.) != feat(A=1., terms=(tpc(a=1),))


def test_feature_equality_includes_term_motif():
    f1 = feat(A=0., terms=(tpc(term_ids=(1,), a=0),))
    f2 = feat(A=0., terms=(TermParametersChoice(OTHER_MOTIF, (ParameterSet({'a': 0}),), (1,)),))

    assert f1 != f2
    assert len({f1, f2}) == 2


def test_feature_hash_matches_equality():
    # equal features must hash equal: amplitude is ignored, and features without terms work too
    assert hash(feat(A=1., amp=1.)) == hash(feat(A=1., amp=5.))
    assert hash(feat(A=1., terms=())) == hash(feat(A=1., terms=()))
    assert len({feat(A=1.), feat(A=1.), feat(A=2.)}) == 2


def test_feature_is_not_equal_or_ordered_with_other_types():
    assert feat(A=0.) != 'feature'
    assert feat(A=0.).__lt__(1.) is NotImplemented # type: ignore


def test_feature_orders_by_absolute_amplitude():
    assert feat(A=0., amp=1.) < feat(A=0., amp=-2.)
    assert not feat(A=0., amp=-3.) < feat(A=0., amp=2.)


def test_feature_order_needs_amplitudes():
    with pytest.raises(ValueError, match='amplitude_coeff'):
        sorted([feat(A=0., amp=None), feat(A=0., amp=1.)])


@pytest.mark.xfail(strict=True, reason='amplitude 0 is treated as a missing amplitude')
def test_feature_order_accepts_zero_amplitude():
    assert feat(A=0., amp=0.) < feat(A=1., amp=1.)


def test_feature_param_set_comes_from_terms_without_zero():
    assert feat(A=0., terms=(tpc(a=0, b=1),)).param_set == {'a': 0, 'b': 1}


def test_feature_param_set_is_none_without_terms():
    assert feat(A=0., terms=()).param_set is None


def test_feature_param_set_needs_one_shared_choice():
    f = feat(A=0., terms=(tpc(a=0), tpc(a=1)))

    with pytest.raises(ValueError, match='different parameter choices'):
        _ = f.param_set


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


def test_normalize_coeffs_rejects_all_zero_amplitudes():
    with pytest.raises(ValueError):
        SpectralFeature.normalize_coeffs_to_max([feat(A=0., amp=0.), feat(A=1., amp=0.)])


def test_get_feats_with_params_matches_first_term_group():
    f0 = feat(A=0., terms=(tpc(a=0, b=1),))
    f1 = feat(A=1., terms=(tpc(a=1, b=1),))

    assert SpectralFeature.get_feats_with_params([f0, f1], {'a': 0, 'b': 1}) == [f0]


def test_get_feats_with_params_skips_features_without_terms():
    assert SpectralFeature.get_feats_with_params([feat(A=0., terms=())], {'a': 0}) == []


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


def test_feature_union_needs_amplitude_of_the_other_feature():
    with pytest.raises(ValueError, match='Other'):
        feat(A=1.).union(feat(A=1., amp=None))


def test_feature_union_keeps_lineshape_parameter():
    merged = feat(A=1., gamma=2.).union(feat(A=1., gamma=2.))

    assert merged.lineshape_parameter == 2.
    assert merged.feat_box is not None


def test_feature_union_does_not_change_inputs():
    f1, f2 = feat(A=1., amp=1.), feat(A=1., amp=2.)

    f1.union(f2)

    assert (f1.amplitude_coeff, f2.amplitude_coeff) == (1., 2.)


## SpectralFeature: relation to a box ---------------------------------------

@pytest.mark.parametrize('a, inside', [(5., True), (0., True), (10., True), (10.1, False), (-1., False)])
def test_feature_is_inside_by_location_includes_edges(a, inside):
    assert feat(A=a).is_inside(Box({'A': (0., 10.)})) == inside


def test_feature_is_inside_by_location_needs_same_dimensionality():
    with pytest.raises(ValueError, match='coords'):
        feat(A=1., B=1.).is_inside(Box({'A': (0., 10.)}))


def test_feature_is_inside_by_box_checks_feature_box_overlap():
    box = Box({'A': (0., 10.)})

    assert feat(A=10.5, gamma=1.).is_inside(box, mode='box')
    assert not feat(A=11., gamma=1.).is_inside(box, mode='box')  # boxes only touch


def test_feature_is_inside_by_box_needs_a_feature_box():
    with pytest.raises(ValueError, match='box'):
        feat(A=5., gamma=None).is_inside(Box({'A': (0., 10.)}), mode='box')


def test_feature_is_inside_rejects_unknown_mode():
    with pytest.raises(ValueError, match='modes'):
        feat(A=5.).is_inside(Box({'A': (0., 10.)}), mode='circle')


@pytest.mark.parametrize('a, contributing', [
    (5., False),    # inside -> a full feature, not a contributing one
    (11., True),    # outside, within 2*gamma
    (12., True),    # exactly 2*gamma away
    (12.5, False),  # beyond 2*gamma
    (-2., True),
])
def test_feature_contributes_to_box_when_outside_but_within_two_gamma(a, contributing):
    assert feat(A=a, gamma=1.).contributes_to(Box({'A': (0., 10.)})) == contributing


def test_feature_contributes_to_needs_same_dimensionality():
    with pytest.raises(ValueError, match='coords'):
        feat(A=11., B=1.).contributes_to(Box({'A': (0., 10.)}))


def test_feature_contributes_to_needs_lineshape_parameter():
    with pytest.raises(ValueError, match='lineshape_parameter'):
        feat(A=11., gamma=None).contributes_to(Box({'A': (0., 10.)}))


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


def test_filter_to_spec_window_does_not_change_input_features():
    f = feat(A=11.)

    first = SpectralFeature.filter_to_spec_window([f], SpectralWindow(Box({'A': (0., 10.)})))
    second = SpectralFeature.filter_to_spec_window([f], SpectralWindow(Box({'A': (10.5, 20.)})))

    assert f.feat_type is None
    assert first.contrib_features[0].feat_type == 'contributing'
    assert second.full_features[0].feat_type == 'full'


## SpectralFeature: intensity -----------------------------------------------

@pytest.mark.parametrize('location, gamma, expected', [
    ({'A': 100.}, 5., 4. / (5. * CM_TO_AU) ** 2),             # gamma > 1e-5 is cm-1 and gets converted
    ({'A': 100., 'B': 200.}, 5., 4. / (5. * CM_TO_AU) ** 4),  # one 1/gamma factor per axis
    ({'A': 100.}, 1e-6, 4. / 1e-6 ** 2),                      # gamma <= 1e-5 is taken as au
])
def test_intensity_is_abs_squared_of_amplitude_over_gamma_power(location, gamma, expected):
    assert feat(amp=2., gamma=gamma, **location).get_intensity() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize('amp, gamma', [(None, 1.), (1., None)])
def test_intensity_needs_amplitude_and_gamma(amp, gamma):
    with pytest.raises(ValueError):
        feat(A=0., amp=amp, gamma=gamma).get_intensity()


@pytest.mark.xfail(strict=True, reason='amplitude 0 is treated as a missing amplitude')
def test_intensity_of_zero_amplitude_is_zero():
    assert feat(A=0., amp=0.).get_intensity() == 0.


def test_intensity_supports_only_abs_squared():
    with pytest.raises(NotImplementedError):
        feat(A=0.).get_intensity('real()')


def test_max_intensity_feat_by_intensity_or_by_amplitude():
    narrow = feat(A=0., amp=2., gamma=1.)  # intensity ~ 2**2 / 1**2 = 4
    broad = feat(A=1., amp=3., gamma=2.)   # intensity ~ 3**2 / 2**2 = 2.25

    assert SpectralFeature.get_max_intensity_feat([broad, narrow]) is narrow
    assert SpectralFeature.get_max_intensity_feat([broad, narrow], intensity_expr=None) is broad


def test_max_intensity_feat_by_amplitude_compares_absolute_values():
    positive, negative = feat(A=0., amp=2.), feat(A=1., amp=-3.)

    assert SpectralFeature.get_max_intensity_feat([positive, negative], intensity_expr=None) is negative


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


def test_dress_rejects_zero_lineshape_override():
    f = feat(A=0., gamma=5.)
    top = f.get_intensity()

    with pytest.raises(ValueError, match='lineshape_parameter'):
        SpectralFeature.dress_these_with_boxes([f], top, top / 101, lineshape_parameter=0.)


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


def test_magn_cond_filter_drops_features_exactly_at_margin():
    feats = [feat(A=0., B=1.), feat(A=2., B=3.)]

    assert SpectralFeature.apply_magn_cond_filter(feats, (('B',),), magn_conditions_margin=1.) == [feats[1]]
    assert SpectralFeature.apply_magn_cond_filter(feats, (('-A', 'B'),), magn_conditions_margin=1.) == []


def test_magn_cond_filter_rejects_unknown_condition():
    with pytest.raises(ValueError):
        SpectralFeature.apply_magn_cond_filter([feat(A=0., B=1.)], (('A',),), 0.)


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


def test_sample_grid_runs_from_min_to_max_with_both_edges():
    window = SpectralWindow(Box({'A': (0., 10.), 'B': (100., 200.)}))

    axes, _ = window.sample_grid({'A': 11, 'B': 5})

    np.testing.assert_allclose(axes['A'], [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    np.testing.assert_allclose(axes['B'], [100., 125., 150., 175., 200.])


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


@pytest.mark.xfail(strict=True, reason='max intensity is taken from full_features only, '
                                       'so a stronger contributing feature raises')
def test_window_dress_with_featboxes_allows_a_stronger_contributing_feature():
    weak_inside = feat(A=5., amp=0.1, gamma=5.)
    strong_near = feat(A=12., amp=1., gamma=5.)
    window = SpectralWindow(Box({'A': (0., 10.)}), full_features=[weak_inside], contrib_features=[strong_near])

    dressed = window.dress_with_featboxes(dynrange=101.)

    assert dressed.full_features == [weak_inside]
    assert dressed.contrib_features == [strong_near]


def test_window_find_clusters_by_featboxes_includes_contributing_features():
    # boxes (8.5, 10.5) and (10, 12) overlap
    inside, near = feat(A=9.5), feat(A=11.)
    window = SpectralWindow(Box({'A': (0., 10.)}), full_features=[inside], contrib_features=[near])

    [domain] = window.find_clusters_by_featboxes()

    assert domain.box == Box({'A': (8.5, 12.)})


## RectangularDomain --------------------------------------------------------

def test_domain_from_features_spans_feature_boxes():
    f0, f1 = feat(A=0.), feat(A=5.)

    domain = RectangularDomain.from_features([f0, f1])

    assert domain.box == Box({'A': (-1., 6.)})
    assert domain.full_features == [f0, f1]


def test_domain_from_features_needs_feature_boxes():
    with pytest.raises(ValueError, match='feat_box'):
        RectangularDomain.from_features([feat(A=0., gamma=None)])


def test_equal_domains_have_equal_hashes():
    # the old GridManager uses domains as dict keys
    d1 = RectangularDomain.from_features([feat(A=0.)])
    d2 = RectangularDomain.from_features([feat(A=0.)])

    assert d1 == d2
    assert hash(d1) == hash(d2)
    assert {d1: 'region'}[d2] == 'region'


def test_domain_add_methods_append_features():
    domain = RectangularDomain(Box({'A': (0., 10.)}))
    f0, f1, f2, f3 = feat(A=0.), feat(A=1.), feat(A=2.), feat(A=3.)

    domain.add_full_features([f0])
    domain.add_a_full_feature(f1)
    domain.add_contrib_features([f2])
    domain.add_a_contrib_feature(f3)

    assert domain.full_features == [f0, f1]
    assert domain.contrib_features == [f2, f3]


## N dimensions (1 to 4 axes) -----------------------------------------------
# The same checks for every spectrum dimensionality in NDIMS. `last` is the highest axis, e.g. 'C' for ndim=3.

@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_feature_box_is_location_plus_minus_gamma(ndim):
    # a different value on each axis: A=10, B=20, ...
    location = {ax: 10. * (i + 1) for i, ax in enumerate(AXES[:ndim])}

    assert feat(location, gamma=2.).feat_box == Box({ax: (v - 2., v + 2.) for ax, v in location.items()})


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_feature_is_inside_only_when_every_axis_is_inside(ndim):
    box, last = cube(ndim, 0., 10.), AXES[ndim - 1]

    assert feat(coords(ndim, 5.)).is_inside(box)
    assert feat(coords(ndim, 10.)).is_inside(box)  # corner
    for ax in AXES[:ndim]:
        assert not feat(coords(ndim, 5., **{ax: 11.})).is_inside(box)
    assert feat(coords(ndim, 5., **{last: 10.5}), gamma=1.).is_inside(box, mode='box')


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_feature_contributes_to_box_within_two_gamma_on_every_axis(ndim):
    box = cube(ndim, 0., 10.)

    assert not feat(coords(ndim, 5.), gamma=1.).contributes_to(box)  # inside
    assert feat(coords(ndim, 11.), gamma=1.).contributes_to(box)  # outside on every axis (corner), within 2*gamma
    for ax in AXES[:ndim]:
        assert feat(coords(ndim, 5., **{ax: 11.5}), gamma=1.).contributes_to(box)
        assert not feat(coords(ndim, 5., **{ax: 12.5}), gamma=1.).contributes_to(box)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_intensity_has_one_gamma_factor_per_axis(ndim):
    f = feat(coords(ndim, 100.), amp=2., gamma=5.)

    assert f.get_intensity() == pytest.approx(4. / (5. * CM_TO_AU) ** (2 * ndim), rel=1e-6)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_dress_box_halfwidth_is_the_same_on_every_axis(ndim):
    f = feat(coords(ndim, 100.), gamma=5.)
    top = f.get_intensity()

    [dressed] = SpectralFeature.dress_these_with_boxes([f], top, top / 101, box_range_safety_margin=0.)

    assert [box_halfwidth(dressed, ax) for ax in AXES[:ndim]] == pytest.approx([50.] * ndim)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_filter_to_spec_window_splits_full_and_contributing_features(ndim):
    window = SpectralWindow(cube(ndim, 0., 10.))
    inside, far = feat(coords(ndim, 5.)), feat(coords(ndim, 50.))
    near = feat(coords(ndim, 5., **{AXES[ndim - 1]: 11.}))

    result = SpectralFeature.filter_to_spec_window([inside, near, far], window)

    assert result.full_features == [inside]
    assert result.contrib_features == [near]


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_window_dress_with_featboxes_drops_weak_features(ndim):
    strong, weak = feat(coords(ndim, 5.), amp=1., gamma=5.), feat(coords(ndim, 6.), amp=1e-3, gamma=5.)
    window = SpectralWindow(cube(ndim, 0., 10.), full_features=[strong, weak])

    dressed = window.dress_with_featboxes(dynrange=101.)

    assert dressed.full_features == [strong]
    assert [box_halfwidth(dressed.full_features[0], ax) for ax in AXES[:ndim]] == pytest.approx([55.] * ndim)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_window_find_clusters_by_featboxes(ndim):
    # f0 and f1 boxes overlap on every axis; f2 sits on f0, except far away on the last axis
    f0, f1 = feat(coords(ndim, 0.)), feat(coords(ndim, 1.5))
    f2 = feat(coords(ndim, 0., **{AXES[ndim - 1]: 10.}))
    window = SpectralWindow(cube(ndim, -5., 15.), full_features=[f0, f1, f2])

    domains = window.find_clusters_by_featboxes()

    assert [d.box for d in domains] == [cube(ndim, -1., 2.5), f2.feat_box]
    assert [d.full_features for d in domains] == [[f0, f1], [f2]]


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_sample_grid_has_both_edges_and_ij_indexing(ndim):
    window = SpectralWindow(cube(ndim, 0., 10.))

    axes, grid = window.sample_grid({ax: 3 for ax in AXES[:ndim]})

    assert list(axes) == list(AXES[:ndim])
    for i, ax in enumerate(AXES[:ndim]):
        np.testing.assert_allclose(axes[ax], [0., 5., 10.])
        assert grid[ax].shape == (3,) * ndim
        for j in range(ndim):
            # ij indexing: grid[ax] steps by 5 along its own array dimension i and is constant along the others
            np.testing.assert_allclose(np.diff(grid[ax], axis=j), 5. if j == i else 0.)

