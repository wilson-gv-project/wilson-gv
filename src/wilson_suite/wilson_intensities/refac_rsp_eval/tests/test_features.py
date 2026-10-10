"""
features.py — contribution rows, spectral features, windows and domains. Runs with zero molecular data.

Tests marked xfail(strict=True) pin known bugs: they start passing (and so fail the run) once the bug is fixed,
which is the reminder to drop the marker.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    ContributionTable,
    RectangularDomain,
    SpectralFeature,
    SpectralWindow,
    features_from_rows,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import (
    Box,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ParameterSet,
    ResLocPoint,
    ResonanceMotif,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
    AXES,
    NDIMS,
    coords,
    cube,
    feat,
    get_box_extent,
    row,
)

CM_TO_AU = 4.556335e-6  # 1 cm-1 in Hartree
OTHER_MOTIF = ResonanceMotif.from_tuples((((('a',), ()), ('A',)),))  # row() uses the empty motif


## ContributionTable --------------------------------------------------------
# R0, R1: empty motif; R2: OTHER_MOTIF. R0, R2: a=0; R1: a=1. R2 also has axis B.

R0 = row(0.3, term_id=0, location=ResLocPoint({'A': 1650.}), a=0)
R1 = row(-0.1, term_id=1, location=ResLocPoint({'A': 1700.}), a=1)
R2 = row(0.2, term_id=2, motif=OTHER_MOTIF, location=ResLocPoint({'A': 1600., 'B': 10.}), a=0)


def test_table_keeps_rows_in_order():
    table = ContributionTable([R0, R1, R2])

    assert list(table) == [R0, R1, R2]
    assert len(table) == 3


def test_table_where_keeps_matching_rows_and_returns_a_table():
    table = ContributionTable([R0, R1, R2])

    kept = table.where(lambda r: r.coeff > 0)

    assert isinstance(kept, ContributionTable)
    assert list(kept) == [R0, R2]
    assert len(table) == 3                  # the original table is unchanged


def test_table_group_by_keeps_first_appearance_order():
    groups = ContributionTable([R0, R1, R2]).group_by(lambda r: r.coeff > 0)

    assert list(groups) == [True, False]
    assert all(isinstance(g, ContributionTable) for g in groups.values())
    assert list(groups[True]) == [R0, R2]
    assert list(groups[False]) == [R1]


def test_table_by_params_and_by_motif():
    table = ContributionTable([R0, R1, R2])

    assert {ps: list(g) for ps, g in table.by_params().items()} == {ParameterSet({'a': 0}): [R0, R2],
                                                                     ParameterSet({'a': 1}): [R1]}
    assert {m: list(g) for m, g in table.by_motif().items()} == {ResonanceMotif(()): [R0, R1],
                                                                 OTHER_MOTIF: [R2]}


def test_table_by_location_rounds_to_steps_of_tol_cm():
    """Steps, not a distance: rows 0.002 apart can be split, rows 0.008 apart can share a step."""
    r_004, r_006, r_014 = (row(location=ResLocPoint({'A': v}), a=0) for v in (1650.004, 1650.006, 1650.014))

    groups = ContributionTable([r_004, r_006, r_014]).by_location(tol_cm=0.01)

    assert [list(g) for g in groups.values()] == [[r_004], [r_006, r_014]]


def test_table_by_location_never_joins_different_axes():
    on_a = row(location=ResLocPoint({'A': 1.}), a=0)
    on_ab = row(location=ResLocPoint({'A': 1., 'B': 0.}), a=0)

    assert len(ContributionTable([on_a, on_ab]).by_location()) == 2


def test_table_axis_range_skips_rows_without_the_axis():
    table = ContributionTable([R0, R1, R2])

    assert table.axis_range('A') == (1600., 1700.)
    assert table.axis_range('B') == (10., 10.)


def test_table_axis_range_needs_a_row_with_the_axis():
    with pytest.raises(ValueError, match="no row has axis 'C'"):
        ContributionTable([R0, R1, R2]).axis_range('C')


## SpectralFeature: basics --------------------------------------------------

def test_feature_has_no_box_until_dressed():
    f = feat(A=100., gamma=5.)
    top = f.get_intensity()

    [dressed] = SpectralFeature.dress_these_with_boxes([f], top, top / 101)

    assert f.feat_box is None
    assert dressed.feat_box is not None


def test_replace_keeps_the_box():
    """dataclasses.replace runs __post_init__ again; that used to reset a dressed box to location +- gamma."""
    dressed = feat(A=100., box_extent=50.)

    assert replace(dressed, scale=0.5).feat_box == Box({'A': (50., 150.)})


def test_feature_amplitude_is_scale_times_sum_of_row_coeffs():
    f = feat(A=0., rows=(row(0.3, a=0), row(-0.12, a=1)))

    assert f.amplitude_coeff == pytest.approx(0.18)
    f.scale = 2.
    assert f.amplitude_coeff == pytest.approx(0.36)


def test_feature_without_rows_has_no_amplitude():
    assert feat(A=0., amp=None).amplitude_coeff is None


@pytest.mark.parametrize('coeffs, expected', [((0.5, -0.4), 0.1 / 0.9),
                                              ((0.5, -0.5), 0.),
                                              ((0.3, 0.5), 1.),
                                              ((-0.3,), 1.)])
def test_feature_net_fraction_is_abs_sum_over_sum_of_abs(coeffs, expected):
    f = feat(A=0., rows=tuple(row(c, term_id=i, a=0) for i, c in enumerate(coeffs)))

    assert f.net_fraction == pytest.approx(expected)
    assert replace(f, scale=3.).net_fraction == pytest.approx(expected)


@pytest.mark.parametrize('rows', [(), (row(0., a=0), row(0., term_id=1, a=0))])
def test_feature_net_fraction_is_none_without_rows_or_with_only_zero_rows(rows):
    assert feat(A=0., rows=rows).net_fraction is None


def test_feature_motif_param_sets_and_term_ids_come_from_rows():
    f = feat(A=0., rows=(row(term_id=0, motif=OTHER_MOTIF, a=0),
                         row(term_id=1, motif=OTHER_MOTIF, a=0),
                         row(term_id=0, motif=OTHER_MOTIF, a=1)))

    assert f.motif == OTHER_MOTIF
    assert f.param_sets == (ParameterSet({'a': 0}), ParameterSet({'a': 1}))
    assert f.term_ids == (0, 1)


def test_feature_without_rows_has_no_motif_param_sets_or_term_ids():
    f = feat(A=0., amp=None)

    assert (f.motif, f.param_sets, f.term_ids) == (None, (), ())


def test_feature_motif_rejects_rows_of_two_motifs():
    f = feat(A=0., rows=(row(a=0), row(motif=OTHER_MOTIF, a=0)))

    with pytest.raises(ValueError, match='one motif per feature'):
        _ = f.motif


def test_feature_equality_uses_location_gamma_rows_and_scale():
    scaled = feat(A=1.)
    scaled.scale = 2.

    assert feat(A=1.) == feat(A=1.)
    assert feat(A=1.) != feat(A=2.)
    assert feat(A=1., gamma=1.) != feat(A=1., gamma=2.)
    assert feat(A=1., amp=1.) != feat(A=1., amp=5.)         # rows differ in coeff
    assert feat(A=1., rows=(row(a=0),)) != feat(A=1., rows=(row(motif=OTHER_MOTIF, a=0),))
    assert scaled != feat(A=1.)


def test_feature_hash_matches_equality():
    assert hash(feat(A=1.)) == hash(feat(A=1.))
    assert len({feat(A=1.), feat(A=1.), feat(A=2.), feat(A=1., amp=5.)}) == 3


def test_feature_is_not_equal_or_ordered_with_other_types():
    assert feat(A=0.) != 'feature'
    assert feat(A=0.).__lt__(1.) is NotImplemented # type: ignore


def test_feature_orders_by_absolute_amplitude():
    assert feat(A=0., amp=1.) < feat(A=0., amp=-2.)
    assert not feat(A=0., amp=-3.) < feat(A=0., amp=2.)


def test_feature_order_needs_amplitudes():
    with pytest.raises(ValueError, match='amplitude_coeff'):
        sorted([feat(A=0., amp=None), feat(A=0., amp=1.)])


def test_feature_order_accepts_zero_amplitude():
    assert feat(A=0., amp=0.) < feat(A=1., amp=1.)


def test_feature_anharm_contributions_sums_per_anharmonicity():
    terms = [SimpleNamespace(anharmonicity='mech'), SimpleNamespace(anharmonicity='mech'),
             SimpleNamespace(anharmonicity='el')]
    f = feat(A=0., rows=(row(1., term_id=0, a=0), row(2., term_id=1, a=0), row(5., term_id=2, a=0)))

    assert f.anharm_contributions(terms) == {'mech': 3., 'el': 5.}


def test_feature_anharm_contributions_add_up_to_the_scaled_amplitude():
    terms = [SimpleNamespace(anharmonicity='mech'), SimpleNamespace(anharmonicity='el')]
    f = feat(A=0., rows=(row(1., term_id=0, a=0), row(3., term_id=1, a=0)))
    f.scale = 0.5

    parts = f.anharm_contributions(terms)

    assert parts == {'mech': 0.5, 'el': 1.5}
    assert sum(parts.values()) == f.amplitude_coeff


def test_feature_anharm_contributions_is_empty_without_rows():
    assert feat(A=0., amp=None).anharm_contributions([]) == {}


## features_from_rows -------------------------------------------------------

LOC_1650 = ResLocPoint({'A': 1650.})


def test_features_from_rows_adds_rows_of_one_motif_at_one_location():
    rows = (row(0.30, term_id=0, location=LOC_1650, a=0), row(-0.12, term_id=1, location=LOC_1650, a=0))

    [f] = features_from_rows(ContributionTable(rows))

    assert f.location == LOC_1650
    assert f.rows == rows
    assert f.amplitude_coeff == pytest.approx(0.18)


def test_features_from_rows_keeps_motifs_apart_at_one_location():
    """Same point, different motif -> different peak shape: two features."""
    rows = (row(0.30, location=LOC_1650, a=0), row(0.20, motif=OTHER_MOTIF, location=LOC_1650, a=0))

    features = features_from_rows(ContributionTable(rows))

    assert [f.motif for f in features] == [ResonanceMotif(()), OTHER_MOTIF]
    assert [f.amplitude_coeff for f in features] == [0.30, 0.20]


def test_features_from_rows_keeps_locations_apart():
    loc_1651 = ResLocPoint({'A': 1651.})
    rows = (row(location=LOC_1650, a=0), row(location=loc_1651, a=1))

    assert [f.location for f in features_from_rows(ContributionTable(rows))] == [LOC_1650, loc_1651]


def test_features_from_rows_groups_equal_locations_that_are_different_objects():
    """Two index sets can land on one point: separate ResLocPoint objects with equal values."""
    rows = (row(0.30, location=ResLocPoint({'A': 1650.}), a=0), row(0.20, location=ResLocPoint({'A': 1650.}), a=1))
    assert rows[0].location is not rows[1].location

    [f] = features_from_rows(ContributionTable(rows))

    assert f.rows == rows
    assert f.amplitude_coeff == pytest.approx(0.5)


def test_features_from_rows_does_not_round_locations():
    """Exact key on purpose: float noise gives two features at almost one point; the grid adds them anyway."""
    rows = (row(location=ResLocPoint({'A': 1650.}), a=0), row(location=ResLocPoint({'A': 1650. + 1e-9}), a=1))

    assert len(features_from_rows(ContributionTable(rows))) == 2


def test_features_from_rows_gives_every_feature_the_lineshape_parameter():
    [f] = features_from_rows(ContributionTable([row(location=LOC_1650, a=0)]), lineshape_parameter=5.)

    assert f.lineshape_parameter == 5.
    assert f.feat_box is None   # boxes come later, from dress_these_with_boxes


def test_features_from_rows_of_empty_table_is_empty():
    assert features_from_rows(ContributionTable([])) == []


## SpectralFeature: list helpers --------------------------------------------

def test_sort_by_params_orders_by_sorted_index_sets():
    f_high = feat(A=0., rows=(row(a=1),))
    f_low = feat(A=1., rows=(row(a=2), row(a=0)))

    assert SpectralFeature.sort_by_params([f_high, f_low]) == [f_low, f_high]


def test_normalize_coeffs_divides_by_largest_absolute_amplitude():
    feats = [feat(A=0., amp=2.), feat(A=1., amp=-4.)]

    result = SpectralFeature.normalize_coeffs_to_max(feats)

    assert [f.amplitude_coeff for f in result] == [0.5, -1.]
    assert [f.amplitude_coeff for f in feats] == [2., -4.]
    assert [f.rows for f in result] == [f.rows for f in feats]      # only scale changes


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
    with pytest.raises(ValueError, match='nonzero amplitude_coeff'):
        SpectralFeature.normalize_coeffs_to_max([feat(A=0., amp=0.), feat(A=1., amp=0.)])


def test_get_feats_with_params_matches_any_row():
    f0 = feat(A=0., rows=(row(a=0, b=1), row(a=1, b=0)))
    f1 = feat(A=1., rows=(row(a=1, b=1),))

    assert SpectralFeature.get_feats_with_params([f0, f1], {'a': 1, 'b': 0}) == [f0]


def test_get_feats_with_params_skips_features_without_rows():
    assert SpectralFeature.get_feats_with_params([feat(A=0., amp=None)], {'a': 0}) == []


## SpectralFeature: relation to a box ---------------------------------------

@pytest.mark.parametrize('method', ['is_inside', 'feat_box_overlaps'])
def test_feature_box_relations_need_the_box_axes(method):
    box = Box({'A': (0., 10.)})

    with pytest.raises(ValueError, match='axes'):
        getattr(feat(B=1.), method)(box)  # same number of axes, other name
    with pytest.raises(ValueError, match='axes'):
        getattr(feat(A=1., B=1.), method)(box)  # one axis too many


@pytest.mark.parametrize('a, inside', [(5., True), (0., True), (10., True), (10.1, False), (-1., False)])
def test_feature_is_inside_includes_edges(a, inside):
    assert feat(A=a).is_inside(Box({'A': (0., 10.)})) == inside


def test_feature_box_overlaps_excludes_touching_edges():
    box = Box({'A': (0., 10.)})

    assert feat(A=10.5, box_extent=1.).feat_box_overlaps(box)
    assert not feat(A=11., box_extent=1.).feat_box_overlaps(box)  # boxes only touch


def test_feature_box_overlaps_needs_a_feature_box():
    with pytest.raises(ValueError, match='Need to add a box'):
        feat(A=5.).feat_box_overlaps(Box({'A': (0., 10.)}))


## SpectralFeature: intensity -----------------------------------------------

@pytest.mark.parametrize('location, gamma, expected', [
    ({'A': 100.}, 5., 4. / (5. * CM_TO_AU) ** 2),             # gamma is in cm-1 and gets converted
    ({'A': 100., 'B': 200.}, 5., 4. / (5. * CM_TO_AU) ** 4),  # one 1/gamma factor per axis
    ({'A': 100.}, 1e-6, 4. / (1e-6 * CM_TO_AU) ** 2),         # a small gamma is still cm-1
])
def test_intensity_is_abs_squared_of_amplitude_over_gamma_power(location, gamma, expected):
    assert feat(amp=2., gamma=gamma, **location).get_intensity() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize('amp, gamma', [(None, 1.), (1., None)])
def test_intensity_needs_amplitude_and_gamma(amp, gamma):
    with pytest.raises(ValueError):
        feat(A=0., amp=amp, gamma=gamma).get_intensity()


def test_intensity_needs_positive_gamma():
    with pytest.raises(ValueError, match='must be > 0'):
        feat(A=0., gamma=0.).get_intensity()


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
    with pytest.raises(ValueError, match='empty list'):
        SpectralFeature.get_max_intensity_feat([])


def test_max_intensity_feat_rejects_features_that_all_have_zero_intensity():
    with pytest.raises(ValueError, match='None of the 2 features has a nonzero intensity'):
        SpectralFeature.get_max_intensity_feat([feat(A=0., amp=0.), feat(A=1., amp=0.)])


## SpectralFeature: dress_these_with_boxes ----------------------------------
# A Lorentzian drops to 1/dynrange of its peak at distance gamma*sqrt(dynrange - 1).
# dynrange = 101 -> 10*gamma ; dynrange = 26 -> 5*gamma

def test_dress_box_extent_is_gamma_times_sqrt_dynrange_minus_one():
    f = feat(A=100., gamma=5.)
    top = f.get_intensity()

    [dressed] = SpectralFeature.dress_these_with_boxes([f], top, top / 101, box_range_safety_margin=0.)

    assert dressed.feat_box.bounds['A'] == pytest.approx((50., 150.)) # type: ignore


def test_dress_safety_margin_widens_the_box():
    f = feat(A=0., gamma=5.)
    top = f.get_intensity()

    [dressed] = SpectralFeature.dress_these_with_boxes([f], top, top / 101, box_range_safety_margin=0.1)

    assert get_box_extent(dressed) == pytest.approx(55.)


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

    assert [get_box_extent(f) for f in result] == pytest.approx([50., 20.])


def test_dress_scaled_boxes_shrink_for_weaker_features():
    # the feature is at half of max_intensity: dynrange 52 is scaled to 26
    f = feat(A=0., gamma=5.)
    top = 2 * f.get_intensity()

    [scaled] = SpectralFeature.dress_these_with_boxes([f], top, top / 52, box_range_safety_margin=0.,
                                                      scale_wrt_max_intensity=True)
    [unscaled] = SpectralFeature.dress_these_with_boxes([f], top, top / 52, box_range_safety_margin=0.)

    assert get_box_extent(scaled) == pytest.approx(25.)
    assert get_box_extent(unscaled) == pytest.approx(5. * 51 ** 0.5)


def test_dress_needs_lineshape_parameter():
    with pytest.raises(ValueError, match='lineshape_parameter'):
        SpectralFeature.dress_these_with_boxes([feat(A=0., gamma=None)], 1., 0.01)


def test_dress_rejects_feature_above_max_intensity():
    f = feat(A=0.)
    top = f.get_intensity()

    with pytest.raises(ValueError, match='higher intensity'):
        SpectralFeature.dress_these_with_boxes([f], top / 2, top / 200)


def test_dress_does_not_change_input_features():
    f = feat(A=0., gamma=5.)
    top = f.get_intensity()

    SpectralFeature.dress_these_with_boxes([f], top, top / 101)

    assert f.feat_box is None


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


def test_sample_grid_is_the_box_grid():
    box = Box({'A': (0., 10.), 'B': (100., 200.)})

    axes, grid = SpectralWindow(box).sample_grid({'A': 5, 'B': 3})
    box_axes, box_grid = box.make_grid({'A': 5, 'B': 3})

    for ax in ('A', 'B'):
        np.testing.assert_array_equal(axes[ax], box_axes[ax])
        np.testing.assert_array_equal(grid[ax], box_grid[ax])


def test_window_find_clusters_by_featboxes_includes_contributing_features():
    # boxes (8.5, 10.5) and (10, 12) overlap
    inside, near = feat(A=9.5, box_extent=1.), feat(A=11., box_extent=1.)
    window = SpectralWindow(Box({'A': (0., 10.)}), full_features=[inside], contrib_features=[near])

    [domain] = window.find_clusters_by_featboxes()

    assert domain.box == Box({'A': (8.5, 12.)})


## SpectralWindow.from_features ---------------------------------------------
# Window A (0, 10). gamma 5 cm-1, dynrange 101, margin 0.1: box_extent = 5 * sqrt(101 - 1) * 1.1 = 55.
# Intensity goes with amp**2, e.g. amp 0.09 -> 1/123 of amp 1.

WINDOW_0_10 = Box({'A': (0., 10.)})


def test_window_from_features_keeps_features_whose_box_overlaps_the_window():
    """
    A = 60: box (5, 115) overlaps the window -> contrib_features.  A = 70: box (15, 125) does not -> dropped.
    (The old rule, location within 2 * gamma of the window, dropped A = 60 too.)
    """
    inside, near, far = feat(A=5., gamma=5.), feat(A=60., gamma=5.), feat(A=70., gamma=5.)

    window = SpectralWindow.from_features(WINDOW_0_10, [inside, near, far], dynrange=101.)

    assert window.box == WINDOW_0_10
    assert window.full_features == [inside]
    assert window.contrib_features == [near]


def test_window_from_features_box_extent_is_gamma_times_sqrt_dynrange_minus_one_plus_margin():
    window = SpectralWindow.from_features(WINDOW_0_10, [feat(A=5., gamma=5.)], dynrange=101.)
    no_margin = SpectralWindow.from_features(WINDOW_0_10, [feat(A=5., gamma=5.)], dynrange=101.,
                                             box_range_safety_margin=0.)

    assert get_box_extent(window.full_features[0]) == pytest.approx(55.)
    assert get_box_extent(no_margin.full_features[0]) == pytest.approx(50.)


def test_window_from_features_boxes_are_the_dress_these_with_boxes_boxes():
    f = feat(A=5., gamma=5.)
    top = f.get_intensity()

    [dressed] = SpectralFeature.dress_these_with_boxes([f], top, top / 101)
    window = SpectralWindow.from_features(WINDOW_0_10, [f], dynrange=101.)

    assert window.full_features[0].feat_box == dressed.feat_box


def test_window_from_features_keeps_weak_features():
    """No feature is dropped for being weak, inside or outside the window: 1/123 and 1e-12 of the strong intensity stay."""
    strong = feat(A=5., amp=1., gamma=5.)
    weak, weakest, weak_outside = feat(A=6., amp=0.09, gamma=5.), feat(A=7., amp=1e-6, gamma=5.), feat(A=12., amp=1e-6, gamma=5.)

    window = SpectralWindow.from_features(WINDOW_0_10, [strong, weak, weakest, weak_outside], dynrange=101.)

    assert window.full_features == [strong, weak, weakest]
    assert window.contrib_features == [weak_outside]


def test_window_from_features_keeps_a_feature_whose_rows_cancel():
    """Rows that cancel (0.5 - 0.5): amplitude 0, but the feature stays with its rows, as a record."""
    loc = {'A': 6.}
    cancelling = feat(loc, gamma=5., rows=(row(0.5, location=ResLocPoint(loc), a=0),
                                           row(-0.5, location=ResLocPoint(loc), a=1)))
    other = feat(A=5., gamma=5.)

    window = SpectralWindow.from_features(WINDOW_0_10, [other, cancelling], dynrange=101.)

    assert window.full_features[1].amplitude_coeff == 0.
    assert window.full_features[1].rows == cancelling.rows
    assert len(window.full_features) == 2


@pytest.mark.parametrize('features', [[], [feat(A=500., gamma=5.)]])
def test_window_from_features_without_features_near_the_window_is_empty(features):
    window = SpectralWindow.from_features(WINDOW_0_10, features, dynrange=101.)

    assert window.full_features == [] and window.contrib_features == []


def test_window_from_features_does_not_change_input_features():
    f = feat(A=5., gamma=5.)

    SpectralWindow.from_features(WINDOW_0_10, [f], dynrange=101.)

    assert f.feat_box is None


def test_window_from_features_can_be_clustered_right_away():
    """inside: box (-50, 60), near: box (5, 115). They overlap, so one domain spans both."""
    window = SpectralWindow.from_features(WINDOW_0_10, [feat(A=5., gamma=5.), feat(A=60., gamma=5.)], dynrange=101.)

    [domain] = window.find_clusters_by_featboxes()

    assert domain.box.bounds['A'] == pytest.approx((-50., 115.))


@pytest.mark.parametrize('dynrange', [1., 0.5])
def test_window_from_features_needs_dynrange_above_one(dynrange):
    with pytest.raises(ValueError, match='dynrange'):
        SpectralWindow.from_features(WINDOW_0_10, [feat(A=5.)], dynrange=dynrange)


def test_window_from_features_needs_lineshape_parameter():
    with pytest.raises(ValueError, match='lineshape_parameter'):
        SpectralWindow.from_features(WINDOW_0_10, [feat(A=5., gamma=None)], dynrange=101.)


def test_window_from_features_needs_rows():
    with pytest.raises(ValueError, match='no rows'):
        SpectralWindow.from_features(WINDOW_0_10, [feat(A=5., amp=None)], dynrange=101.)


## RectangularDomain --------------------------------------------------------

def test_domain_from_features_spans_feature_boxes():
    f0, f1 = feat(A=0., box_extent=1.), feat(A=5., box_extent=1.)

    domain = RectangularDomain.from_features([f0, f1])

    assert domain.box == Box({'A': (-1., 6.)})
    assert domain.full_features == [f0, f1]


def test_domain_from_features_needs_feature_boxes():
    with pytest.raises(ValueError, match='feat_box'):
        RectangularDomain.from_features([feat(A=0.)])


def test_equal_domains_have_equal_hashes():
    # the old GridManager uses domains as dict keys
    d1 = RectangularDomain.from_features([feat(A=0., box_extent=1.)])
    d2 = RectangularDomain.from_features([feat(A=0., box_extent=1.)])

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
def test_nd_feature_is_inside_only_when_every_axis_is_inside(ndim):
    box, last = cube(ndim, 0., 10.), AXES[ndim - 1]

    assert feat(coords(ndim, 5.)).is_inside(box)
    assert feat(coords(ndim, 10.)).is_inside(box)  # corner
    for ax in AXES[:ndim]:
        assert not feat(coords(ndim, 5., **{ax: 11.})).is_inside(box)
    assert feat(coords(ndim, 5., **{last: 10.5}), box_extent=1.).feat_box_overlaps(box)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_intensity_has_one_gamma_factor_per_axis(ndim):
    f = feat(coords(ndim, 100.), amp=2., gamma=5.)

    assert f.get_intensity() == pytest.approx(4. / (5. * CM_TO_AU) ** (2 * ndim), rel=1e-6)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_dress_box_extent_is_the_same_on_every_axis(ndim):
    f = feat(coords(ndim, 100.), gamma=5.)
    top = f.get_intensity()

    [dressed] = SpectralFeature.dress_these_with_boxes([f], top, top / 101, box_range_safety_margin=0.)

    assert [get_box_extent(dressed, ax) for ax in AXES[:ndim]] == pytest.approx([50.] * ndim)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_window_from_features_needs_box_overlap_on_every_axis(ndim):
    """
    Window (0, 10) on every axis; gamma 5, dynrange 101 -> box_extent 55 on every axis.
    near and far sit on `inside`, except on the last axis: near at 60 (box from 5: overlaps), far at 70 (from 15: no).
    """
    last = AXES[ndim - 1]
    inside = feat(coords(ndim, 5.), gamma=5.)
    near = feat(coords(ndim, 5., **{last: 60.}), gamma=5.)
    far = feat(coords(ndim, 5., **{last: 70.}), gamma=5.)

    window = SpectralWindow.from_features(cube(ndim, 0., 10.), [inside, near, far], dynrange=101.)

    assert window.full_features == [inside]
    assert window.contrib_features == [near]
    assert [get_box_extent(window.full_features[0], ax) for ax in AXES[:ndim]] == pytest.approx([55.] * ndim)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_window_find_clusters_by_featboxes(ndim):
    # boxes: f0 (-1, 1) and f1 (0.5, 2.5) overlap on every axis; f2 sits on f0, except (9, 11) on the last axis
    f0, f1 = feat(coords(ndim, 0.), box_extent=1.), feat(coords(ndim, 1.5), box_extent=1.)
    f2 = feat(coords(ndim, 0., **{AXES[ndim - 1]: 10.}), box_extent=1.)
    window = SpectralWindow(cube(ndim, -5., 15.), full_features=[f0, f1, f2])

    domains = window.find_clusters_by_featboxes()

    assert [d.box for d in domains] == [cube(ndim, -1., 2.5), f2.feat_box]
    assert [d.full_features for d in domains] == [[f0, f1], [f2]]

