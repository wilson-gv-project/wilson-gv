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
    amplitude_half_widths,
    amplitude_in_window,
    features_from_rows,
    intensity_half_widths,
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

def test_feature_has_no_box_until_its_window_gives_one():
    f = feat(A=5., gamma=5.)

    [boxed] = SpectralWindow.from_features(WINDOW_0_10, [f], [55.]).full_features

    assert f.feat_box is None
    assert boxed.feat_box is not None


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


def test_features_from_rows_gives_every_feature_the_propagator_param():
    [f] = features_from_rows(ContributionTable([row(location=LOC_1650, a=0)]), propagator_param=5.)

    assert f.propagator_param == 5.
    assert f.feat_box is None   # boxes come later, from SpectralWindow.from_features
    assert f.conditions is None # a number needs no conditions


def test_features_from_rows_without_states_gives_a_propagator_param_function_no_conditions():
    """The conditions come from the states (test_pipeline.py checks them); without states, peak_widths raises."""
    width = lambda vd, freqs: 5.

    [f] = features_from_rows(ContributionTable([row(location=LOC_1650, a=0)]), propagator_param=width)

    assert f.propagator_param is width
    assert f.conditions is None


def test_features_from_rows_of_empty_table_is_empty():
    assert features_from_rows(ContributionTable([])) == []


## SpectralFeature: list helpers --------------------------------------------

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


## SpectralFeature: propagator_param function, peak widths --------------------------
# A propagator_param function gets each condition's pair of states; here stand-ins that only have an energy difference.

TWO_CONDITIONS = (SimpleNamespace(E=1600.), SimpleNamespace(E=500.))
WIDTH_1_PERCENT = lambda vd, freqs: vd.E / 100    # 16 and 5 cm-1


def with_width_per_condition(f: SpectralFeature) -> SpectralFeature:
    return replace(f, propagator_param=WIDTH_1_PERCENT, conditions=TWO_CONDITIONS)


def test_peak_widths_of_a_number_is_one_per_axis():
    assert feat(A=100., B=200., gamma=5.).peak_widths() == (5., 5.)


def test_peak_widths_of_a_function_is_one_per_condition():
    assert with_width_per_condition(feat(A=1600., B=500.)).peak_widths() == pytest.approx((16., 5.))


def test_peak_widths_of_a_function_takes_the_location_as_freqs():
    """width = A / 100 at A = 300 -> 3 cm-1"""
    f = replace(feat(A=300.), propagator_param=lambda vd, freqs: freqs['A'] / 100, conditions=TWO_CONDITIONS[:1])

    assert f.peak_widths() == pytest.approx((3.,))


def test_peak_widths_of_a_function_needs_the_conditions():
    with pytest.raises(ValueError, match='no conditions'):
        replace(feat(A=100.), propagator_param=WIDTH_1_PERCENT).peak_widths()


def test_intensity_with_a_width_per_condition():
    """|2 / ((-16i) * (-5i))|^2, widths in au: 4 / (16 * 5 * CM_TO_AU**2)**2"""
    f = with_width_per_condition(feat(A=1600., B=500., amp=2.))

    assert f.get_intensity() == pytest.approx(4. / (16. * 5. * CM_TO_AU ** 2) ** 2, rel=1e-6)


def test_intensity_needs_positive_widths_from_a_function():
    f = replace(feat(A=0.), propagator_param=lambda vd, freqs: -5., conditions=TWO_CONDITIONS[:1])

    with pytest.raises(ValueError, match='must be > 0'):
        f.get_intensity()


def test_intensity_half_widths_take_the_largest_peak_width():
    """widths 16 and 5 cm-1, dynrange 101, no margin: 16 * sqrt(100) = 160 cm-1, scaled or not (the only feature)"""
    f = with_width_per_condition(feat(A=5., B=5.))
    window = Box({'A': (0., 10.), 'B': (0., 10.)})

    for scaled in (False, True):
        assert intensity_half_widths(window, [f], dynrange=101., box_range_safety_margin=0.,
                                     scale_wrt_max_intensity=scaled) == pytest.approx([160.])


def test_amplitude_half_widths_take_the_largest_peak_width():
    """
    widths 16 and 5 cm-1. Alone, the feature is the max: k = 2 * 101 / 0.1 = 2020 -> 16 * sqrt(2020**2 - 1).
    With an a_max far above it: k < 1 -> the smallest box, 1 gamma = 16 cm-1.
    """
    f = with_width_per_condition(feat(A=5., B=5.))
    window = Box({'A': (0., 10.), 'B': (0., 10.)})

    assert amplitude_half_widths(window, [f], dynrange=101.) == pytest.approx([16. * (2020. ** 2 - 1) ** 0.5])
    assert amplitude_half_widths(window, [f], dynrange=101., a_max=1e6 * f.get_intensity() ** 0.5) == [16.]


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
# Window A (0, 10). Half-width 55 cm-1: a feature at A = 60 has the box (5, 115), at A = 70 the box (15, 125).

WINDOW_0_10 = Box({'A': (0., 10.)})


def test_window_from_features_keeps_features_whose_box_overlaps_the_window():
    """
    A = 60: box (5, 115) overlaps the window -> contrib_features.  A = 70: box (15, 125) does not -> dropped.
    (The old rule, location within 2 * gamma of the window, dropped A = 60 too.)
    """
    inside, near, far = feat(A=5., gamma=5.), feat(A=60., gamma=5.), feat(A=70., gamma=5.)

    window = SpectralWindow.from_features(WINDOW_0_10, [inside, near, far], [55., 55., 55.])

    assert window.box == WINDOW_0_10
    assert window.full_features == [inside]
    assert window.contrib_features == [near]


def test_window_from_features_box_is_the_location_plus_minus_its_half_width():
    """one half-width per feature, in the order of the features; the same on every axis"""
    f0, f1 = feat(A=5., B=1., gamma=5.), feat(A=6., B=2., gamma=5.)

    window = SpectralWindow.from_features(Box({'A': (0., 10.), 'B': (0., 10.)}), [f0, f1], [55., 3.])

    assert [f.feat_box for f in window.full_features] == [Box({'A': (-50., 60.), 'B': (-54., 56.)}),
                                                          Box({'A': (3., 9.), 'B': (-1., 5.)})]


def test_window_from_features_keeps_a_feature_whose_rows_cancel():
    """Rows that cancel (0.5 - 0.5): amplitude 0, but the feature stays with its rows, as a record."""
    loc = {'A': 6.}
    cancelling = feat(loc, gamma=5., rows=(row(0.5, location=ResLocPoint(loc), a=0),
                                           row(-0.5, location=ResLocPoint(loc), a=1)))
    other = feat(A=5., gamma=5.)

    window = SpectralWindow.from_features(WINDOW_0_10, [other, cancelling], [55., 55.])

    assert window.full_features[1].amplitude_coeff == 0.
    assert window.full_features[1].rows == cancelling.rows
    assert len(window.full_features) == 2


@pytest.mark.parametrize('features', [[], [feat(A=500., gamma=5.)]])
def test_window_from_features_without_features_near_the_window_is_empty(features):
    window = SpectralWindow.from_features(WINDOW_0_10, features, [55.] * len(features))

    assert window.full_features == [] and window.contrib_features == []


def test_window_from_features_does_not_change_input_features():
    f = feat(A=5., gamma=5.)

    SpectralWindow.from_features(WINDOW_0_10, [f], [55.])

    assert f.feat_box is None


def test_window_from_features_can_be_clustered_right_away():
    """inside: box (-50, 60), near: box (5, 115). They overlap, so one domain spans both."""
    window = SpectralWindow.from_features(WINDOW_0_10, [feat(A=5., gamma=5.), feat(A=60., gamma=5.)], [55., 55.])

    [domain] = window.find_clusters_by_featboxes()

    assert domain.box.bounds['A'] == pytest.approx((-50., 115.))


def test_window_from_features_needs_one_half_width_per_feature():
    with pytest.raises(ValueError, match='one half-width per feature'):
        SpectralWindow.from_features(WINDOW_0_10, [feat(A=5.), feat(A=6.)], [55.])


@pytest.mark.parametrize('half_width', [-1., float('nan')])
def test_window_from_features_needs_half_widths_of_zero_or_more(half_width):
    with pytest.raises(ValueError, match='half-width must be >= 0'):
        SpectralWindow.from_features(WINDOW_0_10, [feat(A=5.)], [half_width])


def test_window_from_features_needs_propagator_param():
    with pytest.raises(ValueError, match='propagator_param'):
        SpectralWindow.from_features(WINDOW_0_10, [feat(A=5., gamma=None)], [55.])


def test_window_from_features_needs_rows():
    with pytest.raises(ValueError, match='no rows'):
        SpectralWindow.from_features(WINDOW_0_10, [feat(A=5., amp=None)], [55.])


## amplitude_half_widths: box size by amplitude (todo.txt E10) ----------------
# k = 2 * dynrange * a / (eps * a_max), half-width = gamma * sqrt(k**2 - 1), at least min_gammas * gamma.
# dynrange 100, eps 0.1 (default): k = 2000 * a / a_max. One gamma (5 cm-1) for all: a / a_max = amp / max amp.

def three_strengths() -> list[SpectralFeature]:
    """amp 1 (the max), 0.01 and 1e-4, inside WINDOW_0_10: k = 2000, 20, 0.2"""
    return [feat(A=5., amp=1., gamma=5.), feat(A=6., amp=0.01, gamma=5.), feat(A=7., amp=1e-4, gamma=5.)]


def test_amplitude_half_widths_follow_the_rule():
    """k = 2000, 20, 0.2 -> 5 * sqrt(2000**2 - 1), 5 * sqrt(20**2 - 1), and 1 gamma (k <= 1)"""
    half_widths = amplitude_half_widths(WINDOW_0_10, three_strengths(), dynrange=100.)

    assert half_widths == pytest.approx([5. * (2000. ** 2 - 1) ** 0.5, 5. * (20. ** 2 - 1) ** 0.5, 5.])


def test_amplitude_half_widths_smaller_eps_gives_bigger_boxes():
    """eps 0.01: k ten times larger, 20000, 200, 2"""
    half_widths = amplitude_half_widths(WINDOW_0_10, three_strengths(), dynrange=100., eps=0.01)

    assert half_widths == pytest.approx([5. * (20000. ** 2 - 1) ** 0.5, 5. * (200. ** 2 - 1) ** 0.5, 5. * 3 ** 0.5])


@pytest.mark.parametrize('min_gammas, weakest', [(3., 15.), (0., 0.)])
def test_amplitude_half_widths_min_gammas_sets_the_smallest_box(min_gammas, weakest):
    """only the weakest one (k = 0.2) is below the smallest box; the other two keep theirs"""
    half_widths = amplitude_half_widths(WINDOW_0_10, three_strengths(), dynrange=100., min_gammas=min_gammas)

    assert half_widths == pytest.approx([5. * (2000. ** 2 - 1) ** 0.5, 5. * (20. ** 2 - 1) ** 0.5, weakest])


def test_amplitude_half_widths_max_cm_caps_the_boxes():
    """cap 50 cm-1: the two strong boxes (9999.999 and 99.87) are cut to 50, the weakest (5) stays"""
    half_widths = amplitude_half_widths(WINDOW_0_10, three_strengths(), dynrange=100., max_cm=50.)

    assert half_widths == pytest.approx([50., 50., 5.])


def test_amplitude_half_widths_a_max_given_replaces_the_estimate():
    """a_max twice the strongest peak: every k halves, 1000, 10, 0.1"""
    features = three_strengths()
    peak = features[0].get_intensity() ** 0.5

    half_widths = amplitude_half_widths(WINDOW_0_10, features, dynrange=100., a_max=2. * peak)

    assert half_widths == pytest.approx([5. * (1000. ** 2 - 1) ** 0.5, 5. * (10. ** 2 - 1) ** 0.5, 5.])


def test_amplitude_in_window_is_the_peak_inside_and_the_tail_at_the_nearest_point_outside():
    """gamma 5: at A = 5, the peak; at A = 110, 100 cm-1 outside the window: peak * 5 / hypot(100, 5)"""
    inside, outside = feat(A=5., gamma=5.), feat(A=110., gamma=5.)
    peak = inside.get_intensity() ** 0.5

    assert amplitude_in_window(inside, WINDOW_0_10) == pytest.approx(peak)
    assert amplitude_in_window(outside, WINDOW_0_10) == pytest.approx(peak * 5. / np.hypot(100., 5.))


def test_amplitude_half_widths_a_max_comes_from_inside_the_window():
    """
    A feature 10 times stronger, 100 cm-1 outside, reaches the window with 10 * 0.05 = 0.5 of the inside peak.
    So a_max is the inside peak: k = 2000 inside, as if alone; outside k = 20000: its box reaches far into the window.
    """
    inside, strong_outside = feat(A=5., amp=1., gamma=5.), feat(A=110., amp=10., gamma=5.)

    half_widths = amplitude_half_widths(WINDOW_0_10, [inside, strong_outside], dynrange=100.)

    assert half_widths == pytest.approx([5. * (2000. ** 2 - 1) ** 0.5, 5. * (20000. ** 2 - 1) ** 0.5])


def test_amplitude_half_widths_compare_peak_heights_not_coefficients():
    """
    The same coefficient, gamma 1 and 5: the narrow peak is 5 times taller (one condition: amp / gamma), so it is
    the max. Narrow: k = 2000 -> 1 * sqrt(2000**2 - 1). Broad: k = 2000 / 5 = 400 -> 5 * sqrt(400**2 - 1).
    """
    narrow, broad = feat(A=5., amp=1., gamma=1.), feat(A=6., amp=1., gamma=5.)

    half_widths = amplitude_half_widths(WINDOW_0_10, [narrow, broad], dynrange=100.)

    assert half_widths == pytest.approx([(2000. ** 2 - 1) ** 0.5, 5. * (400. ** 2 - 1) ** 0.5])


def test_amplitude_half_widths_with_every_amplitude_zero_are_the_smallest_boxes():
    """rows that cancel everywhere: nothing on the grid, no a_max to scale by -> min_gammas * gamma"""
    assert amplitude_half_widths(WINDOW_0_10, [feat(A=5., amp=0., gamma=5.), feat(A=6., amp=0., gamma=2.)],
                                 dynrange=100.) == [5., 2.]


def test_amplitude_half_widths_of_no_features_is_empty():
    assert amplitude_half_widths(WINDOW_0_10, [], dynrange=100.) == []


def test_window_from_amplitude_boxes_keeps_weak_features():
    """
    No feature is dropped for being weak (decided 2026-10-09). Amplitudes 1, 1e-6 inside, 1e-6 at A = 12, outside:
    the weak ones get the smallest box, 1 gamma; at A = 12 that is (7, 17), so it still reaches into the window.
    """
    strong, weakest, weak_outside = feat(A=5., amp=1., gamma=5.), feat(A=7., amp=1e-6, gamma=5.), feat(A=12., amp=1e-6, gamma=5.)
    features = [strong, weakest, weak_outside]

    window = SpectralWindow.from_features(WINDOW_0_10, features,
                                          amplitude_half_widths(WINDOW_0_10, features, dynrange=101.))

    assert window.full_features == [strong, weakest]
    assert window.contrib_features == [weak_outside]
    assert window.contrib_features[0].feat_box == Box({'A': (7., 17.)})


@pytest.mark.parametrize('setting, value', [('dynrange', 1.), ('dynrange', 0.5), ('eps', 0.), ('eps', -0.1),
                                            ('min_gammas', -1.), ('max_cm', 0.), ('a_max', 0.)])
def test_amplitude_half_widths_reject_settings_out_of_range(setting, value):
    settings = {'dynrange': 100., setting: value}

    with pytest.raises(ValueError, match=setting):
        amplitude_half_widths(WINDOW_0_10, [feat(A=5.)], **settings)


@pytest.mark.parametrize('f, message', [(feat(A=5., gamma=None), 'propagator_param'), (feat(A=5., amp=None), 'no rows')])
def test_amplitude_half_widths_need_a_feature_with_rows_and_width(f, message):
    with pytest.raises(ValueError, match=message):
        amplitude_half_widths(WINDOW_0_10, [f], dynrange=100.)


## intensity_half_widths: the old code's box rule (dress_these_with_boxes until 2026-10-14) ---------
# Unscaled: gamma * sqrt(dynrange - 1), the same for every strength.
# Scaled: gamma * sqrt(dynrange * i / max_intensity - 1), 0 at or below the floor.
# Then times (1 + box_range_safety_margin), at least minimum_box_padding (cm-1). gamma 5 cm-1: dynrange 101 -> 50 cm-1.

def test_intensity_half_widths_unscaled_are_the_same_for_every_strength():
    """three_strengths (intensities 1, 1e-4 and 1e-8 of the max), dynrange 101, no margin: 50 cm-1 each"""
    half_widths = intensity_half_widths(WINDOW_0_10, three_strengths(), dynrange=101., box_range_safety_margin=0.)

    assert half_widths == pytest.approx([50., 50., 50.])


def test_intensity_half_widths_safety_margin_widens_the_boxes():
    """default margin 0.1: 50 * 1.1 = 55 cm-1, the boxes of before 2026-10-14"""
    assert intensity_half_widths(WINDOW_0_10, [feat(A=5., gamma=5.)], dynrange=101.) == pytest.approx([55.])


def test_intensity_half_widths_minimum_box_padding_is_the_smallest_half_width():
    """padding 80 cm-1: above the 50 cm-1 of the formula, so 80, margin or not (the margin is not applied to it)"""
    assert intensity_half_widths(WINDOW_0_10, [feat(A=5., gamma=5.)], dynrange=101.,
                                 minimum_box_padding=80.) == pytest.approx([80.])


def test_intensity_half_widths_scaled_shrink_for_weaker_features():
    """intensities 1 and 1/2 of the max, dynrange 52: k = 52 and 26 -> 5 * sqrt(51) and 5 * sqrt(25) = 25 cm-1"""
    strong, half = feat(A=5., amp=2. ** 0.5, gamma=5.), feat(A=6., amp=1., gamma=5.)

    half_widths = intensity_half_widths(WINDOW_0_10, [strong, half], dynrange=52., box_range_safety_margin=0.,
                                        scale_wrt_max_intensity=True)

    assert half_widths == pytest.approx([5. * 51 ** 0.5, 25.])


@pytest.mark.parametrize('padding', [0., 20.])
def test_intensity_half_widths_scaled_never_drop_a_feature_below_the_floor(padding):
    """
    three_strengths, dynrange 101: k = 101, 0.0101 and 1e-6. The two below the floor (k <= 1) get half-width
    minimum_box_padding and stay in the window. dress_these_with_boxes dropped them when the padding was 0.
    """
    features = three_strengths()

    half_widths = intensity_half_widths(WINDOW_0_10, features, dynrange=101., box_range_safety_margin=0.,
                                        scale_wrt_max_intensity=True, minimum_box_padding=padding)

    assert half_widths == pytest.approx([50., padding, padding])
    assert SpectralWindow.from_features(WINDOW_0_10, features, half_widths).full_features == features


def test_intensity_half_widths_scaled_by_a_given_max_intensity():
    """max_intensity twice the strongest peak: k halves, 101 -> 50.5, so 5 * sqrt(49.5) for the strongest"""
    features = three_strengths()

    half_widths = intensity_half_widths(WINDOW_0_10, features, dynrange=101., box_range_safety_margin=0.,
                                        scale_wrt_max_intensity=True,
                                        max_intensity=2. * features[0].get_intensity())

    assert half_widths == pytest.approx([5. * 49.5 ** 0.5, 0., 0.])


def test_intensity_half_widths_scaled_take_a_stronger_feature_outside_the_window():
    """
    A feature 10 times stronger in amplitude, 100 cm-1 outside: dress_these_with_boxes raised (stronger than its
    max). max_intensity is the strongest peak inside: inside k = 101 -> 50 cm-1; outside k = 101 * 100 = 10100 ->
    5 * sqrt(10099) cm-1.
    """
    inside, strong_outside = feat(A=5., amp=1., gamma=5.), feat(A=110., amp=10., gamma=5.)

    half_widths = intensity_half_widths(WINDOW_0_10, [inside, strong_outside], dynrange=101.,
                                        box_range_safety_margin=0., scale_wrt_max_intensity=True)

    assert half_widths == pytest.approx([50., 5. * 10099 ** 0.5])


def test_intensity_half_widths_scaled_need_a_feature_inside_or_max_intensity():
    """no feature inside: no strongest peak to scale by, as in the old callers. Unscaled needs none."""
    outside = feat(A=60., gamma=5.)

    with pytest.raises(ValueError, match='pass max_intensity'):
        intensity_half_widths(WINDOW_0_10, [outside], dynrange=101., scale_wrt_max_intensity=True)
    assert intensity_half_widths(WINDOW_0_10, [outside], dynrange=101., box_range_safety_margin=0.,
                                 scale_wrt_max_intensity=True,
                                 max_intensity=outside.get_intensity()) == pytest.approx([50.])
    assert intensity_half_widths(WINDOW_0_10, [outside], dynrange=101.) == pytest.approx([55.])


def test_intensity_half_widths_scaled_with_every_amplitude_zero_are_the_smallest_boxes():
    """rows that cancel everywhere: nothing to scale by -> minimum_box_padding"""
    assert intensity_half_widths(WINDOW_0_10, [feat(A=5., amp=0., gamma=5.), feat(A=6., amp=0., gamma=2.)],
                                 dynrange=100., scale_wrt_max_intensity=True, minimum_box_padding=3.) == [3., 3.]


def test_intensity_half_widths_of_no_features_is_empty():
    assert intensity_half_widths(WINDOW_0_10, [], dynrange=100.) == []


@pytest.mark.parametrize('setting, value', [('dynrange', 1.), ('dynrange', 0.5), ('box_range_safety_margin', -0.1),
                                            ('minimum_box_padding', -1.), ('max_intensity', 0.)])
def test_intensity_half_widths_reject_settings_out_of_range(setting, value):
    settings = {'dynrange': 100., setting: value}

    with pytest.raises(ValueError, match=setting):
        intensity_half_widths(WINDOW_0_10, [feat(A=5.)], **settings)


@pytest.mark.parametrize('f, message', [(feat(A=5., gamma=None), 'propagator_param'), (feat(A=5., amp=None), 'no rows')])
def test_intensity_half_widths_need_a_feature_with_rows_and_width(f, message):
    with pytest.raises(ValueError, match=message):
        intensity_half_widths(WINDOW_0_10, [f], dynrange=100.)


## A flat background: rows without resonance conditions (todo.txt E12) ---------------------------
# The rows of a term without conditions sit at the location without axes. The feature has the same amplitude on
# every grid point: no peak, no width, no box of its own.

def flat(amp: float = 2.) -> SpectralFeature:
    return feat({}, amp=amp, gamma=5.)


def test_flat_feature_has_no_axes():
    assert flat().is_flat
    assert not feat(A=0.).is_flat


@pytest.mark.parametrize('propagator_param', [5., WIDTH_1_PERCENT])
def test_flat_feature_has_no_peak_widths(propagator_param):
    """No conditions, so no width; a function needs no conditions here."""
    assert replace(flat(), propagator_param=propagator_param).peak_widths() == ()


def test_flat_feature_intensity_is_its_amplitude_squared():
    """No 1/gamma factor: amp 2 -> 4 (au), on every grid point."""
    assert flat(2.).get_intensity() == 4.


@pytest.mark.parametrize('window', [WINDOW_0_10, Box({'A': (500., 600.), 'B': (0., 1.)})])
def test_amplitude_in_window_of_a_flat_feature_is_its_amplitude_anywhere(window):
    assert amplitude_in_window(flat(2.), window) == 2.


def test_amplitude_half_widths_give_a_flat_feature_0_and_count_it_for_a_max():
    """
    A background twice the strongest peak: a_max is the background, so every k halves, 1000, 10, 0.1 (as in
    test_amplitude_half_widths_a_max_given_replaces_the_estimate). The background itself gets 0.
    """
    features = three_strengths()
    background = flat(2. * features[0].get_intensity() ** 0.5)

    half_widths = amplitude_half_widths(WINDOW_0_10, [*features, background], dynrange=100.)

    assert half_widths == pytest.approx([5. * (1000. ** 2 - 1) ** 0.5, 5. * (10. ** 2 - 1) ** 0.5, 5., 0.])


def test_intensity_half_widths_give_a_flat_feature_0():
    """the peak: gamma 5, dynrange 101, no margin -> 5 * sqrt(100) = 50"""
    assert intensity_half_widths(WINDOW_0_10, [feat(A=5., gamma=5.), flat()], dynrange=101.,
                                 box_range_safety_margin=0.) == pytest.approx([50., 0.])


def test_intensity_half_widths_scaled_count_a_flat_feature_as_inside():
    """
    The peak and a background 4 times its intensity: max_intensity is the background's, so the peak's
    k = 100 / 4 = 25 -> 5 * sqrt(24). A background alone is enough as the feature inside.
    """
    peak = feat(A=5., gamma=5.)
    background = flat(2. * peak.get_intensity() ** 0.5)

    half_widths = intensity_half_widths(WINDOW_0_10, [peak, background], dynrange=100., box_range_safety_margin=0.,
                                        scale_wrt_max_intensity=True)

    assert half_widths == pytest.approx([5. * 24. ** 0.5, 0.])
    assert intensity_half_widths(WINDOW_0_10, [background], dynrange=100., scale_wrt_max_intensity=True) == [0.]


@pytest.mark.parametrize('box', [WINDOW_0_10, Box({'A': (0., 10.), 'B': (-5., 5.)})])
def test_window_from_features_gives_a_flat_feature_the_window_as_its_box(box):
    """In full_features, whatever its half-width (7 here); the peak next to it keeps its own box."""
    peak = feat({ax: 5. for ax in box.axes}, gamma=5.)

    window = SpectralWindow.from_features(box, [peak, flat()], [3., 7.])

    [boxed_peak, boxed_flat] = window.full_features
    assert window.contrib_features == []
    assert boxed_flat.is_flat and boxed_flat.feat_box == box
    assert boxed_peak.feat_box == Box({ax: (2., 8.) for ax in box.axes})


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
@pytest.mark.parametrize('scaled', [False, True])
def test_nd_intensity_half_width_does_not_change_with_the_number_of_axes(ndim, scaled):
    """one gamma factor per axis in the intensity, the same half-width: 5 * sqrt(100) = 50 cm-1 (dynrange 101)"""
    f = feat(coords(ndim, 5.), gamma=5.)

    assert intensity_half_widths(cube(ndim, 0., 10.), [f], dynrange=101., box_range_safety_margin=0.,
                                 scale_wrt_max_intensity=scaled) == pytest.approx([50.])


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_window_from_features_needs_box_overlap_on_every_axis(ndim):
    """
    Window (0, 10) on every axis; half-width 55 on every axis.
    near and far sit on `inside`, except on the last axis: near at 60 (box from 5: overlaps), far at 70 (from 15: no).
    """
    last = AXES[ndim - 1]
    inside = feat(coords(ndim, 5.), gamma=5.)
    near = feat(coords(ndim, 5., **{last: 60.}), gamma=5.)
    far = feat(coords(ndim, 5., **{last: 70.}), gamma=5.)

    window = SpectralWindow.from_features(cube(ndim, 0., 10.), [inside, near, far], [55., 55., 55.])

    assert window.full_features == [inside]
    assert window.contrib_features == [near]
    assert [get_box_extent(window.full_features[0], ax) for ax in AXES[:ndim]] == pytest.approx([55.] * ndim)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_amplitude_in_window_multiplies_the_tail_of_each_axis_outside(ndim):
    """100 cm-1 outside on every axis, gamma 5: peak * (5 / hypot(100, 5)) ** ndim"""
    inside, outside = feat(coords(ndim, 5.), gamma=5.), feat(coords(ndim, 110.), gamma=5.)

    assert amplitude_in_window(outside, cube(ndim, 0., 10.)) == pytest.approx(
        inside.get_intensity() ** 0.5 * (5. / np.hypot(100., 5.)) ** ndim)


@pytest.mark.parametrize('ndim', NDIMS)
def test_nd_window_find_clusters_by_featboxes(ndim):
    # boxes: f0 (-1, 1) and f1 (0.5, 2.5) overlap on every axis; f2 sits on f0, except (9, 11) on the last axis
    f0, f1 = feat(coords(ndim, 0.), box_extent=1.), feat(coords(ndim, 1.5), box_extent=1.)
    f2 = feat(coords(ndim, 0., **{AXES[ndim - 1]: 10.}), box_extent=1.)
    window = SpectralWindow(cube(ndim, -5., 15.), full_features=[f0, f1, f2])

    domains = window.find_clusters_by_featboxes()

    assert [d.box for d in domains] == [cube(ndim, -1., 2.5), f2.feat_box]
    assert [d.full_features for d in domains] == [[f0, f1], [f2]]

