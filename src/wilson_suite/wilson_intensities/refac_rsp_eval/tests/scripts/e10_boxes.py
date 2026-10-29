"""
The box sizes (a box rule and its knobs: features.amplitude_half_widths or intensity_half_widths, the
box_rule of spectrum_on_grid) against all_features_on_grid (no cut). Two lists of settings:
    SETTINGS  tune the knobs of amplitude_half_widths
                  python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.scripts.e10_boxes
    RULES     compare the two rules, at dynrange 100, 500 and 1000: amplitude (eps 0.1); intensity unscaled,
              scaled with minimum_box_padding 0 and 10 cm-1 (the default)
                  python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.scripts.e10_boxes rules
Each takes a few minutes, mostly all_features_on_grid on ACDM.

Per case, all_features_on_grid once (the reference: slow, about 70 s on ACDM in the quick_start window_box), then
spectrum_on_grid with each setting and cut_tails=True (each feature on its own feat_box only). Scores per setting:
    err      the largest intensity error, in floors (floor = max / dynrange: the faintest the plot shows)
    worst    the worst relative error at a grid point above the floor (a point the plot shows)
    >10%     the share of those points more than 10 % off
    grid s   spectrum_on_grid's time (feat_boxes + the values on the grid)
    kept     features evaluated on the grid (inside window_box, or outside with a feat_box that reaches in)
    whole    of those, the ones whose feat_box covers all of window_box: the costly ones (each evaluated on the
             full grid)

Cases: formaldehyde and ACDM; the quick_start window_box (A (700, 3400), B (10, 3500), 700 x 700 points) and a zoom
(+-150 cm-1 around the strongest feature inside it, 301 x 301 points). The 14 EVV terms, anharmonic states,
gamma 4.7 cm-1.

A setting: rule (the box rule; amplitude_half_widths if not given) and its knobs, dynrange required. To tune: edit
SETTINGS or RULES and run as above, or, from a notebook, with your own result and window_box:
    from wilson_suite.wilson_intensities.refac_rsp_eval.tests.scripts.e10_boxes import reference, score
    ref = reference(result, window_box, n_points_per_axis)                                  # once: all_features_on_grid
    score(result, window_box, n_points_per_axis, ref, dynrange=500., eps=0.3)               # as often as wanted
    score(result, window_box, n_points_per_axis, ref, intensity_half_widths, dynrange=500.) # the intensity rule
"""
import sys
import time
from functools import partial

import numpy as np

from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import all_features_on_grid
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    BoxRule,
    amplitude_half_widths,
    intensity_half_widths,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import Box
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import (
    FeatureResult,
    evaluate_response_terms,
    prop_names,
    spectrum_on_grid,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests import (
    make_evv_reference as ref,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.scripts.f2_speed import (
    load_acdm,
)
from wilson_suite.wilson_system.system_data import MolSystemData

GAMMA_CM = 4.7
STATES = 'anharmonic'
QUICK_START = (Box({'A': (700., 3400.), 'B': (10., 3500.)}), {'A': 700, 'B': 700})
ZOOM_HALF_CM, ZOOM_N = 150., 301
SETTINGS = [
    dict(dynrange=dynrange, eps=eps) for dynrange in (100., 500., 1000.) for eps in (0.1, 0.3)
] + [
    dict(dynrange=500., eps=0.1, min_gammas=3.),
]
RULES = [
    setting for dynrange in (100., 500., 1000.) for setting in (
        dict(dynrange=dynrange),
        dict(rule=intensity_half_widths, dynrange=dynrange),
        dict(rule=intensity_half_widths, dynrange=dynrange, scale_wrt_max_intensity=True, minimum_box_padding=0.),
        dict(rule=intensity_half_widths, dynrange=dynrange, scale_wrt_max_intensity=True, minimum_box_padding=10.),
    )
]
SHORT = {'scale_wrt_max_intensity': 'scaled', 'minimum_box_padding': 'padding', 'box_range_safety_margin': 'margin'}


def reference(result: FeatureResult, window_box: Box, n_points_per_axis: dict[str, int]) -> np.ndarray:
    """the intensity with nothing cut off: all_features_on_grid, on spectrum_on_grid's grid"""
    axis_points = window_box.grid_points(n_points_per_axis)
    return np.abs(all_features_on_grid(result.features, result.states, axis_points)) ** 2


def score(result: FeatureResult, window_box: Box, n_points_per_axis: dict[str, int], reference_intensity: np.ndarray,
          rule: BoxRule = amplitude_half_widths, **knobs) -> dict:
    """spectrum_on_grid with `rule` and its `knobs` (dynrange required) against the reference: the scores above"""
    t0 = time.perf_counter()
    spec = spectrum_on_grid(result, window_box=window_box, n_points_per_axis=n_points_per_axis,
                            box_rule=partial(rule, **knobs), cut_tails=True)
    t_on_grid = time.perf_counter() - t0

    intensity = np.abs(spec.amplitude) ** 2
    floor = reference_intensity.max() / knobs['dynrange']
    visible = reference_intensity > floor
    err = np.abs(intensity - reference_intensity)
    rel = err[visible] / reference_intensity[visible]
    kept = spec.spectral_window.features
    return {'err': err.max() / floor, 'worst': rel.max(), '>10%': np.mean(rel > 0.1), 'grid s': t_on_grid,
            'kept': len(kept), 'whole': sum(f.feat_box.contains_box(window_box) for f in kept)}  # type: ignore


def label(rule: BoxRule, knobs: dict) -> str:
    """e.g. 'intensity, dynrange 500, scaled, padding 10'"""
    names = [getattr(rule, '__name__', 'rule').removesuffix('_half_widths')]
    return ', '.join(names + [SHORT.get(k, k) if v is True else f'{SHORT.get(k, k)} {v:g}' for k, v in knobs.items()])


def run_case(name: str, result: FeatureResult, window_box: Box, n_points_per_axis: dict[str, int],
             settings: list[dict]):
    t0 = time.perf_counter()
    reference_intensity = reference(result, window_box, n_points_per_axis)
    inside = sum(f.is_inside(window_box) for f in result.features)
    print(f'\n{name}: {len(result.features)} features, {inside} inside; '
          f'all_features_on_grid {time.perf_counter() - t0:.1f} s')
    print(f'  {"setting":<48} {"err":>6} {"worst":>7} {">10%":>6} {"grid s":>7} {"kept":>6} {"whole":>6}')
    for setting in settings:
        knobs = dict(setting)
        rule = knobs.pop('rule', amplitude_half_widths)
        s = score(result, window_box, n_points_per_axis, reference_intensity, rule, **knobs)
        print(f'  {label(rule, knobs):<48} {s["err"]:>6.2f} {s["worst"]:>7.1%} {s[">10%"]:>6.1%} {s["grid s"]:>7.2f} '
              f'{s["kept"]:>6} {s["whole"]:>6}')


def zoom_around_strongest(result: FeatureResult, window_box: Box) -> tuple[Box, dict[str, int]]:
    strongest = max((f for f in result.features if f.is_inside(window_box)), key=lambda f: f.get_intensity())
    loc = strongest.location.as_dict()
    return Box({ax: (v - ZOOM_HALF_CM, v + ZOOM_HALF_CM) for ax, v in loc.items()}), {ax: ZOOM_N for ax in loc}


def main(settings: list[dict]):
    terms = ref.evv_terms()
    for mol, load in (('formaldehyde', lambda: MolSystemData.load(ref.data_origin('formaldehyde'), prop_names(terms))),
                      ('ACDM', lambda: load_acdm(terms))):
        result = evaluate_response_terms(terms, molsys_data=load(), states_choice=STATES, laser_pol=ref.LASER_POL, propagator_param=GAMMA_CM)
        run_case(f'{mol}, quick_start window_box', result, *QUICK_START, settings)
        zoom, zoom_grid = zoom_around_strongest(result, QUICK_START[0])
        run_case(f'{mol}, zoom {zoom.bounds}', result, zoom, zoom_grid, settings)


if __name__ == '__main__':
    main(RULES if sys.argv[1:] == ['rules'] else SETTINGS)
