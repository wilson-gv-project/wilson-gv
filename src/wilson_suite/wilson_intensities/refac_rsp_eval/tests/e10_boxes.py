"""
E10: the box sizes (a box rule and its knobs: features.amplitude_half_widths or intensity_half_widths, the
box_rule of draw_spectrum) against draw_all (no cut). Two lists of settings:
    SETTINGS  tune the knobs of amplitude_half_widths
                  python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.e10_boxes
    RULES     compare the two rules, at dynrange 100, 500 and 1000: amplitude (eps 0.1); intensity (the old code's
              rule) unscaled, scaled, and scaled with minimum_box_padding 10 cm-1 (the old setup's padding)
                  python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.e10_boxes rules
Each takes a few minutes, mostly draw_all on ACDM.

Per case, draw_all once (the reference: slow, about 70 s on ACDM in the quick_start window), then draw_spectrum
with each setting. Scores per setting:
    err      the largest intensity error, in floors (floor = max / dynrange: the faintest the plot shows)
    worst    the worst relative error at a grid point above the floor (a point the plot shows)
    >10%     the share of those points more than 10 % off
    draw s   draw_spectrum's time (boxes + drawing)
    drawn    features drawn (inside the window, or outside with a box that reaches in)
    whole    of those, the ones whose box covers the whole window: the costly ones (one full-grid draw each)

Cases: formaldehyde and ACDM; the quick_start window (A (700, 3400), B (10, 3500), 700 x 700 points) and a zoom
(+-150 cm-1 around the strongest feature inside it, 301 x 301 points). The 14 EVV terms, anharmonic states,
gamma 4.7 cm-1.

A setting: rule (the box rule; amplitude_half_widths if not given) and its knobs, dynrange required. To tune: edit
SETTINGS or RULES and run as above, or, from a notebook, with your own result and window:
    from wilson_suite.wilson_intensities.refac_rsp_eval.tests.e10_boxes import reference, score
    ref = reference(result, window, dim_sizes)                                  # once: draw_all
    score(result, window, dim_sizes, ref, dynrange=500., eps=0.3, max_cm=300.)  # as often as wanted
    score(result, window, dim_sizes, ref, intensity_half_widths, dynrange=500.) # the old code's rule

Result 2026-10-14 (todo.txt E10, also the boxes before E10 and drawing by domains): with eps 0.1, at most
0.13 floors, the worst visible point 2.2 % off, no visible point more than 10 % off; eps 0.3: 0.47 floors, 4.9 %.
RULES, 2026-10-14: amplitude 0.03-0.13 floors (2.2 %); intensity unscaled 1.7-12 floors, scaled 0.7-39 floors
(visible points up to 229 % off), with padding 10 about the same.
"""
import sys
import time
from functools import partial

import numpy as np

from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import draw_all
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    BoxRule,
    amplitude_half_widths,
    intensity_half_widths,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import Box
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import (
    FeatureResult,
    draw_spectrum,
    evaluate_response_terms,
    load_molsys_data,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests import (
    make_evv_reference as ref,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.f2_speed import load_acdm
from wilson_suite.wilson_utils.builders import make_SpectralAxisSet

GAMMA_CM = 4.7
STATES = 'anharmonic'
QUICK_START = (Box({'A': (700., 3400.), 'B': (10., 3500.)}), {'A': 700, 'B': 700})
ZOOM_HALF_CM, ZOOM_N = 150., 301
SETTINGS = [
    dict(dynrange=dynrange, eps=eps) for dynrange in (100., 500., 1000.) for eps in (0.1, 0.3)
] + [
    dict(dynrange=500., eps=0.1, min_gammas=3.),
    dict(dynrange=500., eps=0.1, max_cm=500.),
]
RULES = [
    setting for dynrange in (100., 500., 1000.) for setting in (
        dict(dynrange=dynrange),
        dict(rule=intensity_half_widths, dynrange=dynrange),
        dict(rule=intensity_half_widths, dynrange=dynrange, scale_wrt_max_intensity=True),
        dict(rule=intensity_half_widths, dynrange=dynrange, scale_wrt_max_intensity=True, minimum_box_padding=10.),
    )
]
SHORT = {'scale_wrt_max_intensity': 'scaled', 'minimum_box_padding': 'padding', 'box_range_safety_margin': 'margin'}


def reference(result: FeatureResult, window: Box, dim_sizes: dict[str, int]) -> np.ndarray:
    """the intensity with nothing cut off: every feature on the whole grid (draw_all), on draw_spectrum's grid"""
    coords, _ = window.make_grid(dim_sizes)
    return np.abs(draw_all(result.features, result.states, coords)) ** 2


def score(result: FeatureResult, window: Box, dim_sizes: dict[str, int], reference_intensity: np.ndarray,
          rule: BoxRule = amplitude_half_widths, **knobs) -> dict:
    """draw_spectrum with `rule` and its `knobs` (dynrange required) against the reference: the scores above"""
    t0 = time.perf_counter()
    spec = draw_spectrum(result, window, dim_sizes, box_rule=partial(rule, **knobs))
    t_draw = time.perf_counter() - t0

    intensity = np.abs(spec.grid_result) ** 2
    floor = reference_intensity.max() / knobs['dynrange']
    visible = reference_intensity > floor
    err = np.abs(intensity - reference_intensity)
    rel = err[visible] / reference_intensity[visible]
    drawn = spec.spectral_window.full_features + spec.spectral_window.contrib_features
    return {'err': err.max() / floor, 'worst': rel.max(), '>10%': np.mean(rel > 0.1), 'draw s': t_draw,
            'drawn': len(drawn), 'whole': sum(f.feat_box.contains_box(window) for f in drawn)}  # type: ignore


def label(rule: BoxRule, knobs: dict) -> str:
    """e.g. 'intensity, dynrange 500, scaled, padding 10'"""
    names = [getattr(rule, '__name__', 'rule').removesuffix('_half_widths')]
    return ', '.join(names + [SHORT.get(k, k) if v is True else f'{SHORT.get(k, k)} {v:g}' for k, v in knobs.items()])


def run_case(name: str, result: FeatureResult, window: Box, dim_sizes: dict[str, int], settings: list[dict]):
    t0 = time.perf_counter()
    reference_intensity = reference(result, window, dim_sizes)
    inside = sum(f.is_inside(window) for f in result.features)
    print(f'\n{name}: {len(result.features)} features, {inside} inside; draw_all {time.perf_counter() - t0:.1f} s')
    print(f'  {"setting":<48} {"err":>6} {"worst":>7} {">10%":>6} {"draw s":>7} {"drawn":>6} {"whole":>6}')
    for setting in settings:
        knobs = dict(setting)
        rule = knobs.pop('rule', amplitude_half_widths)
        s = score(result, window, dim_sizes, reference_intensity, rule, **knobs)
        print(f'  {label(rule, knobs):<48} {s["err"]:>6.2f} {s["worst"]:>7.1%} {s[">10%"]:>6.1%} {s["draw s"]:>7.2f} '
              f'{s["drawn"]:>6} {s["whole"]:>6}')


def zoom_around_strongest(result: FeatureResult, window: Box) -> tuple[Box, dict[str, int]]:
    strongest = max((f for f in result.features if f.is_inside(window)), key=lambda f: f.get_intensity())
    loc = strongest.location.as_dict()
    return Box({ax: (v - ZOOM_HALF_CM, v + ZOOM_HALF_CM) for ax, v in loc.items()}), {ax: ZOOM_N for ax in loc}


def main(settings: list[dict]):
    terms = ref.evv_terms()
    for mol, load in (('formaldehyde', lambda: load_molsys_data(terms, ref.data_origin('formaldehyde'))),
                      ('ACDM', lambda: load_acdm(terms))):
        result = evaluate_response_terms(terms, axes=make_SpectralAxisSet(ref.AXES), molsys_data=load(),  # type: ignore
                                         states_choice=STATES, laser_pol=ref.LASER_POL, propagator_param=GAMMA_CM)
        run_case(f'{mol}, quick_start window', result, *QUICK_START, settings)
        zoom, zoom_grid = zoom_around_strongest(result, QUICK_START[0])
        run_case(f'{mol}, zoom {zoom.bounds}', result, zoom, zoom_grid, settings)


if __name__ == '__main__':
    main(RULES if sys.argv[1:] == ['rules'] else SETTINGS)
