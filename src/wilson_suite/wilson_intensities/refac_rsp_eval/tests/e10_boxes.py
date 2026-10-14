"""
E10: box size by amplitude (todo.txt E10). Against draw_all (no cut):
    today (domains)   today's boxes (SpectralWindow.from_features), drawn by domains (draw_window)
    today, own box    the same boxes, each feature drawn on its own box only: what the domains change
    amplitude         amplitude boxes, each feature drawn on its own box

Cases: formaldehyde and ACDM; the quick_start window (A (700, 3400), B (10, 3500), 700 x 700 points) and a zoom
(+-150 cm-1 around the strongest feature inside it, 301 x 301 points); the plot's dynamic range D = 100, 500, 1000;
eps = 0.3, 0.1. The 14 EVV terms, anharmonic states, gamma 4.7 cm-1.

The amplitude rule:
    k = 2 * D * a_f / (eps * A_max),  half-width d = gamma * sqrt(k^2 - 1)     k <= 1: 1 gamma
    a_f: the feature's peak amplitude; A_max: estimated without drawing, the largest amplitude any single feature
    has inside the window (amp_in_window).
Scores: err, the largest intensity error in floors (max / D); worst, the worst relative error at a grid point above
the floor; and the share of those points more than 10 % off.

Result 2026-10-14 (the full table: todo.txt E10): with eps 0.1, at most 0.13 floors, the worst visible point 2.2 %
off, no visible point more than 10 % off. Today's boxes: 0.8-7.7 floors, up to 100 % off at visible points.

A script, not a test: it runs for minutes (draw_all on ACDM, quick_start window: about 70 s).
    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.e10_boxes
"""
import time
from dataclasses import replace

import numpy as np

from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    build_contributions,
    draw_all,
    draw_window,
    eval_feature_on_grid,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    SpectralWindow,
    features_from_rows,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import Box, box_slices
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import (
    load_molsys_data,
    make_polarization_linear_comb,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests import (
    make_evv_reference as ref,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.f2_speed import load_acdm

GAMMA_CM = 4.7
STATES = 'anharmonic'
QUICK_START = ({'A': (700., 3400.), 'B': (10., 3500.)}, {'A': 700, 'B': 700})
ZOOM_HALF_CM, ZOOM_N = 150., 301
DYNRANGES = (100., 500., 1000.)
EPSILONS = (0.3, 0.1)


def amp_in_window(f, box: Box) -> float:
    """the feature's largest amplitude inside the window: its peak amplitude, less along each axis it lies outside"""
    a, g = np.sqrt(f.get_intensity()), max(f.peak_widths())
    for ax, x in f.location.as_dict().items():
        mn, mx = box.bounds[ax]
        a *= g / np.hypot(max(mn - x, 0., x - mx), g)
    return a


def amplitude_boxes(features, box: Box, dynrange: float, eps: float, a_max: float) -> list:
    """the features with boxes from the amplitude rule, those whose box overlaps the window"""
    out = []
    for f in features:
        g = max(f.peak_widths())
        k = 2 * dynrange * np.sqrt(f.get_intensity()) / (eps * a_max)
        d = g * np.sqrt(k * k - 1) if k > 1 else g
        out.append(replace(f, feat_box=Box({ax: (x - d, x + d) for ax, x in f.location.as_dict().items()})))
    return [f for f in out if f.feat_box.overlaps(box)]


def draw_per_feature(features, states, coords: dict[str, np.ndarray]) -> np.ndarray:
    """each feature on its own box only (no domains); the complex amplitude, laid out as in draw_all"""
    total = np.zeros(tuple(len(p) for p in coords.values()), dtype=complex)
    for f in features:
        s = box_slices(f.feat_box, coords)
        if s is None:
            continue
        sub = {ax: pts[sl] for (ax, pts), sl in zip(coords.items(), s)}
        total[s] += eval_feature_on_grid(f, states, dict(zip(sub, np.meshgrid(*sub.values(), indexing='ij'))))
    return total


def scores(intensity: np.ndarray, reference: np.ndarray, dynrange: float) -> tuple[float, float, float]:
    """(largest error in floors, worst relative error above the floor, share of those points more than 10 % off)"""
    floor = reference.max() / dynrange
    visible = reference > floor
    err = np.abs(intensity - reference)
    rel = err[visible] / reference[visible]
    return err.max() / floor, rel.max(), np.mean(rel > 0.1)


def run_case(name: str, features, states, window: dict, grid: dict[str, int]):
    box = Box(dict(window))
    coords, _ = box.make_grid(grid)
    t0 = time.perf_counter()
    reference = np.abs(draw_all(features, states, coords)) ** 2
    t_ref = time.perf_counter() - t0
    a_max = max(amp_in_window(f, box) for f in features)
    inside = sum(f.is_inside(box) for f in features)
    print(f'\n{name}: {len(features)} features, {inside} inside; draw_all {t_ref:.1f} s; '
          f'A_max estimate / true {a_max / np.sqrt(reference.max()):.3f}')
    print(f'  {"D":>5} {"boxes":<22} {"draw s":>7} {"err/floor":>10} {"worst visible":>14} {"visible >10% off":>17}')

    def line(dynrange, label, t, intensity):
        e, w, n = scores(intensity, reference, dynrange)
        print(f'  {dynrange:>5.0f} {label:<22} {t:>7.2f} {e:>10.2f} {w:>14.1%} {n:>17.1%}')

    for dynrange in DYNRANGES:
        t0 = time.perf_counter()
        window_today = SpectralWindow.from_features(box, features, dynrange)
        intensity = np.abs(draw_window(window_today, states, coords)) ** 2
        line(dynrange, 'today (domains)', time.perf_counter() - t0, intensity)
        t0 = time.perf_counter()
        amp = draw_per_feature(window_today.full_features + window_today.contrib_features, states, coords)
        line(dynrange, 'today, own box', time.perf_counter() - t0, np.abs(amp) ** 2)
        for eps in EPSILONS:
            t0 = time.perf_counter()
            amp = draw_per_feature(amplitude_boxes(features, box, dynrange, eps, a_max), states, coords)
            line(dynrange, f'amplitude, eps {eps}', time.perf_counter() - t0, np.abs(amp) ** 2)


def zoom_around_strongest(features, window: dict) -> tuple[dict, dict[str, int]]:
    box = Box(dict(window))
    strongest = max((f for f in features if f.is_inside(box)), key=lambda f: f.get_intensity())
    loc = strongest.location.as_dict()
    return {ax: (loc[ax] - ZOOM_HALF_CM, loc[ax] + ZOOM_HALF_CM) for ax in loc}, {ax: ZOOM_N for ax in loc}


def main():
    terms, compiled = ref.evv_terms(), ref.compiled_evv_terms()
    pol = make_polarization_linear_comb(terms, ref.LASER_POL)
    for mol, load in (('formaldehyde', lambda: load_molsys_data(terms, ref.data_origin('formaldehyde'))),
                      ('ACDM', lambda: load_acdm(terms))):
        molsys = load()
        table, _ = build_contributions(compiled, molsys, pol, states_choice=STATES)
        features = features_from_rows(table, GAMMA_CM)
        states = molsys.states[STATES]
        run_case(f'{mol}, quick_start window', features, states, *QUICK_START)
        zoom, zoom_grid = zoom_around_strongest(features, QUICK_START[0])
        run_case(f'{mol}, zoom {zoom}', features, states, zoom, zoom_grid)


if __name__ == '__main__':
    main()
