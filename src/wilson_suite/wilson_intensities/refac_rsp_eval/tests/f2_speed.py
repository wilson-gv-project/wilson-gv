"""
F2: runtime on a real bigger molecule (todo.txt F2). ACDM, the acetic acid dimer: 16 atoms, 42 modes.
The 14 EVV terms, axes A = w1, B = -w1 + w2, laser_pol (1, 1, 1), anharmonic states (the file is a GVPT2 run).

Steps:
  1. load: MolSystemData.load, only the terms' properties + frequencies + states. The reader fails on ACDM at
     normal_modes (todo.txt F2, [outside]); the evaluator does not use normal_modes.
  2. build_contributions: wall time without a profiler. Then once under cProfile: the share of the averaging
     (before F1 the loop, compute_for_idx_choice; since F1 calculate_avrg_tensor_einsum + AvrgTensor.at) and of
     the deepcopy (FreqTermsCollection.__post_init__).
  3. the rest: features_from_rows, SpectralWindow.from_features, draw_window on one grid.
     Window, grid, gamma and dynrange as in examples/quick_start.ipynb.

Result 2026-10-12 (also todo.txt F2): load 4 s; build_contributions 78 s (averaging 47 s, deepcopy 16 s, the rest
15 s); features, window and draw_window 2.5 s. Total without load: 80 s.
After F1, the same day (todo.txt F1): build_contributions 22 s (averaging 0.3 s, deepcopy 11 s). Total: 24 s.
After F3, the same day (todo.txt F3): build_contributions 3.9 s (no deepcopy per index set: the split is made
once per term). Total: 6.5 s. The deepcopy line shows 0 calls: the timed run before the profile made the splits.
After the leaves on demand, the same day (todo.txt Done): build_contributions 3.3 s (it builds no leaves).
Total: 5.9 s.

A script, not a test: it runs for minutes.
    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.f2_speed
"""
import cProfile
import pstats
import time

import numpy as np

from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    build_contributions,
    draw_window,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    SpectralWindow,
    features_from_rows,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import Box
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import (
    make_polarization_linear_comb,
    prop_names,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import FreqTermsCollection
from wilson_suite.wilson_intensities.refac_rsp_eval.tests import (
    make_evv_reference as ref,
)
from wilson_suite.wilson_system.system_data import (
    DataOriginInfo,
    MolSystemData,
)
from wilson_suite.wilson_utils.paths import SUITE_ROOT

DATA_FILE = 'wilson_intensities/tests/test_database/dftGaussian/ACDM/B3LYPcc_pVQZ/g16_b3lyp_corr_inputFull_3q.out'
STATES = 'anharmonic'
# drawing, as in examples/quick_start.ipynb
GAMMA_CM = 4.7
DYNRANGE = 500.
WINDOW = {'A': (700., 3400.), 'B': (10., 3500.)}
GRID = {'A': 700, 'B': 700}


def load_acdm(compiled) -> MolSystemData:
    """the terms' properties, frequencies, states; no normal_modes, atoms, geometry: the reader fails on ACDM there"""
    origin = DataOriginInfo(source_type='gaussian', base_file_loc=f'{SUITE_ROOT}/{DATA_FILE}')
    return MolSystemData.load(origin, prop_names(compiled))


def timed(f, *args, **kwargs):
    """(f's result, wall time in s)"""
    t0 = time.perf_counter()
    out = f(*args, **kwargs)
    return out, time.perf_counter() - t0


def profile_build_contributions(compiled, molsys, pol) -> dict:
    """build_contributions under cProfile: total, averaging, deepcopy (cumulative s, number of calls)"""
    prof = cProfile.Profile()
    prof.enable()
    build_contributions(compiled, molsys, pol, states_choice=STATES)
    prof.disable()
    stats = pstats.Stats(prof).stats       # type: ignore # (file, line, name) -> (primitive calls, calls, own s, cumulative s, ...)

    def cum(name: str, file_end: str, line: int | None = None) -> tuple[float, int]:
        hits = [(v[3], v[1]) for (file, ln, fn), v in stats.items()
                if fn == name and file.endswith(file_end) and (line is None or ln == line)]
        return sum(t for t, _ in hits), sum(n for _, n in hits)

    post_init = FreqTermsCollection.__post_init__.__code__
    out = {'total': cum('build_contributions', 'evaluate.py'),
           'averaging, loop (before F1)': cum('compute_for_idx_choice', 'evaluate.py'),
           'averaging, einsum (F1)': cum('calculate_avrg_tensor_einsum', 'evaluate.py'),
           'averaging, lookup (F1)': cum('at', 'evaluate.py'),
           'deepcopy': cum('__post_init__', post_init.co_filename, post_init.co_firstlineno)}
    print('\nbuild_contributions under cProfile, top 15 by own time:')
    pstats.Stats(prof).sort_stats('tottime').print_stats(15)
    return out


def main():
    compiled = ref.compiled_evv_terms()
    pol = make_polarization_linear_comb(compiled, ref.LASER_POL)

    # 1. load
    molsys, t_load = timed(load_acdm, compiled)
    n_modes = len(molsys.eigenvals)  # type: ignore
    print(f'ACDM: {n_modes} modes, {STATES} states, load {t_load:.1f} s')

    # 2. build_contributions
    (table, zero), t_build = timed(build_contributions, compiled, molsys, pol, states_choice=STATES)
    print(f'build_contributions: {t_build:.1f} s, {len(table)} rows, {len(zero)} zero')

    prof = profile_build_contributions(compiled, molsys, pol)
    total = prof['total'][0]
    for part, (t, n) in prof.items():
        if part != 'total':
            print(f'  {part:<28} {t:7.2f} s of {total:.1f} s under cProfile ({t / total:4.0%}), {n} calls, '
                  f'scaled to the plain run: {t / total * t_build:.2f} s')

    # 3. the rest
    features, t_feat = timed(features_from_rows, table, GAMMA_CM)
    window, t_window = timed(SpectralWindow.from_features, Box(dict(WINDOW)), features, DYNRANGE)
    (coords, _), t_grid = timed(window.sample_grid, GRID)
    domains, t_domains = timed(window.find_clusters_by_featboxes)
    spec, t_draw = timed(draw_window, window, molsys.states[STATES], coords)

    n_in = len(window.full_features) + len(window.contrib_features)
    print(f'features_from_rows:          {t_feat:7.2f} s, {len(features)} features')
    print(f'SpectralWindow.from_features: {t_window:6.2f} s, {n_in} features near the window '
          f'({len(window.full_features)} inside)')
    print(f'sample_grid {GRID}:   {t_grid:6.2f} s')
    print(f'find_clusters_by_featboxes:  {t_domains:7.2f} s, {len(domains)} domains '
          f'(draw_window finds them again)')
    print(f'draw_window:                 {t_draw:7.2f} s, max intensity {np.max(np.abs(spec)) ** 2:.3e}')

    t_rest = t_feat + t_window + t_grid + t_draw
    print(f'\ntotal without load: {t_build + t_rest:.1f} s (build_contributions {t_build:.1f} s, rest {t_rest:.1f} s)')


if __name__ == '__main__':
    main()
