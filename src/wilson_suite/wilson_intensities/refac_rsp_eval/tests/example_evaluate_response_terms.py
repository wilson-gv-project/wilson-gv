"""
evaluate_response_terms and spectrum_on_grid, step by step. An example of use, not a test.
The 14 EVV terms, formaldehyde (GVPT2 run, so anharmonic states), axes A = w1, B = -w1 + w2.

    1. the inputs: terms already translated to axes, data file, laser polarization, states choice
    2. load the data, once
    3. features (evaluate_response_terms): how many, and where they are
    4. the spectrum on a grid (spectrum_on_grid), with a width per condition
    5. what came out: the grid, the brightest point, the window as used (its features and their boxes)
    6. another width, same rows: spectrum_on_grid runs again without computing the rows again

    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.example_evaluate_response_terms
"""
import time
from dataclasses import replace
from functools import partial

import numpy as np

from wilson_suite.wilson_intensities.refac_rsp_eval.features import amplitude_half_widths
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import Box
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import (
    evaluate_response_terms,
    prop_names,
    spectrum_on_grid,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.stored_terms import load_setup
from wilson_suite.wilson_system.system_data import DataOriginInfo, MolSystemData
from wilson_suite.wilson_utils.paths import SUITE_ROOT


def main():
    # 1. inputs
    setup = load_setup('evv')       # data_for_tests/translated_terms/evv.json
    terms = setup.terms             # translated to setup.axes: A = w1, B = -w1 + w2
    data_origin = DataOriginInfo(source_type='gaussian',
                                 base_file_loc=f'{SUITE_ROOT}/data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')
    laser_pol = setup.laser_pol     # (1, 1, 1): pulses and detector along x
    states_choice = 'anharmonic'    # follows the vib analysis regime: GVPT2 -> anharmonic

    # 2. data, once, outside the evaluator: the evaluator takes a MolSystemData and reads no file.
    #    prop_names(terms): which properties the terms use, so which ones to load.
    molsys = MolSystemData.load(data_origin, prop_names(terms))

    # 3. features. Width per condition: 3 cm-1 plus 0.2 % of the condition's |E| (energy_difference is signed).
    width = lambda vd, freqs: 3. + 0.002 * abs(vd.energy_difference())

    t0 = time.perf_counter()
    result = evaluate_response_terms(terms, molsys_data=molsys, states_choice=states_choice,
                                     laser_pol=laser_pol, propagator_param=width)
    t_rows = time.perf_counter() - t0

    print(f'{len(terms)} terms -> {len(result.table)} rows, {len(result.zero)} zero pairs, '
          f'{len(result.features)} features')
    for ax in ('A', 'B'):
        values = [f.location[ax] for f in result.features]
        print(f'  features on {ax}: {min(values):8.1f} .. {max(values):8.1f} cm-1')

    # 4. the spectrum around the strongest peak (A = 2682.8, B = 0): A in 1 cm-1 steps, B over the whole
    #    range of the features in 16.65 cm-1 steps (201 points from -230 to 3100)
    window = Box({'A': (600., 2800.), 'B': (5., 3100.)})
    dim_sizes = {'A': 501, 'B': 201}

    # Boxes: the box rule, with its knobs given by partial. dynrange is the plot's dynamic range (max / floor).
    boxes = partial(amplitude_half_widths, dynrange=1001.)

    t0 = time.perf_counter()
    spec = spectrum_on_grid(result, window, dim_sizes, box_rule=boxes)
    t_on_grid = time.perf_counter() - t0

    # 5. what came out
    grid_axes = spec.grid_axes                    # 1D points per axis, cm-1
    mesh = spec.grid_mesh                         # every grid point, mesh['A'][i, j]: what a renderer reads
    intensity = np.abs(spec.grid_result) ** 2     # grid_result: complex amplitude in au, [i_A, i_B]
    print(f'\ngrid {intensity.shape}: ' + ', '.join(
        f'{ax} {pts[0]:.0f} .. {pts[-1]:.0f} cm-1, step {pts[1] - pts[0]:.2f}' for ax, pts in grid_axes.items()))

    i, j = np.unravel_index(np.argmax(intensity), intensity.shape)
    print(f'brightest grid point: A = {mesh["A"][i, j]:.1f}, B = {mesh["B"][i, j]:.1f} cm-1, '
          f'intensity {intensity[i, j]:.3e} au (the grid misses the peak centre on B)')

    used = spec.spectral_window                   # the window as used: features with their boxes (by amplitude)
    whole = sum(f.feat_box.contains_box(window) for f in used.full_features + used.contrib_features)  # type: ignore
    print(f'window: {len(used.full_features)} features inside, {len(used.contrib_features)} outside with a box '
          f'that reaches in; {whole} boxes cover the whole window')

    inside = used.full_features
    print(f'\n{len(inside)} features inside the window, strongest first:')
    print(f'  {"A":>8} {"B":>8}  {"(a, b)":>7}  {"terms":<22} {"widths at peak":<16} {"peak intensity":>14}')
    for f in sorted(inside, key=lambda f: f.get_intensity(), reverse=True):
        (ps,) = f.param_sets
        widths = ', '.join(f'{w:.2f}' for w in f.peak_widths())
        print(f'  {f.location["A"]:8.1f} {f.location["B"]:8.1f}  {str((ps["a"], ps["b"])):>7}  '  # noqa: RUF010
              f'{str(f.term_ids):<22} {widths:<16} {f.get_intensity():14.3e}')  # noqa: RUF010

    print('-----------------------------')
    for i in inside:
        for k in i.rows:
            print(k.location, k.params, k.term_id)
        print('\n')

    print('-----------------------------')


    # ---------------------------------------------------
    # 6. another width, same rows: replace gives a new FeatureResult with the same table, spectrum_on_grid uses it
    t0 = time.perf_counter()
    wider = spectrum_on_grid(replace(result, propagator_param=8.), window, dim_sizes, box_rule=boxes)
    t_again = time.perf_counter() - t0

    print(f'\n8 cm-1 everywhere: brightest intensity {np.max(np.abs(wider.grid_result) ** 2):.3e} au; '
          f'rows reused: {wider.feature_result.table is result.table}; '
          f'rows {t_rows:.3f} s, first grid {t_on_grid:.3f} s, grid again {t_again:.3f} s')


if __name__ == '__main__':
    main()
