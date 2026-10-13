"""
evaluate_response_terms, step by step. An example of use, not a test.
The 14 EVV terms, formaldehyde (GVPT2 run, so anharmonic states), axes A = w1, B = -w1 + w2.

    1. the inputs: terms as derive writes them, axes, data file, laser polarization, states choice
    2. load the data and make the orientation-average recipe, once
    3. features only (no window): how many, and where they are
    4. the spectrum on a grid (with a window), with a width per condition
    5. what came out: the grid, the brightest point, the window as drawn (its features and domains)
    6. another width, same rows: draw_spectrum draws again without computing the rows again

    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.example_evaluate_response_terms
"""
import time
from dataclasses import replace

import numpy as np

from wilson_suite.wilson_derive import term_var_translate
from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import Box
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import (
    draw_spectrum,
    evaluate_response_terms,
    load_molsys_data,
    make_polarization_linear_comb,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import compile_terms
from wilson_suite.wilson_system.system_data import DataOriginInfo
from wilson_suite.wilson_utils.builders import make_SpectralAxisSet
from wilson_suite.wilson_utils.paths import SUITE_ROOT


def main():
    # 1. inputs
    terms = VibPerturbedTerm.load_many_from_json(
        f'{SUITE_ROOT}/wilson_intensities/refac_rsp_eval/tests/test_terms.json')   # in pulse IDs
    axes = make_SpectralAxisSet({'A': [1], 'B': [-1, 2]})                          # A = w1, B = -w1 + w2
    data_origin = DataOriginInfo(source_type='gaussian',
                                 base_file_loc=f'{SUITE_ROOT}/data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')
    laser_pol = (1., 1., 1.)        # pulses and detector along x
    states_choice = 'anharmonic'    # follows the vib analysis regime: GVPT2 -> anharmonic

    # 2. data and recipe, once. Both need the compiled terms: which properties to load, which averaging rank.
    #    The translation works on copies, so `terms` stay in pulse IDs for evaluate_response_terms.
    compiled = compile_terms(term_var_translate.translate_terms_to_axis_variables(terms, axes))
    molsys = load_molsys_data(compiled, data_origin)
    recipe = make_polarization_linear_comb(compiled, laser_pol)

    # 3. features only: no window, so nothing is drawn (grid_axes and grid_result are None)
    result = evaluate_response_terms(terms, axes, molsys, states_choice, recipe, propagator_param=5.)

    features = result.feature_result.features
    print(f'{len(terms)} terms -> {len(result.feature_result.table)} rows, '
          f'{len(result.feature_result.zero)} zero pairs, {len(features)} features')
    for ax in ('A', 'B'):
        values = [f.location[ax] for f in features]
        print(f'  features on {ax}: {min(values):8.1f} .. {max(values):8.1f} cm-1')

    # 4. the spectrum around the strongest peak (A = 2682.8, B = 0), 1 cm-1 steps.
    #    Width per condition: 3 cm-1 plus 0.2 % of the condition's |E| (energy_difference is signed).
    width = lambda vd, freqs: 3. + 0.002 * abs(vd.energy_difference())
    window = Box({'A': (2600., 2800.), 'B': (-230., 3100.)})
    dim_sizes = {'A': 201, 'B': 201}

    t0 = time.perf_counter()
    result = evaluate_response_terms(terms, axes, molsys, states_choice, recipe,
                                     propagator_param=width,
                                     window=window, dim_sizes=dim_sizes,
                                     dynrange=1001.)
    t_full = time.perf_counter() - t0

    # 5. what came out
    grid_axes = result.grid_axes                    # 1D points per axis, cm-1
    mesh = result.grid_mesh                         # every grid point, mesh['A'][i, j]: what a renderer reads
    intensity = np.abs(result.grid_result) ** 2     # grid_result: complex amplitude in au, [i_A, i_B]
    print(f'\ngrid {intensity.shape}: ' + ', '.join(
        f'{ax} {pts[0]:.0f} .. {pts[-1]:.0f} cm-1, step {pts[1] - pts[0]:.1f}' for ax, pts in grid_axes.items()))

    i, j = np.unravel_index(np.argmax(intensity), intensity.shape)
    print(f'brightest point: A = {mesh["A"][i, j]:.1f}, B = {mesh["B"][i, j]:.1f} cm-1, '
          f'intensity {intensity[i, j]:.3e} au')

    drawn = result.spectral_window                  # the window as drawn: features with their boxes
    domains = drawn.find_clusters_by_featboxes()    # overlapping boxes, drawn together
    print(f'drawn: {len(drawn.full_features)} features inside, {len(drawn.contrib_features)} outside with a box '
          f'that reaches in, {len(domains)} domains')

    inside = drawn.full_features
    print(f'\n{len(inside)} features inside the window, strongest first:')
    print(f'  {"A":>8} {"B":>8}  {"(a, b)":>7}  {"terms":<22} {"widths at peak":<16} {"peak intensity":>14}')
    for f in sorted(inside, key=lambda f: f.get_intensity(), reverse=True):
        (ps,) = f.param_sets
        widths = ', '.join(f'{w:.2f}' for w in f.peak_widths())
        print(f'  {f.location["A"]:8.1f} {f.location["B"]:8.1f}  {str((ps["a"], ps["b"])):>7}  '
              f'{str(f.term_ids):<22} {widths:<16} {f.get_intensity():14.3e}')

    # 6. another width, same rows: replace gives a new FeatureResult with the same table, draw_spectrum draws it
    t0 = time.perf_counter()
    wider = draw_spectrum(replace(result.feature_result, propagator_param=8.), window, dim_sizes, dynrange=1001.)
    t_redraw = time.perf_counter() - t0
    print(f'\n8 cm-1 everywhere: brightest intensity {np.max(np.abs(wider.grid_result) ** 2):.3e} au; '
          f'rows reused: {wider.feature_result.table is result.feature_result.table}; '
          f'{t_redraw:.3f} s vs {t_full:.3f} s for the whole run')


if __name__ == '__main__':
    main()
