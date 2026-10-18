"""
Figures with plot_rsp_func and plot_spectrum (wilson_analysis/render/plot_spectrum.py). An example of use, not a
test. The 14 EVV terms, axes A = w1, B = -w1 + w2, gamma 4.7 cm-1, boxes and plot with dynrange 500.

    1. formaldehyde, anharmonic states: the whole window
    2. formaldehyde: +-150 cm-1 around the strongest peak, 1 cm-1 steps
    2b. 1D: the row of 2 through the strongest peak, with plot_spectrum on plain arrays
    3. formaldehyde, harmonic vs anharmonic states side by side, on one shared scale (reference_max)
    4. ACDM (42 modes), anharmonic states: the window of examples/quick_start.ipynb, 700 x 700 points
       (about 10 s: loading the data takes most of it; the loading warns about anharmonic states below 0)

    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.example_plot_spectrum [out_dir]
out_dir: where the png files go; default plot_spectrum_figures (made if missing).
"""
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from wilson_suite.wilson_analysis.render.plot_spectrum import plot_rsp_func, plot_spectrum
from wilson_suite.wilson_intensities.refac_rsp_eval.features import amplitude_half_widths
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
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.f2_speed import load_acdm
from wilson_suite.wilson_system.system_data import MolSystemData

DYNRANGE = 500.         # the boxes (amplitude_half_widths) and the plot use the same
GAMMA_CM = 4.7
BOXES = partial(amplitude_half_widths, dynrange=DYNRANGE)
LABELS = {'A': r'A = $\omega_1/2\pi c$, cm$^{-1}$', 'B': r'B = $(-\omega_1+\omega_2)/2\pi c$, cm$^{-1}$'}


def evaluate(terms, molsys: MolSystemData, states: str) -> FeatureResult:
    return evaluate_response_terms(terms, molsys_data=molsys, states_choice=states, laser_pol=ref.LASER_POL, propagator_param=GAMMA_CM)


def save(fig, path: Path, dpi: int = 110):
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  {path}')


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    terms = ref.evv_terms()
    formaldehyde = MolSystemData.load(ref.data_origin('formaldehyde'), prop_names(terms))
    anharmonic, harmonic = evaluate(terms, formaldehyde, 'anharmonic'), evaluate(terms, formaldehyde, 'harmonic')

    # 1. the whole window
    spec = spectrum_on_grid(anharmonic, Box({'A': (600., 3100.), 'B': (5., 3100.)}), {'A': 501, 'B': 501},
                            box_rule=BOXES)
    fig, ax, _, _ = plot_rsp_func(spec, dynrange=DYNRANGE, axis_labels=LABELS)
    ax.set_title('formaldehyde, 14 EVV terms, anharmonic states, dynrange 500')
    save(fig, out / '1_formaldehyde.png')

    # 2. +-150 cm-1 around the strongest peak, 1 cm-1 steps
    strongest = max(anharmonic.features, key=lambda f: abs(f.amplitude_coeff))  # type: ignore
    a0, b0 = strongest.location['A'], strongest.location['B']
    zoom = Box({'A': (a0 - 150., a0 + 150.), 'B': (b0, b0 + 300.)})
    spec = spectrum_on_grid(anharmonic, zoom, {'A': 301, 'B': 301}, box_rule=BOXES)
    fig, ax, _, _ = plot_rsp_func(spec, dynrange=DYNRANGE, axis_labels=LABELS)
    ax.set_title(f'zoom: strongest peak at A = {a0:.1f}, B = {b0:.1f} cm$^{{-1}}$')
    save(fig, out / '2_formaldehyde_zoom.png')

    # 2b. 1D: the zoom's row at B = b0, with plot_spectrum on plain arrays
    j = int(np.argmin(np.abs(spec.grid_axes['B'] - b0)))
    intensity = np.abs(spec.grid_result[:, j]) ** 2
    fig, ax, _, _ = plot_spectrum(intensity, {'A': spec.grid_axes['A']}, dynrange=DYNRANGE, axis_labels=LABELS)
    ax.set_title(f'1D: the zoom at B = {spec.grid_axes["B"][j]:.1f} cm$^{{-1}}$')
    save(fig, out / '2b_formaldehyde_zoom_1d.png')

    # 3. harmonic vs anharmonic: reference_max = the larger of the two maxima, so both share one scale
    window, sizes = Box({'A': (1100., 3000.), 'B': (5., 3000.)}), {'A': 401, 'B': 401}
    specs = {name: spectrum_on_grid(result, window, sizes, box_rule=BOXES)
             for name, result in (('anharmonic', anharmonic), ('harmonic', harmonic))}
    reference_max = max(np.max(np.abs(s.grid_result)) ** 2 for s in specs.values())
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (name, spec) in zip(axes, specs.items()):
        plot_rsp_func(spec, dynrange=DYNRANGE, reference_max=reference_max, ax=ax, axis_labels=LABELS)
        ax.set_title(f'formaldehyde, {name} states')
    save(fig, out / '3_formaldehyde_harmonic_vs_anharmonic.png', dpi=100)

    # 4. ACDM, the quick_start window
    t0 = time.perf_counter()
    acdm = load_acdm(terms)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    spec = spectrum_on_grid(evaluate(terms, acdm, 'anharmonic'), Box({'A': (700., 3400.), 'B': (10., 3500.)}),
                         {'A': 700, 'B': 700}, box_rule=BOXES)
    t_eval = time.perf_counter() - t0
    t0 = time.perf_counter()
    fig, ax, _, _ = plot_rsp_func(spec, dynrange=DYNRANGE, axis_labels=LABELS)
    ax.set_title('ACDM (42 modes), anharmonic states, dynrange 500')
    save(fig, out / '4_acdm.png')
    t_plot = time.perf_counter() - t0
    print(f'ACDM: load {t_load:.1f} s, rows and grid {t_eval:.1f} s, plot and save {t_plot:.1f} s')


if __name__ == '__main__':
    main(Path(sys.argv[1] if len(sys.argv) > 1 else 'plot_spectrum_figures'))
