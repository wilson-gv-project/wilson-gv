"""
C3: whole spectrum, old vs new (todo.txt C3). The 14 EVV terms, formaldehyde, axes A = w1, B = -w1 + w2.

Cases:
  formaldehyde    GVPT2 / anharmonic states, as the first C2 variation
  degenerate toy  harmonic / harmonic states, mode 5 given the harmonic frequency of mode 3 (degenerate_obtainer).
                  The data is changed where both pipelines read it, so old and new get the same toy molecule.
                  Then several parameter sets share one location, e.g. a = 3 and a = 5 with the same b.

Per case, on one grid (the old SpectralWindow.sample_grid over a window around all features):
  1. features: old (nonzero amplitude) vs new, matched by location
  2. each feature on the whole grid: old evaluate_feature vs new eval_feature_on_grid.
     Expected: old = n * new, n = number of parameter sets at the location (old: one shape per parameter set)
  3. no cut: sum of old evaluate_feature vs draw_all; also the old sum with each feature divided by its n
  4. cut by domains: old EvaluationWorkflow.evaluate vs draw_window, each also vs its own no-cut sum

Result 2026-10-09 (also todo.txt C3), dynrange 100, gamma 4.7 cm-1, grid step 2 cm-1,
after from_features stopped dropping weak features (it drops no feature for its amplitude):
  formaldehyde: 68 = 68 features, every one with n = 1. Old = new: amplitudes, each feature, the sum without a cut
                (1e-15), and the cut by domains (68 = 68 features drawn, 30 = 30 domains, 1e-16).
                Cut vs no cut: 1.05 % of the max intensity, old and new alike.
  degenerate toy: 50 = 50 features, n = 1 (32), 2 (16), 4 (2). Old = n * new for every feature (1e-14):
                the old code draws a peak n times too large (n^2 in intensity). Whole spectrum: 14 % of the max
                amplitude, 5.7 % of the max intensity, with or without the cut. Old with each feature / n = new (5e-16).

Needs the old code. Delete together with it (todo.txt, order of work step 10).
    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.c3_spectrum
"""
import copy
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from unittest import mock

import numpy as np

import wilson_suite as ws
from wilson_suite.wilson_derive import term_var_translate
from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import EvaluationWorkflow
from wilson_suite.wilson_intensities.amplitudes.evaluators import evaluate_feature
from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import (
    Box as OldBox,
)
from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import (
    SpectralWindow as OldWindow,
)
from wilson_suite.wilson_intensities.refac_rsp_eval import pipeline
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    draw_all,
    draw_window,
    eval_feature_on_grid,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import SpectralWindow
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import Box
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import compile_terms
from wilson_suite.wilson_intensities.refac_rsp_eval.tests import c2_variations as c2
from wilson_suite.wilson_system.system_data import DataOriginInfo
from wilson_suite.wilson_utils.builders import make_SpectralAxisSet
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer

AXES = {'A': [1], 'B': [-1, 2]}
POLS = (c2.X, c2.X, c2.X, c2.X)
GAMMA_CM = 4.7
DYNRANGE = 100.          # old EvaluationInfo default
MARGIN = 0.1             # box_range_safety_margin, old default
MIN_PADDING = 30.        # old only; boxes are 4.7 * sqrt(99) * 1.1 = 51 cm-1 anyway
WINDOW_PAD_CM = 60.      # window = feature locations +- this: every feature box inside the window
STEP_CM = 2.             # grid step, about GAMMA_CM / 2
LOC_DIGITS = 4           # locations match to 1e-4 cm-1 (C2: old = new locations to 1e-6)


def degenerate_obtainer(mode: int = 5, like: int = 3) -> Callable:
    """
    wilson_data_obtainer, but mode `mode` gets the harmonic frequency of mode `like`. Every harmonic state energy
    is rebuilt as the sum of its modes' frequencies (in the file they are that sum, to 0.001 cm-1).
    Only harmonic data changes: use harmonic states on both sides.
    """
    def obtain(*args, **kwargs):
        data = copy.deepcopy(wilson_data_obtainer(*args, **kwargs))
        if 'nc_sqrt_eigval' in data:
            freqs = data['nc_sqrt_eigval']
            freqs[mode] = freqs[like]
            if 'harmonic_states' in data:
                data['harmonic_states'] = {k: sum(freqs[int(m)] for m in k) for k in data['harmonic_states']}
        return data
    return obtain


@dataclass
class Case:
    name: str
    old_regime: str
    new_states: str
    obtainer: Callable = wilson_data_obtainer


CASES = [
    Case('formaldehyde', 'GVPT2', 'anharmonic'),
    Case('degenerate toy (mode 5 = mode 3, harmonic)', 'harmonic', 'harmonic', degenerate_obtainer()),
]


def run_old(case: Case):
    """old pipeline, set up as c2_variations.run_old, with the case's data obtainer"""
    path, lvl, basis, natoms = c2.MOLECULES['formaldehyde']
    sim = ws.main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(c2.evv_experiment(POLS))
    sim.addTerms(terms=VibPerturbedTerm.load_many_from_json(c2.TERMS))
    mol_system = ws.main.abstractions.MolecularSystem(name='formaldehyde', natoms=natoms)
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(ws.main.abstractions.VibAnaSetup(system=mol_system, regime=case.old_regime,
                                                        vibana_own_analysis='none'))
    sim.addPropEvalSetup(eval_uniform=ws.main.abstractions.DataOriginInfo(source_type='gaussian', lvl_theory=lvl,
                                                                          basis_set=basis, base_file_loc=path))
    sim.setPropsAndMaxStateLvl()
    sim.dressPropsWithSetup()
    sim.setAxisChoiceAndTranslateTerms(make_SpectralAxisSet(AXES))
    sim.getResults(obtainer=case.obtainer)
    evi = ws.main.spectrum_abstractions.EvaluationInfo(Gamma=GAMMA_CM, Gamma_unit='cm-1', dynamic_range=DYNRANGE,
                                                       box_range_safety_margin=MARGIN, minimum_box_padding=MIN_PADDING)
    sim.addSpecEvalSetup(ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi))
    sim.vib_ana_setup.set_include_modes_list()
    return sim, EvaluationWorkflow(sim)


def run_new(case: Case, laser_pol):
    """new pipeline, the steps of compute_features_from_terms, keeping molsys for drawing"""
    translated = term_var_translate.translate_terms_to_axis_variables(VibPerturbedTerm.load_many_from_json(c2.TERMS),
                                                                      make_SpectralAxisSet(AXES))
    compiled = compile_terms(translated)
    origin = DataOriginInfo(source_type='gaussian', base_file_loc=c2.MOLECULES['formaldehyde'][0])
    with mock.patch.object(pipeline, 'wilson_data_obtainer', case.obtainer):
        molsys = pipeline.load_molsys_data(compiled, origin, case.new_states)
    result = pipeline.compute_features(compiled, molsys, pipeline.make_polarization_linear_comb(compiled, laser_pol),
                                       lineshape_parameter=GAMMA_CM)
    return molsys, result


def loc_key(coord_dict: dict) -> tuple:
    return tuple(sorted((ax, round(float(v), LOC_DIGITS)) for ax, v in coord_dict.items()))


def by_location(features, coord_dict_of) -> dict:
    out = {}
    for f in features:
        k = loc_key(coord_dict_of(f))
        if k in out:
            raise ValueError(f'two features at one location {k}: cannot match old and new by location')
        out[k] = f
    return out


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """max |a - b| / max |b|"""
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def intensity_rel(a: np.ndarray, b: np.ndarray) -> float:
    """amplitudes in, intensities compared: max | |a|^2 - |b|^2 | / max |b|^2"""
    return rel(np.abs(a) ** 2, np.abs(b) ** 2)


def compare(case: Case) -> dict:
    sim, wf = run_old(case)
    molsys, result = run_new(case, sim.exp.polarization_avg_vector)
    new_feats = result.features
    out = {'old_E3_E5': [s.energy for s in sim.vib_ana_setup.states if s.state_label in ('3', '5')],
           'new_E3_E5': [molsys.states.get_energy_by_label(lbl) for lbl in ('3', '5')]}

    # window around every feature, grid from the old sample_grid
    bounds = {}
    for ax in AXES:
        vals = [f.location[ax] for f in new_feats]
        bounds[ax] = (float(min(vals)) - WINDOW_PAD_CM, float(max(vals)) + WINDOW_PAD_CM)
    resolution = {ax: int((mx - mn) / STEP_CM) + 1 for ax, (mn, mx) in bounds.items()}
    old_spec = wf.evaluate(spec_window=OldWindow(box=OldBox(dict(bounds))), grid_resolution=resolution)
    coords = {'A': old_spec.axes['A'][:, 0], 'B': old_spec.axes['B'][0, :]}
    mesh_cm = dict(zip(coords, np.meshgrid(*coords.values(), indexing='ij')))
    mesh_au = {ax: convNu2Ene(v) for ax, v in mesh_cm.items()}
    out['grid'] = {ax: (len(v), round(float(v[0]), 1), round(float(v[-1]), 1)) for ax, v in coords.items()}

    # 1. features
    old_feats = wf.feat_result.features
    if any(len(f.term_contributions) != 1 for f in old_feats):
        raise ValueError('expected one term group per old feature')
    old_by_loc = by_location(old_feats, lambda f: f.location._coord_dict)
    new_by_loc = by_location(new_feats, lambda f: f.location.as_dict())
    common = set(old_by_loc) & set(new_by_loc)
    n_of = {k: len(old_by_loc[k].term_contributions[0].states_parameters) for k in old_by_loc}
    out.update(old_feats=len(old_feats), old_zero_feats=len(wf.feat_result.zero_feats), new_feats=len(new_feats),
               matched=len(common), only_old=sorted(set(old_by_loc) - common), only_new=sorted(set(new_by_loc) - common),
               n_counts=dict(sorted(Counter(n_of.values()).items())),
               amp_max_rel=max(abs(old_by_loc[k].amplitude_coeff - new_by_loc[k].amplitude_coeff)
                               / abs(new_by_loc[k].amplitude_coeff) for k in common))

    # 2. each feature on the whole grid; 3. sums without a cut
    qc = wf.qcdata_ctx
    gamma_au = convNu2Ene(GAMMA_CM)
    old_all = np.zeros(mesh_cm['A'].shape, dtype=complex)
    old_fixed = np.zeros_like(old_all)
    ratio_dev = 0.
    for k in common:
        old_grid = evaluate_feature(old_by_loc[k], qc.vibstates_data, qc.vibdiff_cache, gamma_au, mesh_au)
        new_grid = eval_feature_on_grid(new_by_loc[k], molsys.states, mesh_cm)
        ratio_dev = max(ratio_dev, rel(old_grid, n_of[k] * new_grid))
        old_all += old_grid
        old_fixed += old_grid / n_of[k]
    new_all = draw_all(new_feats, molsys.states, coords)
    out.update(per_feature_old_vs_n_new=ratio_dev,
               nocut_old_vs_new_amp=rel(old_all, new_all),
               nocut_old_vs_new_intensity=intensity_rel(old_all, new_all),
               nocut_old_fixed_vs_new_amp=rel(old_fixed, new_all))

    # 4. cut by domains. Both keep every feature near the window, however weak (old: minimum_box_padding > 0)
    window = SpectralWindow.from_features(Box(dict(bounds)), new_feats, DYNRANGE, MARGIN)
    new_cut = draw_window(window, molsys.states, coords)
    old_cut = old_spec.result
    out.update(window_feats_old_new=(sum(len(r.features) for r in wf.region_eval.regions),
                                     len(window.full_features) + len(window.contrib_features)),
               domains_old_new=(len(wf.region_eval.regions), len(window.find_clusters_by_featboxes())),
               cut_old_vs_own_nocut_intensity=intensity_rel(old_cut, old_all),
               cut_new_vs_own_nocut_intensity=intensity_rel(new_cut, new_all),
               cut_old_vs_new_amp=rel(old_cut, new_cut),
               cut_old_vs_new_intensity=intensity_rel(old_cut, new_cut))
    return out


def main():
    for case in CASES:
        r = compare(case)
        print(f'\n==================== {case.name} ====================')
        for k, v in r.items():
            print(f'{k:<34} {v}')


if __name__ == '__main__':
    main()
