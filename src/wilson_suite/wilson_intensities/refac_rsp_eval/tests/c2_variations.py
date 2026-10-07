"""
C2: old vs new for the 14 EVV terms (test_terms.json), varying one setting at a time.

  states:        old regime (harmonic / VPT2 / GVPT2)  x  new states_choice (harmonic / anharmonic)
  axes:          {'A': [1], 'B': [-1, 2]},  {'A': [1], 'B': [2]},  {'A': [-1], 'B': [2]}
  polarization:  pulse / detector polarization vectors -> polarization_avg_vector
  molecule:      formaldehyde, water, CO (data_for_tests)

Per variation: coefficients (old evaluate_term_coeffs vs new build_contributions) and
locations (old resonances.solve_LSE_motif vs new solve_LSE_motif) for every (term, index set).
c2_leaves.py goes one level deeper: every full label set, factor by factor.

Result 2026-10-07: old = new in every matched variation (todo.txt C2). The two "mismatch" variations
are controls: they must differ, which shows the comparison can fail.

Needs the old code. Delete together with it (todo.txt, order of work step 10).
    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.c2_variations
"""
import traceback
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

import wilson_suite as ws
import wilson_suite.wilson_experiment.experiment_abstractions as wexp
from wilson_suite.wilson_derive import term_var_translate
from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_intensities.amplitudes.averaging import getGeneralPolarizationAveragingExpression
from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import EvaluationWorkflow
from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import Box as OldBox
from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow as OldWindow
from wilson_suite.wilson_intensities.amplitudes.term_parts import ResonanceMotif as OldMotif
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import build_contributions, solve_LSE_motif
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import ParameterSet, compile_terms
from wilson_suite.wilson_system.system_data import (
    DataOriginInfo,
    MolecularProperty,
    MolPropsCollection,
    MolSystemData,
    _sys_info_request,
)
from wilson_suite.wilson_utils.builders import make_SpectralAxisSet
from wilson_suite.wilson_utils.paths import SUITE_ROOT
from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer

TERMS = SUITE_ROOT + '/wilson_intensities/refac_rsp_eval/tests/test_terms.json'
MOLECULES = {
    'formaldehyde': (SUITE_ROOT + '/data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out', 'B3LYP', 'cc-pVQZ', 4),
    'water':        (SUITE_ROOT + '/data_for_tests/g16_h2o_HF_STO3G.out', 'HF', 'STO-3G', 3),
    'CO':           (SUITE_ROOT + '/data_for_tests/g16_co_HF_STO3G.out', 'HF', 'STO-3G', 2),
}
X, Y = (1., 0., 0.), (0., 1., 0.)
D45 = (2 ** -0.5, 2 ** -0.5, 0.)
RTOL = 1e-9


@dataclass
class Variation:
    name: str
    old_regime: str = 'GVPT2'
    new_states: str = 'anharmonic'
    axes: dict = field(default_factory=lambda: {'A': [1], 'B': [-1, 2]})
    pols: tuple = (X, X, X, X)          # pulse 1, pulse 2, UV/VIS pulse, detector
    molecule: str = 'formaldehyde'


VARIATIONS = [
    # states
    Variation('GVPT2 / anharmonic'),
    Variation('VPT2 / anharmonic', old_regime='VPT2'),
    Variation('harmonic / harmonic', old_regime='harmonic', new_states='harmonic'),
    Variation('GVPT2 / harmonic  (mismatch)', new_states='harmonic'),
    Variation('harmonic / anharmonic  (mismatch)', old_regime='harmonic'),
    # axes
    Variation('axes A=w1, B=w2', axes={'A': [1], 'B': [2]}),
    Variation('axes A=-w1, B=w2', axes={'A': [-1], 'B': [2]}),
    # polarization (an odd number of y gives a zero average, so pairs of y)
    Variation('pol: pulses 1, 2 along y', pols=(Y, Y, X, X)),
    Variation('pol: pulse 1, detector along y', pols=(Y, X, X, Y)),
    Variation('pol: pulse 2, detector along y', pols=(X, Y, X, Y)),
    Variation('pol: pulse 1 at 45 deg', pols=(D45, X, X, X)),
    # molecules
    Variation('water', molecule='water'),
    Variation('CO', molecule='CO'),
    Variation('water, harmonic / harmonic', molecule='water', old_regime='harmonic', new_states='harmonic'),
    Variation('CO, harmonic / harmonic', molecule='CO', old_regime='harmonic', new_states='harmonic'),
]


def key(ps) -> tuple:
    """Old ParameterSet, new ParameterSet or a plain dict -> sorted ((label, mode), ...) without 'zero'."""
    d = dict(ps._parameters) if hasattr(ps, '_parameters') else dict(ps)
    return tuple(sorted((k, int(v)) for k, v in d.items() if k != 'zero'))


def evv_experiment(pols):
    """fixtures.evv_experiment with the four polarizations as arguments."""
    p1, p2, p3, det = pols
    pulses = (wexp.make_impulsive_gaussian_pulse(tc=50.0, cf=0.0, cf_uv=0.0, maxstr=1.0e-5, wv=(0., 0., 1.), pol=p1, id=1),
              wexp.make_impulsive_gaussian_pulse(tc=100.0, cf=0.0, cf_uv=0.0, maxstr=1.0e-5, wv=(0., 0., 1.), pol=p2, id=2),
              wexp.make_impulsive_gaussian_pulse(tc=120.0, cf=0.0, cf_uv=0.072, maxstr=1.0e-5, wv=(0., 0., 1.), pol=p3, id=3))
    detector = wexp.SpecDetector(detection_method='freq', detector_location=(0., 0., 1.), detection_polarization=det,
                                 detection_range=[0.003 + 0.0001 * i for i in range(101)], wv_filter=[{1: -1, 2: 1, 3: 1}])
    scan = wexp.SpecScan(scan_objs=(wexp.ScanObject('pulse', 'cf', id=1, coeff=1.0),
                                    wexp.ScanObject('detector', 'detection_range', id=0, coeff=1.0)),
                         range=[0.0001 * i for i in range(101)])
    return wexp.VibExperiment(field=wexp.ElectricField(pulses), detector=detector, scans=(scan,), magn_conditions=((-1, 2),))


def run_old(v: Variation):
    """old pipeline, set up as in test_integration_evaluation.py test_full_integration_EVV_axes"""
    path, lvl, basis, natoms = MOLECULES[v.molecule]
    sim = ws.main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(evv_experiment(v.pols))
    sim.addTerms(terms=VibPerturbedTerm.load_many_from_json(TERMS))
    mol_system = ws.main.abstractions.MolecularSystem(name=v.molecule, natoms=natoms)
    sim.addSystem(mol_system)
    sim.addVibAnaSetup(ws.main.abstractions.VibAnaSetup(system=mol_system, regime=v.old_regime, vibana_own_analysis='none'))
    sim.addPropEvalSetup(eval_uniform=ws.main.abstractions.DataOriginInfo(source_type='gaussian', lvl_theory=lvl,
                                                                          basis_set=basis, base_file_loc=path))
    sim.setPropsAndMaxStateLvl()
    sim.dressPropsWithSetup()
    sim.setAxisChoiceAndTranslateTerms(make_SpectralAxisSet(v.axes))
    sim.getResults(obtainer=wilson_data_obtainer)
    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{
        'spectral_window': OldWindow(box=OldBox({'A': (0., 5000.), 'B': (0., 5000.)})),
        'Gamma': 4.7, 'Gamma_unit': 'cm-1', 'grid_resolution': {'A': 7, 'B': 10}, 'minimum_box_padding': 30.})
    sim.addSpecEvalSetup(ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi))
    sim.vib_ana_setup.set_include_modes_list()
    return sim, EvaluationWorkflow(sim).bound_motifs_ctx


def run_new(v: Variation, laser_pol):
    """new pipeline; laser_pol comes from the old experiment object (shared, not checked here)"""
    path = MOLECULES[v.molecule][0]
    translated = term_var_translate.translate_terms_to_axis_variables(VibPerturbedTerm.load_many_from_json(TERMS),
                                                                      make_SpectralAxisSet(v.axes))
    compiled = compile_terms(terms=translated)
    origin = DataOriginInfo(source_type='gaussian', base_file_loc=path)
    request, props = {}, {}
    for t in compiled:
        request.update(t.all_props.build_request_dict(calc_setup=origin))
        for p in t.all_props:
            mp = MolecularProperty.from_polprop(p)
            props.setdefault(mp.trivial_name, mp)
    request.update(_sys_info_request(origin))
    molsys = MolSystemData.from_datadict(mol_props=MolPropsCollection(properties=list(props.values())),
                                         data_dict=wilson_data_obtainer(requested_data_dict=request),
                                         states_choice=v.new_states)
    iso = getGeneralPolarizationAveragingExpression(rank=4, laser_pol=tuple(laser_pol))
    table, zero, failed = build_contributions(compiled, molsys, polarization_linear_comb=iso)
    return compiled, molsys, table, zero, failed


def compare(v: Variation) -> dict:
    sim, bound = run_old(v)
    laser_pol = sim.exp.polarization_avg_vector
    compiled, molsys, table, zero, failed = run_new(v, laser_pol)
    old_terms = bound.axes.terms

    new_coeff = {i: {} for i in range(len(compiled))}
    for r in table:
        new_coeff[r.term_id][key(r.params)] = r.coeff
    for i, ps in zero:
        new_coeff[i][key(ps)] = 0.
    for i, ps, c in failed:
        new_coeff[i][key(ps)] = c

    out = dict(old_states=tuple(sorted((s.state_label, round(s.energy, 3)) for s in sim.vib_ana_setup.states)),
               pol=[round(x, 4) for x in laser_pol], n_modes=len(molsys.eigenvals), terms=len(compiled),
               rows=len(table), zero=len(zero), failed=len(failed), sets_mismatch=0,
               coeff_diff=0, coeff_max_rel=0., zero_both=0, loc_diff=0, loc_max=0., old_none=0, per_term=[])
    for i, (old_t, new_t) in enumerate(zip(old_terms, compiled)):
        old_c = {key(ps): val[0] for ps, val in bound.coefficients[old_t].items()}
        new_c = new_coeff[i]
        out['sets_mismatch'] += len(set(old_c) ^ set(new_c))
        ratios, n_diff, n_loc_diff = Counter(), 0, 0
        for k in set(old_c) & set(new_c):
            o, n = old_c[k], new_c[k]
            if o == 0. and n == 0.:
                out['zero_both'] += 1
                continue
            out['coeff_max_rel'] = max(out['coeff_max_rel'], abs(n - o) / max(abs(o), abs(n)))
            if n * o < 0:
                out.setdefault('sign_flips', []).append((i, dict(k), o, n))
            if not np.isclose(n, o, rtol=RTOL, atol=0.):
                n_diff += 1
                ratios['inf' if o == 0 else f'{n / o:.3g}'] += 1
        for loc, idxs_list in bound.motif_locs[OldMotif(old_t.res)].items():
            o = None if loc is None else dict(loc._coord_dict)
            for idxs in idxs_list:
                try:
                    n = solve_LSE_motif(new_t.cmp_resmotf, ParameterSet(dict(key(idxs))), molsys.states, unit='cm-1').as_dict()
                except np.linalg.LinAlgError:
                    n = None
                if o is None:
                    out['old_none'] += 1
                same = (o is None and n is None) or (o is not None and n is not None and set(o) == set(n))
                if same and o is not None:
                    dev = max(abs(o[a] - n[a]) for a in o)
                    out['loc_max'] = max(out['loc_max'], dev)
                    same = dev < 1e-6
                if not same:
                    n_loc_diff += 1
        out['coeff_diff'] += n_diff
        out['loc_diff'] += n_loc_diff
        out['per_term'].append((i, len(old_c), n_diff, n_loc_diff, ratios.most_common(3)))
    return out


def main():
    results = []
    for v in VARIATIONS:
        try:
            results.append((v, compare(v), None))
        except Exception as e:
            results.append((v, None, f'{type(e).__name__}: {e}'.splitlines()[0][:150]))
            traceback.print_exc()

    print('\n\n==================== SUMMARY ====================')
    print(f'{"variation":<36} {"pol vector":<22} {"modes":>5} {"pairs":>5} {"rows":>4} {"zero":>4} {"fail":>4} | '
          f'{"coeff diff":>10} {"max rel":>8} | {"loc diff":>8} {"max cm-1":>8} {"old None":>8}')
    for v, r, err in results:
        if err:
            print(f'{v.name:<36} ERROR  {err}')
            continue
        pairs = r['rows'] + r['zero'] + r['failed']
        print(f'{v.name:<36} {str(r["pol"]):<22} {r["n_modes"]:>5} {pairs:>5} {r["rows"]:>4} {r["zero"]:>4} {r["failed"]:>4} | '
              f'{r["coeff_diff"]:>10} {r["coeff_max_rel"]:>8.1e} | {r["loc_diff"]:>8} {r["loc_max"]:>8.2f} {r["old_none"]:>8}'
              + (f'   (index sets only on one side: {r["sets_mismatch"]})' if r['sets_mismatch'] else ''))

    by_name = {v.name: r for v, r, err in results if r is not None}
    g, p, h = (by_name.get(n) for n in ('GVPT2 / anharmonic', 'VPT2 / anharmonic', 'harmonic / harmonic'))
    if g and p and h:
        n_diff = lambda a, b: sum(x != y for x, y in zip(a['old_states'], b['old_states']))
        print(f'\nold states, formaldehyde: {len(g["old_states"])} states; VPT2 vs GVPT2 differ in {n_diff(g, p)}, '
              f'harmonic vs GVPT2 differ in {n_diff(g, h)}')
        print('   first states GVPT2:   ', g['old_states'][:4])
        print('   first states harmonic:', h['old_states'][:4])

    print('\n==================== PER TERM (variations with differences) ====================')
    for v, r, err in results:
        if err or (r['coeff_diff'] == 0 and r['loc_diff'] == 0):
            continue
        flips = r.get('sign_flips', [])
        print(f'\n{v.name}: {len(flips)} coefficients change sign; first: {flips[:3]}')
        print(f'{v.name}:  term | sets | coeff diff | loc diff | new/old ratios')
        for i, n_sets, n_diff, n_loc, ratios in r['per_term']:
            print(f'   {i:>4} | {n_sets:>4} | {n_diff:>10} | {n_loc:>8} | ' + ', '.join(f'{q} x{c}' for q, c in ratios))


if __name__ == '__main__':
    main()
