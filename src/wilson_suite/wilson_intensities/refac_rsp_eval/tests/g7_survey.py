"""
G7: other experiments, through derive and the new evaluator (todo.txt G7; also E8, E11, E12, E13).
The setups: ~/pre_paper3/try.py (its named experiments, explicit runs and sign sweeps), the seminar notebook
~/hyllseminar_12062026/hyllseminar_wilson_demo.ipynb, and two 1D setups (IR absorption, IR + vis SFG).

Per setup:
    terms       derive's terms, all / unique
    axes        the canonical axes (VibExperiment.canonical_axes; else its first valid choice)
    patterns    per term, the axes of each condition after the translation to those axes, e.g. (A ; A+B) = one
                condition on A, one on A + B. Kinds:
                    point{AB}    a point on the axes A, B (B7 passes); the same along any other axis (E11)
                    line, plane  fewer independent conditions than the axes they use (B7 rejects them, E8)
                    flat         no conditions (E12)
    rank        the averaging rank of each term (pipeline.averaging_rank allows only one per run, 2 to 6)
    quanta      the most quanta in a state of any term (the Gaussian files have up to 3)
    missing     the props the Gaussian reader does not give (on formaldehyde)
    derive tr.  derive's own translation: ok, or where it stops (3 axes at most, E13)
    evaluation  compute_features on formaldehyde (anharmonic states, gamma 10 cm-1, the experiment's laser_pol),
                unless something blocks it: B7, more than one rank, props, more than 3 quanta, size
The translation here is derive's algorithm (term_var_translate.find_pulse_id_tuples_as_axis_vars) without its
3-axis stop (E13), so 4-axis setups get their patterns too.

    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.g7_survey                  every setup
    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.g7_survey --only CARS "3 IR (nb)"
    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.g7_survey --terms terms.txt
--only: the setups whose name holds one of the words. --terms: also write every unique term (in axes, to_str) to
that file. Every setup: about 12 min, most of it derive on the 5-pulse sweeps (about 20000 terms each).

Result 2026-10-15: todo.txt G7 (the table of patterns, and what runs). 1D and 2D run end to end (CARS, EVV).
Pulses at different times give only points; pulses at the same time give lines and planes. Setups with more
than 2 axes mix axis sets (3 IR: {A, B} and {A, B, C}). 4 IR + UV: 4 axes, stopped by derive's translation.
"""
import argparse
import copy
import time
from collections import Counter
from dataclasses import replace
from itertools import product

import numpy as np

import wilson_suite.wilson_experiment.experiment_abstractions as wexp
from wilson_suite.wilson_intensities.refac_rsp_eval.compat import term_to_str, translate_terms_to_axis_variables
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import (
    compute_features,
    make_polarization_linear_comb,
    prop_names,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    ResonanceMotif,
    compile_terms,
    generate_LHS_motif,
    split_props,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests import (
    make_evv_reference as ref,
)
from wilson_suite.wilson_system.system_data import MolSystemData

N_MODES = 6             # formaldehyde
MAX_QUANTA = 3          # the most quanta in the Gaussian files' states
MAX_INDEX_SETS = 3e6    # no evaluation above this many index sets (sum over the terms of N_MODES ** labels)
GAMMA_CM = 10.


# ---- pulses and experiments, as in try.py ----
def uv(tc: float, cf_uv: float):
    return wexp.make_impulsive_gaussian_pulse(tc=tc, cf=0.0, cf_uv=cf_uv, maxstr=1e-5, wv=(0., 0., 1.),
                                              pol=(1., 0., 0.), id=1)


def ir(tc: float):
    return wexp.make_impulsive_gaussian_pulse(tc=tc, cf=0.0, cf_uv=0.0, maxstr=1e-5, wv=(0., 0., 1.),
                                              pol=(1., 0., 0.), id=1)


def make_experiment(pulses, wv_filter: dict, magn_conditions=None) -> wexp.VibExperiment:
    """try.py's make_experiment: pulse ids 1..N in the given order, detector along x"""
    pulses = tuple(replace(p, id=i + 1) for i, p in enumerate(pulses))
    detector = wexp.SpecDetector(detection_method='freq', detector_location=(0., 0., 1.),
                                 detection_polarization=(1., 0., 0.),
                                 detection_range=[0.003 + 0.0001 * i for i in range(101)], wv_filter=[wv_filter])
    scan = wexp.SpecScan(scan_objs=(wexp.ScanObject('pulse', 'cf', id=1, coeff=1.0),
                                    wexp.ScanObject('detector', 'detection_range', id=0, coeff=1.0)),
                         range=[0.0001 * i for i in range(101)])
    return wexp.VibExperiment(field=wexp.ElectricField(pulses), detector=detector, scans=(scan,),
                              magn_conditions=magn_conditions)


def sweep(name: str, pulses, fixed: dict | None = None) -> list:
    """try.py's sweep_sign_patterns: one setup per +/- pattern of the filter; `fixed` keeps some signs"""
    fixed = fixed or {}
    free = [i + 1 for i in range(len(pulses)) if i + 1 not in fixed]
    setups = []
    for signs in product([-1, 1], repeat=len(free)):
        wv_filter = {**fixed, **dict(zip(free, signs))}
        label = ' '.join(f"{'+' if wv_filter[k] > 0 else '-'}{k}" for k in sorted(wv_filter))
        setups.append((f'{name} [{label}]', pulses, wv_filter, None))
    return setups


IR_50, IR_80, UV_7 = ir(50.), ir(80.), uv(90., 0.192)
SETUPS = [   # (name, pulses, wv_filter, magn_conditions)
    # try.py, the named experiments
    ('try: CARS', (uv(10., .072), uv(10., .072), uv(120., .092)), {1: 1, 2: -1, 3: 1}, None),
    ('try: CAHRS (3 pulses, 5-entry filter)', (uv(10., .072), uv(10., .072), uv(120., .092)),
     {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, None),
    ('try: beta-alpha CARS', (uv(10., .072), uv(10., .072), uv(10., .144), uv(120., .072)),
     {1: 1, 2: 1, 3: -1, 4: 1}, None),
    ('try: EVV, IR 1 and 2 at the same time', (ir(100.), ir(100.), uv(120., .072)), {1: -1, 2: 1, 3: 1},
     ((-1, 2),)),
    # try.py, the explicit runs
    ('try: CAHRS?? (5 UV)', (uv(10., .05), uv(10., .05), uv(10., .1), uv(110., .05), uv(110., .05)),
     {1: 1, 2: 1, 3: -1, 4: 1, 5: 1}, ((1, 2, -3),)),
    ('try: 3 IR + UV, IR 1 and 2 at the same time', (IR_50, IR_50, IR_80, UV_7), {1: -1, 2: -1, 3: 1, 4: 1}, None),
    ('try: 4 IR + UV, IR in 2 same-time pairs, UV with the 2nd',
     (ir(10.), ir(10.), ir(20.), ir(20.), uv(20., .092)), {1: -1, 2: -1, 3: -1, 4: -1, 5: 1}, None),
    ('try: 4 IR + UV, different times', (ir(10.), ir(20.), ir(40.), ir(70.), uv(120., .092)),
     {1: -1, 2: 1, 3: 1, 4: -1, 5: 1}, None),
    ('try: 7-pulse UV', (uv(10., .072), uv(10., .072), uv(50., .144), uv(50., .144), uv(90., .092),
                         uv(90., .092), uv(90., .192)), {1: 1, 2: -1, 3: 1, 4: -1, 5: 1, 6: -1, 7: 1}, None),
    # the seminar notebook
    ('nb: EVV', (ir(50.), ir(100.), uv(120., .072)), {1: -1, 2: 1, 3: 1}, ((-1, 2),)),
    ('nb: EVV + 2nd UV', (ir(50.), ir(100.), uv(120., .072), uv(120., .092)), {1: -1, 2: 1, 3: 1, 4: 1},
     ((-1, 2),)),
    ('nb: 3 IR (3D)', (ir(50.), ir(100.), ir(120.)), {1: -1, 2: 1, 3: 1}, ((-1, 2),)),
    ('nb: CARS (cf_uv .072 .072 .144)', (uv(10., .072), uv(10., .072), uv(40., .144)), {1: 1, 2: -1, 3: 1},
     ((-1, 2),)),
    ('nb: CARS (cf_uv all .072)', (uv(10., .072), uv(10., .072), uv(40., .072)), {1: 1, 2: -1, 3: 1}, ((-1, 2),)),
    # 1D
    ('1D: IR absorption', (ir(50.),), {1: 1}, None),
    ('1D: IR + vis SFG', (ir(50.), uv(120., .072)), {1: 1, 2: 1}, None),
    # try.py, the sign sweeps
    *sweep('sweep: 3 IR + UV, all 3 IR at the same time', (IR_50, IR_50, IR_50, UV_7)),
    *sweep('sweep: 3 IR + UV, IR 1 and 2 at the same time', (IR_50, IR_50, IR_80, UV_7)),
    *sweep('sweep: 3 IR + UV, different times', (IR_50, ir(70.), IR_80, UV_7)),
    *sweep('sweep: 4 IR + UV, different times', (ir(10.), ir(30.), ir(50.), ir(90.), uv(120., .092)),
           {3: 1, 4: 1, 5: 1}),
    *sweep('sweep: 4 IR + UV, IR in 2 same-time pairs', (ir(10.), ir(10.), ir(20.), ir(20.), uv(120., .092)),
           {3: 1, 4: 1, 5: 1}),
    *sweep('sweep: 4 IR + UV, IR in 2 same-time pairs, UV with the 2nd',
           (ir(10.), ir(10.), ir(20.), ir(20.), uv(20., .092)), {3: 1, 4: -1, 5: 1}),
]


# ---- the translation to axes: derive's algorithm, any number of axes (E13) ----
def axes_dict(axis_set) -> dict[str, list[tuple]]:
    """{'A': [(-1,)], 'B': [(2,)]}: the pulse ids of each axis"""
    return {ax.label: [p.pulse_refs for p in ax.var_set.var_set] for ax in axis_set.axes}


def express(id_tuple: tuple, axes: dict) -> dict[str, int]:
    """a sum of signed pulse ids as a sum of signed axes, e.g. (-1, 2) with A = -w1, B = w2 -> {'A': 1, 'B': 1}"""
    labels = list(axes)
    for signs in product([-1, 0, 1], repeat=len(labels)):
        running = []
        for label, sign in zip(labels, signs):
            for pulses in (axes[label] if sign else ()):
                for k in pulses:
                    if sign == -1:
                        running.remove(k) if k in running else running.append(-k)
                    else:
                        running.remove(-k) if -k in running else running.append(k)
        if tuple(sorted(running)) == id_tuple:
            return {label: sign for label, sign in zip(labels, signs) if sign}
    raise AssertionError(f'{id_tuple} cannot be written in the axes {axes}')


def translate(terms, axes: dict) -> list:
    """copies of the terms with their conditions in axes (term_var_translate does the same up to 3 axes)"""
    translated = []
    for term in terms:
        term = copy.deepcopy(term)
        for rc in term.res:
            rc.pf = [label if sign == 1 else '-' + label
                     for label, sign in express(tuple(sorted(rc.pf)), axes).items()]
        translated.append(term)
    return translated


# ---- per term ----
def pattern(term) -> tuple[tuple[str, ...], str]:
    """(the conditions' axes, the kind), e.g. (('A', 'A+B'), 'point{AB}'); the term in axes"""
    def as_sum(pf):
        s = ''.join(('-' if a.startswith('-') else '+') + a.lstrip('-') for a in sorted(pf, key=lambda a: a.lstrip('-')))
        return s.lstrip('+')

    conditions = tuple(sorted(as_sum(rc.pf) for rc in term.res))
    if not term.res:
        return conditions, 'flat'
    lhs, axes = generate_LHS_motif(ResonanceMotif.from_conditions(term.res))
    rank = int(np.linalg.matrix_rank(lhs))
    free = len(axes) - rank
    if free == 0:
        kind = 'point{' + ''.join(axes) + '}' if len(term.res) == rank else 'over'
    else:
        kind = {1: 'line', 2: 'plane'}.get(free, f'{free}D')
    return conditions, kind


def max_quanta(term) -> int:
    """the most quanta in a state of the term (its denominators and conditions), as main_functions.find_max_state_lvl"""
    states = [s for vd in term.freqterms for s in (vd.sl, vd.sr)]
    states += [s for rc in term.res for s in (rc.diff.sl, rc.diff.sr)]
    return max((len(s.q) for s in states), default=0)


_available: dict[str, bool] = {}


def available(name: str) -> bool:
    """does the Gaussian reader give this property (on formaldehyde)?"""
    if name not in _available:
        try:
            MolSystemData.load(ref.data_origin('formaldehyde'), [name])
            _available[name] = True
        except ValueError:
            _available[name] = False
    return _available[name]


def counts(counter: Counter) -> str:
    return ', '.join(f'{k} {v}' for k, v in sorted(counter.items(), key=lambda kv: str(kv[0])))


# ---- one setup ----
def survey(name, pulses, wv_filter, magn_conditions, terms_file=None) -> None:
    print(f'\n{name}')
    print('  pulses: ' + ', '.join(f"{'UV' if p.cf_uv else 'IR'}(tc {p.tc:g}"
                                   + (f', cf_uv {p.cf_uv:g})' if p.cf_uv else ')') for p in pulses)
          + f'; filter {wv_filter}; magn_conditions {magn_conditions}')
    try:
        t0 = time.perf_counter()
        experiment = make_experiment(pulses, wv_filter, magn_conditions)
        terms = experiment.derive_terms()
        t_derive = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - a setup that derive cannot take is a result too
        print(f'  experiment or derive raises: {type(e).__name__}: {e}')
        return
    unique = list(dict.fromkeys(terms))
    print(f'  terms {len(terms)} (unique {len(unique)}), derive {t_derive:.1f} s')
    if not terms:
        return

    axis_set = experiment.canonical_axes
    if axis_set is None:
        choices = [c for group in experiment.valid_axis_combs for c in group.valid_axis_combs]
        if not choices:
            print('  no canonical axes and no valid axis choice')
            return
        axis_set = choices[0]
        print('  no canonical axes: the first valid choice')
    axes = axes_dict(axis_set)
    try:
        translate_terms_to_axis_variables(unique, axis_set)
        derive_translation = 'ok'
    except (ValueError, AssertionError) as e:
        derive_translation = f'raises: {e}'
    translated = translate(unique, axes)

    patterns = Counter(pattern(t) for t in translated)
    kinds = Counter()
    for (_, kind), n in patterns.items():
        kinds[kind] += n
    ranks = Counter(len(split_props(t)[0].get_cart_axes()) for t in unique)
    quanta = max(max_quanta(t) for t in terms)
    names = prop_names(terms)
    missing = [n for n in names if not available(n)]
    print(f"  axes {' '.join(f'{k} = {v}' for k, v in axes.items())}; derive's translation: {derive_translation}")
    print(f'  kinds: {counts(kinds)}; rank: {counts(ranks)}; quanta: {quanta}; props: {names}; missing: {missing}')
    for (conditions, kind), n in sorted(patterns.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {n:>6}  {kind:<12} ({' ; '.join(conditions) if conditions else 'no conditions'})")

    blocks = []
    if any(kind in ('line', 'plane', 'over') or kind.endswith('D') for kind in kinds):
        blocks.append('B7')
    if len(ranks) > 1:
        blocks.append('more than one rank')
    if missing:
        blocks.append('props')
    if quanta > MAX_QUANTA:
        blocks.append(f'{quanta}-quanta states')
    index_sets = sum(N_MODES ** len(set(t.tellNonSummSummIndices()[0]) | set(t.tellNonSummSummIndices()[1]))
                     for t in terms)
    if index_sets > MAX_INDEX_SETS:
        blocks.append(f'size ({index_sets:.0e} index sets)')
    if blocks:
        print(f"  evaluation: not run ({', '.join(blocks)})")
    else:
        try:
            t0 = time.perf_counter()
            molsys = MolSystemData.load(ref.data_origin('formaldehyde'), names)
            recipe = make_polarization_linear_comb(terms, experiment.polarization_avg_vector)
            result = compute_features(compile_terms(translate(terms, axes)), molsys, recipe, GAMMA_CM,
                                      states_choice='anharmonic')
            print(f'  evaluation: {len(result.table)} rows, {len(result.zero)} zero, {len(result.features)} '
                  f'features ({time.perf_counter() - t0:.1f} s)')
        except Exception as e:  # noqa: BLE001 - an evaluation that raises is a result too
            print(f'  evaluation raises: {type(e).__name__}: {str(e).splitlines()[0]}')

    if terms_file is not None:
        terms_file.write(f'\n{name}\n')
        for t in translated:
            terms_file.write(f'    {term_to_str(t)}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--only', nargs='+', help='only the setups whose name holds one of these')
    parser.add_argument('--terms', help='also write every unique term (in axes) to this file')
    args = parser.parse_args()
    setups = [s for s in SETUPS if not args.only or any(word in s[0] for word in args.only)]
    terms_file = open(args.terms, 'w') if args.terms else None
    try:
        for setup in setups:
            survey(*setup, terms_file=terms_file)
    finally:
        if terms_file is not None:
            terms_file.close()


if __name__ == '__main__':
    main()
