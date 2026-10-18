"""
G7: other experiments, from derive's terms through the new evaluator (todo.txt G7; also E8, E11, E12, E13).
The setups: ~/pre_paper3/try.py (its named experiments, explicit runs and sign sweeps), the seminar notebook
~/hyllseminar_12062026/hyllseminar_wilson_demo.ipynb, and two 1D setups (IR absorption, IR + vis SFG). Their
terms are stored, translated to axes (stored_terms.py; make_translated_terms.py in data_for_tests/translated_terms
builds the experiments, runs derive and translates): this script imports neither wilson_experiment nor
derive.derive.

Per setup:
    terms       derive's terms, all / unique
    axes        the canonical axes (VibExperiment.canonical_axes; else its first valid choice)
    patterns    per term, the axes of each condition in those axes, e.g. (A ; A+B) = one condition on A, one on
                A + B. Kinds:
                    point{AB}    a point on the axes A, B (B7 passes); the same along any other axis (E11)
                    line, plane  fewer independent conditions than the axes they use (B7 rejects them, E8)
                    flat         no conditions (E12)
    rank        the averaging rank of each term (pipeline.averaging_rank allows only one per run, 2 to 6)
    quanta      the most quanta in a state of any term (the Gaussian files have up to 3)
    missing     the props the Gaussian reader does not give (on formaldehyde)
    derive tr.  derive's own translation: ok, or where it stops (3 axes at most, E13); the 4-axis setups are
                stored in the translation without that stop (settings['translation'] 'survey')
    evaluation  compute_features on formaldehyde (anharmonic states, gamma 10 cm-1, the experiment's laser_pol),
                unless something blocks it: B7, more than one rank, props, more than 3 quanta, size

    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.g7_survey                  every setup
    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.g7_survey --only CARS "3 IR (nb)"
    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.g7_survey --terms terms.txt
--only: the setups whose name holds one of the words. --terms: also write every unique term (in axes, to_str) to
that file.

Result 2026-10-15: todo.txt G7 (the table of patterns, and what runs). 1D and 2D run end to end (CARS, EVV).
Pulses at different times give only points; pulses at the same time give lines and planes. Setups with more
than 2 axes mix axis sets (3 IR: {A, B} and {A, B, C}). 4 IR + UV: 4 axes, stopped by derive's translation.
"""
import argparse
import time
from collections import Counter

import numpy as np

from wilson_suite.wilson_intensities.refac_rsp_eval.compat import term_to_str
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
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.stored_terms import index, load_setup, setup_names
from wilson_suite.wilson_system.system_data import MolSystemData

N_MODES = 6             # formaldehyde
MAX_QUANTA = 3          # the most quanta in the Gaussian files' states
MAX_INDEX_SETS = 3e6    # no evaluation above this many index sets (sum over the terms of N_MODES ** labels)
GAMMA_CM = 10.


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
def survey(name, terms_file=None) -> None:
    print(f'\n{name}')
    if index()[name]['file'] is None:
        print(f"  {index()[name]['error']}")
        return
    t0 = time.perf_counter()
    setup = load_setup(name)
    terms = setup.terms
    t_load = time.perf_counter() - t0
    settings = setup.settings
    print('  pulses: ' + ', '.join(f"{'UV' if p['cf_uv'] else 'IR'}(tc {p['tc']:g}"
                                   + (f", cf_uv {p['cf_uv']:g})" if p['cf_uv'] else ')') for p in settings['pulses'])
          + f"; filter { {int(k): v for k, v in settings['wv_filter'].items()} }"
          + f"; magn_conditions {settings['magn_conditions']}")
    unique = list(dict.fromkeys(terms))
    print(f'  terms {len(terms)} (unique {len(unique)}), load {t_load:.1f} s')
    if not terms:
        return

    if settings['axes_from'] != 'canonical_axes':
        print('  no canonical axes: the first valid choice')
    axes = setup.axes

    patterns = Counter(pattern(t) for t in unique)
    kinds = Counter()
    for (_, kind), n in patterns.items():
        kinds[kind] += n
    ranks = Counter(len(split_props(t)[0].get_cart_axes()) for t in unique)
    quanta = max(max_quanta(t) for t in terms)
    names = prop_names(terms)
    missing = [n for n in names if not available(n)]
    print(f"  axes {' '.join(f'{k} = {v}' for k, v in axes.items())}; "
          f"derive's translation: {settings['derive_translation']}")
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
            recipe = make_polarization_linear_comb(terms, setup.laser_pol)
            result = compute_features(compile_terms(terms), molsys, recipe, GAMMA_CM, states_choice='anharmonic')
            print(f'  evaluation: {len(result.table)} rows, {len(result.zero)} zero, {len(result.features)} '
                  f'features ({time.perf_counter() - t0:.1f} s)')
        except Exception as e:  # noqa: BLE001 - an evaluation that raises is a result too
            print(f'  evaluation raises: {type(e).__name__}: {str(e).splitlines()[0]}')

    if terms_file is not None:
        terms_file.write(f'\n{name}\n')
        for t in unique:
            terms_file.write(f'    {term_to_str(t)}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--only', nargs='+', help='only the setups whose name holds one of these')
    parser.add_argument('--terms', help='also write every unique term (in axes) to this file')
    args = parser.parse_args()
    names = [n for n in setup_names('g7') if not args.only or any(word in n for word in args.only)]
    terms_file = open(args.terms, 'w') if args.terms else None
    try:
        for name in names:
            survey(name, terms_file=terms_file)
    finally:
        if terms_file is not None:
            terms_file.close()


if __name__ == '__main__':
    main()
