"""
C2, factor by factor: for every full label set (fixed a, b and summed c), old evaluate_single_index_dict
vs new evaluate_full_index_dict, each of NON_AVRG, AVRG, VIBDIFF_TERMS, VIBENE_DENOM.
Same variations as c2_variations.py, without the two mismatch controls.

Old leaves come from bound.coefficients[term][fixed] = (total, dict_of_sum):
    no summed label:  dict_of_sum = {fixed: factors}
    one summed label: dict_of_sum = {fixed: {full: factors}}
New leaves come from evaluate_term_coeff_sumover(...)[1] = {full: factors}.

A zero factor stops both evaluations early, at different factors (old: NON_AVRG first, new: AVRG first),
so for such leaves only "both zero" and the factors present on both sides are compared.

Result 2026-10-07: formaldehyde 2664 label sets, every factor equal in every variation (todo.txt C2).
Control (old GVPT2 vs new harmonic states): 1152 label sets differ, all in VIBDIFF_TERMS only.

Needs the old code. Delete together with it (todo.txt, order of work step 10).
    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.c2_leaves
"""
from collections import Counter

import numpy as np

from wilson_suite.wilson_intensities.amplitudes.averaging import getGeneralPolarizationAveragingExpression
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import evaluate_term_coeff_sumover
from wilson_suite.wilson_intensities.refac_rsp_eval.tests import c2_variations as c2

FACTORS = ('NON_AVRG', 'AVRG', 'VIBDIFF_TERMS', 'VIBENE_DENOM')


def old_leaves(dict_of_sum: dict) -> dict:
    """{full key: factors}, for both shapes of dict_of_sum"""
    (fixed, inner), = dict_of_sum.items()
    if set(inner) <= set(FACTORS):          # no summed label: the factors themselves
        return {c2.key(fixed): inner}
    return {c2.key(full): factors for full, factors in inner.items()}


def is_zero(factors: dict) -> bool:
    return any(factors.get(f, 1.) == 0. for f in FACTORS)


def compare_leaves(v: c2.Variation) -> dict:
    sim, bound = c2.run_old(v)
    laser_pol = sim.exp.polarization_avg_vector
    compiled, molsys, *_ = c2.run_new(v, laser_pol)
    iso = getGeneralPolarizationAveragingExpression(rank=4, laser_pol=tuple(laser_pol))

    out = dict(leaves=0, only_one_side=0, both_zero=0, zero_mismatch=0, all_four=0,
               max_rel=Counter(), n_diff=Counter(), examples=[])
    for old_t, new_t in zip(bound.axes.terms, compiled):
        for fixed, (_, dict_of_sum) in bound.coefficients[old_t].items():
            old = old_leaves(dict_of_sum)
            _, new_raw = evaluate_term_coeff_sumover(new_t, dict(c2.key(fixed)), molsys,
                                                     polarization_linear_comb=iso)
            new = {c2.key(k): f for k, f in new_raw.items()}

            out['leaves'] += len(old)
            out['only_one_side'] += len(set(old) ^ set(new))
            for k in set(old) & set(new):
                o, n = old[k], new[k]
                if is_zero(o) != is_zero(n):
                    out['zero_mismatch'] += 1
                    out['examples'].append((k, o, n))
                    continue
                if is_zero(o):
                    out['both_zero'] += 1
                elif set(o) == set(n) == set(FACTORS):
                    out['all_four'] += 1
                for f in set(o) & set(n):
                    a, b = float(o[f]), float(n[f])
                    if a == b == 0.:
                        continue
                    rel = abs(a - b) / max(abs(a), abs(b))
                    out['max_rel'][f] = max(out['max_rel'][f], rel)
                    if not np.isclose(a, b, rtol=c2.RTOL, atol=0.):
                        out['n_diff'][f] += 1
                        if len(out['examples']) < 5:
                            out['examples'].append((k, f, a, b))
    return out


def main():
    print(f'{"variation":<32} {"leaves":>6} {"one side":>8} {"all four":>8} {"both 0":>6} {"0 vs !0":>7} | '
          + ' '.join(f'{f:>13}' for f in FACTORS) + ' | n diff')
    for v in c2.VARIATIONS:
        if 'mismatch' in v.name:
            continue
        r = compare_leaves(v)
        print(f'{v.name:<32} {r["leaves"]:>6} {r["only_one_side"]:>8} {r["all_four"]:>8} {r["both_zero"]:>6} '
              f'{r["zero_mismatch"]:>7} | ' + ' '.join(f'{r["max_rel"][f]:>13.1e}' for f in FACTORS)
              + f' | {sum(r["n_diff"].values())}')
        for ex in r['examples']:
            print('      ', ex)


if __name__ == '__main__':
    main()
