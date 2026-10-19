"""
Writes evv_reference.json: the rows of the new pipeline (build_contributions) for the 14 EVV terms
(the setup 'evv' of stored_terms.py: data_for_tests/translated_terms/evv.json, already translated to AXES),
for formaldehyde, water and CO, with harmonic and with anharmonic states.

Settings are the ones of the old EVV integration tests (tests/integration/test_integration_evaluation.py; deleted
with the old code, see git show d93cc7b:src/wilson_suite/wilson_intensities/tests/integration/
test_integration_evaluation.py): axes A = [1], B = [-1, 2]; laser polarization (1, 1, 1), rank 4. The data files are GVPT2 runs,
so the old pipeline uses anharmonic states for them; harmonic is stored too.

Run it again only on purpose (a deliberate change of numbers), then check the diff:
    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.make_evv_reference
"""

import json
from pathlib import Path

from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import build_contributions
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import (
    make_polarization_linear_comb,
    prop_names,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    ParameterSet,
    compile_terms,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.stored_terms import load_setup
from wilson_suite.wilson_system.system_data import DataOriginInfo, MolSystemData
from wilson_suite.wilson_utils.paths import SUITE_ROOT

REFERENCE_FILE = Path(__file__).parent / 'evv_reference.json'

TERMS_FILE = 'data_for_tests/translated_terms/evv.json'   # relative to SUITE_ROOT: the setup 'evv'
AXES = {'A': [1], 'B': [-1, 2]}     # the axes of evv.json, A = w1, B = -w1 + w2 (test_evv_reference checks both agree)
LASER_POL = (1., 1., 1.)
RANK = 4     # recorded in the file; the recipe takes the rank from the terms (test_pipeline checks both agree)
DATA_FILES = {'formaldehyde': 'data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out',
              'water': 'data_for_tests/g16_h2o_HF_STO3G.out',
              'co': 'data_for_tests/g16_co_HF_STO3G.out'}
STATES_CHOICES = ('anharmonic', 'harmonic')


def evv_terms() -> list[VibPerturbedTerm]:
    """the 14 terms, translated to AXES; new objects per call"""
    return load_setup('evv').terms


def compiled_evv_terms() -> list[CompiledTerm]:
    return compile_terms(terms=evv_terms())


def data_origin(molecule: str) -> DataOriginInfo:
    return DataOriginInfo(source_type='gaussian', base_file_loc=f'{SUITE_ROOT}/{DATA_FILES[molecule]}')


def _params(ps: ParameterSet) -> dict[str, int]:
    """the term's labels and their modes, as plain json data"""
    return ps.to_dict()


def compute_block(compiled: list[CompiledTerm], molecule: str, states_choice: str) -> dict:
    """What the new pipeline gives now for one molecule and one states choice, as plain json data."""
    terms = evv_terms()
    molsys = MolSystemData.load(data_origin(molecule), prop_names(terms))
    pol = make_polarization_linear_comb(terms, LASER_POL)
    table, zero = build_contributions(compiled, molsys, pol, states_choice=states_choice)
    rows = list(table)
    return {
        'molecule': molecule,
        'data_file': DATA_FILES[molecule],
        'states': states_choice,
        'n_modes': len(molsys.eigenvals),
        'counts': {'rows': len(rows), 'zero': len(zero)},
        'rows': [{'term': r.term_id, 'params': _params(r.params), 'coeff': r.coeff,
                  'location': r.location.as_dict()} for r in rows],
        'zero': [{'term': t, 'params': _params(ps)} for t, ps in zero],
    }


def _to_json_text(header: dict, blocks: list[dict]) -> str:
    """json with one row (zero entry) per line: short diffs, easy to read"""
    def json_list(name: str, items: list, last: bool) -> list[str]:
        lines = [f'     "{name}": [']
        lines += ['      ' + json.dumps(x) + (',' if i < len(items) - 1 else '') for i, x in enumerate(items)]
        return lines + ['     ]' + ('' if last else ',')]

    lines = ['{']
    lines += [f'  {json.dumps(k)}: {json.dumps(v)},' for k, v in header.items()]
    lines.append('  "results": [')
    for i, b in enumerate(blocks):
        info = {k: v for k, v in b.items() if k not in ('rows', 'zero')}
        lines.append('    {' + json.dumps(info)[1:-1] + ',')
        lines += json_list('rows', b['rows'], last=False)
        lines += json_list('zero', b['zero'], last=True)
        lines.append('    }' + (',' if i < len(blocks) - 1 else ''))
    lines += ['  ]', '}', '']
    return '\n'.join(lines)


def main():
    compiled = compiled_evv_terms()
    header = {
        'about': ('Rows of the new pipeline (build_contributions) for the 14 EVV terms. '
                  'Written by make_evv_reference.py. coeff in au, location in cm-1. '
                  'params: the fixed resonance labels; the other labels are summed.'),
        'settings': {'terms_file': TERMS_FILE, 'axes': AXES, 'laser_pol': list(LASER_POL), 'rank': RANK},
    }
    blocks = [compute_block(compiled, m, sc) for m in DATA_FILES for sc in STATES_CHOICES]
    text = _to_json_text(header, blocks)
    json.loads(text)  # the hand-written layout must stay valid json
    REFERENCE_FILE.write_text(text)
    for b in blocks:
        print(f"{b['molecule']:13s} {b['states']:11s} {b['n_modes']} modes  {b['counts']}")


if __name__ == '__main__':
    main()
