"""
The terms that the tests and scripts evaluate, read from files: already translated to axes, with their settings.
The files are in data_for_tests/translated_terms; make_translated_terms.py there writes them (from the experiment
and derive.derive; no file of the evaluator imports those: test_imports.py).
    setup = load_setup('try: CARS')
    setup.terms          VibPerturbedTerm list, each condition's pf in axes, e.g. ['A', '-B']; new objects per call
    setup.axes           {label: [pulse ids of each variable]}, e.g. {'A': [(-2, 1)]}: A = w1 - w2
    setup.laser_pol      the experiment's polarization_avg_vector, e.g. (1., 1., 1.)
    setup.settings       everything else in the file (make_translated_terms.py lists it), e.g. settings['pulses']
    setup_names('g7')    the setups of g7_survey.py, in its order; 'evv': ['evv']
A setup that the experiment or derive could not take has no file: load_setup raises, and index()[name]['error']
holds the error.
"""
import gzip
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_utils.paths import SUITE_ROOT

TERMS_DIR = Path(SUITE_ROOT) / 'data_for_tests' / 'translated_terms'


@dataclass(frozen=True)
class StoredSetup:
    name: str
    terms: list[VibPerturbedTerm]
    settings: dict

    @property
    def axes(self) -> dict[str, list[tuple[int, ...]]]:
        return {label: [tuple(pulse_ids) for pulse_ids in variables]
                for label, variables in self.settings['axes'].items()}

    @property
    def laser_pol(self) -> tuple[float, ...]:
        return tuple(self.settings['laser_pol'])


@cache
def index() -> dict[str, dict]:
    """{setup name: its entry in index.json}: file, n_terms, error, group ('evv' or 'g7')"""
    return {entry['name']: entry for entry in json.loads((TERMS_DIR / 'index.json').read_text())}


def setup_names(group: str) -> list[str]:
    return [name for name, entry in index().items() if entry['group'] == group]


def load_setup(name: str) -> StoredSetup:
    if name not in index():
        raise KeyError(f'no setup {name!r} in {TERMS_DIR / "index.json"}; setups: {list(index())}')
    entry = index()[name]
    if entry['file'] is None:
        raise ValueError(f'{name}: no terms file: {entry["error"]}')
    raw = (TERMS_DIR / entry['file']).read_bytes()
    settings = json.loads(gzip.decompress(raw) if entry['file'].endswith('.gz') else raw)
    terms = [VibPerturbedTerm.from_dict(t) for t in settings.pop('terms')]
    return StoredSetup(name, terms, settings)
