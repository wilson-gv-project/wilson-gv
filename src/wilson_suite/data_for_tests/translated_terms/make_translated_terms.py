"""
Makes the stored term files in this folder: the terms translated to axes, for the tests of the new evaluator.
The tests and scripts of the new evaluator read these files through
wilson_intensities/refac_rsp_eval/tests/stored_terms.py.

Why the terms are stored: the new evaluator code must not import wilson_experiment, derive.derive or
term_var_translate (refac_rsp_eval/tests/test_imports.py checks the imports). This script imports all three. The
script sits outside the new code and saves their output to files.

Words used below:
    w1              the frequency of pulse 1 (pulse ids start at 1)
    setup           one experiment to derive terms for: its pulses, wv_filter and magn_conditions
    wv_filter       the sign of each pulse in the detected direction. {1: 1, 2: -1, 3: 1}: detect along k1 - k2 + k3.
    magn_conditions sums of pulse frequencies that are never close to 0. ((-1, 2),): -w1 + w2 is well above 0.
    term            one VibPerturbedTerm from derive. A term has resonance conditions (term.res). The pf of a
                    condition lists the pulse frequencies of the resonance: pf [-1, 2] means -w1 + w2.
    axes            the axes of the spectrum, labelled A, B, C, ... Each axis is a signed sum of pulse frequencies,
                    e.g. A = w1, B = -w1 + w2.
    translate       rewrite each pf from pulse ids to axis labels. With A = w1 and B = -w1 + w2:
                    pf [-1] becomes ['-A'], and pf [-1, 2] becomes ['B']. The new evaluator takes only translated terms.
    E13, G7         items in wilson_intensities/refac_rsp_eval/tests/scripts/todo.txt

What the script does for each G7 setup (make_g7_file):
    1. Build the experiment with wilson_experiment.
    2. Run derive on the experiment. derive gives the terms, with pf in pulse ids.
    3. Pick the axes: the experiment's canonical_axes. If the experiment has none: the first of its valid axis
       choices (valid_axis_combs).
    4. Translate the terms twice: with derive's term_var_translate, and with survey_translate below.
       derive's translation raises for 4 or more axes (E13). survey_translate has no such limit.
       Where both translations work, the script checks that both give the same terms.
    5. Check the polarizations (see the keys below), then write the file.

Files:
    evv.json                the 14 EVV terms. Not derived here: read from evv_in_pulse_ids.json (derive's terms,
                            stored earlier, with pf in pulse ids). Translated to A = w1, B = -w1 + w2.
                            laser_pol (1, 1, 1). These are the settings of the old EVV integration tests
                            (refac_rsp_eval/tests/make_evv_reference.py).
    g7/NN_<name>.json.gz    one file per G7 setup. NN is the setup's number in G7_SETUPS. G7 is a survey of other
                            experiments: refac_rsp_eval/tests/scripts/g7_survey.py evaluates the terms of these files.
    index.json              one entry per setup, in order: its file and its number of terms. A setup that the
                            experiment or derive cannot take has no file. Its entry holds the error instead.

The start of a G7 setup's name puts the setups in groups:
    try:            single experiments, e.g. 'try: CARS'
    nb:             the experiments of the seminar notebook ~/hyllseminar_12062026/hyllseminar_wilson_demo.ipynb
    1D:             1D spectra: IR absorption, IR + vis SFG
    sweep:          one set of pulses, once per sign pattern of wv_filter (see sweep below)

Keys of a G7 file:
    name                the setup's name, e.g. 'try: CARS'
    pulses, wv_filter, magn_conditions
                        the input of the experiment. Pulse ids are 1..N, in the order of the setup's pulses.
    axes                {label: [variables]}. A variable is a list of signed pulse ids. The axis is the sum of all
                        its variables. {'A': [[-2, 1]]}: A = w1 - w2. {'B': [[-1], [2]]}: B = -w1 + w2.
    axes_from           'canonical_axes' or 'first valid choice' (step 3)
    translation         the translation of the stored terms: 'derive', or 'survey' where derive's translation raised
    derive_translation  'ok', or the error that derive's translation raised
    polarizations       the polarization vectors in this order: detector, pulse 1, pulse 2, ...
                        The script checks this order against the experiment's all_polarizations.
    laser_pol           the experiment's polarization_avg_vector. The script checks that
                        get_pol_laser(polarizations) gives the same vector.
    source              the hash of the last commit that changed wilson_derive or wilson_experiment. The hash gets
                        ' + uncommitted changes' if either folder had changes not yet committed.
    terms               the translated terms, each as VibPerturbedTerm.to_dict
evv.json has name, axes, axes_from, translation, derive_translation, laser_pol, source and terms, plus
source_terms: the file the terms come from.
The .gz files hold no time stamp. The same terms give the same bytes, so git sees no change.

The stored files were made with wilson_derive and wilson_experiment as at commit
57776f0b7134c418361fa6a2e3640b1a7242e7b0 (2026-06-04, on main). Every stored file has this hash as its source.

Run the script again only on purpose: after derive or the experiment changed.
    python -m wilson_suite.data_for_tests.translated_terms.make_translated_terms                  every setup, ~10 min
    python -m wilson_suite.data_for_tests.translated_terms.make_translated_terms --only evv "3 IR (3D)"
--only runs only the setups whose name contains one of the given words. The other setups keep their files and
their entries in index.json. Case matters: 'evv' picks evv.json only. 'EVV' picks the G7 setups with EVV in the name.
After a run, check the git diff of index.json and evv.json. The .gz files are binary: git diff shows no lines of them.
"""
import argparse
import copy
import gzip
import json
import re
import subprocess
from dataclasses import replace
from itertools import product
from pathlib import Path

import wilson_suite.wilson_experiment.experiment_abstractions as wexp
from wilson_suite.wilson_derive import term_var_translate
from wilson_suite.wilson_derive.derive import get_fully_enhanced_terms
from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_intensities.amplitudes.averaging import get_pol_laser
from wilson_suite.wilson_utils.builders import make_SpectralAxisSet
from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_flat

HERE = Path(__file__).parent
INDEX_FILE = HERE / 'index.json'
SOURCE_PACKAGES = (HERE.parents[1] / 'wilson_derive', HERE.parents[1] / 'wilson_experiment')

EVV_SOURCE = HERE / 'evv_in_pulse_ids.json'
EVV_AXES = {'A': [1], 'B': [-1, 2]}     # the input form of make_SpectralAxisSet: A = w1, B = -w1 + w2
EVV_LASER_POL = (1., 1., 1.)            # pulses and detector along x, rank 4


# ---- pulses and experiments
def uv(tc: float, cf_uv: float):
    """A UV/vis pulse at time tc, with UV carrier frequency cf_uv. Polarized along x, travels along z.
    make_experiment sets the id."""
    return wexp.make_impulsive_gaussian_pulse(tc=tc, cf=0.0, cf_uv=cf_uv, maxstr=1e-5, wv=(0., 0., 1.),
                                              pol=(1., 0., 0.), id=1)


def ir(tc: float):
    """An IR pulse at time tc: as uv, but with UV carrier frequency 0."""
    return wexp.make_impulsive_gaussian_pulse(tc=tc, cf=0.0, cf_uv=0.0, maxstr=1e-5, wv=(0., 0., 1.),
                                              pol=(1., 0., 0.), id=1)


def make_experiment(pulses, wv_filter: dict, magn_conditions=None) -> wexp.VibExperiment:
    """
    The experiment of one setup. The pulses get ids 1..N in the given order. The detector is polarized along x.
    """
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
    """
    One setup per sign pattern of wv_filter.
    The pulses in `fixed` keep their sign. Every other pulse gets -1 and +1.
    Example: 3 pulses, fixed={3: 1} gives 4 setups: '<name> [-1 -2 +3]', '<name> [-1 +2 +3]', '<name> [+1 -2 +3]',
    '<name> [+1 +2 +3]'.
    """
    fixed = fixed or {}
    free = [i + 1 for i in range(len(pulses)) if i + 1 not in fixed]
    setups = []
    for signs in product([-1, 1], repeat=len(free)):
        wv_filter = {**fixed, **dict(zip(free, signs))}
        label = ' '.join(f"{'+' if wv_filter[k] > 0 else '-'}{k}" for k in sorted(wv_filter))
        setups.append((f'{name} [{label}]', pulses, wv_filter, None))
    return setups


IR_50, IR_80, UV_7 = ir(50.), ir(80.), uv(90., 0.192)
G7_SETUPS = [   # (name, pulses, wv_filter, magn_conditions)
    # single experiments
    ('try: CARS', (uv(10., .072), uv(10., .072), uv(120., .092)), {1: 1, 2: -1, 3: 1}, None),
    ('try: CAHRS (3 pulses, 5-entry filter)', (uv(10., .072), uv(10., .072), uv(120., .092)),
     {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, None),
    ('try: beta-alpha CARS', (uv(10., .072), uv(10., .072), uv(10., .144), uv(120., .072)),
     {1: 1, 2: 1, 3: -1, 4: 1}, None),
    ('try: EVV, IR 1 and 2 at the same time', (ir(100.), ir(100.), uv(120., .072)), {1: -1, 2: 1, 3: 1},
     ((-1, 2),)),
    ('try: CAHRS?? (5 UV)', (uv(10., .05), uv(10., .05), uv(10., .1), uv(110., .05), uv(110., .05)),
     {1: 1, 2: 1, 3: -1, 4: 1, 5: 1}, ((1, 2, -3),)),
    ('try: 3 IR + UV, IR 1 and 2 at the same time', (IR_50, IR_50, IR_80, UV_7), {1: -1, 2: -1, 3: 1, 4: 1}, None),
    ('try: 4 IR + UV, IR in 2 same-time pairs, UV with the 2nd',
     (ir(10.), ir(10.), ir(20.), ir(20.), uv(20., .092)), {1: -1, 2: -1, 3: -1, 4: -1, 5: 1}, None),
    ('try: 4 IR + UV, different times', (ir(10.), ir(20.), ir(40.), ir(70.), uv(120., .092)),
     {1: -1, 2: 1, 3: 1, 4: -1, 5: 1}, None),
    ('try: 7-pulse UV', (uv(10., .072), uv(10., .072), uv(50., .144), uv(50., .144), uv(90., .092),
                         uv(90., .092), uv(90., .192)), {1: 1, 2: -1, 3: 1, 4: -1, 5: 1, 6: -1, 7: 1}, None),
    # the seminar notebook (~/hyllseminar_12062026/hyllseminar_wilson_demo.ipynb)
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
    # sign sweeps: one set of pulses, once per sign pattern of wv_filter
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


# ---- derive and the translation to axes ----
def derive_terms(experiment: wexp.VibExperiment) -> list[VibPerturbedTerm]:
    """
    Runs derive on the experiment. Returns all terms in one list, with pf in pulse ids.
    derive itself returns the terms in groups: {number of anharmonicities: {anharmonicities: [terms]}}.
    """
    return derived_terms_flat(get_fully_enhanced_terms(experiment=experiment), tolistonly=True)


def derive_translate(terms, axis_set) -> list[VibPerturbedTerm]:
    """
    Translates a list of terms with derive's term_var_translate. Returns the translated copies.
    derive's function takes and returns groups {i: {j: [terms]}}. The list goes in as the one group {0: {0: terms}}.
    Raises ValueError for 4 or more axes (E13).
    """
    return term_var_translate.translate_terms_to_axis_variables({0: {0: list(terms)}}, axis_set)[0][0]


def axes_dict(axis_set) -> dict[str, list[tuple]]:
    """
    The axes of axis_set as plain data: {label: [variables]}. A variable is a tuple of signed pulse ids.
    Example: {'A': [(1,)], 'B': [(-1,), (2,)]} means A = w1, B = -w1 + w2.
    """
    return {ax.label: [p.pulse_refs for p in ax.var_set.var_set] for ax in axis_set.axes}


def express(id_tuple: tuple, axes: dict) -> dict[str, int]:
    """
    Writes a signed sum of pulse ids as a signed sum of axes. Returns {label: 1 or -1}.
    Example: axes A = w1, B = -w1 + w2. id_tuple (-1, 2) means -w1 + w2, so the result is {'B': 1}.
    The search tries each sign -1, 0, +1 for each axis, as derive's find_pulse_id_tuples_as_axis_vars does,
    but for any number of axes. id_tuple must be sorted. Raises AssertionError if no signs fit.
    """
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


def survey_translate(terms, axes: dict) -> list[VibPerturbedTerm]:
    """
    Translates the terms as derive's translation does, but for any number of axes (derive's stops at 3, E13).
    Returns copies. Each condition's pf goes from pulse ids to axis labels, e.g. [-1, 2] becomes ['B'].
    """
    translated = []
    for term in terms:
        term = copy.deepcopy(term)
        for rc in term.res:
            rc.pf = [label if sign == 1 else '-' + label
                     for label, sign in express(tuple(sorted(rc.pf)), axes).items()]
        translated.append(term)
    return translated


# ---- the files ----
def source() -> str:
    """
    The hash of the last commit that changed wilson_derive or wilson_experiment.
    The hash gets ' + uncommitted changes' if either folder has changes not yet committed.
    """
    paths = [str(p) for p in SOURCE_PACKAGES]
    git = lambda *args: subprocess.run(['git', *args, '--', *paths], cwd=HERE, capture_output=True, text=True,
                                       check=True).stdout.strip()
    return git('log', '-1', '--format=%H') + (' + uncommitted changes' if git('status', '--porcelain') else '')


def file_stem(number: int, name: str) -> str:
    """
    The file name of a G7 setup, without the suffix: the number, then the name in lower case with _ between
    the words. The signs + and - of a sweep become p and m.
    Example: 17, 'sweep: 3 IR + UV, all 3 IR at the same time [-1 +2 -3 +4]'
        -> '17_sweep_3_ir_uv_all_3_ir_at_the_same_time_m1_p2_m3_p4'
    """
    base, _, signs = name.partition(' [')
    stem = re.sub(r'[^a-z0-9]+', '_', base.lower()).strip('_')
    if signs:
        stem += '_' + '_'.join(s.replace('+', 'p').replace('-', 'm') for s in signs.rstrip(']').split())
    return f'{number:02d}_{stem}'


def as_json_data(content: dict) -> dict:
    """
    content as it comes back from a JSON file: tuples become lists, int keys become str.
    Example: {1: (0., 1.)} -> {'1': [0., 1.]}
    """
    return json.loads(json.dumps(content))


def write_file(path: Path, content: dict) -> None:
    """
    Writes content as JSON. A .gz path gets compact JSON, gzipped with time stamp 0.
    Any other path gets indented JSON.
    """
    if path.suffix == '.gz':
        path.write_bytes(gzip.compress(json.dumps(content, separators=(',', ':')).encode(), mtime=0))
    else:
        path.write_text(json.dumps(content, indent=2) + '\n')


def same_terms(a: list[VibPerturbedTerm], b: list[VibPerturbedTerm]) -> bool:
    """True if a and b hold the same terms in the same order. Compares the terms' to_dict."""
    return [t.to_dict() for t in a] == [t.to_dict() for t in b]


def make_evv_file(src: str) -> dict:
    """
    Writes evv.json: the terms of evv_in_pulse_ids.json, translated to EVV_AXES. src: the hash from source().
    Checks that derive's translation and survey_translate give the same terms. Returns the entry for index.json.
    """
    terms = VibPerturbedTerm.load_many_from_json(str(EVV_SOURCE))
    axis_set = make_SpectralAxisSet(EVV_AXES)  # type: ignore
    translated = derive_translate(terms, axis_set)
    assert same_terms(translated, survey_translate(terms, axes_dict(axis_set)))
    content = as_json_data({'name': 'evv', 'source_terms': EVV_SOURCE.name, 'axes': axes_dict(axis_set),
                            'axes_from': 'fixed: the old EVV integration tests (make_evv_reference.py)',
                            'translation': 'derive', 'derive_translation': 'ok',
                            'laser_pol': list(EVV_LASER_POL), 'source': src,
                            'terms': [t.to_dict() for t in translated]})
    write_file(HERE / 'evv.json', content)
    return {'name': 'evv', 'group': 'evv', 'file': 'evv.json', 'n_terms': len(terms), 'error': None}


def make_g7_file(number: int, name: str, pulses, wv_filter: dict, magn_conditions, src: str) -> dict:
    """
    Writes the file of one G7 setup: steps 1-5 of the module docstring. src: the hash from source().
    Returns the entry for index.json. No file is written if the experiment or derive raises, or if the experiment
    has no axes. The entry then holds the error.
    """
    entry = {'name': name, 'group': 'g7', 'file': None, 'n_terms': None, 'error': None}
    try:
        experiment = make_experiment(pulses, wv_filter, magn_conditions)
        terms = derive_terms(experiment)
    except Exception as e:  # noqa: BLE001 - a setup that the experiment or derive cannot take is a result too
        return {**entry, 'error': f'the experiment or derive raises {type(e).__name__}: {e}'}

    if experiment.canonical_axes is not None:
        axis_set, axes_from = experiment.canonical_axes, 'canonical_axes'
    else:
        choices = [c for group in experiment.valid_axis_combs for c in group.valid_axis_combs]
        if not choices:
            return {**entry, 'n_terms': len(terms), 'error': 'no canonical axes and no valid axis choice'}
        axis_set, axes_from = choices[0], 'first valid choice'
    axes = axes_dict(axis_set)

    by_survey = survey_translate(terms, axes)
    try:
        by_derive, derive_translation = derive_translate(terms, axis_set), 'ok'
    except (ValueError, AssertionError) as e:
        by_derive, derive_translation = None, f'raises: {e}'
    if by_derive is not None:
        assert same_terms(by_derive, by_survey), f"{name}: derive's and the survey's translation differ"

    polarizations = [experiment.detector.detection_polarization,
                     *(p.pol for p in sorted(experiment.field.pulses, key=lambda p: p.id))]
    assert [tuple(p) for p in experiment.all_polarizations] == [tuple(p) for p in polarizations], name
    assert list(experiment.polarization_avg_vector) == list(get_pol_laser(experiment.all_polarizations)), name

    file = f'g7/{file_stem(number, name)}.json.gz'
    content = as_json_data({
        'name': name,
        'pulses': [{'id': p.id, 'tc': p.tc, 'cf_uv': p.cf_uv, 'pol': p.pol, 'wv': p.wv}
                   for p in sorted(experiment.field.pulses, key=lambda p: p.id)],
        'wv_filter': wv_filter, 'magn_conditions': magn_conditions,
        'axes': axes, 'axes_from': axes_from,
        'translation': 'derive' if by_derive is not None else 'survey',
        'derive_translation': derive_translation,
        'polarizations': polarizations, 'laser_pol': [float(x) for x in experiment.polarization_avg_vector],
        'source': src,
        'terms': [t.to_dict() for t in (by_derive if by_derive is not None else by_survey)]})
    write_file(HERE / file, content)
    return {**entry, 'file': file, 'n_terms': len(terms)}


def main():
    """Writes the files of the chosen setups (all, or the ones --only picks), then index.json."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--only', nargs='+', help='run only the setups whose name contains one of these words')
    args = parser.parse_args()
    selected = lambda name: not args.only or any(word in name for word in args.only)

    entries = {e['name']: e for e in json.loads(INDEX_FILE.read_text())} if INDEX_FILE.exists() else {}
    (HERE / 'g7').mkdir(exist_ok=True)
    src = source()
    if selected('evv'):
        entries['evv'] = make_evv_file(src)
        print(f"evv: {entries['evv']['n_terms']} terms")
    for number, (name, *setup) in enumerate(G7_SETUPS, start=1):
        if selected(name):
            entries[name] = make_g7_file(number, name, *setup, src)
            print(f"{name}: {entries[name]['error'] or str(entries[name]['n_terms']) + ' terms'}")

    order = ['evv', *(s[0] for s in G7_SETUPS)]
    INDEX_FILE.write_text(json.dumps([entries[n] for n in order if n in entries], indent=2) + '\n')


if __name__ == '__main__':
    main()
