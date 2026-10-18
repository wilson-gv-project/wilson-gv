"""
The evaluator takes terms already translated to axes; the tests and scripts load theirs from files (stored_terms.py).
No file of refac_rsp_eval or wilson_system (tests and scripts too) imports a module of FORBIDDEN, not even for
type hints or inside a function. make_translated_terms.py (data_for_tests/translated_terms) does, outside the
new code: it writes the term files.
Used from wilson_derive: the classes a term is made of (response_terms.VibPerturbedTerm, abstractions); the test
forbids only the modules of FORBIDDEN.
Import lines only: at run time wilson_suite/__init__.py loads wilson_experiment and wilson_derive.derive anyway.
"""
import ast
from pathlib import Path

from wilson_suite.wilson_utils.paths import SUITE_ROOT

NEW_CODE = ('wilson_intensities/refac_rsp_eval', 'wilson_system')     # relative to SUITE_ROOT
FORBIDDEN = (
    'wilson_suite.wilson_experiment',                   # the experiment, every module
    'wilson_suite.wilson_derive.derive',                # get_fully_enhanced_terms
    'wilson_suite.wilson_derive.term_var_translate',    # the translation to axes
    'wilson_suite.wilson_utils.builders',               # make_SpectralAxisSet; imports wilson_experiment
    'wilson_suite.fixtures',                            # builds VibExperiments
)


def imported_modules(source: str, package: str = '') -> list[tuple[int, str]]:
    """
    (line, module) for every import in the source, at any depth (functions, if TYPE_CHECKING). `from a import b`
    gives a and a.b: b may be a module, e.g. from wilson_suite.wilson_derive import term_var_translate.
    package: the file's package, for relative imports, e.g. 'wilson_suite.wilson_system'
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            parts = package.split('.')[:len(package.split('.')) - node.level + 1] if node.level else []
            base = '.'.join([*parts, *([node.module] if node.module else [])])
            found += [(node.lineno, base), *((node.lineno, f'{base}.{alias.name}') for alias in node.names)]
    return found


def forbidden(module: str) -> bool:
    return any(module == f or module.startswith(f + '.') for f in FORBIDDEN)


def new_code_files() -> list[Path]:
    return sorted(p for folder in NEW_CODE for p in (Path(SUITE_ROOT) / folder).rglob('*.py'))


def package_of(path: Path) -> str:
    return '.'.join(['wilson_suite', *path.parent.relative_to(SUITE_ROOT).parts])


def test_new_code_imports_neither_the_experiment_nor_derive():
    files = new_code_files()
    found = [f'{p.relative_to(SUITE_ROOT)}:{line} {module}'
             for p in files for line, module in imported_modules(p.read_text(), package_of(p)) if forbidden(module)]

    assert len(files) > 30      # both folders found
    assert found == []


def test_the_check_finds_each_kind_of_import():
    """at the top, as a module from its package, for type hints only, inside a function, relative"""
    source = '\n'.join([
        'import wilson_suite.wilson_experiment.experiment_abstractions as wexp',
        'from wilson_suite.wilson_derive import term_var_translate',
        'if TYPE_CHECKING:',
        '    from wilson_suite.wilson_experiment.indep_vars_and_axes import SpectralAxisSet',
        'def f():',
        '    from wilson_suite.wilson_derive.derive import get_fully_enhanced_terms',
        'from ...wilson_utils.builders import make_SpectralAxisSet',
        'from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm',   # allowed
    ])
    lines = sorted({line for line, module in imported_modules(source, 'wilson_suite.wilson_intensities.x')
                    if forbidden(module)})

    assert lines == [1, 2, 4, 6, 7]
