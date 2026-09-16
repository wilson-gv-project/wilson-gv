"""
Submodules are imported lazily (PEP 562).

Eagerly importing them here created an import cycle: modules in
wilson_intensities.amplitudes import wilson_main.abstractions, which ran this
__init__ and pulled in workflow_abstractions / spectrum_abstractions, which
import back into a half-built wilson_intensities.amplitudes.
"""

import importlib
from typing import TYPE_CHECKING

_SUBMODULES = ("abstractions", "workflow_abstractions", "main_functions", "spectrum_abstractions")

if TYPE_CHECKING:
	from . import abstractions
	from . import workflow_abstractions
	from . import main_functions
	from . import spectrum_abstractions


def __getattr__(name: str):
	if name in _SUBMODULES:
		return importlib.import_module(f".{name}", __name__)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
	return sorted(list(globals()) + list(_SUBMODULES))
