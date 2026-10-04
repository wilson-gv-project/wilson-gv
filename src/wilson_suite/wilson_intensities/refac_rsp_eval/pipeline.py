"""
Entry point: compiled terms + molecular data -> spectral features.

CompiledTerm list
  │ 1. index sets: fix the motif labels (make_idx_sets over term.idx_nonsumm)
  │ 2. coeff: evaluate_term_coeff_sumover, sums the other labels     ← MolSystemData, polarization
  │    zero coeff -> `zero`, no row
  │ 3. location: solve_LSE_motif, once per (motif, params), cm-1     ← molsys_data.states
  │    no single point -> `failed`
  v
ContributionRow(term_id, motif, params, location, coeff)   "why" layer    evaluate.build_contributions
  │ 4. group by (motif, location), add the coeffs
  v
SpectralFeature(location, rows)                            "what" layer   features.features_from_rows
  │ 5. peak width, spectral window, boxes
  v
grid.py → spectrum array                                   E2, not built yet
"""
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    PrecalculatedData,
    build_contributions,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    ContributionTable,
    SpectralFeature,
    features_from_rows,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import ParameterSet
from wilson_suite.wilson_system.system_data import MolSystemData

if TYPE_CHECKING:
    from wilson_suite.wilson_intensities.refac_rsp_eval.plan import CompiledTerm


@dataclass(frozen=True)
class FeatureResult:
    table: ContributionTable                          # all rows: the source
    zero: list[tuple[int, ParameterSet]]              # pairs with coeff == 0
    failed: list[tuple[int, ParameterSet, float]]     # pairs with no single resonance point
    lineshape_parameter: float | None = None          # cm-1

    @cached_property
    def features(self) -> list[SpectralFeature]:
        """One feature per (motif, location), built from the table on first use."""
        return features_from_rows(self.table, self.lineshape_parameter)


def compute_features(compiled: Sequence['CompiledTerm'], molsys_data: MolSystemData,
                     polarization_linear_comb: dict | None = None,
                     precalculated_data: PrecalculatedData | None = None,
                     lineshape_parameter: float | None = None) -> FeatureResult:
    """
    One SpectralFeature per (motif, location). lineshape_parameter in cm-1.
    row.term_id is the index in `compiled`, and so in the term list it was compiled from.
    """
    table, zero, failed = build_contributions(compiled, molsys_data, polarization_linear_comb, precalculated_data)
    return FeatureResult(table, zero, failed, lineshape_parameter)


 