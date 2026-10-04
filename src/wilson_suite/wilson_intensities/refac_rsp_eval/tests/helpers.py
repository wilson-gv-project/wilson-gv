"""
Small hand-built objects shared by the unit tests of plan.py, evaluate.py, system_data.py, grid.py, features.py
and pipeline.py. No VibPerturbedTerm, no data files.
"""

from dataclasses import replace

import numpy as np

from wilson_suite.wilson_derive.abstractions import (
    HarmOscStateSymbolic,
    PolProp,
    QOperator,
    VibDiffTerm,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    ContributionRow,
    SpectralFeature,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.grid import Box
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    FreqTermsCollection,
    ParameterSet,
    PropsCollection,
    ResLocPoint,
    ResonanceMotif,
)
from wilson_suite.wilson_system.system_data import VibState, VibStatesData


def polprop(ops: tuple[int, ...] = (), inds: str = '') -> PolProp:
    """ops -> QOperator labels, inds -> one-letter mode symbols (dord = len(inds))."""
    p = PolProp(ops=[QOperator(o=i) for i in ops], dord=len(inds))
    p.setInds(list(inds))
    return p


def vibdiff(sl: str = '', sr: str = '', pert: bool = False) -> VibDiffTerm:
    return VibDiffTerm(sl=HarmOscStateSymbolic(list(sl)), sr=HarmOscStateSymbolic(list(sr)), is_pert_wf_diff=pert)


def state(label: str, energy: float) -> VibState:
    return VibState(harm_quanta_coeffs={}, energy=energy, state_label=label)


# Two modes; energies in cm-1 chosen so every difference is distinct.
E0, E1, E01, E00 = 1000., 1500., 2600., 1950.
E0_eigval, E1_eigval = 1100., 1580.


def toy_states() -> VibStatesData:
    return VibStatesData(allstates=(state('0', E0), state('1', E1), state('0,1', E01), state('0,0', E00)),
                         harmonic_osc_states_labels=(0, 1))


def toy_term() -> CompiledTerm:
    """
    0.5 * cff[a,b,c] * <polgrad>[a] * 1/(E_ab - E_a) * 1/omega_a ; sum over b, c ; a fixed.
    """
    return CompiledTerm(
        avrg_props=PropsCollection([polprop(ops=(0, 1), inds='a')]),
        non_avrg_props=PropsCollection([polprop(inds='abc')]),
        cmp_resmotf=ResonanceMotif(()),
        cmp_freqdenom=FreqTermsCollection([vibdiff(sl='a'), vibdiff(sl='ab', sr='a', pert=True)]),
        frac_factor=0.5,
        idx_summ=('b', 'c'),
        idx_nonsumm=('a',),
    )


# --- resonance motifs (evaluate.py, pipeline.py) ---
# A motif is a set of resonance conditions  E_left - E_right = sum_j s_j * w_j , one per
# ResCondKey: `diff` names the two states by quanta labels, `pf` the frequency axes w_j with
# sign s_j ('-B' -> s = -1). Fixing the mode labels (a, b, ...) to modes turns the motif into
# a linear system  LHS @ w = RHS  whose solution is where on the axes the term resonates.
#
# The four motifs below are the ones used for the pre-refactor amplitudes/resonances.py.
# With toy_states() and a=0, b=1:  E_a = E0, E_b = E1, E_ab = E01, E_0 = 0.

MOTIF_AB = (((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))        # one axis per condition
MOTIF_A = (((('a', 'b'), ('a',)), ('A',)),)                                    # single condition
MOTIF_MIXED = ((((), ('a',)), ('B',)), (((), ('a',)), ('A', '-B')))             # A - B in one condition
MOTIF_B_TWICE = ((((), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))           # two conditions, one axis


# --- toy molecule data and a term with resonance labels a, b (build_contributions, compute_features) ---
# The fixtures `states`, `molsys` and `pre_a1_zero` (conftest.py) are built from these.

CFF = np.arange(8, dtype=float).reshape(2, 2, 2) + 1.   # cff[a, b, c] = 1 + 4a + 2b + c

PS_00, PS_01 = ParameterSet({'a': 0, 'b': 0}), ParameterSet({'a': 0, 'b': 1})
PS_10, PS_11 = ParameterSet({'a': 1, 'b': 0}), ParameterSet({'a': 1, 'b': 1})


def ab_term(motif=MOTIF_AB) -> CompiledTerm:
    """toy_term() with resonance labels a, b (fixed by the index sets) and summation label c."""
    return replace(toy_term(), cmp_resmotf=ResonanceMotif.from_tuples(motif), idx_summ=('c',), idx_nonsumm=('a', 'b'))


# --- spectral features (grid.py, features.py) ---

def row(coeff: float = 1., term_id: int = 0, motif: ResonanceMotif = ResonanceMotif(()),
        location: ResLocPoint | None = None, **params) -> ContributionRow:
    """One contribution, e.g. row(0.3, a=0, b=1). Location defaults to A=0."""
    return ContributionRow(term_id=term_id, motif=motif, params=ParameterSet(params),
                           location=ResLocPoint({'A': 0.}) if location is None else location, coeff=coeff)


def feat(location: dict[str, float] | None = None, /, gamma: float | None = 1.0, amp: float | None = 1.0,
         rows: tuple | None = None, **coords: float) -> SpectralFeature:
    """
    A feature at the given coordinates, e.g. feat(A=100.) or feat({'A': 100.}). gamma in cm-1; its box is location +- gamma.
    The dict form is for N-dimensional tests: feat(coords(3, 100.)).
    Without `rows`, the feature gets one row with coeff=amp and a=0 (amp=None -> no rows, so no amplitude).
    """
    loc = ResLocPoint({**(location or {}), **coords})
    if rows is None:
        rows = () if amp is None else (row(amp, location=loc, a=0),)
    return SpectralFeature(location=loc, rows=rows, lineshape_parameter=gamma)


def box_halfwidth(f: SpectralFeature, axis: str = 'A') -> float:
    mn, mx = f.feat_box.bounds[axis] # type: ignore
    return (mx - mn) / 2


# --- N-dimensional locations: the same test runs for every spectrum dimensionality in NDIMS ---

NDIMS = (1, 2, 3, 4)
AXES = 'ABCD'


def coords(ndim: int, value: float = 0., **override: float) -> dict[str, float]:
    """value on the first ndim axes; override sets single axes, e.g. coords(3, 5., C=11.) -> {'A': 5., 'B': 5., 'C': 11.}."""
    return {ax: override.get(ax, value) for ax in AXES[:ndim]}


def cube(ndim: int, mn: float, mx: float, **override: tuple[float, float]) -> Box:
    """(mn, mx) on the first ndim axes; override sets single axes, e.g. cube(2, 0., 1., B=(5., 6.))."""
    return Box({ax: override.get(ax, (mn, mx)) for ax in AXES[:ndim]})
