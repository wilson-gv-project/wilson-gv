"""
Small hand-built objects shared by the unit tests of plan.py, evaluate.py, system_data.py, grid.py, features.py
and pipeline.py. No VibPerturbedTerm, no data files.
Also the averaged part as a plain loop, the reference for its einsum (make_func_to_compute_avrg).
"""

from collections.abc import Callable
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
from wilson_suite.wilson_system.system_data import (
    MolPropsCollection,
    VibState,
    VibStatesData,
)


def polprop(ops: tuple[int, ...] = (), inds: str = '') -> PolProp:
    """ops -> QOperator labels, inds -> one-letter mode symbols (dord = len(inds))."""
    p = PolProp(ops=[QOperator(o=i) for i in ops], dord=len(inds))
    p.setInds(list(inds))
    return p


def vibdiff(sl: str = '', sr: str = '', pert: bool = False) -> VibDiffTerm:
    return VibDiffTerm(sl=HarmOscStateSymbolic(list(sl)), sr=HarmOscStateSymbolic(list(sr)), is_pert_wf_diff=pert)


def state(label: str, energy: float) -> VibState:
    return VibState(harm_quanta_coeffs={}, energy=energy, state_label=label)


# Two modes, cm-1. Harmonic frequencies (Hessian eigenvalues) E0_eigval, E1_eigval; the anharmonic states
# lie below them, as in real data: fundamentals about 5 % lower, 2-quanta states below the sum of their
# fundamentals ('0,1' by 100, '0,0' by 50). Every energy difference is distinct.
E0, E1, E01, E00 = 1000., 1500., 2400., 1950.
E0_eigval, E1_eigval = 1100., 1580.


def toy_states() -> VibStatesData:
    """The anharmonic states, the ones the tests use: states_choice='anharmonic'."""
    return VibStatesData(allstates=(state('0', E0), state('1', E1), state('0,1', E01), state('0,0', E00)),
                         harmonic_osc_states_labels=(0, 1))


def toy_harmonic_states() -> VibStatesData:
    """The harmonic states: fundamentals at the eigenvalues, 2-quanta states at the sums."""
    return VibStatesData(allstates=(state('0', E0_eigval), state('1', E1_eigval),
                                    state('0,1', E0_eigval + E1_eigval), state('0,0', 2 * E0_eigval)),
                         harmonic_osc_states_labels=(0, 1))


def toy_term() -> CompiledTerm:
    """
    0.5 * cff[a,b,c] * <polgrad>[a] * 1/(E_ab - E_a) * 1/omega_a, with one resonance condition,
    A = E_a - E_0. The condition fixes a; b and c are summed.
    Hand-built: a real term has one 1/omega per mode label.
    """
    return CompiledTerm(
        avrg_props=PropsCollection([polprop(ops=(0, 1), inds='a')]),
        non_avrg_props=PropsCollection([polprop(inds='abc')]),
        cmp_resmotf=cumulative_motif(('A',)),
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
# Real motifs are cumulative in pulse ids (condition k on pulses 1..k); the axes of a setup set
# their shape in axes. The condition of pulse 1 always has the ground state on one side.
# With toy_states() and a=0, b=1:  E_a = E0, E_b = E1, E_ab = E01, E_0 = 0.

# Real shapes:
#   MOTIF_AB  the EVV shape (axes A = w1, B = -w1 + w2): -A = E_0 - E_a, B = E_ab - E_a,
#             so A = E_a, B = E_ab - E_a
#   MOTIF_A   one condition: A = E_a - E_0
MOTIF_AB = ((((), ('a',)), ('-A',)), ((('a', 'b'), ('a',)), ('B',)))
MOTIF_A = (((('a',), ()), ('A',)),)

# Edge cases, not real shapes:
#   MOTIF_MIXED    A - B in one condition: a minus sign inside a sum of axes
#   MOTIF_B_TWICE  two conditions on one axis; compile_terms rejects it
#   MOTIF_AB_EXTRA MOTIF_AB plus A + B = E_ab - E_0, which follows from the other two: 3 conditions
#                  on 2 axes. compile_terms rejects it; solve_LSE_motif accepts it, since the
#                  conditions agree.
MOTIF_MIXED = ((((), ('a',)), ('B',)), (((), ('a',)), ('A', '-B')))
MOTIF_B_TWICE = ((((), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))
MOTIF_AB_EXTRA = MOTIF_AB + (((('a', 'b'), ()), ('A', 'B')),)


# --- toy molecule data and a term with resonance labels a, b (build_contributions, compute_features) ---
# The fixtures `states`, `molsys` and `molsys_a1_zero` (conftest.py) are built from these.

# The symmetry of real data: mode 0 is totally symmetric, mode 1 is not. A property entry is then
# nonzero only where mode 1 appears an even number of times.

# cff: symmetric in its mode indices; cff[0,0,1] and cff[1,1,1] are 0 by symmetry.
CFF = np.zeros((2, 2, 2))
CFF[0, 0, 0] = 3.
CFF[0, 1, 1] = CFF[1, 0, 1] = CFF[1, 1, 0] = 2.

# polgrad[mode, i, j]: symmetric in i, j. Mode 0 has a diagonal only, mode 1 one off-diagonal pair (yz).
# TOY_RECIPE is the real rank-2 recipe (laser_pol (1,)): xx, yy, zz, 1/3 each, so <polgrad>[a] is the
# trace / 3 = (0.6 / 3, 0) = (0.2, 0). Every index set with a = 1 has coeff 0, for the symmetry reason.
TOY_POLGRAD = np.zeros((2, 3, 3))
TOY_POLGRAD[0] = np.diag([0.1, 0.3, 0.2])
TOY_POLGRAD[1, 1, 2] = TOY_POLGRAD[1, 2, 1] = 0.08
TOY_RECIPE = {(0, 0): 1 / 3, (1, 1): 1 / 3, (2, 2): 1 / 3}

PS_00, PS_01 = ParameterSet({'a': 0, 'b': 0}), ParameterSet({'a': 0, 'b': 1})
PS_10, PS_11 = ParameterSet({'a': 1, 'b': 0}), ParameterSet({'a': 1, 'b': 1})


def ab_term(motif=MOTIF_AB) -> CompiledTerm:
    """
    toy_term() with `motif`. The labels in the motif are fixed by the index sets, the others of a, b, c
    are summed, as derive splits them: MOTIF_AB fixes a, b and sums c; MOTIF_A fixes a and sums b, c.
    """
    res_motif = ResonanceMotif.from_tuples(motif)
    fixed = tuple(sorted(res_motif.get_nm_indices()))
    return replace(toy_term(), cmp_resmotf=res_motif, idx_summ=tuple(i for i in 'abc' if i not in fixed),
                   idx_nonsumm=fixed)


def flat_term() -> CompiledTerm:
    """
    toy_term() without resonance conditions, as derive writes such a term: no label to fix, a, b and c
    all summed. A flat background: the same amplitude on every grid point.
    """
    return replace(toy_term(), cmp_resmotf=ResonanceMotif(()), idx_summ=('a', 'b', 'c'), idx_nonsumm=())


# --- spectral features (grid.py, features.py) ---

def row(coeff: float = 1., term_id: int = 0, motif: ResonanceMotif = ResonanceMotif(()),
        location: ResLocPoint | None = None, **params) -> ContributionRow:
    """One contribution, e.g. row(0.3, a=0, b=1). Location defaults to A=0."""
    return ContributionRow(term_id=term_id, motif=motif, params=ParameterSet(params),
                           location=ResLocPoint({'A': 0.}) if location is None else location, coeff=coeff)


def cumulative_motif(axes: tuple[str, ...]) -> ResonanceMotif:
    """
    The shape of the stored response terms: condition k on the sum of the first k axes, e.g.
    ('A', 'B', 'C') -> A = E_a - E_0, A + B = E_ab - E_0, A + B + C = E_abc - E_0.
    As many conditions as axes. No axes -> ResonanceMotif(()), a flat background.
    """
    labels = 'abcd'
    return ResonanceMotif.from_tuples([((tuple(labels[:k + 1]), ()), tuple(axes[:k + 1]))
                                       for k in range(len(axes))])


def feat(location: dict[str, float] | None = None, /, gamma: float | None = 1.0, amp: float | None = 1.0,
         rows: tuple | None = None, box_extent: float | None = None, **coords: float) -> SpectralFeature:
    """
    A feature at the given coordinates, e.g. feat(A=100.) or feat({'A': 100.}).
    gamma: the lineshape_param, cm-1 (the Lorentzian's half width).
    The dict form is for N-dimensional tests: feat(coords(3, 100.)).
    Without `rows`, the feature gets one row with coeff=amp, a=0 and the motif
    cumulative_motif(axes), so one lineshape_param per axis (amp=None -> no rows, so no amplitude).
    No box, as before SpectralWindow.from_features gives one. box_extent gives it the box location +- box_extent on
    every axis (box_extent: the half-width in from_features, from the location to each box face), e.g.
    feat(A=0., box_extent=1.) has the box (-1, 1).
    """
    loc = ResLocPoint({**(location or {}), **coords})
    if rows is None:
        rows = () if amp is None else (row(amp, location=loc, motif=cumulative_motif(loc.axes), a=0),)
    box = None if box_extent is None else Box({ax: (v - box_extent, v + box_extent) for ax, v in loc.coordinates})
    return SpectralFeature(location=loc, rows=rows, lineshape_param=gamma, feat_box=box)


def get_box_extent(f: SpectralFeature, axis: str = 'A') -> float:
    """From the feature's location to each face of its box on `axis` (the half-width in from_features)."""
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


# --- the averaged part, one index set at a time (evaluate.py) ---

def make_func_to_compute_avrg(*,
                              avrg_expression: PropsCollection,
                              polarization_linear_comb: dict
                              ) -> Callable[[dict, MolPropsCollection], float]:
    """
    The averaged part for one index set, as a loop over the recipe keys: the reference for
    evaluate.calculate_avrg_tensor_einsum, which gives every index set at once.
    Returns f(index_choices, props_data) -> the value, e.g. f({'a': 0, 'b': 1, 'c': 0}, props).
    """
    props_and_names = tuple(zip(avrg_expression, avrg_expression.names))     # names once, not per call

    def compute_for_idx_choice(index_choices: dict, props_data: MolPropsCollection) -> float:
        """The averaged part at index_choices, from the data in props_data."""

        if not isinstance(props_data, MolPropsCollection):
            raise TypeError(
                f"props_data must be a MolPropsCollection, got {type(props_data).__name__}"
            )

        # Validate index_choices has all required keys
        required_inds = {i for prop in avrg_expression for i in prop.inds} # type: ignore
        missing = required_inds - index_choices.keys()
        if missing:
            raise KeyError(
                f"index_choices is missing required mode indices: {missing}"
            )

        total = 0.

        for cart_axes in polarization_linear_comb:

            # Comment (MR): Noting that I considered if there would be any issues with this in generalized routine,
            # couldn't think of any but want to discuss and double check for safety

            product = 1.

            for prop, prop_tuple_key in props_and_names:

                nm_inds = tuple([index_choices[i] for i in prop.inds]) # type: ignore
                cart_inds = tuple([cart_axes[i.o] for i in prop.ops])
                all_inds = (*nm_inds, *cart_inds)

                # retrieve data for property (prop_key) and idxs_key which is (tuple(mode inds), tuple(cart inds))
                product *= props_data[prop_tuple_key].vals[all_inds]

            total += product * polarization_linear_comb[cart_axes]

        return total

    return compute_for_idx_choice
