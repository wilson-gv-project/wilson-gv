"""
Small hand-built objects shared by the unit tests of plan.py, evaluate.py, system_data.py, grid.py and features.py.
No VibPerturbedTerm, no data files.
"""

from wilson_suite.wilson_derive.abstractions import (
    HarmOscStateSymbolic,
    PolProp,
    QOperator,
    VibDiffTerm,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.features import (
    SpectralFeature,
    TermParametersChoice,
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


# --- spectral features (grid.py, features.py) ---

def tpc(term_ids: tuple = (0,), **params) -> TermParametersChoice:
    """One term group with a single parameter choice, e.g. tpc(a=0, b=1)."""
    return TermParametersChoice(res_motif=ResonanceMotif(()),
                                states_parameters=(ParameterSet(params),),
                                term_ids=term_ids)


def feat(location: dict[str, float] | None = None, /, gamma: float | None = 1.0, amp: float | None = 1.0,
         terms: tuple | None = None, **coords: float) -> SpectralFeature:
    """
    A feature at the given coordinates, e.g. feat(A=100.) or feat({'A': 100.}). gamma in cm-1; its box is location +- gamma.
    The dict form is for N-dimensional tests: feat(coords(3, 100.)).
    """
    return SpectralFeature(location=ResLocPoint({**(location or {}), **coords}),
                           term_contributions=(tpc(a=0),) if terms is None else terms,
                           lineshape_parameter=gamma,
                           amplitude_coeff=amp)


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
