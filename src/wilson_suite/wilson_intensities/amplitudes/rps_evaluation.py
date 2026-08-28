from dataclasses import dataclass, field
import numpy as np

from wilson_suite.wilson_experiment.indep_vars_and_axes import SpectralAxisSet
from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_main.abstractions import DataOriginInfo, MolecularSystem
from wilson_suite.wilson_utils.unit_convertor import convNu2Ene


# -w1 + w2 is always significantly > 0 ==> magn_conditions=((-1, 2),)
MagnConditions = tuple[tuple[int]]

@dataclass(frozen=True)
class TermsInAxes:
    """Everything in axis variables. Ready to evaluate."""
    axis_choice: SpectralAxisSet
    terms: list[VibPerturbedTerm]
    magn_conds: MagnConditions | None

    def need_what(self) -> set[str]: ...

@dataclass(frozen=True)
class MolSystemData:
    """Everything obtained externally. No configuration."""
    name: str
    natoms: int = None
    geo: Any = None
    geo_extra: Any = None
    linear: bool = False
    conformer: str = 'conf1'

    mol_props: MolPropsCollection
    states: tuple
    eigenvals: np.ndarray
    eigenvecs: np.ndarray



@dataclass(frozen=True)
class RspFunEvalSetup:
    terms: TermsInAxes # attributes: magn_conds, axis_choices(all possible), axes
    # methods: input_vars -- what data is needed for evaluation (states vib ene, max_state_lvs, axis_choice, gamma, pulse_polarization_vector)
    # this defines experiment represented
    # here should be with axis_choice
    # terms post axis selection

    # how to eval
    gamma_cm1: float
    polarization: np.ndarray
    grid: SpectralGrid
    # vib_ana: VibAnaConfig
    # calc_config: DataOriginInfo
    mol_system: MolSystemData
    mask_forbidden_region: bool = False

    @property
    def gamma_au(self) -> float:
        return convNu2Ene(self.gamma_cm1)
    
    @property
    def axis_choice(self):
        return self.terms.axis_choice

    @property
    def magn_conds(self):
        return self.terms.magn_conds

    def __post_init__(self):
        if self.mask_forbidden_region and self.terms.magn_conds is None:
            raise ValueError("mask_forbidden_region set but terms carry no magn_conds")

    def make_datarequest(self) -> dict[str, DataOriginInfo]:
        return {name: self.calc_config for name in self.terms.need_what()}


def compute_features(setup: RspFunEvalSetup, data: MolSystemData) -> list[SpectralFeature]:
    terms = _prep_terms(setup.terms)
    vib_data, configs = _prep_data(data, setup.polarization)
    vibdiffs = _precompute_vibdiffs(terms, vib_data)
    motif_locs, terms_for_motifs = _process_resonances(terms, vib_data, vibdiffs)
    precalc = _precalculate(terms, configs)
    coeffs = _term_coefficients(terms, motif_locs, configs, precalc)
    features, _zero = _get_features(motif_locs, terms_for_motifs, coeffs, setup.gamma_cm1)
    return features


def render_grid(features, setup: RspFunEvalSetup, executor=None) -> EvaluatedResult:
    features = _dress_with_boxes(features, setup.boxes)
    if setup.mask_forbidden_region:
        features = _filter_magn_conds(features, setup.magn_conds, setup.magn_conds_margin)
    window = _place_in_window(features, setup.grid.window)
    gm = _make_grid_manager(window, setup.grid.resolution)
    regions = _make_regions(gm)
    results = evaluate_regions(regions, setup.gamma_au, executor)
    gm.place_results_into_grid(results)
    return EvaluatedResult(spec=gm.full_grid, axes=gm.axes, setup=setup)


#--------------------------
def _prep_terms(terms: dict | list) -> list:
    """
    put data in a form for use on the evaluation step
    """
    if isinstance(terms, type([])):
        for t in terms:
            if not isinstance(t, VibPerturbedTerm):
                raise ValueError("Smth that is not a VibPerturbedTerm was given in a list to prepTermsForEval()")
        return terms

    if isinstance(terms, type({})):
        for t_key in terms:
            if isinstance(terms[t_key], type({})):

                terms_as_list = []
                for i in terms:
                    for j in terms[i]:
                        for t in terms[i][j]:
                            terms_as_list.append(t)
                return terms_as_list
            else:
                if not isinstance(terms[t_key], VibPerturbedTerm):
                    raise ValueError("A flat dictionary but has smth other than VibPerturbedTerm as a value")
                return list(terms.values())

def _prep_data(eval_data: MolSystemData, rsp_eval_setup: RspFunEvalSetup, include_states_list):
    vibdiff_cache = VibDiffCache()

    number_of_nmodes = len(eval_data.eigenvals)
    vib_data = VibStatesData(allstates=tuple(eval_data.states), 
                             harmonic_osc_states_labels=include_states_list,
                             number_of_nmodes=number_of_nmodes)
    
    data_configs = EvaluationDataAndConfigs(props_data=eval_data.mol_props,
                                            vibstates_data=vib_data,
                                            number_of_nmodes=number_of_nmodes,
                                            nm_inds_choices=include_states_list,
                                            pulse_polarization_vector=rsp_eval_setup.polarization,
                                            nc_sqrt_eigval=eval_data.eigenvals)

    return vibdiff_cache, vib_data, data_configs

#--------------------------
def _dress_with_boxes(features, dyn_range):
    """
    box_range_safety_margin, scale_wrt_max_intensity, minimum_box_padding

    """

    max_intensity_in_window = SpectralFeature.get_max_intensity_feat(features).get_intensity()
    min_intensity_in_window = max_intensity_in_window / dyn_range

    features = SpectralFeature.dress_these_with_boxes(features,
                                                      max_intensity_in_window, 
                                                      min_intensity_in_window,
                                                      box_range_safety_margin=box_range_safety_margin,
                                                      scale_wrt_max_intensity=scale_wrt_max_intensity,
                                                      minimum_box_padding=minimum_box_padding,
                                                                        )

    return features

def _filter_magn_conds(features, rsp_eval_setup):
    features = SpectralFeature.apply_magn_cond_filter(features,
                                                      magn_conditions=rsp_eval_setup.terms.magn_conds,
                                                      magn_conditions_margin=magn_conditions_margin)

    return features

def _place_in_window(features, spec_window):
    spec_window = SpectralFeature.filter_to_spec_window(features, spec_window)
    if not spec_window.full_features:
        raise ValueError("This SpectralWindow does not contain any features. Change the bounds of the window or use different terms.")

    return spec_window

def _make_grid_manager(spec_window):
    grid_manager = GridManager(spec_window)
    grid_manager.make_fullgrid(grid_resolution)

    return grid_manager

def _make_regions(grid_manager):
    regions = grid_manager.create_regions()
    if not regions:
        raise ValueError("No regions were created")

    return regions

def evaluate_regions(regions, gamma_au, executor=None):
    fn = partial(_eval_one_region, gamma_au=gamma_au)
    if executor is None:
        return [fn(r) for r in regions]
    return list(executor.map(fn, regions))