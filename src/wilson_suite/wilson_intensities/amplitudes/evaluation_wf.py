import numpy as np

from dataclasses import dataclass

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from wilson_suite.wilson_main.workflow_abstractions import WilsonSimulation
    from wilson_suite.wilson_main.spectrum_abstractions import EvaluationInfo
    from wilson_suite.wilson_intensities.amplitudes.grid_manager_evaluator import GridRegion
    from wilson_suite.wilson_intensities.amplitudes.term_parts import PrecalculatedData, VibStatesData
    from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
    from wilson_suite.wilson_intensities.amplitudes.term_parts import EvaluationDataAndConfigs, ParameterSet, ResonanceMotif
    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, ResLocGeoObject
    from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
    from wilson_suite.wilson_derive.term_var_translate import SpectralAxisSet

from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_flat

from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralFeature
from wilson_suite.wilson_intensities.amplitudes.grid_manager_evaluator import GridManager

from wilson_suite.wilson_intensities.amplitudes.evaluators import prepDataForEval
from wilson_suite.wilson_intensities.amplitudes.evaluators import evaluate_terms_coeffs
from wilson_suite.wilson_intensities.amplitudes.full_amplitude_coeff import (precalculate_unique_coeff_parts, 
                                                                             identify_precalc_unique_coeff_parts)
from wilson_suite.wilson_intensities.amplitudes.evaluators import get_features_to_draw
from wilson_suite.wilson_intensities.amplitudes.evaluators import _compute_motif_locs, _get_terms_for_motifs

from wilson_suite.wilson_intensities.amplitudes.evaluators import evaluate_regions

import logging
logger = logging.getLogger("wilson")


###############################################################

@dataclass(frozen=True)
class ExperimentContext:
    """Pulse-indexed, axis-invariant. Determined by the experiment alone."""
    raw_terms: list['VibPerturbedTerm']
    pulse_polarization_vector: tuple
    magn_conditions: tuple
    # avrg_tensors, avrg_expr_tensor_mapping, vibenedenoms_tensors, vibdiff_motifs
    need_precalc: dict[str, Any]

@dataclass(frozen=True)
class AxisContext:
    """Axis choice applied to ExperimentContext."""
    experiment_ctx: ExperimentContext
    axes: 'SpectralAxisSet'
    terms: list['VibPerturbedTerm']
    terms_for_motifs: dict['ResonanceMotif', list['VibPerturbedTerm']]
    magn_conditions: tuple

@dataclass(frozen=True)
class QCDataContext:
    vib_data: 'VibStatesData'
    vibdiff_cache: 'VibDiffCache'
    data_configs: 'EvaluationDataAndConfigs'

@dataclass(frozen=True)
class PrecalcContext:
    """QC + precalc spec. Heavy compute. Survives axis changes."""
    experiment: ExperimentContext        # parent ref
    qc: QCDataContext                    # parent ref
    precalculated: 'PrecalculatedData'

@dataclass(frozen=True)
class BoundMotifs:
    """AxisContext x QCDataContext x PrecalcContext binding.
    Rebuilds on axis change."""
    axes: AxisContext             # parent ref
    precalc: PrecalcContext       # parent ref (covers experiment + qc)
    motif_locs: dict['ResonanceMotif', dict['ResLocGeoObject', list]]
    coefficients: dict['VibPerturbedTerm', dict['ParameterSet', float]]


@dataclass(frozen=True)
class GridContext:
    spec_window: 'SpectralWindow'
    grid_manager: 'GridManager'
    grid_resolution: dict

@dataclass(frozen=True)
class FeatureResult:
    features: list[SpectralFeature]
    zero_feats: list[SpectralFeature]


@dataclass(frozen=True)
class RegionEvaluation:
    """Per-region evaluation outputs. Intermediate."""
    regions: list['GridRegion']
    regions_results: dict[str, np.ndarray]

@dataclass(frozen=True)
class EvaluatedSpectrum:
    """
    The assembled spectrum on the grid.
        axes: {'A': ..., 'B': ...}
    """
    axes: dict[str, np.ndarray]
    result: np.ndarray

@dataclass(frozen=True)
class RenderSettings:
    """Settings applied during region evaluation (post-feature)."""
    box_range_safety_margin: float
    scale_wrt_max_intensity: bool
    minimum_box_padding: float
    apply_magn_cond_filter: bool = False
    exp_magn_conditions: tuple = ()
    magn_conditions_margin: dict = None
    dynamic_range: float = None


def build_experiment_context(simulation: 'WilsonSimulation') -> ExperimentContext:
    flat_terms = derived_terms_flat(simulation.terms, tolistonly=True)
    return ExperimentContext(
        raw_terms=flat_terms,
        pulse_polarization_vector=tuple(simulation.exp.polarization_avg_vector),
        magn_conditions=tuple(simulation.exp.magn_conditions),
        need_precalc=identify_precalc_unique_coeff_parts(flat_terms),
    )

def build_qc_context(simulation: 'WilsonSimulation') -> QCDataContext:
    vib_data, vibdiff_cache, data_configs = prepDataForEval(
        simulation.exp.polarization_avg_vector,
        simulation.vib_ana_setup, 
        simulation.props,
    )
    return QCDataContext(
        vib_data=vib_data,
        vibdiff_cache=vibdiff_cache,
        data_configs=data_configs,
    )

def build_precalc_context(
    experiment: ExperimentContext,
    qc: QCDataContext,
) -> PrecalcContext:
    precalculated = precalculate_unique_coeff_parts(
        need_to_precalc=experiment.need_precalc,
        data_and_configs=qc.data_configs,
    )
    return PrecalcContext(
        experiment=experiment,
        qc=qc,
        precalculated=precalculated,
    )


def build_axis_context(
    experiment_ctx: ExperimentContext,
    simulation: 'WilsonSimulation',
) -> AxisContext:
    if simulation.spec_eval_setup.ev_info.apply_exp_magn_conditions_eval:
        from wilson_suite.wilson_derive.term_var_translate import translate_magn_conditions_to_axisvars
        translated_magn_conditions = translate_magn_conditions_to_axisvars(experiment_ctx.magn_conditions, simulation.axis_choice)
    else:
        translated_magn_conditions = None
    if isinstance(simulation.terms_in_axis_choice, dict):
        translated_terms = derived_terms_flat(simulation.terms_in_axis_choice, tolistonly=True)
    else:
        translated_terms = simulation.terms_in_axis_choice

    return AxisContext(
        experiment_ctx=experiment_ctx,
        axes=simulation.spec_eval_setup.ev_info.spectral_axes,
        terms=translated_terms,
        terms_for_motifs=_get_terms_for_motifs(translated_terms),
        magn_conditions=translated_magn_conditions,
    )


def bind_motifs(
    axis_ctx: AxisContext,
    precalc: PrecalcContext,
) -> BoundMotifs:
    motif_locs = _compute_motif_locs(axis_ctx, precalc.qc)
    coefficients = evaluate_terms_coeffs(
        derived_terms=axis_ctx.terms,
        motif_res_loc=motif_locs,
        data_and_configs=precalc.qc.data_configs,
        precalculated=precalc.precalculated,
    )
    return BoundMotifs(
        axes=axis_ctx,
        precalc=precalc,
        motif_locs=motif_locs,
        coefficients=coefficients,
    )

def compute_features(
    bound: BoundMotifs,
    gamma: float,
) -> FeatureResult:
    features, zero_feats = get_features_to_draw(
        motif_res_loc=bound.motif_locs,
        terms_for_motifs=bound.axes.terms_for_motifs,
        term_coeffs_per_index=bound.coefficients,
        lineshape_parameter=gamma,
    )
    return FeatureResult(features=features, zero_feats=zero_feats)


def build_grid_context(
    spec_window: 'SpectralWindow',
    grid_resolution: dict,
) -> GridContext:
    grid_manager = GridManager(spec_window)
    grid_manager.make_fullgrid(grid_resolution)
    return GridContext(
        spec_window=spec_window,
        grid_manager=grid_manager,
        grid_resolution=grid_resolution
    )


def _settings_from_ev_info(ev_info: 'EvaluationInfo') -> RenderSettings:
    return RenderSettings(
        box_range_safety_margin=ev_info.box_range_safety_margin,
        scale_wrt_max_intensity=ev_info.scale_wrt_max_intensity,
        minimum_box_padding=ev_info.minimum_box_padding,
        apply_magn_cond_filter=ev_info.apply_exp_magn_conditions_eval,
        exp_magn_conditions=ev_info.exp_magn_conditions,
        magn_conditions_margin=ev_info.magn_conditions_margin,
        dynamic_range=ev_info.dynamic_range
    )

def filter_features_to_window(
    features: FeatureResult,
    spec_window: 'SpectralWindow',
) -> 'SpectralWindow':
    window = SpectralFeature.filter_to_spec_window(features.features, spec_window)
    if not (window.full_features + window.contrib_features):
        raise ValueError("No features in this spec window")
    return window


def apply_magn_cond_filter(
    window: 'SpectralWindow',
    settings: RenderSettings,
    verbose: bool = False,
) -> 'SpectralWindow':
    """Filter features by magnitude conditions, if enabled in settings."""
    if not settings.apply_magn_cond_filter:
        return window
    
    surviving_full = SpectralFeature.apply_magn_cond_filter(
        window.full_features,
        magn_conditions=settings.exp_magn_conditions,
        magn_conditions_margin=settings.magn_conditions_margin,
    )
    '''    
    surviving_contrib = SpectralFeature.apply_magn_cond_filter(
        window.contrib_features,
        magn_conditions=settings.exp_magn_conditions,
        magn_conditions_margin=settings.magn_conditions_margin,
    )
    '''
    if not surviving_full:
        raise ValueError("Magn-condition filter left 0 features")
    if verbose:
        print(f" After magn-cond filter: {len(surviving_full)} features")
    window.full_features = surviving_full
    return window

def _get_intensity_bounds(window: 'SpectralWindow'):
    feats = window.full_features
    max_intensity_in_window = SpectralFeature.get_max_intensity_feat(feats).get_intensity()
    min_intensity_in_window = 0.
    return max_intensity_in_window, min_intensity_in_window

def dress_features_with_boxes(
    window: 'SpectralWindow',
    settings: RenderSettings,
) -> 'SpectralWindow':
    """Add bounding boxes around features for region creation."""
    max_int, _ = _get_intensity_bounds(window)
    min_int = max_int / settings.dynamic_range
    
    window.full_features = SpectralFeature.dress_these_with_boxes(
        window.full_features,
        max_int, min_int,
        box_range_safety_margin=settings.box_range_safety_margin,
        scale_wrt_max_intensity=settings.scale_wrt_max_intensity,
        minimum_box_padding=settings.minimum_box_padding,
    )
    window.contrib_features = SpectralFeature.dress_these_with_boxes(
        window.contrib_features,
        max_int, min_int,
        box_range_safety_margin=settings.box_range_safety_margin,
        scale_wrt_max_intensity=settings.scale_wrt_max_intensity,
        minimum_box_padding=settings.minimum_box_padding,
    )

    return window


def prepare_features_for_evaluation(
    features: FeatureResult,
    spec_window: 'SpectralWindow',
    settings: RenderSettings,
    verbose: bool = False,
) -> 'SpectralWindow':
    window = filter_features_to_window(features, spec_window)
    window = apply_magn_cond_filter(window, settings, verbose)
    window = dress_features_with_boxes(window, settings)
    return window


def evaluate_regions_on_grid(
    prepared_window: 'SpectralWindow',
    grid_resolution: dict,
    qc: QCDataContext,
    gamma_au: float,
    verbose: bool = False,
) -> tuple[RegionEvaluation, GridContext]:
    eval_grid = build_grid_context(prepared_window, grid_resolution)
    regions = eval_grid.grid_manager.create_regions()
    if not regions:
        raise ValueError("No regions were created")
    regions_results = evaluate_regions(
        regions, qc.vib_data, qc.vibdiff_cache, gamma_au, verbose
    )
    return RegionEvaluation(regions=regions, regions_results=regions_results), eval_grid


def assemble_spectrum(region_eval: RegionEvaluation, grid: GridContext) -> EvaluatedSpectrum:
    full_grid = grid.grid_manager.place_results_into_grid(region_eval.regions_results)
    result = full_grid.pop('result')
    return EvaluatedSpectrum(axes=full_grid, result=result)


def _get_gamma_au(ev_info: 'EvaluationInfo') -> float:

    if ev_info.Gamma_unit == 'cm-1':
        return convNu2Ene(ev_info.Gamma)
    elif ev_info.Gamma_unit == 'au':
        return ev_info.Gamma
    raise ValueError(f"Gamma cannot be converted from unit {ev_info.Gamma_unit!r} to au")

def _get_gamma_cm(ev_info: 'EvaluationInfo') -> float:

    if ev_info.Gamma_unit == 'au':
        return convNu2Ene(ev_info.Gamma, reverse=True)
    elif ev_info.Gamma_unit == 'cm-1':
        return ev_info.Gamma
    raise ValueError(f"Gamma cannot be converted from unit {ev_info.Gamma_unit!r} to cm-1")


## -------------------------------------
class EvaluationWorkflow:
    def __init__(self, simulation: 'WilsonSimulation'):
        self.simulation = simulation
        # Cache for lazy builds
        self._experiment_ctx = None
        self._qcdata_ctx = None
        self._axis_ctx = None
        self._precalc_ctx = None
        self._bound_motifs_ctx = None
    
        self.feat_result = None
        self.region_eval = None

    @property
    def experiment_ctx(self):
        if self._experiment_ctx is None:
            self._experiment_ctx = build_experiment_context(self.simulation)
        return self._experiment_ctx
    
    @property
    def qcdata_ctx(self):
        if self._qcdata_ctx is None:
            self._qcdata_ctx = build_qc_context(self.simulation)
        return self._qcdata_ctx
    
    @property
    def axis_ctx(self):
        if self._axis_ctx is None:
            self._axis_ctx = build_axis_context(
                self.experiment_ctx,
                self.simulation,
            )
        return self._axis_ctx
    
    @property
    def precalc_ctx(self):
        if self._precalc_ctx is None:
            self._precalc_ctx = build_precalc_context(self.experiment_ctx, self.qcdata_ctx)
        return self._precalc_ctx
    
    @property
    def bound_motifs_ctx(self):
        if self._bound_motifs_ctx is None:
            self._bound_motifs_ctx = bind_motifs(self.axis_ctx, self.precalc_ctx)
        return self._bound_motifs_ctx
    
    
    def prepare(self) -> "EvaluationWorkflow":
        """
        Build all shared contexts now. 
        Returns self for chaining.
        
        triggers a chain of builders
        """
        _ = self.bound_motifs_ctx
        return self
    
    
    def evaluate(self, *, gamma_cm=None, 
                 spec_window=None, grid_resolution=None, 
                 settings=None, verbose=False) -> EvaluatedSpectrum:
        ev_info = self.simulation.spec_eval_setup.ev_info
        spec_window = spec_window if spec_window is not None else ev_info.spectral_window
        grid_resolution = grid_resolution if grid_resolution is not None else ev_info.grid_resolution
        settings = settings if settings is not None else _settings_from_ev_info(ev_info)
        gamma_cm = gamma_cm if gamma_cm is not None else ev_info.Gamma
        gamma_au = convNu2Ene(gamma_cm)
        
        features = compute_features(self.bound_motifs_ctx, gamma_cm)
        prepared_window = prepare_features_for_evaluation(features, spec_window, settings, verbose)
        
        region_eval, render_grid = evaluate_regions_on_grid(
            prepared_window, grid_resolution, self.qcdata_ctx, gamma_au, verbose
        )
        spectrum = assemble_spectrum(region_eval, render_grid)
        
        self.feat_result = features
        self.region_eval = region_eval
        return spectrum

    # ---- Sweeps ----
    
    def sweep_gamma(self, gammas_cm: list[float]) -> dict[float, EvaluatedSpectrum]:
        return {g: self.evaluate(gamma_cm=g) for g in gammas_cm}
    
    def sweep_grid(self, grid_specs: list[tuple['SpectralWindow', dict]]) -> dict[tuple, EvaluatedSpectrum]:
        return {(w, r): self.evaluate(spec_window=w, grid_resolution=r) for w, r in grid_specs}
    
    def sweep_gamma_x_grid(self, gammas_cm, grid_specs) -> dict[tuple[float, tuple], EvaluatedSpectrum]:
        return {
            (g, (w, r)): self.evaluate(gamma_cm=g, spec_window=w, grid_resolution=r)
            for g in gammas_cm for (w, r) in grid_specs
        }
    

###########################
