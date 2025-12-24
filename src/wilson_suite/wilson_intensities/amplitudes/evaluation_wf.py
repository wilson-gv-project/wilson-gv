import time
import numpy as np
from wilson_suite.wilson_intensities.amplitudes.grid_manager_evaluator import FeatureCompiler, PhysicsCalculator

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wilson_suite.wilson_main.workflow_abstractions import WilsonSimulation
    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, SpectralFeature
    from wilson_suite.wilson_intensities.amplitudes.grid_manager_evaluator import GridManager, GridRegion

import logging
logger = logging.getLogger("wilson")


class WorkflowError(Exception):
    """Base class for workflow errors."""
    pass
class InputValidationError(WorkflowError):
    """Raised when input validation fails."""
    pass
class StepExecutionError(WorkflowError):
    """Raised when a specific step fails."""
    def __init__(self, step_name, original_exception):
        super().__init__(f"Step '{step_name}' failed: {original_exception}")
        self.step_name = step_name
        self.original_exception = original_exception

class EvaluationWorkflow:
    """Simple workflow that tracks steps and captures intermediates on error"""
    
    def __init__(self, simulation: "WilsonSimulation"):
        self.simulation = simulation
        self.results = {}
        self.timing = {}
        self.failed_at = None
        self.verbose = False

    def _validate_inputs(self):
        """
        WilsonSimulation at this point should have necessary data
        """
        if not hasattr(self.simulation, 'terms') or not self.simulation.terms:
            raise ValueError("Simulation object must have a non-empty 'terms' attribute.")
        if not hasattr(self.simulation, 'system') or not self.simulation.system:
            raise ValueError("Simulation object must have a non-empty 'system' attribute.")
        if not hasattr(self.simulation, 'props') or not self.simulation.props:
            raise ValueError("Simulation object must have a non-empty 'props' attribute.")
        

    def run(self, *, keep_intermediates: bool = False):
        """Run evaluation, return (spectrum, info_dict)"""
        self._validate_inputs()
        start = time.time()
        
        try:
            # prep steps
            terms_list = self._step('prep_terms', self._prep_terms)
            vib_data, vib_cache, data_configs = self._step('prep_data', self._prep_data)

            # get resonances locations for all terms
            motif_locs, terms_for_motifs = self._step('process_resonances', 
                lambda: self._process_resonances(terms_list, vib_data, vib_cache))
            
            coefficients = self._step('term_coefficients',
                lambda: self._calc_coefficients(terms_list, motif_locs, data_configs))
            
            features = self._step('all_features',
                lambda: self._extract_features(motif_locs, terms_for_motifs, coefficients))
            
            spec_window = self._step('place_in_specwindow',
                lambda: self._place_in_specwindow(features))
            


            grid_manager = self._step('make_GridManager',
                lambda: self._make_GridManager(spec_window))
            grid_manager.make_fullgrid(self.simulation.spec_eval_setup.ev_info.grid_resolution)
            
            regions = self._step('make_regions',
                lambda: self._make_regions(grid_manager))
            
            self._step('prep_complilers',
                lambda: self._prep_complilers(vib_data, vib_cache, self.simulation.spec_eval_setup.ev_info.Gamma))
            
            regions_results = self._step('evaluate_regions',
                lambda: self._evaluate_regions(regions))
            
            self._step('assemble_fullgrid',
                lambda: self._assemble_fullgrid(grid_manager, regions_results))

            
            info = {'timing': self.timing, 'total_time': time.time() - start}
            if keep_intermediates:
                info['intermediates'] = self.results
            
            return grid_manager.full_grid, info
            
        except Exception as e:
            # On error, always include intermediates
            info = {
                'timing': self.timing,
                'total_time': time.time() - start,
                'failed_at': self.failed_at,
                'error': str(e),
                'intermediates': self.results,
            }

            raise type(e)(f"Failed at '{self.failed_at}': {e}") from e

    def _save_checkpoint(self, name):
        import json

        # Save intermediate results to a file or log them
        with open(f'checkpoint_{name}.json', 'w') as f:
            json.dump(self.results, f)
    
    def _step(self, name, func):
        """Execute step, track timing and result"""
        self.failed_at = name
        start = time.time()
        # try:
        result = func()
        # except Exception as e:
        #     raise StepExecutionError(name, e) from e
        self.timing[name] = time.time() - start
        self.results[name] = result
        # self._save_checkpoint(name)  # Save checkpoint
        return result
    
    # Step implementations
    def _prep_terms(self):
        """
        Make flat list of VibPerturbedTerm from the dict
        """
        from wilson_suite.wilson_intensities.amplitudes.evaluators import prepTermsForEval
        terms = prepTermsForEval(self.simulation.terms)
        if not terms:
            raise ValueError("No terms were prepared with prepTermsForEval(). Check the input terms.")
        return terms
    
    def _prep_data(self):
        from wilson_suite.wilson_intensities.amplitudes.evaluators import prepDataForEval
        
        return prepDataForEval(self.simulation.system, self.simulation.exp, 
                               self.simulation.vib_ana_setup, self.simulation.props)
    
    def _prep_complilers(self, vib_data, vibdiff_cache, gamma):
        self.physics = PhysicsCalculator(gamma)
        self.compiler = FeatureCompiler(vib_data, vibdiff_cache)

    def _process_resonances(self, terms_list, vib_data, vib_cache):
        from wilson_suite.wilson_intensities.amplitudes.evaluators import process_resonance_motifs
        return process_resonance_motifs(terms_list, vib_data, vib_cache)
    
    def _calc_coefficients(self, terms_list, motif_locs, data_configs):
        from wilson_suite.wilson_intensities.amplitudes.evaluators import (
            evaluate_terms_coeffs, precalculate_unique_coeff_parts, 
            identify_precalc_unique_coeff_parts
        )
        need_precalc = identify_precalc_unique_coeff_parts(terms=terms_list)
        precalculated = precalculate_unique_coeff_parts(
            need_to_precalc=need_precalc, data_and_configs=data_configs)

        return evaluate_terms_coeffs(terms_list, motif_locs, precalculated=precalculated)
    
    def _extract_features(self, motif_locs, terms_for_motifs, coefficients) -> list["SpectralFeature"]:
        from wilson_suite.wilson_intensities.amplitudes.evaluators import get_features_to_draw
        return get_features_to_draw(
            motif_res_loc=motif_locs, terms_for_motifs=terms_for_motifs,
            term_coeffs_per_index=coefficients,
            lineshape_parameter=self.simulation.spec_eval_setup.ev_info.Gamma)
    
    def _place_in_specwindow(self, features) -> "SpectralWindow":
        from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralFeature
        return SpectralFeature.filter_to_spec_window(
            features, self.simulation.spec_eval_setup.ev_info.spectral_window)
    
    def _make_GridManager(self, spec_window):
        from wilson_suite.wilson_intensities.amplitudes.grid_manager_evaluator import GridManager
        return GridManager(spec_window)

    def _make_fullgrid(self, grid_manager: "GridManager"):
        grid_manager.make_fullgrid()

    def _make_regions(self, grid_manager: "GridManager") -> list["GridRegion"]:
        return grid_manager.create_regions()

    def _evaluate_regions(self, regions: list["GridRegion"]):

        # Step 2: Evaluate each region
        region_results = {}
        for region in regions:
            if self.verbose:
                logger.info(f"\nEvaluating region with {len(region.features)} features")
            
            region_results[region] = self._evaluate_region(region, self.verbose)
            
            if self.verbose:
                intensity = region_results[region]
                logger.info(f"  Region shape: {intensity.shape}")
                logger.info(f"  Max intensity: {np.max(np.abs(intensity))}")
        return region_results
    
    def _evaluate_region(self, 
                        region: "GridRegion",
                        verbose: bool = False) -> np.ndarray:
        """Evaluate all features in a single grid region."""
        # Initialize result array
        result = np.zeros_like(
            next(iter(region.coords.values())), 
            dtype=complex
        )
        
        # Sum contributions from all features
        for feature in region.features:
            if verbose:
                logger.info(f"  Feature: amplitude={feature.amplitude_coeff}")
            
            result += self._evaluate_feature(feature, region.coords, verbose)
            
        return result
    
    def _evaluate_feature(self,
                         feature: 'SpectralFeature',
                         coords: dict[str, np.ndarray],
                         verbose: bool = False) -> np.ndarray:
        """Evaluate a single feature on grid coordinates."""
        # Compile feature to numerical form
        compiled_groups = self.compiler.compile_feature(feature)
        
        if verbose:
            logger.info(f"    Compiled into {len(compiled_groups)} term groups")
        
        # Sum all compiled groups
        feature_sum = np.zeros_like(
            next(iter(coords.values())),
            dtype=complex
        )
        
        for group in compiled_groups:
            feature_sum += self.physics.evaluate_compiled_group(group, coords)
        
        # Apply amplitude coefficient
        return feature.amplitude_coeff * feature_sum
    
    def _assemble_fullgrid(self, grid_manager: "GridManager", region_results):
        grid_manager.place_results_into_grid(region_results)


    def _evaluate_spectrum(self, spec_window, vib_data, vib_cache):
        from wilson_suite.wilson_intensities.amplitudes.grid_manager_evaluator import SpectralEvaluator
        evaluator = SpectralEvaluator(vib_data, vib_cache, 
                                     gamma=self.simulation.spec_eval_setup.ev_info.Gamma)
        return evaluator.evaluate_spectrum(
            spec_window=spec_window,
            grid_resolution=self.simulation.spec_eval_setup.ev_info.grid_resolution,
            return_type='grid')
