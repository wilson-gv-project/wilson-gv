"""
Refactored spectral evaluation with separated concerns.

Architecture:
1. Physics - Pure calculations (resonance conditions, energy differences)
2. Grid Management - Handles meshgrids and spatial organization
3. Feature Evaluation - High-level orchestration
"""

from dataclasses import dataclass, field
from typing import List, Dict, Union
import numpy as np

from wilson_suite.wilson_intensities.amplitudes.numerical_abstractions import CompiledTermGroup, NumericalResonanceMotif
from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import Box, RectangularDomain, SpectralFeature, SpectralWindow

import logging
logger = logging.getLogger("wilson")


# =============================================================================
# 1. PHYSICS LAYER - Pure calculations, no side effects
# =============================================================================

class PhysicsCalculator:
    """Pure physics calculations for resonance conditions."""
    
    def __init__(self, vib_data, vibdiff_cache, gamma: float):
        self.vib_data = vib_data
        self.vibdiff_cache = vibdiff_cache
        self.gamma = gamma
    
    def evaluate_resonance_motif(self, 
                                  motif: 'NumericalResonanceMotif',
                                  coords: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate resonance motif contribution at grid points.
        
        Args:
            motif: Compiled resonance motif with conditions
            coords: Dict of axis_label -> meshgrid array
            
        Returns:
            Complex array with resonance contributions
        """
        total = np.ones_like(next(iter(coords.values())), dtype=complex)
        
        for res_cond in motif.res_conds:
            # Calculate photon frequency: sum over axes
            pfreq = sum(coords[ax] * res_cond.pf_dict[ax] 
                       for ax in res_cond.pf_dict)
            
            # Resonance denominator
            z = res_cond.vib_energy_diff - pfreq - 1j * self.gamma
            total *= 1.0 / z
            
        return total
    
    def evaluate_compiled_group(self,
                                group: 'CompiledTermGroup',
                                coords: Dict[str, np.ndarray]) -> np.ndarray:
        """Sum all resonance motifs in a compiled group."""
        result = np.zeros_like(next(iter(coords.values())), dtype=complex)
        
        for motif in group.resonance_motifs:
            result += self.evaluate_resonance_motif(motif, coords)
            
        return result


# =============================================================================
# 2. GRID MANAGEMENT - Spatial organization
# =============================================================================

@dataclass
class GridRegion:
    """A single region of the evaluation grid."""
    domain: 'RectangularDomain'
    coords: Dict[str, np.ndarray]  # axis_label -> meshgrid
    indices: tuple  # Slice indices into full grid
    region_id: int = field(default=0)  # Unique identifier for hashing
    
    def __hash__(self):
        """Make GridRegion hashable for use as dict keys."""
        return hash((id(self.domain), self.region_id))
    
    def __eq__(self, other):
        """Equality based on domain identity and region_id."""
        if not isinstance(other, GridRegion):
            return False
        return self.domain is other.domain and self.region_id == other.region_id
    
    @property
    def features(self) -> List['SpectralFeature']:
        """All features that contribute to this region."""
        return self.domain.full_features + self.domain.contrib_features


class GridManager:
    """Manages grid partitioning and coordinate systems."""
    
    def __init__(self, spec_window: 'SpectralWindow'):
        self.spec_window = spec_window
        self.full_grid = None  # Store full grid dict
        
    def create_regions(self, grid_resolution: Dict[str, int]) -> List[GridRegion]:
        """
        Partition spectrum into regions based on feature clustering.
        
        Args:
            grid_resolution: Dict of axis_label -> number of points
            
        Returns:
            List of GridRegion objects
        """
        # Get all features
        all_features = (self.spec_window.full_features + 
                       self.spec_window.contrib_features)
        
        # Cluster features into domains
        from . import domains
        clusters = domains.features_to_clusters(features=all_features)
        formal_domains = [
            RectangularDomain(
                box=Box.union([f.feat_box for f in clusters[c]]),
                full_features=clusters[c]
            )
            for c in clusters
        ]
        
        # Create full grid
        coords_vectors, full_grid = self.spec_window.sample_grid(grid_resolution)
        
        # Store full grid for later assembly
        self.full_grid = full_grid
        
        # Cut grid into subgrids
        subgrids = domains.cut_grid_with_coords_nd(
            full_grid, coords_vectors, formal_domains
        )
        
        # Create GridRegion objects
        regions = []
        for idx, (domain, grid_info) in enumerate(subgrids.items()):
            regions.append(GridRegion(
                domain=domain,
                coords=grid_info['grid'],
                indices=grid_info['indices'],
                region_id=idx
            ))
            
        return regions
    
    def place_results_into_grid(self, 
                                results: Dict[GridRegion, np.ndarray],
                                result_key: str = "result",
                                combine_strategy: str = 'sum') -> Dict[str, np.ndarray]:
        """
        Place evaluated region results back into full grid dictionary.
        Matches your original insert_results_to_grid_nd pattern.
        
        Args:
            results: Dict mapping GridRegion -> evaluated intensity array
            result_key: Key to store results in grid dict (default: "result")
            combine_strategy: How to combine overlapping regions:
                - 'sum': Add contributions (default for spectroscopy)
                - 'overwrite': Last region wins
            
        Returns:
            Updated full grid dict with result_key added
            
        Example:
            regions = grid_mgr.create_regions({'A': 100, 'B': 100})
            results = {region: evaluated_array for region in regions}
            full_grid = grid_mgr.place_results_into_grid(results)
            # Now full_grid['result'] contains assembled spectrum
        """
        if self.full_grid is None:
            raise ValueError("Must call create_regions() before place_results_into_grid()")
        
        # Get shape from any axis array
        first_axis = next(iter(self.full_grid.values()))
        
        # Initialize result array if not present
        if result_key not in self.full_grid:
            # Determine dtype from first result
            first_result = next(iter(results.values()))
            dtype = first_result.dtype if hasattr(first_result, 'dtype') else complex
            self.full_grid[result_key] = np.zeros_like(first_axis, dtype=dtype)
        
        # Place each region's result
        if combine_strategy == 'sum':
            for region, result_array in results.items():
                self.full_grid[result_key][region.indices] += result_array
        elif combine_strategy == 'overwrite':
            for region, result_array in results.items():
                self.full_grid[result_key][region.indices] = result_array
        else:
            raise ValueError(f"Unknown combine_strategy: {combine_strategy}")
        
        return self.full_grid


# =============================================================================
# 3. FEATURE COMPILATION - Convert features to numerical form
# =============================================================================

class FeatureCompiler:
    """Compiles spectral features into numerical form for evaluation."""
    
    def __init__(self, vib_data, vibdiff_cache):
        self.vib_data = vib_data
        self.vibdiff_cache = vibdiff_cache
    
    def compile_feature(self, feature: 'SpectralFeature') -> List['CompiledTermGroup']:
        """
        Compile a spectral feature into numerical resonance motifs.
        
        This converts symbolic term contributions into concrete numerical
        resonance conditions ready for grid evaluation.
        """
        # Import here to avoid circular dependencies
        from .evaluators import compile_feature
        return compile_feature(feature, self.vib_data, self.vibdiff_cache)


# =============================================================================
# 4. EVALUATION ORCHESTRATOR - High-level pipeline
# =============================================================================

class SpectralEvaluator:
    """
    Main evaluation pipeline - coordinates physics, grids, and features.
    
    This is the clean interface to the entire evaluation system.
    """
    
    def __init__(self, vib_data, vibdiff_cache, gamma: float):
        self.physics = PhysicsCalculator(vib_data, vibdiff_cache, gamma)
        self.compiler = FeatureCompiler(vib_data, vibdiff_cache)
        self.grid_mgr = None  # Set during evaluate_spectrum
        
    def evaluate_spectrum(self,
                         spec_window: 'SpectralWindow',
                         grid_resolution: Dict[str, int],
                         verbose: bool = False,
                         return_type: str = 'grid') -> Union[Dict[str, np.ndarray], 
                                                              Dict[GridRegion, np.ndarray]]:
        """
        Evaluate entire spectral window.
        
        Args:
            spec_window: Spectral window with features to evaluate
            grid_resolution: Dict of axis_label -> number of points
            verbose: Print progress information
            return_type: What to return:
                - 'grid': Full grid dict with 'result' key (default)
                - 'regions': Dict mapping GridRegion -> intensity array
                - 'both': Tuple of (grid_dict, regions_dict)
            
        Returns:
            Depends on return_type:
            - 'grid': Dict with axis meshgrids + 'result' key
            - 'regions': Dict mapping GridRegion -> evaluated array
            - 'both': (grid_dict, regions_dict)
            
        Example:
            # Get assembled full grid
            full_grid = evaluator.evaluate_spectrum(spec_window, {'A': 100, 'B': 100})
            spectrum = full_grid['result']
            
            # Or get individual regions
            regions = evaluator.evaluate_spectrum(spec_window, {'A': 100, 'B': 100}, 
                                                 return_type='regions')
        """
        # Step 1: Partition into regions
        self.grid_mgr = GridManager(spec_window)
        regions = self.grid_mgr.create_regions(grid_resolution)
        
        if verbose:
            logger.info(f"Created {len(regions)} grid regions")
        
        # Step 2: Evaluate each region
        region_results = {}
        for region in regions:
            if verbose:
                logger.info(f"\nEvaluating region with {len(region.features)} features")
            
            region_results[region] = self._evaluate_region(region, verbose)
            
            if verbose:
                intensity = region_results[region]
                logger.info(f"  Region shape: {intensity.shape}")
                logger.info(f"  Max intensity: {np.max(np.abs(intensity))}")
        
        # Step 3: Assemble into full grid
        if return_type in ['grid', 'both']:
            self.grid_mgr.place_results_into_grid(region_results)
            if verbose:
                logger.info(f"\nAssembled full grid with shape: {self.grid_mgr.full_grid['result'].shape}")
        
        # Return based on type
        if return_type == 'grid':
            return self.grid_mgr.full_grid
        elif return_type == 'regions':
            return region_results
        elif return_type == 'both':
            return self.grid_mgr.full_grid, region_results
        else:
            raise ValueError(f"Unknown return_type: {return_type}")
    
    def get_full_grid(self) -> Dict[str, np.ndarray]:
        """
        Get the assembled full grid after evaluation.
        
        Returns:
            Grid dict with axis meshgrids and 'result' key
            
        Raises:
            ValueError if evaluate_spectrum hasn't been called yet
        """
        if self.grid_mgr is None or self.grid_mgr.full_grid is None:
            raise ValueError("Must call evaluate_spectrum() first")
        return self.grid_mgr.full_grid
    
    def _evaluate_region(self, 
                        region: GridRegion,
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
                         coords: Dict[str, np.ndarray],
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

# =============================================================================
# 5. CONVENIENCE FUNCTION - Simple interface for existing code
# =============================================================================

def evaluate_all_on_grids(grid_info_dict: Dict['RectangularDomain', dict],
                          vib_data, vibdiff_cache, gamma: float) -> dict:
    """
    Backward-compatible wrapper for existing code.
    
    Args:
        grid_info_dict: {domain: {'indices': slices, 'grid': coords}, ...}
        vib_data: Vibrational states data
        vibdiff_cache: Cache for energy differences
        gamma: Linewidth parameter
        
    Returns:
        Updated grid_info_dict with 'result' key added to each domain
    """
    evaluator = SpectralEvaluator(vib_data, vibdiff_cache, gamma)
    
    for domain, info in grid_info_dict.items():
        # Create temporary region
        region = GridRegion(
            domain=domain,
            coords=info['grid'],
            indices=info['indices']
        )
        
        # Evaluate
        info['result'] = evaluator._evaluate_region(region, verbose=True)
    
    return grid_info_dict

