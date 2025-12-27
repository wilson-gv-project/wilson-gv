"""
Refactored spectral evaluation with separated concerns.

Architecture:
1. Physics - Pure calculations (resonance conditions, energy differences)
2. Grid Management - Handles meshgrids and spatial organization
3. Feature Evaluation - High-level orchestration
"""

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import RectangularDomain, SpectralFeature, SpectralWindow

import logging
logger = logging.getLogger("wilson")


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
    
    @property
    def coords_au(self):
        from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
        return {k:convNu2Ene(v) for k,v in self.coords.items()}
    
class GridManager:
    """
    Manages grid partitioning and coordinate systems.

    
    """
    
    def __init__(self, spec_window: 'SpectralWindow'):
        self.spec_window = spec_window
        self.full_grid = None  # Store full grid dict
    
    def make_fullgrid(self, grid_resolution: Dict[str, int]):
        self.coords_vectors, self.full_grid = self.spec_window.sample_grid(grid_resolution)
    
    def _cut_fullgrid_to_domains(self, formal_domains):
        from . import domains
        subgrids = domains.cut_grid_to_domains_nd(
            self.full_grid, self.coords_vectors, formal_domains
        )
        return subgrids

    def create_regions(self) -> List[GridRegion]:
        """
        Partition spectrum into regions based on feature clustering.
        
        Args:
            grid_resolution: Dict of axis_label -> number of points
            
        Returns:
            List of GridRegion objects
        """
        formal_domains = self.spec_window.find_clusters_by_featboxes()
        
        if self.full_grid is None:
            raise ValueError("full_grid is None in this GridManager")
        
        subgrids = self._cut_fullgrid_to_domains(formal_domains)

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
