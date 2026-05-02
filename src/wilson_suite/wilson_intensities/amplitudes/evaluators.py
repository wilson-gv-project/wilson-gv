"""
Evaluator functions for WilsonSimulation
"""
from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import ResLocGeoObject, SpectralFeature
from ..amplitudes.full_amplitude_coeff import evaluate_term_coeffs
from ..amplitudes.resonances import find_resonance_locations_wrt_index_choices, identify_unique_resmotifs

from wilson_suite.wilson_main.abstractions import VibAnaSetup, MolecularSystem, MolPropsCollection
from wilson_suite.wilson_intensities.amplitudes.term_parts import ResonanceMotif, VibStatesData
from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
from wilson_suite.wilson_intensities.amplitudes.term_parts import (EvaluationDataAndConfigs, ParameterSet,
                                                                   TermParametersChoice, PrecalculatedData)
from ...wilson_derive.response_terms import VibPerturbedTerm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wilson_suite.wilson_main.abstractions import VibAnaSetup, MolecularProperty
    from .evaluation_wf import QCDataContext, AxisContext
    from wilson_suite.wilson_intensities.amplitudes.grid_manager_evaluator import GridRegion
    from wilson_suite.wilson_intensities.amplitudes.numerical_abstractions import CompiledTermGroup, NumericalResonanceMotif

from .numerical_abstractions import compile_feature

import numpy as np

import logging
logger = logging.getLogger("wilson."+__name__)


def prepDataForEval(pulse_polarization_vector: np.ndarray,
                    vib_ana_setup: 'VibAnaSetup',
                    props: list['MolecularProperty']) -> tuple[VibStatesData, VibDiffCache, EvaluationDataAndConfigs]:
    """
    put data in a form for use on the evaluation step
    """
    include_list = tuple([v for v in list(vib_ana_setup.nc_sqrt_eigval.keys()) if v not in vib_ana_setup.exclude_modes])
    if include_list == tuple():
        raise ValueError("include_list of included normal modes labels is empty")
    

    vibstates_data = VibStatesData(allstates=tuple(vib_ana_setup.states), 
                                   harmonic_osc_states_labels=vib_ana_setup.include_list,
                                   number_of_nmodes=vib_ana_setup.number_of_modes)
    vibdiff_cache = VibDiffCache()
    props = MolPropsCollection(properties=props)
    
    data_and_configs = EvaluationDataAndConfigs(props_data=props,
                                                vibstates_data=vibstates_data,
                                                number_of_nmodes=vib_ana_setup.number_of_modes,
                                                nm_inds_choices=include_list,
                                                pulse_polarization_vector=pulse_polarization_vector,
                                                nc_sqrt_eigval=vib_ana_setup.nc_sqrt_eigval)

    return vibstates_data, vibdiff_cache, data_and_configs


def _get_terms_for_motifs(derived_terms: list['VibPerturbedTerm']):
    
    unique_res_motifs = identify_unique_resmotifs(derived_terms)
    terms_for_motifs: dict[ResonanceMotif, list[VibPerturbedTerm]] = {res_motif: [] for res_motif in unique_res_motifs}
    
    for vibterm in derived_terms:
        res_motif = ResonanceMotif(vibterm.res)
        for u_motif in unique_res_motifs:
            if u_motif == res_motif:
                terms_for_motifs[u_motif].append(vibterm)
    
    return terms_for_motifs

def _compute_motif_locs(axis_ctx: 'AxisContext', qc: 'QCDataContext'):
    
    unique_res_motifs = identify_unique_resmotifs(axis_ctx.terms)
    motif_res_loc: dict[ResonanceMotif, ResLocGeoObject] = {}
    
    for res_motif in unique_res_motifs:
        this_motif_res_locs = find_resonance_locations_wrt_index_choices(
            motif=res_motif,
            vibstates_data=qc.vib_data,
            vibdiff_cache=qc.vibdiff_cache,
            spec_window=None
        )
        motif_res_loc.update(this_motif_res_locs)

    return motif_res_loc

def evaluate_terms_coeffs(derived_terms: list['VibPerturbedTerm'],
                  motif_res_loc: dict[ResonanceMotif, dict[ResLocGeoObject]],
                  data_and_configs: EvaluationDataAndConfigs,
                  precalculated: PrecalculatedData) -> dict['VibPerturbedTerm', dict[ParameterSet, float]]:
    """
    Evaluate coefficients for all terms.

    terms to dict with (params to coeff)
    """
    term_coeffs_per_index = {}
    for vibterm in derived_terms:
        res_motif = ResonanceMotif(vibterm.res)
        term_coeffs_per_index[vibterm] = evaluate_term_coeffs(
            term=vibterm,
            relevant_indices=[k for i in motif_res_loc[res_motif].values() for k in i],
            necessary_data=(data_and_configs, precalculated)
        )
    
    return term_coeffs_per_index



def get_features_to_draw(motif_res_loc: dict[ResonanceMotif, dict[ResLocGeoObject, list]],
                         terms_for_motifs: dict[ResonanceMotif, list['VibPerturbedTerm']], 
                         term_coeffs_per_index: dict['VibPerturbedTerm', 
                                                     dict[ParameterSet, tuple[float, dict]]]=None,
                         lineshape_parameter: float=None) -> tuple[list[SpectralFeature], list[SpectralFeature]]:
    """

    lineshape_parameter - uniform lineshape parameters (for all axes) for each feature for now
    lineshape_parameter unit -- will be au - follows from the workflow in step("all_features")
    """
    # a SpectralFeature instanse holds a res_location and list of states parameters that give this res_location; 
    #       the amplitude coefficient is a value in the dict
    features_to_draw: list[SpectralFeature] = []
    zero_coeff_feats: list[SpectralFeature] = []

    for res_motif in motif_res_loc:

        for res_geo_obj, list_state_dicts in motif_res_loc[res_motif].items():

            lst_params = tuple([ParameterSet(states_dict) for states_dict in list_state_dicts])
            term_contributions=tuple([TermParametersChoice(res_motif=res_motif,
                                        states_parameters=lst_params,
                                        term_ids=tuple([t.to_str() for t in terms_for_motifs[res_motif]]) )])

            if term_coeffs_per_index is not None:
                list_to_sum = [term_coeffs_per_index[term][ParameterSet(states_dict)][0] for term in terms_for_motifs[res_motif] for states_dict in list_state_dicts]
                dict_of_contribs = {term.to_str(): term_coeffs_per_index[term][ParameterSet(states_dict)] for term in terms_for_motifs[res_motif] for states_dict in list_state_dicts}
                amplitude_coeff = sum(list_to_sum)
            else:
                dict_of_contribs = None
                amplitude_coeff = None
            
            # disregard locations where coefficient is zero
            spec_feature = SpectralFeature(location=res_geo_obj, 
                                        term_contributions=term_contributions,
                                        term_contrib_by_id = dict_of_contribs,
                                        lineshape_parameter=lineshape_parameter, # uniform lineshape parameters (for all axes) for each feature
                                        amplitude_coeff=amplitude_coeff)
            if amplitude_coeff != 0.:

                if spec_feature not in features_to_draw:
                    features_to_draw.append(spec_feature)
                else:
                    new_specfeat = spec_feature.union(features_to_draw[res_geo_obj][1])
                    features_to_draw.append(new_specfeat)
            else:
                zero_coeff_feats.append(spec_feature)

    return features_to_draw, zero_coeff_feats



def evaluate_regions(regions: list["GridRegion"], 
                     vib_data: "VibStatesData", 
                     vibdiff_cache: "VibDiffCache", 
                     gamma: float,
                     verbose: bool):

    region_results = {}
    for region in regions:
        if verbose:
            logger.info(f"\nEvaluating region with {len(region.features)} features")
        
        region_results[region] = evaluate_region(region, vib_data, vibdiff_cache, gamma, verbose)
        
        if verbose:
            intensity = region_results[region]
            logger.info(f"  Region shape: {intensity.shape}")
            logger.info(f"  Max intensity: {np.max(np.abs(intensity))}")
    return region_results

def evaluate_region(region: "GridRegion",
                    vib_data: "VibStatesData", 
                    vibdiff_cache: "VibDiffCache", 
                    gamma: float,
                    verbose: bool = False) -> np.ndarray:
    """Evaluate all features in a single grid region."""
    # Initialize result array
    target_shape = np.broadcast(*(arr for arr in region.coords.values())).shape
    result = np.zeros(target_shape, dtype=complex)
    
    # Sum contributions from all features
    for feature in region.features:
        if verbose:
            logger.info(f"  Feature: amplitude={feature.amplitude_coeff}")
        
        result += evaluate_feature(feature, vib_data, vibdiff_cache, gamma, region.coords_au, verbose)
        
    return result

def evaluate_feature(feature: 'SpectralFeature', 
                     vib_data: "VibStatesData", 
                     vibdiff_cache: "VibDiffCache", 
                     gamma: float,
                     coords: dict[str, np.ndarray],
                     verbose: bool = False) -> np.ndarray:
    """Evaluate a single feature on grid coordinates."""
    # Compile feature to numerical form
    compiled_groups = compile_feature(feature, vib_data, vibdiff_cache)

    
    if verbose:
        logger.info(f"    Compiled into {len(compiled_groups)} term groups")
    
    # Sum all compiled groups
    target_shape = np.broadcast(*(arr for arr in coords.values())).shape
    feature_sum = np.zeros(target_shape, dtype=complex)

    
    for group in compiled_groups:
        feature_sum += evaluate_compiled_group(group, coords, gamma)
    
    # Apply amplitude coefficient
    return feature.amplitude_coeff * feature_sum


def evaluate_resonance_motif(motif: 'NumericalResonanceMotif',
                             coords: dict[str, np.ndarray],
                             gamma: float) -> np.ndarray:
    """
    Calculate resonance motif contribution at grid points.
    
    Args:
        motif: Compiled resonance motif with conditions
        coords: Dict of axis_label -> meshgrid array
        
    Returns:
        Complex array with resonance contributions
    """
    target_shape = np.broadcast(*(arr for arr in coords.values())).shape
    total = np.ones(target_shape, dtype=complex)
    
    for res_cond in motif.res_conds:
        # Calculate photon frequency: sum over axes
        pfreq = sum(coords[ax] * res_cond.pf_dict[ax] 
                    for ax in res_cond.pf_dict)
        # Resonance denominator
        z = res_cond.vib_energy_diff - pfreq - 1j * gamma
        total *= 1.0 / z
        
    return total

def evaluate_compiled_group(group: 'CompiledTermGroup',
                            coords: dict[str, np.ndarray],
                            gamma: float) -> np.ndarray:
    """Sum all resonance motifs in a compiled group."""
    target_shape = np.broadcast(*(arr for arr in coords.values())).shape
    result = np.zeros(target_shape, dtype=complex)
    
    for motif in group.resonance_motifs:
        result += evaluate_resonance_motif(motif, coords, gamma)
    return result


