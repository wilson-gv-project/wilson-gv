"""
Evaluator functions for WilsonSimulation
"""
from .numerical_abstractions import NumericalResonanceMotif, CompiledTermGroup, compile_feature
from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import RectangularDomain, ResLocGeoObject, SpectralFeature, Box
from ..amplitudes.full_amplitude_coeff import evaluate_term_coeffs, precalculate_unique_coeff_parts, identify_precalc_unique_coeff_parts
from ..amplitudes.resonances import find_resonance_locations_wrt_index_choices, identify_unique_resmotifs

from wilson_suite.wilson_main.abstractions import VibAnaSetup, MolecularSystem, MolPropsCollection
from wilson_suite.wilson_intensities.amplitudes.term_parts import ResonanceMotif, VibStatesData
import wilson_suite.wilson_intensities.amplitudes.domains as domfuncs
from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
from wilson_suite.wilson_intensities.amplitudes.term_parts import (EvaluationDataAndConfigs, ParameterSet,
                                                                   TermParametersChoice)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wilson_suite.wilson_main.abstractions import VibAnaSetup, MolecularProperty
    from wilson_suite.wilson_main.spectrum_abstractions import SpecEvalSetup
    from ...wilson_derive.response_terms import VibPerturbedTerm
    from wilson_suite.wilson_experiment.experiment_abstractions import VibExperiment

import numpy as np

import logging
logger = logging.getLogger("wilson."+__name__)


def get_features_from_terms_for_eval(system: 'MolecularSystem',
                                     experiment: 'VibExperiment',
                          derived_terms: list['VibPerturbedTerm'],
                          props: list['MolecularProperty'],
                          spec_eval_setup: 'SpecEvalSetup' = None,
                          vib_ana_setup: 'VibAnaSetup' = None,
                          linkage_domains: str = None,
                          domain_distance_thresholds: dict = None) -> dict[ResLocGeoObject, SpectralFeature]:
    """
    From terms get spectral features.
    """
    
    # Initialize evaluation data
    vibstates_data, vibdiff_cache, data_and_configs = initialize_evaluation_data(
        system, experiment, vib_ana_setup, props
    )
    
    # Precalculate coefficient parts
    to_precalculate = identify_precalc_unique_coeff_parts(terms=derived_terms)
    precalculated = precalculate_unique_coeff_parts(
        need_to_precalc=to_precalculate,
        data_and_configs=data_and_configs
    )
    
    # Process resonance motifs
    motif_res_loc, terms_for_motifs = process_resonance_motifs(
        derived_terms, vibstates_data, vibdiff_cache
    )
    
    # Evaluate term coefficients
    term_coeffs_per_index = evaluate_terms(
        derived_terms, motif_res_loc, precalculated
    )
    
    # Get features to draw
    features_to_draw = get_features_to_draw(
        motif_res_loc, terms_for_motifs, term_coeffs_per_index, spec_eval_setup.ev_info.Gamma
    )
    return features_to_draw

# General term evaluator "as response function"
def terms_evaluator_general_compilation(system: 'MolecularSystem',
                                        experiment: 'VibExperiment',
                                        derived_terms: dict[int, dict[tuple, list['VibPerturbedTerm']]],
                                        props: list['MolecularProperty'],
                                        spec_eval_setup: 'SpecEvalSetup' = None,
                                        vib_ana_setup: 'VibAnaSetup' = None,
                                        domain_distance_thresholds: dict = None,
                                        do_diagn: bool = None) -> dict[ResLocGeoObject, 
                                                                       tuple[float, SpectralFeature]]:
    """
    Evaluate terms and generate spectral features.
    """
    diagn = {}

    # Initialize evaluation/precalculation data
    vibstates_data, vibdiff_cache, data_and_configs = initialize_evaluation_data(
        system, experiment, vib_ana_setup, props
    )

    # FIXME - make a function for a flat list
    terms_as_list = []
    for i in derived_terms:
        for j in derived_terms[i]:
            for t in derived_terms[i][j]:
                terms_as_list.append(t)

    # Precalculate coefficient parts
    to_precalculate = identify_precalc_unique_coeff_parts(terms=terms_as_list)
    precalculated = precalculate_unique_coeff_parts(
        need_to_precalc=to_precalculate,
        data_and_configs=data_and_configs
    )
    
    # Process resonance motifs
    motif_res_loc, terms_for_motifs = process_resonance_motifs(
        terms_as_list, vibstates_data, vibdiff_cache
    )

    # Evaluate term coefficients
    term_coeffs_per_index = evaluate_terms(
        terms_as_list, motif_res_loc, precalculated
    )
    
    # FIXME: maybe should be checked earlier
    if spec_eval_setup is None:
        raise ValueError('SpecEvalSetup is not set')

    # Get features to draw
    all_features = get_features_to_draw(
        motif_res_loc, terms_for_motifs, term_coeffs_per_index, spec_eval_setup.ev_info.Gamma
    )
    
    spec_window = spec_eval_setup.ev_info.spectral_window

    spec_window_with_features = SpectralFeature.filter_to_spec_window(all_features, spec_window)
    
    #### <--- SpectralEvaluator.evaluate_spectrum

    from .grid_manager_evaluator import SpectralEvaluator

    spec_evaluator = SpectralEvaluator(vibstates_data, vibdiff_cache, gamma=4.2) # FIXME gamma value type
    grid_values_all_domains = spec_evaluator.evaluate_spectrum(spec_window=spec_window_with_features, 
                                                               grid_resolution=spec_eval_setup.ev_info.grid_resolution, return_type='grid')

    # print('grid_values_all_domains\n',grid_values_all_domains, type( grid_values_all_domains))
    return grid_values_all_domains, diagn


def get_domain_grids():
    return


def evaluate_all_on_grids(grid_info_dict: dict['RectangularDomain', dict], 
                          vib_data, vibdiff_cache, gamma) -> np.ndarray:
    """
    grid_info_dict = {domain: {'indices': slices,
                               'grid': {'A': meshgrid, 'B': meshgrid}}, ...}

    parameter of res location/feature location

    returns:

    domains_result = {domain: evaluated grid (np.ndarray), ...}
    """

    for domain in grid_info_dict:
        domains_grids = grid_info_dict[domain]['grid']
        grid_info_dict[domain]['result'] = evaluate_domain(domain, domains_grids, 
                                                           vib_data, vibdiff_cache, gamma)
    return grid_info_dict


def evaluate_domain(domain: 'RectangularDomain', dom_subgrids: dict, 
                    vib_data, vibdiff_cache, gamma):
    """
    sum of evaluations for all features in this domain
    """
    domain_result = 0. + 0.j

    dom_all_feats = domain.full_features + domain.contrib_features
    for feature in dom_all_feats:
        compiled_groups = compile_feature(feature, vib_data, vibdiff_cache)
        
        domain_result += evaluate_feature_on_grid(compiled_groups, dom_subgrids, 
                                                  gamma=gamma, amplitude_coeff=feature.amplitude_coeff)
    return domain_result


def evaluate_res_motif_on_grid(compiled_res_motif: NumericalResonanceMotif,
                               meshgrids: dict[str, np.ndarray],
                               gamma: float):
    """
    compiled_res_motif: NumericalResonanceMotif - 1 resonanse motif for a ParameterSet

    Should work for a single specific compiled NumericalResonanceMotif, i.e., with a choice of ParameterSet, 
        and VibStatesData and VibDiffCache for states

    meshgrids - dictionary of axes labels with meshgrid np.ndarrays as values

    gamma - should generally be Gamma_{mn}, so a value for given chioce of vib_diff/normal modes/states
    """
    total = 1.
    for res_conds in compiled_res_motif.res_conds:

        pfreq = sum(meshgrids[ax] * res_conds.pf_dict[ax] for ax in res_conds.pf_dict)
        
        z = res_conds.vib_energy_diff - pfreq - 1j*gamma
        total *= 1. / z
    return total


def evaluate_feature_on_grid(compiled_groups: list[CompiledTermGroup],
                             meshgrids: dict[str, np.ndarray],
                             gamma: float,
                             amplitude_coeff: float):
    """
    
    """
    full = 0.

    for group in compiled_groups:
        for motif in group.resonance_motifs:
            full += evaluate_res_motif_on_grid(motif, meshgrids, gamma)

    return amplitude_coeff * full


def initialize_evaluation_data(system: 'MolecularSystem',
                               experiment: 'VibExperiment',
                               vib_ana_setup: 'VibAnaSetup',
                               props: list['MolecularProperty']) -> tuple[VibStatesData, VibDiffCache, EvaluationDataAndConfigs]:
    """
    Initialize data structures needed for term evaluation.
    """
    # har_states_labels = tuple([list(state.harm_quanta_coeffs.keys())[0][0] 
    #                          for state in vib_ana_setup.states if state.harmonic_WF])

    include_list = tuple([int(v[0]) for v in list(vib_ana_setup.nc_sqrt_eigval.keys()) if int(v[0]) not in vib_ana_setup.exclude_modes])
    vibstates_data = VibStatesData(allstates=tuple(vib_ana_setup.states), harmonic_osc_states_labels=include_list)
    
    vibdiff_cache = VibDiffCache()
    props = MolPropsCollection(properties=props)
    
    data_and_configs = EvaluationDataAndConfigs(props_data=props,
                                                vibstates_data=vibstates_data,
                                                number_of_nmodes=system.Nnmodes,
                                                nm_inds_choices=vib_ana_setup.modes_indices,
                                                pulse_polarization_vector=experiment.polarization_avg_vector)
    
    return vibstates_data, vibdiff_cache, data_and_configs

def process_resonance_motifs(derived_terms: list['VibPerturbedTerm'],
                            vibstates_data: VibStatesData,
                            vibdiff_cache: VibDiffCache) -> tuple[dict[ResonanceMotif, ResLocGeoObject], 
                                                                  dict[ResonanceMotif, list['VibPerturbedTerm']]]:
    """
    Process resonance motifs and find their locations.
    """
    unique_res_motifs = identify_unique_resmotifs(derived_terms)
    motif_res_loc: dict[ResonanceMotif, ResLocGeoObject] = {}
    
    for res_motif in unique_res_motifs:
        this_motif_res_locs = find_resonance_locations_wrt_index_choices(
            motif=res_motif,
            vibstates_data=vibstates_data,
            vibdiff_cache=vibdiff_cache,
            spec_window=None
        )
        motif_res_loc.update(this_motif_res_locs)
    
    terms_for_motifs: dict[ResonanceMotif, list[VibPerturbedTerm]] = {res_motif: [] for res_motif in unique_res_motifs}
    
    for vibterm in derived_terms:
        res_motif = ResonanceMotif(vibterm.res)
        for u_motif in unique_res_motifs:
            if u_motif == res_motif:
                terms_for_motifs[u_motif].append(vibterm)
                
    return motif_res_loc, terms_for_motifs

def evaluate_terms(derived_terms: list['VibPerturbedTerm'],
                  motif_res_loc: dict[ResonanceMotif, dict[ResLocGeoObject]],
                  precalculated: EvaluationDataAndConfigs) -> dict['VibPerturbedTerm', dict[ParameterSet, float]]:
    """
    Evaluate coefficients for all terms.
    """
    term_coeffs_per_index = {}
    for vibterm in derived_terms:
        res_motif = ResonanceMotif(vibterm.res)
        term_coeffs_per_index[vibterm] = evaluate_term_coeffs(
            term=vibterm,
            relevant_indices=[k for i in motif_res_loc[res_motif].values() for k in i],
            necessary_data=precalculated
        )
    
    return term_coeffs_per_index

def get_features_to_draw(motif_res_loc: dict[ResonanceMotif, dict[ResLocGeoObject, list]], 
                         terms_for_motifs: dict[ResonanceMotif, list['VibPerturbedTerm']], 
                         term_coeffs_per_index: dict['VibPerturbedTerm', 
                                                     dict[ParameterSet, float]],
                         lineshape_parameter: dict[str, float]) -> list[SpectralFeature]:
    """

    lineshape_parameter - uniform lineshape parameters (for all axes) for each feature for now
    """
    # a SpectralFeature instanse holds a res_location and list of states parameters that give this res_location; 
    #       the amplitude coefficient is a value in the dict
    features_to_draw: list[SpectralFeature] = []

    for res_motif in motif_res_loc:

        for res_geo_obj, list_state_dicts in motif_res_loc[res_motif].items():
            lst_params = tuple([ParameterSet(states_dict) for states_dict in list_state_dicts])
            list_to_sum = [term_coeffs_per_index[term][ParameterSet(states_dict)] for term in terms_for_motifs[res_motif] for states_dict in list_state_dicts]
            amplitude_coeff = sum(list_to_sum)
            
            # disregard locations where coefficient is zero
            if amplitude_coeff != 0.:
                spec_feature = SpectralFeature(location=res_geo_obj, 
                                            # term_contributions=tuple([TermParametersChoice(terms=tuple(terms_for_motifs[res_motif]),
                                            #                                         states_parameters=lst_params)]),
                                            term_contributions=tuple([TermParametersChoice(res_motif=res_motif,
                                                                                           states_parameters=lst_params, 
                                                                                           term_ids=tuple([t.h() for t in terms_for_motifs[res_motif]]) )]),
                                            lineshape_parameter=lineshape_parameter, # uniform lineshape parameters (for all axes) for each feature
                                            amplitude_coeff=amplitude_coeff)
                if spec_feature not in features_to_draw:
                    features_to_draw.append(spec_feature)
                else:
                    new_specfeat = spec_feature.union(features_to_draw[res_geo_obj][1])
                    features_to_draw.append(new_specfeat)

    return features_to_draw

