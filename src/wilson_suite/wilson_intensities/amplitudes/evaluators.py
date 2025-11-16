"""
Evaluator functions for WilsonSimulation
"""
from .numerical_abstractions import NumericalResonanceMotif
from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import RectangularDomain, ResLocGeoObject, SpectralFeature, SpectralWindow, Box
from ...wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
from ..utils import mainVibStates2arraydict, check_energy_unit, convNu2Ene
from ..amplitudes.full_amplitude_coeff import evaluate_term_coeffs, precalculate_unique_coeff_parts, identify_precalc_unique_coeff_parts
from ..amplitudes.resonances import find_resonance_locations_wrt_index_choices, identify_unique_resmotifs

from wilson_suite.wilson_main.abstractions import VibAnaSetup, MolecularSystem, MolPropsCollection
from wilson_suite.wilson_intensities.amplitudes.term_parts import ResonanceMotif, VibStatesData, ResonanceCondition
import wilson_suite.wilson_intensities.amplitudes.domains as domfuncs
from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
from wilson_suite.wilson_intensities.amplitudes.term_parts import (ResonanceMotif, VibStatesData, EvaluationDataAndConfigs, ParameterSet,
                                                                   TermParametersChoice)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wilson_suite.wilson_main.abstractions import VibAnaSetup, MolecularProperty
    from wilson_suite.wilson_main.spectrum_abstractions import SpecEvalSetup
    from wilson_suite.wilson_derive.abstractions import VibPerturbedTerm
    from wilson_suite.wilson_experiment.abstractions import VibExperiment

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
                                        derived_terms: list['VibPerturbedTerm'],
                                        props: list['MolecularProperty'],
                                        spec_eval_setup: 'SpecEvalSetup' = None,
                                        vib_ana_setup: 'VibAnaSetup' = None,
                                        domain_distance_thresholds: dict = None,
                                        do_diagn: bool = None) -> dict[ResLocGeoObject, 
                                                                       tuple[float, SpectralFeature]]:
    """
    Evaluate terms and generate spectral features.
    """
    # Initialize evaluation/precalculation data
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
    all_features = get_features_to_draw(
        motif_res_loc, terms_for_motifs, term_coeffs_per_index, spec_eval_setup.ev_info.Gamma
    )
    
    # print('\nall_features', all_features, '\n')

    spec_window = spec_eval_setup.ev_info.spectral_window
    # print('\nspec_window', spec_window)

    spec_window_with_features = SpectralFeature.filter_to_spec_window(all_features, spec_window)
    
    # print('\nspec_window_with_features', spec_window_with_features)
    feat_all = spec_window_with_features.full_features + spec_window_with_features.contrib_features
    domains = domfuncs.features_to_clusters(features=feat_all)
    formal_domains = [RectangularDomain(box=Box.union([f.feat_box for f in domains[d]])) for d in domains]
    
    for domain in formal_domains:
        domain.box = domain.box.intersect(spec_window_with_features.box)
    
    spec_grid = spec_window_with_features.sample_grid({'A': 10, 'B': 10})
    subgrids = domfuncs.cut_grid_with_indices_dict_nd(spec_grid, formal_domains)
    
    # modifies spec_grid dict
    domfuncs.insert_results_to_grid_nd(spec_grid, subgrids, result_func=lambda sg: sum(sg.values())) # upd result_func

    grid_values_all_domains = spec_grid['result']


    print('\n\nspec_eval_setup\n', spec_eval_setup)
    return grid_values_all_domains


def get_domain_grids():
    return

def evaluate_on_grid(grid_info_dict: dict['RectangularDomain', dict]) -> np.ndarray:
    """
    grid_info_dict = {domain: {'indices': slices,
                               'grid': {'A': meshgrid, 'B': meshgrid}}, ...}

    parameter of res location/feature location
    """
    domain_grids = grid_info_dict['grid']
    domain_result = 0. + 0.j

    for domain in grid_info_dict:
        domain_result += evaluate_domain_on_grid(domain, domain_grids)
    return

def evaluate_domain_on_grid(domain: 'RectangularDomain', domain_grid):
    return


import wilson_suite.wilson_intensities.amplitudes.vibene_differences as vediff

def evaluate_feature_on_grid(feature: SpectralFeature, 
                             meshgrids: dict[str, np.ndarray],
                             necessary_data: EvaluationDataAndConfigs) -> np.ndarray:

    # one res motif per a group of terms contributing (united by same res motif)
    # one res motif per TermParametersChoice

    term_contributions_rcs: dict[int, list] = {}
    states_parameters: dict[int, list] = {}

    for i, term_group in enumerate(feature.term_contributions):
        states_parameters[i] = []
        term_contributions_rcs[i] = []
        
        for st_params in term_group.states_parameters:
            states_parameters[i].append(st_params)

        for rc in term_group.res_motif:
            term_contributions_rcs[i].append(rc)


    # collect for each term_group_i (one motif group) results for each parameter set
    resonances_t: dict[int, list] = {}

    for term_group_i in states_parameters:
        # in one terms group
        term_rcs: list[ResonanceCondition] = term_contributions_rcs[term_group_i]
        
        resonances_t[term_group_i] = {}

        for params in states_parameters[term_group_i]:
            # for combination of parameters for this terms group
            
            rs_difs_num = []
            
            for rc in term_rcs:
                # for each resonace condition (rc) compute vibdiff part
                vd = vediff.VibDiff.from_symbolic(rc.diff, params, necessary_data.vibstates_data)
                vd.cache_it(necessary_data.vibdiff_cache)
                print('\nvd', vd)
                
                # for this resonance condition get grids for axes
                pfreqs = sum([meshgrids[ax.strip('-')] * rc.pf_dict[ax.strip('-')] for ax in rc.pf])

                # in reciprocal centimeters
                rc_num = vd.energy_difference(au=False) - pfreqs - 1j*feature.lineshape_parameter_single
                
                # in au
                rs_difs_num.append(convNu2Ene(rc_num))

            from functools import reduce
            import operator

            product = reduce(operator.mul, rs_difs_num)
            resonances_t[term_group_i][params] = 1./product

    # now sum up all results with their coeffs
    full_result = 0. + 0.j
    for term_group_i in resonances_t:
        this_motif_result = 0. + 0.j
        for param_set in resonances_t[term_group_i]:
            this_motif_result += resonances_t[term_group_i][param_set]
        full_result += this_motif_result
    # grid multiplied with the total coefficient here
    return full_result * feature.amplitude_coeff

from .term_parts import EvalFeature

def evaluate_eval_feature_on_grid(
    feature: EvalFeature,
    meshgrid: dict[str, np.ndarray],
) -> np.ndarray:
    
    result = 0
    for term in feature.terms:
        term_val = term.resonance_function(meshgrid)
        result += term.prefactor * term_val
    return result * feature.amplitude

def evaluate_resonance_on_grid(compiled: NumericalResonanceMotif,
                               meshgrids: dict[str, np.ndarray],
                               gamma: float):
    total = 0
    for res_conds in compiled.res_conds:
        pfreq = sum(meshgrids[ax] * res_conds.pf_dict[ax] for ax in res_conds.pf_dict)
        z = res_conds.vib_energy_diff - pfreq - 1j*gamma
        total += 1. / z
    return total


def initialize_evaluation_data(system: 'MolecularSystem',
                               experiment: 'VibExperiment',
                               vib_ana_setup: 'VibAnaSetup',
                               props: list['MolecularProperty']) -> tuple[VibStatesData, VibDiffCache, EvaluationDataAndConfigs]:
    """
    Initialize data structures needed for term evaluation.
    """
    har_states_labels = tuple([list(state.harm_quanta_coeffs.keys())[0][0] 
                             for state in vib_ana_setup.states if state.harmonic_WF])
    
    vibstates_data = VibStatesData(
        allstates=tuple(vib_ana_setup.states), 
        harmonic_osc_states_labels=har_states_labels
    )
    
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
                                            term_contributions=tuple([TermParametersChoice(terms=tuple(terms_for_motifs[res_motif]),
                                                                                    states_parameters=lst_params)]),
                                            lineshape_parameter=lineshape_parameter, # uniform lineshape parameters (for all axes) for each feature
                                            amplitude_coeff=amplitude_coeff)
                if spec_feature not in features_to_draw:
                    features_to_draw.append(spec_feature)
                else:
                    new_specfeat = spec_feature.union(features_to_draw[res_geo_obj][1])
                    features_to_draw.append(new_specfeat)

    return features_to_draw

