import wilson_suite.wilson_intensities.amplitudes.evaluators as evaluators #terms_evaluator_general, terms_evaluator_general_compilation
from wilson_suite.wilson_intensities.amplitudes import domains as domfuncs
from ...amplitudes.numerical_abstractions import NumericalResonanceCondition, NumericalResonanceMotif
from ....wilson_main import abstractions as wm_abst
from ....wilson_derive.response_terms import VibPerturbedTerm
import numpy as np

def generate_props_data_Nmodes(N_modes):
    return {'dipgrad': np.ones((N_modes, 3)), 
            'diphess': np.zeros((N_modes, N_modes, 3)),
            'polgrad': np.zeros((N_modes, 3, 3)), 
            'polhess': np.zeros((N_modes, N_modes, 3, 3)),
            'cff':     np.ones((N_modes, N_modes, N_modes)),
            'qff':     np.zeros((N_modes, N_modes, N_modes, N_modes)),
            }

def prep_vibanasetup_with_degen_states():
    # vib_ana_setup needs to have vibstates
    vibana = wm_abst.VibAnaSetup()
    vibana.setStates((
                        wm_abst.VibState(harm_quanta_coeffs={(0,):1.}, state_label='0', energy=964., harmonic_WF=True),
                        wm_abst.VibState(harm_quanta_coeffs={(1,):1.}, state_label='1', energy=1234., harmonic_WF=True),
                        wm_abst.VibState(harm_quanta_coeffs={(2,):1.}, state_label='2', energy=1234., harmonic_WF=True),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 0):1.}, state_label='0,0', energy=1864., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 1):1.}, state_label='0,1', energy=2255., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(1, 1):1.}, state_label='1,1', energy=2368., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(1, 2):1.}, state_label='1,2', energy=2360., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(2, 2):1.}, state_label='2,2', energy=2362., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 2):1.}, state_label='0,2', energy=2274., harmonic_WF=False),
                        # Adding 3-quanta states:
                        wm_abst.VibState(harm_quanta_coeffs={(0, 0, 0):1.}, state_label='0,0,0', energy=2685., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(1, 1, 1):1.}, state_label='1,1,1', energy=3581., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(2, 2, 2):1.}, state_label='2,2,2', energy=3690., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 1, 2):1.}, state_label='0,1,2', energy=3742., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 1, 1):1.}, state_label='0,1,1', energy=3318., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 2, 2):1.}, state_label='0,2,2', energy=3680., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 0, 1):1.}, state_label='0,0,1', energy=3155., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(0, 0, 2):1.}, state_label='0,0,2', energy=3498., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(1, 1, 2):1.}, state_label='1,1,2', energy=3594., harmonic_WF=False),
                        wm_abst.VibState(harm_quanta_coeffs={(1, 2, 2):1.}, state_label='1,2,2', energy=3642., harmonic_WF=False),
                    ))
    
    return vibana


def props_with_values(system: wm_abst.MolecularSystem):
    props_data = generate_props_data_Nmodes(system.Nnmodes)
    # cart axes (0, 1, 1, 0) - 0 1 2 3
    props_data['dipgrad'][0, 1] = 0.45
    props_data['dipgrad'][1, 0] = 0.45
    props_data['polgrad'][0, 0, 0] = 0.3
    props_data['polgrad'][1, 1, 0] = 0.3
    props_data['polgrad'][0, 1, 0] = 0.3
    props_data['polhess'][0, 0, 0, 0] = 0.15
    props_data['diphess'][0, 0, 1] = 0.15
    
    props_data['cff'][0, 1, 1] = 0.7
    
    props = []
    
    # props needs to have properties tensors, values    
    for trname in props_data:
        p = wm_abst.MolecularProperty(prop_spec={}, trivial_name=trname, vals=props_data[trname])
        p.addValues(values=props_data[trname])
        props.append(p)
    
    return props


def test_terms_evaluator_general_compilation():
    print()
    from .test_domains import get_data_evaluators_tests
    datadict = get_data_evaluators_tests()

    evaluators.terms_evaluator_general_compilation(**datadict)


def get_necessary_data(terms_select: list[VibPerturbedTerm]):
    import wilson_suite.wilson_intensities.amplitudes.full_amplitude_coeff as fac
    from wilson_suite.wilson_main.abstractions import MolPropsCollection, MolecularProperty, VibState
    from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, VibStatesData, EvaluationDataAndConfigs
    from .test_full_coeff import generate_props_data4modes
    props_data = generate_props_data4modes()
    # cart axes (0, 1, 1, 0) - 0 1 2 3
    props_data['dipgrad'][0, 1] = 0.45
    props_data['dipgrad'][1, 0] = 0.45
    props_data['polgrad'][0, 0, 0] = 0.3
    props_data['polgrad'][1, 1, 0] = 0.3
    props_data['polgrad'][0, 1, 0] = 0.3
    props_data['polhess'][0, 0, 0, 0] = 0.15
    props_data['diphess'][0, 0, 1] = 0.15
    
    props_data['cff'][0, 1, 1] = 0.7

    props = []
    for trname in props_data:
        p = MolecularProperty(prop_spec={}, trivial_name=trname, vals=props_data[trname])
        p.addValues(values=props_data[trname])
        props.append(p)

    vibdata = VibStatesData(allstates=(VibState(harm_quanta_coeffs={(0,):1.}, state_label='0', energy=964., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(1,):1.}, state_label='1', energy=1234., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(2,):1.}, state_label='2', energy=3644., harmonic_WF=True),
                                       VibState(harm_quanta_coeffs={(0, 1, 1):1.}, state_label='0,1,1', energy=3318., harmonic_WF=False),
                                       ),
                                   harmonic_osc_states_labels=(0, 1, 2))

    need_to_precalc = fac.identify_precalc_unique_coeff_parts(terms_select)

    from .test_domains import get_data_evaluators_tests
    datadict = get_data_evaluators_tests()

    data_and_configs = EvaluationDataAndConfigs(props_data=props,
                                                vibstates_data=vibdata,
                                                number_of_nmodes=4,
                                                nm_inds_choices=[0,1],
                                                pulse_polarization_vector=datadict['experiment'].polarization_avg_vector)


    results = fac.precalculate_unique_coeff_parts(need_to_precalc=need_to_precalc,
                                                  data_and_configs=data_and_configs)
    return results


def test_evaluate_feature_on_grid():
    print()
    from .test_domains import get_features_from_terms, get_data_evaluators_tests
    datadict = get_data_evaluators_tests()
    
    features = get_features_from_terms()

    spec_window = datadict['spec_eval_setup'].ev_info.spectral_window

    from ...amplitudes.spectrum_composition import SpectralFeature, Box, RectangularDomain
    spec_window_with_features = SpectralFeature.filter_to_spec_window(features, spec_window)

    feat_all = spec_window_with_features.full_features + spec_window_with_features.contrib_features
    domains = domfuncs.features_to_clusters(features=feat_all)
    domains_in_window = [RectangularDomain(box=Box.union([f.feat_box for f in domains[d]]), full_features=domains[d]) for d in domains]
    
    
    domain3 = domains_in_window[3]
    d3_all_feats = domain3.full_features + domain3.contrib_features

    coords_vectors, spec_grid = spec_window_with_features.sample_grid({'A': 10, 'B': 10})

    domains_with_subgrids = domfuncs.cut_grid_to_domains_nd(full_meshgrids=spec_grid, 
                                                             axis_coords=coords_vectors,
                                                             domains=domains_in_window)
    

    from ...amplitudes.term_parts import EvaluationDataAndConfigs, VibStatesData
    from ...amplitudes.vibene_differences import VibDiffCache
    from wilson_suite.wilson_main.abstractions import VibAnaSetup

    from ...amplitudes.numerical_abstractions import compile_feature

    vas: VibAnaSetup = datadict['vib_ana_setup']
    dd = EvaluationDataAndConfigs(vibstates_data=VibStatesData(allstates=vas.states),
                                  vibdiff_cache=VibDiffCache())
    

    compiled_groups = compile_feature(
        d3_all_feats[0],
        dd.vibstates_data,
        dd.vibdiff_cache
    )
    final = evaluators.evaluate_feature_on_grid(compiled_groups=compiled_groups, 
                                                meshgrids=domains_with_subgrids[domain3]['grid'],
                                                gamma=2.0,
                                                amplitude_coeff=d3_all_feats[0].amplitude_coeff)
    print(final)

def test_evaluate_feature_on_grid_new():
    from ...amplitudes.evaluators import evaluate_feature_on_grid
    from ...amplitudes.numerical_abstractions import CompiledTermGroup
    # Simple 1D grid for "A"
    A = np.array([0.0, 1.0, 2.0])
    mesh = {"A": A}

    # gamma = 0 for simplicity
    gamma = 0.0

    # Motif 1:
    # z = 10 - 1*x
    rc1 = NumericalResonanceCondition({"A": 1.0}, vib_energy_diff=10.0)
    motif1 = NumericalResonanceMotif([rc1])

    # Motif 2:
    # z = 20 - 2*x
    rc2 = NumericalResonanceCondition({"A": 2.0}, vib_energy_diff=20.0)
    motif2 = NumericalResonanceMotif([rc2])

    # One term group containing both motifs
    group = CompiledTermGroup([motif1, motif2])

    amplitude = 1.0

    # run evaluation
    result = evaluate_feature_on_grid([group], mesh, gamma, amplitude)

    # expected:
    expected = 1/(10 - A) + 1/(20 - 2*A)

    assert np.allclose(result, expected)


def test_evaluate_feature_on_grid_new_2():
    """
    gamma nonzero, two resonance conditions per motif
    """
    from ...amplitudes.evaluators import evaluate_feature_on_grid
    from ...amplitudes.numerical_abstractions import CompiledTermGroup
    # Simple 1D grid for "A"
    A = np.array([0.0, 1.0, 2.0])
    mesh = {"A": A}

    # gamma
    gamma = 0.5

    # Motif 1:
    # z = 10 - 1*x
    rc1 = NumericalResonanceCondition({"A": 1.0}, vib_energy_diff=10.0)
    # z = 15 - 0.5*x
    rc2 = NumericalResonanceCondition({"A": 0.5}, vib_energy_diff=15.0)
    motif1 = NumericalResonanceMotif([rc1, rc2])

    # Motif 2:
    # z = 20 - 2*x
    rc3 = NumericalResonanceCondition({"A": 2.0}, vib_energy_diff=20.0)
    # z = 25 - 1*x
    rc4 = NumericalResonanceCondition({"A": 1.0}, vib_energy_diff=25.0)
    motif2 = NumericalResonanceMotif([rc3, rc4])

    # One term group containing both motifs
    group = CompiledTermGroup([motif1, motif2])

    amplitude = 1.0

    # run evaluation
    result = evaluate_feature_on_grid([group], mesh, gamma, amplitude)

    # expected:
    expected = 1/((10 - A) - 1j*gamma) / ((15 - 0.5*A) - 1j*gamma) + 1/((20 - 2*A) - 1j*gamma) / ((25 - 1*A) - 1j*gamma)
    assert np.allclose(result, expected)



def test_evaluate_resonance_simple():
    # 1. Make a compiled motif with one term
    motif = NumericalResonanceMotif(res_conds=[
        NumericalResonanceCondition(
            pf_dict={"A": 1, "B": -1},
            vib_energy_diff=10.0,
        )
    ])

    # 2. Make a simple 1D grid
    A = np.array([0.0, 1.0, 2.0])
    B = np.array([3.0, 1.0, 2.0])
    mesh = {"A": A, "B": B}

    # 3. Evaluate with some gamma
    gamma = 0.5
    result = evaluators.evaluate_res_motif_on_grid(motif, mesh, gamma)

    # 4. Expected output (do by hand)
    expected = 1 / (10 - A + B - 1j*gamma)

    # 5. Assert correctness
    np.testing.assert_allclose(result, expected)

def test_evaluate_resonance_simple_2conds():
    # 1. Make a compiled motif with one term
    motif = NumericalResonanceMotif(res_conds=[
        NumericalResonanceCondition(
            pf_dict={"A": 1, "B": -1},
            vib_energy_diff=10.0,
        ),
        NumericalResonanceCondition(
            pf_dict={"A": 1},
            vib_energy_diff=10.0,
        )
    ])

    # 2. Make a simple 1D grid
    A = np.array([0.0, 1.0, 2.0])
    B = np.array([3.0, 1.0, 2.0])
    mesh = {"A": A, "B": B}

    # 3. Evaluate with some gamma
    gamma = 0.5
    result = evaluators.evaluate_res_motif_on_grid(motif, mesh, gamma)

    # 4. Expected output (do by hand)
    expected = 1 / (10 - A + B - 1j*gamma) / (10 - A - 1j*gamma)

    # 5. Assert correctness
    np.testing.assert_allclose(result, expected)

