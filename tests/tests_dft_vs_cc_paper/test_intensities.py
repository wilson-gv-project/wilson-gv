"""
- [ ] single feature intensities (at resonance)
    - [ ] several terms contribution (they would always share the same resonance motif):
        - [ ] 2 mech terms
        - [ ] mech and el term?? - idk if possible
- [x] 1. 5x5 grid around feature centrepoint with single term contribution
- [x] 2. composite grid including 5x5 grid around feature centrepoints for 2 non-overlapping features | this isn't much different from single feature but still different because grids are generated here
- [x] 3. composite grid including 5x5 grid around feature centrepoints for 2 overlapping features (2 cases: 1) same sign coeff, 2) diff sign coeff)
- [ ] gamma = 0 case
- [ ] feature located exactly on a grid boundary
- [ ] negative tests - ensure there is no silent errors
- [ ]
"""
from .testutils import MakeObjects
import wilson_suite as ws
import numpy as np
np.set_printoptions(linewidth=180, precision=3)


def test_single_feat_5by5grid_as_evaluate_regions():
    """
    Testing intensity evaluation for a single feature.
    Ultimately, evaluation is done with `evaluate_regions()` function: needs a list[GridRegion].
    
    This test was constracted with SpectralFeature centrepoint being the staring point:
        - assumes a spectrum with axes A and B (both A > 0 and B > 0)
        - vib states data and resonance motif was adjusted to the resonance location and these axes

    Need to have:
    - a single feature with its box
    - a spectral window with 5x5 grid around the centrepoint of the feature - feature location
    - grid_coords dict for this 5x5 grid
    - feature is dressed with its box, but that is not used since there is only one feature, 
        and evaluation is done simply over the whole SpectralWindow (5x5 - this given by custom here grid)
    """
    
    feature = MakeObjects.mk_feature_single()
    # just confirming here and for the information
    assert feature.amplitude_coeff == -1.12e-06
    assert feature.location == ws.intensities.amplitudes.spectrum_composition.ResLocGeoObject({'A': 1119.5, 'B': 2921.})
    
    term_contrib = feature.term_contributions[0]
    # just confirming here and for the information
    assert term_contrib.states_parameters[0] == ws.intensities.amplitudes.term_parts.ParameterSet({'a': 0, 'b': 1})
    
    # '-A' so that with (vibdiff - pf = 0), and (-pf > 0) ==> -(-A) and then 'A' can be a positive value (as set now in ResLocGeoObject)
    # in EVV axis A in this case would be (1,) and not (-1) --- there should be no issue with such interpretation 
    #                                                           because it has nothing to do with the pulse direction or sign, 
    #                                                           it is purely mathematical transformation
    assert set(i.pf for i in term_contrib.res_motif) == set([('-A',), ('B',)])

    rcs = [ws.intensities.amplitudes.term_parts.ResonanceCondition.make_from_tuples(left_state=(), right_state=('a',), pert_freqs=('-A',)).h(),
           ws.intensities.amplitudes.term_parts.ResonanceCondition.make_from_tuples(left_state=('a', 'b'), right_state=('b',), pert_freqs=('B',)).h()]
    assert sorted(rcs) == sorted([i.h() for i in term_contrib.res_motif])


    spec_wind_box = ws.intensities.amplitudes.spectrum_composition.Box({'A': (998.5, 1240.5), 
                                                                        'B': (2800., 3022.)})

    # step is 60.5
    grid_coords = {'A': np.array([[998.5 , 998.5 , 998.5 , 998.5 , 998.5],
                                  [1059.0, 1059.0, 1059.0, 1059.0, 1059.0],
                                  [1119.5, 1119.5, 1119.5, 1119.5, 1119.5],
                                  [1180.0, 1180.0, 1180.0, 1180.0, 1180.0],
                                  [1240.5, 1240.5, 1240.5, 1240.5, 1240.5]]), 
                   'B': np.array([[2800., 2860.5, 2921.0, 2961.5, 3022.], 
                                  [2800., 2860.5, 2921.0, 2961.5, 3022.],
                                  [2800., 2860.5, 2921.0, 2961.5, 3022.],
                                  [2800., 2860.5, 2921.0, 2961.5, 3022.],
                                  [2800., 2860.5, 2921.0, 2961.5, 3022.]])}
    
    spec_window = ws.intensities.amplitudes.spectrum_composition.SpectralWindow(box=spec_wind_box)
    spec_window.full_features = [feature]

    grid_region = MakeObjects.mk_gridregion_for_specwindow(spec_window, grid_coords)
    assert grid_region.domain.full_features == spec_window.full_features
    
    from wilson_suite.wilson_intensities.tests.unit.test_domains import get_data_evaluators_tests
    vbana = get_data_evaluators_tests()['vib_ana_setup']

    vibstates_data = ws.intensities.amplitudes.term_parts.VibStatesData(allstates=MakeObjects.mk_vibstates_states(), 
                                                                        harmonic_osc_states_labels=vbana.include_list,
                                                                        number_of_nmodes=vbana.number_of_modes)
    assert vibstates_data.get_state_by_label('0').energy == 1119.5
    assert vibstates_data.get_state_by_label('0,1').energy == 3885.

    from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
    vibdiff_cache = VibDiffCache()
    
    # regions evaluation - here a single region that matches the whole SpectralWindow in this case
    # at this point all units should be au in evaluations (including gamma and coordinates of the spec grids)
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    res = ws.intensities.amplitudes.evaluation_wf.evaluate_regions(regions=[grid_region], # holds grid coordinates
                                                                   vib_data=vibstates_data, 
                                                                   vibdiff_cache=vibdiff_cache, 
                                                                   gamma=convNu2Ene(feature.lineshape_parameter),
                                                                   verbose=True)
    regions_eval = list(res.values())[0]
    assert np.count_nonzero(np.isreal(regions_eval)) == 1 # only centrepoint is at resonance - imaginary part is zero
    # centrepoint intensity vs refference
    assert regions_eval[2,2] == feature.amplitude_coeff/(-1j*convNu2Ene(feature.lineshape_parameter))**2

    # now down to the feature evaluation
    grid_coords_au = {k: convNu2Ene(v) for k,v in grid_coords.items()}
    feat_eval = ws.intensities.amplitudes.evaluation_wf.evaluate_feature(feature=feature,
                                                                         vib_data=vibstates_data,
                                                                         vibdiff_cache=vibdiff_cache,  # won't be empty now but it's fine
                                                                         gamma=convNu2Ene(feature.lineshape_parameter),
                                                                         coords=grid_coords_au, # extra wrt evaluate_regions because features should be evaluated on regions/grids
                                                                         verbose=True)
    assert np.count_nonzero(np.isreal(feat_eval)) == 1 # only centrepoint is at resonance - imaginary part is zero

    assert np.allclose(feat_eval, regions_eval)

    # reference value construction 
    # amplitude = feat_coeff  / (w_0,a + A) / (w_a+b,b - B) with a=0,b=1
    assert vibdiff_cache._cache == {('zero', '0'): -1119.5, ('0,1', '1'): 2921.0}
    resonance_part = 1. / (convNu2Ene(-1119.5) + grid_coords_au['A'] - 1j* convNu2Ene(feature.lineshape_parameter)) / (convNu2Ene(2921.0) - grid_coords_au['B'] - 1j* convNu2Ene(feature.lineshape_parameter))
    ref_res = feature.amplitude_coeff * resonance_part
    
    assert np.allclose(ref_res, feat_eval)


def test_2_nonoverl_feat_5by5grid_as_evaluate_regions():
    """
    There are 2 non-overlapping features being evaluated.
    Grids and regions are craated via GridManager.
    Evaluation of features is done on those grids, so all points are off-resonances.
    """

    features = MakeObjects.mk_features_non_ovrl()
    # just confirming here and for the information
    assert features[0].amplitude_coeff == -4.32e-06
    assert features[0].location == ws.intensities.amplitudes.spectrum_composition.ResLocGeoObject({'A': 1119.5, 'B': 2921.})
    
    term_contrib0 = features[0].term_contributions[0]
    # just confirming here and for the information
    assert term_contrib0.states_parameters[0] == ws.intensities.amplitudes.term_parts.ParameterSet({'a': 0, 'b': 1})

    assert features[1].amplitude_coeff == -1.12e-05
    assert features[1].location == ws.intensities.amplitudes.spectrum_composition.ResLocGeoObject({'A': 964., 'B': 270.})
    
    term_contrib1 = features[1].term_contributions[0]
    # just confirming here and for the information
    assert term_contrib1.states_parameters[0] == ws.intensities.amplitudes.term_parts.ParameterSet({'a': 1, 'b': 2})

    assert set(i.pf for i in term_contrib0.res_motif) == set([('-A',), ('B',)])
    assert set(i.pf for i in term_contrib1.res_motif) == set([('-A',), ('B',)])
    
    # do the boxes of these features overlap? boxes constructed with SpectralFeature.dress_these_with_boxes()
    assert not features[0].feat_box.overlaps(features[1].feat_box)


    # now need a bigger grid to include 2 features that don't overlap
    # will use now grid_manager
    box0 = ws.intensities.amplitudes.spectrum_composition.Box({'A': (998.5, 1240.5), 
                                                               'B': (2800., 3022.)})
    box1 = ws.intensities.amplitudes.spectrum_composition.Box({'A': (843.0, 1085.0), 
                                                               'B': (149.0, 391.0)})

    union_box = ws.intensities.amplitudes.spectrum_composition.Box.union([box0, box1])

    assert union_box.bounds == {'A': (843.0, 1240.5), 'B': (149.0, 3022.0)}

    spec_window = ws.intensities.amplitudes.spectrum_composition.SpectralWindow(box=union_box)
    spec_window.full_features = features
    
    # grid prepared via GridManager
    from wilson_suite.wilson_intensities.amplitudes.grid_manager_evaluator import GridManager
    grid_manager = GridManager(spec_window)
    grid_manager.make_fullgrid({'A': 50, 'B': 90})
    assert grid_manager.full_grid['A'].shape == grid_manager.full_grid['B'].shape == (50, 90)

    regions = grid_manager.create_regions()
    assert len(regions) == 2
    assert regions[0].domain.box == ws.intensities.amplitudes.spectrum_composition.Box({'A': (1089.5, 1149.5), 'B': (2891.0, 2951.0)})
    assert regions[1].domain.box == ws.intensities.amplitudes.spectrum_composition.Box({'A': (934.0, 994.0), 'B': (240.0, 300.0)})

    from wilson_suite.wilson_intensities.tests.unit.test_domains import get_data_evaluators_tests
    vbana = get_data_evaluators_tests()['vib_ana_setup']

    vibstates_data = ws.intensities.amplitudes.term_parts.VibStatesData(allstates=MakeObjects.mk_vibstates_states(), 
                                                                        harmonic_osc_states_labels=vbana.include_list,
                                                                        number_of_nmodes=vbana.number_of_modes)
    assert vibstates_data.get_state_by_label('0').energy == 1119.5
    assert vibstates_data.get_state_by_label('1').energy == 964.
    assert vibstates_data.get_state_by_label('2').energy == 1234.
    assert vibstates_data.get_state_by_label('0,1').energy == 3885.

    from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
    vibdiff_cache = VibDiffCache()
    
    # regions evaluation - here a single region that matches the whole SpectralWindow in this case
    # at this point all units should be au in evaluations (including gamma and coordinates of the spec grids)
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    res = ws.intensities.amplitudes.evaluation_wf.evaluate_regions(regions=regions, # holds grid coordinates
                                                                   vib_data=vibstates_data, 
                                                                   vibdiff_cache=vibdiff_cache, 
                                                                   gamma=convNu2Ene(features[0].lineshape_parameter), # btw, assuming features have uniform lineshape parameter
                                                                   verbose=True)

    region0_eval = list(res.values())[0]
    region1_eval = list(res.values())[1]

    # reference value construction 
    # amplitude = feat_coeff  / (w_0,a + A) / (w_a+b,b - B) with a=0,b=1
    assert vibdiff_cache._cache == {('zero', '0'): -1119.5, ('0,1', '1'): 2921.0, ('zero', '1'): -964.0, ('1', '2'): 270.0}

    resonance_part_0 = 1. / (convNu2Ene(-1119.5 + regions[0].coords['A']) - 1j* convNu2Ene(features[0].lineshape_parameter)) / (convNu2Ene(2921.0 - regions[0].coords['B']) - 1j* convNu2Ene(features[0].lineshape_parameter))
    resonance_part_1 = 1. / (convNu2Ene(-964. + regions[1].coords['A']) - 1j* convNu2Ene(features[0].lineshape_parameter)) / (convNu2Ene(270.0 - regions[1].coords['B']) - 1j* convNu2Ene(features[0].lineshape_parameter))

    ref_res0 = features[0].amplitude_coeff * resonance_part_0
    ref_res1 = features[1].amplitude_coeff * resonance_part_1
    
    assert np.allclose(ref_res0, region0_eval)
    assert np.allclose(ref_res1, region1_eval)


def test_2_overl_feat_5by5grid_as_evaluate_regions():
    """
    There are 2 overlapping(boxes) features being evaluated.
    Grids and regions are craated via GridManager.
    Evaluation of features is done on those grids, so all points are off-resonances.
    """

    features = MakeObjects.mk_features_ovrl()

    # just confirming here and for the information
    assert features[0].amplitude_coeff == -4.32e-06
    assert features[0].location == ws.intensities.amplitudes.spectrum_composition.ResLocGeoObject({'A': 1000., 'B': 520.})
    
    term_contrib0 = features[0].term_contributions[0]
    # just confirming here and for the information
    assert term_contrib0.states_parameters[0] == ws.intensities.amplitudes.term_parts.ParameterSet({'a': 0, 'b': 1})

    assert features[1].amplitude_coeff == -1.12e-05
    assert features[1].location == ws.intensities.amplitudes.spectrum_composition.ResLocGeoObject({'A': 1000., 'B': 500.})
    
    term_contrib1 = features[1].term_contributions[0]
    # just confirming here and for the information
    assert term_contrib1.states_parameters[0] == ws.intensities.amplitudes.term_parts.ParameterSet({'a': 0, 'b': 1})

    assert set(i.pf for i in term_contrib0.res_motif) == set([('-A',), ('B',)])
    assert set(i.pf for i in term_contrib1.res_motif) == set([('-A',), ('B',)])
    
    # do the boxes of these features overlap? boxes constructed with SpectralFeature.dress_these_with_boxes()
    assert features[0].feat_box.overlaps(features[1].feat_box)

    # now need a bigger grid to include 2 features that don't overlap
    # will use now grid_manager
    box0 = ws.intensities.amplitudes.spectrum_composition.Box({'A': (879.0, 1121.0),
                                                               'B': (399.0, 641.0)})
    box1 = ws.intensities.amplitudes.spectrum_composition.Box({'A': (879.0, 1121.0), 
                                                               'B': (379.0, 621.0)})

    union_box = ws.intensities.amplitudes.spectrum_composition.Box.union([box0, box1])
    assert union_box.bounds == {'A': (879.0, 1121.0), 'B': (379.0, 641.0)}

    spec_window = ws.intensities.amplitudes.spectrum_composition.SpectralWindow(box=union_box)
    spec_window.full_features = features
    
    # grid prepared via GridManager
    from wilson_suite.wilson_intensities.amplitudes.grid_manager_evaluator import GridManager
    grid_manager = GridManager(spec_window)
    grid_manager.make_fullgrid({'A': 50, 'B': 90})
    assert grid_manager.full_grid['A'].shape == grid_manager.full_grid['B'].shape == (50, 90)

    regions = grid_manager.create_regions()
    # just one because boxes overlap and they form a domain/grid region
    assert len(regions) == 1

    assert regions[0].domain.box == ws.intensities.amplitudes.spectrum_composition.Box({'A': (970.0, 1030.0), 'B': (470.0, 550.0)})


    from wilson_suite.wilson_intensities.tests.unit.test_domains import get_data_evaluators_tests
    vbana = get_data_evaluators_tests()['vib_ana_setup']

    vibstates_data = ws.intensities.amplitudes.term_parts.VibStatesData(allstates=MakeObjects.mk_vibstates_states(), 
                                                                        harmonic_osc_states_labels=vbana.include_list,
                                                                        number_of_nmodes=vbana.number_of_modes)
    assert vibstates_data.get_state_by_label('0').energy == 1119.5
    assert vibstates_data.get_state_by_label('1').energy == 964.
    assert vibstates_data.get_state_by_label('2').energy == 1234.
    assert vibstates_data.get_state_by_label('0,1').energy == 3885.

    from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
    vibdiff_cache = VibDiffCache()
    
    # regions evaluation - here a single region that matches the whole SpectralWindow in this case
    # at this point all units should be au in evaluations (including gamma and coordinates of the spec grids)
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    res = ws.intensities.amplitudes.evaluation_wf.evaluate_regions(regions=regions, # holds grid coordinates
                                                                   vib_data=vibstates_data, 
                                                                   vibdiff_cache=vibdiff_cache, 
                                                                   gamma=convNu2Ene(features[0].lineshape_parameter), # btw, assuming features have uniform lineshape parameter
                                                                   verbose=True)

    region0_eval = list(res.values())[0]

    # reference value construction 
    # amplitude = feat_coeff  / (w_0,a + A) / (w_a+b,b - B) with a=0,b=1
    assert vibdiff_cache._cache == {('zero', '0'): -1119.5, ('0,1', '1'): 2921.0, ('0', '1'): -155.5}

    # since 2 features share the GridRegion, they both get evaluated over its grid and summed
    resonance_part_0 = 1. / (convNu2Ene(-1119.5 + regions[0].coords['A']) - 1j* convNu2Ene(features[0].lineshape_parameter)) / (convNu2Ene(2921.0 - regions[0].coords['B']) - 1j* convNu2Ene(features[0].lineshape_parameter))

    resonance_part_1 = 1. / (convNu2Ene(-1119.5 + regions[0].coords['A']) - 1j* convNu2Ene(features[0].lineshape_parameter)) / (convNu2Ene(-155.5 - regions[0].coords['B']) - 1j* convNu2Ene(features[0].lineshape_parameter))

    ref_res0 = features[0].amplitude_coeff * resonance_part_0
    ref_res1 = features[1].amplitude_coeff * resonance_part_1
    
    assert np.allclose(ref_res0 + ref_res1, region0_eval)


def test_2posneg_overl_feat_5by5grid_as_evaluate_regions():
    """
    There are 2 overlapping(boxes) features being evaluated - one coeff is positive, one - negative
    Grids and regions are craated via GridManager.
    Evaluation of features is done on those grids, so all points are off-resonances.
    """

    features = MakeObjects.mk_features_ovrl()
    
    # upd coefficients for this test
    features[0].amplitude_coeff = -4.32e-05
    features[1].amplitude_coeff = 1.12e-05

    # just confirming here and for the information
    assert features[0].amplitude_coeff == -4.32e-05
    assert features[0].location == ws.intensities.amplitudes.spectrum_composition.ResLocGeoObject({'A': 1000., 'B': 520.})
    
    term_contrib0 = features[0].term_contributions[0]
    # just confirming here and for the information
    assert term_contrib0.states_parameters[0] == ws.intensities.amplitudes.term_parts.ParameterSet({'a': 0, 'b': 1})

    assert features[1].amplitude_coeff == 1.12e-05
    assert features[1].location == ws.intensities.amplitudes.spectrum_composition.ResLocGeoObject({'A': 1000., 'B': 500.})
    
    term_contrib1 = features[1].term_contributions[0]
    # just confirming here and for the information
    assert term_contrib1.states_parameters[0] == ws.intensities.amplitudes.term_parts.ParameterSet({'a': 0, 'b': 1})

    assert set(i.pf for i in term_contrib0.res_motif) == set([('-A',), ('B',)])
    assert set(i.pf for i in term_contrib1.res_motif) == set([('-A',), ('B',)])
    
    # do the boxes of these features overlap? boxes constructed with SpectralFeature.dress_these_with_boxes()
    assert features[0].feat_box.overlaps(features[1].feat_box)

    # now need a bigger grid to include 2 features that don't overlap
    # will use now grid_manager
    box0 = ws.intensities.amplitudes.spectrum_composition.Box({'A': (879.0, 1121.0),
                                                               'B': (399.0, 641.0)})
    box1 = ws.intensities.amplitudes.spectrum_composition.Box({'A': (879.0, 1121.0), 
                                                               'B': (379.0, 621.0)})

    union_box = ws.intensities.amplitudes.spectrum_composition.Box.union([box0, box1])
    assert union_box.bounds == {'A': (879.0, 1121.0), 'B': (379.0, 641.0)}

    spec_window = ws.intensities.amplitudes.spectrum_composition.SpectralWindow(box=union_box)
    spec_window.full_features = features
    
    # grid prepared via GridManager
    from wilson_suite.wilson_intensities.amplitudes.grid_manager_evaluator import GridManager
    grid_manager = GridManager(spec_window)
    grid_manager.make_fullgrid({'A': 50, 'B': 90})
    assert grid_manager.full_grid['A'].shape == grid_manager.full_grid['B'].shape == (50, 90)

    regions = grid_manager.create_regions()
    # just one because boxes overlap and they form a domain/grid region
    assert len(regions) == 1

    assert regions[0].domain.box == ws.intensities.amplitudes.spectrum_composition.Box({'A': (970.0, 1030.0), 'B': (470.0, 550.0)})


    from wilson_suite.wilson_intensities.tests.unit.test_domains import get_data_evaluators_tests
    vbana = get_data_evaluators_tests()['vib_ana_setup']

    vibstates_data = ws.intensities.amplitudes.term_parts.VibStatesData(allstates=MakeObjects.mk_vibstates_states(), 
                                                                        harmonic_osc_states_labels=vbana.include_list,
                                                                        number_of_nmodes=vbana.number_of_modes)
    assert vibstates_data.get_state_by_label('0').energy == 1119.5
    assert vibstates_data.get_state_by_label('1').energy == 964.
    assert vibstates_data.get_state_by_label('2').energy == 1234.
    assert vibstates_data.get_state_by_label('0,1').energy == 3885.

    from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
    vibdiff_cache = VibDiffCache()
    
    # regions evaluation - here a single region that matches the whole SpectralWindow in this case
    # at this point all units should be au in evaluations (including gamma and coordinates of the spec grids)
    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    res = ws.intensities.amplitudes.evaluation_wf.evaluate_regions(regions=regions, # holds grid coordinates
                                                                   vib_data=vibstates_data, 
                                                                   vibdiff_cache=vibdiff_cache, 
                                                                   gamma=convNu2Ene(features[0].lineshape_parameter), # btw, assuming features have uniform lineshape parameter
                                                                   verbose=True)

    region0_eval = list(res.values())[0]

    # reference value construction 
    # amplitude = feat_coeff  / (w_0,a + A) / (w_a+b,b - B) with a=0,b=1
    assert vibdiff_cache._cache == {('zero', '0'): -1119.5, ('0,1', '1'): 2921.0, ('0', '1'): -155.5}

    # since 2 features share the GridRegion, they both get evaluated over its grid and summed
    resonance_part_0 = 1. / (convNu2Ene(-1119.5 + regions[0].coords['A']) - 1j* convNu2Ene(features[0].lineshape_parameter)) / (convNu2Ene(2921.0 - regions[0].coords['B']) - 1j* convNu2Ene(features[0].lineshape_parameter))

    resonance_part_1 = 1. / (convNu2Ene(-1119.5 + regions[0].coords['A']) - 1j* convNu2Ene(features[0].lineshape_parameter)) / (convNu2Ene(-155.5 - regions[0].coords['B']) - 1j* convNu2Ene(features[0].lineshape_parameter))

    ref_res0 = features[0].amplitude_coeff * resonance_part_0
    ref_res1 = features[1].amplitude_coeff * resonance_part_1
    
    assert np.allclose(ref_res0 + ref_res1, region0_eval)



def test_integration_evv_experiment_until_after_evaluation():

    from wilson_suite.fixtures import evv_experiment
    from wilson_suite.wilson_utils.paths import SUITE_ROOT
    from wilson_suite.wilson_experiment.indep_vars_and_axes import SpectralAxisSet, IndependentVariableSet, \
        SignedPulseTuple, SpectralAxis

    evv_exp = evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)
    #axes_choice = evv_exp.valid_axis_combs[0].valid_axis_combs[1]  # {'A': [(-1,)], 'B': [(2,)]}
    axes_choice = evv_exp.valid_axis_combs[0].valid_axis_combs[0] # {'A': [(-1,)], 'B': [(-1,), (2,)]}
    # axis_choice = SpectralAxisSet(
    #axes = (SpectralAxis(label='A', var_set=IndependentVariableSet(var_set=(SignedPulseTuple(pulse_refs=(1,)),))),
    #        SpectralAxis(label='B', var_set=IndependentVariableSet(
    #            var_set=(SignedPulseTuple(pulse_refs=(-1,)),
    #                     SignedPulseTuple(pulse_refs=(2,)))))))  # {'A': [(1,)], 'B': [(-1,), (2,)]}



    calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian',
                                                     lvl_theory='B3LYP',
                                                     basis_set='cc-pVQZ',
                                                     base_file_loc=SUITE_ROOT+'/../data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')

    sim = ws.main.workflow_abstractions.WilsonSimulation()
    sim.addExperiment(evv_exp)
    sim.addTerms(terms=terms)  # terms

    mol_system = ws.main.abstractions.MolecularSystem(name='formaldehyde', natoms=4)

    vib_ana = ws.main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2', vibana_own_analysis='none')

    sim.addSystem(mol_system)
    sim.addVibAnaSetup(vib_ana)
    sim.addPropEvalSetup(eval_uniform=calc_setup)

    sim.setPropsAndMaxStateLvl()  # setting up self.props/sim.props
    sim.dressPropsWithSetup()

    from wilson_suite.wilson_utils.some_reprs import make_SpectralAxisSet
    axes_choice: ws.main.spectrum_abstractions.SpectralAxisSet = make_SpectralAxisSet({'A': [1], 'B': [-1,2]}) # this makes A and B > 0
    sim.setAxisChoiceAndTranslateTerms(axes_choice)

    from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box

    # These windows capture many of the same features for the respective axis set choices
    #bounds_dict = {'A': (-3000, -1200.), 'B': (1200., 6000.)} # {'A': [(-1,)], 'B': [(2,)]}
    # bounds_dict = {'A': (-3000, -1200.), 'B': (500., 3000.)} # {'A': [(-1,)], 'B': [(-1,), (2,)]}
    bounds_dict = {'A': (1200., 3000.), 'B': (500., 3000.)} # {'A': [(1,)], 'B': [(-1,), (2,)]}

    # These bounds raise a "no features in window" error but I thought they would correspond to what I used for
    # {'A': [(-1,)], 'B': [(-1,), (2,)]} right above here
    # bounds_dict = {'A': (1200, 3000.), 'B': (500., 3000.)} # {'A': [(1,)], 'B': [(-1,), (2,)]}

    spectral_window = SpectralWindow(box=Box(bounds_dict))

    dynrange_log10 = 9 # 3 = dynamic range 1000

    evi = ws.main.spectrum_abstractions.EvaluationInfo(**{'spectral_window': spectral_window,
                                                          'Gamma': 4.7, 'Gamma_unit': 'cm-1',
                                                          'grid_resolution': {'A': 360, 'B': 600},
                                                          'dynamic_range': 10**dynrange_log10,
                                                          'box_range_safety_margin': 0.1,
                                                          'scale_wrt_max_intensity': True,
                                                          'minimum_box_padding': 30.0
                                                          }
                                                       )


    eval_setup = ws.main.spectrum_abstractions.SpecEvalSetup(ev_info=evi)

    sim.addSpecEvalSetup(eval_setup)

    from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
    sim.getResults(obtainer=wilson_data_obtainer)
    sim.vib_ana_setup.set_include_modes_list()

    sim.evaluate()

    sw_here = sim._workflow.artifacts.spec_window
    print('\n-----\nall full_features here')
    ws.intensities.amplitudes.spectrum_composition.SpectralFeature.print_list_features(sw_here.full_features)
    print(len(sw_here.full_features))

    for f in sw_here.full_features:
        print(f.amplitude_coeff)
    # print(sw_here.full_features)
    # print(sw_here.contrib_features)

    # import matplotlib.pyplot as plt


    # Z = np.log(np.abs(sim.spec)**2)

    # zmax = np.amax(Z)

    # for i in range(Z.shape[0]):
    #     for j in range(Z.shape[1]):
    #         if Z[i, j] < (zmax - dynrange_log10):
    #             Z[i, j] = (zmax - dynrange_log10)


    # x = np.unique(sim.spec_eval_setup.grid['A'])
    # y = np.unique(sim.spec_eval_setup.grid['B'])

    # # if Z.shape == (len(y), len(x)) -> no transpose; if Z.shape == (len(x), len(y)) -> transpose
    # # matplotlib expects [y, x] ordering for images
    # toplot = Z.T

    # plt.pcolormesh(x, y, toplot, vmax=np.amax(Z), vmin=np.amax(Z)-dynrange_log10, shading="auto")
    # plt.xlabel('A')
    # plt.ylabel('B')
    # plt.colorbar(label='log intensity')
    # plt.show()
