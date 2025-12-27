import numpy as np
from .test_evaluators import prep_vibanasetup_with_degen_states

from wilson_suite.wilson_intensities.amplitudes.term_parts import (
    VibStatesData, TermParametersChoice, ResonanceMotif,
    ResonanceCondition, ParameterSet
)
from wilson_suite.wilson_intensities.amplitudes.vibene_differences import VibDiffCache
from ...amplitudes.spectrum_composition import SpectralFeature, Box, ResLocGeoObject, SpectralWindow

import logging
from ....wilson_utils.logger import setup_logger
setup_logger("wilson", level=logging.INFO)

def test_example_new_api_asserts():
    """Example of using the new clean API with actual assertions instead of prints."""

    # -------------------------
    # Setup
    # -------------------------
    spec_window = SpectralWindow(box=Box({'A': (5., 30.), 'B': (45., 60.)}))
    print('\nspec_window before', spec_window)
    
    rc1 = ResonanceCondition.make_from_tuples(
        left_state=('a', 'b'),
        right_state=('a',),
        pert_freqs=('A',)
    )
    rc2 = ResonanceCondition.make_from_tuples(
        left_state=('b',),
        right_state=('a',),
        pert_freqs=('B',)
    )

    states_parameters = (
        ParameterSet({'a': 0, 'b': 0}),
        ParameterSet({'a': 0, 'b': 1}),
        ParameterSet({'a': 1, 'b': 1}),
    )

    term_contributions = (
        TermParametersChoice(
            res_motif=ResonanceMotif([rc1, rc2]),
            states_parameters=states_parameters,
        ),
    )

    sf1 = SpectralFeature(
        location=ResLocGeoObject({'A': 15., 'B': 55.}),
        lineshape_parameter={'A': 2.5, 'B': 1.5},
        term_contributions=term_contributions,
        amplitude_coeff=10.0,
    )

    states_parameters_small = (ParameterSet({'a': 0, 'b': 0}),)
    term_contributions_small = (
        TermParametersChoice(
            res_motif=ResonanceMotif([rc1, rc2]),
            states_parameters=states_parameters_small,
        ),
    )

    sf4 = SpectralFeature(
        location=ResLocGeoObject({'A': 4.0, 'B': 43.8}),
        lineshape_parameter={'A': 2.5, 'B': 1.5},
        term_contributions=term_contributions_small,
        amplitude_coeff=40.0,
    )

    # Filter spectral features to the window
    spec_window = SpectralFeature.filter_to_spec_window([sf1, sf4], spec_window)
    print('\nspec_window after', spec_window)

    vib_ana_setup = prep_vibanasetup_with_degen_states()
    vib_data = VibStatesData(vib_ana_setup.states)
    vibdiff_cache = VibDiffCache()

    evaluator = SpectralEvaluator(vib_data, vibdiff_cache, gamma=2.0)

    # -------------------------
    # Evaluate: full grid
    # -------------------------
    full_grid = evaluator.evaluate_spectrum(
        spec_window=spec_window,
        grid_resolution={'A': 10, 'B': 10}, # 10 points in each axis
        verbose=True,
        return_type='grid',
    )

    spectrum = full_grid['result']
    axis_A = full_grid['A']
    axis_B = full_grid['B']

    # -------------------------
    # Assertions (replace prints)
    # -------------------------

    # 1. Correct grid shape
    assert spectrum.shape == (10, 10)
    assert axis_A.shape == (10, 10)
    assert axis_B.shape == (10, 10)

    # 2. Values reasonable – non-zero but finite
    assert np.isfinite(spectrum).all()
    assert np.max(np.abs(spectrum)) > 0

    # 3. Spectrum depends only on features inside window
    #    Here we check that the feature outside was removed (sf4 is outside A=5..30)
    #    So only 1 feature should remain:
    assert len(spec_window.full_features) == 1
    assert len(spec_window.contrib_features) == 1

    # -------------------------
    # Test region-based output
    # -------------------------
    region_results = evaluator.evaluate_spectrum(
        spec_window=spec_window,
        grid_resolution={'A': 10, 'B': 10},
        verbose=True,
        return_type='regions',
    )

    # Regions should be non-empty and have the expected shape
    assert len(region_results) >= 1
    for i, (region, arr) in enumerate(region_results.items()):
        if i == 0:
            assert arr.shape == (4, 3)
        if i == 1:
            assert arr.shape == (2, 2)
        
        assert np.isfinite(arr).all()

    # -------------------------
    # Test return_type="both"
    # -------------------------
    full2, r2 = evaluator.evaluate_spectrum(
        spec_window=spec_window,
        grid_resolution={'A': 10, 'B': 10},
        verbose=True,
        return_type='both',
    )
    np.set_printoptions(linewidth=180, precision=3)

    print('\n', full2['result'])
    assert 'result' in full2
    assert isinstance(r2, dict)
