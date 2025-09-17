"""
Integration tests for full procedure with figure rendering using TermND functionality.
"""
import numpy as np
from ..testing_utils import require_asserts
from ...spectrum import debug_mode
from ....wilson_analysis.render.simple_plot import render_spectrum

from ....wilson_utils import printing as debug
import CQCParse.debug as cqc_debug

debug.level = 0
cqc_debug.level = 0


np.set_printoptions(precision=4,suppress=False)


def compare_amplitudes(amplitudes1: np.ndarray, amplitudes2: np.ndarray) -> None:
    """
    Helper function to compare two sets of amplitudes.
    """
    assert amplitudes1.shape == amplitudes2.shape, "Amplitude shapes do not match"
    diff = np.abs(amplitudes1 - amplitudes2)
    max_diff = np.max(diff)
    print(f"Maximum difference between amplitudes: {max_diff:.2e}")
    assert max_diff < 1e-6, "Amplitudes differ significantly"

@require_asserts
def test_terms_collection_calculation(terms_collection: dict, spectrum_setup: dict, conditions: dict) -> None:
    """
    get a spectrum figure
    """
    # terms_collection is a fixture
    te, precalc_dict = terms_collection['FORM']
    spectrum_setup = spectrum_setup['FORM']
    conditions = conditions['FORM']

    expected_keys = ['vibene_denoms', 'avrg_tensors', 'res_conds', 'vibdiffs']
    for key in expected_keys:
        assert key in precalc_dict, f"Key '{key}' missing in precalculated data"
    assert precalc_dict['vibene_denoms'], "vibene_denoms data is empty"

    
    # can run with debug prints if > 0
    with debug_mode(0):
        amplitudes = 0.0
        for id, term in te.terms.items():
            term.precalc_data = precalc_dict
            
            # can run with debug prints if > 0
            with debug_mode(0):
                intensity = term.get_amplitudes(spectrum_setup.w1m,
                                                spectrum_setup.w2m,
                                                3.8, 1.0, debugprint=True, collect_all=False)
            amplitudes += intensity

    print(amplitudes.shape)

    assert np.isfinite(amplitudes).all(), "Amplitudes contain non-finite values (NaN or Inf)"
    max_amplitude = np.max(np.abs(amplitudes))
    print(f"Maximum amplitude: {max_amplitude:.2e}")
    print(f"Maximum intensity: {np.max(np.abs(amplitudes)**2):.3e}")

    np.set_printoptions(precision=4)

    intensities = np.abs(amplitudes)**2

    hist, bin_edges = np.histogram(intensities, bins=10)
    print('\nintensities from test:')
    print("Histogram counts:", hist)
    print("Bin edges:", bin_edges)

    render_spectrum(intensities, spectrum_setup.w1m, spectrum_setup.w2m,
                    filename=f'yo_terms_{spectrum_setup.molecule}.svg',
                    dynamic_range=conditions.dynamic_range_n,
                    nicetitle='TermsEvaluator')


@require_asserts
def test_terms_collection_calculation_derived(terms_collection_derived: dict, 
                                              spectrum_setup: dict, conditions: dict) -> None:
    """
    get a spectrum figure
    """
    # terms_collection is a fixture
    te, precalc_dict = terms_collection_derived['FORM']
    spectrum_setup = spectrum_setup['FORM']
    conditions = conditions['FORM']

    expected_keys = ['vibene_denoms', 'avrg_tensors', 'res_conds', 'vibdiffs']
    for key in expected_keys:
        assert key in precalc_dict, f"Key '{key}' missing in precalculated data"
    assert precalc_dict['vibene_denoms'], "vibene_denoms data is empty"

    # can run with debug prints if > 0
    with debug_mode(0):
        amplitudes = 0.0
        for id, term in te.terms.items():
            term.precalc_data = precalc_dict
            # with np.printoptions(precision=2,legacy='1.25'):
                # formatted_resonances = {
                    # key: (f"{value[0]:.2f}", f"{value[1]:.2f}")
                    # for key, value in term.get_all_resonances(w2mw1=True).items()
                # }
                # print(formatted_resonances)
            # can run with debug prints if > 0
            with debug_mode(0):
                intensity = term.get_amplitudes(spectrum_setup.w1m,
                                                spectrum_setup.w2m,
                                                3.8, 1.0, debugprint=True, collect_all=False)
            amplitudes += intensity

    print(amplitudes.shape)

    assert np.isfinite(amplitudes).all(), "Amplitudes contain non-finite values (NaN or Inf)"
    max_amplitude = np.max(np.abs(amplitudes))
    print(f"Maximum amplitude: {max_amplitude:.2e}")
    print(f"Maximum intensity: {np.max(np.abs(amplitudes)**2):.3e}")

    np.set_printoptions(precision=4)

    intensities = np.abs(amplitudes)**2

    hist, bin_edges = np.histogram(intensities, bins=10)
    print('\n')
    print("Histogram counts:", hist)
    print("Bin edges:", bin_edges)

    render_spectrum(intensities, spectrum_setup.w1m, spectrum_setup.w2m,
                    filename=f'yo_terms_derived_{spectrum_setup.molecule}.svg',
                    dynamic_range=conditions.dynamic_range_n,
                    nicetitle='TermsEvaluator')



def test_spectrum2d_calculation(intensity_data: dict, spectrum_setup: dict, conditions: dict) -> None:
    """
    Test the intensity calculation and rendering of the spectrum.
    """
    intensity_data = intensity_data['FORM']
    spectrum_setup = spectrum_setup['FORM']
    conditions = conditions['FORM']

    print(intensity_data.shape)
    print(f"Maximum amplitude: {np.max(abs(intensity_data)):.2e}")

    assert intensity_data is not None, "Intensity data is None"
    assert np.isfinite(intensity_data).all(), "Intensity data contains NaN or Inf values"
    max_intensity = np.max(abs(intensity_data) ** 2)
    print(f"Maximum intensity: {max_intensity:.3e}")

    np.set_printoptions(precision=4)

    intensities = np.abs(intensity_data) ** 2

    hist, bin_edges = np.histogram(intensities, bins=10)
    print('\n')
    print("Histogram counts:", hist)
    print("Bin edges:", bin_edges)

    assert np.all(np.isfinite(intensities)), "Data contains NaN or Inf"
    assert np.min(intensities) >= 0, "Negative intensities detected!"

    render_spectrum(intensities.T, spectrum_setup.w1m, spectrum_setup.w2m,
                    filename=f'yo_spec2d_{spectrum_setup.molecule}.svg',
                    dynamic_range=conditions.dynamic_range_n,
                    nicetitle='Spectrum2D')

@require_asserts
def test_compare_amplitudes(terms_amplitudes: dict, intensity_data: dict) -> None:
    """
    Compare amplitudes calculated using TermsEvaluator and Spectrum2D.
    """
    terms_amplitudes = terms_amplitudes['FORM']
    intensity_data = intensity_data['FORM'].T # a bug there; this is a quick fix

    # Validate TermsEvaluator amplitudes
    assert terms_amplitudes is not None, "TermsEvaluator amplitudes are None"
    assert np.isfinite(terms_amplitudes).all(), "TermsEvaluator amplitudes contain NaN or Inf values"
    max_terms_amplitude = np.max(np.abs(terms_amplitudes))
    print(f"Maximum amplitude (TermsEvaluator): {max_terms_amplitude:.2e}")
    # Validate Spectrum2D intensity data
    assert intensity_data is not None, "Spectrum2D intensity data is None"
    assert np.isfinite(intensity_data).all(), "Spectrum2D intensity data contains NaN or Inf values"
    max_spectrum2d_intensity = np.max(np.abs(intensity_data))
    print(f"Maximum intensity (Spectrum2D): {max_spectrum2d_intensity:.2e}")
    # Compare the two datasets
    assert terms_amplitudes.shape == intensity_data.shape, "Shapes of the datasets do not match"
    diff = np.abs(terms_amplitudes - intensity_data)

    print("TermsEvaluator Data Statistics:")
    print(f"Min: {np.min(terms_amplitudes)}, Max: {np.max(terms_amplitudes)}, Mean: {np.mean(terms_amplitudes)}")
    print("Spectrum2D Data Statistics:")
    print(f"Min: {np.min(intensity_data)}, Max: {np.max(intensity_data)}, Mean: {np.mean(intensity_data)}")

    print("Spectrum2D Zero Values:")
    print(f"Number of zeros: {np.sum(intensity_data == 0)}")
    print(f"Number of very small values (<1e-6): {np.sum(np.abs(intensity_data) < 1e-6)}")

    print("TermsEvaluator Zero Values:")
    print(f"Number of zeros: {np.sum(terms_amplitudes == 0)}")
    print(f"Number of very small values (<1e-6): {np.sum(np.abs(terms_amplitudes) < 1e-6)}")
    print('\n', terms_amplitudes.shape[0] * terms_amplitudes.shape[1], '\n')
    # Mask zeros in the Spectrum2D data
    non_zero_mask = intensity_data != 0
    masked_int_data = intensity_data[non_zero_mask]
    # Check the statistics of the masked data
    print("Masked Spectrum2D Data Statistics:")
    print(f"Min: {np.min(masked_int_data)}, Max: {np.max(masked_int_data)}, Mean: {np.mean(masked_int_data)}")

    diff_indices = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"Largest difference at index: {diff_indices}")

    max_diff = np.max(diff)
    print(f"Maximum difference between datasets: {max_diff:.2e}")


