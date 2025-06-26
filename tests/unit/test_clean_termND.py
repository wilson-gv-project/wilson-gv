import numpy as np
import pytest
from tests.testing_utils import require_asserts, debug_mode

import wilson.debug as debug
import CQCParse.debug as cqc_debug

debug.level = 0
cqc_debug.level = 0

np.set_printoptions(precision=4)

from wilson_analysis import render
def render_spectrum_with_debug(intensities, w1m, w2m, filename, nicetitle='yes'):
    """
    Helper function to render the spectrum figure with debugging.
    """
    print(f"Rendering spectrum: {filename}")
    print(f"Intensity data stats - Min: {np.min(intensities)}, Max: {np.max(intensities)}, Mean: {np.mean(intensities)}")
    # Normalize intensities for rendering
    normalized_intensities = intensities / np.max(intensities)
    print(f"Normalized intensity stats - Min: {np.min(normalized_intensities)}, Max: {np.max(normalized_intensities)}")
    fig, ax = render.set_figure(figsize=(40, 60), font_dict={'size': 20}, to_save=True)
    levels_nums, levels_ticks, levels_nums_str = render.prep_levels(
        d_max=np.max(normalized_intensities),
        dynamic_range=100,
        num_level_ticks=10
    )
    print(f"Levels: {levels_nums}")
    print(f"Ticks: {levels_ticks}")
    intensity_plot = render.prep_intensity_log10(normalized_intensities, normalized='01')
    render.set_xyz(
        w1m, w2m, intensity_plot, fig, ax,
        w1mw2=True, nicetitle=nicetitle,
        levels=levels_ticks, saturation_color='#FF00FF',
        levels_ticks=levels_ticks,
        levels_nums_str=levels_nums_str,
        maxYX=3000., minY=None
    )
    render.finilize_ax(ax, filename=filename, dpi=250, to_save=True)
def render_spectrum(intensities, w1m, w2m, filename, nicetitle='yes'):
    """
    Helper function to render the spectrum figure.
    """
    fig, ax = render.set_figure(figsize=(40, 60), font_dict={'size': 20}, to_save=True)
    levels_nums, levels_ticks, levels_nums_str = render.prep_levels(
        d_max=np.max(intensities),
        dynamic_range=100,
        num_level_ticks=10
    )
    assert all(upper > lower for upper, lower in zip(levels_nums[1:], levels_nums[:-1])), "Invalid contour"
    print('\nlevels_nums', levels_nums)
    print('levels_ticks', levels_ticks, '\n')
    # print('levels_nums_str', levels_nums_str)

    intensity_plot = render.prep_intensity_log10(intensities, normalized='01')
    render.set_xyz(
        w1m, w2m, intensity_plot, fig, ax,
        w1mw2=True, nicetitle=nicetitle,
        levels=levels_ticks, saturation_color='#FF00FF',
        levels_ticks=levels_ticks,
        levels_nums_str=levels_nums_str,
        maxYX=3000., minY=None
    )
    render.finilize_ax(ax, filename=filename, dpi=250, to_save=True)
def compare_amplitudes(amplitudes1, amplitudes2):
    """
    Helper function to compare two sets of amplitudes.
    """
    assert amplitudes1.shape == amplitudes2.shape, "Amplitude shapes do not match"
    diff = np.abs(amplitudes1 - amplitudes2)
    max_diff = np.max(diff)
    print(f"Maximum difference between amplitudes: {max_diff:.2e}")
    assert max_diff < 1e-6, "Amplitudes differ significantly"

# @pytest.mark.parametrize("molecule", ["FORM", "OXAC2"])
@require_asserts
def test_terms_collection_calculation(terms_collection, spectrum_setup):
    """
    get a spectrum figure
    """
    # terms_collection is a fixture
    te, big_dict = terms_collection
    assert len(te.terms) == 4, "Expected 4 terms in the TermsEvaluator"

    expected_keys = ['vibene_denoms', 'avrg_tensors', 'res_conds', 'vibdiffs']
    for key in expected_keys:
        assert key in big_dict, f"Key '{key}' missing in precalculated data"
    assert big_dict['vibene_denoms'], "vibene_denoms data is empty"

    with debug_mode(0):
        amplitudes = 0.0
        for id, term in te.terms.items():
            intensity = term.get_intensity(spectrum_setup.w1m, spectrum_setup.w2m,
                                           3.8, 0.0, debugprint=True, collect_all=False)
            amplitudes += intensity

    assert np.isfinite(amplitudes).all(), "Amplitudes contain non-finite values (NaN or Inf)"
    max_amplitude = np.max(np.abs(amplitudes))
    print(f"Maximum amplitude: {max_amplitude:.2e}")
    print(f"Maximum intensity: {np.max(np.abs(amplitudes)**2):.3e}")

    print(amplitudes.shape)

    np.set_printoptions(precision=4)
    print(amplitudes)

    intensities = np.abs(amplitudes)**2
    render_spectrum(intensities, spectrum_setup.w1m, spectrum_setup.w2m,
                    filename=f'yo_terms_{spectrum_setup.molecule}.svg', nicetitle='TermsEvaluator')


def test_spectrum2d_calculation(intensity_data, spectrum2d, spectrum_setup):
    """
    Test the intensity calculation and rendering of the spectrum.
    """
    print(intensity_data.shape)
    print(f"Maximum amplitude: {np.max(abs(intensity_data)):.2e}")

    assert intensity_data is not None, "Intensity data is None"
    assert np.isfinite(intensity_data).all(), "Intensity data contains NaN or Inf values"
    max_intensity = np.max(abs(intensity_data) ** 2)
    print(f"Maximum intensity: {max_intensity:.3e}")

    np.set_printoptions(precision=4)

    intensities = np.abs(intensity_data) ** 2
    assert np.all(np.isfinite(intensities)), "Data contains NaN or Inf"
    assert np.min(intensities) >= 0, "Negative intensities detected!"

    render_spectrum(intensities, spectrum_setup.w1m, spectrum_setup.w2m,
                    filename=f'yo_spec2d_{spectrum_setup.molecule}.svg', nicetitle='Spectrum2D')

@require_asserts
def test_compare_amplitudes(terms_amplitudes, intensity_data, spectrum_setup):
    """
    Compare amplitudes calculated using TermsEvaluator and Spectrum2D.
    """
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
    # import matplotlib.pyplot as plt
    # Visualize the zero mask
    # plt.figure(figsize=(10, 8))
    # plt.imshow(non_zero_mask, cmap='gray', interpolation='none')
    # plt.title("Zero Mask for Spectrum2D Data")
    # plt.colorbar(label="Non-Zero (1) vs Zero (0)")
    # plt.savefig("zero_mask_spectrum2d.svg", dpi=300)
    # # Flatten the data to 1D arrays for histogram plotting
    # terms_abs_values = np.abs(terms_amplitudes).flatten()
    # spectrum2d_abs_values = np.abs(intensity_data).flatten()
    # # Plot histograms
    # plt.figure(figsize=(12, 6))
    # plt.hist(terms_abs_values, bins=100, alpha=0.7, label='TermsEvaluator', color='green')
    # plt.hist(spectrum2d_abs_values, bins=100, alpha=0.3, label='Spectrum2D', color='red')
    # plt.yscale('log')  # Use a log scale to better visualize the distribution
    # plt.xlabel('Absolute Value')
    # plt.ylabel('Frequency (log scale)')
    # plt.title('Distribution of Absolute Values')
    # plt.legend()
    # plt.savefig("hist.svg", dpi=300)
    # Calculate the difference
    # difference = np.abs(terms_amplitudes - intensity_data)
    # Render the difference
    # render_spectrum(difference, spectrum_setup.w1m, spectrum_setup.w2m, filename='yo_difference.svg', nicetitle='Difference')

    # intensities_terms = np.abs(terms_amplitudes) ** 2
    # intensities_spectrum2d = np.abs(intensity_data) ** 2

    # render_spectrum_with_debug(intensities_terms, spectrum_setup.w1m, spectrum_setup.w2m,
    #                 filename='yo_terms_comparison.svg', nicetitle='TermsEvaluator')
    # render_spectrum_with_debug(intensities_spectrum2d, spectrum_setup.w1m, spectrum_setup.w2m,
    #                 filename='yo_spectrum2d_comparison.svg', nicetitle='Spectrum2D')
    # render_spectrum(intensities_terms, spectrum_setup.w1m, spectrum_setup.w2m,
    #                 filename=f'yo_terms_comparison_{spectrum_setup.molecule}.svg', nicetitle='TermsEvaluator')
    # render_spectrum(intensities_spectrum2d, spectrum_setup.w1m, spectrum_setup.w2m,
    #                 filename=f'yo_spectrum2d_comparison_{spectrum_setup.molecule}.svg', nicetitle='Spectrum2D')

    diff_indices = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"Largest difference at index: {diff_indices}")

    max_diff = np.max(diff)
    print(f"Maximum difference between datasets: {max_diff:.2e}")
    # assert max_diff < 1e-6, "Datasets differ significantly"


