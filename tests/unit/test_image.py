import numpy as np
from tests.testing_utils import require_asserts

from wilson.utils import prep_data_load
from wilson.spectrum.termND import TermND
from wilson.spectrum.termsEvaluator import TermsEvaluator

# import sys
# sys.path.append('/home/vlev/wilson-suite/')
# import wilson_suite as ws
# print(dir(ws))
# from ws.analysis import render
from wilson_analysis import render

import wilson.debug as debug
debug.level = 0

@require_asserts
def test_amplitude_1term_grid(dict_8terms, MOL_setup_parser, spectrum_setup):
    print()
    parsed_data = MOL_setup_parser.parse(linear_molecule=False)

    parsed_data.get_vpt2(vpt2settings={'anharmonic_type': 'GVPT2'}, list2exclude=None, print_level=0)
    parsed_data.upd_indices_several_parts(spectrum_setup.old_new_dict)
    deriv_data, allstates, harmonic_states, mode_indices = prep_data_load(parsed_data) # wrapper func

    t0 = TermND(0, dict_8terms[0])
    t1 = TermND(1, dict_8terms[1])
    t2 = TermND(2, dict_8terms[2])
    t3 = TermND(3, dict_8terms[3])

    t0.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
    t1.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
    t2.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)
    t3.load_calc_data(properties_data=deriv_data, allstates=allstates, harmonic_states=harmonic_states,
                      mode_indices=mode_indices, gammaCompsAll=spectrum_setup.gammaCompsAll)

    ###########################################################################################################
    terms = [t0, t1, t2, t3]
    te = TermsEvaluator(terms)

    Nnmodes = 6
    print(t0.properties_data.keys())
    data = {
        (1, 1): t0.properties_data['mu_Q'],
        (1, 2): t0.properties_data['mu_QQ'],
        (2, 1): t0.properties_data['alpha_Q'],
        (2, 2): t0.properties_data['alpha_QQ'],
    }
    avrg_terms = spectrum_setup.gammaCompsAll

    w1 = np.arange(spectrum_setup.start1, spectrum_setup.end1, spectrum_setup.step1)
    w2 = np.arange(spectrum_setup.start2, spectrum_setup.end2, spectrum_setup.step2)
    w1m, w2m = np.meshgrid(w1, w2)

    axes_dict = {1: w1m, 2: w2m}

    alldata = [Nnmodes, data, avrg_terms, axes_dict, t2.states_arrays_Eh, t2.harmonic_arrays_Eh] # todo: set this up better
    te.identify_to_precalculate()
    big_dict = te.precalculate(alldata)
    for t in terms:
        t.precalc_data = big_dict
    ###########################################################################################################

    debug.level = 1

    amplitudes = 0.
    for t in terms:
        e = t.get_intensity(w1m, w2m, 3.8, 0., debugprint=False, collect_all=True)
        print('\n-----', t)
        print(e)
        print('any nan???? ', np.isnan(e).any())  # True if there's at least one NaN
        amplitudes += e
    debug.level = 0
    print('\n---- amplitudes')
    print(amplitudes.shape)
    print(np.max(np.abs(amplitudes)))
    print(f'{np.max(np.abs(amplitudes)**2):.2e}')
    print(spectrum_setup.start1, spectrum_setup.end1, spectrum_setup.step1)
    print(spectrum_setup.start2, spectrum_setup.end2, spectrum_setup.step2)
    print(w1m.shape)
    print(w1m)

    intensities = np.abs(amplitudes)**2

    fig, ax = render.set_figure(figsize=(40,60), font_dict={'size': 20}, to_save=True)
    levels_nums, levels_ticks, levels_nums_str = render.prep_levels(d_max=np.max(np.abs(amplitudes)**2),
                                                                    dynamic_range=50,
                                                                    num_level_ticks=10)
    intensity_plot = render.prep_intensity_log10(intensities, normalized='01')
    render.set_xyz(w1m, w2m, intensity_plot, fig, ax, w1mw2=True, nicetitle='yes',
                   levels=levels_ticks, saturation_color='#FF00FF',
                   levels_ticks=levels_ticks,
                   levels_nums_str=levels_nums_str,
                   maxYX=3200., minY=None)
    render.finilize_ax(ax, filename='yo.svg', dpi=250, to_save=True)
