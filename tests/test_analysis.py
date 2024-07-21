import numpy as np
from wilson.dataanalysis import get_resonances
from wilson import spectrum


def test_get_resonances():
    terms_selection = [0, 1], [0, 1, 2, 3, 4, 5]

    omega1 = np.arange(2810., 3210., 10.)
    omega2 = np.arange(5510., 6050., 10.)
    datain = spectrum.make_DatainputDict('gaussian', ('FORM', 'HF', 'cc_pVQZ'))

    computedSpectrum = spectrum.SpectrumEVV(omega1, omega2, input_data_info=datain, vib_levels_harmonic=False)
    computedSpectrum.addTerms(*terms_selection)
    print('\nvib_levels_harmonic bool:', computedSpectrum.vib_levels_harmonic)

    elTermsDict = dict(zip(terms_selection[0], computedSpectrum.electrical_terms))
    mechTermsDict = dict(zip(terms_selection[1], computedSpectrum.mechanical_terms))

    combinations = (computedSpectrum.coords_ab, computedSpectrum.coords_abc)
    print('\n------------------------')
    print("All states anharmonic\n", computedSpectrum.all_states)
    print("All states harmonic\n", computedSpectrum.all_states_harmonic)
    print('\n------------------------\n')

    get_resonances(elTermsDict, mechTermsDict, computedSpectrum.all_states, combinations, rec_cm=False)

def test_get_avrg_tensors():
    """
    """
    terms_selection = [0, 1], [0, 1, 2, 3, 4, 5]
    omega1 = np.arange(2810., 3210., 10.)
    omega2 = np.arange(5510., 6050., 10.)
    datain = spectrum.make_DatainputDict('gaussian', ('FORM', 'HF', 'cc_pVQZ'))

    computedSpectrum = spectrum.SpectrumEVV(omega1, omega2, input_data_info=datain, vib_levels_harmonic=False)
    computedSpectrum.addTerms(*terms_selection)

    print('\n-------------------')
    for et in computedSpectrum.el_avrg_tensors:
        print(et.shape)
        print(et)
    print('-------------------')
    for mt in computedSpectrum.mech_avrg_tensors:
        print(mt.shape)
        print(mt)
    print('\ncubic force constants')
    print(computedSpectrum.deriv_data['F_abc'])