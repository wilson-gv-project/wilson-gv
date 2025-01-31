import numpy as np
np.set_printoptions(legacy='1.25')

from wilson.spectrum.spectrum2D import Spectrum2D
from CQCParse.parsing import GaussianDataParser
from CQCParse.relay import DataVault

path_to_files = '' # set up your path to the directory shared on OneDrive
# https://universitetetitromso-my.sharepoint.com/:f:/g/personal/vle014_uit_no/EgH4Rjk0_YtNvH1BXVMdh3gBnB5H5j68lDC7EROXiBM3Ag?email=magnus.ringholm%40uit.no&e=ScmRmw
data_vault = DataVault('/files_fram/files_database.csv')

omega1 = np.arange(1130., 2050., 2.91)
omega2 = np.arange(1300., 5150., 2.91)


def test_VPT2():
    molecule = 'FORM'
    method = 'HF'
    basis = 'STO_3G_VPT2'
    Gamma_rc = 5.1
    list2exclude = []
    terms_selection = [0, 1], [2, 3]

    datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    gParser = GaussianDataParser(datadict)

    dictInputs = {'parserObject': gParser,
                  'el_terms_select': terms_selection[0], 'mech_terms_select': terms_selection[1]}

    spectrumObj = Spectrum2D(omega1, omega2)
    spectrumObj.load_data(dictInputs['parserObject'], vpt2=True, vpt2settings={'anharmonic_type':'VPT2'})

    spectrumObj.set_spectrum_settings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=True)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.add_terms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculate_parts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))
        # print(spectrumObj.all_states[i], spectrumObj.all_states_corr[i])
        assert np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3)
    assert all(freqs)


def test_GVPT2():
    """
    """
    molecule = 'FORM'
    method = 'HF'
    basis = 'STO_3G'
    Gamma_rc = 5.1
    list2exclude = []
    terms_selection = [0, 1], [2, 3]

    datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    gParser = GaussianDataParser(datadict)

    dictInputs = {'parserObject': gParser,
                  'el_terms_select': terms_selection[0], 'mech_terms_select': terms_selection[1]}

    spectrumObj = Spectrum2D(omega1, omega2)
    spectrumObj.load_data(dictInputs['parserObject'], vpt2=True, vpt2settings={'anharmonic_type':
                                                                                   'GVPT2'
                                                                               })

    spectrumObj.set_spectrum_settings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=True)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.add_terms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculate_parts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))

    assert all(freqs)


def test_GVPT2_b3lyp_cc_pvqz():
    """
    """
    molecule = 'FORM'
    method = 'B3LYP'
    basis = 'cc_pVQZ'
    Gamma_rc = 5.1
    list2exclude = []
    terms_selection = [0, 1], [2, 3]

    datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    gParser = GaussianDataParser(datadict)

    dictInputs = {'parserObject': gParser,
                  'el_terms_select': terms_selection[0], 'mech_terms_select': terms_selection[1]}

    spectrumObj = Spectrum2D(omega1, omega2)

    spectrumObj.load_data(dictInputs['parserObject'], vpt2=True, vpt2settings={'anharmonic_type':
                                                                                   'GVPT2'
                                                                               })

    spectrumObj.set_spectrum_settings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=False)

    # currently requires diag_margin_rc attribute to be set
    spectrumObj.add_terms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])

    spectrumObj.precalculate_parts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))

    assert all(freqs)


def test_GVPT2_b3lyp_cc_pvtz():
    """
    """
    molecule = 'FORM'
    method = 'B3LYP'
    basis = 'cc_pVTZ'
    Gamma_rc = 5.1
    list2exclude = []
    terms_selection = [0, 1], [2, 3]

    datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    gParser = GaussianDataParser(datadict)

    dictInputs = {'parserObject': gParser,
                  'el_terms_select': terms_selection[0], 'mech_terms_select': terms_selection[1]}

    spectrumObj = Spectrum2D(omega1, omega2)
    spectrumObj.load_data(dictInputs['parserObject'], vpt2=True, vpt2settings={'anharmonic_type':
                                                                                   'GVPT2'
                                                                               })

    spectrumObj.set_spectrum_settings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=True)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.add_terms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculate_parts(list2exclude=list2exclude)


    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))

    assert all(freqs)


def test_GVPT2_b3lyp_cc_pvdz():
    """
    """
    molecule = 'FORM'
    method = 'B3LYP'
    basis = 'cc_pVDZ'
    Gamma_rc = 5.1
    list2exclude = []
    terms_selection = [0, 1], [2, 3]

    datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    gParser = GaussianDataParser(datadict)

    dictInputs = {'parserObject': gParser,
                  'el_terms_select': terms_selection[0], 'mech_terms_select': terms_selection[1]}

    spectrumObj = Spectrum2D(omega1, omega2)
    spectrumObj.load_data(dictInputs['parserObject'], vpt2=True, vpt2settings={'anharmonic_type':
                                                                                   'GVPT2'
                                                                               })

    spectrumObj.set_spectrum_settings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=True)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.add_terms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculate_parts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))

    assert all(freqs)

def test_GVPT2_FOAC_hf_cc_pvqz():
    """
    """
    molecule = 'FOAC'
    method = 'HF'
    basis = 'cc_pVQZ'
    Gamma_rc = 5.1
    list2exclude = []
    terms_selection = [0, 1], [2, 3]

    datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    gParser = GaussianDataParser(datadict)

    dictInputs = {'parserObject': gParser,
                  'el_terms_select': terms_selection[0], 'mech_terms_select': terms_selection[1]}

    spectrumObj = Spectrum2D(omega1, omega2)
    spectrumObj.load_data(dictInputs['parserObject'], vpt2=True, vpt2settings={'anharmonic_type':
                                                                                   'GVPT2'
                                                                               })

    spectrumObj.set_spectrum_settings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=False)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.add_terms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculate_parts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))

    assert all(freqs)


def test_GVPT2_FOAC_b3lyp_cc_pvqz():
    """
    """
    molecule = 'FOAC'
    method = 'B3LYP'
    basis = 'cc_pVQZ'
    Gamma_rc = 5.1
    list2exclude = []
    terms_selection = [0, 1], [2, 3]

    datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    gParser = GaussianDataParser(datadict)

    dictInputs = {'parserObject': gParser,
                  'el_terms_select': terms_selection[0], 'mech_terms_select': terms_selection[1]}

    spectrumObj = Spectrum2D(omega1, omega2)
    spectrumObj.load_data(dictInputs['parserObject'], vpt2=True, vpt2settings={'anharmonic_type':
                                                                                   'GVPT2'
                                                                               })

    spectrumObj.set_spectrum_settings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=False)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.add_terms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculate_parts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))

    assert all(freqs)


def test_GVPT2_FOAC_b3lyp_cc_pvdz():
    """
    """
    molecule = 'FOAC'
    method = 'B3LYP'
    basis = 'cc_pVDZ'
    Gamma_rc = 5.1
    list2exclude = []
    terms_selection = [0, 1], [2, 3]

    datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    gParser = GaussianDataParser(datadict)

    dictInputs = {'parserObject': gParser,
                  'el_terms_select': terms_selection[0], 'mech_terms_select': terms_selection[1]}

    spectrumObj = Spectrum2D(omega1, omega2)
    spectrumObj.load_data(dictInputs['parserObject'], vpt2=True, vpt2settings={'anharmonic_type':
                                                                                   'GVPT2'
                                                                               })

    spectrumObj.set_spectrum_settings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=False)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.add_terms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculate_parts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))

    assert all(freqs)


def test_GVPT2_OXAC2_b3lyp_cc_pvqz():
    """
    """
    molecule = 'OXAC2'
    method = 'B3LYP'
    basis = 'cc_pVQZ'
    Gamma_rc = 5.1
    list2exclude = []
    terms_selection = [0, 1], [2, 3]

    datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    gParser = GaussianDataParser(datadict)

    dictInputs = {'parserObject': gParser,
                  'el_terms_select': terms_selection[0], 'mech_terms_select': terms_selection[1]}

    spectrumObj = Spectrum2D(omega1, omega2)
    spectrumObj.load_data(dictInputs['parserObject'], vpt2=True, vpt2settings={'anharmonic_type':
                                                                                   'GVPT2'
                                                                               })

    spectrumObj.set_spectrum_settings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=False)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.add_terms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculate_parts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))

    assert all(freqs)
