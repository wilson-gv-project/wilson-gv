import numpy as np
np.set_printoptions(legacy='1.25')

from wilson.spectrum.spectrum2D import Spectrum2D
from CQCParse.parsing import GaussianDataParser
from CQCParse.relay import DataVault

data_vault = DataVault('/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/files_database.csv')

omega1 = np.arange(1130., 2050., 2.91)
omega2 = np.arange(1300., 5150., 2.91)


def test_corrected_levels():
    molecule = 'FORM'  # METH, ACDM, ACAC, ACDM, FORM, FOAC, OXAC1, OXAC2
    method = 'HF'
    basis = 'STO_3G_VPT2'  # 'STO_3G_VPT2'
    Gamma_rc = 5.1
    list2exclude = []
    terms_selection = [0, 1], [2, 3]

    datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    gParser = GaussianDataParser(datadict)

    dictInputs = {'parserObject': gParser,
                  'el_terms_select': terms_selection[0], 'mech_terms_select': terms_selection[1]}

    spectrumObj = Spectrum2D(omega1, omega2)
    spectrumObj.load_data(dictInputs['parserObject'], vpt2=True, vpt2settings={'anharmonic_type':'VPT2'})

    spectrumObj.setSpectrumSettings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=True)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.addTerms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculateParts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))
        # print(spectrumObj.all_states[i], spectrumObj.all_states_corr[i])
        assert np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3)
    assert all(freqs)


def test_corrected_levels_default0():
    """
    """
    molecule = 'FORM'  # METH, ACDM, ACAC, ACDM, FORM, FOAC, OXAC1, OXAC2
    method = 'HF' # 'B3LYP' 'HF'
    basis = 'STO_3G_VPT2'  # 'STO_3G_VPT2' 'cc_pVQZ' 'STO_3G'
    Gamma_rc = 5.1
    list2exclude = []
    terms_selection = [0, 1], [2, 3]

    datadict = data_vault.make_DatainputDict('gaussian', (molecule, method, basis), '')
    gParser = GaussianDataParser(datadict)

    dictInputs = {'parserObject': gParser,
                  'el_terms_select': terms_selection[0], 'mech_terms_select': terms_selection[1]}

    spectrumObj = Spectrum2D(omega1, omega2)
    spectrumObj.load_data(dictInputs['parserObject'], vpt2=True, vpt2settings={'anharmonic_type':
                                                                                   'VPT2'
                                                                               })

    spectrumObj.setSpectrumSettings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=True)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.addTerms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculateParts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        # spectrumObj.all_states (from gaussian) vs spectrumObj.all_states_corr (from vpt2 code)
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))

    assert all(freqs)


def test_corrected_levels_default1():
    """
    """
    molecule = 'FORM'  # METH, ACDM, ACAC, ACDM, FORM, FOAC, OXAC1, OXAC2
    method = 'HF' # 'B3LYP' 'HF'
    basis = 'STO_3G'  # 'STO_3G_VPT2' 'cc_pVQZ' 'STO_3G'
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

    spectrumObj.setSpectrumSettings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=True)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.addTerms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculateParts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))

    assert all(freqs)


def test_corrected_levels_default2():
    """
    """
    molecule = 'FORM'  # METH, ACDM, ACAC, ACDM, FORM, FOAC, OXAC1, OXAC2
    method = 'B3LYP' # 'B3LYP' 'HF'
    basis = 'cc_pVQZ'  # 'STO_3G_VPT2' 'cc_pVQZ' 'STO_3G' 'cc_pVDZ' 'cc_pVDZ_VPT2' 'cc_pVTZ' 'cc_pVQZ'
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

    spectrumObj.setSpectrumSettings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=False)

    # currently requires diag_margin_rc attribute to be set
    spectrumObj.addTerms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])

    spectrumObj.precalculateParts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))

    assert all(freqs)


def test_corrected_levels_default3():
    """
    """
    molecule = 'FORM'  # METH, ACDM, ACAC, ACDM, FORM, FOAC, OXAC1, OXAC2
    method = 'B3LYP' # 'B3LYP' 'HF'
    basis = 'cc_pVTZ'  # 'STO_3G_VPT2' 'cc_pVQZ' 'STO_3G'
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
                                                                                   # 'Anharmonic: VPT2'
                                                                                   # 'Anharmonic: DVPT2'
                                                                               })

    spectrumObj.setSpectrumSettings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=True)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.addTerms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculateParts(list2exclude=list2exclude)

    # print(spectrumObj.nmodes)
    # print([i for i in spectrumObj.all_states.keys() if len(i)==1])
    # print(spectrumObj.all_states)
    # print(spectrumObj.all_states.keys())
    # print('-----------------------')
    # print([i for i in spectrumObj.all_states_corr.keys() if len(i)==1])
    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))
        # print(spectrumObj.all_states[i], spectrumObj.all_states_corr[i])

    print(np.sum(np.array(freqs)))
    print(freqs)
    assert all(freqs)
        # assert np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3)


def test_corrected_levels_default3p():
    """
    """
    molecule = 'FORM'  # METH, ACDM, ACAC, ACDM, FORM, FOAC, OXAC1, OXAC2
    method = 'B3LYP' # 'B3LYP' 'HF'
    basis = 'cc_pVDZ'  # 'STO_3G_VPT2' 'cc_pVQZ' 'STO_3G'
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

    spectrumObj.setSpectrumSettings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=True)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.addTerms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculateParts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))
        # print(spectrumObj.all_states[i], spectrumObj.all_states_corr[i])

    print(np.sum(np.array(freqs)))
    print(freqs)
    assert all(freqs)

def test_corrected_levels_default4():
    """
    """
    molecule = 'FOAC'  # METH, ACDM, ACAC, ACDM, FORM, FOAC, OXAC1, OXAC2
    method = 'HF' # 'B3LYP' 'HF'
    basis = 'cc_pVQZ'  # 'STO_3G_VPT2' 'cc_pVQZ' 'STO_3G'
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
                                                                                   # 'VPT2'
                                                                                   # 'DVPT2'
                                                                               })

    spectrumObj.setSpectrumSettings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=False)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.addTerms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculateParts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))
        # print(spectrumObj.all_states[i], spectrumObj.all_states_corr[i])

    # print(np.sum(np.array(freqs)))
    # print(freqs)
    assert all(freqs)


def test_corrected_levels_default5():
    """
    """
    molecule = 'FOAC'  # METH, ACDM, ACAC, ACDM, FORM, FOAC, OXAC1, OXAC2
    method = 'B3LYP' # 'B3LYP' 'HF'
    basis = 'cc_pVQZ'  # 'STO_3G_VPT2' 'cc_pVQZ' 'STO_3G'
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
                                                                                   # 'VPT2'
                                                                                   # 'DVPT2'
                                                                               })

    spectrumObj.setSpectrumSettings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=False)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.addTerms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculateParts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))
        # print(spectrumObj.all_states[i], spectrumObj.all_states_corr[i])

    # print(np.sum(np.array(freqs)))
    # print(freqs)
    assert all(freqs)


def test_corrected_levels_default6():
    """
    """
    molecule = 'FOAC'  # METH, ACDM, ACAC, ACDM, FORM, FOAC, OXAC1, OXAC2
    method = 'B3LYP' # 'B3LYP' 'HF'
    basis = 'cc_pVDZ'  # 'STO_3G_VPT2' 'cc_pVQZ' 'STO_3G'
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
                                                                                   # 'VPT2'
                                                                                   # 'DVPT2'
                                                                               })

    spectrumObj.setSpectrumSettings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=False)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.addTerms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculateParts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))
        # print(spectrumObj.all_states[i], spectrumObj.all_states_corr[i])

    # print(np.sum(np.array(freqs)))
    # print(freqs)
    assert all(freqs)


def test_corrected_levels_default7():
    """
    """
    molecule = 'OXAC2'  # METH, ACDM, ACAC, ACDM, FORM, FOAC, OXAC1, OXAC2
    method = 'B3LYP' # 'B3LYP' 'HF'
    basis = 'cc_pVQZ'  # 'STO_3G_VPT2' 'cc_pVQZ' 'STO_3G'
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
                                                                                   # 'VPT2'
                                                                                   # 'DVPT2'
                                                                               })

    spectrumObj.setSpectrumSettings(Gamma_rc=Gamma_rc, diag_margin_rc=3., vib_levels_harmonic=False)
    # currently requires diag_margin_rc attribute to be set
    spectrumObj.addTerms(dictInputs['el_terms_select'], dictInputs['mech_terms_select'])
    spectrumObj.precalculateParts(list2exclude=list2exclude)

    freqs = []
    for i in spectrumObj.all_states:
        freqs.append(np.isclose(spectrumObj.all_states[i], spectrumObj.all_states_corr[i], atol=1e-3))
        # print(spectrumObj.all_states[i], spectrumObj.all_states_corr[i])

    # print(np.sum(np.array(freqs)))
    # print(freqs)
    assert all(freqs)
