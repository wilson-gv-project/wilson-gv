from calculations.parseGaussian_forWilson import *
from calculations.parseCFOUR_forWilson import *

def test_anharmonicHF_DZ_freqs():

    datadict_gaussian = {'log': '/mnt/c/Users/vle014/Downloads/files_fram/dftGaussian/FORM/HFcc_pVDZ/g16_inputFull_3q.out',
                '3quanta': '/mnt/c/Users/vle014/Downloads/files_fram/dftGaussian/FORM/HFcc_pVDZ/g16_inputFull_3q.out'}
    parserGaussian = GaussianDataParser({'files': datadict_gaussian})
    parserGaussian.getData()

    essential_gaussian = [parserGaussian.harmonic_states,
                          parserGaussian.anharmonic_states,
                          parserGaussian.fundamentals_harmonic_str,
                          parserGaussian.fundamentals_anharmonic_str,
                          ]
    g16_harmFreqs = sorted(list(essential_gaussian[2].values()))
    g16_anharmFreqs = sorted(list(essential_gaussian[3].values()))
    print('\nharmonic_states', g16_harmFreqs)
    print('\nanharmonic_states', g16_anharmFreqs)

    datadict_cfour = {'out_anharm_final': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVDZ/out',
                      'dipolexyz': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVDZ/dipole',
                      'polar_pkl': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVDZ/polar.pkl',
                      'cubic': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVDZ/cubic'}
    parserC4 = CFOURdataParser({'files':datadict_cfour})
    parserC4.getData()
    essential_CFOUR = [parserC4.harmonic_states,
                       parserC4.anharmonic_states,
                       parserC4.fundamentals_harmonic_str,
                       parserC4.fundamentals_anharmonic_str,
                       ]
    c4_harmFreqs = sorted(list(essential_CFOUR[2].values()))
    c4_anharmFreqs = sorted(list(essential_CFOUR[3].values()))
    print('\nharmonic_states', c4_harmFreqs)
    print('\nanharmonic_states', c4_anharmFreqs)

    assert g16_anharmFreqs != c4_anharmFreqs
    assert np.allclose(g16_harmFreqs, c4_harmFreqs, atol=10**(-4))

def test_anharmonicHF_QZ_freqs():

    datadict_gaussian = {'log': '/mnt/c/Users/vle014/Downloads/files_fram/dftGaussian/FORM/HFcc_pVQZ/g16_inputFull_3q.out',
                '3quanta': '/mnt/c/Users/vle014/Downloads/files_fram/dftGaussian/FORM/HFcc_pVQZ/g16_inputFull_3q.out'}
    parserGaussian = GaussianDataParser({'files': datadict_gaussian})
    parserGaussian.getData()

    essential_gaussian = [parserGaussian.harmonic_states,
                          parserGaussian.anharmonic_states,
                          parserGaussian.fundamentals_harmonic_str,
                          parserGaussian.fundamentals_anharmonic_str,
                          ]
    g16_harmFreqs = sorted(list(essential_gaussian[2].values()))
    g16_anharmFreqs = sorted(list(essential_gaussian[3].values()))
    print('\nharmonic_states', g16_harmFreqs)
    print('\nanharmonic_states', g16_anharmFreqs)

    datadict_cfour = {'out_anharm_final': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVQZ/out',
                      'dipolexyz': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVQZ/dipole',
                      'polar_pkl': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVQZ/polar.pkl',
                      'cubic': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVQZ/cubic'}
    parserC4 = CFOURdataParser({'files':datadict_cfour})
    parserC4.getData()
    essential_CFOUR = [parserC4.harmonic_states,
                       parserC4.anharmonic_states,
                       parserC4.fundamentals_harmonic_str,
                       parserC4.fundamentals_anharmonic_str,
                       ]

    c4_harmFreqs = sorted(list(essential_CFOUR[2].values()))
    c4_anharmFreqs = sorted(list(essential_CFOUR[3].values()))
    print('\nharmonic_states', c4_harmFreqs)
    print('\nanharmonic_states', c4_anharmFreqs)

    assert g16_anharmFreqs != c4_anharmFreqs
    assert np.allclose(g16_harmFreqs, c4_harmFreqs, atol=10**(-4))

def test_anharmonicHF_CFF():
    pass

def test_anharmonicHF_dipoleF():
    pass

def test_anharmonicHF_dipoleS():
    pass

