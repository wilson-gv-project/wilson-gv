from wilson_utils.logger import setup_logger
import numpy as np

from wilson_utils.paths import SUITE_ROOT

import wilson_main.externalDataProcessor as dataprc

import logging
setup_logger("wilson", level=logging.DEBUG)

# gaussian fixtures
logfile = SUITE_ROOT+'/wilson_intensities/tests/test_database/dftGaussian/FORM/B3LYPcc_pVQZ/g16_inputFull_3q.out'

datafile_dict_g16 = {'molecule': 'FORM', 'method': 'B3LYP', 'basis': 'cc-pVQZ', 
            'program': 'gaussian', 'log_file': logfile}
cd_from_files1 = dataprc.prepareDataFromFiles(datafile_dict_g16)

# CFOUR fixtures
out_file = SUITE_ROOT+'/wilson_intensities/tests/test_database/refinedc4/CCSDTcc_pVQZ/out'
polar_file = SUITE_ROOT+'/wilson_intensities/tests/test_database/refinedc4/CCSDTcc_pVQZ/polar.pkl'
dipole_file = SUITE_ROOT+'/wilson_intensities/tests/test_database/refinedc4/CCSDTcc_pVQZ/dipole'
cubic_file = SUITE_ROOT+'/wilson_intensities/tests/test_database/refinedc4/CCSDTcc_pVQZ/cubic'
quartic_file = SUITE_ROOT+'/wilson_intensities/tests/test_database/refinedc4/CCSDTcc_pVQZ/quartic'
molden_f = SUITE_ROOT+'/wilson_intensities/tests/test_database/refinedc4/CCSDTcc_pVQZ/MOLDEN'

datafile_dict_c4 = {'molecule': 'FORM', 'method': 'CCSD(T)', 'basis': 'cc-pVQZ', 
                    'program': 'cfour', 'out_file': out_file, 'molden_file':molden_f,
                    'cubic_file': cubic_file, 'quartic_file': quartic_file, 
                    'dipole_file': dipole_file, 'polar_pkl': polar_file}
cd_from_files2 = dataprc.prepareDataFromFiles(datafile_dict_c4)



def test_CalculatedDataFromOutput():
    from wilson_suite.calcdata import datadict1
    cd1 = dataprc.CalculatedDataFromOutput(**datadict1)
    assert getattr(cd1, 'dipgrad') == datadict1['dipgrad']
    assert getattr(cd1, 'coriolis') == datadict1['coriolis']


def test_prepareDataFromFiles_g16():

    assert np.allclose(getattr(cd_from_files2, 'B'), [1.1507743, 1.3090557, 9.5174003])
    assert {k:v for k,v in getattr(cd_from_files2, 'harmonic_states').items() if len(k)==1} == {('0',): 2878.687, ('1',): 1820.416, 
                                                                                               ('2',): 1534.549, ('3',): 1203.179, 
                                                                                               ('4',): 2933.526, ('5',): 1268.91}

def test_prepareDataFromFiles_c4():

    assert np.allclose(getattr(cd_from_files1, 'B'), [1.14505171, 1.30063729, 9.57220394])
    assert {k:v for k,v in getattr(cd_from_files1, 'harmonic_states').items() if len(k)==1} == {('0',): 1195.515081, ('1',): 1278.448678, 
                                                                                               ('2',): 1544.478183, ('3',): 1791.306846, 
                                                                                               ('4',): 2944.822345, ('5',): 3014.592218}
