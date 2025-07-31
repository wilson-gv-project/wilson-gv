import wilson_main.abstractions as wm_abst
from wilson_utils.printing import printtest
from wilson_utils.logger import setup_logger
import numpy as np

from wilson_utils.paths import SUITE_ROOT

import wilson_main.externalDataProcessor as dataprc

import logging
setup_logger("wilson", level=logging.DEBUG)

def test_CalculatedDataFromOutput():
    from wilson_suite.calcdata import datadict1
    cd1 = dataprc.CalculatedDataFromOutput(**datadict1)
    assert getattr(cd1, 'dipgrad') == datadict1['dipgrad']
    assert getattr(cd1, 'coriolis') == datadict1['coriolis']

def test_prepareDataFromFiles():
    logfile = SUITE_ROOT+'/wilson_intensities/tests/test_database/dftGaussian/FORM/B3LYPcc_pVQZ/g16_inputFull_3q.out'

    datafile_dict_g16 = {'molecule': 'FORM', 'method': 'B3LYP', 'basis': 'cc-pVQZ', 
                'program': 'gaussian', 'log_file': logfile}
    cd_from_files = dataprc.prepareDataFromFiles(datafile_dict_g16)
    assert np.allclose(getattr(cd_from_files, 'B'), [1.1507743, 1.3090557, 9.5174003])
    assert {k:v for k,v in getattr(cd_from_files, 'harmonic_states').items() if len(k)==1} == {('0',): 2878.687, ('1',): 1820.416, 
                                                                                               ('2',): 1534.549, ('3',): 1203.179, 
                                                                                               ('4',): 2933.526, ('5',): 1268.91}

def test_CalcDataStorage_getbySysCalc():
    pass

def test_CalcDataStorage_addResult():
    pass
