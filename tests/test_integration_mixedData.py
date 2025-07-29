"""
Goal:
- construct a calculation using mixed sources of data
"""
import wilson_main.abstractions as wm_abst
from wilson_utils.printing import printtest
from wilson_utils.logger import setup_logger

import wilson_main.calculationManagement as manage

import logging
setup_logger("wilson", level=logging.DEBUG)

storage = manage.CalcDataStorage()

mol1 = wm_abst.MolecularSystem(name='mol1', natoms=3)
mol2 = wm_abst.MolecularSystem(name='mol2', natoms=5)

setup1 = wm_abst.ExternalCalcSetup('p1', 'lvl1', 'b1')
setup2 = wm_abst.ExternalCalcSetup('p1', 'lvl2', 'b2')
setup3 = wm_abst.ExternalCalcSetup('p1', 'lvl2', 'b3')
setup4 = wm_abst.ExternalCalcSetup('p2', 'lvl3', 'b2')

datadict1 = {'system': mol1, 'calc_setup': setup1, 
             'hess': None, 'cff': None, 'qff': None, 
             'dipgrad': None, 'diphess': None, 'polgrad': None, 'polhess': None}

datadict2 = {'system': mol2, 'calc_setup': setup2, 
             'hess': None, 'cff': None, 'qff': None, 
             'dipgrad': None, 'diphess': None, 'polgrad': None, 'polhess': None}

datadict3 = {'system': mol1, 'calc_setup': setup3, 
             'hess': None, 'cff': None, 'qff': None, 
             'dipgrad': None, 'diphess': None, 'polgrad': None, 'polhess': None}

datadict4 = {'system': mol2, 'calc_setup': setup3, 
             'hess': None, 'cff': None, 'qff': None, 
             'dipgrad': None, 'diphess': None, 'polgrad': None, 'polhess': None}

datadict5 = {'system': mol2, 'calc_setup': setup4, 
             'hess': None, 'cff': None, 'qff': None, 
             'dipgrad': None, 'diphess': None, 'polgrad': None, 'polhess': None}

# NOTE manage.CalculatedDataFromOutput(datadict1) - then need keys in dict to be in order
# NOTE manage.CalculatedDataFromOutput(**datadict1) - doesn't need ordered keys

cd1 = manage.CalculatedDataFromOutput(**datadict1)
cd2 = manage.CalculatedDataFromOutput(**datadict2)
cd3 = manage.CalculatedDataFromOutput(**datadict3)
cd4 = manage.CalculatedDataFromOutput(**datadict4)
cd5 = manage.CalculatedDataFromOutput(**datadict5)

storage.addResult(cd1)
storage.addResult(cd2)
storage.addResult(cd3)
storage.addResult(cd4)
storage.addResult(cd5)


def test_mixed_sources_calc():
    """
    What should happen:

    - wilsonsim is set up for getting result from mixed data calculations

    - there would be several CalculationBatches
    - molprops would have different sources (should keep that info with them somehow, point to a system,calcsetup hash?)
    - vib states would also have another data source/calcsetup

    should be possible to get info about sources of data
    """

    pass