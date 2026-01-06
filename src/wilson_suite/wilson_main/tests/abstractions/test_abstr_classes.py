"""
check_if_jsonsafe - is it a format that can be written in JSON file with json library. 
Generally, it would also mean that the object is serialized into a dictionary here.

Not all classes from abstraction.py are repeated/rewritten here.
Skipped from abstraction.py: CollEvalSetup, WilsonSimulations, SpectralAxisAdvanced (these classes aren't really implemented yet there)

Integration test is in wilson_suitetests/evv_tester_dataclasses.py (different from wilson_suitetests/evv_tester.py only by the import from different abstractions module)
"""
from ... import abstractions as wm_abst
import numpy as np
from ....wilson_utils.logger import setup_logger

import logging
setup_logger("wilson", level=logging.DEBUG)

def test_MolecularSystem():

    mol_system_datacls = wm_abst.MolecularSystem(name='Mock_datacls', natoms=3)
    assert mol_system_datacls.geo is None
    mol_system_datacls.geo = np.array([[1., -0.3, 2.2], [-1.3, 0.0, -2.1], [0.0, 0.0, -0.1]])

    assert mol_system_datacls.Nnmodes == 3*3-6


def test_DataOriginInfo():
    """To be implemented"""
    pass

def test_MolecularProperty():

    from ....wilson_utils.prop_trivname import prop_trivname

    hess = wm_abst.MolecularProperty(
					{'ops': tuple(['g', 'g']), 'freq': (0.0, 0.0)},
					trivial_name=prop_trivname(ord_geo=2))
    assert hess.vals is None
    assert hess.to_dict() == {'prop_spec': {'ops': ('g', 'g'), 'freq': (0.0, 0.0)}, 
                              'trivial_name': 'hess'}

    rot_const = wm_abst.MolecularProperty(
						{'ops': tuple(['r']), 'freq': (0.0)},
						trivial_name=prop_trivname(ord_rot=1))
    assert rot_const.vals is None
    assert rot_const.to_dict() == {'prop_spec': {'ops': ('r',), 'freq': 0.0}, 
                                   'trivial_name': 'B'}


    pdict = {'ops': tuple(['g', 'f']), 'freq': tuple([0.0 * k for k in range(len(['g', 'f']))])}
    dipgrad = wm_abst.MolecularProperty(pdict, trivial_name=prop_trivname(ord_geo=1, ord_el=1))
    assert dipgrad.vals is None
    assert dipgrad.to_dict() == {'prop_spec': {'ops': ('g', 'f'), 'freq': (0.0, 0.0)}, 
                                 'trivial_name': 'dipgrad'}
    
    dipgrad_vals = np.array([[0.67, 0.05, 0.11],
                             [0.42, 0.59, 0.98]])
    dipgrad.addValues(dipgrad_vals)
    assert np.all(dipgrad.vals == dipgrad_vals)

    # keys are strings because it's JSON-compatible

    # providing vals in init, serial_vals will be made from input vals    
    dipgrad2 = wm_abst.MolecularProperty(pdict, trivial_name=prop_trivname(ord_geo=1, ord_el=1), vals=dipgrad_vals)
    

def test_MolecularPropertyEncoder():
    """To be implemented"""
    pass

def test_VibState():
    """To be implemented"""
    pass

def test_SpectralAxis():
    """To be implemented"""
    pass

def test_SpectralGrid():
    """To be implemented"""
    pass

def test_SpecEvalSetup():
    """To be implemented"""
    pass

def test_CalculationBatch():
    """To be implemented"""
    pass

def test_WilsonSimulation():
    """To be implemented"""
    pass