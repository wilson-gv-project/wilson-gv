"""
check_if_jsonsafe - is it a format that can be written in JSON file with json library. 
Generally, it would also mean that the object is serialized into a dictionary here.

Not all classes from abstraction.py are repeated/rewritten here.
Skipped from abstraction.py: CollEvalSetup, WilsonSimulations, SpectralAxisAdvanced (these classes aren't really implemented yet there)

Integration test is in wilson_suitetests/evv_tester_dataclasses.py (different from wilson_suitetests/evv_tester.py only by the import from different abstractions module)
"""
import wilson_main.abstractions as wm_abst_dataclass
import numpy as np

def test_MolecularSystem():

    mol_system_datacls = wm_abst_dataclass.MolecularSystem(name='Mock_datacls', natoms=3)
    assert mol_system_datacls.geo is None
    mol_system_datacls.geo = np.array([[1., -0.3, 2.2], [-1.3, 0.0, -2.1], [0.0, 0.0, -0.1]])

    assert hasattr(mol_system_datacls, 'h')
    assert mol_system_datacls.Nnmodes == 3*3-6


def test_ExternalCalcSetup():
    pass

def test_MolecularProperty():
    pass

def test_MolecularPropertyEncoder():
    """
    Helper class
    """
    pass

def test_VibState():
    pass

def test_VibAnaSetup():
    pass

def test_SpectralAxis():
    pass

def test_SpectralGrid():
    pass

def test_SpecEvalSetup():
    pass

def test_CalculationBatch():
    """
    CalculationBatch is not a dataclass
    """
    pass

def test_WilsonSimulation():
    """
    WilsonSimulation is not a dataclass
    """
    pass