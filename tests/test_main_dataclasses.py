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
    from wilson_utils.prop_trivname import prop_trivname

    hess = wm_abst_dataclass.MolecularProperty(
					{'ops': tuple(['g', 'g']), 'freq': (0.0, 0.0)},
					trivial_name=prop_trivname(ord_geo=2),
					target_basis='cart',
					target_units='au')
    assert hess.serial_vals is None
    assert hess.vals is None
    assert hess.to_dict() == {'prop_spec': {'ops': ('g', 'g'), 'freq': (0.0, 0.0)}, 
                              'trivial_name': 'hess', 'in_basis': None, 'in_units': None, 
                              'target_basis': 'cart', 'target_units': 'au', 'serial_vals': None}

    rot_const = wm_abst_dataclass.MolecularProperty(
						{'ops': tuple(['r']), 'freq': (0.0)},
						trivial_name=prop_trivname(ord_rot=1),
						target_basis='nm',
						target_units='au')
    assert rot_const.serial_vals is None
    assert rot_const.vals is None
    assert rot_const.to_dict() == {'prop_spec': {'ops': ('r',), 'freq': 0.0}, 
                                   'trivial_name': 'B', 'in_basis': None, 'in_units': None, 
                                   'target_basis': 'nm', 'target_units': 'au', 'serial_vals': None}


    pdict = {'ops': tuple(['g', 'f']), 'freq': tuple([0.0 * k for k in range(len(['g', 'f']))])}
    dipgrad = wm_abst_dataclass.MolecularProperty(pdict, trivial_name=prop_trivname(ord_geo=1, ord_el=1),
													 target_basis='nm', target_units='au')
    assert dipgrad.serial_vals is None
    assert dipgrad.vals is None
    assert dipgrad.to_dict() == {'prop_spec': {'ops': ('g', 'f'), 'freq': (0.0, 0.0)}, 
                                 'trivial_name': 'dipgrad', 'in_basis': None, 'in_units': None, 
                                 'target_basis': 'nm', 'target_units': 'au', 'serial_vals': None}
    
    dipgrad_vals = np.array([[0.67, 0.05, 0.11],
                             [0.42, 0.59, 0.98]])
    dipgrad.addValues(dipgrad_vals)
    assert np.all(dipgrad.vals == dipgrad_vals)
    assert dipgrad.serial_vals is None

    dipgrad.make_serial_vals()
    # keys are strings because it's JSON-compatible
    assert dipgrad.serial_vals == {'(0, 0)': 0.67, '(0, 1)': 0.05, '(0, 2)': 0.11, 
                                   '(1, 0)': 0.42, '(1, 1)': 0.59, '(1, 2)': 0.98}

    # providing vals in init, serial_vals will be made from input vals    
    dipgrad2 = wm_abst_dataclass.MolecularProperty(pdict, trivial_name=prop_trivname(ord_geo=1, ord_el=1),
                                                    target_basis='nm', target_units='au', vals=dipgrad_vals)
    assert dipgrad2.serial_vals == {'(0, 0)': 0.67, '(0, 1)': 0.05, '(0, 2)': 0.11, 
                                    '(1, 0)': 0.42, '(1, 1)': 0.59, '(1, 2)': 0.98}
    

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