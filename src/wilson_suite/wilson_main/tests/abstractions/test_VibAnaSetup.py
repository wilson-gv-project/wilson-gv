
from ... import abstractions as wm_abst
from ....wilson_utils.printing import printtest
from ....wilson_utils.logger import setup_logger
import pytest

import copy

import logging
setup_logger("wilson", level=logging.INFO)

def test_VibAnaSetup_init():

    try:
        wm_abst.VibAnaSetup()
    except Exception as e:
        pytest.fail(f"VibAnaSetup initialization raised an exception: {e}")

def test_VibAnaSetup_emptyinit_exclude_modes():
    """
    No parameters VibAnaSetup init.
    modes_indices attribute is no constructed.
    """
    vibana = wm_abst.VibAnaSetup()

    assert vibana.exclude_modes is None
    assert not hasattr(vibana, 'modes_indices')

def test_VibAnaSetup_nosystem_exclude_modes(caplog):
    """
    Checking if a warning for VibAnaSetup().exclude_modes works

    No system but with exclude_modes.
    modes_indices make no sence without a system, so are not constructed
    """
    with caplog.at_level(logging.INFO, logger="wilson.wilson_main.abstractions"):
        vibana2 = wm_abst.VibAnaSetup(exclude_modes=[])

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "VibAnaSetup().exclude_modes attribute is not meaningfull without having set the VibAnaSetup().system attribute"

    assert vibana2.exclude_modes == []
    assert not hasattr(vibana2, 'modes_indices')

def test_VibAnaSetup_system_exclude_modes():

    mol_system = wm_abst.MolecularSystem(name='Mock_datacls', natoms=3)
    vibana2 = wm_abst.VibAnaSetup(system=mol_system, exclude_modes=[])

    assert vibana2.exclude_modes == []
    assert hasattr(vibana2, 'modes_indices')

def mock_anhanalyser():
    """
    dummy anharmonic_analyzer function
    """
    return

'''
def test_VAS_anharm_context_regime():
    vibana = wm_abst.VibAnaSetup()
    try:
        vibana.doAnharmonicAnalysis(props=[], anharmonic_analyzer=mock_anhanalyser)
        pytest.fail('Should have raised an error without set attribute VibAnaSetup().regime')
    except AssertionError as e:
        assert "Vibrational analysis cannot be carried out without having chosen an analysis regime" in str(e), \
            f"Unexpected error message: {e}"

def test_VAS_anharm_context_system():
    vibana = wm_abst.VibAnaSetup()
    vibana.regime = 'GVPT2'
    try:
        vibana.doAnharmonicAnalysis(props=[], anharmonic_analyzer=mock_anhanalyser)
        pytest.fail('Should have raised an error without set attribute VibAnaSetup().regime')
    except AssertionError as e:
        assert "Vibrational analysis cannot be carried out without having set the system attribute" in str(e), \
            f"Unexpected error message: {e}"

def test_VAS_anharm_context_regime_name():
    """
    Should validate 'regime' choice?
    """
    vibana = wm_abst.VibAnaSetup()
    vibana.regime = 4
    try:
        vibana.doAnharmonicAnalysis(props=[], anharmonic_analyzer=mock_anhanalyser)
        pytest.fail('Should have raised an error without set attribute VibAnaSetup().regime')
    except NotImplementedError as e:
        assert 'Implemented regime choices are: "GVPT2", "VPT2"' in str(e), \
            f"Unexpected error message: {e}"

def test_VAS_anharm_context_regime_harm():
    """
    anharmonic_analyzer should be a function
    """
    mol_system = wm_abst.MolecularSystem(name='Mock_datacls', natoms=3)
    vibana = wm_abst.VibAnaSetup()
    vibana.system = mol_system
    vibana.regime = 'harmonic'
    try:
        vibana.doAnharmonicAnalysis(props=[], anharmonic_analyzer=mock_anhanalyser)
        pytest.fail('Should have raised an error without set attribute VibAnaSetup().regime')
    except ValueError as e:
        assert 'Anharmonic analysis requested but chosen vibrational regime is harmonic.' in str(e), \
            f"Unexpected error message: {e}"
        
def test_VAS_anharm_analyzerfunc():
    """
    anharmonic_analyzer should be a function
    """
    mol_system = wm_abst.MolecularSystem(name='Mock_datacls', natoms=3)
    vibana = wm_abst.VibAnaSetup()
    vibana.system = mol_system
    vibana.regime = 'GVPT2'
    try:
        vibana.doAnharmonicAnalysis(props=[], anharmonic_analyzer=copy.copy(mol_system))
        pytest.fail('Should have raised an error without set attribute VibAnaSetup().regime')
    except TypeError as e:
        assert 'anharmonic_analyzer should be a function' in str(e), \
            f"Unexpected error message: {e}"

def test_VAS_anharm_nc_sqrt_eigval():
    """
    anharmonic_analyzer should be a function
    """
    mol_system = wm_abst.MolecularSystem(name='Mock_datacls', natoms=3)
    vibana = wm_abst.VibAnaSetup()
    vibana.system = mol_system
    vibana.regime = 'GVPT2'
    try:
        vibana.doAnharmonicAnalysis(props=[], anharmonic_analyzer=mock_anhanalyser)
        pytest.fail('Should have raised an error without set attribute VibAnaSetup().regime')
    except ValueError as e:
        assert 'Missing values for nc_sqrt_eigval, cannot proceed with anharmonic analysis' in str(e), \
            f"Unexpected error message: {e}"
'''