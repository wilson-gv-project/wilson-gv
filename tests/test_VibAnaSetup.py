
import wilson_main.abstractions as wm_abst
import numpy as np
from wilson_utils.printing import printtest
from wilson_utils.logger import setup_logger
import pytest

import logging
setup_logger("wilson", level=logging.DEBUG)

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

    # without system provided    
    assert vibana.exclude_modes is None
    # modes_indices make no sence without a system, so are not constructed
    assert not hasattr(vibana, 'modes_indices')

def test_VibAnaSetup_nosystem_exclude_modes(caplog):
    """
    Checking if a warning for VibAnaSetup().exclude_modes works
    """
    with caplog.at_level(logging.INFO, logger="wilson.wilson_main.abstractions"):
        vibana2 = wm_abst.VibAnaSetup(exclude_modes=[])

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "VibAnaSetup().exclude_modes attribute is not meaningfull without having set the VibAnaSetup().system attribute"

    # without system provided    
    assert vibana2.exclude_modes == []
    # modes_indices make no sence without a system, so are not constructed
    assert not hasattr(vibana2, 'modes_indices')

def test_VibAnaSetup_system_exclude_modes():

    mol_system = wm_abst.MolecularSystem(name='Mock_datacls', natoms=3)

    vibana2 = wm_abst.VibAnaSetup(system=mol_system, exclude_modes=[])

    # without system provided    
    assert vibana2.exclude_modes == []
    # modes_indices make no sence without a system, so are not constructed
    assert hasattr(vibana2, 'modes_indices')

    # vibana = ws_main.abstractions.VibAnaSetup(system=mol_system, regime='GVPT2',
    #                                         vibana_prop_need='anharm', # should this vary? take minimal needed for regime unless specified? 
    #                                         allow_skip_eigvec=True, external_fill_from=calc_setup)
    
