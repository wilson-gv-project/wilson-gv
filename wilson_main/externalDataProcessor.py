"""
Functionality for prepping data from files to WilsonSimulation

"""
from dataclasses import dataclass, field
import numpy as np
from .abstractions import MolecularSystem, ExternalCalcSetup

import logging
logger = logging.getLogger("wilson."+__name__)

@dataclass(frozen=True)
class CalculatedDataFromOutput:
    """
    For a given calculation (by self.system, self.calc_setup). 
    Would have an associated input for a specific program.
    Holds possible molecular properties data and VibAnaSetup data.

    Immutable (frozen=True)

    tuple[np.ndarray, str, str] - data, basis, units
    tuple[np.ndarray, None, str] - data, basis, units
    tuple[np.ndarray, str] - data, units

	# TODO units dictionary? basis? 
    # TODO validation of consistent units/bases? should be part of WilsonSimulation pipeline
    """
    hess: tuple[np.ndarray, str, str] = None
    cff: tuple[np.ndarray, str, str] = None
    qff: tuple[np.ndarray, str, str] = None
    dipgrad: tuple[np.ndarray, str, str] = None
    diphess: tuple[np.ndarray, str, str] = None
    polgrad: tuple[np.ndarray, str, str] = None
    polhess: tuple[np.ndarray, str, str] = None
    B: tuple[np.ndarray, None, str] = None
    coriolis: tuple[np.ndarray, str, str] = None
    anharmonic_states: dict = None
    harmonic_states: dict = None
    normal_modes: tuple[np.ndarray, str] = None
    system: MolecularSystem = None
    calc_setup: ExternalCalcSetup = None

    def h(self):
        return hash((self.system, self.calc_setup))
    
    def __hash__(self):
        return hash((self.system, self.calc_setup))

    def __eq__(self, other):
        if not isinstance(other, CalculatedDataFromOutput):
            return False
        
        return self.system == other.system and self.calc_setup == self.calc_setup
    

@dataclass
class CalcDataStorage:
    """
    This looks more like "vault" now.
    
    systems: list [MolecularSystem, ...]
    setups: list [ExternalCalcSetup]
    data: dict {CalculatedDataFromOutput.h(): CalculatedDataFromOutput} ==
               {hash((self.system, self.calc_setup)): CalculatedDataFromOutput}

    could also store/generate inputs for QC programs ID-ing this way
    """
    systems: list = field(default_factory=lambda: list())
    setups: list = field(default_factory=lambda: list())
    data: dict = field(default_factory=lambda: dict())

    def getbySystem(self):
        pass

    def getbyCalcSetup(self):
        pass

    def getbySysCalc(self, system: MolecularSystem, calc_setup: ExternalCalcSetup):
        """
        returns None if key not in data dict
        """
        return self.data.get(hash((system, calc_setup)))

    def addResult(self, calc_data: CalculatedDataFromOutput):
        system, calc_setup = calc_data.system, calc_data.calc_setup
        if hash((system, calc_setup)) in self.data:
            logger.warning('Data is already registered for:'+
                           f'\n  system: {system.name}'
                           f'\n  level of theory: {calc_setup.lvl_theory}'
                           f'\n  basis set: {calc_setup.basis}'
                           f'\n  program: {calc_setup.program}')
        else:
            self.data[hash((system, calc_setup))] = calc_data


def retrieveFromCQCParse():
    pass

def setUpCQCParsing():
    pass