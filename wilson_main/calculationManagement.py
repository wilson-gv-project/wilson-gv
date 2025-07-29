"""
User cases:

CalculatedData has all parts that are associated with a setup

Notes:
- a set of results is identified by hash((self.system, self.calc_setup))
- a CalculationBatch requests data for a given calc_setup - relates to inputs and outputs
- eval_by_prop_name is a dictionary {trivial name: ExternalCalcSetup}.
For given property request a specific ExternalCalcSetup
- eval_uniform - one ExternalCalcSetup for all - [a special case of eval_by_prop_name]
eval_by_prop_name - be constructed using eval_uniform setup

Goal:
- construct a calculation using mixed sources of data
"""
# NOTE - is itself imported to .abstractions
from .abstractions import VibState, MolecularSystem, ExternalCalcSetup, MolecularProperty, VibAnaSetup
import numpy as np
from dataclasses import dataclass, field

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
        print(system)
        if hash((system, calc_setup)) in self.data:
            logger.warning('Data is already registered for:'+
                           f'\n  system: {system.name}'
                           f'\n  level of theory: {calc_setup.lvl_theory}'
                           f'\n  basis set: {calc_setup.basis}'
                           f'\n  program: {calc_setup.program}')
        else:
            self.data[hash((system, calc_setup))] = calc_data


def getPropValsStorage(system: MolecularSystem, 
                props_to_fill: list[MolecularProperty], 
                eval_by_prop_name: dict, calcdatasets: CalcDataStorage):
    """
    For all props_to_fill add vals from their respective calc_setup for this system

    eval_by_prop_name: dict {trivial name: ExternalCalcSetup}
    calcdatasets: dict {ExternalCalcSetup.h(): CalculatedDataFromOutput} - is like a valut?

    maybe could be a pure function? should be?
    """
    for i in props_to_fill:
        calc_setup = eval_by_prop_name[i.trivial_name]
        vals, basis, units = calcdatasets.getbySysCalc(system, calc_setup)
        i.addValues(values=getattr(vals, i.trivial_name),
                    in_basis=basis, in_units=units)


def getVibAnaValsStorage(system: MolecularSystem, 
                vib_ana_setup_to_fill: VibAnaSetup, 
                calcdatasets: CalcDataStorage):
    """
    For all props_to_fill add vals from their respective calc_setup for this system

    eval_by_prop_name: dict {trivial name: ExternalCalcSetup}
    calcdatasets: dict {ExternalCalcSetup.h(): CalculatedDataFromOutput} - is like a valut?

    maybe could be a pure function? should be?
    """
    calc_setup = vib_ana_setup_to_fill.external_fill_from
    calcData = calcdatasets.getbySysCalc(system, calc_setup)

    # Take harmonic vibrational analysis results
    # FIXME? why not also for 'all'?
    if vib_ana_setup_to_fill.vibana_prop_need in ['none', 'anharm']:

        vib_ana_setup_to_fill.nc_sqrt_eigval = {k:v for k,v in calcData.harmonic_states.items() if len(k)==1}

        if not vib_ana_setup_to_fill.allow_skip_eigvec:
            # FIXME: Find out if these are proper coordinates (and precision) for the intended use (transformation)
            if calcData.normal_modes is None:
                raise AssertionError('Normal coordinates (eigenvectors) not found')
            vib_ana_setup_to_fill.nc_eigvec = calcData.normal_modes

    # Take states
    # TODO units
	# FIXME vibana_prop_need? confusing name; need for what/whom? I get from context of VibAnaSetup init that "need" means "need to calculate with Wilson internally"
    if vib_ana_setup_to_fill.vibana_prop_need in ['none']:

        if vib_ana_setup_to_fill.regime not in ['harmonic']:
            extracted_states = calcData.anharmonic_states

        else:
            extracted_states = calcData.harmonic_states

        processed_states = []

        # For now taking only "single harmonic oscillator state" states when getting from output file
        # TODO: Add parsing capability for re-resolved states with possible admixtures
        for i in extracted_states:
            if len(i) <= vib_ana_setup_to_fill.max_state_lvl:

                # TODO: Exclusion based on mode index or freq cutoff - when should this happen? should define places for exclusion
                # FIXME: Change to integer indexing - what does it mean?
                processed_states.append(VibState(s={i: 1.0}, e=extracted_states[i]))
        vib_ana_setup_to_fill.states = processed_states

