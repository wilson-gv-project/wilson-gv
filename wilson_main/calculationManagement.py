"""
User cases:

CalculatedData has all parts that are associated with a setup

Notes:
- a set of results is identified by hash(self.system, self.calc_setup)
- a CalculationBatch requests data for a given calc_setup - relates to inputs and outputs
- eval_by_prop_name is a dictionary {trivial name: ExternalCalcSetup}.
For given property request a specific ExternalCalcSetup
- eval_uniform - one ExternalCalcSetup for all - [a special case of eval_by_prop_name]
eval_by_prop_name - be constructed using eval_uniform setup
- 
"""
# NOTE - is itself imported to .abstractions
from .abstractions import VibState, MolecularSystem, ExternalCalcSetup, MolecularProperty
import numpy as np
from dataclasses import dataclass

import logging
logger = logging.getLogger("wilson."+__name__)

@dataclass(frozen=True)
class CalculatedDataFromOutput:
    """
    For a given calculation (by self.system, self.calc_setup). 
    Would have an associated input for a specific program.
    Holds possible molecular properties data and VibAnaSetup data.

    Immutable (frozen=True)
    """
    hess: np.ndarray
    cff: np.ndarray
    qff: np.ndarray
    dipgrad: np.ndarray
    diphess: np.ndarray
    polgrad: np.ndarray
    polhess: np.ndarray
    B: np.ndarray
    coriolis: np.ndarray
    nc_sqrt_eigval: np.ndarray
    nc_eigvec: np.ndarray
    system: MolecularSystem
    calc_setup: ExternalCalcSetup

    def h(self):
        return hash(self.system, self.calc_setup)

@dataclass
class CalcDataStorage:
    """
    This looks more like "vault" now.
    
    systems: list [MolecularSystem, ...]
    setups: list [ExternalCalcSetup]
    data: dict {CalculatedDataFromOutput.h(): CalculatedDataFromOutput} ==
               {hash(self.system, self.calc_setup): CalculatedDataFromOutput}

    could also store/generate inputs for QC programs ID-ing this way
    """
    systems: list
    setups: list
    data: dict

    def getbySystem(self):
        pass

    def getbyCalcSetup(self):
        pass

    def getbySysCalc(self, syscalcTuple: tuple[MolecularSystem, ExternalCalcSetup]):
        """
        returns None if key not in data dict
        """
        return self.data.get(hash(*syscalcTuple))

    def addResult(self, calc_data: CalculatedDataFromOutput, 
                  syscalcTuple: tuple[MolecularSystem, ExternalCalcSetup]):
        system, calc_setup = syscalcTuple
        if hash(system, calc_setup) in self.data:
            logger.warning('Data is already registered for:'+
                           f'\n  system: {system.name}'
                           f'\n  level of theory: {calc_setup.lvl_theory}'
                           f'\n  basis set: {calc_setup.basis}'
                           f'\n  program: {calc_setup.program}')
        else:
            self.data[hash(system, calc_setup)] = calc_data


def getPropVals(system: MolecularSystem, 
                props_to_fill: list[MolecularProperty], 
                eval_by_prop_name: dict, calcdatasets: CalcDataStorage):
    """
    For all props_to_fill add vals from their respective calc_setup for this system

    eval_by_prop_name: dict {trivial name: ExternalCalcSetup}
    calcdatasets: dict {ExternalCalcSetup.h(): CalculatedDataFromOutput}
    """
    # for i in props_to_fill:
    #     # if property belongs to this CalculationBatch
    #     if i.calc_setup.h() == calc_batch.calc_setup.h():
    #         i.addValues(getattr(calcData, i.trivial_name))

    pass

def getResultsToSimulation(calc_batch, 
                           props_to_fill,
                           vib_ana_setup_to_fill,
                           sources: dict):
    """
    Based on CalculationBatch.getResults.
    Fill in props input and vibanasetup; 
    uses batch hash to connect property to the batch

def getResults(self, 
            props_to_fill: list[MolecularProperty],
            vib_ana_setup_to_fill: VibAnaSetup=None, 
            source_type: str='',
            source_types: list[str]=[], 
            source_loc: Any=None, 
            datavault: Any = None):
    
    Example use:
self.calc_batches[i].getResults(self.props, source_type=source_type, source_loc=source_loc, datavault=datavault)

    """
    calcData = sources.get('calcData')

    #! i is an instance of MolecularProperty
    for i in props_to_fill:
        # if property belongs to this CalculationBatch
        if i.calc_setup.h() == calc_batch.calc_setup.h():
            i.addValues(getattr(calcData, i.trivial_name))

    #! vib_ana_setup_to_fill is an instance of VibAnaSetup
    if vib_ana_setup_to_fill is not None:

        # Take harmonic vibrational analysis results
        # FIXME? why not also for 'all'?
        if vib_ana_setup_to_fill.vibana_prop_need in ['none', 'anharm']:

            # vib_ana_setup_to_fill.nc_sqrt_eigval = calcData.fundamentals_harmonic_int # todo: tests...
            vib_ana_setup_to_fill.nc_sqrt_eigval = calcData.nc_sqrt_eigval # todo: tests...

            if not vib_ana_setup_to_fill.allow_skip_eigvec:
                # FIXME: Find out if these are proper coordinates (and precision) for the intended use (transformation)
                if calcData.normal_modes is None:
                    raise AssertionError('Normal coordinates (eigenvectors) not found')
                vib_ana_setup_to_fill.nc_eigvec = calcData.normal_modes

        # Take states
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

                    # TODO: Exclusion based on mode index or freq cutoff
                    # FIXME: Change to integer indexing
                    processed_states.append(VibState(s={i: 1.0}, e=extracted_states[i]))
            vib_ana_setup_to_fill.states = processed_states

