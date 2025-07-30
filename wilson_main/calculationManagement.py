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
# NOTE - VibState is itself imported to .abstractions
from .abstractions import VibState, MolecularSystem, ExternalCalcSetup, MolecularProperty, VibAnaSetup, CalculationBatch
import numpy as np
from dataclasses import dataclass, field
import copy

from wilson_utils.prop_trivname import prop_trivname

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
        if hash((system, calc_setup)) in self.data:
            logger.warning('Data is already registered for:'+
                           f'\n  system: {system.name}'
                           f'\n  level of theory: {calc_setup.lvl_theory}'
                           f'\n  basis set: {calc_setup.basis}'
                           f'\n  program: {calc_setup.program}')
        else:
            self.data[hash((system, calc_setup))] = calc_data


class AttributeIndex:
    """
    Sets up a retrieval by attribute from a list of custom class instances

    objects: list of class instances

    Example:
    index_molecules = AttributeIndex(molecules)

    index_molecules.get_by("name", "Methane")    #  MolecularSystem(...)
    index_molecules.get_by("natoms", 4)          #  MolecularSystem(...)
    
    [here seems unnecessary, but generally allows to retrieve by different attributes]
    """
    def __init__(self, objects: list):
        self.objects = objects
        self._indexes = {}

    def get_by(self, attr, value):
        """
        retrieve an instance by value of an attribute

        attr is an atribute of the class of objects
        """
        if attr not in self._indexes:
            self._indexes[attr] = {getattr(obj, attr): obj for obj in self.objects}
        return self._indexes[attr].get(value)


def getPropValsFromStorage(system: MolecularSystem, 
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
        entry = calcdatasets.getbySysCalc(system, calc_setup)
        if entry is None:
            raise ValueError(f"No data found for system={system.name}, calc_setup={calc_setup}")
        try:
            vals, basis, units = getattr(entry, i.trivial_name)
            i.addValues(values=vals, in_basis=basis, in_units=units)
            
            # fills in serial_vals attribute; doesn't have to be done here
            i.make_serial_vals()

        # when entry is None
        except TypeError:
            logger.info(f' --> Attention! Did not find results for: {i.trivial_name}; with calc_setup: {calc_setup}')


def getVibAnaValsFromStorage(system: MolecularSystem, vib_ana_setup_to_fill: VibAnaSetup, calcdatasets: CalcDataStorage):
    """
    vib_ana_setup_to_fill

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

def groupDataForCalcSetups(vibanasetup, calc_props_setup, props_needed: list[MolecularProperty]):
    """
    Similar to WilsonSimulation.makeCalculationBatches()

    Collect all comp setups from settings and group into batches.
    Expected calc setups are in vibanasetup and eval_by_prop_name or eval_uniform (?) 
    FIXME? [could generalize and do only eval_by_prop_name?]
    """
    collected = copy.deepcopy(calc_props_setup)
    
    if vibanasetup.external_fill_from is not None:
        collected['vibana'] = vibanasetup.external_fill_from
    
    missing_props = [i.trivial_name for i in props_needed if i.trivial_name not in calc_props_setup]
    
    if missing_props:
        raise ValueError(f"Some properties do not have a calculation setup associated with them: {missing_props}")
    
    # index of props_needed = list[MolecularProperty]
    index_props_needed = AttributeIndex(props_needed)
    
    grouped_names = {}
    for p in collected:
        if collected[p] not in grouped_names:
            grouped_names[collected[p]] = []
        grouped_names[collected[p]].append(p)
    
    logger.debug('grouped_names')
    logger.debug(grouped_names)

    grouped = {}
    for p in grouped_names:
        grouped[p] = [index_props_needed.get_by('trivial_name', i) for i in grouped_names[p]]

    return grouped


def makeBatchesFromGroups(system, grouped_calcs):

    return [CalculationBatch(system=system, calc_setup=calc_setup, properties=grouped_calcs[calc_setup]) for calc_setup in grouped_calcs]


def findPropsAndMaxStateLvlNeeded(terms, vib_ana_setup: VibAnaSetup, freqs: str='static') -> tuple[list[MolecularProperty], VibAnaSetup]:
    """
    copy of WilsonSimulation.findPropsAndMaxStateLvl

    Make property instances needed to fulfill tasks and set maximum state level in vibrational analysis

    freqs: String: For terms involving properties that may be frequency dependent, use
    experiment information ('exp') or use the static ('static') properties?
    """

    props = []

    if terms is None:
        raise AssertionError('There must be terms present to determine needed properties')
    if vib_ana_setup is None:
        raise AssertionError('There must be a vibrational analysis setup present to')

    # FIXME: Consider checking if terms are VibPerturbedTerm instances
    for i in terms:

        for a in terms[i]:
            for t in terms[i][a]:
                for j in t.props:

                    ops = []

                    m = j.dord
                    for k in range(m):
                        ops.append('g')

                    n = len(j.ops)

                    for k in range(n):
                        ops.append('f')

                    if freqs == 'static':
                        pdict = {'ops': tuple(ops), 'freq': tuple([0.0 * k for k in range(len(ops))])}

                    else:
                        raise AssertionError('Managing electronic properties for non-static frequencies not yet implemented')

                    new_prop = MolecularProperty(pdict, trivial_name=prop_trivname(ord_geo=m, ord_el=n),
                                                    target_basis='nm', target_units='au')

                    if new_prop.h(1) not in [k.h(1) for k in props]:
                        props.append(copy.deepcopy(new_prop))

                # Currently registering these states without regard to whether harmonic or other regime
                # TODO: Find out if this should be changed

                max_state_lvl = 0

                for j in t.freqterms:

                    if len(j.sl.q) > max_state_lvl:
                        max_state_lvl = len(j.sl.q)
                    if len(j.sr.q) > max_state_lvl:
                        max_state_lvl = len(j.sr.q)

                for j in t.res:

                    if len(j.diff.sl.q) > max_state_lvl:
                        max_state_lvl = len(j.diff.sl.q)
                    if len(j.diff.sr.q) > max_state_lvl:
                        max_state_lvl = len(j.diff.sr.q)

                vib_ana_setup.max_state_lvl = max_state_lvl

    for i in vib_ana_setup.tellNeededProps():
        if i.h(1) not in [k.h(1) for k in props]:
            props.append(copy.deepcopy(i))
    
    return props, vib_ana_setup