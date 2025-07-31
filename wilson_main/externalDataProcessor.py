"""
Functionality for prepping data from files to WilsonSimulation

"""
from dataclasses import dataclass
import numpy as np
from .abstractions import MolecularSystem, ExternalCalcSetup

from CQCParse.parsing  import parser_template as parsing

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


def _retrieveFromCQCParse(parsed_data: parsing.ParsedData):
    """
    Preparing CalculatedDataFromOutput instance
    """
    derkeys = ['hess', 'cff', 'qff', 'dipgrad', 'diphess', 'polgrad', 'polhess']

    # parser.structure is an instance of StructureData
    datadict = {'system': MolecularSystem(name=parsed_data.molecule, natoms=len(parsed_data.structure.atoms), geo=parsed_data.structure),
                'calc_setup': ExternalCalcSetup(program=parsed_data.program, lvl_theory=parsed_data.method, basis=parsed_data.basis),
                'B': parsed_data.anharm_correction_data.rotational_constants,
                'coriolis': parsed_data.anharm_correction_data.coriolis_constants,
                'harmonic_states': parsed_data.vib_states.harmonic_states,
                'anharmonic_states': parsed_data.vib_states.anharmonic_states}
    for k in derkeys:
        datadict[k] = getattr(parsed_data.derivatives, k)
    
    return CalculatedDataFromOutput(**datadict)


def _setUpCQCParsing(datafile_dict: dict):
    """
    datafile_dict has a specific structure depending on program (also included in this dict)

    ----------------
        Gaussian 16:
    logfile = '/home/vlev/wilson-suite/wilson_intensities/tests/test_database/dftGaussian/FORM/B3LYPcc_pVQZ/g16_inputFull_3q.out'
    
    FORM_g16_1 = {'molecule': 'FORM', 'method': 'B3LYP', 'basis': 'cc-pVQZ', 
                'program': 'gaussian', 'log_file': logfile}
    ----------------
        CFOUR 2.1:
    out_file = '/home/vlev/wilson-suite/wilson_intensities/tests/test_database/refinedc4/CCSDTcc_pVQZ/out'
    polar_file = '/home/vlev/wilson-suite/wilson_intensities/tests/test_database/refinedc4/CCSDTcc_pVQZ/polar.pkl'
    dipole_file = '/home/vlev/wilson-suite/wilson_intensities/tests/test_database/refinedc4/CCSDTcc_pVQZ/dipole'
    cubic_file = '/home/vlev/wilson-suite/wilson_intensities/tests/test_database/refinedc4/CCSDTcc_pVQZ/cubic'
    quartic_file = '/home/vlev/wilson-suite/wilson_intensities/tests/test_database/refinedc4/CCSDTcc_pVQZ/quartic'
    outout_file = '/home/vlev/wilson-suite/wilson_intensities/tests/test_database/refinedc4/CCSDTcc_pVQZ/outfile0.out'
    molden_f = '/home/vlev/wilson-suite/wilson_intensities/tests/test_database/refinedc4/CCSDTcc_pVQZ/MOLDEN'
    
    FORM_c4_1 = {'molecule': 'FORM', 'method': 'CCSD(T)', 'basis': 'cc-pVQZ', 
                'program': 'cfour', 
                'out_file': out_file, 
                'molden_file':molden_f,
                'cubic_file': cubic_file,
                'quartic_file': quartic_file,
                'dipole_file': dipole_file,
                'polar_pkl': polar_file,
                }
    """
    if datafile_dict['program'] == 'cfour':
        from CQCParse.parsing import cfour_parser as c4p
        output = c4p.CFOUROutput(**datafile_dict)
        out_parser = c4p.CFOURParser(output)

    elif datafile_dict['program'] == 'gaussian':
        from CQCParse.parsing import gaussian_parser as g16p
        output = g16p.GaussianOutput(**datafile_dict)
        out_parser = g16p.GaussianParser(output)
    
    return out_parser


def prepareDataFromFiles(datafile_dict: dict) -> CalculatedDataFromOutput:
    """
    A wrapper. Two steps:
    1. Take a dictionary with file paths and make a parser object
    2. Prepare the parser with load()
    3. Get a parsed data object with parse()
    """
    parser = _setUpCQCParsing(datafile_dict)
    parser.load()
    parsed_data = parser.parse()

    return _retrieveFromCQCParse(parsed_data)