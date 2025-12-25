from enum import Enum, auto
from typing import Any, Callable
from functools import wraps
from .abstractions import (VibAnaSetup, MolecularProperty,
						   MolecularSystem, DataOriginInfo)
from wilson_suite.wilson_derive.abstractions import VibPerturbedTerm
from wilson_suite.wilson_experiment.abstractions import VibExperiment
from wilson_suite.wilson_main.abstractions import VibState
from .spectrum_abstractions import SpecEvalSetup
from .main_functions import find_props_and_max_state_lvl

from wilson_suite.wilson_intensities.amplitudes.evaluation_wf import EvaluationWorkflow

class SimulationState(Enum):
    """States in the simulation lifecycle"""
    INITIALIZED = auto()
    CONFIGURED = auto()
    READY = auto()           # Properties resolved and data loaded
    EVALUATED = auto()
    RENDERED = auto()


class SimulationError(Exception):
    """Raised when operation attempted in wrong state"""
    pass


def needs(*required_attrs):
    """Simple decorator: check required attributes exist before running method"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            missing = [attr for attr in required_attrs if getattr(self, attr, None) is None]
            if missing:
                raise SimulationError(
                    f"Cannot {func.__name__}: missing {', '.join(missing)}"
                )
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


class WilsonSimulation:
    """Simplified WilsonSimulation with minimal state tracking"""
    
    def __init__(self, **kwargs):
        # Core components
        self.exp = kwargs.get('exp')
        self.terms = kwargs.get('terms')
        self.vib_ana_setup = kwargs.get('vib_ana_setup')
        self.spec_eval_setup = kwargs.get('spec_eval_setup')
        self.system = kwargs.get('system')
        
        # Property configuration
        self.eval_uniform = kwargs.get('eval_uniform')
        self.eval_by_prop_name = kwargs.get('eval_by_prop_name')
        
        # Results
        self.props = kwargs.get('props')
        self.residual_vib_info = kwargs.get('residual_vib_info')
        self.spec = None
        self.rendering = None
        self.diagn = kwargs.get('diagn')
        self.name = kwargs.get('name')
    
    # ==================== Simple State Checks ====================
    
    @property
    def is_configured(self) -> bool:
        """Check if basic configuration is complete"""
        return all([
            self.exp is not None,
            self.system is not None,
            self.terms is not None,
            self.vib_ana_setup is not None,
            self.spec_eval_setup is not None
        ])
    
    @property
    def is_ready(self) -> bool:
        """Check if ready to evaluate (has properties with data)"""
        # print(f'self.is_configured {self.is_configured}\nself.props is not None {self.props is not None}\nlen(self.props) > 0 {len(self.props) > 0}')
        # print(f"for p in self.props: { {p.trivial_name: p.vals is not None for p in self.props} }") 
        # print(f'self.vib_ana_setup.isAllSet {self.vib_ana_setup.isAllSet}')       
        conds = {'not self.is_configured': self.is_configured , 
                'not self.props is not None': self.props is not None , 
                'not len(self.props) > 0': len(self.props) > 0 ,
                f'p.trivial_name: p.vals is not None for p in self.props: {{p.trivial_name: p.vals is not None for p in self.props}}': all(p.vals is not None for p in self.props) ,
                'not self.vib_ana_setup.isAllSet': self.vib_ana_setup.isAllSet}
        final = True

        for c in conds:
            final &= conds[c]
            if not conds[c]:
                print(c)
                return final

        return final
    
    @property
    def has_spectrum(self) -> bool:
        """Check if spectrum has been evaluated"""
        return self.spec is not None
    
    @property
    def state(self) -> SimulationState:
        """Infer current state from object state"""
        if self.rendering is not None:
            return SimulationState.RENDERED
        if self.has_spectrum:
            return SimulationState.EVALUATED
        if self.is_ready:
            return SimulationState.READY
        if self.is_configured:
            return SimulationState.CONFIGURED
        return SimulationState.INITIALIZED
    
    # ==================== Configuration Methods ====================
    
    def addExperiment(self, experiment: VibExperiment):
        """
        Add an experiment

        experiment: VibExperiment instance: The experiment to be added
        """
        self.exp = experiment
    
    def addSystem(self, system: MolecularSystem):
        """
        Add a system

        system: MolecularSystem instance: The experiment to be added
        """
        self.system = system
    
    def addTerms(self, terms: dict[int, dict[tuple[int,int], VibPerturbedTerm]], 
                 extend: bool = False):
        """
        Add terms

        terms: dict of VibPerturbedTerm instances: The terms to be added
        extend: Boolean: Add this to (possibly already existing) terms or (default) set up this
        attribute afresh (possibly overwriting existing terms)?
        """
        if not extend:
            self.terms = terms
        else:
            if self.terms is None:
                self.terms = {}
            self.terms.update(terms)
    
    def addVibAnaSetup(self, vib_ana_setup: VibAnaSetup):
        """
        Add a vibrational analysis setup

        vib_ana_setup: VibAnaSetup instance: The vibrational analysis to be added
        """
        self.vib_ana_setup = vib_ana_setup
    
    def addSpecEvalSetup(self, spec_eval_setup: SpecEvalSetup):
        """
        Add a spectral evaluation/rendering setup

        spec_eval_setup: SpecEvalSetup instance: The setup to be added
        """
        self.spec_eval_setup = spec_eval_setup
    
    def addPropEvalSetup(self, eval_uniform: DataOriginInfo=None, 
                         eval_by_prop_name: dict[str: DataOriginInfo]=None):
        """
        Add a property evaluation setup

        See argument explanation of __init__ method of this class for explanation of these arguments

        VL: What if done after self.props is filled? 
        then can check if all props have calculation setup specified in parameters here.
        Also can warn user about the use of eval_uniform for props not mentioned in eval_by_prop_name
        """
        self.eval_uniform = eval_uniform
        self.eval_by_prop_name = eval_by_prop_name
    
    # ==================== Property Resolution ====================
    
    @needs('terms', 'vib_ana_setup')
    def setPropsAndMaxStateLvl(self, freqs: str = 'static'):
        """
        freqs: String: For terms involving properties that may be frequency dependent, use
        experiment information ('exp') or use the static ('static') properties?
        """
        
        self.props, self.residual_vib_info, self.vib_ana_setup.max_state_lvl = \
            find_props_and_max_state_lvl(self.terms, self.vib_ana_setup, freqs)
    
    @needs('props')
    def dressPropsWithSetup(self):
        """
        Dress my self.properties with computational setups according to how they are specified in
        self.eval_uniform or self.eval_by_prop_name
        """
        if not self.props:
            return
        
        for prop in self.props:
            dressed = False
            
            # Try specific setup first
            if self.eval_by_prop_name and prop.trivial_name in self.eval_by_prop_name:
                prop.addCalcSetup(self.eval_by_prop_name[prop.trivial_name])
                dressed = True
            # Fall back to uniform setup
            elif self.eval_uniform:
                prop.addCalcSetup(self.eval_uniform)
                dressed = True
            
            if not dressed:
                raise SimulationError(f'No calculation setup for property: {prop}')
        
        # Handle residual vib info
        if self.residual_vib_info:
            for key in self.residual_vib_info:
                if self.eval_by_prop_name and key in self.eval_by_prop_name:
                    self.residual_vib_info[key] = self.eval_by_prop_name[key]
                elif self.eval_uniform:
                    self.residual_vib_info[key] = self.eval_uniform
    
    # ==================== Data Loading ====================
    
    @needs('props')
    def requestData(self) -> dict:
        """
        data_dict: dict - {data_name: DataOriginInfo}
        """
        data_dict = {}
        
        for p in self.props:
            data_dict[p.trivial_name] = p.calc_setup
        
        if self.residual_vib_info:
            for k, v in self.residual_vib_info.items():
                data_dict[k] = v
        
        return data_dict
    
    @needs('props')
    def fillResults(self, data_dict: dict):
        """
        loading data into self.props (and optionally to self.vib_ana_setup)

        data_dict: dict - {data_name: values}

        """
        # Fill property values
        for p in self.props:
            value = data_dict.get(p.trivial_name)
            if value is not None:
                p.addValues(value)
        
        # Handle vibrational states
        if self.residual_vib_info:
            for k in self.residual_vib_info:
                if k in ['anharmonic_states', 'harmonic_states']:
                    states_dict = data_dict.get(k, {})
                    if states_dict:
                        states_list = []
                        for state, energy in states_dict.items():
                            states_list.append(VibState(
                                harm_quanta_coeffs={state: 1.0},
                                energy=energy,
                                state_label=','.join(state)
                            ))
                        self.vib_ana_setup.setStates(states=states_list)
                else:
                    value = data_dict.get(k)
                    if value is not None:
                        self.residual_vib_info[k] = value
                        setattr(self.vib_ana_setup, k, value)
    
    def getResults(self, obtainer: Callable[[dict[str,DataOriginInfo]], dict]):
        """
        obtainer must return : a dictionary:
            keys: trivial_name for properties or residual_vib_info keys
            values: values
        """
        data_request = self.requestData()
        print('data_request', data_request.keys())
        try:
            data = obtainer(data_request)
        except Exception as e:
            # print("data_request", data_request)
            # print(e)
            raise ValueError('Smth went wrong in the obtainer')
        self.fillResults(data)
    
    
    # ==================== Evaluation & Rendering ====================
    
    def evaluateSpectrum(self, evaluator, do_diagn: bool = False):
        """Evaluate the spectrum"""
        # Check we're ready
        if not self.is_ready:
            raise SimulationError(
                f"Cannot evaluate: simulation not ready (state: {self.state.name})"
            )
        
        if not self.vib_ana_setup.isAllSet:
            raise SimulationError('VibAnaSetup not ready')
        
        # Prepare context
        context = {
            'system': self.system,
            'experiment': self.exp,
            'derived_terms': self.terms,
            'props': self.props,
            'spec_eval_setup': self.spec_eval_setup,
            'vib_ana_setup': self.vib_ana_setup,
            'do_diagn': do_diagn
        }
        
        # Evaluate
        if do_diagn:
            self.spec, diagn = evaluator(**context)
            self.updDiagnostics(diagn)
        else:
            self.spec, _ = evaluator(**context)

    def evaluate(self):
        from ..wilson_intensities.amplitudes.evaluation_wf import make_evaluation_inputs
        eval_inputs = make_evaluation_inputs(simulation=self)
        workflow = EvaluationWorkflow(inputs=eval_inputs)
        self._workflow = workflow

        self.spec  = workflow.run()

        # if self.diagn is None:
        #     self.diagn = {}
        # self.diagn.update(info)


    def render(self, renderer, do_diagn: bool = False):
        """Render the spectral data"""
        if not self.has_spectrum:
            raise SimulationError(
                f"Cannot render: no spectrum data (state: {self.state.name})"
            )
        
        if self.diagn is None:
            self.diagn = {}
        
        context = {
            'spec_data': self.spec,
            'system': self.system,
            'experiment': self.exp,
            'diagn': self.diagn,
            'name': self.name,
            'spec_eval_setup': self.spec_eval_setup,
            'do_diagn': do_diagn
        }
        
        if do_diagn:
            self.rendering, diagn = renderer(**context)
            self.updDiagnostics(diagn)
        else:
            self.rendering, _ = renderer(**context)
    
    # ==================== Utility ====================
    
    def updDiagnostics(self, upd_dict: dict):
        """
        add info to self.diagn dictionary
        """
        if self.diagn is None:
            self.diagn = {}
        self.diagn.update(upd_dict)
    
    def status_report(self) -> dict:
        """Get current simulation status"""
        return {
            'state': self.state.name,
            'is_configured': self.is_configured,
            'is_ready': self.is_ready,
            'has_spectrum': self.has_spectrum,
            'has_rendering': self.rendering is not None,
            'name': self.name or 'Unnamed',
            'can_evaluate': self.is_ready,
            'can_render': self.has_spectrum,
        }
    
    def __repr__(self):
        """Quick status view"""
        return f"<WilsonSimulation '{self.name}' state={self.state.name}>"
