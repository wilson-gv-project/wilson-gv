"""
Evaluator functions for WilsonSimulation
"""
from ...wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
from ..utils import mainVibStates2arraydict, check_energy_unit, convNu2Ene
from ...wilson_utils.unit_convertor import convertor

import numpy as np

import logging
logger = logging.getLogger("wilson."+__name__)


# TermND with TermsEvaluator
def terms_evaluator(system,
                    derived_terms, props,
                    spec_eval_setup, vib_ana_setup,
                    do_diagn: bool,
                    selected_combs: list = None,
                    collect_all: bool = False) -> complex|float|np.ndarray:
    """
    >> Orchestrating spectrum amplitudes evaluation with TermND setup.

      Should ultimately return: spectrum array (and diagnostics)
      Diagnostics: ?? (spectrum calculation related)

    Called in wilsonSimulation class instance:
        evaluator(self.system, self.exp, self.terms, self.props, self.spec_eval_setup, self.vib_ana_setup)

    - system - ws.main.abstractions.molecularSystem(name='ACAC')
    - exp/experiment - ws.experiment.abstractions.vibExperiment(order, field_a,
                                                              detector_a, [scan_a],
                                                              magn_conditions=[[-1, 2]])
    - terms - return from ws.derive.main.get_fully_enhanced_terms() - a dict.
        E.g.: {1: {(1, 0): [<wilson_derive.abstractions.vibPerturbedTerm object at 0x7ff3dfd33590>, <wilson_derive.abstractions.vibPerturbedTerm object at 0x7ff3dfd339b0>],
         (0, 1): [<wilson_derive.abstractions.vibPerturbedTerm object at 0x7ff3dfd33170>, <wilson_derive.abstractions.vibPerturbedTerm object at 0x7ff3dfd32180>,
         <wilson_derive.abstractions.vibPerturbedTerm object at 0x7ff3dfd31220>, <wilson_derive.abstractions.vibPerturbedTerm object at 0x7ff3dfd32570>,
         <wilson_derive.abstractions.vibPerturbedTerm object at 0x7ff3dfd32720>, <wilson_derive.abstractions.vibPerturbedTerm object at 0x7ff3dfd31b80>,
         <wilson_derive.abstractions.vibPerturbedTerm object at 0x7ff3dfd32a20>, <wilson_derive.abstractions.vibPerturbedTerm object at 0x7ff3dfd31460>,
         <wilson_derive.abstractions.vibPerturbedTerm object at 0x7ff3dfd30920>, <wilson_derive.abstractions.vibPerturbedTerm object at 0x7ff3dfd30230>,
         <wilson_derive.abstractions.vibPerturbedTerm object at 0x7ff3dfd30320>, <wilson_derive.abstractions.vibPerturbedTerm object at 0x7ff3dfd31520>]},
        0: {(0, 0): []}}
    - props - a list of wilson_main.abstractions.molecularProperty objects
        E.g.: [molecularProperty dipgrad: values are not None, molecularProperty polhess: values are not None,
                molecularProperty polgrad: values are not None, molecularProperty diphess: values are not None,
                molecularProperty cff: values are not None, molecularProperty qff: values are not None,
                molecularProperty B: values are not None, molecularProperty coriolis: values are not None]
    - spec_eval_setup - wilson_main.abstractions.specEvalSetup instance
    - vib_ana_setup - wilson_main.abstractions.vibAnaSetup instance

    data: props

        What should be done?

    
    - Step 1: Set up terms
    evaluation_terms = setup_terms(exp, terms, props)
    - Step 2: Load data
    data = load_data(evaluation_terms)
    - Step 3: Precalculate
    precalc_data = precalculate(data, evaluation_terms)
    - Step 4: Calculate amplitudes
    amplitudes = calculate_amplitudes(precalc_data, evaluation_terms)
    - Step 5: Postprocess results
    result = postprocess_results(amplitudes)
    - Step 6: Generate diagnostics
    diagnostics = generate_diagnostics(precalc_data, amplitudes)

    spec_eval_setup  is a wilson_main.abstractions.specEvalSetup instance
    """
    from ..spectrum import TermND, TermsEvaluator
    from ..utils.spectrum_utils import DataForPrecalc
    from ..spectrum.averaging import getPolarizationAveragingExpression

    amplitudes = 0. + 0.j

    # fixme: logging decorator instead?
    diagn = {}

    #! 1.1 transform terms from derive to evaluate form
    dict_terms = derived_terms_dict_to_dicts(derived_terms)

    # 1 complete
    terms_to_eval = [TermND(i, dict_terms[i]) for i in range(len(dict_terms))]
    for t in terms_to_eval:
        t.vibstates = vib_ana_setup.states
    # 3 complete
    te = TermsEvaluator(terms_to_eval)

    # 4 complete
    te.identify_to_precalculate()

    # for p in props:
    #     if p.trivial_name == 'cff':
    #         if p.in_units != 'au':
    #             p.target_units = 'au'
    #             p.convertValues(convertor=convertor,
    #                             convertor_info={'harm_freqs': list(vib_ana_setup.nc_sqrt_eigval.values())})

    # 5.1
    props_data = {prop.trivial_name: prop.vals for prop in props}

    if 'cff' in [p.trivial_name for p in props]:
        # format transformation 
        cff_data = {'cff': props_data['cff']}

    # todo: make a func for checking units - cm-1 vs Eh - energy_unit_check in spectrum_utils
    # format transformation
    harm_states_arr = np.array(list(vib_ana_setup.nc_sqrt_eigval.values()))

    if check_energy_unit(harm_states_arr[1]) == 'cm-1':
        harmonic_arrays_Eh = {1: convNu2Ene(harm_states_arr)}
    else:
        harmonic_arrays_Eh = {1: harm_states_arr}

    avrg_terms, prefactorAvrg = getPolarizationAveragingExpression("ZZZZ") # "ZZZZ" should come from some setup dataobject
    axes_dict = spec_eval_setup.ev_info.freq_variables
    Gamma_rc = spec_eval_setup.ev_info.Gamma
    # format transformation
    states_arrays_Eh = mainVibStates2arraydict(vib_ana_setup.states, system.Nnmodes)

    data_for_precalc = DataForPrecalc(Nnmodes=system.Nnmodes,
                                      props_data=props_data,
                                      avrg_terms=(avrg_terms, prefactorAvrg),
                                      axes_dict=axes_dict,
                                      states_arrays_Eh=states_arrays_Eh,
                                      harmonic_arrays_Eh=harmonic_arrays_Eh)
    
    # 5 - complete
    precalculated_data = te.precalculate(data_for_precalc)

    diagn['evaluator diagn'] = {}

    # logger.warning(f'precalculated_data \n{precalculated_data['vibdiffs']}')
    # logger.warning(f'precalculated_data \n{precalculated_data['avrg_tensors']}')
    logger.warning(f'precalculated_data \n{precalculated_data['res_conds']}')

    # 6
    from ..utils import debug_mode
    for id, term in te.terms.items():
        if 'cff' in [p.trivial_name for p in props]:
            term.properties_data = cff_data
        term.precalc_data = precalculated_data
        term.mode_indices = vib_ana_setup.modes_indices

        term.vibstates = vib_ana_setup.states
        # context manager - setting debug level
        with debug_mode(0):
            logger.warning(f'now term {id}')

            a_intermediate = term.get_amplitudes(w1=axes_dict['w1'], w2=axes_dict['w2'],
                                                 Gamma_rc=Gamma_rc, margin=0.0, 
                                                 debugprint=True, collect_all=collect_all,
                                                 sel_abs=selected_combs)
        amplitudes += a_intermediate

        diagn['evaluator diagn'][id] = term.diagnostics

    logger.warning(f'amplitudes \n{amplitudes}')
    if do_diagn:
        return amplitudes, diagn
    else:
        return amplitudes, {}