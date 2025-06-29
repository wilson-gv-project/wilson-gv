import numpy as np

def eval_spec2D():
    from wilson.spectrum import wilsonmain_integration
    return wilsonmain_integration.spectrum2D


from wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
from wilson.spectrum import mainVibStates2arraydict
# from wilson.utils import prep_data_load

# TermND with TermsEvaluator
# with_diagnostics=True because wilsonSimulation.evaluate() - FIXME please!
# fixme? no need to know 'experiment'?
def terms_evaluator(system, experiment,
                    derived_terms, props,
                    spec_eval_setup, vib_ana_setup,
                    with_diagnostics=True):
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

    1. set up evaluation terms - from derived_terms
    2. load data into terms??? or smth
    3. make TermsEvaluator instance
    4. identify what to precalculate with TermsEvaluator
    5. precalculate if any with TermsEvaluator
    5.1. prepare data_for_precalc (related to 2.) - postprocess loaded data?..
    6. calculate amplitudes - loop over terms in TermsEvaluator(?), use precalculated data

    """
    from wilson.spectrum import TermND, TermsEvaluator, DataForPrecalc
    from wilson.spectrum.averaging import get_AlphaBetaGammaDelta_indices

    amplitudes = 0. + 0.j
    print('\nws.intensities.spectrum.evaluators.terms_evaluator:    Just doing nothing for now\n')

    # print('\n--- Printing spec_eval_setup:')
    # print(spec_eval_setup)
    # print('  spec_eval_setup.grid:', spec_eval_setup.grid)

    # print('\n--- Printing vib_ana_setup:')
    # print(vib_ana_setup)
    # print('vib_ana_setup.nc_sqrt_eigval', vib_ana_setup.nc_sqrt_eigval)

    # fixme: logging decorator instead?
    if with_diagnostics:
        """
    # Step 1: Set up terms
    evaluation_terms = setup_terms(exp, terms, props)
    # Step 2: Load data
    data = load_data(evaluation_terms)
    # Step 3: Precalculate
    precalc_data = precalculate(data, evaluation_terms)
    # Step 4: Calculate amplitudes
    amplitudes = calculate_amplitudes(precalc_data, evaluation_terms)
    # Step 5: Postprocess results
    result = postprocess_results(amplitudes)
    # Step 6: Generate diagnostics
    diagnostics = generate_diagnostics(precalc_data, amplitudes)
        """
        diagn = {}

        #! 1.1 transform terms from derive to evaluate form
        dict_terms = derived_terms_dict_to_dicts(derived_terms)

        # 1 complete
        terms_to_eval = [TermND(i, dict_terms[i]) for i in range(len(dict_terms))]
        from wilson_utils.termdict_from_symb_term import flip_modes_indices
        f = flip_modes_indices(terms_to_eval[3].expression,{'b':'a', 'c':'b','a':'c'})
        for i, t in enumerate(terms_to_eval):
            if i==3:
                # because res conds have indices b,c instead of a,b
                terms_to_eval[3] = TermND(3, f)

        # 3 complete
        te = TermsEvaluator(terms_to_eval)

        # 4 complete
        te.identify_to_precalculate()

        # 5.1
        props_data = {prop.triv_name: prop.vals for prop in props}
        props_dict = {'dipgrad':(1, 1),
                      'diphess':(1, 2),
                      'polgrad':(2, 1),
                      'polhess':(2, 2)}
        props_data_ready = {props_dict[p]:props_data[p] for p in props_dict}
        cff_data = {'F_abc': props_data['cff']}
        # todo: make a func for checking units - cm-1 vs Eh - energy_unit_check in spectrum_utils
        harmonic_arrays_Eh = {1: np.array(list(vib_ana_setup.nc_sqrt_eigval.values()))}

        avrg_terms = get_AlphaBetaGammaDelta_indices(num_f=4)
        axes_dict = spec_eval_setup.grid.make_mesh_numpy()
        states_arrays_Eh = mainVibStates2arraydict(vib_ana_setup.states, system.Nnmodes)

        data_for_precalc = DataForPrecalc(Nnmodes=system.Nnmodes,
                                          props_data=props_data_ready,
                                          avrg_terms=avrg_terms,
                                          axes_dict=axes_dict,
                                          states_arrays_Eh=states_arrays_Eh,
                                          harmonic_arrays_Eh=harmonic_arrays_Eh)

        # 5 - complete
        precalculated_data = te.precalculate(data_for_precalc)

        # 6
        from wilson.spectrum import debug_mode
        # for id, term in te.terms.items():
        for id, term in [(3, te.terms[3])]:
            print(f'term {term} starting')
            term.properties_data = cff_data
            term.precalc_data = precalculated_data
            term.mode_indices = vib_ana_setup.modes_indices
            with debug_mode(0):
                intensity = term.get_intensity(axes_dict[1], axes_dict[2],
                                               3.8, 0.0, debugprint=True, collect_all=False)
            amplitudes += intensity
        return amplitudes, diagn

    else:
        return amplitudes