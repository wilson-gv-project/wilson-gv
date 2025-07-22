# Notes from 30.06.2025

1. **wilson-main**: externalCalcSetup: dataclass + immutable. other_setup: custom data from user? input_generation functionality - separate class
2. **wilson-main**: wilsonSimulation: report method; saving instances/setups; saving results
3. **wilson-main**: evaluate: self.spec - np.ndarray, self.diagn - dict;
4. **wilson-main**: more general evaluator would take experiment info. Other evaluator (evaluate_as_response) just evaluates response function
5. **wilson-derive**: Canonical indices in terms
6. **😺wilson-intensities**: Identification in TermsEvaluate - clean up
7. **😺wilson-intensities**: Use dictionaries to hold properties values? - lower priority
8. **😺wilson-intensities**: terms_evaluator: always identify and precalculate? - check if it actually works without precalc;
9. **😺wilson-intensities**: TermsEvaluator.precalc_avrg_tensors: generalize greek indices -- dict of greek letters
10. **😺wilson-intensities**: make indices abc into tuples 
11. **😺wilson-intensities/CQCParse**: Non-averaged data - check namings and functions
12. **😺wilson-main**: calculationBatch.getResultsFromVault - VL
13. **wilson-main**: vibAnaSetup clarifications: class name, external_fill_from name; 
14. **wilson-analysis/suite test**: basic renderer
15. **😺wilson-intensities**: harmonic precal data - fix
16. **😺wilson-intensities**: test_clean_termND  - produces different molecule????
17. **😺wilson-analysis/suite test**: include renderer to evv_tester
18. **😺wilson_suite**: update.sh file to go to specific commit on a specific branch for each repo
19. **wilson-derive**: another look at signs of perturbing freqs in terms and signs of spectroscopic axes


------------------------------------------------

# Notes from 16.07.2025

## Discussion points:

1. return datatype(s) from evaluator
   Several arrays: when precalc data is reused -> batch evaluator (return precalc data and several spec ranges);  return ndarray always now
2. what should be diagnostics info: dictionary
   get_factor_summed() result... will be accumulated in a dict, should not be strict. we'll see what we need. empty now. 
3. output and logging: configs for wilsonsimulations; make "bare bones" output. print_level settings (logging) and diagnostics_settings: issue for wilson-main and utils: set also sdtout -> VL move utils out of wilson-intensities; MR make infrastructure in WilsonSimulation
4. general names for indices and stuff (e.g. get_full_factor)

all indices                 -> nm_full_indices
non-summation indices       -> nm_nonsumm_indices
summation indices           -> nm_summ_indices
ABGD greek                  -> cart_indices

clear the uses of indices in terms, distinction between

5. responsibilities
6. conversions of units


## Issues

1. conftest.py - clear up
2. test_vpt2_new.py - write vpt2 module tests
3. reorganize utils
4. spectral grid in get_amplitudes -> arbitrary number of axes
5. modes comparison, mode indices in CQCParse; make a canonical labeling of normal modes: two parts (short term CQCPArse fix and long term canonical labels)
6. window_check should be a generalized utility function
7. Stored data in a dictionary (set of spec points)
8. write anharmonic_analyzer to use vpt2.py
9. conversions of units
10. tests/checks about precission of energy unit conversions
11. product_all = self.get_factor_summed() - should have stored all prefactor - atribute for each term: after precalculate assemble full factors all combinations of core indices

precalculated_data contains all unique precalc data, right? So why register it as an attribute of each term? Only some of the data in precalculated_data is relevant for each term, right? (Maybe instead pass precalculated as a "bank of data" to term.get_amplitudes?)


12. get_factor_summed rename: 
13. make uses of wilson-main abstractions of properties
14. fix weirdness about cff
15. clear up init of TermND
16. make general: get_resonance_location_general for diff indices in res conds
17. collective_n_idx_max

wasnt self.collective_n_idx_max an attribute of all the terms? We might rework this to look in each term by itself which indices are represented in the resonance conditions and which ones are not (i.e. the latter can be summed over)

18. precalc_avrg_tensors

Stored appears to now contain both 2-index (i.e. a, b) and 3-index (i.e. a, b,c) quantities? I think I need a demonstration of the addressing


## Conclusion for now:

1. Do what's trivial for the PR
2. Make issues for all other things needed


------------------------------------------------

# Notes from 22.07.2025 - VL

WilsonSimulation: 
    - what is the metadata for a spectrum result? 
    - what is needed to reproduce this result?
    - what was needed for evaluation?
         --- system, derived_terms, props, spec_eval_setup, vib_ana_setup
            - system <- WilsonSimulation.system
            - derived_terms <- WilsonSimulation.terms but translated <- how were they derived <- experiment
            - props <- WilsonSimulation.props with vals <- WilsonSimulation.eval_uniform
            - spec_eval_setup <- WilsonSimulation.spec_eval_setup
            - vib_ana_setup <- WilsonSimulation.vib_ana_setup ---- states info, system, calc info (external_fill_from <- ExternalCalcSetup)
    - what is needed for rendering?
         --- spec, system, exp, diagn, name, spec_eval_setup ---- why exp? why diag?
            - spec <- evaluator
            - system <- WilsonSimulation.system
            - exp <- WilsonSimulation.exp; why optional for eval but not for render?
            - spec_eval_setup <- WilsonSimulation.spec_eval_setup
terms <- 

Should be possible:
  1. to set up and run a calculation from json file starting from derivation (experiment and calc setups info)
  2. to set up and run a calculation from json file starting from derived terms (terms and calc setups info)
  3. to set up and run a calculation from json file starting from derived terms, vibdata and props, spec_eval_setup 
Inputs generation is not supported yet, so it is also not mentioned for now


exp                 dict_keys(['order', 'field', 'detector', 'scans', 'magn_conditions']) ---- VibExperiment
-------
vib_ana_setup       dict_keys(['regime', 'system', 'regime_subinfo', 'max_state_lvl', 'states', 
                                'nc_sqrt_eigval', 'nc_eigvec', 'allow_skip_eigvec', 'vibana_prop_need', 'external_fill_from', 'exclude_modes']) ---- VibAnaSetup
-------
spec_eval_setup     dict_keys(['grid', 'ev_info', 'rnd_info']) ---- SpecEvalSetup
-------
system              dict_keys(['name', 'natoms', 'geo', 'geo_extra'])
-------
eval_uniform        dict_keys(['program', 'lvl_theory', 'basis', 'other_setup', 'other_setup_identifier']) ---- ExternalCalcSetup
-------
eval_by_prop_name   None
-------
props               dict_keys(['prop_spec', 'trivial_name', 'in_basis', 'in_units', 
                                'target_basis', 'target_units', 'serial_vals'])
-------
calc_batches        dict_keys(['system', 'calc_setup', 'properties'])

dict_keys(['system', 'calc_setup', 'properties'])
{-830484654937960533: {'system': {'name': 'ACAC', 'natoms': 8, 'geo': None, 'geo_extra': None}, 
                       
                       'calc_setup': {'program': 'gaussian', 'lvl_theory': 'B3LYP', 'basis': 'cc_pVQZ', 'other_setup': {}, 'other_setup_identifier': {}}, ---- ExternalCalcSetup
                       
                       'properties': [{'prop_spec': {'ops': ('g', 'f'), 'freq': (0.0, 0.0)}, 'trivial_name': 'dipgrad', 'in_basis': None, 'in_units': None, 'target_basis': 'nm', 'target_units': 'au', 'serial_vals': None}, {'prop_spec': {'ops': ('g', 'g', 'f', 'f'), 'freq': (0.0, 0.0, 0.0, 0.0)}, 'trivial_name': 'polhess', 'in_basis': None, 'in_units': None, 'target_basis': 'nm', 'target_units': 'au', 'serial_vals': None}]}}
-------
spec                 None
-------
diagn                None
-------
rendering            None
-------
name                 None
-------
terms                dict_keys(['averaged_props', 'non_averaged_props', 'termA_pref', 'termB_pref', 'vibene_denom', 'vibenediff', 'resonances'])


------------------------------------------------

# Notes from 23.07.2025 

Finding sysytems with relatively small effects , to show off the ability to find differences with broad dynamic range.
Rotamers? what are the weak effect?

Propene - rotate CH3.

### More calculations


1. Formic acid -  CCSD(T) with TZ

