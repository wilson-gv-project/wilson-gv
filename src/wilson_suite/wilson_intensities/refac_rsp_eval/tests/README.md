# refac_rsp_eval tests

Tests of the evaluator (plan.py, evaluate.py, features.py, grid.py, lineshape.py, pipeline.py),
with their helpers and reference data. Run them with pytest:

    python -m pytest src/wilson_suite/wilson_intensities/refac_rsp_eval/tests

Scripts that are run by hand are in [scripts/](scripts/): speed, box sizes, and two examples of
use. pytest collects none of them. The refactor's notes are
there too: todo.txt and todo_oldcode_rm.txt. [scripts/README.md](scripts/README.md) lists them all.

## Test files

| file | what it checks | data |
|---|---|---|
| test_plan.py | plan.py, the compile stage: property and denominator collections, resonance motifs, ParameterSet, ResLocPoint, CompiledTerm, compile_terms | none |
| test_evaluate.py | evaluate.py, the numeric stage: the resonance location, each factor (non-averaged, averaged, 1/omega and energy-difference denominators), the einsum average, the sums over labels, build_contributions, excluded modes, the line shape on the grid, boxed_features_on_grid against all_features_on_grid | toy molecule |
| test_features.py | features.py: tables of rows, SpectralFeature (amplitude, intensity, widths), features_from_rows, SpectralWindow, both box rules, flat features, 1 to 4 axes | none |
| test_grid.py | grid.py: Box, grid_points, make_mesh, box_slices (also against a point-by-point search on random boxes), 1 to 4 axes | none |
| test_lineshape.py | lineshape.py: the Lorentzian factor; peak, tail and distance agree with it | none |
| test_pipeline.py | pipeline.py: compute_features and FeatureResult on the toy molecule; prop_names, averaging_rank, make_polarization_linear_comb; evaluate_response_terms and spectrum_on_grid on the 14 EVV terms (box rules, cut_tails, reuse of the rows) | toy molecule, formaldehyde |
| test_dimensions.py | 1 to 4 axes: one toy term per number of axes; CARS (1D), IR absorption (1D, with a flat background), 3 IR (3D) | mock data, formaldehyde |
| test_evv_reference.py | the rows of the 14 EVV terms (coefficients, locations, zero pairs) have not changed, against evv_reference.json | formaldehyde, water, CO |
| test_grid_reference.py | the whole spectrum has not changed, grid point by grid point, against grid_reference.npz. Run as a script, it writes that file again | formaldehyde |
| test_hand_calculation.py | the numbers are right: rows of EVV, CARS and 3 IR terms by hand, with numpy and textbook formulas; control cases with the wrong denominator | water |
| test_imports.py | no file of refac_rsp_eval or wilson_system imports the experiment or derive | the source files |
| test_integration.py | the 14 EVV terms compiled; the data reader; the factors of one term; the einsum average against the plain loop on real data, also with the molecule rotated | formaldehyde |

## Support files

| file | what it holds | used by |
|---|---|---|
| \_\_init\_\_.py | empty: makes tests a package, so the files can import each other | every file here |
| conftest.py | pytest fixtures of the toy molecule: states, molsys, molsys_a1_zero | test_evaluate.py, test_pipeline.py |
| helpers.py | small hand-built objects: toy states and terms, resonance motifs, row(), feat(), helpers for 1 to 4 axes; the averaged part as a plain loop, the reference for its einsum | most test files here, wilson_system/tests/test_system_data.py |
| stored_terms.py | reads the stored terms, already translated to axes (data_for_tests/translated_terms): load_setup, index | the real-data tests, make_evv_reference.py, the scripts |
| make_evv_reference.py | the settings and loaders of the 14 EVV terms (evv_terms, data_origin, LASER_POL, ...). Run as a script, it writes evv_reference.json | test_evv_reference.py, test_pipeline.py, test_dimensions.py, test_grid_reference.py, the scripts |
| evv_reference.json | the rows of the 14 EVV terms: formaldehyde, water and CO, harmonic and anharmonic states | test_evv_reference.py, test_hand_calculation.py, test_pipeline.py |
| grid_reference.npz | the spectra of the test_grid_reference.py cases | test_grid_reference.py |
