# refac_rsp_eval scripts

Scripts that are run by hand. pytest collects none of these files.
Run a script by its module name:

    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.scripts.<name> [options]

Each script's docstring says what it prints and which settings it uses. The scripts load the
stored terms with ../stored_terms.py and the settings of the 14 EVV terms from
../make_evv_reference.py. The stored terms are in src/wilson_suite/data_for_tests/translated_terms;
make_translated_terms.py there writes them (see the last section). ACDM below is the acetic acid
dimer: 16 atoms, 42 modes.
The open items of the refactor are in [issue #71](https://github.com/wilson-gv-project/wilson-gv/issues/71).

## Scripts

| script | what it does | options | output | run time |
|---|---|---|---|---|
| example_evaluate_response_terms.py | Example of use, step by step: the 14 EVV terms on formaldehyde through evaluate_response_terms (a lineshape_param per condition) and spectrum_on_grid; then a new lineshape_param on the same rows | none | text: counts, the grid, the brightest point, the features in window_box | 1 s |
| example_plot_spectrum.py | Example of plots with plot_rsp_func and plot_spectrum (wilson_analysis/render/plot_spectrum.py): formaldehyde, every feature in window_box, a zoom, a 1D cut through the strongest peak, harmonic and anharmonic states on one scale; ACDM | out_dir (default: plot_spectrum_figures) | 5 png files in out_dir | 20 s |
| f2_speed.py | Run time of each step on ACDM (14 EVV terms, anharmonic states, a 700 x 700 grid): load, build_contributions (also under cProfile), features, boxes, window, grid | none | text: the time of each step, the 15 slowest functions of build_contributions | 20 s |
| e10_boxes.py | Box sizes (a feat_box is the part of the grid where one feature is evaluated): spectrum_on_grid with the tails cut, for each setting of a box rule, against all_features_on_grid with nothing cut. Formaldehyde and ACDM, each on a big window_box and on a zoom | `rules`: compare the amplitude and intensity rules; default: settings of amplitude_half_widths | text, one line per setting: the largest error, the worst visible point, the grid time, the features kept | 1 min (default) |

Parts used elsewhere: f2_speed.load_acdm loads ACDM for e10_boxes.py and example_plot_spectrum.py.
e10_boxes.reference and e10_boxes.score also run from a notebook, on any result and window_box.

## Code from outside the refactor

The refactor's code is refac_rsp_eval and wilson_system. The scripts also use:

| module | what it gives | used by |
|---|---|---|
| wilson_analysis/render/plot_spectrum.py (written for the refactor) | plot_rsp_func, plot_spectrum | example_plot_spectrum.py |
| wilson_intensities/amplitudes/averaging.py (old evaluator code), through pipeline.py | the averaging recipe | every script |
| wilson_utils.wilson_data_obtainer (the reader), through MolSystemData.load | the data of a Gaussian output file | every script |

No script imports the experiment or derive. The stored terms come from
data_for_tests/translated_terms/make_translated_terms.py, which runs main's code outside the refactor:

| module | what it gives |
|---|---|
| wilson_experiment.experiment_abstractions | VibExperiment: the setups, their canonical axes, polarizations and polarization vector (laser_pol) |
| wilson_derive.derive | get_fully_enhanced_terms: the terms |
| wilson_derive.term_var_translate | the translation of the terms to axes |
| wilson_utils.termdict_from_symb_term | derived_terms_flat: derive's terms as one list |
| wilson_utils.builders | make_SpectralAxisSet: the EVV axes, A = w1, B = -w1 + w2 |
| wilson_intensities/amplitudes/averaging.py (old evaluator code) | get_pol_laser: checks each setup's polarization vector |
