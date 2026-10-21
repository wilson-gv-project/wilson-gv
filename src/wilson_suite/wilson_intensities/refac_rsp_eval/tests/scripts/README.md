# refac_rsp_eval scripts

Scripts that are run by hand, and the refactor's notes. pytest collects none of these files.
Run a script by its module name:

    python -m wilson_suite.wilson_intensities.refac_rsp_eval.tests.scripts.<name> [options]

Each script's docstring says what it prints and which settings it uses. The scripts load the
stored terms with ../stored_terms.py and the settings of the 14 EVV terms from
../make_evv_reference.py. ACDM below is the acetic acid dimer: 16 atoms, 42 modes.
The todo items are those of todo.txt here.

## Scripts

| script | what it does | options | output | run time | todo items |
|---|---|---|---|---|---|
| example_evaluate_response_terms.py | Example of use, step by step: the 14 EVV terms on formaldehyde through evaluate_response_terms (a width per condition) and spectrum_on_grid; then a new width on the same rows | none | text: counts, the grid, the brightest point, the features in the window | 1 s | E6 (inputs and outputs), A17 |
| example_plot_spectrum.py | Example of plots with plot_rsp_func and plot_spectrum (wilson_analysis/render/plot_spectrum.py): formaldehyde in the whole window, a zoom, a 1D cut through the strongest peak, harmonic and anharmonic states on one scale; ACDM | out_dir (default: plot_spectrum_figures) | 5 png files in out_dir | 20 s | E7 (plots) |
| f2_speed.py | Run time of each step on ACDM (14 EVV terms, anharmonic states, a 700 x 700 grid): load, build_contributions (also under cProfile), features, boxes, window, grid | none | text: the time of each step, the 15 slowest functions of build_contributions | 20 s | F (speed; F2 was its first run), A17 |
| e10_boxes.py | Box sizes (a box is the part of the grid where one feature is evaluated): spectrum_on_grid with the tails cut, for each setting of a box rule, against all_features_on_grid with nothing cut. Formaldehyde and ACDM, each on a big window and on a zoom | `rules`: compare the amplitude and intensity rules; default: settings of amplitude_half_widths | text, one line per setting: the largest error, the worst visible point, the grid time, the features kept | 1 min (default) | E10 (tune the knobs, open), A17 |
| g7_survey.py | Survey of other experiments, from the stored terms. Per setup: the axes, the kinds of resonance conditions, the averaging rank, the most quanta, missing properties; then an evaluation on formaldehyde if nothing blocks it | `--only WORD ...`: setups whose name holds a word; `--terms FILE`: also write every unique term | text per setup; with --terms, the terms in FILE | 2 min, 530 MB for every setup | G7 (the survey), E8 (lines and planes), E13 (4 axes) |

Parts used elsewhere: f2_speed.load_acdm loads ACDM for e10_boxes.py and example_plot_spectrum.py.
e10_boxes.reference and e10_boxes.score also run from a notebook, on any result and window.

## Notes

| file | what it holds |
|---|---|
| todo.txt | the refactor's list: status, order of work, decisions with dates, checks, open items |
| todo_oldcode_rm.txt | every use of the old evaluator code and what happened to it (todo item E9) |
