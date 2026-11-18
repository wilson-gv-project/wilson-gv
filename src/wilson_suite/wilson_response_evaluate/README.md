# wilson_response_evaluate

Turns response terms into a spectrum on a frequency grid.

The package does not derive terms, does not translate them, and reads no file.
The caller gives it the terms and the data of one molecule.

```
terms + molecular data
   │   evaluate_response_terms      computes the rows: once per term list and data
   v
result: rows and features, no grid yet
   │   spectrum_on_grid             once per grid
   v
spec: the complex amplitude on a grid
   │   plot_rsp_func                wilson_analysis/render/plot_spectrum.py
   v
plot
```

The sections below follow these steps. A full example that runs:
[tests/scripts/example_evaluate_response_terms.py](tests/scripts/example_evaluate_response_terms.py).

## 1. The terms

A list of `VibPerturbedTerm` from wilson_derive, translated to axes.

- Each resonance condition's pf (perturbing frequencies) is in axis labels, e.g. `['A', '-B']`.
  Terms in pulse ids, e.g. `[-1, 2]`, raise ValueError.
- One call takes one set of term axes. Terms with two sets raise ValueError, e.g. 3 IR with
  (A, B) and (A, B, C). `terms_by_axis_set(terms)` splits them: one call per set.
- Terms without resonance conditions are a flat background. They join any set.
- All terms of one call need the same averaged slots, so that one averaging recipe fits all.
- In the results, `row.term_id` is the index of the term in this list.

## 2. The molecular data

`prop_names(terms)` lists the properties the terms need, e.g.
`['polhess', 'dipgrad', 'polgrad', 'diphess', 'cff']`.

A `MolSystemData` holds the data. There are two ways to make one:

1. `MolSystemData.load(origin, prop_names(terms))` reads one source: a Gaussian or CFOUR output,
   through `wilson_data_obtainer`.
2. `MolSystemData.from_datadict(mol_props, data_dict)` takes a data dict the caller already has.
   A new data source only has to produce this dict.

Both check the dict first (`check_data_dict`): key names, mode numbers, shapes, symmetry.

```python
from wilson_suite.wilson_response_evaluate.pipeline import prop_names
from wilson_suite.wilson_system.system_data import DataOriginInfo, MolSystemData

origin = DataOriginInfo(source_type='gaussian', base_file_loc='formaldehyde.out')
molsys = MolSystemData.load(origin, prop_names(terms))
```

### Required data

| key | what | form | unit | needed |
|---|---|---|---|---|
| `nc_sqrt_eigval` | harmonic frequency of each mode | `{mode: value}`, keys 0 .. n-1 | cm-1 | always: gives n, the number of modes, and the 1/omega factors |
| `harmonic_states` | harmonic state energies | `{quanta: energy}`. quanta: one mode number (str) per quantum, e.g. `('0',)`, `('0', '1')`, `('1', '1')`. Must have one fundamental per mode | cm-1 | always |
| `anharmonic_states` | anharmonic state energies (VPT2, GVPT2) | as `harmonic_states` | cm-1 | for `states_choice='anharmonic'` |
| one key per property | the property values | numpy array: mode indices first, then x, y, z (next table). Symmetric in the mode indices | au | each name in `prop_names(terms)` |

| property | what | shape, n modes |
|---|---|---|
| `dipgrad` | dipole moment, 1st derivative by the normal modes | (n, 3) |
| `diphess` | dipole moment, 2nd derivative | (n, n, 3) |
| `polgrad` | polarizability, 1st derivative | (n, 3, 3) |
| `polhess` | polarizability, 2nd derivative | (n, n, 3, 3) |
| `cff` | energy, 3rd derivative (cubic force constants) | (n, n, n) |

Other names follow the same rule, e.g. `qff` (energy, 4th derivative): (n, n, n, n).

Rules for the data:

1. Units and coordinates are an agreement with the source. The evaluator multiplies the property
   values as they are. It converts only frequencies and state energies, from cm-1 to Eh. Nothing
   checks the units.
2. The ground state needs no entry. If a state list has the key `()`, its energy must be 0.
3. The states must cover every state the terms name, for every choice of modes.
   Example: a term with the state "two quanta in a, one in b" needs states such as `('0', '0', '2')`.
   A missing state raises `ValueError: Requested state label - 0,0,2 - is not in VibStatesData`.
4. Data from two sources: mode k must be the same vibration in both. Nothing checks this.

### Example: the setups stored for the tests

| setup | terms | term axes | properties | `laser_pol` | most quanta in one state |
|---|---|---|---|---|---|
| EVV | 14 | A, B | polhess, dipgrad, polgrad, diphess, cff | 3 numbers | 3 |
| CARS | 4 | A | polgrad | 3 numbers | 1 |
| IR absorption | 5 | A (2 terms); none (3 terms, a flat background) | polhess, polgrad, cff, dipgrad | 1 number | 1 |
| 3 IR | 16 | A, B (14 terms); A, B, C (2 terms): two calls | polhess, dipgrad, polgrad, diphess, cff | 3 numbers | 3 |

"Most quanta in one state" says which states the data must hold. For CARS the fundamentals are
enough. For EVV the data needs every state with up to 3 quanta.

## 3. Evaluate the terms: `evaluate_response_terms`

```python
from wilson_suite.wilson_response_evaluate.pipeline import evaluate_response_terms

result = evaluate_response_terms(terms, molsys_data=molsys, states_choice='anharmonic',
                                 laser_pol=(1, 1, 1), lineshape_param=5.)
```

For each term and each choice of modes for its resonance labels, it computes:
- the coefficient, summed over the other labels;
- the location of the peak, cm-1.

Rows with the same resonance conditions and the same location add up into one feature.

| argument | what it is | example |
|---|---|---|
| `terms` | section 1 | |
| `molsys_data` | section 2 | `molsys` |
| `states_choice` | which state energies to use. Must match the vibrational analysis: harmonic → `'harmonic'`; VPT2, GVPT2 → `'anharmonic'`. No default | `'anharmonic'` |
| `laser_pol` | the experiment's polarization_avg_vector. Its length is set by the averaging rank (the number of light fields the terms couple to): rank 2 or 3 → 1 number, 4 → 3, 5 → 6, 6 → 15 | `(1, 1, 1)`: all pulses and the detector along x |
| `lineshape_param` | the half width gamma of each condition's Lorentzian, cm-1. A number, or a function `(vd, freqs) -> cm-1` | `5.`, or `lambda vd, freqs: 3. + 0.002 * abs(vd.energy_difference())` |
| `exclude_modes` | modes to leave out, counted from 0 as in the data. Optional. The other peaks change too: their sums lose the excluded modes | `(5,)` |

| output | what it holds |
|---|---|
| `result.table` | one ContributionRow per term and choice of modes, with coeff ≠ 0: term_id, motif, params (the modes), location (cm-1), coeff (au) |
| `result.zero` | the (term_id, params) pairs with coeff 0. They get no row |
| `result.features` | one SpectralFeature per (motif, location): the coefficients of its rows added up |
| `result.factors_by_index_set(row)` | the factors behind one row, computed again on request |
| `result.cancelling(rel_tol)` | the features whose rows cancel |

## 4. Put the spectrum on a grid: `spectrum_on_grid`

```python
from functools import partial
import numpy as np

from wilson_suite.wilson_response_evaluate.features import amplitude_half_widths
from wilson_suite.wilson_response_evaluate.grid import Box
from wilson_suite.wilson_response_evaluate.pipeline import spectrum_on_grid

spec = spectrum_on_grid(result, window_box=Box({'A': (600., 2800.), 'B': (5., 3100.)}),
                        n_points_per_axis={'A': 501, 'B': 201},
                        box_rule=partial(amplitude_half_widths, dynrange=1000.))
intensity = np.abs(spec.amplitude) ** 2      # shape (501, 201)
```

1. `box_rule` gives each feature a half-width. Location ± half-width is its feat_box.
2. The features whose feat_box overlaps `window_box` are kept.
3. The kept features go on the grid. Their amplitudes add before squaring, so peaks interfere.

One result serves many grids. Call `spectrum_on_grid` again with another `window_box`, or
with `dataclasses.replace(result, lineshape_param=8.)`. The rows are not computed again.

| argument | what it is | example |
|---|---|---|
| `window_box` | the range of each axis, cm-1. Its axes are the term axes | `Box({'A': (600., 2800.), 'B': (5., 3100.)})` |
| `n_points_per_axis` | grid points on each axis, both edges included | `{'A': 501, 'B': 201}` |
| `box_rule` | how far around each peak to look (its feat_box). `dynrange` is the plot's dynamic range, max / floor. Required | `partial(amplitude_half_widths, dynrange=1000.)` |
| `cut_tails` | `False` (default): each kept peak on the whole grid. `True`: each peak only inside its feat_box; faster | `True` |

| output | what it holds |
|---|---|
| `spec.axis_points` | the points along each axis, cm-1. The grid is every combination of them |
| `spec.amplitude` | complex amplitude, au, one value per grid point. intensity = abs(amplitude)**2 |
| `spec.mesh` | every grid point, one array per axis: what the renderer reads |
| `spec.spectral_window` | the features used on the grid, each with its feat_box |
| `spec.feature_result` | `result`, the input |

## 5. Plot

`plot_rsp_func(spec, dynrange=1000.)` in wilson_analysis/render/plot_spectrum.py.

## Known limits

- Ranks 3, 5 and 6: the averaging recipe has the slots in reverse order (FIXME in
  `calculate_avrg_tensor_einsum`). Ranks 2 and 4 are right.
- The line shape is a Lorentzian. [lineshape.py](lineshape.py) is the one place to change it.

## Files

| file | what it does |
|---|---|
| [pipeline.py](pipeline.py) | the two calls; FeatureResult, EvalRspResult; prop_names, terms_by_axis_set, the averaging recipe. Its docstring draws the whole flow |
| [plan.py](plan.py) | terms → CompiledTerm. No molecular data. The only module that imports wilson_derive when it runs (its abstractions); the others import it for type hints only |
| [evaluate.py](evaluate.py) | the numbers: coefficients, locations, the features on the grid |
| [features.py](features.py) | ContributionRow, SpectralFeature, SpectralWindow, the two box rules |
| [grid.py](grid.py) | Box, grid points, mesh, box_slices. No physics |
| [lineshape.py](lineshape.py) | the Lorentzian: its factor, peak, tail and distance |
| [compat.py](compat.py) | term_to_str, until VibPerturbedTerm.to_str is on main |

Code used from other packages: wilson_system (MolSystemData, the data reader),
wilson_orientational_averaging (the averaging recipe).

Tests: `python -m pytest src/wilson_suite/wilson_response_evaluate/tests`.
[tests/README.md](tests/README.md) lists the test files.
[tests/scripts/README.md](tests/scripts/README.md) lists the scripts that are run by hand.
