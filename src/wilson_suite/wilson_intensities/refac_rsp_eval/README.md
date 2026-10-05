# Cross imports

All paths below are under `wilson_suite.`. "type-only" means the import sits inside `if TYPE_CHECKING:` and
does not run at runtime. "lazy" means the import is inside a function body.

## Between modules of `refac_rsp_eval`

```
grid.py      <- (nothing)
plan.py      <- (nothing)
features.py  <- grid, plan
evaluate.py  <- features, plan
pipeline.py  <- evaluate, features, plan
```

| importer | from | names |
|---|---|---|
| `features.py` | `refac_rsp_eval.grid` | `Box`, `compute_box_adjacency`, `connected_components_from_adjacency`, `points_to_bounds`; type-only: `Dim_bounds` |
| `features.py` | `refac_rsp_eval.plan` | `ParameterSet`, `ResonanceMotif`; type-only: `ResLocPoint` |
| `evaluate.py` | `refac_rsp_eval.features` | `ContributionRow`, `ContributionTable` |
| `evaluate.py` | `refac_rsp_eval.plan` | `ParameterSet`, `ResLocPoint`; type-only: `CompiledTerm`, `FreqTermsCollection`, `PropsCollection`, `ResonanceMotif` |
| `pipeline.py` | `refac_rsp_eval.evaluate` | `PrecalculatedData`, `build_contributions` |
| `pipeline.py` | `refac_rsp_eval.features` | `ContributionTable`, `SpectralFeature`, `features_from_rows` |
| `pipeline.py` | `refac_rsp_eval.plan` | `ParameterSet`; type-only: `CompiledTerm` |

## Outgoing: `refac_rsp_eval` -> other `wilson_suite` packages

| importer | from | names |
|---|---|---|
| `grid.py` | — | numpy only |
| `plan.py` | `wilson_derive.abstractions` | `PolProp`, `ResonanceCondition`, `VibDiffTerm` |
| `plan.py` | `wilson_utils.prop_trivname` | `prop_trivname` |
| `plan.py` | `wilson_derive.response_terms` | type-only: `VibPerturbedTerm` |
| `plan.py` | `wilson_system.system_data` | type-only: `DataOriginInfo` |
| `features.py` | `wilson_utils.unit_convertor` | `convNu2Ene` |
| `evaluate.py` | `wilson_system.system_data` | `MolecularProperty`, `MolPropsCollection`, `MolSystemData`, `VibDiff`; type-only: `VibStatesData` |
| `evaluate.py` | `wilson_utils.prop_trivname` | `prop_trivname` (also re-imported lazily in one function) |
| `evaluate.py` | `wilson_utils.unit_convertor` | `convNu2Ene` |
| `evaluate.py` | `wilson_intensities.amplitudes.utils` | lazy: `generate_index_choices_general` |
| `pipeline.py` | `wilson_system.system_data` | `MolSystemData` |
| `rps_evaluation.py` (old code) | `wilson_experiment.indep_vars_and_axes` | `SpectralAxisSet` |
| `rps_evaluation.py` | `wilson_derive.response_terms` | `VibPerturbedTerm` |
| `rps_evaluation.py` | `wilson_main.abstractions` | `MolPropsCollection` |
| `rps_evaluation.py` | `wilson_utils.unit_convertor` | `convNu2Ene` |
| `rps_evaluation.py` | `wilson_intensities.amplitudes.*` | `SpectralFeature` (`spectrum_composition`), `VibDiffCache` (`vibene_differences`), `EvaluationDataAndConfigs`, `VibStatesData` (`term_parts`), `process_resonance_motifs`, `evaluate_terms_coeffs`, `get_features_to_draw` (`evaluators`), `precalculate_unique_coeff_parts`, `identify_precalc_unique_coeff_parts` (`full_amplitude_coeff`), `GridManager` (`grid_manager_evaluator`), `evaluate_region` (`evaluation_wf`) |

`rps_evaluation.py` is imported by nothing inside `refac_rsp_eval`. It is the only file here that depends on
`wilson_main`, `wilson_experiment`, and (apart from one lazy helper in `evaluate.py`) on `amplitudes`.

## Incoming: other packages -> `refac_rsp_eval`

| importer | from | names |
|---|---|---|
| `wilson_system/system_data.py` | `refac_rsp_eval.plan` | type-only: `CompiledTerm` |
| `wilson_system/tests/test_system_data.py` | `refac_rsp_eval.plan` | `CompiledTerm`, `FreqTermsCollection`, `ParameterSet`, `PropsCollection`, `ResCondKey`, `ResonanceMotif` |
| `wilson_system/tests/test_system_data.py` | `refac_rsp_eval.tests.helpers` | `E0`, `E01`, `E1`, `polprop`, `state`, `toy_states`, `toy_term`, ... |

Cycle note: `plan.py` <-> `wilson_system.system_data` import each other, but both directions are type-only.
`evaluate.py` -> `system_data` is a real import; `system_data` -> `plan` is type-only, so there is no runtime cycle.

## Tests (`tests/`)

Besides the package's own modules and `tests.helpers`, tests import:
`wilson_derive.abstractions` (`ResonanceCondition`, `HarmOscStateSymbolic`, `PolProp`, `QOperator`, `VibDiffTerm`),
`wilson_derive.term_var_translate`, `wilson_derive.response_terms.VibPerturbedTerm`,
`wilson_intensities.amplitudes.averaging.getGeneralPolarizationAveragingExpression`,
`wilson_system.system_data` (`MolecularProperty`, `MolPropsCollection`, `MolSystemData`, `VibState`, `VibStatesData`,
`DataOriginInfo`, `_sys_info_request`), and `wilson_utils` (`builders.make_SpectralAxisSet`, `paths.SUITE_ROOT`,
`prop_trivname`, `unit_convertor.convNu2Ene`, `wilson_data_obtainer`).

---

The data then passes through these steps:

```
1. build the request      "please fetch 'polgrad' from CFOUR"          plan.py:158
2. obtainer returns       data_dict = {'polgrad': array, ...}
3. fill mol_props         put data_dict['polgrad'] into the 'polgrad' slot   (slot name from from_polprop, evaluate.py:423)
4. evaluate               read mol_props['polgrad'].vals[mode, i, j]   evaluate.py:725 (averaged)                                                                       evaluate.py:618 (non-averaged)
```

# rsp_eval — evaluation of response-function terms

Derived terms plus molecular data in, spectrum out. Three modules, one direction of imports:

| module | in | out | note |
|---|---|---|---|
| `plan.py` | terms, axis choice | `tuple[CompiledTerm]` | only file that imports `wilson_derive`; no molecule |
| `evaluate.py` | compiled terms, `MolData` | `list[SpectralFeature]` | locations and coefficients become numbers |
| `grid.py` | features, window, resolution | spectrum array | full-grid Lorentzians |

`evaluate.py` and `grid.py` import numpy and `spectrum_composition`, nothing from `wilson_main` or
`wilson_derive`. The adapter that turns `VibAnaSetup` and `MolPropsCollection` into `MolData` lives in `wilson_main`.

## Signatures

```python
compile_terms(terms) -> (tuple[CompiledTerm])

compute_features(terms, data: MolData, gamma_cm1) -> list[SpectralFeature]

evaluate_on_grid(features, window, resolution, gamma_cm1, data, magn_conditions=None) -> (axes, spectrum)
```

Two boundary types, both frozen:

- `CompiledTerm`: term id, coefficient, averaged property slots, plain property slots, resonance motif
  (tuple of conditions with axes already substituted), perturbed-wavefunction differences, harmonic
  denominator labels, sorted mode labels, fixed labels. All tuples.
- `MolData`: number of modes, included modes, harmonic frequencies (cm-1), state energies as
  `{quanta tuple: cm-1}` with `()` the ground state, properties as `{name: array[modes..., components...]}`,
  laser polarization vector.

An index assignment is a `tuple[int]` aligned to the term's sorted labels.

## How the numbers are made

- Orientational average: one `np.einsum` of the property blocks against the dense polarization recipe
  (`getGeneralPolarizationAveragingExpression` as a `(3,)*rank` array). No shared tensors, no key mapping.
- Energy difference: `energies[left] - energies[right]`. No cache class.
- Harmonic denominator: product of `1/omega` over the labels.
- Coefficient: sum over free labels with `itertools.product`, product of the four factors above times
  the term coefficient. Terms sharing motif and location are summed into one feature; exact zeros dropped.
- Grid: every feature on the full grid, product over conditions of `1/(dE - sum(sign*axis) - i*gamma)`.
  Boxes and regions are gone; add per-feature index slicing only if a measured case is too slow.

## Rules

- Prefer deleting to designing. A type, cache or optimization needs a measured reason to exist.
- Frozen, tuples not lists, no derived value stored as a field, no reference to a later stage.
- Units in the name (`gamma_cm1`), converted once at the boundary, never guessed from magnitude.
- Pure functions; type checks at entry points only; validate at construction.
- A Python loop over index choices means an einsum was missed.
- One reason to open each file. Exactly one module imports `wilson_derive`; a test asserts it.
- Introduce an abstraction on the second concrete instance, not the first. No base classes, registries
  or second orchestrator. Delete rather than comment out.

## What was tried and dropped

- A six-module layout with a compile plan, work manifest and precomputed tables. Only worth it when
  precomputation is expensive; with einsum it is not.
- Deduplicating averaged tensors across terms by index-repetition pattern (`averaged_props.py`). Same reason.
- Wrapper classes around lists of derive objects (`PropsCollection`, `FreqTermsCollection`,
  `ResonanceMotif`, `ParameterSet`). Their only content was a canonical tuple; they also had to erase
  `PolProp.inds`, which gave one field two meanings and cost a deepcopy per motif.
- Copying the data model into intensities to break the `wilson_main` cycle. Taking plain arrays does it
  without a new package.
- A builder/sealed-setup orchestrator beside `WilsonSimulation`. One orchestrator.
- Region partitioning of the grid. An optimization that also truncates lineshapes.


---

"""
rsp_eval draft — response-function evaluation: design summary and refactoring TODO
==================================================================================

Long-form reasoning lives in refac_rsp_eval/q.md. This docstring is the actionable list.


Design summary
--------------
The pipeline has three independent inputs, hence three stages:

    symbolic   (wilson_derive)   terms with free symbols; no molecule, no experiment
    compiled   (plan)            terms + axis choice -> RspEvalTerm + WorkManifest; no molecule
    evaluated  (numeric)         plan + MolSystemData -> tables -> coefficients -> features


Ownership rule: a type belongs to the layer that defines its identity (__eq__ / __hash__ /
canonical form). Derive keeps PolProp.inds inside identity; the motif key here erases it.
One __eq__ per class => PropsCollection & co. are evaluator types, whatever they hold.

Boundary rules (all checkable):
    R1  exactly one module in wilson_intensities imports wilson_derive   (today: 8)
    R2  identity ownership, as above
    R3  never write to derive-owned objects   (today held only by remembering to deepcopy)
    R4  a method that builds a key into an external table is reusable by nobody without
        that table -> a class with such methods is not a generic container
    R5  the plan stage runs, and its output is assertable, with zero molecular data
    R6  shared-looking code goes at the consumer; promote on the second real use

Target layout (import direction one-way, top to bottom):

    rsp_eval/
      plan.py        # ONLY importer of wilson_derive
                     #   in:  terms, axis choice        out: EvalPlan + WorkManifest
                     #   holds: PropsCollection, FreqTermsCollection, ResonanceMotif,
                     #          parse_vibpert_term, index bookkeeping, motif keys
      ingest.py      # MolSystemData, VibStatesData — the data door
      precompute.py  # manifest + data -> Tables (resolution tensors, vibenedenom, vibdiff energies)
      kernel.py      # arrays and floats only; the hierarchical sum; picklable
      features.py    # locations + coefficients -> SpectralFeature
      render.py      # grid

Boundary types: EvalPlan (frozen, the compiled program), WorkManifest (what data/tensors are
needed; answers need_what()), Tables (numeric precomputation keyed by manifest entries),
Coefficients (dict[IndexAssignment, complex]), SpectralFeature (exists).



Open decisions
--------------
[ ] Does the plan depend on the axis choice, or only on the terms? TermsInAxes bundles
    both; this decides what a sweep over axes invalidates.
[ ] Motif identification (plan, no molecule) vs motif location (needs vibstates_data): one
    function today — split them.
[ ] Who owns the molecular data model: wilson_main (and accept the cycle) or a leaf package.
[ ] Is single-component Cartesian selection a wanted feature?

zero_tol fix and get_ind_tuple_from_base docstring TODOs are already done on this branch
"""
