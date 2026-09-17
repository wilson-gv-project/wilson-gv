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
