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


TODO — structure
----------------
[ ] Decide the fate of this file vs amplitudes/term_parts.py, averaged_props.py,
    vibene_differences.py. The classes below duplicate those. One tree survives, not both.
[ ] Delete the copied data-model classes: DataOriginInfo, MolecularProperty,
    MolPropsCollection, VibState (the tab-indented block is the copy from
    wilson_main/abstractions.py). Import them from a leaf package that sits below both
    wilson_main and wilson_intensities. Package-level cycle today:
    wilson_main.spectrum_abstractions -> amplitudes; amplitudes.evaluators -> wilson_main.
    Three VibState definitions exist (wilson_main/abstractions.py, utils/spectrum_utils.py,
    here).
[ ] MolSystemData: keep as the single data door. Also defined in rps_evaluation.py:115 —
    one home.
[ ] Split EvaluationDataAndConfigs by reader: config (plan stage) vs data (numeric stage).
    Split PrecalculatedData into named tables. No None default that is dereferenced
    unconditionally (README rule 2).
[ ] Make RspEvalTerm the only thing the evaluator sees. evaluate_term_coeffs re-derives
    avrg_expr / non_avrg_expr / freqterms / index split from the raw term inside the loop,
    while RspEvalTerm is consumed by nothing.
[ ] parse_vibpert_term is the one derive-facing function. Bugs: term_id is never passed
    (required field -> TypeError); num_coeff is typed FreqTermsCollection, assigned float.
[ ] RspEvalTerm frozen; all_indices as a @property, not a stored derived field (README rule 4).
[ ] evaluate_single_index_dict takes the whole term only to read term.coeff — pass the float.
[ ] Normalise indentation (mixed tabs/spaces from the copied block).


TODO — value types
------------------
[ ] PropsCollection / FreqTermsCollection __eq__ is `all(p in other)`: asymmetric, ignores
    length and multiplicity — and these are dict keys. Derive __eq__ and __hash__ from one
    canonical tuple (ResonanceMotif._tuplify already does this correctly).
[ ] PropsCollection.sort() mutates self.props (and turns the tuple back into a list) on an
    object used as a dict key. Frozen; sorted() returns a new instance.
[ ] Split PropsCollection along its three method families:
      (a) payload predicates — get_cart_axes, get_mode_indices, bool(p.ops) — may move to
          derive as free functions over Sequence[PolProp];
      (b) identity / canonical form — a frozen key type, evaluator-owned;
      (c) key builders for external tables — evaluator-owned, named as such.
[ ] identify_avrg_motif: translate into an own key type instead of deepcopy + writing
    inds=None onto PolProp. inds=None means "not yet assigned" to derive and
    "index-agnostic" here; that collision is why this class hand-rolls __hash__ and guards
    None in three places. It also returns None implicitly on empty, and that None becomes a
    dict key in group_PropsColls_by_numerator. Make it total (raise) or handle None at the
    call site.
[ ] get_mode_indices_group_template: the None branch returns [] inside a list of ints.
[ ] ResonanceMotif.resonance_location_class is a @property with a required parameter.
    to_str is EVV/paper-specific — move it out of the type. Drop the UNUSED methods.
[ ] ParameterSet: declared Mapping[str, int] but injects 'zero': 'zero' (a str) and remaps
    '' -> 'zero'; __lt__ hardcodes ('a'..'h'). Decide: generic frozen index assignment (the
    conventions live in a labelling module) or a domain type (IndexAssignment) with the
    conventions explicit and tested. The ground state is spelled '', 'zero', and
    state_label == 'zero' within this file.
[ ] VibDiff.cache_it / VibDiffCache thread a cache through a domain object (README rules 5, 7).
    The energy difference is a pure function of a normalised label pair — memoise that
    function. make_vibdiff_key is a key builder -> plan stage. VibDiff.from_symbolic takes a
    VibDiffTerm, i.e. a derive-boundary crossing outside plan.py.
[ ] VibStatesData._fill_storage unpacks dict keys as pairs (broken; UNUSED).
    harmonic_osc_states_labels defaults to None and is dereferenced in
    get_harmonic_osc_states.
[ ] MolecularProperty.h reads self.system (not a field) and calls calc_setup.h() (no such
    method); MolPropsCollection.of_order reads p.order (not a field). Moot once the copies go.


TODO — numerics / physics
-------------------------
[ ] Cache-key gap (a live defect, independent of any design choice): the tensors in
    avrg_tensors are already contracted with the polarization recipe, but the key is
    (Cartesian motif, repetition pattern) only. ZZZZ and XXYY collide. The key must include
    the recipe. Manifest entry = motif x repetition pattern x CartesianResolution.
[ ] Name the recipe: CartesianResolution = {component tuple: coefficient}. Today it exists
    only as the local polarization_linear_comb inside make_gen_func_to_compute_avrg, which
    constructs it (getGeneralPolarizationAveragingExpression) rather than receiving it.
    pulse_polarization_vector is threaded through EvaluationDataAndConfigs ->
    calculate_avrg_tensor -> make_gen_func_to_compute_avrg as a stand-in for that recipe.
[ ] Single molecule-frame component selection (e.g. gamma_xxyz) is a one-entry recipe
    {(x,x,y,z): 1.0}; the evaluation loop already handles it unchanged. DECIDE whether the
    feature is wanted. If yes: the avrg vocabulary (~29 identifiers, ~180 sites, plus the
    filename averaged_props.py) has to move, as its own commit, never mixed with behaviour
    changes. If no: leave avrg scoped and honest and document the limitation.
[ ] Zero test: np.isclose(x, zero_tol) with zero_tol=1e-18 evaluates
    abs(x - 1e-18) <= 1e-8 + 1e-5*1e-18, i.e. effectively abs(x) <= 1e-8. The zero_tol
    argument is ignored. Use abs(x) < tol (eval_non_avrg_per_indexdict,
    eval_avrg_per_indexdict).
[ ] get_ind_tuple_from_base: sorted(set(base symbols)) is a no-op only because
    nm_indices_repetition_decoding allocates letters in first-appearance order — assert
    that invariant. The `<` branch is the repeated-index rank-reduction path
    ((a,b,a) -> rank-2 tensor), not dead code; say so in its docstring. The function is
    duplicated from averaged_props.py.
[ ] The motif is one of two keys. Stripping inds is correct — mode indices are spectators
    to orientational averaging; the repetition pattern recovers the coincidence structure.
    Keep the two-level grouping and document it where the motif is defined.
[ ] eval_non_avrg_per_indexdict indexes vals with mode indices only. Correct for ops=[]
    props (F_abc); returns a sub-array for any ops-carrying prop routed there. Currently
    unreachable — assert `not prop.ops`.


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
