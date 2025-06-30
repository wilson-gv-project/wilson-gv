# 🧠 Conceptual Design: [Project Title]

## 🎯 Goal
Concise description of what this software will do.

---

## 📦 Modules and Responsibilities

### `spectrum_simulator.py`
**Responsibility**: Core nonlinear signal simulation (rephasing, non-rephasing)

#### Classes & Key Functions

##### `SpectrumSimulator`
- **Role**: Main class for computing time-domain IR signals
- **Methods**:
  - `simulate_signal(...)`
    - **Inputs**: energy states, dipoles, dephasing params
    - **Output**: 2D time-domain array
    - **Notes**: Should be stateless if possible for testability
    - 🔧 *Idea*: Consider functional API in parallel to OO for flexibility

  - `apply_lineshape(...)`
    - Apply damping window; could be extracted into utility module
    - ❗ *Performance bottleneck* if FFTs are used poorly

---

### `vibrational_model.py`
**Responsibility**: Define system energy levels & couplings

#### Classes & Key Functions

##### `VibrationalSystem`
- **Attributes**:
  - `hamiltonian_matrix`
  - `dipole_matrix`
  - `coupling_constants`
- **Design Note**:
  - Use `@dataclass` for readability and immutability
  - Maybe split into a builder (`VibrationalSystemBuilder`) vs pure data object

##### `diagonalize_hamiltonian(H)`
- Should be a pure function
- 🔍 *Edge case*: Degeneracy handling unclear — needs test

---

## 🔁 Data Flow

1. `VibrationalSystem` → `SpectrumSimulator`
2. → Simulate signal
3. → Apply FT + lineshape
4. → Plot / export

---

## 🔗 Inter-module Interfaces

| From | To | Data |
|------|----|------|
| `vibrational_model` | `spectrum_simulator` | `energy_states`, `mu_matrix` |
| `spectrum_simulator` | `plotting.py` | 2D array |
| All | `utils/units.py` | frequency/energy conversion |

---

## 🚧 Design Notes (General or Cross-Cutting)

- **Memory model**: avoid keeping raw time-domain and FT arrays in memory at once
- **Parallelism**: explore `joblib` or `numba` for FFT sections
- **Logging**: allow optional verbose/debug modes for numerical tracking

---

## ❓ Open Questions

- Should we cache diagonalization results if Hamiltonian doesn’t change?
- Do we want full OO design or hybrid with functional helpers?

---

## 🛠️ Implementation Roadmap

1. Define `VibrationalSystem` as immutable container
2. Stub out `SpectrumSimulator.simulate_signal`
3. Write test for expected dipole input/output dimensions
4. Design plotting and file export API last


------------------------------------------------
# Notes from 30.06.2025

1. externalCalcSetup: dataclass + immutable. other_setup: custom data from user? input_generation functionality - separate class
2. wilsonSimulation: report method; saving instances/setups; saving results
3. evaluate: self.spec - np.ndarray, self.diagn - dict;
4. more general evaluator would take experiment info. Other evaluator (evaluate_as_response) just evaluates response function
5. Canonical indices in terms
6. Identification in TermsEvalua - clean up
7. Use dictionaries to hold properties values? - lower priority
8. terms_evaluator: always identify and precalculate? - check if it actually works without precalc;
9. TermsEvaluator.precalc_avrg_tensors: generalize greek indices -- dict of greek letters
10. make indices abc into tuples 
11. Non-averaged data - check namings and functions
12. calculationBatch.getResultsFromVault - VL
13. vibAnaSetup clarifications: class name, external_fill_from name; 
14. basic renderer
15. harmonic precal data - fix
16. test_clean_termND  - produces different molecule????
17. include renderer to evv_tester
18. wilson_suite: update.sh file to go to specific commit on a specific branch for each repo
19. another look at signs of perturbing freqs in terms and signs of spectroscopic axes

