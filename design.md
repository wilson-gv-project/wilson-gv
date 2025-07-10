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


## 🛠️ Implementation Roadmap

1. Define `VibrationalSystem` as immutable container
2. Stub out `SpectrumSimulator.simulate_signal`
3. Write test for expected dipole input/output dimensions
4. Design plotting and file export API last


------------------------------------------------
# Notes from 30.06.2025

1. **wilson-main**: externalCalcSetup: dataclass + immutable. other_setup: custom data from user? input_generation functionality - separate class
2. **wilson-main**: wilsonSimulation: report method; saving instances/setups; saving results
3. **wilson-main**: evaluate: self.spec - np.ndarray, self.diagn - dict;
4. **😺❔wilson-main**: more general evaluator would take experiment info. Other evaluator (evaluate_as_response) just evaluates response function
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

