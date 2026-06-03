[![docs absent](https://img.shields.io/badge/docs-absent-red)](https://www.gnu.org/licenses/lgpl-3.0.html)
[![License LGPL 3.0](https://img.shields.io/badge/license-LGPL_v3.0-blue)](https://www.gnu.org/licenses/lgpl-3.0.html)
[![DOI:](https://img.shields.io/badge/DOI-none-red)](https://www.gnu.org/licenses/lgpl-3.0.html)

# Wilson Suite

Wilson Suite is a Python package for computing response functions for vibrational wave-mixing spectroscopy.

## Key features

<!-- - Derivation of response function contribution expressions for vibrational wave-mixing spectroscopy experiments. -->

- Parse force constants, molecular properties, and derivatives from Gaussian and CFOUR outputs via CQCParse package
- Compute vibrational state energies across harmonic, VPT2, GVPT2, and DVPT2 regimes, with Fermi resonance detection
- Define spectroscopic experiments via parameters such as pulse configuration, detection method, phase-matching, and polarization
- Derive response function terms symbolically, with configurable anharmonicity limits and automatic perturbation expansion
- Evaluate response function amplitudes numerically on multi-dimensional spectral grids, with localized evaluation near resonances
- Identify which vibrational transitions produce resonant features in a given spectral region
- Compute isotropic orientational averages for a given polarization configuration
- Render 2D contour spectra with logarithmic normalization, dynamic range control, and configurable styling
- Run the full symbolic-to-spectrum pipeline through a single simulation container with built-in validation, diagnostics, and intermediate caching

## Installation

### From source
1. Get source code
2. Prepare the environment
3. Install

```
git clone git@github.com:wilson-gv-project/wilson-gv.git
cd wilson-gv
conda env create -f environment.yml
pip install .
```

### `CQCParse`  - package for parting quantum chemistry software outputs

```
git clone git@github.com:wlevand/CQCParse.git
cd CQCParse
pip install .
```

## Quick start

The example below runs the full EVV pipeline for formaldehyde.
A step-by-step walkthrough with intermediate outputs is in
[`examples/quick_start.ipynb`](examples/quick_start.ipynb).

### 1. Derive response function terms

Define the experiment and derive the symbolic response function terms, then
translate them to your chosen spectral axis variables.

```python
import wilson_suite as ws
from wilson_suite.fixtures import evv_experiment
from wilson_suite.wilson_utils.some_reprs import make_SpectralAxisSet
from wilson_suite.wilson_utils.paths import SUITE_ROOT

experiment = evv_experiment()

terms = ws.derive.derive.get_fully_enhanced_terms(experiment=experiment)
axis_choice = make_SpectralAxisSet({'A': [1], 'B': [-1, 2]})
translated_terms = ws.derive.term_var_translate.translate_terms_to_axis_variables(terms, axis_choice)
```

### 2. Configure the simulation

```python
from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
from wilson_suite.wilson_analysis.render.render_utils import NormalizationType
from wilson_suite.wilson_intensities.amplitudes.spectrum_composition import SpectralWindow, Box

sim = ws.main.workflow_abstractions.WilsonSimulation()
sim.addExperiment(experiment)
sim.addTerms(terms=translated_terms)

sim.addSystem(ws.main.abstractions.MolecularSystem(name='FORM', natoms=4))
vibana = ws.main.abstractions.VibAnaSetup(system=sim.system, regime='GVPT2', vibana_own_analysis='none')
sim.addVibAnaSetup(vibana)
sim.addPropEvalSetup(eval_uniform=ws.main.abstractions.DataOriginInfo(
    source_type='gaussian',
    lvl_theory='B3LYP',
    basis_set='cc-pVQZ',
    base_file_loc=SUITE_ROOT + '/../data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out',
))

sim.addSpecEvalSetup(ws.main.spectrum_abstractions.SpecEvalSetup(
    ev_info=ws.main.spectrum_abstractions.EvaluationInfo(
        Gamma=4.7,
        Gamma_unit='cm-1',
        dynamic_range=500,
        grid_resolution={'A': 700, 'B': 700},
        spectral_window=SpectralWindow(box=Box({'A': (700., 3400.), 'B': (10., 3500.)})),
    ),
    rnd_info=ws.main.spectrum_abstractions.RenderingInfo(
        intensity_normalization_type=NormalizationType.LOG_RATIO,
        filename='evv_spectrum.svg',
    ),
))
```

### 3. Evaluate and render

```python
sim.setPropsAndMaxStateLvl()
sim.dressPropsWithSetup()
sim.getResults(obtainer=wilson_data_obtainer)
sim.vib_ana_setup.set_include_modes_list()

sim.axis_choice = axis_choice
sim.terms_in_axis_choice = translated_terms
sim.evaluate()

sim.spec_eval_setup.rnd_info.style_config.figsize = (10, 10)
sim.render_spectrum(do_diagn=False)
# spectrum saved to evv_spectrum.svg
```


## Documentation

readthedocs.io?

