[![docs absent](https://img.shields.io/badge/docs-absent-red)](https://www.gnu.org/licenses/lgpl-3.0.html)
[![License LGPL 3.0](https://img.shields.io/badge/license-LGPL_v3.0-blue)](https://www.gnu.org/licenses/lgpl-3.0.html)
[![DOI:](https://img.shields.io/badge/DOI-set_it_up-red)](https://www.gnu.org/licenses/lgpl-3.0.html)

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


## Quick start

### Derived EVV terms

```python
import wilson_suite as ws
from wilson_suite.fixtures import evv_experiment

EVV_EXPERIMENT = evv_experiment()
DERIVED_EVV_TERMS = ws.derive.derive.get_fully_enhanced_terms(experiment=EVV_EXPERIMENT)
```

### Independent variables and possible axes choices

```python

```

### What data is needed for evaluation

```python

```



## Documentation

readthedocs.io?

