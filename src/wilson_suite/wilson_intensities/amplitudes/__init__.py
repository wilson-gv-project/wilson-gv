"""
Spectrum subpackage deals with evaluation of spectra

Evaluation is based on "terms", the result is the sum of results from terms.
Single "term" requires 6 parts to be computed:
    1. vibenediff denominator --    (vibenediff evaluation should be a function)
    2. averaged_props --            (evaluation should be a function)
    3. non_averaged_props --        (evaluation should be a function)
    4. vibene_denom --              (evaluation should be a function)
    5. resonance --                 (vibenediff + eval vars grid)
    6. additional float prefactors
Then all 6 parts are multiplied together into the result.


"""
from .averaging import get_iso_f, get_AlphaBetaGammaDelta_indices
from . import spectrum_composition
from . import evaluation_wf
from . import term_parts
from . import grid_manager_evaluator