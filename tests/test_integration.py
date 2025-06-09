"""
~/Wilson$ pytest --cov=wilson --cov-report term --cov=wilson --cov-config=../.coveragerc tests/test_integration.py::test_spectrum_compute
"""
import sys
import os
import numpy as np

from CQCParse.parsing import GaussianDataParser
from wilson.spectrum.tools import Conditions


np.set_printoptions(legacy=False)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



molecule, method, basis = "FORM", "B3LYP", "cc_pVQZ"


dynamic_range_n = 500
num_level_ticks = 15
Gamma = 4.7
diag_margin = 5.0

start1, end1 = 1000.0, 3150.0
step1 = 3.8

start2, end2 = 1000.0, 6150.0
step2 = 3.8

el_terms, mech_terms = [0, 1], [2, 3]

old_new_dict = {3: 0, 5: 1, 2: 2, 1: 3, 0: 4, 4: 5}
list2exclude = []

figmake = False
compute_intensity = True

maxmax = 3.276e9

common_fig_settings = {
    "w1mw2": True,
    "font_dict": {"size": 14},
    "figsize": (23, 37),
    "norm_max": None,
    "norm_min": None,
    "dynamic_range_n": None,
    "num_level_ticks": num_level_ticks,
    "levels_ticks": None,
    "levels": None,
    "maxYX": 4050,
    "max_int": maxmax,
    # 'minY': 1400,
    "directory": "./pics/",
    "prefix_name": "run",
    "saturation_color": "#FF00FF",
}


program = "gaussian"
parser = GaussianDataParser
upd_idx = old_new_dict

elevels = "anharm"
enelvl = False

print(f"\n{molecule}", basis, method, program, elevels)


def test_spectrum_compute():
    from wilson.utils import run_experiment1

    conds = Conditions(
        Gamma,
        diag_margin,
        dynamic_range_n,
        np.arange(start1, end1, step1),
        np.arange(start2, end2, step2),
        program,
        parser,
        molecule,
        method,
        basis,
        upd_idx,
        el_terms,
        mech_terms,
        list2exclude=list2exclude,
        only_modes=None,
        vib_levels_harmonic=enelvl,
        preview=False,
    )

    d = run_experiment1(
        conds,
        common_fig_settings,
        get_max=False,
        sparse=0.0,
        compute_intensity=compute_intensity,
        figmake=figmake,
    )

    assert np.count_nonzero(d['sec_hypol_dataALL_ref']) != 0


