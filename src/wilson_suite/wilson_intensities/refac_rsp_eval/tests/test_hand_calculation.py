"""
G1 (todo.txt): two EVV coefficients by hand, for water, against evv_reference.json.
C2 and C3 show old = new; this shows the numbers are right. The checking side is numpy and textbook formulas,
no wilson code, so an error in code that old and new share (derive's terms, units, the averaging) shows here.

Water has 3 modes (0, 1, 2): every pair of fixed labels (a, b) is checked, and term 12's sum over c has 3 entries.
Inputs: the data dict of data_for_tests/g16_h2o_HF_STO3G.out, read with wilson_data_obtainer (input side only;
G2 checks the data dict against the .out file). A change in the reader would still show: the stored numbers
of evv_reference.json would no longer match.
Settings of evv_reference.json: laser_pol (1, 1, 1), i.e. all pulses and the detector along x.

Two kinds of denominator (rule from the user, 2026-10-10):
    w            the Hessian eigenvalue (nc_sqrt_eigval), the same for any states choice
    E(1) - E(0)  a state energy difference: harmonic or anharmonic states
to_str prints both as <['a'],[]>; derive's is_pert_wf_diff says which is which (True: state energy difference).

Checked 2026-10-10, both states choices: term 0, 9 pairs, largest relative difference 5.4e-13; term 12, 6 pairs at
1.07e-12, and the 3 pairs with a = 2 exactly 0, as stored (cff_(2,c,c) = 0: mode 2 is antisymmetric). All of the
difference comes from the cm-1 per Eh constant (see REL): term 0 has 2 energy factors, term 12 has 4.
(First done for CO, 1 mode; replaced by water so that the sum over c has more than one entry.)
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wilson_suite.wilson_system.system_data import DataOriginInfo
from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer

REFERENCE_FILE = Path(__file__).parent / 'evv_reference.json'
WATER_FILE = Path(__file__).parents[3] / 'data_for_tests' / 'g16_h2o_HF_STO3G.out'
CM1_PER_EH = 219474.6313632     # CODATA 2018 (CODATA 2014's 219474.6313702 would miss by 6e-11)
STATES = ('harmonic', 'anharmonic')
REL = 5e-12                     # wilson's cm-1 per Eh and CODATA 2018's differ by 2.7e-13, once per energy factor;
                                # abs=0 always: pytest.approx's default abs=1e-12 would allow 1e-4 on these coefficients


@pytest.fixture(scope='module')
def water() -> SimpleNamespace:
    """The data dict as numpy arrays, energies in Eh. Cartesian x, y, z; modes 0, 1, 2."""
    origin = DataOriginInfo(source_type='gaussian', base_file_loc=str(WATER_FILE))
    keys = ('dipgrad', 'polgrad', 'polhess', 'cff', 'nc_sqrt_eigval', 'harmonic_states', 'anharmonic_states')
    data = wilson_data_obtainer(requested_data_dict={k: origin for k in keys})
    modes = range(len(data['nc_sqrt_eigval']))
    return SimpleNamespace(
        modes=modes,
        dipgrad=data['dipgrad'],        # [a, x]
        polgrad=data['polgrad'],        # [a, x, y]
        polhess=data['polhess'],        # [a, b, x, y]
        cff=data['cff'],                # [a, b, c]
        w=np.array([data['nc_sqrt_eigval'][a] for a in modes]) / CM1_PER_EH,
        # E(1_a) - E(0); the stored energies are relative to the ground state
        e1={s: np.array([data[f'{s}_states'][(str(a),)] for a in modes]) / CM1_PER_EH for s in STATES})


def iso_average_xxxx(t: np.ndarray) -> float:
    """
    Average of t_xxxx over all orientations of the molecule (rank 4, everything along x):
        (1/15) * sum over p, q of (t_ppqq + t_pqpq + t_pqqp)
    Only the Cartesian indices are averaged: rotating the molecule never changes which mode is which.
    """
    return (np.einsum('iijj', t) + np.einsum('ijij', t) + np.einsum('ijji', t)) / 15


def term_0(mol: SimpleNamespace, a: int, b: int, states: str) -> float:
    """
    -1/4 * dipgrad_(a) * dipgrad_(b) * polhess_(a,b) / w_a / w_b
    Slots (Cartesian index positions): polhess 0 and 3, dipgrad_(a) 1, dipgrad_(b) 2.
    states: 'harmonic' or 'anharmonic', which state energies to use. Unused here: both denominators are w
    (the Hessian eigenvalue), and no state energy enters, so both choices give the same number.
    """
    t = np.einsum('il,j,k->ijkl', mol.polhess[a, b], mol.dipgrad[a], mol.dipgrad[b])
    return -1 / 4 * iso_average_xxxx(t) / (mol.w[a] * mol.w[b])


def term_12(mol: SimpleNamespace, a: int, b: int, states: str) -> float:
    """
    1/16 * polgrad_(b) * dipgrad_(a) * dipgrad_(b) * cff_(a,c,c) / (E(1_a) - E(0)) / w_a / w_b / w_c, summed over c
    Slots: polgrad 0 and 3, dipgrad_(a) 1, dipgrad_(b) 2. cff is not averaged, and it is the only part with c,
    so the sum over c is a factor of its own.
    states: 'harmonic' or 'anharmonic', which state energies to use. Used here: one of the two <['a'],[]> is the
    state energy difference E(1_a) - E(0).
    """
    t = np.einsum('il,j,k->ijkl', mol.polgrad[b], mol.dipgrad[a], mol.dipgrad[b])
    sum_over_c = sum(mol.cff[a, c, c] / mol.w[c] for c in mol.modes)
    return 1 / 16 * iso_average_xxxx(t) * sum_over_c / (mol.e1[states][a] * mol.w[a] * mol.w[b])


def reference_coeffs(term_id: int, states: str) -> dict[tuple[int, int], float]:
    """{(a, b): coeff} of evv_reference.json for water; a pair in the file's `zero` list gets 0."""
    reference =json.loads(REFERENCE_FILE.read_text())
    assert reference['settings']['laser_pol'] == [1., 1., 1.]
    (block,) = [b for b in reference['results'] if b['molecule'] == 'water' and b['states'] == states]
    coeffs = {(r['params']['a'], r['params']['b']): r['coeff'] for r in block['rows'] if r['term'] == term_id}
    coeffs.update({(z['params']['a'], z['params']['b']): 0. for z in block['zero'] if z['term'] == term_id})
    return coeffs


@pytest.mark.parametrize('states', STATES)
@pytest.mark.parametrize('term_id, by_hand', [(0, term_0), (12, term_12)])
def test_hand_calculation_gives_the_reference_coefficients(water, term_id, by_hand, states):
    stored = reference_coeffs(term_id, states)
    assert set(stored) == {(a, b) for a in water.modes for b in water.modes}

    wrong = [(pair, coeff, by_hand(water, *pair, states)) for pair, coeff in stored.items()
             if by_hand(water, *pair, states) != pytest.approx(coeff, rel=REL, abs=0)]

    assert wrong == [], '((a, b), stored, by hand)'


def test_term_12_needs_the_state_energy_difference_not_w(water):
    """Control: anharmonic states, but w where E(1_a) - E(0) belongs. Every pair is off by E(1_a) / w_a."""
    stored = reference_coeffs(12, 'anharmonic')
    with_w = SimpleNamespace(**{**vars(water), 'e1': {'anharmonic': water.w}})

    for (a, b), coeff in stored.items():
        if coeff != 0.:
            ratio = term_12(with_w, a, b, 'anharmonic') / coeff
            assert ratio == pytest.approx(water.e1['anharmonic'][a] / water.w[a], rel=REL, abs=0)
            assert abs(ratio - 1) > 0.01


def test_term_0_needs_w_not_the_state_energy_difference(water):
    """Control: E(1) - E(0) of the anharmonic states in place of w: every nonzero pair is off by 2 % or more."""
    stored = reference_coeffs(0, 'anharmonic')
    with_e1 = SimpleNamespace(**{**vars(water), 'w': water.e1['anharmonic']})

    for (a, b), coeff in stored.items():
        if coeff != 0.:
            assert term_0(with_e1, a, b, 'anharmonic') != pytest.approx(coeff, rel=0.02, abs=0)
