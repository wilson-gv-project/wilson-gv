"""
lineshape.py — the line shape of one resonance condition. Runs with zero molecular data.
peak, tail and distance are each checked against factor, so they stay one shape.
"""
import numpy as np
import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval import lineshape


def test_factor_real_and_imaginary_parts():
    """Lorentzian, -i*gamma: Im = gamma / (x^2 + gamma^2) > 0, Re = x / (x^2 + gamma^2)"""
    x, gamma = np.linspace(-3., 3., 7), 0.5

    value = lineshape.factor(x, gamma)

    np.testing.assert_allclose(value.imag, gamma / (x**2 + gamma**2))
    np.testing.assert_allclose(value.real, x / (x**2 + gamma**2))


def test_peak_is_the_factor_at_resonance():
    assert lineshape.peak(0.5) == pytest.approx(1. / (-1j * 0.5))


@pytest.mark.parametrize('d, expected', [(0., 1.), (5., 2 ** -0.5), (100., 5. / np.hypot(100., 5.))])
def test_tail_is_the_factor_over_the_peak(d, expected):
    """gamma 5, Lorentzian: gamma / sqrt(d**2 + gamma**2)"""
    assert lineshape.tail(d, 5.) == pytest.approx(expected)


@pytest.mark.parametrize('drop', [0.5, 0.1, 1e-4])
def test_distance_is_where_the_tail_drops_to_drop(drop):
    assert lineshape.tail(lineshape.distance(drop, 5.), 5.) == pytest.approx(drop)


@pytest.mark.parametrize('drop', [1., 2.])
def test_distance_is_0_at_or_above_the_peak(drop):
    assert lineshape.distance(drop, 5.) == 0.


def test_peak_and_tail_follow_factor(monkeypatch):
    """Another factor changes peak and tail with it: only factor and distance are per line shape."""
    monkeypatch.setattr(lineshape, 'factor', lambda resloc, gamma: np.exp(-(resloc / gamma) ** 2))

    assert lineshape.peak(2.) == 1.
    assert lineshape.tail(2., 2.) == pytest.approx(np.exp(-1.))
