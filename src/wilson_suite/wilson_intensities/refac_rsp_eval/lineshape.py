"""
The line shape of one resonance condition, in one place: its factor on the grid, its value
at resonance, and how fast it falls off from there.

  holds: lorentzian_factor, lorentzian_distance   the Lorentzian
         factor, distance                         the line shape in use: the Lorentzian
         peak, tail                               from factor, for any line shape

The evaluation takes the line shape only from here, through factor, distance, peak and tail:
    evaluate.py   the factor on the grid (eval_resonance_factor)
    features.py   the peak height (SpectralFeature.get_intensity), the amplitude reaching
                  into a window (amplitude_in_window) and the box sizes (the box rules)
So these always agree. Another line shape, e.g. a Gaussian, adds its own factor and distance
functions and sets factor and distance to them.
gamma: the line shape's parameter, in the same unit as resloc and d (cm-1 or Eh).

Imports nothing from the rest of the package.
"""
import math

# the Lorentzian, 1 / (E - freq - i*gamma); gamma is its half width

def lorentzian_factor(resloc, gamma):
    """The factor on the grid at resloc = E - freq, one number or an array: 1 / (resloc - i*gamma)."""
    return 1 / (resloc - 1j * gamma)


def lorentzian_distance(drop: float, gamma: float) -> float:
    """
    The distance from resonance where tail falls to `drop`, the inverse of tail:
    gamma * sqrt(1 / drop**2 - 1); 0 for drop >= 1 (the peak is already at or below it).
        e.g. gamma 5, drop 0.1 -> 49.7
    """
    if drop >= 1.:
        return 0.
    return gamma * math.sqrt(1. / drop ** 2 - 1.)


# the line shape in use; peak, tail and the evaluation call it through these two names

factor = lorentzian_factor
distance = lorentzian_distance


# for any line shape, from factor

def peak(gamma) -> complex:
    """The factor at resonance, factor(0, gamma); for the Lorentzian 1 / (-i*gamma) = i / gamma."""
    return factor(0., gamma)


def tail(d, gamma) -> float:
    """
    |factor| at distance d from resonance, as a fraction of |peak|: 1 at d = 0.
    For the Lorentzian gamma / sqrt(d**2 + gamma**2), about gamma / d far away.
        e.g. gamma 5: tail(5) = 0.71, tail(100) = 0.05
    """
    return abs(factor(d, gamma) / factor(0., gamma))
