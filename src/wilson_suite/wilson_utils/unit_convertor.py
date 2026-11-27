from scipy import constants
import numpy as np

per_cm_x_fs = 5308.837458876145


def convNu2Ene(values: float | np.ndarray, reverse: bool = False) -> float | np.ndarray:
    """
    Convert a wavenumber (cm-1) to energy (Hartree) and reverse if specified
    """
    hartree2J = constants.physical_constants['hartree-joule relationship'][0]
    if not reverse:
        return values * (100 * constants.h * constants.c / hartree2J)
    else:
        return values / (100 * constants.h * constants.c / hartree2J)
