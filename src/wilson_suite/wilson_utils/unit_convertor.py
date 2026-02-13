from scipy import constants
import numpy as np
from typing import Any

bohr_in_angstroms = constants.physical_constants['Bohr radius'][0]/10**(-10)

def rcm2Eh_a0(rcm_array: np.ndarray, harm_freqs) -> np.ndarray:
    """ Convert from [cm-1] to [Hartree/Bohr^n] """
    
    if rcm_array.ndim == 3:
        n = len(harm_freqs)
        K3 = np.zeros((n, n, n), dtype=np.float64)

        for i in range(rcm_array.shape[0]):
            for j in range(rcm_array.shape[1]):
                for k in range(rcm_array.shape[2]):
                    d = rcm_array[i, j, k]

                    d *= np.sqrt(harm_freqs[i] * harm_freqs[j] * harm_freqs[k])

                    a = np.sqrt(constants.h / constants.c / constants.physical_constants['unified atomic mass unit'][0] / 100)
                    b = 10 ** 10 / 2 / np.pi / constants.physical_constants['Bohr radius'][0] / 10 ** 10
                    Fact3R = (constants.physical_constants['hartree-joule relationship'][
                                    0] / constants.h / constants.c / 100) * (a * b) ** 3

                    d /= Fact3R

                    K3[i, j, k] = d
                    K3[i, k, j] = d
                    K3[k, j, i] = d
                    K3[k, i, j] = d
                    K3[j, i, k] = d
                    K3[j, k, i] = d
        return K3
    
def un_massweight(FF_array):
    """
    Convert a force constant array from mass-weighted to un-mass-weighted
    e.g., from [Hartree*amu(-3/2)*Bohr(-3)] to [Hartree*m_e(-3/2)*a0(-3)] - gaussian K(I,J,K)
    """
    # amu to au mass unit (m_e)
    amc_au = constants.physical_constants['atomic mass constant'][0] / \
                constants.physical_constants['atomic unit of mass'][0]

    return FF_array / amc_au**(0.5 * FF_array.ndim)


def GHz2Nu(ghz: float | np.ndarray) -> float | np.ndarray:
    """Conversion from GHz to cm-1"""
    return ghz*10**9/(constants.c*100)

def convNu2Ene(values: float | np.ndarray, reverse: bool = False) -> float | np.ndarray:
    """
    Convert a wavenumber (cm-1) to energy (Hartree) and reverse if specified
    """
    hartree2J = constants.physical_constants['hartree-joule relationship'][0]
    if not reverse:
        return values * (100 * constants.h * constants.c / hartree2J)
    else:
        return values / (100 * constants.h * constants.c / hartree2J)


def convertor(system,
              prop_spec: dict, vals: Any,
              in_basis: str, target_basis: str,
              in_units: str, target_units: str,
              convertor_info: dict={}) -> Any:
    """ Convert vals from in_basis to target_basis and from in_units to target_units
        prop_spec is a dictionary with property specification (e.g. trivial_name)
        convertor_info is further information for the convertor if needed
    """
    # Basis conversion
    if in_basis != target_basis:
        if in_basis == 'mw' and target_basis == 'reduced':
            vals = un_massweight(vals)
        else:
            raise ValueError(f'Basis conversion from {in_basis} to {target_basis} not implemented')

    # Units conversion
    if in_units != target_units:
        # conversion of states energies from cm-1 to Eh
        if in_units == 'cm-1' and target_units == 'Eh':
            vals = convNu2Ene(values=vals, reverse=False)

        # conversion of force constants from cm-1 to Eh/Bohr^n
        elif in_units == 'cm-1' and target_units == 'au':
            vals = rcm2Eh_a0(rcm_array=vals, harm_freqs=convertor_info['harm_freqs'])

        # conversion of frequencies from cm-1 to GHz (rotational constants)
        elif in_units == 'GHz' and target_units == 'cm-1':
            vals = GHz2Nu(ghz=vals)

        # energy from Eh/Bohr^n to cm-1
        elif in_units == 'Eh' and target_units == 'cm-1':
            vals = convNu2Ene(values=vals, reverse=True)

        else:
            raise ValueError(f'Units conversion from {in_units} to {target_units} not implemented')

    return vals


def recip_cm_or_au(energy):
    """
    Distinguish between cm-1 and atomic units based on magnitude.
    Uses the fact that 1 hartree ≈ 219,475 cm-1
    
    Threshold logic:
    - Values > 0.5 are likely cm-1 (even weak transitions are >100 cm-1)
    - Values < 0.5 are likely au (even strong transitions are <1 hartree)
    """
    if energy > 0.5:
        return 'cm-1'
    elif energy > 0:
        return 'au'
    else:
        raise ValueError(f"Energy must be positive, got {energy}")


def linewidth_cm_or_au(width):
    """
    Distinguish between cm-1 and atomic units based on magnitude.
    For linewidths/broadening parameters
    """
    if width > 1e-5:
        return 'cm-1'
    else:
        return 'au'
