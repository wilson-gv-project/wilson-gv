# test_config.py
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

@dataclass
class SimulationConfig:
    gammaCompsAll: Any
    molecule: str
    method: str
    basis: str
    Gamma: float
    diag_margin: float
    start1: float
    end1: float
    step1: float
    start2: float
    end2: float
    step2: float
    old_new_dict: Dict[int, int]
    elevels: str
    enelvl: bool
    w1m: np.ndarray
    w2m: np.ndarray

import string
abc_list = list(string.ascii_lowercase)