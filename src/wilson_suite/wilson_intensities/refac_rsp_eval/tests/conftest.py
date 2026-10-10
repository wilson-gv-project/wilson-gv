"""
Fixtures shared by the refac_rsp_eval tests: the toy molecule built from helpers.py.
"""

import numpy as np
import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import PrecalculatedData
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import PropsCollection
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
    CFF,
    E0_eigval,
    E1_eigval,
    polprop,
    toy_states,
)
from wilson_suite.wilson_system.system_data import (
    MolecularProperty,
    MolPropsCollection,
    MolSystemData,
    VibStatesData,
)


@pytest.fixture
def states() -> VibStatesData:
    return toy_states()


@pytest.fixture
def molsys(states) -> MolSystemData:
    props = MolPropsCollection([MolecularProperty(trivial_name='cff', vals=CFF, extra_data={})])
    return MolSystemData(name='toy', eigenvals={0: E0_eigval, 1: E1_eigval}, eigenvecs=None, mol_props=props, states=states)


@pytest.fixture
def pre_a1_zero() -> PrecalculatedData:
    """Precalculated <polgrad>[a] = (10, 0): every index set with a=1 has coeff 0."""
    avrg_key = PropsCollection([polprop(ops=(0, 1), inds='a')])
    return PrecalculatedData(avrg_tensors={avrg_key: np.array([10., 0.])},
                             avrg_expr_tensor_mapping={avrg_key: avrg_key})
