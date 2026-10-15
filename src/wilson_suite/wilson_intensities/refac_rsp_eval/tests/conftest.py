"""
Fixtures shared by the refac_rsp_eval tests: the toy molecule built from helpers.py.
"""

from dataclasses import replace

import pytest

from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import (
    CFF,
    TOY_POLGRAD,
    E0_eigval,
    E1_eigval,
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
    return MolSystemData(name='toy', eigenvals={0: E0_eigval, 1: E1_eigval}, eigenvecs=None, mol_props=props,
                         states={'harmonic': states})


@pytest.fixture
def molsys_a1_zero(molsys) -> MolSystemData:
    """molsys plus polgrad: with helpers.TOY_RECIPE, <polgrad>[a] = (10, 0), so every index set with a=1 has coeff 0."""
    polgrad = MolecularProperty(trivial_name='polgrad', vals=TOY_POLGRAD, extra_data={})
    return replace(molsys, mol_props=MolPropsCollection([*molsys.mol_props, polgrad]))
