import itertools

import pytest

from wilson_suite.wilson_orientational_averaging.averaging import (
    getGeneralPolarizationAveragingExpression,
)
from wilson_suite.wilson_response_evaluate.evaluate import (
    calculate_avrg_tensor_einsum,
    make_avrg_einsum_subscripts,
)
from wilson_suite.wilson_response_evaluate.pipeline import prop_names
from wilson_suite.wilson_response_evaluate.plan import compile_terms
from wilson_suite.wilson_response_evaluate.tests.helpers import (
    LASER_POL_DISTINCT_PAIRINGS,
    make_func_to_compute_avrg,
)
from wilson_suite.wilson_response_evaluate.tests.stored_terms import load_setup
from wilson_suite.wilson_system.system_data import (
    DataOriginInfo,
    MolSystemData,
)
from wilson_suite.wilson_utils.paths import SUITE_ROOT

# the 14 EVV terms, already translated to A = w1, B = -w1 + w2 (stored_terms.py)
translated_terms = load_setup('evv').terms
evv_terms = translated_terms    # prop_names reads only the props
cmpl_terms = compile_terms(terms=translated_terms)


## All 14 EVV terms: the einsum tensor = the on-the-fly loop, formaldehyde -----------------------------
# Each term reads the tensor with its own labels, e.g. term 8: 'bIL,aJ,cK,IJKL->abc', read at {'a', 'b', 'c'}.
# Tolerance: rel 1e-12; abs 1e-18, the evaluator's zero_tol (below it a value counts as 0): where the loop gives 0,
# einsum gives rounding noise.

ALL_ABC = [dict(zip('abc', combo)) for combo in itertools.product(range(6), repeat=3)]

EVV_SUBSCRIPTS = ['abIL,aJ,bK,IJKL->ab', 'bIL,aJ,abK,IJKL->ab', 'bIL,aJ,bK,IJKL->ab', 'cIL,aJ,bK,IJKL->abc',
                  'aIL,aJ,bK,IJKL->ab', 'aIL,aJ,bK,IJKL->ab', 'cIL,aJ,bK,IJKL->abc', 'bIL,aJ,bK,IJKL->ab',
                  'bIL,aJ,cK,IJKL->abc', 'bIL,aJ,bK,IJKL->ab', 'bIL,aJ,aK,IJKL->ab', 'bIL,aJ,aK,IJKL->ab',
                  'bIL,aJ,bK,IJKL->ab', 'bIL,aJ,cK,IJKL->abc']


@pytest.fixture(scope='module')
def evv_molsys() -> MolSystemData:
    """all properties of the 14 terms; the states do not enter the average"""
    origin = DataOriginInfo(source_type='gaussian', base_file_loc=SUITE_ROOT+'/data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')
    return MolSystemData.from_output_file(origin, prop_names(evv_terms))


def test_evv_einsum_subscripts():
    """
    term 0:  polhess[a, b] (slots 0, 3) dipgrad[a] (1) dipgrad[b] (2)      'abIL,aJ,bK,IJKL->ab'
    term 1:  polgrad[b] (0, 3) dipgrad[a] (1) diphess[a, b] (2)            'bIL,aJ,abK,IJKL->ab'
    term 8:  polgrad[b] (0, 3) dipgrad[a] (1) dipgrad[c] (2)               'bIL,aJ,cK,IJKL->abc'
    """
    assert [make_avrg_einsum_subscripts(t.avrg_props) for t in cmpl_terms] == EVV_SUBSCRIPTS


@pytest.mark.parametrize('term_id', range(14))
def test_evv_einsum_tensor_is_the_on_the_fly_average_for_every_index_set(term_id, evv_molsys):
    term = cmpl_terms[term_id]
    pol = getGeneralPolarizationAveragingExpression(rank=4, laser_pol=LASER_POL_DISTINCT_PAIRINGS)
    func = make_func_to_compute_avrg(avrg_expression=term.avrg_props, polarization_linear_comb=pol)

    tensor = calculate_avrg_tensor_einsum(term.avrg_props, evv_molsys.mol_props, pol)

    on_the_fly = [func(idx, evv_molsys.mol_props) for idx in ALL_ABC]
    assert any(abs(v) > 1e-12 for v in on_the_fly)
    assert [tensor.at(idx) for idx in ALL_ABC] == pytest.approx(on_the_fly, rel=1e-12, abs=1e-18)
