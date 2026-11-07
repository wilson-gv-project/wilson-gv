import dataclasses
import itertools
from typing import TYPE_CHECKING

import pytest

from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_intensities.amplitudes.averaging import (
    getGeneralPolarizationAveragingExpression,
)
from wilson_suite.wilson_response_evaluate.evaluate import (
    calculate_avrg_tensor_einsum,
    evaluate_full_index_dict,
    evaluate_term_coeff_sumover,
    make_avrg_einsum_subscripts,
)
from wilson_suite.wilson_response_evaluate.pipeline import prop_names
from wilson_suite.wilson_response_evaluate.plan import (
    CompiledTerm,
    compile_terms,
)
from wilson_suite.wilson_response_evaluate.tests.helpers import (
    make_func_to_compute_avrg,
)
from wilson_suite.wilson_response_evaluate.tests.stored_terms import load_setup
from wilson_suite.wilson_system.data_obtainer import wilson_data_obtainer
from wilson_suite.wilson_system.system_data import (
    DataOriginInfo,
    MolecularProperty,
    MolPropsCollection,
    MolSystemData,
)
from wilson_suite.wilson_utils.paths import SUITE_ROOT
from wilson_suite.wilson_utils.prop_trivname import prop_trivname

if TYPE_CHECKING:
    from wilson_suite.wilson_derive.abstractions import ResonanceCondition


def res_to_str(res: 'ResonanceCondition'):
    """
    ResonanceCondition to str for printing
    """
    upd_pf_sign = ['-'+ax if '-' not in ax else '+'+ax.strip('-') for ax in res.pf]
    pf_string = ''.join(upd_pf_sign)
    zip_res = f'w_[{res.diff.to_latex()}] {pf_string} -iG'
    return zip_res

def term_to_str(term: VibPerturbedTerm):
    """
    VibPerturbedTerm to str for printing
    """
    ft = ' '.join([f'w_[{i.to_latex()}]' for i in term.freqterms])
    res = ' / '.join([f'( {res_to_str(i)} )' for i in term.res])
    p_names = [(prop_trivname(ord_geo=len(i.inds), ord_el=len(i.ops)),i.inds, i.ops) for i in term.props] # type: ignore
    pp = ' '.join([f'{i[0]}[{",".join([f"{j.o}" for j in i[2]])};{",".join([f"{k}" for k in i[1]])}]' for i in p_names]) # type: ignore
    return f"{float(term.coeff)} * ( {pp} ) / ( {ft} ) / {res}"


# the 14 EVV terms, already translated to A = w1, B = -w1 + w2 (stored_terms.py)
translated_terms = load_setup('evv').terms
evv_terms = translated_terms    # prop_names reads only the props
cmpl_terms = compile_terms(terms=translated_terms)


def test_plan_compiled_term():
    """
    CompiledTerm for EVV
    """
    print('\n\n')

    assert isinstance(cmpl_terms[0], CompiledTerm)

    print(f"{term_to_str(translated_terms[0])}\n")
    assert cmpl_terms[0].max_state_lvl == 1
    assert cmpl_terms[0].frac_factor == -0.25

    print(f"{term_to_str(translated_terms[8])}\n")
    assert cmpl_terms[8].max_state_lvl == 2
    assert cmpl_terms[8].frac_factor == -0.125

    print(f"{term_to_str(translated_terms[13])}\n")
    assert cmpl_terms[13].max_state_lvl == 3
    assert cmpl_terms[13].frac_factor == 0.125

    with pytest.raises(dataclasses.FrozenInstanceError):
        cmpl_terms[8].frac_factor = 0.5 # type: ignore

    freqdenom_idx = sorted([freqt.to_latex() for freqt in cmpl_terms[8].cmp_freqdenom])
    # VibDiffTerm.to_latex keeps the comma of an empty ket: 'a,'
    assert freqdenom_idx == sorted(['a,', 'b,', 'c,', 'a+b,c'])


def test_eval_molsys_data():
    """
    getting data into MolPropsCollection (MolecularProperty instances hold values)
    """
    term0_el = cmpl_terms[0]

    # property collection from term0_el: one empty property per name (dipgrad appears twice in the term, once here)
    molprops = MolPropsCollection(properties=[MolecularProperty(trivial_name=n) for n in term0_el.prop_names])

    
    calc_dataorigin = DataOriginInfo(source_type='gaussian',
                                     base_file_loc=SUITE_ROOT+'/data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')
    # request dict: one entry per property name
    request = dict.fromkeys(term0_el.prop_names, calc_dataorigin)

    # base info about system, including vib states, normal modes and geometry (MolSystemData.load asks only for
    # the frequencies and the states: the evaluator reads no more)
    request.update(dict.fromkeys(['anharmonic_states', 'harmonic_states', 'nc_sqrt_eigval', 'normal_modes',
                                  'atoms', 'equilibrium_geometry'], calc_dataorigin))
    # obtainer is using DataOriginInfo to parse sources with CQCParse
    datadict = wilson_data_obtainer(requested_data_dict=request)

    assert set(request.keys()) == {'dipgrad', 'polhess',
                                   'anharmonic_states', 'harmonic_states',
                                   'nc_sqrt_eigval', 'normal_modes',
                                   'atoms', 'equilibrium_geometry'}
    assert set(datadict.keys()) == {'equilibrium_geometry', 'atoms', 'normal_modes',
                                    'anharmonic_states', 'harmonic_states',
                                    'nc_sqrt_eigval', 'dipgrad', 'polhess', 'reindex_modes'}
    assert list(datadict.keys()) != list(request.keys())

    # finally, putting data in -- requires empty MolPropsCollection and dict with data
    molsys = MolSystemData.from_datadict(mol_props=molprops, data_dict=datadict)
    print(molsys)

    import numpy as np
    # mode 0, one row per atom (O, C, H, H): dx/dQ in atomic units (m_e^(-1/2)), the printed mode (0.04294, -0.17408,
    # 0.69565, 0.69565 along x) scaled to sum_a m_a |d_a|^2 = 1 with m_a in electron masses
    np.testing.assert_allclose(molsys.eigenvecs[0], [[0.0008597016824121677, 0, 0], [-0.0034852554465372655, 0, 0],  # pyright: ignore[reportOptionalSubscript]
                                                     [0.013927607717047614, 0, 0], [0.013927607717047614, 0, 0]],
                               rtol=1e-12, atol=0)
    assert molsys.eigenvals == {0: 1203.179, 1: 1268.91, 2: 1534.549, 3: 1820.416, 4: 2878.687, 5: 2933.526}

    # with empty datadict: no frequencies and no states, so it does not fit (check_data_dict)
    with pytest.raises(ValueError, match=r"no 'nc_sqrt_eigval'.*\n.*no 'harmonic_states'"):
        MolSystemData.from_datadict(mol_props=molprops, data_dict={})


def test_evaluate_term():
    print()

    term = cmpl_terms[2]
    calc_dataorigin = DataOriginInfo(source_type='gaussian',
                                     base_file_loc=SUITE_ROOT+'/data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')
    molsys = MolSystemData.load(calc_dataorigin, term.prop_names)
    
    from wilson_suite.wilson_intensities.amplitudes.averaging import (
        getGeneralPolarizationAveragingExpression,
    )
    polarization_linear_comb = getGeneralPolarizationAveragingExpression(rank=4,laser_pol=(1.,1.,1.))
    avrg_tensor = calculate_avrg_tensor_einsum(term.avrg_props, molsys.mol_props, polarization_linear_comb)

    value, contribs = evaluate_full_index_dict(term, {'a': 4, 'b': 3, 'c': 3}, 
                                                 molsys_data=molsys,
                                                 avrg_tensor=avrg_tensor,
                                                 zero_tol=1e-18, states_choice='harmonic')
    print(value)
    print(contribs)

    import numpy as np

    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    
    w = {m: convNu2Ene(e) for m, e in molsys.eigenvals.items()}  # harmonic omega, Eh

    assert contribs['NON_AVRG'] == molsys.mol_props['cff'].vals[4, 3, 3]        # cff[a, c, c]
    assert contribs['VIBDIFF_TERMS'] == pytest.approx(1 / w[4])                  # E_a - E_0
    assert contribs['VIBENE_DENOM'] == pytest.approx(1 / (w[4] * w[3] * w[3]))   # 1/(w_a w_b w_c)
    assert value == pytest.approx(-0.0625 * np.prod(list(contribs.values())))

    with pytest.raises(ValueError, match='term has indices that do not have values'):
        evaluate_full_index_dict(term, {'a': 4, 'b': 3}, molsys_data=molsys, avrg_tensor=avrg_tensor,
                                 states_choice='harmonic')

    value, contribs = evaluate_full_index_dict(term, {'a': 4, 'b': 3, 'c': 2}, 
                                                 molsys_data=molsys,
                                                 avrg_tensor=avrg_tensor,
                                                 zero_tol=1e-18, states_choice='harmonic')
    print(value)
    print(contribs)

    # term 2: resonance labels a, b are fixed; summation label c runs over every mode
    total, factors = evaluate_term_coeff_sumover(term, {'a': 4, 'b': 3},
                                                 molsys_data=molsys, polarization_linear_comb=polarization_linear_comb,
                                                 states_choice='harmonic')
    n_modes = len(molsys.eigenvals)
    by_hand = sum(evaluate_full_index_dict(term, {'a': 4, 'b': 3, 'c': c}, molsys_data=molsys, avrg_tensor=avrg_tensor,
                                           states_choice='harmonic')[0]
                  for c in range(n_modes))
    assert by_hand != 0.
    assert total == pytest.approx(by_hand)
    assert len(factors) == n_modes

    # b is a resonance label: summing over b would add peaks at different positions
    with pytest.raises(ValueError, match=r"resonance labels \['b'\] must be fixed"):
        evaluate_term_coeff_sumover(term, {'a': 4},
                                    molsys_data=molsys, polarization_linear_comb=polarization_linear_comb,
                                    states_choice='harmonic')



## Term 8, averaged properties on the fly, with the formaldehyde data ---------------------
# Term 8 averages polgrad[b] (ops 0, 3) * dipgrad[a] (op 1) * dipgrad[c] (op 2) over four cartesian
# slots: for polarization key (i, j, k, l), polgrad reads (i, l), the first dipgrad j, the second k.

ISO4 = getGeneralPolarizationAveragingExpression(rank=4, laser_pol=(1., 1., 1.))
ALL_ABC = [dict(zip('abc', combo)) for combo in itertools.product(range(6), repeat=3)]


@pytest.fixture(scope='module')
def term8():
    """Term 8 and its MolSystemData from the formaldehyde file."""
    term = cmpl_terms[8]
    calc_dataorigin = DataOriginInfo(source_type='gaussian',
                                     base_file_loc=SUITE_ROOT+'/data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')
    return term, MolSystemData.load(calc_dataorigin, term.prop_names)


def evv_props(polgrad, dipgrad) -> MolPropsCollection:
    return MolPropsCollection([MolecularProperty(trivial_name='polgrad', vals=polgrad, extra_data={}),
                               MolecularProperty(trivial_name='dipgrad', vals=dipgrad, extra_data={})])


## All 14 EVV terms: the einsum tensor = the on-the-fly loop, formaldehyde -----------------------------
# Each term reads the tensor with its own labels, e.g. term 8: 'bIL,aJ,cK,IJKL->abc', read at {'a', 'b', 'c'}.
# Tolerance: rel 1e-12; abs 1e-18, the evaluator's zero_tol (below it a value counts as 0): where the loop gives 0,
# einsum gives rounding noise.

EVV_SUBSCRIPTS = ['abIL,aJ,bK,IJKL->ab', 'bIL,aJ,abK,IJKL->ab', 'bIL,aJ,bK,IJKL->ab', 'cIL,aJ,bK,IJKL->abc',
                  'aIL,aJ,bK,IJKL->ab', 'aIL,aJ,bK,IJKL->ab', 'cIL,aJ,bK,IJKL->abc', 'bIL,aJ,bK,IJKL->ab',
                  'bIL,aJ,cK,IJKL->abc', 'bIL,aJ,bK,IJKL->ab', 'bIL,aJ,aK,IJKL->ab', 'bIL,aJ,aK,IJKL->ab',
                  'bIL,aJ,bK,IJKL->ab', 'bIL,aJ,cK,IJKL->abc']


@pytest.fixture(scope='module')
def evv_molsys() -> MolSystemData:
    """all properties of the 14 terms; the states do not enter the average"""
    origin = DataOriginInfo(source_type='gaussian', base_file_loc=SUITE_ROOT+'/data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')
    return MolSystemData.load(origin, prop_names(evv_terms))


def test_evv_einsum_subscripts():
    """
    term 0:  polhess[a, b] (slots 0, 3) dipgrad[a] (1) dipgrad[b] (2)      'abIL,aJ,bK,IJKL->ab'
    term 1:  polgrad[b] (0, 3) dipgrad[a] (1) diphess[a, b] (2)            'bIL,aJ,abK,IJKL->ab'
    term 8:  polgrad[b] (0, 3) dipgrad[a] (1) dipgrad[c] (2)               'bIL,aJ,cK,IJKL->abc'
    """
    assert [make_avrg_einsum_subscripts(t.avrg_props) for t in cmpl_terms] == EVV_SUBSCRIPTS


@pytest.mark.parametrize('laser_pol', [(1., 1., 1.), (1., 0., 0.), (0.3, -1.2, 2.)])
@pytest.mark.parametrize('term_id', range(14))
def test_evv_einsum_tensor_is_the_on_the_fly_average_for_every_index_set(term_id, laser_pol, evv_molsys):
    term = cmpl_terms[term_id]
    pol = getGeneralPolarizationAveragingExpression(rank=4, laser_pol=laser_pol)
    func = make_func_to_compute_avrg(avrg_expression=term.avrg_props, polarization_linear_comb=pol)

    tensor = calculate_avrg_tensor_einsum(term.avrg_props, evv_molsys.mol_props, pol)

    on_the_fly = [func(idx, evv_molsys.mol_props) for idx in ALL_ABC]
    assert any(abs(v) > 1e-12 for v in on_the_fly)
    assert [tensor.at(idx) for idx in ALL_ABC] == pytest.approx(on_the_fly, rel=1e-12, abs=1e-18)
