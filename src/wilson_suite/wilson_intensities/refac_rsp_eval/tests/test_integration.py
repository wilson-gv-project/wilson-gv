import dataclasses
import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_intensities.amplitudes.averaging import (
    getGeneralPolarizationAveragingExpression,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    calculate_avrg_tensor_einsum,
    evaluate_full_index_dict,
    evaluate_term_coeff_sumover,
    make_avrg_einsum_subscripts,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.pipeline import prop_names
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    compile_terms,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.helpers import make_func_to_compute_avrg
from wilson_suite.wilson_intensities.refac_rsp_eval.tests.stored_terms import load_setup
from wilson_suite.wilson_system.system_data import (
    DataOriginInfo,
    MolecularProperty,
    MolPropsCollection,
    MolSystemData,
)
from wilson_suite.wilson_utils.paths import SUITE_ROOT
from wilson_suite.wilson_utils.prop_trivname import prop_trivname
from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer

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
    # main's VibDiffTerm.to_latex keeps the comma of an empty ket: 'a,' (debug_rendering drops it for harmonic factors: 'a')
    assert freqdenom_idx == sorted(['a,', 'b,', 'c,', 'a+b,c'])


def test_compiled_terms():
    """
    CompiledTerm for EVV
    """
    print('\n\n')

    u = {term.cmp_resmotf for term in cmpl_terms}
    print(f"unique resonance conditions: {len(u)}\n")
    for i in u:
        print(i)


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
    assert np.all(molsys.eigenvecs[0] == np.array([ 0.04, -0.,  0., -0.17,  0., -0.,  0.7, -0., -0., 0.7,  0.,  0.])) # pyright: ignore[reportOptionalSubscript]
    assert molsys.eigenvals == {0: 2878.687, 1: 1820.416, 2: 1534.549, 3: 1203.179, 4: 2933.526, 5: 1268.91}

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

    value, contribs = evaluate_full_index_dict(term, {'a': 0, 'b': 1, 'c': 1}, 
                                                 molsys_data=molsys,
                                                 avrg_tensor=avrg_tensor,
                                                 zero_tol=1e-18, states_choice='harmonic')
    print(value)
    print(contribs)

    import numpy as np

    from wilson_suite.wilson_utils.unit_convertor import convNu2Ene
    
    w = {m: convNu2Ene(e) for m, e in molsys.eigenvals.items()}  # harmonic omega, Eh

    assert contribs['NON_AVRG'] == molsys.mol_props['cff'].vals[0, 1, 1]        # cff[a, c, c]
    assert contribs['VIBDIFF_TERMS'] == pytest.approx(1 / w[0])                  # E_a - E_0
    assert contribs['VIBENE_DENOM'] == pytest.approx(1 / (w[0] * w[1] * w[1]))   # 1/(w_a w_b w_c)
    assert value == pytest.approx(-0.0625 * np.prod(list(contribs.values())))

    with pytest.raises(ValueError, match='term has indices that do not have values'):
        evaluate_full_index_dict(term, {'a': 0, 'b': 1}, molsys_data=molsys, avrg_tensor=avrg_tensor,
                                 states_choice='harmonic')

    value, contribs = evaluate_full_index_dict(term, {'a': 0, 'b': 1, 'c': 2}, 
                                                 molsys_data=molsys,
                                                 avrg_tensor=avrg_tensor,
                                                 zero_tol=1e-18, states_choice='harmonic')
    print(value)
    print(contribs)

    # term 2: resonance labels a, b are fixed; summation label c runs over every mode
    total, factors = evaluate_term_coeff_sumover(term, {'a': 0, 'b': 1},
                                                 molsys_data=molsys, polarization_linear_comb=polarization_linear_comb,
                                                 states_choice='harmonic')
    n_modes = len(molsys.eigenvals)
    by_hand = sum(evaluate_full_index_dict(term, {'a': 0, 'b': 1, 'c': c}, molsys_data=molsys, avrg_tensor=avrg_tensor,
                                           states_choice='harmonic')[0]
                  for c in range(n_modes))
    assert by_hand != 0.
    assert total == pytest.approx(by_hand)
    assert len(factors) == n_modes

    # b is a resonance label: summing over b would add peaks at different positions
    with pytest.raises(ValueError, match=r"resonance labels \['b'\] must be fixed"):
        evaluate_term_coeff_sumover(term, {'a': 0},
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


def test_term8_on_the_fly_average_is_the_einsum_contraction(term8):
    """
    Every index set: sum over (i, j, k, l) of C[i, j, k, l] * polgrad[b, i, l] * dipgrad[a, j] * dipgrad[c, k].
    Tolerance: rel 1e-12, abs 0, so the small values count too: they go from 1e-23 to 5e-6. Measured 2026-10-14:
    rel at most 1.6e-15.
    """
    term, molsys = term8
    assert [(tuple(o.o for o in p.ops), p.inds) for p in term.avrg_props] == [((0, 3), ['b']), ((1,), ['a']), ((2,), ['c'])]
    polgrad, dipgrad = molsys.mol_props['polgrad'].vals, molsys.mol_props['dipgrad'].vals
    C = np.zeros((3,) * 4)
    for key, coeff in ISO4.items():
        C[key] = coeff
    func = make_func_to_compute_avrg(avrg_expression=term.avrg_props, polarization_linear_comb=ISO4)

    for idx in ALL_ABC:
        expected = np.einsum('ijkl,il,j,k->', C, polgrad[idx['b']], dipgrad[idx['a']], dipgrad[idx['c']])
        assert func(idx, molsys.mol_props) == pytest.approx(expected, rel=1e-12, abs=0)


def test_term8_on_the_fly_average_does_not_change_when_the_molecule_is_rotated(term8):
    """
    Rotating every cartesian index of the real polgrad and dipgrad by the same R leaves the average unchanged.
    Tolerance: rel 1e-12; abs 1e-18, the evaluator's zero_tol (below it a value counts as 0). Measured 2026-10-14:
    72 of the 216 values are 6.4e-9 or more and agree to rel 2e-15; the other 144 are rounding noise, below 2.4e-19,
    and change with the rotation by up to 34 % of their size.
    """
    term, molsys = term8
    polgrad, dipgrad = molsys.mol_props['polgrad'].vals, molsys.mol_props['dipgrad'].vals
    ca, sa, cb, sb = np.cos(0.7), np.sin(0.7), np.cos(1.9), np.sin(1.9)
    R = np.array([[1., 0., 0.], [0., cb, -sb], [0., sb, cb]]) @ np.array([[ca, -sa, 0.], [sa, ca, 0.], [0., 0., 1.]])
    rotated = evv_props(np.einsum('ip,jq,npq->nij', R, R, polgrad), np.einsum('ip,np->ni', R, dipgrad))
    func = make_func_to_compute_avrg(avrg_expression=term.avrg_props, polarization_linear_comb=ISO4)

    values = [func(idx, molsys.mol_props) for idx in ALL_ABC]
    assert any(v != 0. for v in values)
    assert [func(idx, rotated) for idx in ALL_ABC] == pytest.approx(values, rel=1e-12, abs=1e-18)


## All 14 EVV terms: the einsum tensor = the on-the-fly loop, formaldehyde (todo.txt F1) ----------------
# Each term reads the tensor with its own labels, e.g. term 8: 'bIL,aJ,cK,IJKL->abc', read at {'a', 'b', 'c'}.
# Tolerance: rel 1e-12; abs 1e-18, the evaluator's zero_tol (below it a value counts as 0). Measured 2026-10-12:
# rel at most 7e-14 (a value of 6e-13 next to values of 5e-6); where the loop gives 0, einsum gives at most 4e-19.

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
