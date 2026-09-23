import dataclasses
from typing import TYPE_CHECKING

import pytest

from wilson_suite.wilson_derive import term_var_translate
from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    DataOriginInfo,
    MolecularProperty,
    MolPropsCollection,
    MolSystemData,
    _sys_info_request,
    evaluate_single_index_dict,
    evaluate_term_coeffs,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    compile_terms,
)
from wilson_suite.wilson_utils.builders import make_SpectralAxisSet
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


# loading EVV terms from json file
evv_terms = VibPerturbedTerm.load_many_from_json(SUITE_ROOT+'/wilson_intensities/refac_rsp_eval/tests/test_terms.json')

axis_choice = make_SpectralAxisSet({'A': [1], 'B': [-1, 2]}) # type: ignore
translated_terms = term_var_translate.translate_terms_to_axis_variables(evv_terms, axis_choice)
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
    assert freqdenom_idx == sorted(['a', 'b', 'c', 'a+b,c'])


def test_eval_molsys_data():
    """
    getting data into MolPropsCollection (MolecularProperty instances hold values)
    """
    term0_el = cmpl_terms[0]

    # property collection from term0_el
    molprops = MolPropsCollection(properties=[MolecularProperty.from_polprop(i) for i in term0_el.cmp_props])
    
    calc_dataorigin = DataOriginInfo(source_type='gaussian',
                                     base_file_loc=SUITE_ROOT+'/data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')
    # request dict is built from PropsCollection
    request = term0_el.cmp_props.build_request_dict(calc_setup=calc_dataorigin)

    # base info about system, including vib states
    request.update(_sys_info_request(calc_dataorigin))
    # obtainer is using DataOriginInfo to parse sources with CQCParse
    datadict = wilson_data_obtainer(requested_data_dict=request)

    assert list(request.keys()) == ['dipgrad', 'polhess', 
                                    'anharmonic_states', 'harmonic_states', 
                                    'nc_sqrt_eigval', 'normal_modes', 
                                    'atoms', 'equilibrium_geometry']
    assert list(datadict.keys()) == ['equilibrium_geometry', 'atoms', 'normal_modes', 
                                     'anharmonic_states', 'harmonic_states', 
                                     'nc_sqrt_eigval', 'dipgrad', 'polhess', 'reindex_modes']
    assert list(datadict.keys()) != list(request.keys())

    # finally, putting data in -- requires empty MolPropsCollection and dict with data
    # also, resets values, because props in collection shold be from the same source
    molsys = MolSystemData.from_datadict(mol_props=molprops, data_dict=datadict)
    print(molsys)

    import numpy as np
    assert np.all(molsys.eigenvecs[0] == np.array([ 0.04, -0.,  0., -0.17,  0., -0.,  0.7, -0., -0., 0.7,  0.,  0.])) # pyright: ignore[reportOptionalSubscript]
    assert molsys.eigenvals == {0: 2878.687, 1: 1820.416, 2: 1534.549, 3: 1203.179, 4: 2933.526, 5: 1268.91}

    # with empty datadict
    molsys = MolSystemData.from_datadict(mol_props=molprops, data_dict={})
    assert molsys.eigenvals is None
    assert molsys.eigenvecs is None
    assert molsys.natoms == 0
    assert molsys.geo is None
    assert molsys.linear == False
    assert molsys.data_origin is None
    assert molsys.data_filled == False

    assert molsys.mol_props.is_filled == False
    assert len(molsys.mol_props) == 3

    assert molsys.states.harmonic_osc_states_labels == ()
    assert molsys.states.number_of_nmodes == 0


def test_evaluate_term():
    print()

    term = cmpl_terms[2]
    molprops = MolPropsCollection(properties=[MolecularProperty.from_polprop(i) for i in term.cmp_props])
    calc_dataorigin = DataOriginInfo(source_type='gaussian',
                                     base_file_loc=SUITE_ROOT+'/data_for_tests/g16_formaldehyde_B3LYPcc_pVQZ.out')
    request = term.cmp_props.build_request_dict(calc_setup=calc_dataorigin)
    request.update(_sys_info_request(calc_dataorigin))
    datadict = wilson_data_obtainer(requested_data_dict=request)

    molsys = MolSystemData.from_datadict(mol_props=molprops, data_dict=datadict)

    value, contribs = evaluate_single_index_dict(term, {'a': 0, 'b': 1, 'c': 1}, 
                                                 molsys_data=molsys,
                                                 pol_prop_vec=(1.,1.,1.),
                                                 precalculated_data=None, 
                                                 zero_tol=1e-18)
    print(value)
    print(contribs)


    with pytest.raises(ValueError) as e:
        evaluate_single_index_dict(term, {'a': 0, 'b': 1}, 
                                                 molsys_data=molsys,
                                                 pol_prop_vec=(1.,1.,1.),
                                                 precalculated_data=None, 
                                                 zero_tol=1e-18)
        assert e.value == 'term has indices that do not have values in index_dict.'

    value, contribs = evaluate_single_index_dict(term, {'a': 0, 'b': 1, 'c': 2}, 
                                                 molsys_data=molsys,
                                                 pol_prop_vec=(1.,1.,1.),
                                                 precalculated_data=None, 
                                                 zero_tol=1e-18)
    print(value)
    print(contribs)

    results = evaluate_term_coeffs(term, [{'a': 0}], precalculated_data=None, 
                                   molsys_data=molsys, pol_prop_vec=(1.,1.,1.))
    print(results)
