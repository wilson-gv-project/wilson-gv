from wilson_suite.wilson_derive import term_var_translate
from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_intensities.refac_rsp_eval.evaluate import (
    MolPropsCollection,
    MolSystemData,
)
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    compile_terms,
)
from wilson_suite.wilson_utils.builders import make_SpectralAxisSet
from wilson_suite.wilson_utils.prop_trivname import prop_trivname
from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer


def res_to_str(res):
    upd_pf_sign = ['-'+ax if '-' not in ax else '+'+ax.strip('-') for ax in res.pf]
    pf_string = ''.join(upd_pf_sign)
    zip_res = f'w_[{res.diff.to_latex()}] {pf_string} -iG'
    return zip_res

def term_to_str(term):
    ft = ' '.join([f'w_[{i.to_latex()}]' for i in term.freqterms])
    res = ' / '.join([f'( {res_to_str(i)} )' for i in term.res])
    p_names = [(prop_trivname(ord_geo=len(i.inds), ord_el=len(i.ops)),i.inds, i.ops) for i in term.props] # type: ignore
    pp = ' '.join([f'{i[0]}[{",".join([f"{j.o}" for j in i[2]])};{",".join([f"{k}" for k in i[1]])}]' for i in p_names]) # type: ignore
    return f"{float(term.coeff)} * ( {pp} ) / ( {ft} ) / {res}"

def compl_evv_terms():
    terms = VibPerturbedTerm.load_many_from_json('./test_terms.json')

    axis_choice = make_SpectralAxisSet({'A': [1], 'B': [-1, 2]}) # type: ignore
    translated_terms = term_var_translate.translate_terms_to_axis_variables(terms, axis_choice)

    for i, term in enumerate(translated_terms):
        print(f"Term {i}:")
        print(f"{term_to_str(term)}\n")

    return compile_terms(terms=translated_terms)


def test_plan_compiled_term():
    """Integration test for plan module."""
    print('\n\n')
    compiled_terms = compl_evv_terms()

    assert isinstance(compiled_terms[0], CompiledTerm)
    assert compiled_terms[0].max_state_lvl == 1
    assert compiled_terms[8].max_state_lvl == 2
    assert compiled_terms[13].max_state_lvl == 3



def test_eval_molsys_data():
    molprops = MolPropsCollection()

    compiled_terms = compl_evv_terms()
    print(compiled_terms[0].cmp_props.build_request_dict())

    # request = {}
    # datadict = wilson_data_obtainer(requested_data_dict=request)
    # molsys = MolSystemData.from_datadict(data_dict=datadict)

