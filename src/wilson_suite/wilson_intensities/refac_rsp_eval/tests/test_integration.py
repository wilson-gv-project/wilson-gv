from wilson_suite.wilson_derive.response_terms import VibPerturbedTerm
from wilson_suite.wilson_intensities.refac_rsp_eval.plan import (
    CompiledTerm,
    compile_terms,
)


def test_plan_integration():
    """Integration test for plan module."""

    terms = VibPerturbedTerm.load_many_from_json('./test_terms.json')
    for i, term in enumerate(terms):
        print(f"Term {i}: {term.to_latex()}\n\n")

    compiled_terms = compile_terms(terms=terms)


    assert isinstance(compiled_terms[0], CompiledTerm)

