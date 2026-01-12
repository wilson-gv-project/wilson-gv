import pytest

def setup_evv_terms_for_test():
    """
    Helper function: Set up EVV-2D-IR terms for use in other test(s)
    """

    import wilson_suite as ws

    experiment_a = ws.fixtures.evv_experiment()
    fully_enhanced_terms = ws.derive.derive.get_fully_enhanced_terms(experiment_a)

    return fully_enhanced_terms


def test_dict_from_term():
    """
    Testing of dict_form_term
    """

    import wilson_suite as ws
    from fractions import Fraction

    my_terms = setup_evv_terms_for_test()

    reference_dict_a = {'termA_pref': Fraction(1, 4),
                        'termB_pref': 1.0,
                        'averaged_props': (('dipgrad', ('a',), ('B',)), ('dipgrad', ('b',), ('G',)), ('polhess', ('a', 'b'), ('A', 'D'))),
                        'non_averaged_props': None,
                        'vibene_denom': ('a', 'b'),
                        'vibenediff': None,
                        'resonances': (('zero,a', (-1,)), ('b,a', (-1, 2)))
                        }

    reference_dict_b = {'termA_pref': Fraction(-1, 16),
                        'termB_pref': 1.0,
                        'averaged_props': (('polgrad', ('b',), ('A', 'D')), ('dipgrad', ('a',), ('B',)), ('dipgrad', ('b',), ('G',))),
                        'non_averaged_props': (('cff', ('a', 'c', 'c')),),
                        'vibene_denom': ('a', 'b', 'c'),
                        'vibenediff': ('a+b,b',),
                        'resonances': (('zero,a', (-1,)), ('b,a', (-1, 2)))
                        }

    # Choosing a reasonably representative sample
    #my_terms[1][(1,0)][0].present()
    result_dict_a = ws.utils.termdict_from_symb_term.dict_from_term(my_terms[1][(1,0)][0])

    #my_terms[1][(0, 1)][5].present()
    result_dict_b = ws.utils.termdict_from_symb_term.dict_from_term(my_terms[1][(0, 1)][5])

    for i in reference_dict_a:
        try:
            assert reference_dict_a[i] == result_dict_a[i]
        except AssertionError:
            print(reference_dict_a[i], 'does not match', result_dict_a[i])

    for i in reference_dict_b:
        try:
            assert reference_dict_b[i] == result_dict_b[i]
        except AssertionError:
            print(reference_dict_b[i], 'does not match', result_dict_b[i])

    return