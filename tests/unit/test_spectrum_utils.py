from wilson.utils.spectrum_utils import get_allparts_indices, make_abc_tuple

def test_get_indices() -> None:
    print()
    term = {'resonances': (('a+b,a', (-1, 2)), ('zero,a', (-1,))),
              'vibenediff': ('a+b+c,zero', 'c,a+b'),
              'averaged_props': (('mu_Q', ('a',), ('B',)), ('alpha_Q', ('b',), ('A', 'D')), ('mu_Q', ('c',), ('G',))),
              'non_averaged_props': (('F', ('a', 'b', 'c',)),),
              'vibene_denom': ('a','b','c'),
              'termB_pref': 1.,
              'termA_pref': -1/48.}

    print(term)
    allidx, res_idx = get_allparts_indices(term)
    assert allidx == 3
    assert res_idx == 2


def test_make_abc_tuple() -> None:
    print()
    a = make_abc_tuple((1, 2), 4)
    print(a)
    assert a == (1, 2, None, None)


def test_abc_list() -> None:
    print()
    from wilson.utils.spectrum_utils import make_abc_dict
    abc_comb = (1, 2, 3, 4)
    idx_str = make_abc_dict(abc_comb)
    print(idx_str)