from ...amplitudes import precalculations as prec
import wilson_suite.wilson_intensities.amplitudes.pre_eval_treatment as pet
import wilson_suite.wilson_derive.abstractions as wd_abst
import numpy as np


polhess = np.zeros((4, 4, 3, 3))

polhess[0, 0, 0, 0] = 0.3
polhess[0, 2, 1, 1] = 0.3
polhess[0, 1, 0, 0] = 0.3
polhess[2, 3, 1, 1] = 0.15
polhess[2, 1, 2, 0] = 0.15
polhess[1, 3, 2, 0] = 0.3
polhess[1, 3, 2, 1] = 0.3
polhess[0, 3, 1, 2] = 0.3
polhess[1, 0, 2, 1] = 0.15
polhess[2, 0, 2, 1] = 0.3
polhess[2, 2, 2, 1] = 0.3
polhess[3, 3, 2, 1] = 0.3
polhess[2, 1, 2, 2] = 0.3
polhess[0, 2, 2, 2] = 0.3

for i in range(4):
    for j in range(i + 1, 4):
        polhess[j, i] = polhess[i, j]

props_data = {'dipgrad': np.array([[0.3 , 0.0 , 0.0],
                                    [0.15, 0.0 , 0.0],
                                    [0.0 , 0.3 , 0.0],
                                    [0.3 , 0.15, 0.0]]), 
                    
                    'diphess': np.array([[[0. , 0.15, 0.15 ],
                                        [0. , 0.15, 0.15 ],
                                        [0.3, 0.3 , 0.15 ],
                                        [0.3, 0.  , 0.15 ]],

                                        [[0.  , 0.15, 0.15 ],
                                        [0.  , 0.15, 0.15 ],
                                        [0.15, 0.3 , 0.15 ],
                                        [0.  , 0.15, 0.15 ]],

                                        [[0.3 , 0.3 , 0.15 ],
                                        [0.15, 0.3 , 0.15 ],
                                        [0.  , 0.15, 0.  ],
                                        [0.15, 0.15, 0.  ]],

                                        [[0.3 , 0.  , 0.15 ],
                                        [0.  , 0.15, 0.15 ],
                                        [0.15, 0.15, 0.  ],
                                        [0.  , 0.3 , 0.  ]]]), 

                    'polgrad': np.array([[[0.0, 0.3, 0.0],
                                        [0.3, 0.0, 0.0],
                                        [0.0, 0.0, 0.3]],

                                        [[0.3, 0.0 , 0.0],
                                        [0.0, 0.15, 0.0],
                                        [0.0, 0.0 , 0.3]],

                                        [[0.0, 0.15, 0.15],
                                        [0.3, 0.0 , 0.0 ],
                                        [0.3, 0.0 , 0.0 ]],

                                        [[0.0 , 0.0, 0.3 ],
                                        [0.0 , 0.3, 0.0 ],
                                        [0.15, 0.0, 0.0 ]]]), 

                    'polhess': polhess}


def test_make_func_to_compute_avrg():
    print()
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)
    t_inds = [0, 1, -2, -1]
    # t_inds = range(len(terms_fuller_flat))
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]
    
    motifs_coll = []

    for t in terms_select:
        motifs_coll.append(pet.PropsCollection(props=t.props).get_averaged_props())
    
    for i in motifs_coll:
        print(i)

    # print('--------')
    for i in motifs_coll[:2]:
        # print(i)
        # index_choices={'a': 1, 'b': 2}
        f01 = prec.make_func_to_compute_avrg(avrg_expression=i, 
                                             polarization='ZZZZ')
        # print('f01 res1', f01(index_choices={'a': 1, 'b': 2}, props_data=props_data))
        # print('f01 res2', f01(index_choices={'a': 1, 'b': 0}, props_data=props_data))
    
    print('--------')
    funcs3dtensors = []
    for i in motifs_coll[2:]:
        # print(i)
        # index_choices={'a': 1, 'b': 0, 'c': 1}
        f02 = prec.make_func_to_compute_avrg(avrg_expression=i,
                                             polarization='ZZZZ')
        funcs3dtensors.append(f02)

        # print('f02 res1', f02(index_choices={'a': 1, 'b': 0, 'c': 1}, props_data=props_data))
        # print('f02 res2', f02(index_choices={'a': 3, 'b': 1, 'c': 1}, props_data=props_data))
    
    shortlist = motifs_coll[2:]
    for i, f in enumerate(funcs3dtensors):
        print('func', i, shortlist[i])
        res_f = f(index_choices={'a': 3, 'b': 1, 'c': 1}, props_data=props_data)
        print("{'a': 3, 'b': 1, 'c': 1}", res_f)
        res_f1 = f(index_choices={'a': 3, 'b': 0, 'c': 1}, props_data=props_data)
        print("{'a': 3, 'b': 0, 'c': 1}", res_f1)
        res_f2 = f(index_choices={'a': 3, 'b': 0, 'c': 0}, props_data=props_data)
        print("{'a': 3, 'b': 0, 'c': 0}", res_f2)
        res_f3 = f(index_choices={'a': 3, 'b': 0, 'c': 3}, props_data=props_data)
        print("{'a': 3, 'b': 0, 'c': 3}", res_f3)

        res_f4 = f(index_choices={'a': 1, 'b': 0, 'c': 1}, props_data=props_data)
        print("{'a': 1, 'b': 0, 'c': 1}", res_f4)
        res_f5 = f(index_choices={'a': 1, 'b': 0, 'c': 0}, props_data=props_data)
        print("{'a': 1, 'b': 0, 'c': 0}", res_f5)
        res_f6 = f(index_choices={'a': 1, 'b': 0, 'c': 3}, props_data=props_data)
        print("{'a': 1, 'b': 0, 'c': 3}", res_f6)
        res_f7 = f(index_choices={'a': 2, 'b': 0, 'c': 0}, props_data=props_data)
        print("{'a': 2, 'b': 0, 'c': 0}", res_f7)

def test_precalculate_avrg_tensor():
    # polgrad['b'][0, 3] * dipgrad['a'][1] * dipgrad['b'][2]
    polhess = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=2)
    polhess.inds = ['a', 'b']
    dipgrad1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    dipgrad1.inds = ['a']
    dipgrad2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    dipgrad2.inds = ['b']

    avrg_expr = pet.PropsCollection(props=[polhess, dipgrad1, dipgrad2])

    t = prec.precalculate_avrg_tensor(avrg_expression=avrg_expr, polarization='ZZZZ', number_of_nmodes=4, props_data=props_data)
    print(t)

    # polgrad['b'][0, 3] * dipgrad['a'][1] * dipgrad['c'][2]
    polhess = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=2)
    polhess.inds = ['a', 'b']
    dipgrad1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    dipgrad1.inds = ['a']
    dipgrad2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    dipgrad2.inds = ['c']

    avrg_expr = pet.PropsCollection(props=[polhess, dipgrad1, dipgrad2])
    t = prec.precalculate_avrg_tensor(avrg_expression=avrg_expr, polarization='ZZZZ', number_of_nmodes=4, props_data=props_data)
    print(t)