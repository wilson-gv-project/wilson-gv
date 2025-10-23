from ...spectrum import precalculations as prec
import wilson_suite.wilson_intensities.spectrum.pre_eval_treatment as pet
import numpy as np

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
    
    print('--------')
    for i in motifs_coll[:2]:
        print(i)
        # index_choices={'a': 1, 'b': 2}
        f01 = prec.make_func_to_compute_avrg(avrg_expression=i, 
                                             polarization='ZZZZ')
        print('f01 res1', f01(index_choices={'a': 1, 'b': 2}, props_data=props_data))
        print('f01 res2', f01(index_choices={'a': 1, 'b': 0}, props_data=props_data))
    
    print('--------')
    for i in motifs_coll[2:]:
        print(i)
        # index_choices={'a': 1, 'b': 0, 'c': 1}
        f02 = prec.make_func_to_compute_avrg(avrg_expression=i,
                                             polarization='ZZZZ')
        print('f02 res1', f02(index_choices={'a': 1, 'b': 0, 'c': 1}, props_data=props_data))
        print('f02 res2', f02(index_choices={'a': 3, 'b': 1, 'c': 1}, props_data=props_data))
        