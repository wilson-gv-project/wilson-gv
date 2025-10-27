import wilson_suite.wilson_intensities.amplitudes.averaged_props
import wilson_suite.wilson_intensities.amplitudes.term_parts as tparts
import wilson_suite.wilson_intensities.amplitudes.averaged_props as avrgprops

import logging
from ....wilson_utils.logger import setup_logger
setup_logger("wilson", level=logging.DEBUG)

def get_expressions():
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)
    
    t_inds = [0, 1,-1, -2, -3]
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]
    # terms_select = terms_fuller_flat

    return [tparts.PropsCollection(t.props) for t in terms_select]

def test_expr1():
    expression = get_expressions()[0]
    print()
    for prop in expression:
        print(prop)
    nm_indices_symb = sorted(set(expression.get_mode_indices()))
    
    from ...amplitudes.utils import generate_index_choices_general
    idxs = generate_index_choices_general(indlabels_in_motif=nm_indices_symb, labels=['1', '2', '3'])
    print()
    for i in idxs:
        print(i)


def test_identify_unique_avrgmotifs():
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)

    t_inds = [0, 1,-1, -2, -3]
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]
    terms_select = terms_fuller_flat

    unique = wilson_suite.wilson_intensities.amplitudes.averaged_props.identify_unique_avrgmotifs(terms_select)
    print('\n\n')
    for i in unique:
        print(i)
    print(len(unique))


def test_make_avrg_props_motif():
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)

    collect_simple = []
    t_inds = [0, -2]
    for tID in t_inds:
        term = terms_fuller_flat[tID]

        # get only avrg props
        # props_with_cart_axes = [prop.to_latex() for prop in term.props if prop.ops]
        props_with_cax_simple = [wilson_suite.wilson_intensities.amplitudes.averaged_props.simple_prop_ID(prop) for prop in term.props if prop.ops]

        collect_simple.append(set(tuple(props_with_cax_simple)))
        pp = wilson_suite.wilson_intensities.amplitudes.averaged_props.make_avrg_props_motif(term.props)
        print(props_with_cax_simple)
        print(pp)

import wilson_suite.wilson_derive.abstractions as wd_abst
import wilson_suite.wilson_intensities.amplitudes.term_parts as term_abst
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
        motifs_coll.append(term_abst.PropsCollection(props=t.props).get_averaged_props())
    
    for i in motifs_coll:
        print(i)

    # print('--------')
    for i in motifs_coll[:2]:
        # print(i)
        # index_choices={'a': 1, 'b': 2}
        f01 = avrgprops.make_func_to_compute_avrg(avrg_expression=i, 
                                             polarization='ZZZZ')
        # print('f01 res1', f01(index_choices={'a': 1, 'b': 2}, props_data=props_data))
        # print('f01 res2', f01(index_choices={'a': 1, 'b': 0}, props_data=props_data))
    
    print('--------')
    funcs3dtensors = []
    for i in motifs_coll[2:]:
        # print(i)
        # index_choices={'a': 1, 'b': 0, 'c': 1}
        f02 = avrgprops.make_func_to_compute_avrg(avrg_expression=i,
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

    avrg_expr = term_abst.PropsCollection(props=[polhess, dipgrad1, dipgrad2])

    # t1 = avrgprops.precalculate_avrg_tensor(avrg_expression=avrg_expr, polarization='ZZZZ', number_of_nmodes=4, props_data=props_data)
    # print(t1)

    # polgrad['b'][0, 3] * dipgrad['b'][1] * dipgrad['a'][2]
    polhess = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=2)
    polhess.inds = ['a', 'b']
    dipgrad1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    dipgrad1.inds = ['b']
    dipgrad2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    dipgrad2.inds = ['a']

    avrg_expr = term_abst.PropsCollection(props=[polhess, dipgrad1, dipgrad2])

    # t2 = avrgprops.precalculate_avrg_tensor(avrg_expression=avrg_expr, polarization='ZZZZ', number_of_nmodes=4, props_data=props_data)
    # print(t2)

    # polgrad['b'][0, 3] * dipgrad['a'][1] * dipgrad['c'][2]
    polhess = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=2)
    polhess.inds = ['a', 'b']
    dipgrad1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    dipgrad1.inds = ['a']
    dipgrad2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    dipgrad2.inds = ['c']

    avrg_expr = term_abst.PropsCollection(props=[polhess, dipgrad1, dipgrad2])
    t3 = avrgprops.precalculate_avrg_tensor(avrg_expression=avrg_expr, polarization='ZZZZ', number_of_nmodes=4, props_data=props_data)
    print(t3)

    # polgrad['b'][0, 3] * dipgrad['a'][1] * dipgrad['c'][2]
    polhess = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=2)
    polhess.inds = ['a', 'b']
    dipgrad1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    dipgrad1.inds = ['c']
    dipgrad2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    dipgrad2.inds = ['a']

    avrg_expr = term_abst.PropsCollection(props=[polhess, dipgrad1, dipgrad2])
    t4 = avrgprops.precalculate_avrg_tensor(avrg_expression=avrg_expr, polarization='ZZZZ', number_of_nmodes=4, props_data=props_data)
    print(t4)
    print(np.allclose(t3, t4))

    # polgrad['b'][0, 3] * dipgrad['a'][1] * dipgrad['c'][2]
    polgrad = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=1)
    polgrad.inds = ['b']
    dipgrad1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    dipgrad1.inds = ['c']
    dipgrad2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    dipgrad2.inds = ['a']

    avrg_expr = term_abst.PropsCollection(props=[polgrad, dipgrad1, dipgrad2])
    t5 = avrgprops.precalculate_avrg_tensor(avrg_expression=avrg_expr, polarization='ZZZZ', number_of_nmodes=4, props_data=props_data)
    print(t5)

    # polgrad['b'][0, 3] * dipgrad['a'][1] * dipgrad['c'][2]
    polgrad = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=1)
    polgrad.inds = ['c']
    dipgrad1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    dipgrad1.inds = ['b']
    dipgrad2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    dipgrad2.inds = ['a']

    avrg_expr = term_abst.PropsCollection(props=[polgrad, dipgrad1, dipgrad2])
    t6 = avrgprops.precalculate_avrg_tensor(avrg_expression=avrg_expr, polarization='ZZZZ', 
                                            number_of_nmodes=4, props_data=props_data)
    print(t6)

    print(np.allclose(t5, t6))

    # polgrad['b'][0, 3] * dipgrad['a'][1] * dipgrad['c'][2]
    polgrad = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=1)
    polgrad.inds = ['b']
    dipgrad1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    dipgrad1.inds = ['a']
    dipgrad2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    dipgrad2.inds = ['a']

    avrg_expr = term_abst.PropsCollection(props=[polgrad, dipgrad1, dipgrad2])
    t7 = avrgprops.precalculate_avrg_tensor(avrg_expression=avrg_expr, polarization='ZZZZ', number_of_nmodes=4, props_data=props_data)
    print(t7)

def generate_props_data4modes():
    return {'dipgrad': np.zeros((4, 3)), 
            'diphess': np.zeros((4, 4, 3)),
            'polgrad': np.zeros((4, 3, 3)), 
            'polhess': np.zeros((4, 4, 3, 3))}

def test_precalculate_avrg_tensor_focused():
    print()
    props_data = generate_props_data4modes()
    
    # cart axes (0, 1, 1, 0) - 0 1 2 3
    # nm modes (a=0, b=1, c=2)
    # props_data['dipgrad'][0, 1] = 0.15

    # props_data['dipgrad'][2, 1] = 0.15
    # props_data['polgrad'][1, 0, 0] = 0.3
    props_data['dipgrad'][0, 1] = 0.15
    props_data['dipgrad'][1, 1] = 0.3
    props_data['polgrad'][0, 0, 0] = 0.3

    polgrad = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=1)
    polgrad.inds = ['b']
    dipgrad1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    dipgrad1.inds = ['c']
    dipgrad2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    dipgrad2.inds = ['a']

    avrg_expr1 = term_abst.PropsCollection(props=[polgrad, dipgrad1, dipgrad2])
    t5 = avrgprops.precalculate_avrg_tensor(avrg_expression=avrg_expr1, polarization='ZZZZ', 
                                            number_of_nmodes=4, props_data=props_data)
    print(avrg_expr1)
    print(np.transpose(np.nonzero(t5)))
    # print(t5.shape)

    # polgrad['b'][0, 3] * dipgrad['a'][1] * dipgrad['c'][2]
    polgrad = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=1)
    polgrad.inds = ['c']
    dipgrad1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    dipgrad1.inds = ['b']
    dipgrad2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    dipgrad2.inds = ['a']

    avrg_expr2 = term_abst.PropsCollection(props=[polgrad, dipgrad1, dipgrad2])
    t6 = avrgprops.precalculate_avrg_tensor(avrg_expression=avrg_expr2, polarization='ZZZZ', 
                                            number_of_nmodes=4, props_data=props_data)
    # print(t6)
    print(avrg_expr2)
    print(np.transpose(np.nonzero(t6)))

    # cart axes (0, 1, 1, 0) - 0 1 2 3
    props_data = generate_props_data4modes()
    props_data['dipgrad'][0, 1] = 0.15
    props_data['dipgrad'][1, 1] = 0.3
    props_data['polgrad'][0, 0, 0] = 0.3

    polgrad = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=1)
    polgrad.inds = ['a']
    dipgrad1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    dipgrad1.inds = ['b']
    dipgrad2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    dipgrad2.inds = ['a']

    avrg_expr3 = term_abst.PropsCollection(props=[polgrad, dipgrad1, dipgrad2])
    t7 = avrgprops.precalculate_avrg_tensor(avrg_expression=avrg_expr3, polarization='ZZZZ', 
                                            number_of_nmodes=4, props_data=props_data)
    # print(t7)
    print(avrg_expr3)
    print(np.transpose(np.nonzero(t7)))

    num_pulses = len(avrg_expr1.get_cart_axes())
    polarization = 'ZZZZ'

    from wilson_suite.wilson_intensities.amplitudes.averaging import getPolarizationAveragingExpression
    polarization_avrg_terms, prefactor = getPolarizationAveragingExpression(num_pulses=num_pulses, polarization=polarization)
    # print('polarization_avrg_terms', polarization_avrg_terms)
    # print('prefactor', prefactor)
    # print('set polarization_avrg_terms\n', set([tuple(i) for i in polarization_avrg_terms]))
    # print(len(polarization_avrg_terms))
    # print(len(set([tuple(i) for i in polarization_avrg_terms])))