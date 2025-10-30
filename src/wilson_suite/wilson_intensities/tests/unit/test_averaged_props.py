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
        f01 = avrgprops.make_func_to_compute_avrg(avrg_expression=i, polarization='ZZZZ')
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
    t3 = avrgprops.calculate_avrg_tensor(avrg_expression=avrg_expr, polarization='ZZZZ', number_of_nmodes=4, props_data=props_data)
    print(t3)

    # polgrad['b'][0, 3] * dipgrad['a'][1] * dipgrad['c'][2]
    polhess = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=2)
    polhess.inds = ['a', 'b']
    dipgrad1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    dipgrad1.inds = ['c']
    dipgrad2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    dipgrad2.inds = ['a']

    avrg_expr = term_abst.PropsCollection(props=[polhess, dipgrad1, dipgrad2])
    t4 = avrgprops.calculate_avrg_tensor(avrg_expression=avrg_expr, polarization='ZZZZ', number_of_nmodes=4, props_data=props_data)
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
    t5 = avrgprops.calculate_avrg_tensor(avrg_expression=avrg_expr, polarization='ZZZZ', number_of_nmodes=4, props_data=props_data)
    print(t5)

    # polgrad['b'][0, 3] * dipgrad['a'][1] * dipgrad['c'][2]
    polgrad = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=1)
    polgrad.inds = ['c']
    dipgrad1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
    dipgrad1.inds = ['b']
    dipgrad2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
    dipgrad2.inds = ['a']

    avrg_expr = term_abst.PropsCollection(props=[polgrad, dipgrad1, dipgrad2])
    t6 = avrgprops.calculate_avrg_tensor(avrg_expression=avrg_expr, polarization='ZZZZ', 
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
    t7 = avrgprops.calculate_avrg_tensor(avrg_expression=avrg_expr, polarization='ZZZZ', number_of_nmodes=4, props_data=props_data)
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
    t5 = avrgprops.calculate_avrg_tensor(avrg_expression=avrg_expr1, polarization='ZZZZ', 
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
    t6 = avrgprops.calculate_avrg_tensor(avrg_expression=avrg_expr2, polarization='ZZZZ', 
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
    t7 = avrgprops.calculate_avrg_tensor(avrg_expression=avrg_expr3, polarization='ZZZZ', 
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


polhess_ab03 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=2)
polhess_ab03.inds = ['a', 'b']
dipgrad_a0 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0)], dord=1)
dipgrad_a0.inds = ['a']
dipgrad_b1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
dipgrad_b1.inds = ['b']
dipgrad_c2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
dipgrad_c2.inds = ['c']
dipgrad_d3 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=3)], dord=1)
dipgrad_d3.inds = ['d']
dipgrad_d4 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=4)], dord=1)
dipgrad_d4.inds = ['d']

dipgrad_b5 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=5)], dord=1)
dipgrad_b5.inds = ['b']

avrg_expr_00 = term_abst.PropsCollection(props=[dipgrad_a0, dipgrad_b1, dipgrad_c2, dipgrad_d3, dipgrad_d4])

dipgrad_a0 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0)], dord=1)
dipgrad_a0.inds = ['a']
dipgrad_a1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
dipgrad_a1.inds = ['a']
dipgrad_b2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
dipgrad_b2.inds = ['b']
dipgrad_c3 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=3)], dord=1)
dipgrad_c3.inds = ['c']

avrg_expr_01 = term_abst.PropsCollection(props=[dipgrad_a0, dipgrad_a1, dipgrad_b2, dipgrad_c3, dipgrad_d4])

avrg_expr_02 = term_abst.PropsCollection(props=[polhess_ab03, dipgrad_a1, dipgrad_b2, dipgrad_d4])
# print('avrg_expr_02', avrg_expr_02)

shyp_dd024 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=2), wd_abst.QOperator(o=4)], dord=2)
shyp_dd024.inds = ['d', 'd']

shyp_ab024 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=2), wd_abst.QOperator(o=4)], dord=2)
shyp_ab024.inds = ['a', 'b']
dipgrad_d1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
dipgrad_d1.inds = ['d']

avrg_expr_03 = term_abst.PropsCollection(props=[shyp_dd024, dipgrad_a1, dipgrad_c3, dipgrad_b5])
avrg_expr_04 = term_abst.PropsCollection(props=[shyp_ab024, dipgrad_d1, dipgrad_c3, dipgrad_b5])

shyp_aa024 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=2), wd_abst.QOperator(o=4)], dord=2)
shyp_aa024.inds = ['a', 'a']

avrg_expr_05 = term_abst.PropsCollection(props=[shyp_aa024, dipgrad_d1, dipgrad_c3, dipgrad_b5])

shyp_cc024 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=2), wd_abst.QOperator(o=4)], dord=2)
shyp_cc024.inds = ['c', 'c']
dipgrad_a3 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=3)], dord=1)
dipgrad_a3.inds = ['a']

avrg_expr_06 = term_abst.PropsCollection(props=[shyp_cc024, dipgrad_d1, dipgrad_a3, dipgrad_b5])

def make_avrg_expr(proptuples: list[tuple]):
    """
    proptuples = [((0, 2), ('a', 'c')), ((1,), ('b', 'c')),]
    """
    pp = []
    for ptuple in proptuples:
        prop = wd_abst.PolProp(ops=[wd_abst.QOperator(o=i) for i in ptuple[0]], dord=len(ptuple[1]))
        prop.inds = list(ptuple[1])
        pp.append(prop)

    return term_abst.PropsCollection(props=pp)

avrg_expr_0000 = make_avrg_expr([((0,), ('a',)), 
                                 ((1,), ('b',)),
                                 ((2,), ('a',)),
                                 ((3,), ('c',)),
                                 ((4,), ('a',)),])

avrg_expr_0001 = make_avrg_expr([((0,), ('a',)), 
                                 ((1,), ('b',)),
                                 ((2,), ('c',)),
                                 ((3,), ('d',)),
                                 ((4,), ('e',)),])

def test_group_PropsColls_by_numerator():
    print('\n')
    avrg_expr_00 = make_avrg_expr([((0,), ('a',)), 
                                    ((1,), ('b',)),
                                    ((2,), ('c',)),
                                    ((3,), ('d',)),
                                    ((4,), ('d',)),])

    avrg_expr_01 = make_avrg_expr([((0,), ('a',)), 
                                    ((1,), ('a',)),
                                    ((2,), ('b',)),
                                    ((3,), ('c',)),
                                    ((4,), ('d',)),])

    avrg_expr_02 = make_avrg_expr([((0,2,4), ('d','d')), 
                                    ((1,), ('a',)),
                                    ((3,), ('c',)),
                                    ((5,), ('b',)),])

    avrg_expr_03 = make_avrg_expr([((0,2,4), ('a','b')),
                                    ((1,), ('d',)),
                                    ((3,), ('c',)),
                                    ((5,), ('b',)),])
    
    avrg_expr_04 = make_avrg_expr([((0,2), ('a','b')),
                                    ((1,4), ('d',)),
                                    ((3,), ('c',)),
                                    ((5,), ('b',)),])
    
    avrg_expr_05 = make_avrg_expr([((0,2), ('a','b')),
                                    ((1,), ('d',)),
                                    ((3,5), ('c',)),
                                    ((4,), ('b',)),])

    avrg_expr_06 = make_avrg_expr([((0,2), ('a','b')),
                                    ((1,), ('d',)),
                                    ((3,5), ('c','c',)),
                                    ((4,), ('b',)),])
    
    props_colls = [avrg_expr_00, avrg_expr_01, 
                   avrg_expr_02, avrg_expr_03,
                   avrg_expr_04, avrg_expr_05,
                   avrg_expr_06]
    
    numerator_groups = avrgprops.group_PropsColls_by_numerator(props_colls)

    print(f'original: {len(props_colls)}, set: {len(numerator_groups)}\n')

    for group in numerator_groups:
        print(group, ':', numerator_groups[group], '\n')

def test_group_PropsColls_by_repetition_pattern():
    print()
    # props_colls = [avrg_expr_00, avrg_expr_01, avrg_expr_02, avrg_expr_03, avrg_expr_04]
    props_colls = [avrg_expr_00, avrg_expr_01, avrg_expr_03, avrg_expr_04, avrg_expr_05, avrg_expr_06]
    print(props_colls[2].get_mode_indices_grouped())
    print(props_colls[2].get_mode_indices_group_template())
    print(props_colls[2].get_mode_indices())
    print(avrgprops.group_nm_indices(props_colls[2].get_mode_indices(), props_colls[2].get_mode_indices_group_template()))
    # exit()
    # props_colls = [avrg_expr_00, avrg_expr_01]
    numerator_groups = avrgprops.group_PropsColls_by_numerator(props_colls)
    print(numerator_groups)
    print('\n')

    avrg_mapping = {}
    for group in numerator_groups:
        print('\n', group)
        avrg_mapping[group] = {}
        for p in numerator_groups[group]:
            avrg_mapping[group][p] = avrgprops.nm_indices_repetition_encoding(p.get_mode_indices())
            print(p, p.get_mode_indices(), len(set(p.get_mode_indices())), avrgprops.nm_indices_repetition_encoding(p.get_mode_indices()))
        uniques = avrgprops.group_PropsColls_by_repetition_pattern(numerator_groups[group])
        print('\nunique patterns?', uniques)
        
        # for k, u in uniques.items():
        #     for p in u:
        #         print(' unique avrg tensor pattern:', p, avrgprops.nm_indices_repetition_decoding(p))
    print('--- avrg_mapping')
    for k in avrg_mapping:
        print(len(avrg_mapping[k]))
        print(avrg_mapping[k])
    print('avrg_mapping', len(avrg_mapping))

    print('just check')
    pp_uniques = avrgprops.group_PropsColls_by_repetition_pattern(props_colls)
    print(pp_uniques)


def test_group_PropsColls_by_repetition_pattern2():
    print()

    avrg_expr_00 = make_avrg_expr([((0,), ('a',)), 
                                    ((1,), ('b',)),
                                    ((2,), ('c',)),
                                    ((3,), ('d',)),
                                    ((4,), ('d',)),])

    avrg_expr_01 = make_avrg_expr([((0,), ('a',)), 
                                    ((1,), ('a',)),
                                    ((2,), ('b',)),
                                    ((3,), ('c',)),
                                    ((4,), ('d',)),])

    avrg_expr_02 = make_avrg_expr([((0,2,4), ('d','d')), 
                                    ((1,), ('a',)),
                                    ((3,), ('c',)),
                                    ((5,), ('b',)),])

    avrg_expr_03 = make_avrg_expr([((0,2,4), ('a','b')),
                                    ((1,), ('d',)),
                                    ((3,), ('c',)),
                                    ((5,), ('b',)),])
    
    avrg_expr_04 = make_avrg_expr([((0,2), ('a','b')),
                                    ((1,4), ('d',)),
                                    ((3,), ('c',)),
                                    ((5,), ('b',)),])
    
    avrg_expr_05 = make_avrg_expr([((0,2), ('a','b')),
                                    ((1,), ('d',)),
                                    ((3,5), ('c',)),
                                    ((4,), ('b',)),])

    avrg_expr_06 = make_avrg_expr([((0,2), ('a','b')),
                                    ((1,), ('d',)),
                                    ((3,5), ('c','c',)),
                                    ((4,), ('b',)),])

    avrg_expr_07 = make_avrg_expr([((0,2), ('a','b')),
                                    ((1,), ('c',)),
                                    ((3,5), ('d','d',)),
                                    ((4,), ('b',)),])
    
    avrg_expr_08 = make_avrg_expr([((0,2,4), ('a','a')), 
                                    ((1,), ('d',)),
                                    ((3,), ('c',)),
                                    ((5,), ('b',)),])
    avrg_expr_09 = make_avrg_expr([((0,2,4), ('a','a')), 
                                    ((1,), ('a',)),
                                    ((3,), ('c',)),
                                    ((5,), ('b',)),])
    avrg_expr_10 = make_avrg_expr([((0,2,4), ('c','c')), 
                                    ((1,), ('c',)),
                                    ((3,), ('b',)),
                                    ((5,), ('a',)),])

    avrg_expr_11 = make_avrg_expr([((0,2,4), ('c','c')), 
                                    ((1,), ('a',)),
                                    ((3,), ('b',)),
                                    ((5,), ('c',)),])
    
    props_colls = [avrg_expr_00, avrg_expr_01, 
                   avrg_expr_02, avrg_expr_03,
                   avrg_expr_04, avrg_expr_05,
                   avrg_expr_06, avrg_expr_07,
                   avrg_expr_08, avrg_expr_09, 
                   avrg_expr_10, avrg_expr_11]
    # props_colls = [avrg_expr_00, avrg_expr_01, avrg_expr_02, avrg_expr_03, avrg_expr_04]
    # props_colls = [avrg_expr_00, avrg_expr_01, avrg_expr_03, avrg_expr_04, avrg_expr_05, avrg_expr_06]
    # props_colls = [avrg_expr_00, avrg_expr_01]
    numerator_groups = avrgprops.group_PropsColls_by_numerator(props_colls)
    print(numerator_groups)
    print('\n')

    avrg_mapping = {}
    for group in numerator_groups:
        print('\n', group)
        avrg_mapping[group] = {}
        for p in numerator_groups[group]:
            avrg_mapping[group][p] = avrgprops.nm_indices_repetition_encoding(p.get_mode_indices())
            print(p, p.get_mode_indices(), len(set(p.get_mode_indices())), avrgprops.nm_indices_repetition_encoding(p.get_mode_indices()))
        uniques = avrgprops.group_PropsColls_by_repetition_pattern(numerator_groups[group])
        print('\nunique patterns?', uniques)
        
        # for k, u in uniques.items():
        #     for p in u:
        #         print(' unique avrg tensor pattern:', p, avrgprops.nm_indices_repetition_decoding(p))
    print('--- avrg_mapping')
    for k in avrg_mapping:
        print(len(avrg_mapping[k]))
        print(avrg_mapping[k])
    print('avrg_mapping', len(avrg_mapping))

    print('just check')
    pp_uniques = avrgprops.group_PropsColls_by_repetition_pattern(props_colls)
    print(pp_uniques)


def test_id_unique_avrg_tensors_for_num_motif_group2():
    print()
    avrg_expr_00 = make_avrg_expr([((0,), ('a',)), 
                                    ((1,), ('b',)),
                                    ((2,), ('c',)),
                                    ((3,), ('d',)),
                                    ((4,), ('d',)),])

    avrg_expr_01 = make_avrg_expr([((0,), ('a',)), 
                                    ((1,), ('a',)),
                                    ((2,), ('b',)),
                                    ((3,), ('c',)),
                                    ((4,), ('d',)),])

    avrg_expr_02 = make_avrg_expr([((0,2,4), ('d','d')), 
                                    ((1,), ('a',)),
                                    ((3,), ('c',)),
                                    ((5,), ('b',)),])

    avrg_expr_03 = make_avrg_expr([((0,2,4), ('a','b')),
                                    ((1,), ('d',)),
                                    ((3,), ('c',)),
                                    ((5,), ('b',)),])
    
    avrg_expr_04 = make_avrg_expr([((0,2), ('a','b')),
                                    ((1,4), ('d',)),
                                    ((3,), ('c',)),
                                    ((5,), ('b',)),])
    
    avrg_expr_05 = make_avrg_expr([((0,2), ('a','b')),
                                    ((1,), ('d',)),
                                    ((3,5), ('c',)),
                                    ((4,), ('b',)),])

    avrg_expr_06 = make_avrg_expr([((0,2), ('a','b')),
                                    ((1,), ('d',)),
                                    ((3,5), ('c','c',)),
                                    ((4,), ('b',)),])

    avrg_expr_07 = make_avrg_expr([((0,2), ('a','b')),
                                    ((1,), ('c',)),
                                    ((3,5), ('d','d',)),
                                    ((4,), ('b',)),])
    
    avrg_expr_08 = make_avrg_expr([((0,2,4), ('a','a')), 
                                    ((1,), ('d',)),
                                    ((3,), ('c',)),
                                    ((5,), ('b',)),])
    avrg_expr_09 = make_avrg_expr([((0,2,4), ('a','a')), 
                                    ((1,), ('a',)),
                                    ((3,), ('c',)),
                                    ((5,), ('b',)),])
    avrg_expr_10 = make_avrg_expr([((0,2,4), ('c','c')), 
                                    ((1,), ('c',)),
                                    ((3,), ('b',)),
                                    ((5,), ('a',)),])

    avrg_expr_11 = make_avrg_expr([((0,2,4), ('c','c')), 
                                    ((1,), ('a',)),
                                    ((3,), ('b',)),
                                    ((5,), ('c',)),])
    
    props_colls = [avrg_expr_00, avrg_expr_01, 
                   avrg_expr_02, avrg_expr_03,
                   avrg_expr_04, avrg_expr_05,
                   avrg_expr_06, avrg_expr_07,
                   avrg_expr_08, avrg_expr_09, 
                   avrg_expr_10, avrg_expr_11]
    print(props_colls)
    rr = avrgprops.make_unique_avrg_tensors_mapping(props_colls)
    print(rr)
    print('len(props_colls), len(rr)', len(props_colls), len(rr))
    exit()

    # props_colls = [avrg_expr_00, avrg_expr_01]
    numerator_groups = avrgprops.group_PropsColls_by_numerator(props_colls)
    # print(numerator_groups)
    print('\n')

    avrg_mapping = {}
    for group in numerator_groups:
        print('\n', group)
        avrg_mapping[group] = avrgprops.make_unique_avrg_tensors_mapping(numerator_groups[group], group)
    print('----')
    # print('\navrg_mapping', avrg_mapping)
    for numerator_g in avrg_mapping:
        print(numerator_g)
        for t_avrg in avrg_mapping[numerator_g]:
            print(t_avrg, '|||', avrg_mapping[numerator_g][t_avrg])
    print(avrg_mapping)

# {dipNone[0] * dipNone[1] * dipNone[2] * dipNone[3] * dipNone[4]: {dipgrad['a'][0] * dipgrad['b'][1] * dipgrad['c'][2] * dipgrad['d'][3] * dipgrad['d'][4]: (1, 1, 0, 0, 0)}, 
#  hypNone[0, 2, 4] * dipNone[1] * dipNone[3] * dipNone[5]: {hyphess['d', 'd'][0, 2, 4] * dipgrad['a'][1] * dipgrad['c'][3] * dipgrad['b'][5]: (1, 1, 0, 0, 0)}}

def test_nm_indices_repetition_encoding():
    r1 = avrgprops.nm_indices_repetition_encoding(['a', 'a', 'c'])
    print(r1)
    r2 = avrgprops.nm_indices_repetition_encoding(['a', 'c', 'b'])
    print(r2)

    r3 = avrgprops.nm_indices_repetition_encoding(['a', 'a', 'c', 'c'])
    print(r3)

    r4 = avrgprops.nm_indices_repetition_encoding(['a', 'b', 'c', 'b'])
    print(r4)

    r5 = avrgprops.nm_indices_repetition_encoding(['b', 'c', 'a'])
    print(r5)

    r6 = avrgprops.nm_indices_repetition_encoding(['a', 'b', 'c', 'b', 'a'])
    print(r6)

def test_nm_indices_repetition_decoding():
    print()
    r1 = avrgprops.nm_indices_repetition_encoding(['a', 'a', 'c'])
    print(r1, avrgprops.nm_indices_repetition_decoding(r1))
    r2 = avrgprops.nm_indices_repetition_encoding(['a', 'c', 'b'])
    print(r2, avrgprops.nm_indices_repetition_decoding(r2))

    r3 = avrgprops.nm_indices_repetition_encoding(['a', 'a', 'c', 'c'])
    print(r3, avrgprops.nm_indices_repetition_decoding(r3))
    r4 = avrgprops.nm_indices_repetition_encoding(['a', 'b', 'c', 'b'])
    print(r4, avrgprops.nm_indices_repetition_decoding(r4))
    r5 = avrgprops.nm_indices_repetition_encoding(['b', 'c', 'a'])
    print(r5, avrgprops.nm_indices_repetition_decoding(r5))
    r6 = avrgprops.nm_indices_repetition_encoding(['a', 'b', 'c', 'b', 'a'])
    print(r6, avrgprops.nm_indices_repetition_decoding(r6))

def test_reconstruct_unique_avrg_expression():
    print()
    
    props_colls = [avrg_expr_00, avrg_expr_01, avrg_expr_02, avrg_expr_03, 
                   avrg_expr_04, avrg_expr_05, avrg_expr_06,
                   avrg_expr_0000, avrg_expr_0001]
    print('len(props_colls)', len(props_colls))
    # props_colls = [avrg_expr_00, avrg_expr_01]
    numerator_groups = avrgprops.group_PropsColls_by_numerator(props_colls)
    print(numerator_groups)
    print('\n')

    all_uniques_list = []
    for group in numerator_groups:
        print('\n', group, len(numerator_groups[group]))
        uniques = avrgprops.group_PropsColls_by_repetition_pattern(numerator_groups[group])
        print('unique patterns?', uniques, '\n')
        # print('numerator_groups[group]', numerator_groups[group])
        for pattern, expr in uniques.items():
            # print(' max_inds:', max_inds)
            # for expression in expr:
            new_inds = avrgprops.nm_indices_repetition_decoding(pattern)
            reconstructed = avrgprops.reconstruct_unique_avrg_expression(group, new_inds)
            all_uniques_list.append(reconstructed)
            print('len(expr)', len(expr))

    print('\nAll reconstructed unique avrg expressions:')
    for au in all_uniques_list:
        print(au)

def test_identify_unique_avrg_tensors():
    print()
    props_colls = [avrg_expr_00, avrg_expr_01, avrg_expr_02, avrg_expr_03, avrg_expr_04, avrg_expr_05, avrg_expr_06]

    result = avrgprops.identify_unique_avrg_tensors(props_colls)
    print(result)
    print(len(result))

def test_identify_unique_avrg_tensors2():
    print()
    from wilson_suite.fixtures import SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH
    from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_dict_to_dicts
    terms_fuller = SIMPLE_REPRESENTATIVE_FIXTURE_OR_SMTH()
    terms_fuller_flat = derived_terms_dict_to_dicts(terms_fuller, tolistonly=True)
    
    t_inds = [0, 1, -1, -2, -3]
    terms_select = [terms_fuller_flat[tID] for tID in t_inds]
    props_colls = [avrgprops.PropsCollection(props=term.props).get_averaged_props() for term in terms_select]
    print(props_colls, len(props_colls))
    result = avrgprops.identify_unique_avrg_tensors(props_colls)
    print(result)
    print(len(result))

# [dipNone[1]_d1 * dipNone[2]_d1 * polNone[0, 3]_d2, 
#  polNone[0, 3]_d1 * dipNone[1]_d1 * dipNone[2]_d2, 
#  polNone[0, 3]_d1 * dipNone[1]_d1 * dipNone[2]_d1]

polgrad_a03 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=0), wd_abst.QOperator(o=3)], dord=1)
polgrad_a03.inds = ['a']
dipgrad_b1 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=1)], dord=1)
dipgrad_b1.inds = ['b']
dipgrad_b2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
dipgrad_b2.inds = ['b']
dipgrad_a2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
dipgrad_a2.inds = ['a']
dipgrad_c2 = wd_abst.PolProp(ops=[wd_abst.QOperator(o=2)], dord=1)
dipgrad_c2.inds = ['c']

avrg_expr_000 = term_abst.PropsCollection(props=[polgrad_a03, dipgrad_b1, dipgrad_a2])
avrg_expr_001 = term_abst.PropsCollection(props=[polgrad_a03, dipgrad_b1, dipgrad_b2])
avrg_expr_002 = term_abst.PropsCollection(props=[polgrad_a03, dipgrad_b1, dipgrad_c2])

# dipgrad['a'][1] * dipgrad['b'][2] * polhess['a', 'b'][0, 3], 
# polgrad['a'][0, 3] * dipgrad['b'][1] * diphess['b', 'a'][2], 

# polgrad['a'][0, 3] * dipgrad['b'][1] * dipgrad['a'][2], 
# polgrad['a'][0, 3] * dipgrad['b'][1] * dipgrad['b'][2], 
# polgrad['a'][0, 3] * dipgrad['b'][1] * dipgrad['c'][2]

def test_id_unique_avrg_tensors_forAllAvrgs_second():
    print()
    props_colls = [avrg_expr_000, avrg_expr_001, avrg_expr_002]

    result = avrgprops.identify_unique_avrg_tensors(props_colls)
    print(result)
    print(len(result))

def test_group_nm_indices():
    print()
    print(avrgprops.group_nm_indices(['a', 'd', 'b', 'c', 'd'], [2,1,2]))
    print(avrgprops.group_nm_indices(['a', 'd', 'b', 'c', 'd'], [1,2,2]))
    print(avrgprops.group_nm_indices(['a', 'd', 'b', 'c', 'd'], [2,2,1]))
    print(avrgprops.group_nm_indices(['a', 'd', 'b', 'c', 'd'], [2,3]))
    print(avrgprops.group_nm_indices(['a', 'd', 'b', 'c', 'd'], [3,2]))

def test_nm_indices_repetition_reduce_deriv_symmetry():
    print()

    ae1 = make_avrg_expr([((0,2), ('c','b')),
                          ((1,), ('c',)),
                          ((3,5), ('d','d',)),
                          ((4,), ('b',)),])
    
    ae2 = make_avrg_expr([((0,2), ('d','b')),
                          ((1,), ('c',)),
                          ((3,5), ('d','a',)),
                          ((4,), ('b',)),])
    
    print(ae2.get_mode_indices())
    print(avrgprops.nm_indices_repetition_reduce_deriv_symmetry(ae2))

def test_get_avrg_motif_relation():
    print()

    ae11 = make_avrg_expr([((0,2), ('c','b')),
                          ((1,), ('c',)),
                          ((3,5), ('d','d',)),
                          ((4,), ('b',)),])
    
    ae22 = make_avrg_expr([((0,2), ('d','b')),
                          ((1,), ('c',)),
                          ((3,5), ('d','a',)),
                          ((4,), ('b',)),])

    ae1 = make_avrg_expr([((0,), ('a',)),
                          ((1,), ('b',)),
                          ((2,), ('c',)),
                          ((3,), ('d',)),])
    
    ae2 = make_avrg_expr([((0,), ('a',)),
                          ((1,), ('b',)),
                          ((2,), ('c',)),
                          ((3,), ('b',)),])

    ae3 = make_avrg_expr([((0,), ('a',)),
                          ((1,), ('b',)),
                          ((2,), ('a',)),
                          ((3,), ('c',)),])
    
    ae211 = make_avrg_expr([((0,2), ('a','b')),
                          ((1,), ('a',)),
                          ((3,), ('b',)),])
    ae311 = make_avrg_expr([((0,2), ('a','a')),
                          ((1,), ('b',)),
                          ((3,), ('b',)),])
    
    # r = avrgprops.get_avrg_motif_relation(avrg_expr_main=ae1, avrg_expr_sub=ae2, index_dict={'a': 0, 'b': 1, 'c': 2, 'd': 3})
    # print(r)

    g = avrgprops.get_ind_tuple_from_base(expr=ae2, base_expr=ae1, index_dict={'a': 2, 'b': 7, 'c': 4, 'd': 5})
    print(g, '---------\n')
    
    g = avrgprops.get_ind_tuple_from_base(expr=ae3, base_expr=ae1, index_dict={'a': 2, 'b': 7, 'c': 4, 'd': 5})
    print(g, '---------\n')

    g = avrgprops.get_ind_tuple_from_base(expr=ae1, base_expr=ae1, index_dict={'a': 2, 'b': 7, 'c': 4, 'd': 5})
    print(g, '---------\n')

    g = avrgprops.get_ind_tuple_from_base(expr=ae211, base_expr=ae211, index_dict={'a': 2, 'b': 7, 'c': 4, 'd': 5})
    print(g, '---------\n')

    g = avrgprops.get_ind_tuple_from_base(expr=ae311, base_expr=ae311, index_dict={'a': 2, 'b': 7, 'c': 4, 'd': 5})
    print(g, '---------\n')