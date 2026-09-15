from wilson_suite.fixtures import evv_experiment
from wilson_suite.wilson_derive.derive import get_fully_enhanced_terms
from wilson_suite.wilson_derive.term_var_translate import translate_terms_to_axis_variables, translate_magn_conditions_to_axisvars
from wilson_suite.wilson_utils.termdict_from_symb_term import derived_terms_flat
from wilson_suite.wilson_experiment.indep_vars_and_axes import SpectralAxisSet, IndependentVariableSet, SignedPulseTuple, SpectralAxis

def test_find_pulse_id_tuples_as_axis_vars():

    pass

def test_translate_one_term_to_axis_variables():

    pass

def test_translate_terms_to_axis_variables():

    evv_exp = evv_experiment()
    terms = get_fully_enhanced_terms(experiment=evv_exp)
    flat_terms = derived_terms_flat(terms, tolist=True)
    evv_exp.tell_axis_options()

    # Testing axis A = -w1, axis B = w2
    axis_choice_mw1_pw2 = evv_exp.valid_axis_combs[0].valid_axis_combs[1]
    assert axis_choice_mw1_pw2.axes[0].label == 'A'
    assert axis_choice_mw1_pw2.axes[0].var_set.var_set[0].pulse_refs == (-1,)
    assert axis_choice_mw1_pw2.axes[1].label == 'B'
    assert axis_choice_mw1_pw2.axes[1].var_set.var_set[0].pulse_refs == (2,)
    translated_terms = translate_terms_to_axis_variables(flat_terms, axis_choice_mw1_pw2)

    # Arbitrary el. anharm term
    t = [t for t in flat_terms if t.anharmonicity==(1,0)][0]
    # t = terms[1][(1, 0)][0]

    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [-1]
    assert t.res[1].diff.sl.q == ['b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [-1, 2]

    # tt = translated_terms[1][(1, 0)][0]
    tt = [t for t in translated_terms if t.anharmonicity==(1,0)][0]

    assert tt.res[0].diff.sl.q == []
    assert tt.res[0].diff.sr.q == ['a']
    assert tt.res[0].pf == ['A']
    assert tt.res[1].diff.sl.q == ['b']
    assert tt.res[1].diff.sr.q == ['a']
    assert tt.res[1].pf == ['A', 'B']

    # Extra check: Is the rest of this term unchanged?
    # -1/4
    assert tt.coeff == -1 / 4

    assert len(tt.freqterms) == 2

    # 1/w_a
    assert not (tt.freqterms[0].is_pert_wf_diff)
    assert tt.freqterms[0].sl.q == ['a']
    assert tt.freqterms[0].sr.q == []

    # 1/w_b
    assert not (tt.freqterms[1].is_pert_wf_diff)
    assert tt.freqterms[1].sl.q == ['b']
    assert tt.freqterms[1].sr.q == []

    assert len(tt.props) == 3

    # d mu_beta/ d Q_a
    assert len(tt.props[0].ops) == 1
    assert tt.props[0].ops[0].o == 1
    assert tt.props[0].dord == 1
    assert tt.props[0].inds == ['a']

    # d mu_gamma/ d Q_b
    assert len(tt.props[1].ops) == 1
    assert tt.props[1].ops[0].o == 2
    assert tt.props[1].dord == 1
    assert tt.props[1].inds == ['b']

    # d 2 alpha_{alpha delta}/ d Q_a d Q_b
    assert len(tt.props[2].ops) == 2
    assert tt.props[2].ops[0].o == 0
    assert tt.props[2].ops[1].o == 3
    assert tt.props[2].dord == 2
    assert tt.props[2].inds == ['a', 'b']

    assert len(tt.res) == 2

    # Arbitrary mech. anharm term (here testing only res. cond.)
    # t = terms[1][(0, 1)][8]
    t = [t for t in flat_terms if t.anharmonicity==(0,1)][8]

    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [-1]
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [-1, 2]

    # tt = translated_terms[1][(0, 1)][8]
    tt = [t for t in translated_terms if t.anharmonicity==(0,1)][8]

    assert tt.res[0].diff.sl.q == []
    assert tt.res[0].diff.sr.q == ['a']
    assert tt.res[0].pf == ['A']
    assert tt.res[1].diff.sl.q == ['a', 'b']
    assert tt.res[1].diff.sr.q == ['a']
    assert tt.res[1].pf == ['A', 'B']

    print(evv_exp.valid_axis_combs[0].valid_axis_combs[0])

    # Now testing other main EVV axes choice: A = -w1, B = w2 - w1
    axis_choice_mw1_mw1_w2 = evv_exp.valid_axis_combs[0].valid_axis_combs[0]
    assert axis_choice_mw1_mw1_w2.axes[0].label == 'A'
    assert axis_choice_mw1_mw1_w2.axes[0].var_set.var_set[0].pulse_refs == (-1,)
    assert axis_choice_mw1_mw1_w2.axes[1].label == 'B'
    assert axis_choice_mw1_mw1_w2.axes[1].var_set.var_set[0].pulse_refs == (-1,)
    assert axis_choice_mw1_mw1_w2.axes[1].var_set.var_set[1].pulse_refs == (2,)

    translated_terms = translate_terms_to_axis_variables(flat_terms, axis_choice_mw1_mw1_w2)

    # Testing the same terms again
    # tt = translated_terms[1][(1, 0)][0]
    tt = [t for t in translated_terms if t.anharmonicity==(1,0)][0]

    assert tt.res[0].diff.sl.q == []
    assert tt.res[0].diff.sr.q == ['a']
    assert tt.res[0].pf == ['A']
    assert tt.res[1].diff.sl.q == ['b']
    assert tt.res[1].diff.sr.q == ['a']
    assert tt.res[1].pf == ['B']

    # tt = translated_terms[1][(0, 1)][8]
    tt = [t for t in translated_terms if t.anharmonicity==(0,1)][8]

    assert tt.res[0].diff.sl.q == []
    assert tt.res[0].diff.sr.q == ['a']
    assert tt.res[0].pf == ['A']
    assert tt.res[1].diff.sl.q == ['a', 'b']
    assert tt.res[1].diff.sr.q == ['a']
    assert tt.res[1].pf == ['B']

    # Now testing custom axes choice: A = w1, B = w2 - w1

    custom_axis_choice = SpectralAxisSet(
        axes=(SpectralAxis(label='A', var_set=IndependentVariableSet(var_set=(SignedPulseTuple(pulse_refs=(1,)),))),
              SpectralAxis(label='B', var_set=IndependentVariableSet(
                           var_set=(SignedPulseTuple(pulse_refs=(-1,)), SignedPulseTuple(pulse_refs=(2,)))))))


    translated_terms = translate_terms_to_axis_variables(flat_terms, custom_axis_choice)

    # Testing the same terms again
    # tt = translated_terms[1][(1, 0)][0]
    tt = [t for t in translated_terms if t.anharmonicity==(1,0)][0]

    assert tt.res[0].diff.sl.q == []
    assert tt.res[0].diff.sr.q == ['a']
    assert tt.res[0].pf == ['-A']
    assert tt.res[1].diff.sl.q == ['b']
    assert tt.res[1].diff.sr.q == ['a']
    assert tt.res[1].pf == ['B']

    # tt = translated_terms[1][(0, 1)][8]
    tt = [t for t in translated_terms if t.anharmonicity==(0,1)][8]

    assert tt.res[0].diff.sl.q == []
    assert tt.res[0].diff.sr.q == ['a']
    assert tt.res[0].pf == ['-A']
    assert tt.res[1].diff.sl.q == ['a', 'b']
    assert tt.res[1].diff.sr.q == ['a']
    assert tt.res[1].pf == ['B']

def test_translate_magn_conditions_to_axisvars():
    custom_axis_choice = SpectralAxisSet(
        axes=(SpectralAxis(label='A', var_set=IndependentVariableSet(var_set=(SignedPulseTuple(pulse_refs=(1,)),))),
              SpectralAxis(label='B', var_set=IndependentVariableSet(
                           var_set=(SignedPulseTuple(pulse_refs=(-1,)), SignedPulseTuple(pulse_refs=(2,)))))))

    evv_exp = evv_experiment()

    ax_mang_cond = translate_magn_conditions_to_axisvars(magn_conditions=evv_exp.magn_conditions, 
                                                         axis_choice=custom_axis_choice)
    assert ax_mang_cond == (('B',),)

    custom_axis_choice = SpectralAxisSet(
        axes=(SpectralAxis(label='A', var_set=IndependentVariableSet(var_set=(SignedPulseTuple(pulse_refs=(1,)),))),
              SpectralAxis(label='B', var_set=IndependentVariableSet(
                           var_set=(SignedPulseTuple(pulse_refs=(2,)),)))))
    ax_mang_cond = translate_magn_conditions_to_axisvars(magn_conditions=evv_exp.magn_conditions, 
                                                         axis_choice=custom_axis_choice)
    assert ax_mang_cond == (('-A', 'B'),)

    import pytest

    with pytest.raises(NotImplementedError):
        custom_axis_choice_wrong = SpectralAxisSet(
            axes=(SpectralAxis(label='A', var_set=IndependentVariableSet(var_set=(SignedPulseTuple(pulse_refs=(-1,)),))),
                SpectralAxis(label='B', var_set=IndependentVariableSet(
                            var_set=(SignedPulseTuple(pulse_refs=(2,)),)))))
        ax_mang_cond = translate_magn_conditions_to_axisvars(magn_conditions=evv_exp.magn_conditions, 
                                                         axis_choice=custom_axis_choice_wrong)

    with pytest.raises(NotImplementedError):
        evv_exp.magn_conditions = ((1,-2),)
        ax_mang_cond = translate_magn_conditions_to_axisvars(magn_conditions=evv_exp.magn_conditions, 
                                                         axis_choice=custom_axis_choice)
