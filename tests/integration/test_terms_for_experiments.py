import wilson_suite as ws

def test_evv_experiment_terms():
    # The assertions in this test depend on wilson-derive retaining ordering of terms

    evv_exp = ws.fixtures.evv_experiment()
    terms = ws.derive.derive.get_fully_enhanced_terms(experiment=evv_exp)

    # Two total order of anharmonicity
    assert len(terms) == 2
    # Two categories at 1st order of anharmonicity ([1, 0] and [0, 1])
    assert len(terms[1]) == 2
    # One category at 0th order ([0, 0])
    assert len(terms[0]) == 1

    assert len(terms[1][(1, 0)]) == 2
    assert len(terms[1][(0, 1)]) == 12

    # NOTE about coefficients: A factor -1 difference w.r.t. the form of the reference terms is to be expected
    # because they are expressed in terms of the polarizability, which is -1 times the electric dipole response
    # function used in this code
    # TODO: As part of verification, verify that this factor -1 is in fact accounted for during evaluation
    # However, note that this should not have any bearing on the intensity results since it is common
    # to all terms and is squared

    # Now comparing terms to paper 1 form. All paper 1 terms re-checked w.r.t. own (pen/paper) derivation.

    # Electrical anharmonicity

    # Paper 1 term #1

    t = terms[1][(1,0)][0]

    # -1/4
    assert t.coeff == -1/4

    assert len(t.freqterms) == 2

    # 1/w_a
    assert not(t.freqterms[0].is_pert_wf_diff)
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_b
    assert not (t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == ['b']
    assert t.freqterms[1].sr.q == []

    # d mu_beta/ d Q_a
    assert len(t.props[0].ops) == 1
    assert t.props[0].ops[0].o == 1
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['a']

    # d mu_gamma/ d Q_b
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 2
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['b']

    # d 2 alpha_{alpha delta}/ d Q_a d Q_b
    assert len(t.props[2].ops) == 2
    assert t.props[2].ops[0].o == 0
    assert t.props[2].ops[1].o == 3
    assert t.props[2].dord == 2
    assert t.props[2].inds == ['a', 'b']

    assert len(t.res) == 2

    # 1/w_(0,a)^[-1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [-1]

    # 1/w_(b,a)^[-1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [-1, 2]

    # Paper 1 term #0
    t = terms[1][(1, 0)][1]

    print(terms[1][(1, 0)][1])

    # -1/4
    assert t.coeff == -1 / 4

    assert len(t.freqterms) == 2

    # 1/w_a
    assert not (t.freqterms[0].is_pert_wf_diff)
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_b
    assert not (t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == ['b']
    assert t.freqterms[1].sr.q == []

    # d alpha_{alpha delta}/ d Q_b
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['b']

    # d mu_beta/ d Q_b
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 1
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['a']

    # d 2 mu_gamma/ d Q_a d Q_b
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 2
    assert t.props[2].inds == ['a', 'b']

    assert len(t.res) == 2

    # 1/w_(0,a)^[-1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [-1]

    # 1/w_(a + b,a)^[-1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [-1, 2]


    # Mechanical anharmonicity

    terms[1][(0, 1)][].present()



