import pytest

from wilson_suite.wilson_derive.derive import (get_terms_by_comp_pattern_and_magn_conds,
                                               do_dbl_pert_expand_and_hermaut,
                                               ResponseTermsCollection)

from wilson_suite.wilson_system.system_data import MolPropsCollection

def test_response_terms_collection():

    # Setting up with EVV data
    order = 3
    tot_anharm_limit = 1
    el_anharm_limit = 1
    mech_anharm_limit = 1
    comp_patt = (1, 1, 2)

    # At -k1 + k2 (+ k3): w1 is negative, w2 is positive, here constricting to w1 + w2 signif. > 0
    mc_plus = ( (1,), (0,1,) , )
    mc_minus = ((0,),)

    rsp_coll = ResponseTermsCollection(order = order, total_anharm_limit=tot_anharm_limit,
                                       el_anharm_limit=el_anharm_limit, mech_anharm_limit=mech_anharm_limit,
                                       comp_pattern=comp_patt, magn_conds=(mc_plus, mc_minus))

    # There should be 14 terms
    assert len(rsp_coll.terms) == 14

    # There should be these (derivative) properties identified and they should be a MolPropsCollection instance
    assert isinstance(rsp_coll.props, MolPropsCollection)
    assert sorted(rsp_coll.props.names()) == sorted(['dipgrad', 'polhess', 'polgrad', 'diphess', 'cff'])

    # There should be up to 3 quanta in the state references
    assert rsp_coll.max_state_lvl == 3

    # Pair quanta should be (0, 1) (e.g. 0, a), (1, 1) (e.g. (a, b)), and (2, 1) (e.g. ( (a, b), b ) )
    assert rsp_coll.pair_quanta == {(0, 1), (1, 1), (2, 1)}


def test_do_dbl_pert_expand_and_hermaut():

    from wilson_suite.wilson_derive.vib_rsp_sos import get_vib_sos

    R_sos = get_vib_sos(3)

    R_sos_evv = []

    for t in R_sos:
        if t.comp_pattern == (1, 1, 2):
            R_sos_evv.append(t)

    assert len(R_sos_evv) == 4

    # Magnitude conditions for post-filtering
    mc_plus = ( [2], [1,2] )
    mc_minus = ([1],)

    # First term

    # No double harmonic terms
    R_result_raw = do_dbl_pert_expand_and_hermaut([R_sos_evv[0]], 0, 0)

    assert len(R_result_raw) == 0


    # Should be terms here but none satisfying the magnitude conditions
    R_result_raw = do_dbl_pert_expand_and_hermaut([R_sos_evv[0]], 1, 0)

    assert len(R_result_raw) > 0

    R_result = []
    for t in R_result_raw:
        if t.full_enhancement_possible(mcp=mc_plus, mcm=mc_minus):
            R_result.append(t)

    assert len(R_result) == 0

    # Should be terms here but none satisfying the magnitude conditions
    R_result_raw = do_dbl_pert_expand_and_hermaut([R_sos_evv[0]], 0, 1)

    assert len(R_result_raw) > 0

    R_result = []
    for t in R_result_raw:
        if t.full_enhancement_possible(mcp=mc_plus, mcm=mc_minus):
            R_result.append(t)

    assert len(R_result) == 0

    # Second term:

    # No double harmonic terms
    R_result_raw = do_dbl_pert_expand_and_hermaut([R_sos_evv[1]], 0, 0)

    assert len(R_result_raw) == 0


    # Should be terms here but none satisfying the magnitude conditions
    R_result_raw = do_dbl_pert_expand_and_hermaut([R_sos_evv[1]], 1, 0)

    assert len(R_result_raw) > 0

    R_result = []
    for t in R_result_raw:
        if t.full_enhancement_possible(mcp=mc_plus, mcm=mc_minus):
            R_result.append(t)

    assert len(R_result) == 0

    # Should be terms here but none satisfying the magnitude conditions
    R_result_raw = do_dbl_pert_expand_and_hermaut([R_sos_evv[1]], 0, 1)

    assert len(R_result_raw) > 0

    R_result = []
    for t in R_result_raw:
        if t.full_enhancement_possible(mcp=mc_plus, mcm=mc_minus):
            R_result.append(t)

    assert len(R_result) == 0

    # Third term

    # No double harmonic terms
    R_result_raw = do_dbl_pert_expand_and_hermaut([R_sos_evv[2]], 0, 0)

    assert len(R_result_raw) == 0

    # Two el. anharm. terms make it: Contents already tested in test_get_fully_enhanced_terms
    R_result_raw = do_dbl_pert_expand_and_hermaut([R_sos_evv[2]], 1, 0)

    R_result = []
    for t in R_result_raw:
        if t.full_enhancement_possible(mcp=mc_plus, mcm=mc_minus):
            R_result.append(t)

    assert len(R_result) == 2


    # Twelve mech. anharm. terms make it: Contents already tested in test_get_fully_enhanced_terms
    R_result_raw = do_dbl_pert_expand_and_hermaut([R_sos_evv[2]], 0, 1)

    R_result = []
    for t in R_result_raw:
        if t.full_enhancement_possible(mcp=mc_plus, mcm=mc_minus):
            R_result.append(t)

    assert len(R_result) == 12

    # Fourth term:

    # No double harmonic terms
    R_result_raw = do_dbl_pert_expand_and_hermaut([R_sos_evv[3]], 0, 0)

    assert len(R_result_raw) == 0


    # Should be terms here but none satisfying the magnitude conditions
    R_result_raw = do_dbl_pert_expand_and_hermaut([R_sos_evv[3]], 1, 0)

    assert len(R_result_raw) > 0

    R_result = []
    for t in R_result_raw:
        if t.full_enhancement_possible(mcp=mc_plus, mcm=mc_minus):
            R_result.append(t)

    assert len(R_result) == 0

    # Should be terms here but none satisfying the magnitude conditions
    R_result_raw = do_dbl_pert_expand_and_hermaut([R_sos_evv[3]], 0, 1)

    assert len(R_result_raw) > 0

    R_result = []
    for t in R_result_raw:
        if t.full_enhancement_possible(mcp=mc_plus, mcm=mc_minus):
            R_result.append(t)

    assert len(R_result) == 0



def test_get_terms_by_comp_pattern_and_magn_conds():
    # Setting up with EVV data
    order = 3
    tot_anharm_limit = 1
    el_anharm_limit = 1
    mech_anharm_limit = 1
    comp_patt = (1, 1, 2)

    # At -k1 + k2 (+ k3): w1 is negative, w2 is positive, here constricting to w1 + w2 signif. > 0
    mc_plus = ((1,), (0, 1,),)
    mc_minus = ((0,),)

    # Getting terms
    # The assertions in this test depend on wilson-derive retaining ordering of terms
    terms = get_terms_by_comp_pattern_and_magn_conds(order=order, total_anharm_limit=tot_anharm_limit,
                                                     el_anharm_limit=el_anharm_limit,
                                                     mech_anharm_limit=mech_anharm_limit,
                                                     comp_pattern=comp_patt,
                                                     magn_conds=(mc_plus, mc_minus))


    tot_mech = 0
    tot_el = 0

    # Terms should be anharm orders (1, 0) or (0, 1)
    for t in terms:

        assert t.order_mech == 1 or t.order_el == 1

        if t.order_mech == 1:

            assert t.order_el == 0
            tot_mech += 1

        elif t.order_el == 1:

            tot_el += 1

    # Two terms from electrical anharmonicity
    assert tot_el == 2

    # Twelve terms from mechanical anharmonicity
    assert tot_mech == 12

    # NOTE about coefficients: A factor -1 difference w.r.t. the form of the reference terms is to be expected
    # because they are expressed in terms of the polarizability, which is -1 times the electric dipole response
    # function used in this code
    # TODO: As part of verification, verify that this factor -1 is in fact accounted for during evaluation
    # However, note that this should not have any bearing on the intensity results since it is common
    # to all terms and is squared

    # Now comparing terms to paper 1 form. All paper 1 terms re-checked w.r.t. own (pen/paper) derivation.

    # Paper 1 terms     Here                        Coeff sign wrt. paper
    #
    # Term 1            terms[1]                     -1
    # Term 2            terms[0]                     -1
    # Term 3a           terms[13]                    -1
    # Term 3b           terms[8]                      1 (pert. WF diff. term opposite order)
    # Term 4a           terms[6]                     -1
    # Term 4b           terms[3]                     -1
    # Term 5a           terms[10]                     1 (pert. WF diff. term opposite order)
    # Term 5b           terms[11]                    -1
    # Term 6a           terms[9]                      1 (pert. WF diff. term opposite order)
    # Term 6b           terms[12]                    -1
    # Term 7a           terms[4]                      1 (pert. WF diff. term opposite order)
    # Term 7b           terms[5]                     -1
    # Term 8a           terms[7]                      1 (pert. WF diff. term opposite order)
    # Term 8b           terms[2]                     -1

    # Cross-checks: Relative signs/magnitudes between term coefficients
    # Depends on both relative signs of terms as written and "coeff sign" column in above table

    assert terms[0].coeff == terms[1].coeff

    assert terms[0].coeff == -2 * terms[13].coeff
    assert terms[0].coeff ==  2 * terms[8].coeff
    assert terms[0].coeff == -2 * terms[6].coeff
    assert terms[0].coeff == -2 * terms[3].coeff
    assert terms[0].coeff ==  4 * terms[10].coeff
    assert terms[0].coeff == -4 * terms[11].coeff

    # Example: I am on the next code line comparing term 1 and 6a:
    # Term 1's coefficient should be -4 times term 5a's coefficient according to the derivation
    # However, wilson-derive outputs the term corresponding to 6a with the pert WF diff. term in
    # opposite order (it did not do so for the term corresponding to term 1)
    # Therefore, the relative coefficient when compared w.r.t. the wilson-derive terms should here be (plus) 4
    assert terms[0].coeff == 4 * terms[9].coeff
    assert terms[0].coeff == -4 * terms[12].coeff
    assert terms[0].coeff == -4 * terms[4].coeff
    assert terms[0].coeff == 4 * terms[5].coeff
    assert terms[0].coeff == -4 * terms[7].coeff
    assert terms[0].coeff == 4 * terms[2].coeff

    # Term details

    # Electrical anharmonicity

    # Paper 1 term 2
    t = terms[0]

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

    assert len(t.props) == 3

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

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [1]

    # 1/w_(b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [1, 2]

    # Paper 1 term 1
    t = terms[1]

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

    assert len(t.props) == 3

    # d alpha_{alpha delta}/ d Q_b
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['b']

    # d mu_beta/ d Q_a
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

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [1]

    # 1/w_(a + b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [1, 2]

    # Mechanical anharmonicity

    # Paper 1 term 8b
    t = terms[2]

    # -1/16
    assert t.coeff == -1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a (pert WF frequency difference)
    assert t.freqterms[0].is_pert_wf_diff
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_a
    assert not (t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == ['a']
    assert t.freqterms[1].sr.q == []

    # 1/w_b
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == ['b']
    assert t.freqterms[2].sr.q == []

    # 1/w_c
    assert not (t.freqterms[3].is_pert_wf_diff)
    assert t.freqterms[3].sl.q == ['c']
    assert t.freqterms[3].sr.q == []

    assert len(t.props) == 4

    # d alpha_{alpha delta}/ d Q_b
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['b']

    # d mu_beta/ d Q_a
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 1
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['a']

    # d mu_gamma/ d Q_b
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 1
    assert t.props[2].inds == ['b']

    # F_acc
    assert len(t.props[3].ops) == 0
    assert t.props[3].dord == 3
    assert t.props[3].inds == ['a', 'c', 'c']

    assert len(t.res) == 2

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [1]

    # 1/w_(b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [1, 2]

    # Paper 1 term 4b
    t = terms[3]

    # 1/8
    assert t.coeff == 1 / 8

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not (t.freqterms[0].is_pert_wf_diff)
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_b
    assert not (t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == ['b']
    assert t.freqterms[1].sr.q == []

    # 1/w_c
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == ['c']
    assert t.freqterms[2].sr.q == []

    # 1/(w_bc - w_a)  (pert WF frequency difference)
    assert t.freqterms[3].is_pert_wf_diff
    assert t.freqterms[3].sl.q == ['b', 'c']
    assert t.freqterms[3].sr.q == ['a']

    assert len(t.props) == 4

    # d alpha_{alpha delta}/ d Q_c
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['c']

    # d mu_beta/ d Q_a
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 1
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['a']

    # d mu_gamma/ d Q_b
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 1
    assert t.props[2].inds == ['b']

    # F_abc
    assert len(t.props[3].ops) == 0
    assert t.props[3].dord == 3
    assert t.props[3].inds == ['a', 'b', 'c']

    assert len(t.res) == 2

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [1]

    # 1/w_(b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [1, 2]

    # Paper 1 term 7a
    t = terms[4]

    # 1/16 (factor -1 on top of the "polarizability -1 sign"
    # because pert WF diff term is opposite order to paper)
    assert t.coeff == 1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not (t.freqterms[0].is_pert_wf_diff)
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_b
    assert not (t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == ['b']
    assert t.freqterms[1].sr.q == []

    # 1/w_c
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == ['c']
    assert t.freqterms[2].sr.q == []

    # 1/(w_ab - w_a)  (pert WF frequency difference)
    assert t.freqterms[3].is_pert_wf_diff
    assert t.freqterms[3].sl.q == ['a', 'b']
    assert t.freqterms[3].sr.q == ['a']

    assert len(t.props) == 4

    # d alpha_{alpha delta}/ d Q_a
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['a']

    # d mu_beta/ d Q_a
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 1
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['a']

    # d mu_gamma/ d Q_b
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 1
    assert t.props[2].inds == ['b']

    # F_bcc
    assert len(t.props[3].ops) == 0
    assert t.props[3].dord == 3
    assert t.props[3].inds == ['b', 'c', 'c']

    assert len(t.res) == 2

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [1]

    # 1/w_(b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [1, 2]

    # Paper 1 term 7b
    t = terms[5]

    # -1/16
    assert t.coeff == -1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not (t.freqterms[0].is_pert_wf_diff)
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_b  (pert WF frequency difference)
    assert t.freqterms[1].is_pert_wf_diff
    assert t.freqterms[1].sl.q == ['b']
    assert t.freqterms[1].sr.q == []

    # 1/w_b
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == ['b']
    assert t.freqterms[2].sr.q == []

    # 1/w_c
    assert not (t.freqterms[3].is_pert_wf_diff)
    assert t.freqterms[3].sl.q == ['c']
    assert t.freqterms[3].sr.q == []

    assert len(t.props) == 4

    # d alpha_{alpha delta}/ d Q_a
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['a']

    # d mu_beta/ d Q_a
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 1
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['a']

    # d mu_gamma/ d Q_b
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 1
    assert t.props[2].inds == ['b']

    # F_abc
    assert len(t.props[3].ops) == 0
    assert t.props[3].dord == 3
    assert t.props[3].inds == ['b', 'c', 'c']

    assert len(t.res) == 2

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [1]

    # 1/w_(b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [1, 2]

    # Paper 1 term 4a
    t = terms[6]

    # 1/8
    assert t.coeff == 1 / 8

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not (t.freqterms[0].is_pert_wf_diff)
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_b
    assert not (t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == ['b']
    assert t.freqterms[1].sr.q == []

    # 1/w_c
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == ['c']
    assert t.freqterms[2].sr.q == []

    # 1/(w_ac - w_b) (pert WF frequency difference)
    assert t.freqterms[3].is_pert_wf_diff
    assert t.freqterms[3].sl.q == ['a', 'c']
    assert t.freqterms[3].sr.q == ['b']

    assert len(t.props) == 4

    # d alpha_{alpha delta}/ d Q_c
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['c']

    # d mu_beta/ d Q_a
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 1
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['a']

    # d mu_gamma/ d Q_b
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 1
    assert t.props[2].inds == ['b']

    # F_abc
    assert len(t.props[3].ops) == 0
    assert t.props[3].dord == 3
    assert t.props[3].inds == ['a', 'b', 'c']

    assert len(t.res) == 2

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [1]

    # 1/w_(b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [1, 2]

    # Paper 1 term 8a
    t = terms[7]

    # 1/16 (factor -1 on top of the "polarizability -1 sign"
    # because pert WF diff term is opposite order to paper)
    assert t.coeff == 1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not (t.freqterms[0].is_pert_wf_diff)
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_b
    assert not (t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == ['b']
    assert t.freqterms[1].sr.q == []

    # 1/w_c
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == ['c']
    assert t.freqterms[2].sr.q == []

    # 1/(w_ab - w_b) (pert WF frequency difference)
    assert t.freqterms[3].is_pert_wf_diff
    assert t.freqterms[3].sl.q == ['a', 'b']
    assert t.freqterms[3].sr.q == ['b']

    assert len(t.props) == 4

    # d alpha_{alpha delta}/ d Q_b
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['b']

    # d mu_beta/ d Q_a
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 1
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['a']

    # d mu_gamma/ d Q_b
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 1
    assert t.props[2].inds == ['b']

    # F_acc
    assert len(t.props[3].ops) == 0
    assert t.props[3].dord == 3
    assert t.props[3].inds == ['a', 'c', 'c']

    assert len(t.res) == 2

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [1]

    # 1/w_(b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [1, 2]

    # Paper 1 term 3b
    t = terms[8]

    # -1/8 (factor -1 on top of the "polarizability -1 sign"
    # because pert WF diff term is opposite order to paper)
    assert t.coeff == -1 / 8

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not (t.freqterms[0].is_pert_wf_diff)
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_b
    assert not (t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == ['b']
    assert t.freqterms[1].sr.q == []

    # 1/w_c
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == ['c']
    assert t.freqterms[2].sr.q == []

    # 1/(w_ab - w_c) (pert WF frequency difference)
    assert t.freqterms[3].is_pert_wf_diff
    assert t.freqterms[3].sl.q == ['a', 'b']
    assert t.freqterms[3].sr.q == ['c']

    assert len(t.props) == 4

    # d alpha_{alpha delta}/ d Q_b
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['b']

    # d mu_beta/ d Q_a
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 1
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['a']

    # d mu_gamma/ d Q_c
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 1
    assert t.props[2].inds == ['c']

    # F_abc
    assert len(t.props[3].ops) == 0
    assert t.props[3].dord == 3
    assert t.props[3].inds == ['a', 'b', 'c']

    assert len(t.res) == 2

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [1]

    # 1/w_(b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [1, 2]

    # Paper 1 term 6a
    t = terms[9]

    # -1/16 (factor -1 on top of the "polarizability -1 sign"
    # because pert WF diff term is opposite order to paper)
    assert t.coeff == -1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not (t.freqterms[0].is_pert_wf_diff)
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_b
    assert not (t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == ['b']
    assert t.freqterms[1].sr.q == []

    # 1/w_c
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == ['c']
    assert t.freqterms[2].sr.q == []

    # 1/(w_ab - w_b) (pert WF frequency difference)
    assert t.freqterms[3].is_pert_wf_diff
    assert t.freqterms[3].sl.q == ['a', 'b']
    assert t.freqterms[3].sr.q == ['b']

    assert len(t.props) == 4

    # d alpha_{alpha delta}/ d Q_b
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['b']

    # d mu_beta/ d Q_a
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 1
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['a']

    # d mu_gamma/ d Q_b
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 1
    assert t.props[2].inds == ['b']

    # F_acc
    assert len(t.props[3].ops) == 0
    assert t.props[3].dord == 3
    assert t.props[3].inds == ['a', 'c', 'c']

    assert len(t.res) == 2

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [1]

    # 1/w_(b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [1, 2]

    # Paper 1 term 5a
    t = terms[10]

    # -1/16 (factor -1 on top of the "polarizability -1 sign"
    # because pert WF diff term is opposite order to paper)
    assert t.coeff == -1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not (t.freqterms[0].is_pert_wf_diff)
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_b
    assert not (t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == ['b']
    assert t.freqterms[1].sr.q == []

    # 1/w_c
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == ['c']
    assert t.freqterms[2].sr.q == []

    # 1/(w_ab - w_a) (pert WF frequency difference)
    assert t.freqterms[3].is_pert_wf_diff
    assert t.freqterms[3].sl.q == ['a', 'b']
    assert t.freqterms[3].sr.q == ['a']

    assert len(t.props) == 4

    # d alpha_{alpha delta}/ d Q_b
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['b']

    # d mu_beta/ d Q_a
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 1
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['a']

    # d mu_gamma/ d Q_a
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 1
    assert t.props[2].inds == ['a']

    # F_bcc
    assert len(t.props[3].ops) == 0
    assert t.props[3].dord == 3
    assert t.props[3].inds == ['b', 'c', 'c']

    assert len(t.res) == 2

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [1]

    # 1/w_(b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [1, 2]

    # Paper 1 term 5b
    t = terms[11]

    # 1/16
    assert t.coeff == 1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not (t.freqterms[0].is_pert_wf_diff)
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_b (pert WF frequency difference)
    assert t.freqterms[1].is_pert_wf_diff
    assert t.freqterms[1].sl.q == ['b']
    assert t.freqterms[1].sr.q == []

    # 1/w_b
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == ['b']
    assert t.freqterms[2].sr.q == []

    # 1/w_c
    assert not (t.freqterms[3].is_pert_wf_diff)
    assert t.freqterms[3].sl.q == ['c']
    assert t.freqterms[3].sr.q == []

    assert len(t.props) == 4

    # d alpha_{alpha delta}/ d Q_b
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['b']

    # d mu_beta/ d Q_a
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 1
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['a']

    # d mu_gamma/ d Q_a
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 1
    assert t.props[2].inds == ['a']

    # F_bcc
    assert len(t.props[3].ops) == 0
    assert t.props[3].dord == 3
    assert t.props[3].inds == ['b', 'c', 'c']

    assert len(t.res) == 2

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [1]

    # 1/w_(b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [1, 2]

    # Paper 1 term 6b
    t = terms[12]

    # 1/16
    assert t.coeff == 1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a (pert WF frequency difference)
    assert t.freqterms[0].is_pert_wf_diff
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_a
    assert not (t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == ['a']
    assert t.freqterms[1].sr.q == []

    # 1/w_b
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == ['b']
    assert t.freqterms[2].sr.q == []

    # 1/w_c
    assert not (t.freqterms[3].is_pert_wf_diff)
    assert t.freqterms[3].sl.q == ['c']
    assert t.freqterms[3].sr.q == []

    assert len(t.props) == 4

    # d alpha_{alpha delta}/ d Q_b
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['b']

    # d mu_beta/ d Q_a
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 1
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['a']

    # d mu_gamma/ d Q_b
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 1
    assert t.props[2].inds == ['b']

    # F_acc
    assert len(t.props[3].ops) == 0
    assert t.props[3].dord == 3
    assert t.props[3].inds == ['a', 'c', 'c']

    assert len(t.res) == 2

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [1]

    # 1/w_(b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [1, 2]

    # Paper 1 term 3a
    t = terms[13]

    # 1/8
    assert t.coeff == 1 / 8

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not (t.freqterms[0].is_pert_wf_diff)
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_abc (pert WF frequency difference)
    assert t.freqterms[1].is_pert_wf_diff
    assert t.freqterms[1].sl.q == ['a', 'b', 'c']
    assert t.freqterms[1].sr.q == []

    # 1/w_b
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == ['b']
    assert t.freqterms[2].sr.q == []

    # 1/w_c
    assert not (t.freqterms[3].is_pert_wf_diff)
    assert t.freqterms[3].sl.q == ['c']
    assert t.freqterms[3].sr.q == []

    assert len(t.props) == 4

    # d alpha_{alpha delta}/ d Q_b
    assert len(t.props[0].ops) == 2
    assert t.props[0].ops[0].o == 0
    assert t.props[0].ops[1].o == 3
    assert t.props[0].dord == 1
    assert t.props[0].inds == ['b']

    # d mu_beta/ d Q_a
    assert len(t.props[1].ops) == 1
    assert t.props[1].ops[0].o == 1
    assert t.props[1].dord == 1
    assert t.props[1].inds == ['a']

    # d mu_gamma/ d Q_c
    assert len(t.props[2].ops) == 1
    assert t.props[2].ops[0].o == 2
    assert t.props[2].dord == 1
    assert t.props[2].inds == ['c']

    # F_abc
    assert len(t.props[3].ops) == 0
    assert t.props[3].dord == 3
    assert t.props[3].inds == ['a', 'b', 'c']

    assert len(t.res) == 2

    # 1/w_(0,a)^[1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [1]

    # 1/w_(b,a)^[1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [1, 2]




