import pytest

from wilson_suite.wilson_derive.derive import (get_dressed_vib_sos_with_exp_filtering,
                                               do_dbl_pert_expand_and_hermaut_with_enh_filtering,
                                               get_fully_enhanced_terms)
from wilson_suite.fixtures import evv_experiment

def test_get_dressed_vib_sos_with_exp_filtering():

    from fractions import Fraction

    # Test case: EVV terms

    from wilson_suite.wilson_derive.vib_rsp_sos import get_vib_sos

    evv_exp = evv_experiment()

    R_sos = get_dressed_vib_sos_with_exp_filtering(evv_exp.order, evv_exp.int_sequences, evv_exp.epochs,
                                                   evv_exp.cfuv)
    print('\nAFTER FILTERING\n')
    k = 0

    for i in R_sos:
        print('Term, ', k, '\n')
        i.present()
        print('\n\n')
        k += 1

    # Surviving terms here are actually the same terms which are currently tested in
    # test_vib_sos (filtration there was done "manually"), except now dressed with
    # interactions according to -k1 + k2 + k3 phase-matching condition

    # Should have four terms
    assert (len(R_sos) == 4)

    # Corresponds to derivation p.22 res. term 1
    t = R_sos[0]

    assert t.coeff == Fraction(1)
    assert len(t.ints) == 3

    assert t.ints[0].bra.s == '0'
    assert len(t.ints[0].prop.ops) == 2
    assert t.ints[0].prop.ops[0].o == 0
    assert t.ints[0].prop.ops[1].o == 3
    assert t.ints[0].prop.dord == 0
    assert t.ints[0].ket.s == 'n'

    assert t.ints[1].bra.s == 'n'
    assert len(t.ints[1].prop.ops) == 1
    assert t.ints[1].prop.ops[0].o == 2
    assert t.ints[1].prop.dord == 0
    assert t.ints[1].ket.s == 'm'

    assert t.ints[2].bra.s == 'm'
    assert len(t.ints[2].prop.ops) == 1
    assert t.ints[2].prop.ops[0].o == 1
    assert t.ints[2].prop.dord == 0
    assert t.ints[2].ket.s == '0'

    assert len(t.res) == 2
    assert t.res[0].diff.sl.s == 'm'
    assert t.res[0].diff.sr.s == '0'
    assert t.res[0].pf == [-1]

    assert t.res[1].diff.sl.s == 'n'
    assert t.res[1].diff.sr.s == '0'
    assert t.res[1].pf == [-1, 2]

    # Corresponds to derivation p.21 res. term 2
    t = R_sos[1]

    assert t.coeff == Fraction(-1)
    assert len(t.ints) == 3

    assert t.ints[0].bra.s == '0'
    assert len(t.ints[0].prop.ops) == 1
    assert t.ints[0].prop.ops[0].o == 2
    assert t.ints[0].prop.dord == 0
    assert t.ints[0].ket.s == 'n'

    assert t.ints[1].bra.s == 'n'
    assert len(t.ints[1].prop.ops) == 2
    assert t.ints[1].prop.ops[0].o == 0
    assert t.ints[1].prop.ops[1].o == 3
    assert t.ints[1].prop.dord == 0
    assert t.ints[1].ket.s == 'm'

    assert t.ints[2].bra.s == 'm'
    assert len(t.ints[2].prop.ops) == 1
    assert t.ints[2].prop.ops[0].o == 1
    assert t.ints[2].prop.dord == 0
    assert t.ints[2].ket.s == '0'

    assert len(t.res) == 2
    assert t.res[0].diff.sl.s == 'm'
    assert t.res[0].diff.sr.s == '0'
    assert t.res[0].pf == [-1]

    assert t.res[1].diff.sl.s == 'm'
    assert t.res[1].diff.sr.s == 'n'
    assert t.res[1].pf == [-1, 2]

    # Corresponds to derivation p.21 res. term 1
    t = R_sos[2]

    assert t.coeff == Fraction(-1)
    assert len(t.ints) == 3

    assert t.ints[0].bra.s == '0'
    assert len(t.ints[0].prop.ops) == 1
    assert t.ints[0].prop.ops[0].o == 1
    assert t.ints[0].prop.dord == 0
    assert t.ints[0].ket.s == 'm'

    assert t.ints[1].bra.s == 'm'
    assert len(t.ints[1].prop.ops) == 2
    assert t.ints[1].prop.ops[0].o == 0
    assert t.ints[1].prop.ops[1].o == 3
    assert t.ints[1].prop.dord == 0
    assert t.ints[1].ket.s == 'n'

    assert t.ints[2].bra.s == 'n'
    assert len(t.ints[2].prop.ops) == 1
    assert t.ints[2].prop.ops[0].o == 2
    assert t.ints[2].prop.dord == 0
    assert t.ints[2].ket.s == '0'

    assert len(t.res) == 2
    assert t.res[0].diff.sl.s == '0'
    assert t.res[0].diff.sr.s == 'm'
    assert t.res[0].pf == [-1]

    assert t.res[1].diff.sl.s == 'n'
    assert t.res[1].diff.sr.s == 'm'
    assert t.res[1].pf == [-1, 2]

    # Corresponds to derivation p.22 res. term 2
    t = R_sos[3]

    assert t.coeff == Fraction(1)
    assert len(t.ints) == 3

    assert t.ints[0].bra.s == '0'
    assert len(t.ints[0].prop.ops) == 1
    assert t.ints[0].prop.ops[0].o == 1
    assert t.ints[0].prop.dord == 0
    assert t.ints[0].ket.s == 'm'

    assert t.ints[1].bra.s == 'm'
    assert len(t.ints[1].prop.ops) == 1
    assert t.ints[1].prop.ops[0].o == 2
    assert t.ints[1].prop.dord == 0
    assert t.ints[1].ket.s == 'n'

    assert t.ints[2].bra.s == 'n'
    assert len(t.ints[2].prop.ops) == 2
    assert t.ints[2].prop.ops[0].o == 0
    assert t.ints[2].prop.ops[1].o == 3
    assert t.ints[2].prop.dord == 0
    assert t.ints[2].ket.s == '0'

    assert len(t.res) == 2
    assert t.res[0].diff.sl.s == '0'
    assert t.res[0].diff.sr.s == 'm'
    assert t.res[0].pf == [-1]

    assert t.res[1].diff.sl.s == '0'
    assert t.res[1].diff.sr.s == 'n'
    assert t.res[1].pf == [-1, 2]


def test_do_dbl_pert_expand_and_hermaut_with_enh_filtering():

    # Test case: EVV

    # TODO: For one term, do both mech and el expansion, verify results and make requisite asserts

    pass

def test_get_fully_enhanced_terms():

    # Test case: EVV terms

    # The assertions in this test depend on wilson-derive retaining ordering of terms
    evv_exp = evv_experiment()
    terms = get_fully_enhanced_terms(experiment=evv_exp)

    # Two total order of anharmonicity
    assert len(terms) == 2

    # Two categories at 1st order of anharmonicity ([1, 0] and [0, 1])
    assert len(terms[1]) == 2

    # One category at 0th order ([0, 0])
    assert len(terms[0]) == 1

    # Two terms from electrical anharmonicity
    assert len(terms[1][(1, 0)]) == 2

    # Twelve terms from mechanical anharmonicity
    assert len(terms[1][(0, 1)]) == 12

    # No terms at zeroth order of anharmonicity
    assert len(terms[0][(0, 0)]) == 0

    # NOTE about coefficients: A factor -1 difference w.r.t. the form of the reference terms is to be expected
    # because they are expressed in terms of the polarizability, which is -1 times the electric dipole response
    # function used in this code
    # TODO: As part of verification, verify that this factor -1 is in fact accounted for during evaluation
    # However, note that this should not have any bearing on the intensity results since it is common
    # to all terms and is squared

    # Now comparing terms to paper 1 form. All paper 1 terms re-checked w.r.t. own (pen/paper) derivation.

    # Paper 1 terms     Here                        Coeff sign wrt. paper
    #
    # Term 1            terms[1][(1,0)][1]          -1
    # Term 2            terms[1][(1,0)][0]          -1
    # Term 3a           terms[1][(0, 1)][11]        -1
    # Term 3b           terms[1][(0, 1)][6]          1 (pert. WF diff. term opposite order)
    # Term 4a           terms[1][(0, 1)][4]         -1
    # Term 4b           terms[1][(0, 1)][1]         -1
    # Term 5a           terms[1][(0, 1)][8]          1 (pert. WF diff. term opposite order)
    # Term 5b           terms[1][(0, 1)][9]         -1
    # Term 6a           terms[1][(0, 1)][7]          1 (pert. WF diff. term opposite order)
    # Term 6b           terms[1][(0, 1)][10]        -1
    # Term 7a           terms[1][(0, 1)][2]          1 (pert. WF diff. term opposite order)
    # Term 7b           terms[1][(0, 1)][3]         -1
    # Term 8a           terms[1][(0, 1)][5]          1 (pert. WF diff. term opposite order)
    # Term 8b           terms[1][(0, 1)][0]         -1

    # Cross-checks: Relative signs/magnitudes between term coefficients
    # Depends on both relative signs of terms as written and "coeff sign" column in above table

    assert terms[1][(1, 0)][0].coeff ==      terms[1][(1, 0)][1].coeff

    assert terms[1][(1, 0)][0].coeff == -2 * terms[1][(0, 1)][11].coeff
    assert terms[1][(1, 0)][0].coeff ==  2 * terms[1][(0, 1)][6].coeff
    assert terms[1][(1, 0)][0].coeff == -2 * terms[1][(0, 1)][4].coeff
    assert terms[1][(1, 0)][0].coeff == -2 * terms[1][(0, 1)][1].coeff
    assert terms[1][(1, 0)][0].coeff ==  4 * terms[1][(0, 1)][8].coeff
    assert terms[1][(1, 0)][0].coeff == -4 * terms[1][(0, 1)][9].coeff

    # Example: I am on the next code line comparing term 1 and 6a:
    # Term 1's coefficient should be -4 times term 5a's coefficient according to the derivation
    # However, wilson-derive outputs the term corresponding to 6a with the pert WF diff. term in
    # opposite order (it did not do so for the term corresponding to term 1)
    # Therefore, the relative coefficient when compared w.r.t. the wilson-derive terms should here be (plus) 4
    assert terms[1][(1, 0)][0].coeff ==  4 * terms[1][(0, 1)][7].coeff
    assert terms[1][(1, 0)][0].coeff == -4 * terms[1][(0, 1)][10].coeff
    assert terms[1][(1, 0)][0].coeff == -4 * terms[1][(0, 1)][2].coeff
    assert terms[1][(1, 0)][0].coeff ==  4 * terms[1][(0, 1)][3].coeff
    assert terms[1][(1, 0)][0].coeff == -4 * terms[1][(0, 1)][5].coeff
    assert terms[1][(1, 0)][0].coeff ==  4 * terms[1][(0, 1)][0].coeff

    # Term details

    # Electrical anharmonicity

    # Paper 1 term 2
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


    # Paper 1 term 1
    t = terms[1][(1, 0)][1]

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


    # Paper 1 term 8b
    t = terms[1][(0, 1)][0]

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


    # Paper 1 term 4b
    t = terms[1][(0, 1)][1]

    # 1/8
    assert t.coeff == 1 / 8

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not(t.freqterms[0].is_pert_wf_diff)
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


    # Paper 1 term 7a
    t = terms[1][(0, 1)][2]

    # 1/16 (factor -1 on top of the "polarizability -1 sign"
    # because pert WF diff term is opposite order to paper)
    assert t.coeff == 1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not(t.freqterms[0].is_pert_wf_diff)
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


    # Paper 1 term 7b
    t = terms[1][(0, 1)][3]

    # -1/16
    assert t.coeff == -1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not(t.freqterms[0].is_pert_wf_diff)
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


    # Paper 1 term 4a
    t = terms[1][(0, 1)][4]

    # 1/8
    assert t.coeff == 1 / 8

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not(t.freqterms[0].is_pert_wf_diff)
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


    # Paper 1 term 8a
    t = terms[1][(0, 1)][5]

    # 1/16 (factor -1 on top of the "polarizability -1 sign"
    # because pert WF diff term is opposite order to paper)
    assert t.coeff == 1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not(t.freqterms[0].is_pert_wf_diff)
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


    # Paper 1 term 3b
    t = terms[1][(0, 1)][6]

    # -1/8 (factor -1 on top of the "polarizability -1 sign"
    # because pert WF diff term is opposite order to paper)
    assert t.coeff == -1 / 8

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not(t.freqterms[0].is_pert_wf_diff)
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

    # 1/w_(0,a)^[-1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [-1]

    # 1/w_(b,a)^[-1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [-1, 2]


    # Paper 1 term 6a
    t = terms[1][(0, 1)][7]

    # -1/16 (factor -1 on top of the "polarizability -1 sign"
    # because pert WF diff term is opposite order to paper)
    assert t.coeff == -1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not(t.freqterms[0].is_pert_wf_diff)
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

    # 1/w_(0,a)^[-1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [-1]

    # 1/w_(b,a)^[-1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [-1, 2]


    # Paper 1 term 5a
    t = terms[1][(0, 1)][8]

    # -1/16 (factor -1 on top of the "polarizability -1 sign"
    # because pert WF diff term is opposite order to paper)
    assert t.coeff == -1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not(t.freqterms[0].is_pert_wf_diff)
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

    # 1/w_(0,a)^[-1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [-1]

    # 1/w_(b,a)^[-1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [-1, 2]


    # Paper 1 term 5b
    t = terms[1][(0, 1)][9]

    # 1/16
    assert t.coeff == 1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not(t.freqterms[0].is_pert_wf_diff)
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
    assert not(t.freqterms[3].is_pert_wf_diff)
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

    # 1/w_(0,a)^[-1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [-1]

    # 1/w_(b,a)^[-1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [-1, 2]


    # Paper 1 term 6b
    t = terms[1][(0, 1)][10]

    # 1/16
    assert t.coeff == 1 / 16

    assert len(t.freqterms) == 4

    # 1/w_a (pert WF frequency difference)
    assert t.freqterms[0].is_pert_wf_diff
    assert t.freqterms[0].sl.q == ['a']
    assert t.freqterms[0].sr.q == []

    # 1/w_a
    assert not(t.freqterms[1].is_pert_wf_diff)
    assert t.freqterms[1].sl.q == ['a']
    assert t.freqterms[1].sr.q == []

    # 1/w_b
    assert not (t.freqterms[2].is_pert_wf_diff)
    assert t.freqterms[2].sl.q == ['b']
    assert t.freqterms[2].sr.q == []

    # 1/w_c
    assert not(t.freqterms[3].is_pert_wf_diff)
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

    # 1/w_(0,a)^[-1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [-1]

    # 1/w_(b,a)^[-1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [-1, 2]


    # Paper 1 term 3a
    t = terms[1][(0, 1)][11]

    # 1/8
    assert t.coeff == 1 / 8

    assert len(t.freqterms) == 4

    # 1/w_a
    assert not(t.freqterms[0].is_pert_wf_diff)
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
    assert not(t.freqterms[3].is_pert_wf_diff)
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

    # 1/w_(0,a)^[-1]
    assert not (t.res[0].diff.is_pert_wf_diff)
    assert t.res[0].diff.sl.q == []
    assert t.res[0].diff.sr.q == ['a']
    assert t.res[0].pf == [-1]

    # 1/w_(b,a)^[-1, 2]
    assert not (t.res[1].diff.is_pert_wf_diff)
    assert t.res[1].diff.sl.q == ['a', 'b']
    assert t.res[1].diff.sr.q == ['a']
    assert t.res[1].pf == [-1, 2]

