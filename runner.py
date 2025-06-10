import abstractions as abst
import simplify
import vib_rsp_sos
import dbl_pert_expansion
import hermaut

nm_inds = ['a', 'b', 'c', 'd', 'e', 'f']

states = [abst.vibState('0', is_ground=True), abst.vibState('m'), abst.vibState('n'), abst.vibState('p'), abst.vibState('q'),
          abst.vibState('r'), abst.vibState('s'), abst.vibState('t'), abst.vibState('u'), abst.vibState('v'),
          abst.vibState('w'), abst.vibState('x')]

op_omega = abst.qOperator('a', 1)

ops_pert = (
abst.qOperator('b', 1),
abst.qOperator('c', 1),
abst.qOperator('d', 1),
abst.qOperator('e', 1),
abst.qOperator('f', 1),
)

maxord = 3

R_sos = vib_rsp_sos.get_vib_sos(op_omega, ops_pert, maxord, states)

print('len R sos', len(R_sos))

R_dbl_pert_00 = []
R_dbl_pert_10 = []
R_dbl_pert_01 = []
R_dbl_pert_11 = []
R_dbl_pert_20 = []

for i in R_sos:

    R_dbl_pert_00.extend((dbl_pert_expansion.expand_term(i, order_el=0, order_mech=0)))
    R_dbl_pert_10.extend((dbl_pert_expansion.expand_term(i, order_el=1, order_mech=0)))
    R_dbl_pert_01.extend((dbl_pert_expansion.expand_term(i, order_el=0, order_mech=1)))
    #R_dbl_pert_11.extend((dbl_pert_expansion.expand_term(i, order_el=1, order_mech=1)))
    #R_dbl_pert_20.extend((dbl_pert_expansion.expand_term(i, order_el=2, order_mech=0)))


new_terms_00 = []
for i in R_dbl_pert_00:
    #i.present()
    new_terms_00.extend(hermaut.do_hermaut(i, nm_inds))

print('len new terms 00', len(new_terms_00))
simplified_terms_00 = simplify.terms_simplify(new_terms_00, nm_inds)
print('simplified terms len 00', len(simplified_terms_00))


new_terms_10 = []
for i in R_dbl_pert_10:
    #i.present()
    new_terms_10.extend(hermaut.do_hermaut(i, nm_inds))

print('len new terms 10', len(new_terms_10))
simplified_terms_10 = simplify.terms_simplify(new_terms_10, nm_inds)
print('simplified terms len 10', len(simplified_terms_10))
print('simplified terms len 10', simplified_terms_10)

new_terms_01 = []
for i in R_dbl_pert_01:
    #i.present()
    new_terms_01.extend(hermaut.do_hermaut(i, nm_inds))

print('len new terms 01', len(new_terms_01))
simplified_terms_01 = simplify.terms_simplify(new_terms_01, nm_inds)
print('simplified terms len 01', len(simplified_terms_01))

for i in simplified_terms_10:
    print('New term')
    simplified_terms_10[i].present()


'''
new_terms_20 = []
for i in R_dbl_pert_20:
    #i.present()
    new_terms_20.extend(hermaut.do_hermaut(i, nm_inds))

print('len new terms 20', len(new_terms_20))
simplified_terms_20 = simplify.terms_simplify(new_terms_20, nm_inds)
print('simplified terms len 20', len(simplified_terms_20))

new_terms_11 = []
for i in R_dbl_pert_11:
    #i.present()
    new_terms_11.extend(hermaut.do_hermaut(i, nm_inds))

print('len new terms 11', len(new_terms_11))
simplified_terms_11 = simplify.terms_simplify(new_terms_11, nm_inds)
print('simplified terms len 11', len(simplified_terms_11))
'''
