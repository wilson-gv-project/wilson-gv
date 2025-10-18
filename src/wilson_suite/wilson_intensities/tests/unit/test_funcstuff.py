"""
- [x] resonance location function 
- [x] vibenedif calculation
- [ ] vibenediff denominator
- [ ] averaged_props
- [ ] non_averaged_props
- [ ] vibene_denom
- [ ] resonance part
"""
from wilson_suite.wilson_intensities.spectrum import func_abstractions as f_abst
import json
import numpy as np

def test_ParameterSet():
    print()
    o1 = f_abst.ParameterSet(dict(a=3, b=4, c=4))
    o2 = f_abst.ParameterSet(dict(a=3, b=4, c=4))
    o3 = f_abst.ParameterSet(dict(a=1, b=4, c=4))

    assert o1.indices() == [3, 4, 4]
    assert o1.parameter_labels() == ['a', 'b', 'c']

    assert o1 == o2
    assert o1 != o3

    print({o1:2.3, o2:3.4, o3:4.3})
    
    json_str = json.dumps(o1.to_dict())
    print(json_str)
    
    assert True

def test_ResonanceWaveMatch():
    print()
    wm1 = f_abst.ResonanceWaveMatch({'1': -1, '2': 1})
    print(wm1)

def test_EvaluationTerm():
    print()
    # (('b,a', (-1, 2)), ('zero,a', (-1,)))
    rr1 = f_abst.ResonanceWaveMatch({'1': -1, '2': 1})
    rr2 = f_abst.ResonanceWaveMatch({'1': -1})
    
    r1 = f_abst.VibDiffSymbolic(left='b', right='a', wavematching=rr1)
    r2 = f_abst.VibDiffSymbolic(left='zero', right='a', wavematching=rr2)

    #  ('b,a+b', 'a,zero')
    vd1 = f_abst.VibDiffSymbolic(left='b', right='a+b')
    vd2 = f_abst.VibDiffSymbolic(left='a', right='zero')
    
    p1 = f_abst.MolPropertySymbolic(trivial_name='dipgrad', cart_axes=('B',), nm_indices=('a',))
    p2 = f_abst.MolPropertySymbolic(trivial_name='diphess', cart_axes=('G',), nm_indices=('a', 'b'))
    p3 = f_abst.MolPropertySymbolic(trivial_name='polgrad', cart_axes=('A', 'D'), nm_indices=('b',))
    p4 = f_abst.MolPropertySymbolic(trivial_name='cubic', cart_axes=(), nm_indices=('a', 'b', 'c'))

    avrgGroup = f_abst.GroupPropsSymbolic(props=(p1,p2,p3))
    nonavrgGroup = f_abst.GroupPropsSymbolic(props=(p4,))
    # grouped properties part
    allprops = f_abst.PropertiesGrouped(averaged=avrgGroup, non_averaged=nonavrgGroup)

    coeffs = f_abst.TermCoefficients(term_a=0.5, term_b=-1./8)
    anharmonicity = f_abst.AnharmonicLevelInfo(level=2, el_mech=(1,0))

    # grouped part involving vibrational states energies
    vibene1 = f_abst.VibEneSymbolic(resonances=(r1, r2), 
                                    energy_differences=(vd1, vd2),
                                    denominators=('a', 'b', 'c'))
    
    et = f_abst.EvaluationTerm(vib_structure=vibene1,
                               properties=allprops, 
                               coefficients=coeffs,
                               anharmonicity=anharmonicity)
    assert et.short_id == 'T001(1_0)'

    params = f_abst.ParameterSet({'a': '1', 'b': '3', 'zero': 'zero'})
    vibdata = f_abst.VibStatesData(allstates=(f_abst.VibState(s={}, state_label='1', e=1234.),
                                              f_abst.VibState(s={}, state_label='3', e=3644.),
                                              f_abst.VibState(s={}, state_label='zero', e=0.)))
    
    res = f_eval.solve_LSE_resonace(resonances=(r1, r2), parameters=params, vibdata=vibdata)

    res_point = f_abst.ResonancePoint(location=(res['w1'], res['w2']), 
                                      term_id=et.short_id, 
                                      parameters=params,
                                      factor_value=None,
                                      Gamma=3.14)
    print(res_point)

from wilson_suite.wilson_intensities.spectrum import func_evaluation as f_eval

def test_generate_LHS():
    rr1 = f_abst.ResonanceWaveMatch({'1': -1, '2': 1})
    rr2 = f_abst.ResonanceWaveMatch({'1': -1})
    
    r1 = f_abst.VibDiffSymbolic(left='b', right='a', wavematching=rr1)
    r2 = f_abst.VibDiffSymbolic(left='zero', right='a', wavematching=rr2)

    LHS = f_eval.generate_LHS(resonances=(r1, r2))

    assert np.all(LHS==np.array([[ 1., -1.], [ 1.,  0.]]))


def test_get_RHS():
    rr1 = f_abst.ResonanceWaveMatch({'1': -1, '2': 1})
    rr2 = f_abst.ResonanceWaveMatch({'1': -1})
    
    r1 = f_abst.VibDiffSymbolic(left='b', right='a', wavematching=rr1)
    r2 = f_abst.VibDiffSymbolic(left='zero', right='a', wavematching=rr2)

    params = f_abst.ParameterSet({'a': '3', 'b': '1', 'zero': 'zero'})
    vibdata = f_abst.VibStatesData(allstates=(f_abst.VibState(s={}, state_label='1', e=1234.),
                                              f_abst.VibState(s={}, state_label='3', e=3644.),
                                              f_abst.VibState(s={}, state_label='zero', e=0.)))

    RHS = f_eval.get_RHS(resonances=(r1, r2), parameters=params, vibdata=vibdata)
    
    assert np.all(RHS==np.array([2410.0, 3644.0]))

def test_solve_LSE_resonace():
    print()
    rr1 = f_abst.ResonanceWaveMatch({'1': -1, '2': 1})
    rr2 = f_abst.ResonanceWaveMatch({'1': -1})
    
    r1 = f_abst.VibDiffSymbolic(left='b', right='a', wavematching=rr1)
    r2 = f_abst.VibDiffSymbolic(left='zero', right='a', wavematching=rr2)

    params = f_abst.ParameterSet({'a': '1', 'b': '3', 'zero': 'zero'})
    vibdata = f_abst.VibStatesData(allstates=(f_abst.VibState(s={}, state_label='1', e=1234.),
                                              f_abst.VibState(s={}, state_label='3', e=3644.),
                                              f_abst.VibState(s={}, state_label='zero', e=0.)))

    res = f_eval.solve_LSE_resonace(resonances=(r1, r2), parameters=params, vibdata=vibdata)

    assert res == {'w1': np.float64(1234.0), 'w2': np.float64(3644.0)}

def test_generate_LHS_motif():
    print()
    motif1 = (((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))
    motif2 = (((('a', 'b'), ('a',)), ('A',)),)
    motif3 = (((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))
    motif4 = (((('',), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))
    
    from ...spectrum.func_evaluation import generate_LHS_motif
    
    r1 = generate_LHS_motif(motif=motif1)
    assert np.allclose(r1, np.array([[-1.,  0.], [ 0., -1.]]))
    
    r2 = generate_LHS_motif(motif=motif2)
    assert np.allclose(r2, np.array([[-1.]])) # ???

    r3 = generate_LHS_motif(motif=motif3)
    assert np.allclose(r3, np.array([[ 0., -1.], [ -1.,  1.]]))

    r4 = generate_LHS_motif(motif=motif4)
    print(r4)

def test_generate_RHS_motif():
    motif1 = (((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))
    motif2 = (((('a', 'b'), ('a',)), ('A',)),)
    motif3 = (((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))
    
    # (res_cond1, res_cond2, res_cond3, ...)
    # (res_cond1, (wibdiff_mn, axes), ...)
    # (res_cond1, ((m_inds, n_inds), (ax1, ax2, ax3, ...)), ...)
    # (res_cond1, (((m1, m2, ...), (n1, n2, ...)), (ax1, ax2, ax3, ...)), ...)

    # vibdiff: (m_inds, n_inds) <=== ((m1, m2, ...), (n1, n2, ...))
    motif4 = (((('',), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))
    
    from ...spectrum.func_evaluation import get_RHS_motif
    params = f_abst.ParameterSet({'a': '1', 'b': '3', 'zero': 'zero'})
    vibdata = f_abst.VibStatesData(allstates=(f_abst.VibState(s={}, state_label='1', e=1234.),
                                              f_abst.VibState(s={}, state_label='3', e=3644.),
                                              f_abst.VibState(s={}, state_label='1+3', e=4164.),
                                              f_abst.VibState(s={}, state_label='zero', e=0.)))
    
    r1 = get_RHS_motif(motif=motif1, parameters=params, vibdata=vibdata, unit='cm-1')
    print(r1, motif1, params)
    assert np.all(r1==np.array([-2930.0, -2410.0]))
    
    r2 = get_RHS_motif(motif=motif2, parameters=params, vibdata=vibdata, unit='cm-1')
    assert np.all(r2==np.array([-2930.0]))

    r3 = get_RHS_motif(motif=motif3, parameters=params, vibdata=vibdata, unit='cm-1')
    assert np.all(r3==np.array([1234.0, 1234.0]))

    r4 = get_RHS_motif(motif=motif4, parameters=params, vibdata=vibdata, unit='cm-1')
    assert np.all(r4==np.array([1234.0, -2410.0]))


def test_solve_LSE_motif():
    motif1 = (((('a', 'b'), ('a',)), ('A',)), ((('b',), ('a',)), ('B',)))
    motif2 = (((('a', 'b'), ('a',)), ('A',)),)
    motif3 = (((('',), ('a',)), ('B',)), ((('',), ('a',)), ('A', '-B')))
    motif4 = (((('',), ('a',)), ('B',)), ((('b',), ('a',)), ('B',)))

    from ...spectrum.func_evaluation import solve_LSE_motif
    params = f_abst.ParameterSet({'a': '1', 'b': '3', 'zero': 'zero'})
    vibdata = f_abst.VibStatesData(allstates=(f_abst.VibState(s={}, state_label='1', e=1234.),
                                              f_abst.VibState(s={}, state_label='3', e=3644.),
                                              f_abst.VibState(s={}, state_label='1+3', e=4164.),
                                              f_abst.VibState(s={}, state_label='zero', e=0.)))
    print()
    r1 = solve_LSE_motif(motif=motif1, parameters=params, vibdata=vibdata, unit='cm-1')
    print(r1, motif1, params)

    r2 = solve_LSE_motif(motif=motif2, parameters=params, vibdata=vibdata, unit='cm-1')
    print(r2, motif2, params)

    r3 = solve_LSE_motif(motif=motif3, parameters=params, vibdata=vibdata, unit='cm-1')
    print(r3, motif3, params)

    r4 = solve_LSE_motif(motif=motif4, parameters=params, vibdata=vibdata, unit='cm-1')
    print(r4, motif4, params)

