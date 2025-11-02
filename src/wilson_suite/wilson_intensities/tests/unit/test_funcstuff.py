"""
- [x] resonance location function 
- [x] vibenedif calculation
- [ ] vibenediff denominator
- [ ] averaged_props
- [ ] non_averaged_props
- [ ] vibene_denom
- [ ] resonance part
"""
from wilson_suite.wilson_intensities.amplitudes.term_parts import ParameterSet, VibStatesData
from wilson_suite.wilson_intensities.amplitudes import func_abstractions as f_abst
from wilson_suite.wilson_intensities.amplitudes.resonances import solve_LSE_motif
import json

import wilson_suite.wilson_main.abstractions

def test_ParameterSet():
    print()
    o1 = ParameterSet(dict(a=3, b=4, c=4))
    o2 = ParameterSet(dict(a=3, b=4, c=4))
    o3 = ParameterSet(dict(a=1, b=4, c=4))
    print(o1.indices())
    print(o1.parameter_labels())
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

    params = ParameterSet({'a': '1', 'b': '3', 'zero': 'zero'})
    vibdata = VibStatesData(allstates=(wilson_suite.wilson_main.abstractions.VibState(harm_quanta_coeffs={}, state_label='1', energy=1234.),
                                       wilson_suite.wilson_main.abstractions.VibState(harm_quanta_coeffs={}, state_label='3', energy=3644.),
                                       wilson_suite.wilson_main.abstractions.VibState(harm_quanta_coeffs={}, state_label='zero', energy=0.)),
                                   harmonic_osc_states_labels=(1, 3))
    
    # res = solve_LSE_motif(resonances=(r1, r2), parameters=params, vibdata=vibdata)

    # res_point = f_abst.ResonancePoint(location=(res['w1'], res['w2']), 
    #                                   term_id=et.short_id, 
    #                                   parameters=params,
    #                                   factor_value=None,
    #                                   Gamma=3.14)
    # print(res_point)

