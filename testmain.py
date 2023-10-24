###########################################################
##                                                       ##
##    Tests : 2DIR spectra calculation examples          ##
##                                                       ##
###########################################################

import os
import sys

# project_root = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(project_root[:-7])
# print(project_root[:-7])

import main2DIR as dd_ir
import numpy as np


# Terms in expressions
electrical_terms = [('a+b,a', 'zero,a'), ('b,a', 'zero,a') ]

# derivatives:
# 1. mu_Q, mu QQ, alpha_Q - electric dipole (1st and 2nd derivatives), polarizability (1st der.)
# 2. mu_Q, alpha_QQ - electric dipole (1st der.), polarizability (2nd der.)
electric_avrg = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',))],
                 [('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',))] ]

mechanical_terms = [[('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')],
                    [('c,a', 'zero,a'), ('a+b,c', 'b+c,a')],
                    [('a+b,a', 'zero,a'), ('a,a+b', 'b,zero')],
                    [('b,a', 'zero,a'), ('b,a+b', 'a,zero')],
                    [('b,a', 'zero,a'), ('a,a+b', 'b,zero')],
                    [('b,a', 'zero,a'), ('b,a+b', 'a,zero')] ]

# derivatives:
# mu_Q, alpha_Q - for all 6 terms
mechanical_avrg = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',)), 'bcc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',)), 'bcc'],
                   [('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',)), 'acc'] ]


def picks(pool, listofinds):
    return [pool[i] for i in listofinds]


printDerivs = False

test1 = False
test11 = False
ee1, mm1 = [0, 1], []

test22 = False
test222 = True
# ee1, mm1 = [0, 1], [0]

test2 = False
waterBool = False
acetonitrileBool = False

w1mw2 = False

np.set_printoptions(suppress=True, linewidth=150)

###    Test 1
if test1:
    # todo 3 + todo 2: should be coming from SpectroscPy
    # funds = dict(zip(['0', '1', '2'], [25., 40., 85.]))
    funds = dict(zip(['0', '1'], [25., 40.]))
    delta = {'00': 47.0, '01': 57., '11': 77., '000': 65.0, '001': 90.0, '011': 105.0, '110': 105.0, '111': 120.0}
    # set up frequencies for x and y axes: # todo 1 (starting point for rendering)
    # w1, w2 = np.arange(0., 110, 1.), np.arange(0, 110, 1.)

    a = list(funds.values())
    import itertools
    template_w = list(set([i[0]+i[1] for i in list(itertools.product(a, a))]))
    template_w.append(0.)
    template_w.append(max(template_w)+5.)
    template_w = sorted(template_w+[25., 40.])
    w1, w2 = [25.]*len(template_w), template_w.copy()
    w1 = [25., 25., 25., 40., 40., 40.]
    w2 = [40., 47., 57., 25., 57., 77.]

    w1 = [25., 25., 40., 40.]
    w2 = [47., 57., 57., 77.]

    print('w1', w1)
    print('w2', w2)

    # print(w1)
    # create class instance: initialize a 2dir spectrum instance
    # h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=10 ** (-0.003), avrg_ones=True)  # gamma 0.9931160484209338
    h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=0.00000001, avrg_ones=True, Delta=delta)  # gamma 0.9931160484209338
    h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=10 ** (-0.003), avrg_ones=True, Delta=delta)  # gamma 0.9931160484209338
    h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=1., avrg_ones=True, Delta=delta)  # gamma 0.9931160484209338

    # get derivatives - mu_Q, mu QQ, alpha_Q, alphaQQ, F_abc
    # todo: setup PyOpenrsp calculations (should be more flexible)
    derivData = h.getDerivs()
    print(derivData) if printDerivs else None

    # selection of terms by index in the complete list
    ee, mm = ee1, mm1
    # ee, mm = [0, 1], [0, 1]
    # ee, mm = [0, 1], [0, 1, 2, 3, 4, 5]

    # add mechanical and electrical anharmonicities terms and orientational averages (symbolic setup)
    h.addTerms(picks(electrical_terms, ee), picks(mechanical_terms, mm),
               picks(electric_avrg, ee), picks(mechanical_avrg, mm))

    # plot (and save plot) 2D spectrum
    mech = 'mechNone' if not picks(mechanical_terms, mm) else f'mech{len(picks(mechanical_terms, mm))}'
    elec = 'elecNone' if not picks(electrical_terms, ee) else f'elec{len(picks(electrical_terms, ee))}'
    test1_name = f'test1_{mech}_{mm}_{elec}_{ee}' if h.coords_abc is not None else f'test1_{elec}_{ee}'
    if w1mw2: test1_name += '_w1mw2'

    print(h.totInt(style='scatter'))
    h.plot2D(figname=test1_name, w1mw2=w1mw2, style='scatter')

if test11:
    # todo 3 + todo 2: should be coming from SpectroscPy
    funds = dict(zip(['0', '1', '2'], [25., 40., 85.]))
    funds = dict(zip(['0', '1'], [25., 40.]))
    delta = {'00': 47.0, '01': 57., '11': 77., '000': 65.0, '001': 90.0, '011': 105.0, '110': 105.0, '111': 120.0}

    # set up frequencies for x and y axes: # todo 1 (starting point for rendering)
    w1, w2 = np.arange(15., 50, .2), np.arange(15, 80, .2)

    # create class instance: initialize a 2dir spectrum instance
    # h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=10 ** (-0.003), avrg_ones=True)  # gamma 0.9931160484209338
    h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=1., avrg_ones=True, Delta=delta)  # gamma 0.9931160484209338

    # get derivatives - mu_Q, mu QQ, alpha_Q, alphaQQ, F_abc
    # todo: setup PyOpenrsp calculations (should be more flexible)
    derivData = h.getDerivs()
    print(derivData) if printDerivs else None

    # selection of terms by index in the complete list
    ee, mm = ee1, mm1
    # ee, mm = [0, 1], [0, 1]
    # ee, mm = [0, 1], [0, 1, 2, 3, 4, 5]

    # add mechanical and electrical anharmonicities terms and orientational averages (symbolic setup)
    h.addTerms(picks(electrical_terms, ee), picks(mechanical_terms, mm),
               picks(electric_avrg, ee), picks(mechanical_avrg, mm))

    # plot (and save plot) 2D spectrum
    mech = 'mechNone' if not picks(mechanical_terms, mm) else f'mech{len(picks(mechanical_terms, mm))}'
    elec = 'elecNone' if not picks(electrical_terms, ee) else f'elec{len(picks(electrical_terms, ee))}'
    test1_name = f'test11_{mech}_{mm}_{elec}_{ee}' if h.coords_abc is not None else f'test1_{elec}_{ee}'
    if w1mw2: test1_name+='_w1mw2'

    # print(h.totInt())
    h.plot2D(figname=test1_name, w1mw2=w1mw2, style='contour')


if test22:
    # todo 3 + todo 2: should be coming from SpectroscPy
    # funds = dict(zip(['0', '1', '2'], [25., 40., 85.]))
    funds = dict(zip(['0', '1'], [858., 2309.]))
    delta = {'00': 1713.15, '01': 3162.61, '11': 4615.27, '000': 65.0, '001': 90.0, '011': 105.0, '110': 105.0, '111': 120.0}
    # set up frequencies for x and y axes: # todo 1 (starting point for rendering)
    # w1, w2 = np.arange(0., 110, 1.), np.arange(0, 110, 1.)

    a = list(funds.values())
    import itertools
    template_w = list(set([i[0]+i[1] for i in list(itertools.product(a, a))]))
    template_w.append(0.)
    template_w.append(max(template_w)+5.)
    template_w = sorted(template_w+[25., 40.])
    w1, w2 = [25.]*len(template_w), template_w.copy()
    w1 = [25., 25., 25., 40., 40., 40.]
    w2 = [40., 47., 57., 25., 57., 77.]

    w1 = [25., 25., 40., 40.]
    w2 = [47., 57., 57., 77.]

    print('w1', w1)
    print('w2', w2)

    # print(w1)
    # create class instance: initialize a 2dir spectrum instance
    # h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=10 ** (-0.003), avrg_ones=True)  # gamma 0.9931160484209338
    h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=0.00000001, avrg_ones=True, Delta=delta)  # gamma 0.9931160484209338
    h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=10 ** (-0.003), avrg_ones=True, Delta=delta)  # gamma 0.9931160484209338
    h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=1., avrg_ones=True, Delta=delta)  # gamma 0.9931160484209338

    # get derivatives - mu_Q, mu QQ, alpha_Q, alphaQQ, F_abc
    # todo: setup PyOpenrsp calculations (should be more flexible)
    derivData = h.getDerivs()
    print(derivData) if printDerivs else None

    # selection of terms by index in the complete list
    ee, mm = ee1, mm1
    # ee, mm = [0, 1], [0, 1]
    # ee, mm = [0, 1], [0, 1, 2, 3, 4, 5]

    # add mechanical and electrical anharmonicities terms and orientational averages (symbolic setup)
    h.addTerms(picks(electrical_terms, ee), picks(mechanical_terms, mm),
               picks(electric_avrg, ee), picks(mechanical_avrg, mm))

    # plot (and save plot) 2D spectrum
    mech = 'mechNone' if not picks(mechanical_terms, mm) else f'mech{len(picks(mechanical_terms, mm))}'
    elec = 'elecNone' if not picks(electrical_terms, ee) else f'elec{len(picks(electrical_terms, ee))}'
    test22_name = f'test22_{mech}_{mm}_{elec}_{ee}' if h.coords_abc is not None else f'test1_{elec}_{ee}'
    if w1mw2: test22_name += '_w1mw2'

    print(h.totInt(style='scatter'))
    h.plot2D(figname=test22_name, w1mw2=w1mw2, style='scatter')

if test222:
    import time
    c0 = time.process_time()


    # todo 3 + todo 2: should be coming from SpectroscPy
    # funds = dict(zip(['0', '1', '2'], [25., 40., 85.]))
    # funds = dict(zip(['0', '1'], [25., 40.]))
    funds = dict(zip(['0', '1'], [858., 2309.]))

    delta = {'00': 1713.15, '01': 3162.61, '11': 4615.27, '000': 65.0, '001': 90.0, '011': 105.0, '110': 105.0, '111': 120.0}

    # set up frequencies for x and y axes: # todo 1 (starting point for rendering)
    w1, w2 = np.arange(800.2, 2327.3, 1.2), np.arange(800.2, 3192.6, 1.)
    # w1 = [2309., 2309., 2309.]
    # w2 = [2309., 4615.27, 3162.61]
    # w1 = [2309., 2309.]
    # w2 = [2319., 3162.61]

    # create class instance: initialize a 2dir spectrum instance
    # h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=10 ** (-0.003), avrg_ones=True, Delta=delta)  # gamma 0.9931160484209338
    h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=100., avrg_ones=True, Delta=delta)  # gamma 0.9931160484209338

    # get derivatives - mu_Q, mu QQ, alpha_Q, alphaQQ, F_abc
    # todo: setup PyOpenrsp calculations (should be more flexible)
    derivData = h.getDerivs()
    print(derivData) if printDerivs else None

    # selection of terms by index in the complete list
    ee, mm = ee1, mm1
    # ee, mm = [0, 1], [0, 1]
    # ee, mm = [0, 1], [0, 1, 2, 3, 4, 5]

    # add mechanical and electrical anharmonicities terms and orientational averages (symbolic setup)
    h.addTerms(picks(electrical_terms, ee), picks(mechanical_terms, mm),
               picks(electric_avrg, ee), picks(mechanical_avrg, mm))

    # plot (and save plot) 2D spectrum
    mech = 'mechNone' if not picks(mechanical_terms, mm) else f'mech{len(picks(mechanical_terms, mm))}'
    elec = 'elecNone' if not picks(electrical_terms, ee) else f'elec{len(picks(electrical_terms, ee))}'
    test222_name = f'test222_{mech}_{mm}_{elec}_{ee}' if h.coords_abc is not None else f'test1_{elec}_{ee}'
    if w1mw2: test222_name+='_w1mw2'

    # print(h.totInt())

    c1 = time.process_time()
    print('before h.plot2D', c1-c0)

    h.plot2D(figname=test222_name, w1mw2=w1mw2, style='contour')



###    Test 2 (calling pyopenrsp for props)
if test2:

    # set up frequencies for x and y axes: # todo 1 (starting point for rendering)
    w1, w2 = np.arange(0., 120, 1.), np.arange(0, 120, 1.)

    # todo 3 + todo 2: should be coming from SpectroscPy/VeloxChem
    funds = dict(zip(['0', '1', '2'], [20., 30., 50.]))

    # create class instance: initialize a 2dir spectrum instance
    h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=10 ** (-0.003), avrg_ones=True)

    # get derivatives - mu_Q, mu QQ, alpha_Q, alphaQQ, F_abc
    # todo: setup PyOpenrsp calculations (should be more flexible)
    derivData = h.getDerivs(source='pyorsp')
    print(derivData) if printDerivs else None

    # selection of terms by index in from complete lists (mechanical_terms, electric_terms)
    ee, mm = [0, 1], [0]

    # add mechanical and electrical anharmonicities terms and orientational averages (symbolic setup)
    h.addTerms(picks(electrical_terms, ee), picks(mechanical_terms, mm),
               picks(electric_avrg, ee), picks(mechanical_avrg, mm))

    # plot (and save plot) 2D spectrum
    mech = 'mechNone' if not picks(mechanical_terms, mm) else f'mech{len(picks(mechanical_terms, mm))}'
    elec = 'elecNone' if not picks(electrical_terms, ee) else f'elec{len(picks(electrical_terms, ee))}'
    test1_name = f'test1_{mech}_{mm}_{elec}_{ee}' if h.coords_abc is not None else f'test1_{elec}_{ee}'
    h.plot2D(figname=test1_name, w1mw2=False, surface=False)


###    Test 2 (H2O)
if waterBool:

    # set up frequencies for x and y axes
    w1, w2 = np.arange(1200., 3420, 15), np.arange(2300., 5320, 15)
    # should be a routine to get these numbers from vlx
    funds = dict(zip(['0', '1', '2'], [1775.31613305, 4176.50018401, 4267.11828147]))

    # create class instance
    h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=10 ** (-0.003), avrg_ones=False)

    # get derivatives - mu_Q, mu QQ, alpha_Q, alphaQQ, F_abc
    derivData = h.getDerivs(molfile='MOLECULE.INP', rspfile='rsp_tensor')
    print(derivData) if printDerivs else None

    ee, mm = [0, 1], [0]
    # add mechanical and electrical anharmonicities terms and orientational averages (symbolic setup)
    h.addTerms(picks(electrical_terms, ee), picks(mechanical_terms, mm),
               picks(electric_avrg, ee), picks(mechanical_avrg, mm))

    # plot (and save plot) 2D spectrum
    mech = 'mechNone' if not picks(mechanical_terms, mm) else f'mech{len(picks(mechanical_terms, mm))}'
    elec = 'elecNone' if not picks(electrical_terms, ee) else f'elec{len(picks(electrical_terms, ee))}'
    h2o_name = f'h2o_{mech}_{mm}_{elec}_{ee}' if h.coords_abc is not None else f'h2o_{elec}_{ee}'

    h.plot2D(figname=h2o_name, w1mw2=False, surface=False)


###    Test 3 (CH3CN)
if acetonitrileBool:

    # set up frequencies for x and y axes
    w1, w2 = np.arange(1200., 3420, 15), np.arange(2300., 5320, 15)
    funds = dict(zip(['0', '1', '2', '3', '4', '5'],
                     [1354.33983658, 1362.02594387, 1618.89707406, 1860.47711028, 2981.21568204, 3057.08908481]))

    # create class instance
    h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=10 ** (-0.003), avrg_ones=False)

    # get derivatives - mu_Q, mu QQ, alpha_Q, alphaQQ, F_abc from the rsp_tensor file
    # includes basis transformation
    derivData = h.getDerivs(molfile='./ch3cn/MOLECULE.INP', rspfile='./ch3cn/rsp_tensor')
    print(derivData) if printDerivs else None

    ee, mm = [0, 1], [0, 1, 2, 3, 4, 5]
    # add mechanical and electrical anharmonicities terms and orientational averages (symbolic setup)
    h.addTerms(picks(electrical_terms, ee), picks(mechanical_terms, mm),
               picks(electric_avrg, ee), picks(mechanical_avrg, mm))

    # plot (and save plot) 2D spectrum
    mech = 'mechNone' if not picks(mechanical_avrg, mm) else f'mech{len(picks(mechanical_avrg, mm))}'
    elec = 'elecNone' if not picks(electrical_terms, ee) else f'elec{len(picks(electrical_terms, ee))}'
    ch3cn_name = f'ch3cn_{mech}_{mm}_{elec}_{ee}' if h.coords_abc is not None else f'ch3cn_{elec}_{ee}'
    h.plot2D(figname=ch3cn_name, w1mw2=False, surface=False)



def funplot():
    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
    plt.show()