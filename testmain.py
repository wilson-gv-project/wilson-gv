import main2DIR as dd_ir
import numpy as np

# Terms in expressions
electrical_terms = [('a+b,a', 'zero,a')
                   ,('b,a', 'zero,a')
                   ]
#
electric_avrg = [[ ('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_QQ', ('a', 'b',)) ]
                  ,[ ('mu_Q', ('a',)), ('alpha_QQ', ('a', 'b',)), ('mu_Q', ('b',)) ]
                ]
#
# mechanical_terms = [ [('a+b,a', 'zero,a'), ('a+b+c,zero', 'c,a+b')],
#                      #[('c,a', 'zero,a'), ('a+b,c', 'b+c,a')],
#                      #[('a+b,a', 'zero,a'), ('a,a+b', 'b,zero')],
#                      #[('b,a', 'zero,a'), ('b,a+b', 'a,zero')],
#                      #[('b,a', 'zero,a'), ('a,a+b', 'b,zero')],
#                      #[('b,a', 'zero,a'), ('b,a+b', 'a,zero')]
#                    ]
#
# mechanical_avrg = [[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',)), 'abc'],
#                     #[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('c',))],
#                     #[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('a',))],
#                     #[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',))],
#                     #[('mu_Q', ('a',)), ('alpha_Q', ('a',)), ('mu_Q', ('b',))],
#                     #[('mu_Q', ('a',)), ('alpha_Q', ('b',)), ('mu_Q', ('b',))]
#                   ]
#
# # electrical_terms, electric_avrg = [], []
mechanical_terms, mechanical_avrg = [], []

test1 = False
###    Test 1
if test1:
    # set up frequencies for x and y axes
    w1, w2 = np.arange(0., 120, 1.), np.arange(0, 120, 1.)
    funds = {'0': 20, '1': 30
                    , '2': 50
                  }

    # create class instance
    h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=10**(-0.003), avrg_ones=True)

    # get derivatives - mu_Q, mu QQ, alpha_Q, alphaQQ, F_abc
    derivData = h.getDerivs()

    # add mechanical and electrical anharmonicities terms and oriantational averages (symbolic setup)
    h.addTerms(electrical_terms, mechanical_terms, electric_avrg, mechanical_avrg)

    # pairs of normal mode indices
    coords_ab = dd_ir.get_abc(2, len(h.fundamentals))

    # plot (and save plot) 2D spectrum
    h.plot2D(coords_ab, Qabc=[], figname='test1', w1mw2=False, surface=False)


waterBool = True
###    Test 2 (H2O)
if waterBool:
    # set up frequencies for x and y axes
    w1, w2 = np.arange(1200., 3420, 15), np.arange(2300., 5320, 15)
    funds = dict(zip(['0', '1', '2'], [1775.31613305, 4176.50018401, 4267.11828147]))

    # create class instance
    h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=10**(-0.003), avrg_ones=False)

    # get derivatives - mu_Q, mu QQ, alpha_Q, alphaQQ, F_abc
    derivData = h.getDerivs(molfile='MOLECULE.INP', rspfile='rsp_tensor')

    # add mechanical and electrical anharmonicities terms and oriantational averages (symbolic setup)
    h.addTerms(electrical_terms, mechanical_terms, electric_avrg, mechanical_avrg)

    # pairs of normal mode indices
    coords_ab = dd_ir.get_abc(2, len(h.fundamentals))

    # plot (and save plot) 2D spectrum
    h.plot2D(coords_ab, Qabc=[], figname='h2o', w1mw2=False, surface=False)


acetonitrileBool = False
###    Test 2 (CH3CN)
if acetonitrileBool:
    # set up frequencies for x and y axes
    w1, w2 = np.arange(1200., 3420, 15), np.arange(2300., 5320, 15)
    funds = dict(zip(['0', '1', '2', '3', '4', '5'],
                     [1354.33983658, 1362.02594387, 1618.89707406, 1860.47711028, 2981.21568204, 3057.08908481]))

    # create class instance
    h = dd_ir.SpectrumEVV(w1, w2, funds, Gamma=10 ** (-0.003), avrg_ones=False)

    # get derivatives - mu_Q, mu QQ, alpha_Q, alphaQQ, F_abc from the rsp_tensor file
    # includes basis transformation
    dd = h.getDerivs(molfile='./ch3cn/MOLECULE.INP', rspfile='./ch3cn/rsp_tensor')

    # add mechanical and electrical anharmonicities terms and oriantational averages (symbolic setup)
    h.addTerms(electrical_terms, mechanical_terms, electric_avrg, mechanical_avrg)

    # pairs of normal mode indices
    coords_ab = dd_ir.get_abc(2, len(h.fundamentals))

    # plot (and save plot) 2D spectrum
    h.plot2D(coords_ab, Qabc=[], figname='ch3cn', w1mw2=False, surface=False)

