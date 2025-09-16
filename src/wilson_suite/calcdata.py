from .wilson_main import abstractions as wm_abst
import numpy as np


mol1 = wm_abst.MolecularSystem(name='mol1', natoms=3)
mol2 = wm_abst.MolecularSystem(name='mol2', natoms=5)

setup1 = wm_abst.ExternalCalcSetup('p1', 'lvl1', 'b1')
setup2 = wm_abst.ExternalCalcSetup('p1', 'lvl2', 'b2')
setup3 = wm_abst.ExternalCalcSetup('p1', 'lvl2', 'b3')
setup4 = wm_abst.ExternalCalcSetup('p2', 'lvl3', 'b2')

np3b3 = np.array([[0.48, 0.53, 0.52],
                [0.42, 0.81, 0.47],
                [0.23, 0.66, 0.8 ]])
np1b3 = np.array([[0.81, 0.51, 0.3 ]])
np3b3b3 = np.array([[[0.21, 0.44, 0.16],
                    [0.96, 0.98, 0.43],
                    [0.69, 0.5 , 0.05]],

                    [[0.21, 0.68, 0.11],
                    [0.55, 0.22, 0.61],
                    [0.34, 0.11, 0.32]],
                    
                    [[0.2 , 0.63, 0.52],
                    [0.95, 0.49, 0.22],
                    [0.17, 0.84, 0.27]]])
np3b3b3b3 = np.array([[[[0.93, 0.84, 0.13],
                        [0.96, 0.52, 0.5 ],
                        [0.38, 0.71, 0.16]],

                        [[0.38, 0.35, 0.03],
                        [0.49, 0.54, 0.47],
                        [0.56, 0.95, 0.25]],

                        [[0.47, 0.81, 0.13],
                        [0.12, 0.6 , 0.97],
                        [0.55, 0.15, 0.05]]],

                    [[[0.89, 0.13, 0.08],
                        [0.48, 0.45, 0.14],
                        [0.33, 0.15, 0.78]],

                        [[0.38, 0.6 , 0.82],
                        [0.36, 0.64, 0.58],
                        [0.83, 0.52, 0.05]],

                        [[0.14, 0.89, 0.69],
                        [0.88, 0.95, 0.64],
                        [0.21, 0.14, 0.4 ]]],

                    [[[0.03, 0.59, 0.85],
                        [0.82, 0.2 , 0.09],
                        [0.14, 0.37, 0.36]],

                        [[0.23, 0.25, 0.  ],
                        [0.43, 0.9 , 0.47],
                        [0.47, 0.37, 0.35]],

                        [[0.3 , 0.8 , 0.54],
                        [0.18, 0.08, 0.38],
                        [0.57, 0.22, 0.06]]]])


datadict1 = {'system': mol1, 'calc_setup': setup1, 
            'B': (np1b3, None, 'cm-1'), 'coriolis': (np3b3, 'bu', 'cm-1'),
            'hess': (np3b3, 'bu', 'cm-1'), 'cff': (np3b3b3, 'bu', 'cm-1'), 'qff': (np3b3b3b3, 'bu', 'cm-1'), 
            'dipgrad': (np3b3, 'bu', 'cm-1'), 'diphess': (np3b3b3, 'bu', 'cm-1'), 
            'polgrad': (np3b3b3, 'bu', 'cm-1'), 'polhess': (np3b3b3b3, 'bu', 'cm-1'),
            'harmonic_states': {(3,):4, (5,):2, (6,):46}, 'anharmonic_states': {(3,):14, (5,):32, (6,):96}}
