import sys
sys.path.append('./utils')
import parseCFOUR as pc4

import numpy as np

class CFOURdata:

    def __init__(self, data):
        self.sourcetype = data['type']
        self.files = data['files']

    def getFundamentals(self):
        if self.sourcetype == 'out':
            f = pc4.getAnhFundamentals(self.files['out'], type='out')
            return f

        elif self.sourcetype == 'pkl':
            vibdatapkl = self.files['vibdata']
            import pickle
            with open(vibdatapkl, 'rb') as file:
                f = pickle.load(file)
            ff = pc4.getAnhFundamentals(f, type='pkl')
            return ff

    def getAllStates(self):
        if self.sourcetype == 'out':
            ls0, ls1, ls2, ls3, ls4 = pc4.pOut(self.files['out'])

        elif self.sourcetype == 'pkl':
            vibdatapkl = self.files['vibdata']
            import pickle
            with open(vibdatapkl, 'rb') as file:
                ls0, ls1, ls2, ls3, ls4 = pickle.load(file)

        ls0 = [pc4.modes_array2tuple(arr) for arr in ls0]

        combd = dict(zip(ls0, ls2))
        Delta = {tuple([o - 7 for o in k]): v for k, v in combd.items()}
        # Delta = {(k[0] if len(k) == 1 else k): v for k, v in Delta.items()}

        return Delta

    def getDipDers(self):
        if self.sourcetype == 'out':
            mu = pc4.getDipoleDers(self.files['dipolexyz'], self.files['out'])
            return mu

        elif self.sourcetype == 'pkl':
            dipolepkl = self.files['dipole']
            import pickle
            with open(dipolepkl, 'rb') as file:
                d = pickle.load(file)
            return d

    def getPolarDers(self):

        if self.sourcetype == 'out':
            dalpha, d2alpha = pc4.getPolarDers(self.files['polardir'])

        elif self.sourcetype == 'pkl':
            dipolepkl = self.files['polar']
            import pickle
            with open(dipolepkl, 'rb') as file:
                dalpha, d2alpha = pickle.load(file)

        return dalpha, d2alpha

    def getCFF(self):

        if self.sourcetype == 'out':
            cff = pc4.getCubicPost(self.files['out'], self.files['cubic'])
            return cff

        elif self.sourcetype == 'pkl':
            cubicpkl = self.files['cubic']
            import pickle
            with open(cubicpkl, 'rb') as file:
                cff = pickle.load(file)
            return cff

class VeloxChemdata:

    def __init__(self, data):
        pass

