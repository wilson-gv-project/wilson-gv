from scriptsHPC.utils import parseCFOUR
import numpy as np

class CFOURdata:

    def __init__(self, data: dict[str:[str, dict]]):
        self.sourcetype = data['type']
        self.files = data['files']

    def getFundamentals(self) -> dict[int:float]:
        """
        Fundamental frequency with anharmonic corrections
        Returns: dict[int:float]
        """
        if self.sourcetype == 'out':
            fundamentals = parseCFOUR.get_anharmonic_fundamentals(self.files['out'], filetype='out')
            return fundamentals

        elif self.sourcetype == 'pkl':
            fundamentals = parseCFOUR.get_anharmonic_fundamentals(self.files['vibdata'], filetype='pkl')
            return fundamentals

    def getAllStates(self) -> dict[tuple[int]: float, tuple[int, int]: float,
                                   tuple[int, int, int]: float]:
        """
        Dictionary of all the states and their frequencies
        Return: dict[tuple[int]: float, tuple[int, int]: float, tuple[int, int, int]: float]
        """
        if self.sourcetype == 'out':
            ls0, ls1, ls2, ls3, ls4 = parseCFOUR.parse_output_file(self.files['out'])

        elif self.sourcetype == 'pkl':
            vibdatapkl = self.files['vibdata']
            import pickle
            with open(vibdatapkl, 'rb') as file:
                ls0, ls1, ls2, ls3, ls4 = pickle.load(file)

        combd = dict(zip(ls0, ls2))
        Delta = {tuple([o - 7 for o in k]): v for k, v in combd.items()}
        # Delta = {(k[0] if len(k) == 1 else k): v for k, v in Delta.items()}

        return Delta

    def getDipDers(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Dipole derivatives: first order and second order
        Return: tuple[np.ndarray - shape(NM, 3), np.ndarray - shape(NM, NM, 3)]
        """
        if self.sourcetype == 'out':
            mu = parseCFOUR.getDipoleDers(self.files['dipolexyz'], self.files['out'])
            return mu

        elif self.sourcetype == 'pkl':
            dipolepkl = self.files['dipole']
            import pickle
            with open(dipolepkl, 'rb') as file:
                d = pickle.load(file)
            return d

    def getPolarDers(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Polarizability derivatives: first order and second order
        Return: tuple[np.ndarray - shape(NM, 3, 3), np.ndarray - shape(NM, NM, 3, 3)]
        """
        if self.sourcetype == 'out':
            dalpha, d2alpha = parseCFOUR.getPolarDers(self.files['polardir'])

        elif self.sourcetype == 'pkl':
            dipolepkl = self.files['polar']
            import pickle
            with open(dipolepkl, 'rb') as file:
                dalpha, d2alpha = pickle.load(file)

        return dalpha, d2alpha

    def getCFF(self) -> np.ndarray:
        """
        CFF: cubic force constant tensor
        Return: np.ndarray - shape(NM, NM, NM)
        """
        if self.sourcetype == 'out':
            cff = parseCFOUR.getCubicPost(self.files['out'], self.files['cubic'])
            return cff

        elif self.sourcetype == 'pkl':
            cubicpkl = self.files['cubic']
            import pickle
            with open(cubicpkl, 'rb') as file:
                # first 3 columns are the normal mode indices, the last column holds the derivatives
                cff = pickle.load(file)

        freq = self.getFundamentals()
        cubicFC = parseCFOUR.getCubicPost(freq, cff)

        return cubicFC

class VeloxChemdata:

    def __init__(self, data):
        pass

