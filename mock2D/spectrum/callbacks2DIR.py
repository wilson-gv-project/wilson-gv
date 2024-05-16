from scriptsHPC.utils import parseCFOUR
from scriptsHPC.utils import parseGaussian

import numpy as np
np.set_printoptions(linewidth=250, suppress=False, precision=12)

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
            cubic = parseCFOUR.pCubicORQuartic(self.files['cubic'])
            freq = self.getFundamentals()
            cff = parseCFOUR.getCubicPost(freq, cubic)
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

def str_einsum(origstr, same_ind, lenshape):
    origstr = origstr[:lenshape]
    neworigstr = origstr[:same_ind] + 'q' + origstr[same_ind + 1:]
    return origstr + f',{origstr[same_ind]}q->' + neworigstr

def getDimensionlessNM(datafile: str = None) -> dict:
    """
    Reduced (dimensionless) normal coordinates
    Return: a transformation matrix with dimensionless normal coordinates
    """

    if datafile[-3:] == 'pkl':
        import pickle
        with open(datafile, 'rb') as file:
            # first 3 columns are the normal mode indices, the last column holds the derivatives
            undisplaced_matrix, dimless, freqs = pickle.load(file)
        return dimless

    else:
        undisplaced_matrix, freqs, dimless  = parseCFOUR.pQUADRATURE(datafile)
        # print(dimless)
        return dimless

def cart2normalGen(tensor, transfMatrix, geoDims):

    import copy
    new_tensor = copy.deepcopy(tensor)
    sh = new_tensor.shape

    for d in geoDims:
        einstr = str_einsum('ijkl', d, len(sh))
        print(einstr)
        new_tensor = np.einsum(einstr, new_tensor, transfMatrix)

    return new_tensor
def tensors2dimlessNMbasis(prop, geoDims, dimlessFile: str = None) -> np.ndarray:
    """

    :param dimlessFile:
    :return:
    """
    # a dictionary is returned
    rr = getDimensionlessNM(dimlessFile)
    # print(rr)
    mass_weighted_eigenvectors = np.concatenate([i.reshape(-1, 1) for i in rr.values()], axis=1)
    # print(mass_weighted_eigenvectors, mass_weighted_eigenvectors.shape)

    transformed = cart2normalGen(prop, mass_weighted_eigenvectors, geoDims)

    return transformed


class VeloxChemdata:

    def __init__(self, data: dict[str:[str, dict]]):
        self.sourcetype = data['type']
        self.files = data['files']


class LSDaltondata:

    def __init__(self, data: dict[str:[str, dict]]):
        self.sourcetype = data['type']
        self.files = data['files']
        self.props = None


    def getTensors(self):
        """
        Adding tensors to self.props
            hessianProp = props_list[-1]
            hessian_tensor = hessianProp.tensor
        :return:
        """
        from mock2D.fromspectroscpy import openrsp_tensor_reader

        props_list, tens_list = openrsp_tensor_reader.read_openrsp_tensor_file(self.files['rsp_tensor'])
        # print(f'There are {len(props_list)} in this rsp_tensor file')

        for i in range(len(props_list)):
            if props_list[i].operator == ['GEO', 'GEO', 'GEO']:
                # print('LDALTON CFF')
                # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
                # print(repr(tens_list[i]))
                # make in cm-1 for comparison with CFOUR
                props_list[i].addTensor(tens_list[i])
            else:
                props_list[i].addTensor(tens_list[i])
            # props_list[i].tellProp()

        self.props = props_list[:-1]
        # print('\nType self.props is', type(self.props), '\n')
        # for kk in self.props:
        #     print(repr(kk.tensor))

        return props_list

    def get_hessian_tensor(self) -> np.ndarray:
        """Getting Hessian from provided rsp_tensor file"""
        from mock2D.fromspectroscpy import openrsp_tensor_reader

        props, tensors = openrsp_tensor_reader.read_openrsp_tensor_file(self.files['rsp_tensor_hessian'])

        for i, p in enumerate(props):
            if p.operator == ['GEO', 'GEO']:
                p.addTensor(tensors[i])
                return p.tensor

    def vibrational_analysis(self) -> tuple:
        """
        Fundamental frequency with anharmonic corrections
        Returns: dict[int:float]
        """
        from mock2D.fromspectroscpy import vib_analysis
        coordshere, chargeshere, masseshere = vib_analysis.read_mol(self.files['moleculeinp'])
        # print('coordshere', coordshere, type(coordshere))
        # print('masseshere', masseshere, type(masseshere))

        hes = self.get_hessian_tensor()
        # print('hes from rsp_tensor\n', hes, type(hes), hes.shape)

        from scriptsHPC.vibanalysis import jonas
        mol = jonas.Molecule(np.array(coordshere), np.array(masseshere))
        newnewhess = jonas.project_hessian(mol, hes, is_linear=False, internal_coordinates=False)
        # print(newnewhess[0], type(newnewhess[0]), newnewhess[0].shape)

        eigenvalues, mass_weighted_eigenvectors = jonas.hessian_diagonalizer(newnewhess[0])
        # eigenvalues, mass_weighted_eigenvectors = jonas.hessian_diagonalizer(hes)
        eigenvalues = eigenvalues[:-6]
        mass_weighted_eigenvectors = mass_weighted_eigenvectors[:, :-6]
        eigenvalues = jonas.hartree_amu_bohr_2_wavenumbers(eigenvalues)
        # print('#####################_________')
        # print(mass_weighted_eigenvectors)

        sqrtmmm = np.repeat(np.sqrt(np.array(masseshere)), 3)
        sqrtmmminv = np.divide(1.0, sqrtmmm)
        # Reshape sqrtmmminv to have shape (12, 1) for broadcasting
        sqrtmmminv_reshaped = sqrtmmminv[:, np.newaxis]
        # Now you can multiply the 2D array by the reshaped 1D array
        mass_weighted_eigenvectors = mass_weighted_eigenvectors * sqrtmmminv_reshaped
        # print('#####################_________')
        # print(mass_weighted_eigenvectors)

        return np.flip(eigenvalues, 0), np.flip(mass_weighted_eigenvectors, 1)
        # return np.flip(eigenvalues, 0), np.flip(xL, 1)

    def tensors2NMbasis(self, dimlessFile: str = None) -> None:
        """

        :param dimlessFile:
        :return:
        """
        if dimlessFile is None:
            eigenvalues, mass_weighted_eigenvectors = self.vibrational_analysis()
            # print(mass_weighted_eigenvectors, mass_weighted_eigenvectors.shape)
        else:
            # a dictionary is returned
            rr = getDimensionlessNM(dimlessFile)
            # print(rr)
            mass_weighted_eigenvectors = np.concatenate([i.reshape(-1, 1) for i in rr.values()], axis=1)
            # print(mass_weighted_eigenvectors, mass_weighted_eigenvectors.shape)


        for p in self.props:
            # if p.operator == ['EL', 'GEO'] or p.operator == ['GEO', 'EL']:
            # if p.operator == ['GEO', 'GEO', 'EL']:
            self.cart2normal(p, mass_weighted_eigenvectors)


    # transform any tensor's GEO cartesian to normal
    def cart2normal(self, property, transfMatrix):

        import copy
        shape = property.tensor.shape  # rspProperty (SpectroscPy class)

        new_tensor = copy.deepcopy(property.tensor)

        for indx, op in enumerate(property.operator):

            if op == 'GEO':
                einstr = str_einsum('ijkl', indx, len(shape))
                # print(einstr)
                new_tensor = np.einsum(einstr, new_tensor, transfMatrix)

        property.tensor = new_tensor
        return new_tensor

    def getFundamentals(self) -> dict[int:float]:
        """
        Fundamental frequency with anharmonic corrections
        Returns: dict[int:float]
        """
        eigenvalues, mass_weighted_eigenvectors = self.vibrational_analysis()

        return dict(zip(np.arange(len(eigenvalues)), eigenvalues))

    def getAllStates(self) -> dict[tuple[int]: float, tuple[int, int]: float,
                                   tuple[int, int, int]: float]:
        """
        Dictionary of all the states and their frequencies
        Return: dict[tuple[int]: float, tuple[int, int]: float, tuple[int, int, int]: float]
        """

        # Delta = {tuple([o - 7 for o in k]): v for k, v in combd.items()}
        Delta = {(0,):0.}

        return Delta


class GaussianData:

    def __init__(self, data: dict[str:[str, dict]]):
        self.sourcetype = data['type']
        self.files = data['files']

    def getDipDersCart(self):

        fchk_parser = parseGaussian.FormchkInterface(self.files['fchk'])
        dipderCart = fchk_parser.dipolederiv()
        return dipderCart


    def getPolarDersCart(self):
        fchk_parser = parseGaussian.FormchkInterface(self.files['fchk'])
        polder = fchk_parser.polarderiv()
        return polder

    def get_hessian_tensor(self):
        if self.sourcetype == 'fchk':
            fchk_parser = parseGaussian.FormchkInterface(self.files['fchk'])
            hessian = fchk_parser.hessian()

            return hessian

from abc import ABC, abstractmethod
from typing import Callable, Iterator, Union, Optional, Any
# callback: Callable[[str], int]

class DerivativesInterface(ABC):
    """Base class for derivatives Interfaces."""

    @property
    @abstractmethod
    def check_version(self) -> Union[None, str]:
        """Check the version."""

    @property
    @abstractmethod
    def author(self) -> str:
        """Set the authors email adress."""

    @abstractmethod
    def get_energy(self, molecule: Any) -> Optional[float]:
        """Compute Energy."""

    @abstractmethod
    def get_gradient(self, molecule: Any) -> Optional[np.array]:
        """Compute Gradient."""

    @abstractmethod
    def get_hessian(self, molecule: Any) -> Optional[np.array]:
        """Compute Hessian."""


