from scriptsHPC.utils import parseCFOUR
from scriptsHPC.utils import parseGaussian
from typing import Any

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
            # fundamentals_harm = parseCFOUR.get_anharmonic_fundamentals(self.files['out'], filetype='out')
            allstates_CFOUR, allstates_CFOUR_harm = self.getAllStates()
            funds_c4 = {k: v for k, v in allstates_CFOUR_harm.items() if len(k) == 1}
            sorted_data_c4 = {k[0]: funds_c4[k] for k in sorted(funds_c4)}

            return fundamentals, sorted_data_c4

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
        combd_harm = dict(zip(ls0, ls4))
        Delta_harm = {tuple([o - 7 for o in k]): v for k, v in combd_harm.items()}

        return Delta, Delta_harm

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
        # if self.sourcetype == 'out':
        #     dalpha, d2alpha = parseCFOUR.getPolarDers(self.files['polardir'])
        #
        # elif self.sourcetype == 'pkl':
        #     dipolepkl = self.files['polar']
        #     import pickle
        #     with open(dipolepkl, 'rb') as file:
        #         dalpha, d2alpha = pickle.load(file)
        #
        # return dalpha, d2alpha

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
            freq, freq_harm = self.getFundamentals()
            cff = parseCFOUR.getCubicPost(freq_harm, cubic)
            return cff

        elif self.sourcetype == 'pkl':
            cubicpkl = self.files['cubic']
            import pickle
            with open(cubicpkl, 'rb') as file:
                # first 3 columns are the normal mode indices, the last column holds the derivatives
                cff = pickle.load(file)

            freq, freq_harm = self.getFundamentals()
            cubicFC = parseCFOUR.getCubicPost(freq_harm, cff)

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
    # print('helloooo')
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
        # print(einstr)
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
    """

    Also fundamental frequencies with anharmonic corrections are needed
    Overtones and combination bands too
    data is a list of np.arrays 'mu_Q', 'mu_QQ', 'alpha_Q', 'alpha_QQ', 'F_abc'
    """
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

    def getFundamentals(self) -> dict[int:float]:
        """
        Fundamental frequency with anharmonic corrections
        Returns: dict[int:float]
        """
        if self.sourcetype == 'fchk':
            pass
        elif self.sourcetype == 'log':
            results = parseGaussian.parse_frequencies(self.files['log'])
            # print(results['Fundamental Bands'])
            funddict = {int(k)-1: float(v) for k, v in zip(results['Fundamental Bands']['mode_a'], results['Fundamental Bands'][2])}
            funddict_harm = {int(k)-1: float(v) for k, v in zip(results['Fundamental Bands']['mode_a'], results['Fundamental Bands'][1])}
            return funddict, funddict_harm

    def getAllStates(self) -> dict[tuple[int]: float, tuple[int, int]: float,
                                    tuple[int, int, int]: float]:
          """
          Dictionary of all the states and their frequencies
          Return: dict[tuple[int]: float, tuple[int, int]: float, tuple[int, int, int]: float]
          """
          if self.sourcetype == 'fchk':
                pass
          elif self.sourcetype == 'log':
                results = parseGaussian.parse_frequencies(self.files['3quanta'])
                results['Combination Bands']['mode_c'] = results['Combination Bands']['mode_c'].fillna(0)
                results['Combination Bands']['n_c'] = results['Combination Bands']['n_c'].fillna(0)

                # print(results['Combination Bands'])
                funddict = {tuple([int(k)-1]): float(v) for k, v in
                            zip(results['Fundamental Bands']['mode_a'], results['Fundamental Bands'][2])}
                states = {
                    tuple(sorted([int(k) - 1 for k in t.split()] * int(n))): float(v)
                    for t, v, n in
                    zip(results['Overtones']['mode_a'], results['Overtones'][2], results['Overtones']['n_a'])
                }
                combinationbands = {
                    tuple(
                        sorted([int(k) - 1 for k in t1.split()] * int(n1) + [int(l) - 1 for l in t2.split()] * int(n2) + [(int(t3)-1)] * int(n3))
                    ): float(v)
                    for t1, t2, t3, v, n1, n2, n3 in
                    zip(results['Combination Bands']['mode_a'], results['Combination Bands']['mode_b'],
                        results['Combination Bands']['mode_c'],
                        results['Combination Bands'][4], results['Combination Bands']['n_a'],
                        results['Combination Bands']['n_b'], results['Combination Bands']['n_c'])
                }
                # print(combinationbands)
                allstates_anharm = {**funddict, **states, **combinationbands}

                funddict1 = {tuple([int(k)-1]): float(v) for k, v in
                            zip(results['Fundamental Bands']['mode_a'], results['Fundamental Bands'][1])}
                states1 = {tuple(sorted([int(k)-1 for k in t.split()]*int(n))): float(v) for t, v, n in
                          zip(results['Overtones']['mode_a'], results['Overtones'][1], results['Overtones']['n_a'])}
                combinationbands1 = {
                    tuple(
                        sorted([int(k) - 1 for k in t1.split()] * int(n1) + [int(l) - 1 for l in t2.split()] * int(n2) + [(int(t3)-1)] * int(n3))
                    ): float(v)
                    for t1, t2, t3, v, n1, n2, n3 in
                    zip(results['Combination Bands']['mode_a'], results['Combination Bands']['mode_b'],
                        results['Combination Bands']['mode_c'],
                        results['Combination Bands'][3], results['Combination Bands']['n_a'],
                        results['Combination Bands']['n_b'], results['Combination Bands']['n_c'])
                }
                allstates_harm = {**funddict1, **states1, **combinationbands1}
                # print('\n', combinationbands1)

                allstates_anharm = {key:allstates_anharm[key] for key in sorted(allstates_anharm,key=allstates_anharm.get)}
                return allstates_anharm, allstates_harm

    def getDipDers(self) -> tuple[Any, ...]:
        """
        Dipole derivatives: first order and second order
        Return: tuple[np.ndarray - shape(NM, 3), np.ndarray - shape(NM, NM, 3)]
        """
        if self.sourcetype == 'fchk':
            dipderCart = self.getDipDersCart()
            pass

        elif self.sourcetype == 'log':
            dipl, units = parseGaussian.parse_dipole_moment(self.files['log'])
            a2d = dipl.loc[dipl['P'] == 'P1', ['X', 'Y', 'Z']].to_numpy()
            a2d2 = dipl.loc[dipl['P'] == 'P2', ['X', 'Y', 'Z']].to_numpy()
            a2d2_3d = np.zeros((6, 6, 3))

            for i, j, xyz in zip(dipl.loc[dipl['P'] == 'P2', 'i'], dipl.loc[dipl['P'] == 'P2', 'j'], a2d2):
                a2d2_3d[int(i) - 1, int(j) - 1] = xyz
                a2d2_3d[int(j) - 1, int(i) - 1] = xyz

            return tuple([a2d, a2d2_3d])

    def getPolarDers(self) -> tuple[Any, ...]:
        """
        Polarizability derivatives: first order and second order
        Return: tuple[np.ndarray - shape(NM, 3, 3), np.ndarray - shape(NM, NM, 3, 3)]
        """
        if self.sourcetype == 'fchk':
            polder = self.getPolarDersCart()
            pass

        elif self.sourcetype == 'log':
            pol = parseGaussian.parse_polarizability(self.files['log'])
            nm = int(pol.loc[pol[0] == 'P1', 1].max())
            p1_3d = np.zeros((nm, 3, 3))
            for i in pol.loc[pol[0] == 'P1', 1].unique():
                # Get the rows with the current 'i' value and 'P' is 'P1', and select columns 5, 6, and 7
                xyz = pol.loc[(pol[0] == 'P1') & (pol[1] == i), [5, 6, 7]].values
                # Assign the 2D array 'xyz' to the corresponding slice of the 3D array
                p1_3d[int(i) - 1] = xyz

            nm_i = int(pol.loc[pol[0] == 'P2', 1].max())
            nm_j = int(pol.loc[pol[0] == 'P2', 2].max())

            p2_4d = np.zeros((nm_i, nm_j, 3, 3))

            for i in pol.loc[pol[0] == 'P2', 1].unique():
                for j in pol.loc[pol[0] == 'P2', 2].unique():
                    # Get the rows with the current 'i' and 'j' values and 'P' is 'P2', and select columns 5, 6, and 7
                    xyz = pol.loc[(pol[0] == 'P2') & (pol[1] == i) & (pol[2] == j), [5, 6, 7]].values
                    # Check if 'xyz' is not empty
                    if xyz.shape[0] != 0:
                        # Assign the 2D array 'xyz' to the corresponding slice of the 4D array
                        p2_4d[int(i) - 1, int(j) - 1] = xyz
                        p2_4d[int(j) - 1, int(i) - 1] = xyz

            return tuple([p1_3d, p2_4d])

    def getCFF(self) -> np.ndarray:
        """
        CFF: cubic force constant tensor
        Return: np.ndarray - shape(NM, NM, NM)
        """
        if self.sourcetype == 'fchk':
            pass

        elif self.sourcetype == 'log':
            cubic_df = parseGaussian.parse_cubic_constants(self.files['log'])[0]
            # Select the required columns
            selected_df = cubic_df[['I', 'J', 'K', 'K(I,J,K)']]
            # Convert to numpy array
            cubic = selected_df.to_numpy()
            freq, freq_harm = self.getFundamentals()

            cff = parseGaussian.get_cubic_post(freq_harm, cubic)
            return cff




from abc import ABC, abstractmethod
from typing import Callable, Iterator, Union, Optional, Any, Tuple


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


