from operator import itemgetter
import copy

# TODO: SpecDetector and SpecScan as dataclasses?
# TODO: Expand functionality according to below TODOs

class SpecDetector:
    """
    Class to represent a spectral detector
    """

    def __init__(self, detection_method: str, detector_location: list[float]=None,
                 detection_polarization: list[float]=None, detection_range: list[float]=None,
                 wv_filter: list[dict]=None, ignore_collinear: bool=True):
        """
        detection_method: String: A detection method: "integrated", "time", "freq"
        If detection_method is "time" or "freq", then the detection data is one spectral dimension
        If detection_method is "integrated", then the detection data is a scalar
        Currently, only "freq" (frequency-range) detection is supported.

        detector_location: List of floats: Optional explicit vector location in space. Currently not used.

        detection_polarization: List of floats: Detect only light with this specific polarization vector.
        Currently not used.

        detection_range: List of floats: For "time" or "freq" detection, tell over which points (the range)
        in either t/E space as relevant the data is collected

        wv_filter: List of dictionaries {pulse label: [sign], ...}:  Detect only light along this/these particular
        phase-matching direction(s)
        # FIXME: Putting sign in a list seems unnecessary, consider rm and rework. Or maybe OK if considering multiple
        directions for same puls at once?

        ignore_collinear: If using a wavevector filter, ignore other effects collinear with this/these direction(s)?
        Currently not used.
        """

        if not(detection_method == 'time' or detection_method == 'freq' or detection_method == 'int'):
            raise AssertionError("The detection type must be either 'time', 'freq'(uency) or 'int'(egrated)" )

        self.dmethod = detection_method

        # TODO add check (len 3 array or list)
        self.dloc = detector_location

        # TODO add check
        self.dpol = detection_polarization

        if detection_range is None and (self.dmethod in ['time']):
            raise AssertionError("The detection range must be specified when the detector is set to 'time' or 'freq' detection")

        # Comment: detection_range as None and detection_method as 'freq' is valid but results in no dimensionality
        self.detection_range = detection_range

        self.wv_filter = wv_filter
        self.ignore_collinear = ignore_collinear

    def __repr__(self):
        return f'THIS IS specDetector with detection_method - {self.dmethod}'

class SpecScan:
    """
    Class to represent a spectral scan (adding to the dimensionality of a spectrum)
    """

    # TODO: Later the scans could involve a greater variety of parameters that can be varied on a range
    #
    #
    def __init__(self, scan_objs: list, range):
        """
        scan_objs: [['object 1 category', 'object 1 id' [or dummy], 'object 1 attribute', 'multiplier'],
                    ['object 2 category', ...]]: Tells what is being scanned here

        range: Iterable of numbers over which scan objects are varied (scaled by their multipliers)
        """

        valid_scan_objs = ['pulse', 'detector']
        valid_scan_attributes = {'pulse': ['cf', 'tc', 'dev'], 'detector': ['detection_range']}

        for i in scan_objs:
            print('scan obj i', i)
            if not i[0] in valid_scan_objs:
                raise AssertionError("The scan object must be one of", valid_scan_objs, 'but is instead', i[0])
            if not i[2] in valid_scan_attributes[i[0]]:
                raise AssertionError("The scan attribute for", i[0], 'must be one of', valid_scan_attributes[i[0]], 'but is instead', i[2])

        # TODO: Check if ranges are valid

        self.scan_objs = scan_objs
        self.range = range

    def __repr__(self):
        return f'THIS IS specScan - {self.scan_objs}'

class EmPulse:
    """
    Class to represent an electromagnetic pulse
    """

    def __init__(self, env: str, maxstr: float, tc: float=None, cf: float=None, cf_uv: float=0.0, dev: float=None,
                 wv: list[float]=None, pol: list[float]=None, id: int=None):
        """
        env: String: Pulse time-domain envelope: Valid choices are "impulsive", "ideal" (frequency-domain impulsive and
        time-domain impulsive), "cw" (continuous wave, currently not supported), and "gaussian".
        maxstr: Float: Pulse amplitude at maximum of envelope
        tc: Float: Point in time at which pulse envelope is at maximum
        cf: float: (Infrared-range) Carrier frequency
        cf_uv: float: Designated "UV/VIS range" part of carrier frequency (for e.g. CARS-style cancellation)
        For pulses where cf_uv != 0.0, then cf must be 0.0
        dev: float: Deviation parameter (e.g. broadness of Gaussian pulse)
        wv: float: Wavevector travel direction
        pol: float: Polarization (only linearly polarized light currently countenanced)
        id: integer: Pulse ID label
        """

        # TODO: Generalize to arbitrary pulses and chirped pulses

        allowed_envelopes = ['impulsive', 'ideal', 'cw', 'gaussian']

        if not env in allowed_envelopes:
            raise AssertionError('Allowed pulse envelope choices are: "impulsive", "ideal", "cw", "gaussian"')
        self.env = env

        # Centerpoint of temporal envelope
        self.tc = tc

        # Maximum of temporal envelope
        self.maxstr = maxstr

        self.cf_uv = cf_uv
        if not (cf_uv == 0.0):
            if not (cf == 0.0):
                raise AssertionError('Pulses with non-zero UV/VIS carrier freqs. must have IR carrier freq part set to zero')

        if self.env == "gaussian":
            if cf == None or dev == None:
                raise AssertionError('A Gaussian pulse must have a carrier frequency and a deviation parameter')
            self.cf = cf
            self.dev = dev

        if self.env == "ideal":
            if cf == None:
                raise AssertionError('An pulse of the "ideal" type must have a (monochromatic) "carrier" frequency')
            self.cf = cf

        # Wavevector: In which unit vector direction is the pulse wave travelling
        if wv is None:
            print('No wavevector was specified for pulse, defaulting to unit z direction wavevector')
            self.wv = [0.0, 0.0, 1.0]
        else:
            if isinstance(wv, list):
                if len(wv) == 3:
                    if all([isinstance(i, float) for i in wv]):
                        wv_len = (wv[0]**2.0 + wv[1]**2.0 + wv[2]**2.0)**0.5
                        if not wv_len == 1.0:
                            print('Wavevector was normalized')
                        self.wv = [i/wv_len for i in wv]

                    else:
                        raise AssertionError('The pulse wavevector must be a len 3 list of floats')
                else:
                    raise AssertionError('The pulse wavevector must be a len 3 list of floats')
            else:
                raise AssertionError('The pulse wavevector must be a len 3 list of floats')

        # Polarization: Specify the polarization of the pulse
        # Currently supports unit linear polarization (TODO: Add support for circular polarization (as function)?)
        if pol is None:
            print('No polarization was specified for pulse, defaulting to unit z direction wavevector')
            self.pol = [0.0, 0.0, 1.0]
        else:
            if isinstance(pol, list):
                if len(pol) == 3:
                    if all([isinstance(i, float) for i in pol]):
                        pol_len = (pol[0]**2.0 + pol[1]**2.0 + pol[2]**2.0)**0.5
                        if not pol_len == 1.0:
                            print('Wavevector was normalized')
                        self.pol = [i/pol_len for i in pol]

                    else:
                        raise AssertionError('The pulse wavevector must be a len 3 list of floats')
                else:
                    raise AssertionError('The pulse wavevector must be a len 3 list of floats')
            else:
                raise AssertionError('The pulse wavevector must be a len 3 list of floats')

        # Pulse id
        self.id = id

    def __repr__(self):
        return f'THIS IS emPulse with pulse envelope {self.env}'

# The field consists of a collection of pulses
class ElectricField:
    """
    Class to represent an electromagnetic field consisting of one or more pulses
    """

    def __init__(self, pulses: list[EmPulse]):
        """
        pulses: List of EmPulse instances: The pulses making up the field
        """

        # FIXME: Consider: Pulses as dictionary with IDs?
        self.pulses = pulses

    def findEpochs(self, tol: float=0.0) -> list:
        """
        Divide field into epochs either with zero or finite tolerance
        Currently only supported for a field consisting of ideal or impulsive pulses

        tol: Float: Tolerance for non-temporal coincidence (currently not supported)
        # TODO: Add support for tolerance

        Returns: List of lists: [[epoch 1 pulse 1, epoch 1 pulse 2, ...], [epoch 2 pulse 1, ...], ...]
        """

        for i in self.pulses:
            if not(i.env in ['ideal', 'impulsive']):
                raise AssertionError('Can currently only determine epochs for fields consisting of only ideal or impulsive pulses')
            if i.id is None:
                raise AssertionError('All pulses must have IDs for valid epoch determination')

        times_ids = sorted([(i.tc, i.id) for i in self.pulses], key=itemgetter(0))
        epochs = [[]]
        epoch = 0
        curr_time = times_ids[0][0]

        for i in times_ids:
            if not(i[0] == curr_time):
                epochs.append([])
                epoch += 1
                curr_time = i[0]
            epochs[epoch].append(i[1])

        return epochs

    def getCarrierFreqsUV(self) -> dict:
        """
        Get dictionary of UV/VIS-range part of carrier frequencies

        Returns: Dictionary {pulse 1: UV/VIS carrier freq., ...}
        """
        cfuv_dict = {}
        for i in self.pulses:
            cfuv_dict[i.id] = i.cf_uv

        return cfuv_dict

class VibExperiment:
    """
    Class to represent a vibrational wave-mixing experiment
    """

    # field should be an electricField instance:

    # scans is a list of specRange instances
    # Each scan adds a spectral dimension
    # magn_conditions is a list of imposed pulse magnitude conditions
    # Format: [[(i, -j, k], ...]: w_i - w_j + w_k sign. > 0, ...

    def __init__(self, order: int, field: ElectricField, detector: SpecDetector, scans: list[SpecScan]=None,
                 magn_conditions: list=[]):
        """
        field: ElectricField instance: A "base" perturbing field (upon which scans may be imposed)
        detector: SpecDetector instance: The detector for this experiment
        scans: List of SpecScan instances: Tells which parameters will be scanned over (and how) in this experiment
        magn_conditions: List of lists [[sign*pulse i (is always of lower frequency than...), sign*pulse j], ...]:
        Magnitude conditions for later use in identifying terms that will not become fully resononant in this experiment
        """


        self.order = order
        self.field = field
        self.detector = detector

        self.scans = []

        if scans is not None:
            self.scans.extend(scans)

        self.magn_conditions = magn_conditions

        self.dim = self.findDimensionality()
        self.epochs = self.field.findEpochs()
        self.int_sequences = self.findInteractionSequences()
        self.cfuv = self.field.getCarrierFreqsUV()

    def findDimensionality(self) -> int:
        """
        Using detector and scans information, determine the dimensionality of the spectral data
        that carrying out this experiment would produce

        Returns an integer d telling this dimensionality
        """

        d = 0
        d += len(self.scans)

        if self.detector.dmethod == 'time':
            return d + 1

        elif self.detector.dmethod == 'freq':
            if self.detector.detection_range is not None:
                return d + 1

        return d

    def tellDimensions(self):
        """
        Using detector and scans information, make formatted print report of each dimension of the spectral data
        that carrying out this experiment would produce
        """

        d = 0
        for i in self.scans:
            d += 1
            print('Dimension', i, 'is a scan:', i.scan_objs, 'over the range', i.range)

        if self.detector.dmethod == 'time':
            print('Dimension', i, 'is a scan over the time-domain detection range', self.detector.detection_range)

        elif self.detector.dmethod == 'freq':
            if self.detector.detection_range is not None:
                print('Dimension', i, 'is a scan over the frequency-domain detection range', self.detector.detection_range)

        return

    def findInteractionSequences(self) -> list:
        """
        Based on the experiment information, find out if there must be a specific sequence/sequences of interactions with pulses

        Returns a list [[{sequence 1 interaction 1: pulse i}, {seq. 1 int. 2: pulse j}, ...],
                        [{seq. 2 int. 1: pulse k}, ... ], ...]
        """

        def interactionRecurse(res: list, curr_int: list, rem_wv: dict, curr_epoch: int, epochs: list):
            """
            Tail-recursive routine for finding interaction sequences

            res: List of lists: Results accumulator
            curr_int: List of dictionaries: Result currently being assembled
            rem_wv: Dictionary: One wavevector filter dictionary
            curr_epoch: Epoch counter
            epochs: List of epochs as determined by ElectricField.findEpochs
            """


            # Termination condition
            # If this interaction sequence satisfied the wavevector filter, append it
            if rem_wv == {}:
                res.append(curr_int)

            # Recursion
            else:
                # Can one or more pulses in requested wv be found at the current or later epoch?
                # If so, make all combinations, update rem wv and recurse further at same epoch

                for t in range(curr_epoch, len(epochs)):
                    for i in rem_wv:

                        if i in epochs[t]:
                            for j in range(len(rem_wv[i])):
                                new_rem_wv = copy.deepcopy(rem_wv)
                                new_int = copy.deepcopy(curr_int)
                                new_int.append({i: new_rem_wv[i][j]})

                                del new_rem_wv[i][j]
                                if new_rem_wv[i] == []:
                                    del new_rem_wv[i]

                                interactionRecurse(res, new_int, new_rem_wv, t, epochs)

        if self.detector.wv_filter is None:
            raise AssertionError('Interaction sequence determination currently only implemented for wavevector filter detector')

        int_sequences = []
        int_seed = []

        for i in self.detector.wv_filter:

            interactionRecurse(int_sequences, int_seed, i, 0, self.field.findEpochs())

        return int_sequences


