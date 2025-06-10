from operator import itemgetter
import copy

# detection_method is a detection method: "integrated", "time", "freq"
# If detection_method is "time" or "freq", then the data is one spectral dimension
# If detection_method is "int", then the data is a scalar
# detector_location: Optional explicit vector location in space
# detection_polarization: Detect only light polarized in this specific way
# detection_range: For "time" or "freq" detection, tell over which points (the range) in either t/E space as relevant the data is collected
# wv_filter: Detect only light along these particular wavevectors
# (Format: [{wv 1 pulse A1 id: [wv direction (boolean) per interaction], wv 1 pulse B1 id: [...] }, {wv 2 pulse A2...}])
# ignore_collinear: If wavevector filter, ignore other effects collinear with this/these direction(s)?
class specDetector:

    def __init__(self, detection_method, detector_location=None, detection_polarization=None, detection_range=None, wv_filter=None, ignore_collinear=True):

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


class specScan:

    # TODO: Later the scans could involve a greater variety of parameters that can be varied on a range
    # scan_objs: [['object 1 category', 'object 1 id' [or dummy], 'object 1 attribute', 'multiplier'], ['object 2 category', ...]]
    # range: Iterable of numbers over which objects are varied scaled by their multipliers
    def __init__(self, scan_objs, range):

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



class vibExperiment:

    # field should be an electricField instance: A "base" perturbing field upon which scans can be imposed

    # scans is a list of specRange instances
    # Each scan adds a spectral dimension
    # magn_conditions is a list of imposed pulse magnitude conditions
    # Format: [[(i, -j, k], ...]: w_i - w_j + w_k sign. > 0, ...

    def __init__(self, order, field, detector, scans=None, magn_conditions=[]):

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

    def findDimensionality(self):
        d = 0
        d += len(self.scans)

        if self.detector.dmethod == 'time':
            return d + 1

        elif self.detector.dmethod == 'freq':
            if self.detector.detection_range is not None:
                return d + 1

        return d

    def tellDimensions(self):
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

    # Take field, scans and detector information and find out if any detector range or (impulsive)
    # pulse frequency range is implied by what is hitherto specified
    def findImpliedFreqRanges(self):

        pass

    # Find out if any pulse magnitude statements are implied from the experiment specifications and/or imposed magnitude conditions
    def findImpliedPulseMagnitudeStatements(self):
        pass

    # Impose pulse magnitude conditions
    def imposePulseMagnitudeConditions(self, conditions):
        pass

    # Find out which lower- or same-order effects can be collinear
    def findCollinear(self):

        pass

    # Based on the experiment information, find out if there must be a specific sequence of interactions with pulses
    def findInteractionSequences(self):

        def interactionRecurse(res, curr_int, rem_wv, curr_epoch, epochs):

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

    def present(self):

        pass


# The field consists of a collection of pulses
class electricField:

    def __init__(self, pulses):

        # FIXME: Consider: Pulses as dictionary with IDs?
        self.pulses = pulses

    # Set pulse IDs according to (priorities) a) time max, b) time deviation, c) carrier freq, d) max strength, e) arbitrary choice
    def setCanonicalPulseIds(self):

        pass

    # Divide field into epochs either with zero or finite tolerance
    def findEpochs(self, tol=0.0):

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

    def getCarrierFreqsUV(self):
        cfuv_dict = {}
        for i in self.pulses:
            cfuv_dict[i.id] = i.cf_uv

        return cfuv_dict


class emPulse:

    # Starting with Gaussian, impulsive or "frequency-domain ideal" pulses
    # Total carrier frequency is a sum of "IR" carrier frequency part cf and "UVVIS" part cf_uv
    # The "UVVIS" part would used for manifold-level cancellations
    # The "IR" part is the part considered scannable
    # For pulses where cf_uv != 0.0, then cf must be 0.0
    def __init__(self, env, maxstr, tc=None, cf=None, cf_uv=0.0, dev=None, wv=None, pol=None, id=None):

        # TODO: Generalize to arbitrary pulses and chirped pulses

        # Envelope choices: "impulsive", "ideal", "
        # or "gaussian"
        # "impulsive" pulses have a point temporal intensity and a constant frequency envelope
        # "ideal" is an idealized pulse type which is both point temporal and monochromatic
        # "cw" is a continuous-wave monochromatic pulse
        # Gaussian pulses have a Gaussian temporal envelope with a carrier frequency (and thus also a Gaussian
        # frequency envelope)
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
        # Currently supports unit linear polarization (TODO: Add support for circular polarization: Function?)
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
