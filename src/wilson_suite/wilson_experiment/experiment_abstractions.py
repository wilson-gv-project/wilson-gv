from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Iterable
from operator import itemgetter
import copy

# TODO: Expand functionality according to below TODOs

@dataclass
class SpecDetector:
    """
    Class to represent a spectral detector

    ----
    detection_method: String: A detection method: "integrated", "time", "freq"
    If detection_method is "time" or "freq", then the detection data is one spectral dimension
    If detection_method is "integrated", then the detection data is a scalar
    Currently, only "freq" (frequency-range) detection is supported.

    detector_location: List of floats: Unit vector describing the direction in which the detector is facing.
    Default: [0.0, 0.0, 1.0]. Currently only used for defining polarization filtering.

    detection_polarization: List of floats: Detect only light with this specific polarization vector. Default:
    [1.0, 0.0, 0.0].

    detection_range: List of floats: For "time" or "freq" detection, tell over which points (the range)
    in either t/E space as relevant the data is collected

    wv_filter: List of dictionaries {pulse label: sign, ...}:  Detect only light along this/these particular
    phase-matching direction(s)
    # FIXME: Putting sign in a list seems unnecessary, consider rm and rework. Or maybe OK if considering multiple
    directions for same puls at once?

    ignore_collinear: If using a wavevector filter, ignore other effects collinear with this/these direction(s)?
    Currently not used.
    """
    detection_method: str
    
    detector_location: Optional[tuple[float]] = None
    
    detection_polarization: Optional[tuple[float]] = None
    
    # Comment: detection_range as None and detection_method as 'freq' is valid but results in no dimensionality
    detection_range: Optional[List[float]] = None
    wv_filter: Optional[List[dict]] = None
    ignore_collinear: bool = True

    overall_phase: complex = 1.0 + 0.0j

    def __post_init__(self):
        if self.detection_method not in {'time', 'freq', 'int'}:
            raise ValueError("The detection type must be either 'time', 'freq'(uency), or 'int'(egrated)")

        if self.detection_range is None and (self.detection_method in ['time']):
            raise AssertionError("The detection range must be specified when the detector is set to 'time' or 'freq' detection")

        if not self.overall_phase == 1.0 + 0.0j:
            raise AssertionError('Detector overall phase currently restricted to zero shift')

        if not(abs(self.overall_phase) - 1.0 < 1e-10):
            raise AssertionError('Detector overall phase must be of unit length')


@dataclass
class SpecScan:
    """
    Class to represent a spectral scan (adding to the dimensionality of a spectrum)

    ----
    scan_objs: [['object 1 category', 'object 1 id' [or dummy], 'object 1 attribute', 'multiplier'],
                ['object 2 category', ...]]: Tells what is being scanned here

    range: Iterable of numbers over which scan objects are varied (scaled by their multipliers)
    """

    # TODO: Later the scans could involve a greater variety of parameters that can be varied on a range
    
    # TODO: Check if ranges are valid
    scan_objs: List
    range: Iterable

    def __post_init__(self):
        valid_scan_objs = ['pulse', 'detector']
        valid_scan_attributes = {'pulse': ['cf', 'tc', 'dev'], 'detector': ['detection_range']}

        for i in self.scan_objs:
            if i[0] not in valid_scan_objs:
                raise AssertionError("The scan object must be one of", valid_scan_objs, 'but is instead', i[0])
            if i[2] not in valid_scan_attributes[i[0]]:
                raise AssertionError("The scan attribute for", i[0], 'must be one of', valid_scan_attributes[i[0]], 'but is instead', i[2])


@dataclass
class EmPulse:
    """
    Class to represent an electromagnetic pulse
    
    ----
    env: String: Pulse time-domain envelope: Valid choices are "impulsive", "ideal" (frequency-domain impulsive and
    time-domain impulsive), "cw" (continuous wave, currently not supported), and "gaussian".
    maxstr: Float: Pulse amplitude at maximum of envelope
    tc: Float: Point in time at which pulse envelope is at maximum
    cf: float: (Infrared-range) Carrier frequency
    cf_uv: float: Designated "UV/VIS range" part of carrier frequency (for e.g. CARS-style cancellation).
        - cf_uv should be specified as a nonnegative value: Any cancellation should follow from the phase-matching
        wavevector--frequency combination in the experiment
        - For pulses where cf_uv != 0.0, then cf must be 0.0
    dev: float: Deviation parameter (e.g. broadness of Gaussian pulse)
    wv: floats: Unit wavevector propagation direction with respect to laboratory axes
    pol: floats: Polarization: Unit vector describing polarization direction with respect to laboratory axes
        - Must be orthogonal to wavevector
        - Only linear polarization currently supported (no phase difference between orthogonal components of
        polarization vector in plane of polarization)
        - Default: (1.0, 0.0, 0.0)
    overall_phase: complex number defining a unit vector in the complex plane: Overall phase of pulse. Currently enforced as (1.0, 0.0)

    id: integer: Pulse ID label
    """
    env: str
    
    # Maximum of temporal envelope
    maxstr: float
    
    # Centerpoint of temporal envelope
    tc: float = None
    
    cf: float = None
    cf_uv: float = 0.0
    dev: float = None
    wv: tuple[float] = (0.0, 0.0, 1.0)
    pol: tuple[float] = (1.0, 0.0, 0.0)
    overall_phase: complex = 1.0 + 0.0j
    id: int = None

    def __post_init__(self):
        
        # TODO: Generalize to arbitrary pulses and chirped pulses
        
        allowed_envelopes = ['impulsive', 'ideal', 'cw', 'gaussian']

        if self.env not in allowed_envelopes:
            raise AssertionError('Allowed pulse envelope choices are: "impulsive", "ideal", "cw", "gaussian"')

        if not (self.cf_uv == 0.0):
            if not (self.cf == 0.0):
                raise AssertionError('Pulses with non-zero UV/VIS carrier freqs. must have IR carrier freq part set to zero')

        if self.env == "gaussian":
            if self.cf is None or self.dev is None:
                raise AssertionError('A Gaussian pulse must have a carrier frequency and a deviation parameter')
            
        if self.env == "ideal":
            if self.cf is None:
                raise AssertionError('An pulse of the "ideal" type must have a (monochromatic) "carrier" frequency')
        
        # Wavevector: In which unit vector direction is the pulse wave travelling
        if isinstance(self.wv, tuple):
            if len(self.wv) == 3:
                if all([isinstance(i, float) for i in self.wv]):
                    wv_len = (self.wv[0]**2.0 + self.wv[1]**2.0 + self.wv[2]**2.0)**0.5
                    if not wv_len == 1.0:
                        print('Wavevector was normalized')
                    self.wv = tuple([i/wv_len for i in self.wv])

                else:
                    raise AssertionError('The pulse wavevector must be a len 3 tuple of floats')
            else:
                raise AssertionError('The pulse wavevector must be a len 3 tuple of floats')
        else:
            raise AssertionError('The pulse wavevector must be a len 3 tuple of floats')

        # Polarization: Specify the polarization of the pulse
        # Currently supports unit linear polarization
        if isinstance(self.pol, tuple):
            if len(self.pol) == 3:
                if all([isinstance(i, float) for i in self.pol]):
                    pol_len = (self.pol[0]**2.0 + self.pol[1]**2.0 + self.pol[2]**2.0)**0.5
                    if not pol_len == 1.0:
                        print('Wavevector was normalized')
                    self.pol = [i/pol_len for i in self.pol]

                    wv_pol_dot = self.pol[0] * self.wv[0] + self.pol[1] * self.wv[1] + self.pol[2] * self.wv[2]

                    if not(wv_pol_dot == 0.0):
                        raise AssertionError('Error: Wavevector of pulse not orthogonal to polarization vector')

                else:
                    raise AssertionError('The polarization vector must be a len 3 tuple of floats')
            else:
                raise AssertionError('The polarization vector must be a len 3 tuple of floats')
        else:
            raise AssertionError('The polarization must be a len 3 tuple of floats')

        if not self.overall_phase == 1.0 + 0.0j:
            raise AssertionError('Overall phase currently restricted to zero shift')

        if not((abs(self.overall_phase) - 1.0) < 1e-10):
            raise AssertionError('The overall phase must be of unit length')


# The field consists of a collection of pulses
@dataclass
class ElectricField:
    """
    Class to represent an electromagnetic field consisting of one or more pulses
    
    ----
    pulses: List of EmPulse instances: The pulses making up the field. While EmPulse itself does not require an ID to
    be specified, the use of pulses in ElectricField must have each pulse be assigned an ordinal integer ID starting from 1
    """

    pulses: tuple[EmPulse]

    def __post_init__(self):

        pulse_id_target = [i + 1 for i in range(len(self.pulses))]
        for i in self.pulses:
            if not i.id in pulse_id_target:
                raise ValueError('Pulse ID', i.id, ' of field not found in ordinal list')
            else:
                pulse_id_target.remove(i.id)

        # This condition should never be met but just to be safe
        if not pulse_id_target == []:
            raise AssertionError('Collection of pulse IDs do not correspond to ordinal list')

@dataclass
class VibExperiment:
    """
    Class to represent a vibrational wave-mixing experiment

    ----
    field: ElectricField instance: A "base" perturbing field (upon which scans may be imposed)
    detector: SpecDetector instance: The detector for this experiment
    scans: List of SpecScan instances: Tells which parameters will be scanned over (and how) in this experiment
    magn_conditions: List of lists [[sign*pulse i (is always of lower frequency than...), sign*pulse j], ...]:
    Magnitude conditions for later use in identifying terms that will not become fully resononant in this experiment
    """

    # field should be an electricField instance:

    # scans is a list of specRange instances
    # Each scan adds a spectral dimension
    # magn_conditions is a list of imposed pulse magnitude conditions
    # Format: [[(i, -j, k], ...]: w_i - w_j + w_k sign. > 0, ...

    order: int
    field: ElectricField
    detector: SpecDetector
    scans: list[SpecScan] = None
    magn_conditions: list = dc_field(default_factory=lambda: list)

    def __post_init__(self):

        from wilson_suite.wilson_experiment.indep_vars_and_axes import (PhaseMatchingCondition, SignedPulseTuple,
                                                                        find_indep_exp_variables, find_valid_axes,
                                                                        find_canonical_axes)

        relevant_phasematch = []

        # If no specified phase-matching (wavevector) filter, all are (potentially) relevant
        if self.detector.wv_filter is None:

            from itertools import product as iter_prod

            k = 0

            for i in iter_prod([1, -1], repeat=len(self.field.pulses)):

                new_phasematch = []

                for j in range(len(self.field.pulses)):
                    new_phasematch.append(self.field.pulses[j].id * i[j])

                relevant_phasematch.append(PhaseMatchingCondition(SignedPulseTuple(tuple(new_phasematch)), k))
                k += 1

        else:

            k = 0

            for i in range(len(self.detector.wv_filter)):

                new_phasematch = []

                for j in self.detector.wv_filter[i]:
                    new_phasematch.append(j * self.detector.wv_filter[i][j])

                relevant_phasematch.append(PhaseMatchingCondition(SignedPulseTuple(tuple(new_phasematch)), k))
                k += 1


        self.relevant_phasematch = relevant_phasematch

        self.dim = self.findDimensionality()
        self.epochs = find_epochs(self.field)
        self.int_sequences = self.findInteractionSequences()
        self.cfuv = get_carrier_freqs_uv(self.field.pulses)
        self.indep_vars = find_indep_exp_variables(self.field.pulses, self.epochs, self.relevant_phasematch)
        self.valid_axis_combs = find_valid_axes(self.indep_vars)
        self.canonical_axes = find_canonical_axes(self.indep_vars)


        # Here I establish a convention: Macroscopic ranks are with respect to pulse IDs but first rank refers to the
        # detected signal (so detected, pulse ID 1, pulse ID 2, ...)
        all_polarizations = [copy.deepcopy(self.detector.detection_polarization)]

        # Could probably be done more elegantly but works
        for i in range(len(self.field.pulses)):
            for j in self.field.pulses:
                if j.id == i + 1:
                    all_polarizations.append(copy.deepcopy(j.pol))

        self.all_polarizations = all_polarizations

        from wilson_suite.wilson_intensities.amplitudes.averaging import get_pol_laser

        self.polarization_avg_vector = get_pol_laser(self.all_polarizations)


    def findDimensionality(self) -> int:
        """
        Using detector and scans information, determine the dimensionality of the spectral data
        that carrying out this experiment would produce

        Returns an integer d telling this dimensionality
        """

        d = 0
        d += len(self.scans)

        if self.detector.detection_method == 'time':
            return d + 1
        
        elif self.detector.detection_method == 'freq':
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

        if self.detector.detection_method == 'time':
            print('Dimension', i, 'is a scan over the time-domain detection range', self.detector.detection_range)

        elif self.detector.detection_method == 'freq':
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
                res.append(tuple(curr_int))

            # Recursion
            else:
                # Can one or more pulses in requested wv be found at the current or later epoch?
                # If so, make all combinations, update rem wv and recurse further at same epoch

                for t in range(curr_epoch, len(epochs)):
                    for i in rem_wv:

                        if i in epochs[t]:

                            new_rem_wv = copy.deepcopy(rem_wv)
                            new_int = copy.deepcopy(curr_int)
                            new_int.append({i: new_rem_wv[i]})

                            del new_rem_wv[i]

                            interactionRecurse(res, new_int, new_rem_wv, t, epochs)

        if self.detector.wv_filter is None:
            raise AssertionError('Interaction sequence determination currently only implemented for wavevector filter detector')

        int_sequences = []
        int_seed = []

        for i in self.detector.wv_filter:

            interactionRecurse(int_sequences, int_seed, i, 0, find_epochs(self.field))

        return int_sequences


def get_carrier_freqs_uv(pulses) -> dict:
    """
    Get dictionary of UV/VIS-range part of carrier frequencies

    Returns: Dictionary {pulse 1: UV/VIS carrier freq., ...}
    """
    cfuv_dict = {}
    for i in pulses:
        cfuv_dict[i.id] = i.cf_uv

    return cfuv_dict

def find_epochs(field, tol: float=0.0) -> list:
    """
    Divide field into epochs with either zero or finite tolerance
    Currently only supported for a field consisting of ideal or impulsive pulses

    tol: Float: Tolerance for non-temporal coincidence (currently not supported)
    # TODO: Add support for tolerance

    Returns: List of lists: [[epoch 1 pulse 1, epoch 1 pulse 2, ...], [epoch 2 pulse 1, ...], ...]
    """

    for i in field.pulses:
        if i.env not in ['ideal', 'impulsive']:
            raise AssertionError('Can currently only determine epochs for fields consisting of only ideal or impulsive pulses')
        if i.id is None:
            raise AssertionError('All pulses must have IDs for valid epoch determination')

    times_ids = sorted([(i.tc, i.id) for i in field.pulses], key=itemgetter(0))
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

def uv_cancels(coll: tuple, cfs_uv: dict, tol: float=1e-10) -> bool:

    acc = 0.0

    for i in coll:
        sgn = (i > 0) - (i < 0)
        acc += cfs_uv[sgn*i]

    sgnacc = (acc > 0) - (acc < 0)

    return ((sgnacc * acc) <= tol)