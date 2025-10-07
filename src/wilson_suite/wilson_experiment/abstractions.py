from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Iterable
from operator import itemgetter
import copy

# TODO: SpecDetector and SpecScan as dataclasses?
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
    detection_method: str
    
    # TODO add check (len 3 array or list)
    detector_location: Optional[List[float]] = None
    
    # TODO add check
    detection_polarization: Optional[List[float]] = None
    
    # Comment: detection_range as None and detection_method as 'freq' is valid but results in no dimensionality
    detection_range: Optional[List[float]] = None
    wv_filter: Optional[List[dict]] = None
    ignore_collinear: bool = True

    def __post_init__(self):
        if self.detection_method not in {'time', 'freq', 'int'}:
            raise ValueError("The detection type must be either 'time', 'freq'(uency), or 'int'(egrated)")

        if self.detection_range is None and (self.detection_method in ['time']):
            raise AssertionError("The detection range must be specified when the detector is set to 'time' or 'freq' detection")

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
    wv: float: Wavevector travel direction
    pol: float: Polarization (only linearly polarized light currently countenanced)
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
    wv: List[float] = None
    pol: List[float] = None
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
        if self.wv is None:
            print('No wavevector was specified for pulse, defaulting to unit z direction wavevector')
            self.wv = [0.0, 0.0, 1.0]
        else:
            if isinstance(self.wv, list):
                if len(self.wv) == 3:
                    if all([isinstance(i, float) for i in self.wv]):
                        wv_len = (self.wv[0]**2.0 + self.wv[1]**2.0 + self.wv[2]**2.0)**0.5
                        if not wv_len == 1.0:
                            print('Wavevector was normalized')
                        self.wv = [i/wv_len for i in self.wv]

                    else:
                        raise AssertionError('The pulse wavevector must be a len 3 list of floats')
                else:
                    raise AssertionError('The pulse wavevector must be a len 3 list of floats')
            else:
                raise AssertionError('The pulse wavevector must be a len 3 list of floats')

        # Polarization: Specify the polarization of the pulse
        # Currently supports unit linear polarization (TODO: Add support for circular polarization (as function)?)
        if self.pol is None:
            print('No polarization was specified for pulse, defaulting to unit z direction wavevector')
            self.pol = [0.0, 0.0, 1.0]
        else:
            if isinstance(self.pol, list):
                if len(self.pol) == 3:
                    if all([isinstance(i, float) for i in self.pol]):
                        pol_len = (self.pol[0]**2.0 + self.pol[1]**2.0 + self.pol[2]**2.0)**0.5
                        if not pol_len == 1.0:
                            print('Wavevector was normalized')
                        self.pol = [i/pol_len for i in self.pol]

                    else:
                        raise AssertionError('The pulse wavevector must be a len 3 list of floats')
                else:
                    raise AssertionError('The pulse wavevector must be a len 3 list of floats')
            else:
                raise AssertionError('The pulse wavevector must be a len 3 list of floats')


# The field consists of a collection of pulses
@dataclass
class ElectricField:
    """
    Class to represent an electromagnetic field consisting of one or more pulses
    
    ----
    pulses: List of EmPulse instances: The pulses making up the field
    """
    # FIXME: Consider: Pulses as dictionary with IDs?
    pulses: List[EmPulse]


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

def uv_cancels(coll: tuple, cfs_uv: dict, tol: float=0.0) -> bool:

    acc = 0.0

    for i in coll:
        acc += cfs_uv[i]

    sgnacc = (acc > 0) - (acc < 0)

    return ((sgnacc * acc) <= tol)


def find_subsets_making_orig(subsets, acc, orig, res):

    def collapse(s):
        c = []
        for i in s:
            c.extend(list(i))
        return c

    if len(collapse(acc)) > len(orig):
        return

    if (collapse(sorted(acc)) == orig) and (len(orig) > 0):
        res.append(acc)

    if len(subsets) > 0:

        for i in range(len(subsets)):

            new_acc = copy.deepcopy(acc)
            new_acc.append(subsets[i])

            if i - 1 < len(subsets):
                find_subsets_making_orig(subsets[i + 1:], new_acc, orig, res)

            else:
                find_subsets_making_orig([], new_acc, orig, res)


def find_indep_vars_for_one_phasematch(field, epochs, pm_dir):

    # For each phase-matching condition
    ind_vars_p = []

    cfuv = get_carrier_freqs_uv(field.pulses)

    for i in range(len(epochs)):

        ind_vars_p_epoch = []
        cfuv_this_pm = {}
        uv_this = []
        ir_this = []

        for k in epochs[i]:
            if not(cfuv[k] == 0.0):
                cfuv_this_pm[k] = cfuv[k] * pm_dir[k][0]
                uv_this.append(k)
            else:
                ir_this.append(k)

        from itertools import chain, combinations

        uv_superset = set(list(chain.from_iterable(combinations(uv_this, r) for r in range(len(uv_this) + 1)))[1:])
        uv_superset_cancel = []

        for j in uv_superset:

            if uv_cancels(j, cfuv_this_pm):
                uv_superset_cancel.append(j)

        acc = []
        uv_subs_res = []
        find_subsets_making_orig(uv_superset_cancel, acc, uv_this, uv_subs_res)

        for j in ir_this:
            ind_vars_p_epoch.append(j)

        for j in uv_subs_res:
            ind_vars_p_epoch.append(j)

        if (len(ind_vars_p_epoch) > 0):
            ind_vars_p.append(copy.deepcopy(ind_vars_p_epoch))

    print('ind vars p', ind_vars_p)

    return ind_vars_p

# FIXME: Update when working to use attributes and not field instance
def find_indep_exp_variables(field, epochs, phasematch_dirs):

    all_ind_var_cfgs_p = []

    for p in phasematch_dirs:

        all_ind_var_cfgs_p.append(copy.deepcopy(find_indep_vars_for_one_phasematch(field, epochs, p)))

    return all_ind_var_cfgs_p

def find_canonical_axes(ind_vars_cfg_p):

    canonical_axis_cfg = []

    return canonical_axis_cfg
# FIXME: THE NEW ROUTINES HERE MUST HAVE TESTS TO ESTABLISH THEIR FUNCTIONING
# AND VERIFY CHOICES OF CONVENTION

def find_axes_recursion(ind_vars_cfg_p, valid_axes_p, history):

    # Tail recursive into ind_var_cfgs_p
    pass

def find_valid_axes_cfgs_for_one_phasematch(ind_vars):

    # Recurse over the independent variable collections to determine all valid axis cfgs
    seed_hist = []
    valid_ax_cfgs = []
    find_axes_recursion(ind_vars_p, valid_axes, seed_hist)

    # Valid ax cfg format:
    # [[signed pulse id(s) for one axis], [signed pulse id(s) for other axis], ...]

    return valid_ax_cfgs

def find_valid_axes(all_ind_var_cfgs_p):

    # Find valid axes for each phase-matching direction
    valid_axes_p = []

    for i in all_ind_var_cfgs_p:

        valid_axes_p.append(find_valid_axes_cfgs_for_one_phasematch(i))

    final_valid_ind_vars = []

    for i in valid_axes_p:

        # for several PM directions, take intersection of cfgs shared between all PM directions and
        # assign this to final_valid_ind_vars
        pass

    # Format of final_valid_ind_vars: [valid axis cfg 1, ..]

    return final_valid_ind_vars


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

        self.dim = self.findDimensionality()
        self.epochs = find_epochs(self.field)
        self.int_sequences = self.findInteractionSequences()
        self.cfuv = get_carrier_freqs_uv(self.field.pulses)

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

            interactionRecurse(int_sequences, int_seed, i, 0, find_epochs(self.field))

        return int_sequences
