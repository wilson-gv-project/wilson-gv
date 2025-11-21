from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Iterable
from operator import itemgetter
import copy

from wilson_suite.wilson_derive.abstractions import HarmOscStateSymbolic

@dataclass
class SignedPulseTuple:
    """
    Class to represent a collection of signed references to pulse IDs. Applicable as a representation of an independent
    variable, but not limited to this.

    pulse_refs: A tuple of signed references to pulse IDs
    Example: For the independent variable -w1 + w2, a pulse_refs representation is (-1, 2)
    Example: For the phase-matching condition -k1 + k2 + k3, a pulse_refs representation is (-1, 2, 3)
    """
    pulse_refs: tuple

@dataclass
class PhaseMatchingCondition:
    """
    Class to represent a phase-matching condition

    pulse_refs: A SignedPulseTuple instance defining the phase-matching conditions
    phasematch_cond_id: Optional integer argument defining an identifier for this phase-matching condition (default: None)
    """

    pulses: SignedPulseTuple
    phasematch_cond_id: int = None

@dataclass
class IndependentVariableSet:
    """
    Class to represent a group of independent variables.

    var_set: A tuple of SignedPulseTuple instances
    """

    var_set = tuple[SignedPulseTuple]

@dataclass
class IndependentVariableChoices:
    """
    Class to represent a valid set of independent variables choices for a phase-matching condition

    phasematch_cond: A PhaseMatchingCondition instance defining the phase-matching condition for which this variable
    set is specified

    ind_var_groups: A three-fold tuple structure defining valid groups of independent variables as
    (((group 1 variable 1), (group 1 variable 2), ...), ((group 2 variable 1), (group 2 variable 2), ...)

    Example: If the valid groups of independent variables are the pair (-w1, w2) or the combination -w1 + w2,
    ind_var_groups is (A, B), where:
        - A is a IndependentVariableSet instance of the form (I, II), where:
            - I is a SignedPulseTuple instance with pulse_refs attribute = (-1,)
            - II is a SignedPulseTuple instance with pulse_refs attribute = (2,)
        - A is a IndependentVariableSet instance of the form (III,), where:
            - III is a SignedPulseTuple instance with pulse_refs attribute = (-1, 2)
    """
    phasematch_cond: PhaseMatchingCondition
    var_groups: tuple[IndependentVariableSet]

@dataclass
class SpectralAxis:
    """
    Class to represent a choice of spectral axis

    label: A string describing the name of this axis
    """

    label: str
    var_group: IndependentVariableSet

@dataclass
class SpectralAxisChoice:
    """
    Class to represent a full choice of spectral axes

    label: A string describing the name of this axis

    """
    axes: tuple


@dataclass
class SpectralAxisChoiceSet:
    """
    Class to represent a valid set of choices of axes for a choice of independent variables

    ind_vars: A two-fold tuple describing a choice ("group") of independent variables in terms of (signed) pulse IDs
    ((variable 1 (all signed) pulse ID 1, ID 2, ...), (variable 2 (all signed) pulse ID 1, ID 2, ...)). See explanation
    of ind_var_groups in class IndependentVariableChoice set for more details.

    valid_axis_combs: A tuple of dictionaries, each dictionary describing a valid choice of axes for this choice of
    independent variables. Each of these dictionaries is structured in the form
    {axis label: tuple of independent variables whose sum defines the axis}.

    Example 1: For the independent variables -w1 and w2, suppose that two valid choices of axes were identified:
        - Axis 'A': -w1 and axis 'B': w2
        - Axis 'A': -w1 and axis 'B': -w1 + w2

    Then valid_axis_combs would be of the form (I, II) where
        - I is SpectralAxisChoice instance with 'axes' attribute of the form (i, ii), where
            - i is a SpectralAxis instance with label attribute 'A' and var_group attribute with the var_set attribute
            on the form (X,), where
                - X is a SignedPulseTuple instance with the pulse_refs attribute (-1,)
            - ii is a SpectralAxis instance with label attribute 'B' and var_group attribute with the var_set attribute
            on the form (X,), where
                - X is a SignedPulseTuple instance with the pulse_refs attribute (2,)
        - II is SpectralAxisChoice instance with 'axes' attribute of the form (i, ii), where
            - i is a SpectralAxis instance with label attribute 'A' and var_group attribute with the var_set attribute
            on the form (X,), where
                - X is a SignedPulseTuple instance with the pulse_refs attribute (-1,)
            - ii is a SpectralAxis instance with label attribute 'B' and var_group attribute with the var_set attribute
            on the form (X, Y), where
                - X is a SignedPulseTuple instance with the pulse_refs attribute (-1,)
                - X is a SignedPulseTuple instance with the pulse_refs attribute (2,)

    Example 2: If there were two independent variables -w1 + w2 and w3, suppose that two valid choices of axes were
    identified
        - Axis 'A': -w1 + w2 and axis 'B': w3
        - Axis 'A': -w1 + w2 and axis 'B': -w1 + w2 + w3

    Then valid_axis_combs would be of the form (I, II) where
    - I is SpectralAxisChoice instance with 'axes' attribute of the form (i, ii), where
        - i is a SpectralAxis instance with label attribute 'A' and var_group attribute with the var_set attribute
        on the form (X,), where
            - X is a SignedPulseTuple instance with the pulse_refs attribute (-1, 2)
        - ii is a SpectralAxis instance with label attribute 'B' and var_group attribute with the var_set attribute
        on the form (X,), where
            - X is a SignedPulseTuple instance with the pulse_refs attribute (3,)
    - II is SpectralAxisChoice instance with 'axes' attribute of the form (i, ii), where
        - i is a SpectralAxis instance with label attribute 'A' and var_group attribute with the var_set attribute
        on the form (X,), where
            - X is a SignedPulseTuple instance with the pulse_refs attribute (-1, 2)
        - ii is a SpectralAxis instance with label attribute 'B' and var_group attribute with the var_set attribute
        on the form (X, Y), where
            - X is a SignedPulseTuple instance with the pulse_refs attribute (-1, 2)
            - X is a SignedPulseTuple instance with the pulse_refs attribute (3,)

    """

    ind_vars: IndependentVariableSet
    valid_axis_combs: tuple[SpectralAxisChoice]


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
                    self.wv = [i/wv_len for i in self.wv]

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


def find_subsets_making_orig(subsets: list, acc: list, orig: list, res: list):
    """
    Identify which collections of subsets are valid full partitions of original set orig; tail-recursive.

    subsets: A list of subsets to be checked by this function
    acc: list: In-recursion accumulator (candidate)
    orig: list: The original set w.r.t. which to compare
    res: Accumulated results
    """

    # Helper function: Collapse list of iterables into one-fold list
    def collapse(s):
        c = []
        for i in s:
            c.extend(list(i))
        return c

    # Pruning criterion: No use in recursion if candidate list is already longer than orig
    if len(collapse(acc)) > len(orig):
        return

    # Check at recursion end: Does acc correspond to a original list which is non-empty?
    if (sorted(collapse(sorted(acc))) == sorted(orig)) and (len(orig) > 0):
        res.append(acc)

    # Recursion
    if len(subsets) > 0:

        for i in range(len(subsets)):

            new_acc = copy.deepcopy(acc)
            new_acc.append(subsets[i])

            # Pass remaining subsets to further recursion
            if i - 1 < len(subsets):
                find_subsets_making_orig(subsets[i + 1:], new_acc, orig, res)

            else:
                find_subsets_making_orig([], new_acc, orig, res)

def find_branching_indep_var_combs(combs: list, orig_vars: list, curr_comb: list, curr_epoch: int):
    """
    Make branching choices of different valid UV/VIS partitioning choices (tail-recursive)

    combs: list: Accumulated combinations (tail-recursion result)
    orig_vars: list: Original variables (with branches as multi-entry lists)
    curr_comb: list: Branching choice in-recursion accumulator
    curr_epoch: int: Current epoch counter
    """

    if curr_epoch == len(orig_vars):
        combs.append(copy.deepcopy(curr_comb))

    else:
        new_comb_ir = copy.deepcopy(curr_comb)

        uv_start = 0
        for i in orig_vars[curr_epoch]:
            if not isinstance(i, list):
                uv_start += 1
                new_comb_ir.append(i)

        if not(uv_start == (len(orig_vars[curr_epoch]))):

            for i in orig_vars[curr_epoch][uv_start]:

                new_comb_ir_uv = copy.deepcopy(new_comb_ir)
                new_comb_ir_uv.extend(i)

                find_branching_indep_var_combs(combs, orig_vars, new_comb_ir_uv, curr_epoch + 1)

        else:

            find_branching_indep_var_combs(combs, orig_vars, new_comb_ir, curr_epoch + 1)


def find_indep_vars_for_one_phasematch(pulses: list[EmPulse], epochs: list, pm_dir: dict) -> list:
    """
    Determine possible (non-ordered) configurations of (IR-range) independent variables for a set of IR or UV/VIS-range pulses
    for a given phase-matching condition.

    pulses: List of EmPulse instances
    epochs: List of epochs where the pulses (referred to by their IDs) are grouped in time
    pm_dir: Dictionary of (pulse ID: phase-matching condition sign) pairs

    Returns: ind_vars_p: The collection of identified independent variables for that phase-matching condition, formatted as a
                        list of lists, where the outer list represents one full collection of independent
                        variables, and the inner list represents the specific independent variables for that entry.
    """

    raw_ind_vars_p = []
    cfuv = get_carrier_freqs_uv(pulses)

    # Loop over epochs
    for i in range(len(epochs)):

        raw_ind_vars_p_epoch = []
        cfuv_this_pm = {}
        uv_this = []
        ir_this = []

        # Pulses in epoch
        for k in epochs[i]:

            # Dress UV/VIS-range pulses with phase-matching sign
            if not(cfuv[k] == 0.0):
                cfuv_this_pm[k] = cfuv[k] * pm_dir[k]
                uv_this.append(k * pm_dir[k])

            else:
                ir_this.append(k * pm_dir[k])

        uv_this = sorted(uv_this)
        ir_this = sorted(ir_this)

        from itertools import chain, combinations

        # Find the superset (less empty entry) of all UV/VIS pulses in this epoch
        uv_superset = set(list(chain.from_iterable(combinations(uv_this, r) for r in range(len(uv_this) + 1)))[1:])
        uv_superset_cancel = []

        # Which subsets have UV/VIS components that cancel?
        for j in uv_superset:

            if uv_cancels(j, cfuv_this_pm):
                uv_superset_cancel.append(j)

        acc = []
        uv_subs_res = []

        # Which collections of UV/VIS pulses that cancel do together add up to the full set of UV/VIS pulses in this epoch?
        if i < (len(epochs) - 1):
            find_subsets_making_orig(uv_superset_cancel, acc, uv_this, uv_subs_res)
        else:
            # In last epoch they do not need to cancel since the resulting signal is presumed to go to the detector
            for j in uv_superset:
                acc = []
                find_subsets_making_orig(uv_superset_cancel, acc, list(j), uv_subs_res)

        # IR-range pulses become independent variables directly
        for j in ir_this:
            raw_ind_vars_p_epoch.append(tuple([j]))

        # Different UV/VIS partitions of all the UV/VIS pulses in this epoch become branching options
        if not(uv_subs_res == []):
            raw_ind_vars_p_epoch.append(uv_subs_res)

        # No need to add if no independent variables found
        # In this case, there could be a "missing" set of frequencies if no UV/VIS collections were found to
        # be valid in this epoch, but in that case, there should in all likelihood also be no surviving wilson-derive terms
        if (len(raw_ind_vars_p_epoch) > 0):
            raw_ind_vars_p.append(copy.deepcopy(raw_ind_vars_p_epoch))

    ind_vars_p = []
    seed_comb = []

    # Do the branching combinatorics over any UV/VIS partitions with more than one option
    find_branching_indep_var_combs(ind_vars_p, raw_ind_vars_p, seed_comb, 0)

    return ind_vars_p

def find_indep_exp_variables(pulses: list[EmPulse], epochs: list, phasematch_dirs: dict) -> dict:
    """
    Outer loop over phase-matching conditions for use with find_indep_vars_for_one_phasematch.

    pulses: List of EmPulse instances
    epochs: List of epochs where the pulses (referred to by their IDs) are grouped in time
    phasematch_dirs: Dictionary: Each entry (key: phase-matching condition ID)
                    corresponds to a phase-matching condition/direction. Each direction is
                    itself a dictionary with pulse ID: sign pairs for entries.

    Returns: all_ind_var_cfgs_p: A dictionary over the phase-matching condition IDs. Each entry is the set of
                                 found independent variables for that phase-matching condition, formatted as a
                                 list of lists, where the outer list represents one full collection of independent
                                 variables, and the inner list represents the specific independent variables for that
                                 entry.
    """

    all_ind_var_cfgs_p = {}

    for p in phasematch_dirs:
        all_ind_var_cfgs_p[p] = copy.deepcopy(find_indep_vars_for_one_phasematch(pulses, epochs, phasematch_dirs[p]))

    return all_ind_var_cfgs_p

def find_axes_recursion(ind_vars: tuple, valid_axes: list, curr_ax_list: list, pos: int):
    """
    Find valid axes for one independent variables choice (tail-recursive)

    ind_vars: tuple of tuples: One independent variables choice.
    valid_axes: list: Accumulated sets of valid axes
    curr_ax_list: intra-recursion accumulator
    pos: recursion depth counter: Which indep. var. position is considered here?

    Valid axis choices either use the independent variable by itself or together with any nonempty subset of the
    preceding independent variables
    """

    # Recursion termination
    if pos == len(ind_vars):
        if not [sorted(i) for i in curr_ax_list] in valid_axes:
            valid_axes.append([sorted(i) for i in curr_ax_list])

    else:

        from itertools import chain, combinations

        # Make superset and recurse over it
        prev_var_superset = set(list(chain.from_iterable(combinations(ind_vars[:pos], r) for r in range(len(ind_vars[:pos]) + 1)))[1:])

        tmp_ax_list = copy.deepcopy(curr_ax_list)
        tmp_ax_list.append([ind_vars[pos]])
        new_ax_list = copy.deepcopy(tmp_ax_list)

        # Recursion without contribution from superset ("independent variable by itself")
        find_axes_recursion(ind_vars, valid_axes, new_ax_list, pos + 1)

        for i in prev_var_superset:

            new_ax_list = copy.deepcopy(tmp_ax_list)
            new_ax_list[len(new_ax_list) - 1].extend(i)

            # New recursion with entry from superset
            find_axes_recursion(ind_vars, valid_axes, new_ax_list, pos + 1)

def find_valid_axes_cfgs_for_one_phasematch(ind_vars: list) -> dict[tuple, list[dict]]:
    """
    Find valid axes choices for one phase-matching direction.

    ind_var: List of lists of tuples: independent variables for one phase-matching condition.
                    See return structure of find_indep_exp_variables.

    Returns: dict: valid_axes: For each set of independent variable choices (keys), return a list of
    valid axis choices. Each such list entry is a dictionary {dummy axis label: list of independent variables
    comprising axis}
    """


    from wilson_suite.wilson_utils.common_labels import cap_alpha_labels
    from itertools import permutations

    valid_axes = {}
    seed_ax_list = []

    # Loop over independent variable collection choices
    for i in ind_vars:

        curr_valid_axes = []

        # Permute each independent variable collection for full combinatorics; accumulate new entries (inside recursion)
        for j in permutations(i):
            find_axes_recursion(j, curr_valid_axes, seed_ax_list, 0)

        curr_valid_axes = sorted(copy.deepcopy(curr_valid_axes))

        # Dress the found axis entries with dummy labels
        dressed_valid_axes = []

        for j in curr_valid_axes:

            new_dress_v_a = {}

            for k in range(len(j)):
                if k > 2:
                    raise ValueError('Current version enables maximum 3 axes')
                new_dress_v_a[cap_alpha_labels[k]] = j[k]

            dressed_valid_axes.append(copy.deepcopy(new_dress_v_a))

        # Enter in dictionary by sorted independent variable tuples
        valid_axes[tuple(sorted(i))] = copy.deepcopy(dressed_valid_axes)

    return valid_axes

def find_canonical_axes_for_one_phasematch(ind_var_cfgs_p: list) -> dict[str, list[tuple]]:
    """
    Find canonical axes for one phase-matching direction.

    ind_var_cfgs_p: List of lists: independent variables for one phase-matching condition.
                    See return structure of find_indep_exp_variables.

    Returns: dict: canonical_axes: For the canonical independent variable choice,
                   return a dictionary {axis dummy label: independent variable}

    The canonical independent variable choice is taken to be that which gives the greatest number of
    independent variables. If more than one such entry, then the canonical choice is the minimum
    such choice after numerically sorting

    The canonical axis choice is the choice of axes that uses each of the canonical independent variables
    exactly once for each axis, with dummy axis labels affixed in the same order as the ordering for the
    canonical independent variables
    """

    from wilson_suite.wilson_utils.common_labels import cap_alpha_labels
    from itertools import permutations

    max_len_ind = 0

    canonical_axes = {}

    # Find entry/-ies with the most independent variables
    for i in ind_var_cfgs_p:
        if len(i) == max_len_ind:
            max_len_entries.append(copy.deepcopy(sorted(i)))

        elif len(i) > max_len_ind:
            max_len_ind = len(i)
            max_len_entries = [copy.deepcopy(sorted(i))]

    max_len_entries = sorted(max_len_entries)

    # Since max len entries is sorted, I can make a canonical choice with the first entry
    for i in range(len(max_len_entries[0])):
        if i > 2:
            raise ValueError('Current version enables maximum 3 axes')
        
        # Dress canonical indepentent variables with axis labels
        canonical_axes[cap_alpha_labels[i]] = [max_len_entries[0][i]]

    return canonical_axes

# TODO: Have option to let user fix one or more axes and recurse starting from that instead
def find_canonical_axes(all_ind_var_cfgs_p: dict) -> dict[str, list[tuple]]:
    """
    Find canonical axes for a collection of phase-matching directions. FIXME: > 1 pm directions not yet supported

    all_ind_var_cfgs_p: Dictionary of independent variables. See return structure of find_indep_exp_variables.

    Returns: dict: final_canonical_axes: For the canonical independent variable choice
    (see find_canonical_axes_for_one_phasematch), return a dictionary {axis dummy label: independent variable}
    """

    from itertools import permutations

    canonical_axes_p = {}

    for i in all_ind_var_cfgs_p:
        canonical_axes_p[i] = find_canonical_axes_for_one_phasematch(all_ind_var_cfgs_p[i])

    final_canonical_axes = canonical_axes_p[0]

    # For several PM directions, take intersection of cfgs shared between all PM directions
    if len(canonical_axes_p) > 1:
        raise NotImplementedError('Support for axis determination over more than one phasematching direction not implemented')
        for i in valid_axes_p[1:]:
            final_canonical_axes = final_canonical_axes.intersection(i)

    return final_canonical_axes

# TODO: Have option to let user fix one or more axes and recurse starting from that instead
def find_valid_axes(all_ind_var_cfgs_p: dict) -> dict[tuple, list[dict[str, list[tuple]]]]:
    """
    Find valid axes for a collection of phase-matching directions. FIXME: > 1 pm directions not yet supported

    all_ind_var_cfgs_p: Dictionary of independent variables. See return structure of find_indep_exp_variables.

    Returns: dict: final_valid_axes: For each set of independent variable choices (keys), return a list of
    valid axis choices. Each such list entry is a dictionary {dummy axis label: list of independent variables
    comprising axis}
    """

    # Find valid axes for each phase-matching direction
    valid_axes_p = {}


    for i in all_ind_var_cfgs_p:

        valid_axes_p[i] = find_valid_axes_cfgs_for_one_phasematch(all_ind_var_cfgs_p[i])

    # Format of final_valid_ind_vars: set(valid axis cfg 1, ...)
    final_valid_axes = valid_axes_p[0]

    # For several PM directions, take intersection of cfgs shared between all PM directions
    if len(valid_axes_p) > 1:
        raise NotImplementedError('Support for axis determination over more than one phasematching direction not implemented')
        for i in valid_axes_p[1:]:
            final_valid_axes = final_valid_axes.intersection(i)

    return final_valid_axes


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

        relevant_phasematch = {}

        # If no specified phase-matching (wavevector) filter, all are (potentially) relevant
        if self.detector.wv_filter is None:

            from itertools import product as iter_prod

            k = 0

            for i in iter_prod([1, -1], repeat=len(self.field.pulses)):

                new_phasematch = {}
                for j in range(len(self.field.pulses)):
                    new_phasematch[self.field.pulses[j].id] = i[j]

                relevant_phasematch[k] = copy.deepcopy(new_phasematch)
                k += 1

        else:

            for i in range(len(self.detector.wv_filter)):

                relevant_phasematch[i] = self.detector.wv_filter[i]

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

