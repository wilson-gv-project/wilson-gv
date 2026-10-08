from code import interact
from dataclasses import dataclass
from itertools import cycle
from typing import Optional, Iterable
from operator import itemgetter
import copy
from math import inf as infinity
from math import exp as exp
import numpy as np

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

    detector_location: List of floats: Taking the system to be positioned at the origin,
    this parameter is the (unit) vector describing the direction along which the detector is located (i.e., the
    detector is located along this vector from the system and facing the opposite direction).
    Default: [0.0, 0.0, 1.0].

    detection_polarization: List of floats: Detect only light with this specific polarization vector. Default:
    [1.0, 0.0, 0.0].

    detection_range: List of floats: For "time" or "freq" detection, tell over which points (the range)
    in either t/E space as relevant the data is collected

    interaction_filter: List of interaction patterns as lists: Filter signal to include only this/these
    (signed) interaction pattern(s). Note that any ordering here is not taken as a causal interaction order requirement
    and also note that while this parameter can be thought of as close to a phase-matching filter, it is not exactly the
    same (VibExperiment setup may further consider other signals in the same phase-matching direction(s) or isolate
    specific interaction sequences)

    ignore_collinear: If using a wavevector filter, ignore other effects collinear with this/these direction(s)?
    Currently not used.
    """
    detection_method: str
    
    detector_location: Optional[tuple[float]] = None
    
    detection_polarization: Optional[tuple[float]] = None

    # FIXME: Rework to upper/lower bounds and (possibly) granularity?
    detection_range: Optional[list[float]] = None

    interaction_filter: Optional[list[list]] = None

    # FIXME: Unsure if this is a relevant attribute
    overall_phase: float = 0.0

    def __post_init__(self):
        if self.detection_method not in {'time', 'freq', 'int'}:
            raise ValueError("The detection type must be either 'time', 'freq'(uency), or 'int'(egrated)")

        if (self.detection_method in {'time', 'freq'}) and self.detection_range is None:
            raise ValueError("If the detection type is 'time' or 'freq', the detection range must be specified")

        # Derived property: Detection granularity
        if self.detection_range is not None:
            self.dlen = len(self.detection_range)

        if not self.overall_phase == 0.0:
            raise ValueError('Detector overall phase currently restricted to zero shift')

        # Will currently not be reach because of restriction to zero shift, but will be relevant when that is lifted
        if not(abs(self.overall_phase) - 1.0 < 1e-10):
            raise ValueError('Detector overall phase must be of unit length')

        if self.interaction_filter is not None:
            raise NotImplementedError('Non-specification of detector interaction pattern filter')

@dataclass
class ScanObject:
    """
    Class to represent some attribute of the experiment which could be scanned. Currently not in use.

    category: String: The (main) category of the object to be scanned, e.g. "pulse"
    subcategory: String: The subcategory of the object to be scanned, e.g. "cf" (carrier frequency)
    id: Integer: An integer identifier for the specific instance of the main category, e.g. if 'category' is "pulse"
        (of which there are typically several, labelled by integer indices), then the present 'id' attribute
        tells which of these pulses are meant in this scan object
    coeff: Float: Coefficient: As a scan is performed across a range (see SpecScan), 'coeff' tells which scaling to use
    for the scan range increments when applying them to the present scan object. For example, one might want to have a
    scan that increases one parameter according to the scan range while concurrently decreasing another parameter twice
    as rapidly. This can be represented by associating the scan with two scan objects referring to the respective
    parameters, where the first has 'coeff' == 1.0 and the second has 'coeff' == -2.0. Default: 1.0.
    """

    category: str
    subcategory: str
    id: int = 0
    coeff: float = 1.0

    def __post_init__(self):

        # TODO: Consider reintroducing detector range as allowed scanning attribute
        #  The intent is to allow shifting the detector range in lockstep with scanning freq/time maxima if that
        #  facilitates data organization

        # Checking against currently recognized scan categories/subcategories.
        # Presence in these lists is not sufficient to
        # indicate actual support for a specific scan.
        valid_scan_objs = ['pulse']
        valid_scan_attributes = {'pulse': ['cf', 'tc', 'dev']}

        if not self.category in valid_scan_objs:
            raise ValueError('Scan category not supported')
        if not self.subcategory in valid_scan_attributes[self.category]:
            raise ValueError('Scan subcategory not supported')

        # Classify if the scan affects only the pulse integration (convolution)
        # (e.g. is invariant over the response function), affects only the response function (is invariant over
        # the pulse integration), or affects both

        self.scan_affects = None

        if self.category == 'pulse':
            if self.subcategory in ['cf', 'tc', 'dev']:
                self.scan_affects = 'integration'

        if self.scan_affects == None:
            raise ValueError('Scan subcategory domain of effect is indeterminate')


@dataclass
class SpecScan:
    """
    Class to represent a spectral scan (adding to the dimensionality of a spectrum)

    scan_objs: Tuple of ScanObject instances: Tells what this scan will vary. Their IDs must be different

    range: Iterable over which scan objects are varied (scaled by their multipliers as represented by
    their 'coeff' attributes)

    TODO: Support scanning concurrently over several attributes: Either extend this in ScanObject or
     here in SpecScan specify dependencies between singleton scan objects

    TODO: Support several ranges and link each to a scan object for situations where attributes to be scanned
     simultaneously don't live in the same space (e.g. polarization (vector) and carrier frequency (scalar))

    TODO: Consider allowing detector range as a scan object when combined with a relevant attribute (for
     having detector results "lockstep" with such scan if that facilitates data organization)
    """

    scan_objs: tuple[ScanObject]
    range: Iterable

    def __post_init__(self):

        from collections.abc import Iterable

        if not (isinstance(self.scan_objs, tuple)):
            raise TypeError('scan_objs must be a tuple of ScanObject instances')
        else:
            for i in self.scan_objs:
                if not (isinstance(i, ScanObject)):
                    raise TypeError('scan_objs must be a list of ScanObject instances')

        if not isinstance(self.range, Iterable):
            raise TypeError('range is not iterable')

@dataclass
class EmPulse:
    """
    Class to represent an electromagnetic pulse.

    Conventions: In other functions using this class with Gaussian envelopes, parameters with dimensions of time are
    assumed to be given in units of fs, and parameters with dimensions of frequency (energy) are asseumed
    to be given in units of cm^-1.
    
    ----
    env: String: Pulse time-domain envelope type: The only current valid choice is "gaussian".
    maxstr: Float: Pulse amplitude at maximum of envelope. Default: Zero strength
    tc: Float: Point in time at which pulse envelope is at maximum
    cf: float: Carrier (angular) frequency
    dev: float: Time deviation parameter (e.g. time-domain sqrt(variance) of Gaussian pulse: In other functions using
    this class with Gaussian envelopes, dev = sigma in A * e^( -(t- tc)**2 / (2 sigma^2)).
    An impulsive-like pulse can be considered as the dev -> 0.0 limit of a Gaussian pulse.
    "continuous-wave"-like pulse can be considered as the dev -> infty limit of a Gaussian pulse.
    wv: floats: Unit wavevector propagation direction with respect to laboratory axes
    pol: floats: Polarization: Unit vector describing polarization direction with respect to laboratory axes
        - Must be orthogonal to wavevector
        - Only linear polarization currently supported (no phase difference between orthogonal components of
        polarization vector in plane of polarization)
        - Default: (1.0, 0.0, 0.0)

    pol_comp: Tuple of tuples of floats: Polarizations under pulse compounding. Assumed ordering is same ordering
    as id_comp

    overall_phase: complex number defining a unit vector in the complex plane: Overall phase of pulse, expressed as a
    float and understood as the exponent theta of e^(i theta)
    Currently enforced as 0.0

    id: integer: Pulse ID label

    id_comp: Tuple: Pulse ID label(s) (used for pulse compounding)

    is_compound: Boolean: Is this a compound pulse?
    """

    env: str
    
    tc: float = None
    cf: float = None
    dev: float = None

    maxstr: float = 0.0

    wv: tuple[float] = None
    pol: tuple[float] = None
    pol_comp:  tuple[tuple[float]] = None
    overall_phase: complex = 0.0

    id: int = None
    id_comp: tuple = None
    is_compound = False

    def __post_init__(self):
        
        # Currently only Gaussian envelopes allowed
        allowed_envelopes = ['gaussian']

        if self.env not in allowed_envelopes:
            raise ValueError('The only current recognized pulse envelope choice is "gaussian"')

        if self.env == 'gaussian':
            if self.cf is None:
                if not self.dev == 0.0:
                    raise AssertionError('A non-impulsive-tending Gaussian pulse must have a carrier frequency')
            if self.tc is None:
                if not self.dev == infinity:
                    raise AssertionError('A non-continuous-wave-tending Gaussian pulse must have a parameter describing the time at which the temporal envelope is at its maximum')
            if self.dev is None:
                raise AssertionError('A Gaussian pulse must have a time deviation parameter')

        # Wavevector: In which unit vector direction is the pulse wave travelling
        if isinstance(self.wv, tuple):
            if len(self.wv) == 3:
                if all([isinstance(i, float) for i in self.wv]):
                    self.wv = tuple(self.wv)
                else:
                    raise AssertionError('The pulse wavevector must be a len 3 tuple of floats')
            else:
                raise AssertionError('The pulse wavevector must be a len 3 tuple of floats')
        else:
            raise AssertionError('The pulse wavevector must be a len 3 tuple of floats')

        # Polarization: Specify the polarization of the pulse
        # Currently supports unit linear polarization
        # Checks disregarded for compound polarization
        if self.pol_comp is None:
            if isinstance(self.pol, tuple):
                if len(self.pol) == 3:
                    if all([isinstance(i, float) for i in self.pol]):
                        pol_len = (self.pol[0]**2.0 + self.pol[1]**2.0 + self.pol[2]**2.0)**0.5
                        if not pol_len == 1.0:
                            print('Polarization vector was normalized')
                        self.pol = tuple([i/pol_len for i in self.pol])

                        wv_pol_dot = self.pol[0] * self.wv[0] + self.pol[1] * self.wv[1] + self.pol[2] * self.wv[2]

                        if not(wv_pol_dot == 0.0):
                            raise AssertionError('Error: Wavevector of pulse not orthogonal to polarization vector')

                    else:
                        raise AssertionError('The polarization vector must be a len 3 tuple of floats')
                else:
                    raise AssertionError('The polarization vector must be a len 3 tuple of floats')
            else:
                raise AssertionError('The polarization must be a len 3 tuple of floats')

        if not self.overall_phase == 0.0:
            raise AssertionError('Overall phase currently restricted to zero shift')

        if not((abs(self.overall_phase) - 1.0) < 1e-10):
            raise AssertionError('The overall phase must be of unit length')

    def tendsImpulsive(self):
        """
        Does this pulse tend to impulsive?
        """

        if self.env == 'gaussian':
            if self.dev == 0.0:
                return True
            return False
        else:
            raise ValueError('"Tends-impulsive" check currently not implemented for non-Gaussian pulses')

    def tendsContinuous(self):
        """
        Does this pulse tend to being continuous-wave-like?
        """

        if self.env == 'gaussian':
            if self.dev == infinity:
                return True
            return False
        else:
            raise ValueError('"Tends-impulsive" check currently not implemented for non-Gaussian pulses')

# FIXME: Here and next two fns: Change to be in terms of f(*, kw1=kw1, ...) style
def make_gaussian_pulse(tc: float, cf: float, dev: float, cf_uv: float = 0.0, maxstr: float=0.0,
                        wv: tuple[float] = (0.0, 0.0, 1.0), pol: tuple[float] = (1.0, 0.0, 0.0),
                        overall_phase: complex = 1.0 + 0.0j, id: int = None):
    """
    Helper function: Makes a Gaussian EmPulse instance with the selected parameters.
    See EmPulse for a definition of these parameters.

    Required arguments: tc, cf, dev.
    """

    return EmPulse(env = 'gaussian', tc = tc, cf = cf, dev = dev, maxstr = maxstr, wv = wv,
                   pol = pol, overall_phase = overall_phase, id = id)


def make_impulsive_gaussian_pulse(tc: float, cf: float = None, cf_uv: float = 0.0, maxstr: float=0.0,
                         wv: tuple[float] = (0.0, 0.0, 1.0), pol: tuple[float] = (1.0, 0.0, 0.0),
                         overall_phase: complex = 1.0 + 0.0j, id: int = None):
    """
    Helper function: Makes an impulsive-tending Gaussian EmPulse instance with the selected parameters.
    See EmPulse for definitions of these parameters. The present function does not accept a dev parameter
    because it is zero for an impulsive pulse and this is carried out explicitly in the call below. All
    other parameters are transferred directly to the EmPulse instance creation.

    Required arguments: tc. All other arguments optional.
    """

    return EmPulse(env = 'gaussian', tc = tc, cf = cf, dev = 0.0, maxstr = maxstr, wv = wv,
                   pol = pol, overall_phase = overall_phase, id = id)

def make_cw_gaussian_pulse(cf: float, tc: float = None, cf_uv: float = 0.0, maxstr: float=0.0,
                         wv: tuple[float] = (0.0, 0.0, 1.0), pol: tuple[float] = (1.0, 0.0, 0.0),
                         overall_phase: complex = 1.0 + 0.0j, id: int = None):
    """
    Helper function: Makes a continuous-wave-tending Gaussian EmPulse instance with the selected parameters.
    See EmPulse for definitions of these parameters. The present function does not accept a dev parameter
    because it is infinity for a cw pulse and this is carried out explicitly in the call below. All
    other parameters are transferred directly to the EmPulse instance creation.

    Required arguments: cf. All other arguments optional. NOTE changed ordering of cf and tc arguments compared to
    other make_..._pulse functions because of this optionality.
    """

    return EmPulse(env = 'gaussian', tc = tc, cf = cf, dev = infinity, maxstr = maxstr, wv = wv,
                   pol = pol, overall_phase = overall_phase, id = id)

def make_compound_gaussian_pulse(p1: EmPulse, p2: EmPulse = None) -> EmPulse:
    """
    Make a compound Gaussian pulse from two EmPulse instances.

    p1, p2: The EmPulse instances to be compounded (p2 is optional)

    Returns: An EmPulse instance representing the compounded pulse
    """

    if not p1.env == 'gaussian':
        raise ValueError('Pulses to be compounded must be Gaussian')

    if p2 is None:



        comp_pulse = copy.deepcopy(p1)

        comp_pulse.pol_comp = (copy.deepcopy(p1.pol),)
        comp_pulse.pol = None

        comp_pulse.id_comp = (copy.deepcopy(p1.id),)
        comp_pulse.id = None

        return comp_pulse

    else:

        if not p2.env == 'gaussian':
            raise ValueError('Pulses to be compounded must be Gaussian')

        # Combining attributes of p1 and p2
        # Here using the fact that the product of Gaussians is another Gaussian with specific expressions for
        # the new means and variances
        p1dsq = (p1.dev) ** 2
        p2dsq = (p2.dev) ** 2
        p12dsq = p1dsq + p2dsq

        # New time maximum, carrier frequency, deviation, overall coefficient
        new_tc = (p1.tc * p1dsq + p2.tc * p2dsq) / p12dsq
        new_cf = p1.cf + p2.cf
        new_dev = (p1dsq * p2dsq / p12dsq) ** 0.5
        new_maxstr = p1.maxstr * p2.maxstr * exp(-1 * ((p1.tc - p2.tc) ** 2) / (2 * p12dsq))

        # New wavevector as sum
        new_wv = tuple([p1.wv[j] + p2.wv[j] for j in range(3)])

        # Update polarization
        if p1.is_compound:
            lcppc = list(p1.pol_comp)
        else:
            lccpc = [p1.pol]

        if p2.is_compound:
            lcppc.extend(p2.pol_comp)
        else:
            lcppc.append(p2.pol)

        new_pol_comp = tuple(lcppc)

        # New phase is sum
        new_overall_phase = p1.overall_phase + p2.overall_phase

        # Update ID
        if p1.is_compound:
            nic = list(p1.id_comp)
        else:
            nic = [p1.id]

        if p2.is_compound:
            nic.extend(p2.id_comp)
        else:
            nic.append(p2.id)

        new_id_comp = tuple(nic)

        # Make compound pulse
        return EmPulse(env='gaussian', tc=new_tc, cf=new_cf, dev=new_dev, maxstr=new_maxstr,
                       wv=new_wv, pol=None, pol_comp=new_pol_comp, overall_phase=new_overall_phase,
                       id=None, id_comp=new_id_comp, is_compound=True)


def compound_pulses_in_list(pulses: list[EmPulse]) -> EmPulse:

    if not all([i.env == 'gaussian' for i in pulses]):
        raise ValueError('Pulses to be compounded must be Gaussian')

    comp_acc = make_compound_gaussian_pulse(p1 = pulses[0])

    for i in pulses[1:]:
        comp_acc = make_compound_gaussian_pulse(p1 = comp_acc, p2 = i)

    return comp_acc

# The field consists of a collection of pulses
@dataclass
class ElectricField:
    """
    Class to represent an electromagnetic field consisting of one or more pulses
    
    ----
    pulses: List of EmPulse instances: The pulses making up the field. While EmPulse itself does not require an ID to
    be specified, the use of pulses in ElectricField must have each pulse be assigned an ordinal integer ID starting from 1
    These pulses are understood as "hosting" a cos(w_c t + k*r) oscillatory factor; post-init
    will further divide them into individual positive/negative e^(i (-w_c t + k*r) + c.c. components
    """

    pulses: tuple[EmPulse]

    def __post_init__(self):

        # NOTE: This restriction can be softened somewhat to pulse IDs needing to be a positive integer
        # (but not necessarily together correspond exactly to the ordinal list)
        pulse_id_target = [i + 1 for i in range(len(self.pulses))]
        for i in self.pulses:
            if not i.id in pulse_id_target:
                raise ValueError('Pulse ID', i.id, ' of field not found in ordinal list')
            else:
                pulse_id_target.remove(i.id)

        # This condition should never be met but just to be safe
        if not pulse_id_target == []:
            raise AssertionError('Collection of pulse IDs do not correspond to ordinal list')

        if all([i.tendsContinuous() for i in self.pulses]):
            self.cw_field = True
        else:
            self.cw_field = False

        if all([i.tendsImpulsive() for i in self.pulses]):
            self.impulsive_field = True
        else:
            self.impulsive_field = False

        # Making signed field: A dictionary corresponding to pulse IDs, divided into positive/negative
        # freq/wavevector components. Negative components will have IDs = -1 * original ID.
        # The pulses created here will be understood to have an oscillatory factor e^(i (-w_c t + k*r) or c.c.
        # Organizing in dictionary when the pulse ID is also in the instance is redundant but kept for convenience

        signed_pulses = {}

        for i in self.pulses:

            # Positive component
            signed_pulses[i.id] = copy.deepcopy(i)

            # Negative component
            # FIXME: I think the polarization and overall phase are complex conjugated here, should verify this
            signed_pulses[-1 * i.id] = EmPulse(env = i.env, tc = i.tc, cf = -1 * i.cf, dev = i.dev, maxstr = i.maxstr,
                                               wv = tuple([-1.0*j for j in i.wv]),
                                               pol = tuple([j.real - j.imag for j in i.pol]),
                                               overall_phase = -1 * i.overall_phase, id = -1 * i.id)

            self.signed_pulses = signed_pulses


    # Take a reference to an interaction pattern and return all interaction patterns whose vector sum
    # corresponds to the same direction
    def wavevectors_matching_ids(self, ids: list | tuple, filter: str='same_order') -> list[tuple]:
        """
        ids: List or tuple making up an interaction pattern (ordering arbitrary).
        Example: To represent the phase-matching direction k1 - k1 + k2 + k3, ids = [2, -1, 3, 1] is a valid choice
        (as are all its permutations)

        filter: A flag to constrain the search (default and currently only supported: 'same_order', which will
        make this function return only those patterns which both correspond to the same direction and contain the
        same number of interactions as ids, and 'up_to_order', which will additionally return all lower-order
        combinations summing to this pattern.

        Returns: A list of tuples: Each tuple is a (sorted and filtered to unique after sorting) interaction pattern
        whose wavevector will be oriented in the same direction as that corresponding to ids
        """
        from itertools import combinations_with_replacement as combs

        valid_filters = ['same_order', 'up_to_order', 'only_specified']

        if not filter in valid_filters:
            raise NotImplementedError('Unrecognized filter in wavevectors_matching_ids')

        sorted_ids = sorted(list(ids))

        # Make target wavevector
        target_wv = [0.0, 0.0, 0.0]
        for i in sorted_ids:
            target_wv = [target_wv[j] + self.signed_pulses[i].wv[j] for j in range(3)]

        # Normalize
        t_len = (sum([k ** 2 for k in target_wv])) ** 0.5
        if not t_len == 0.0:
            target_wv = [j/t_len for j in target_wv]

        # The input interaction pattern trivially matches
        matching_wv = [tuple(sorted_ids)]

        if filter == 'only_specified':
            return matching_wv

        elif filter == 'same_order':
            ord_start = len(sorted_ids)
            ord_end = len(sorted_ids) + 1

        elif filter == 'up_to_order':
            ord_start = 1
            ord_end = len(sorted_ids) + 1

        else:
            raise NotImplementedError('Unrecognized filter in wavevectors_matching_ids')


        all_ids = self.signed_pulses.keys()

        parallel_tol = 1.e-10

        for ord in range(ord_start, ord_end):

            # Make all unique combinations of signed pulses keys of len ord
            for c in combs(all_ids, ord):

                # Make candidate wavevector
                cand_wv = [0.0, 0.0, 0.0]
                for i in c:
                    cand_wv = [cand_wv[j] + self.signed_pulses[i].wv[j] for j in range(3)]

                c_len = (sum([k ** 2 for k in cand_wv])) ** 0.5
                if not c_len == 0.0:
                    cand_wv = [j / c_len for j in cand_wv]

                if abs(sum([cand_wv[i] * target_wv[i] for i in range(3)]) - 1) < parallel_tol:

                    tc = tuple(sorted(c))
                    if not tc in matching_wv:
                        matching_wv.append(tc)

        return matching_wv

    def overlapping_pulses_for_interaction_pattern(self, pattern: list | tuple, tol_n_dev: float | int = 5.0) -> tuple[tuple]:
        """
        Calculate and tell which pulses of this field have overlapping temporal envelopes
        under a tolerance parameter for the given interaction pattern.
        This routine currently only supports this calculation for Gaussian-envelope pulses.

        pattern: List or tuple describing interactions with field. Ordering is arbitrary and
        duplicates are fine.

        tol_n_dev: Tolerance parameter (for Gaussian-envelope pulses): Each pulse's
        "region of influence" is taken as its arg(time envelope max) +/- tol_n_dev * the pulse's
        time-domain deviation parameter. A tuple of pulses is ruled to have overlapping temporal
        envelope if the intersection of all relevant pulse regions is nonempty.

        Returns: A tuple of tuples. Each "inner" tuple is a tuple of pulse IDs ruled to be
        overlapping in time. Any tuple not in the return data was ruled to not overlap in time.
        """

        #TODO: Make this w.r.t. a phase-matching condition and signed fields

        from itertools import combinations, chain
        from math import inf as infinity

        pattern_sorted = sorted(list(pattern))

        # Make interval around each pulse as dictionary
        pulse_intervals = {}

        for i in pattern_sorted:

            if not self.signed_pulses[i].env == 'gaussian':
                raise ValueError('overlapping_pulses currently only supports Gaussian-envelope pulses')

            if self.signed_pulses[i].tendsImpulsive():
                pulse_intervals[i] = [self.signed_pulses[i].tc, self.signed_pulses[i].tc]
            elif self.signed_pulses[i].tendsContinuous():
                pulse_intervals[i] = [-infinity, infinity]
            else:
                pulse_intervals[i] = [self.signed_pulses[i].tc - tol_n_dev * self.signed_pulses[i].dev,
                                      self.signed_pulses[i].tc + tol_n_dev * self.signed_pulses[i].dev]

        # Make powerset of pulse IDs
        pulse_powerset = list(chain.from_iterable(combinations(pattern_sorted, i) for i in range(len(pattern_sorted) + 1) ))

        # For each element in powerset (smaller to larger sets): Determine intersection
        # If intersection is nonempty, add to return data

        valid_intersections = []

        for i in pulse_powerset:

            # The empty tuple is not relevant
            if len(i) == 0:
                continue

            # Single pulses are trivially overlapping
            elif len(i) == 1:
                if not i in valid_intersections:
                    valid_intersections.append(tuple(i))

            # General case
            else:

                curr_interval = copy.deepcopy(pulse_intervals[i[0]])

                for j in i[1:]:

                    # If no overlap, break iterations
                    if (pulse_intervals[j][0] > curr_interval[1]) or (pulse_intervals[j][1] < curr_interval[0]):
                        break

                    # Otherwise narrow current interval of overlap
                    curr_interval[0] = max(curr_interval[0], pulse_intervals[j][0])
                    curr_interval[1] = min(curr_interval[1], pulse_intervals[j][1])

                # If loop was not broken, consider this a valid intersection
                else:
                    if not i in valid_intersections:
                        valid_intersections.append(tuple(i))

        return valid_intersections


    def all_resonance_screened_compound_pulses(self, cand_pulse_tuples: list[tuple[int]], thres_freq:float | int=8000.,
                                               tol_n_dev: float | int = 5.0) -> dict[tuple, EmPulse]:
        """
        Take a list of candidate pulse tuples and decide if a compound pulse created from them
        can produce resonance in the vibrational as judged by whether the compound's frequency components
        (with a cutoff) has a bandwidth that to a non-zero extent intersects with frequency components
        falling beneath a threshold, returning all such valid candidates as a {pulse ID tuple:EmPulse instance} dictionary
        This routine currently only supports Gaussian-envelope pulses.

        cand_pulse_tuples: A list of pulse tuples specified by (signed) pulse IDs. The main intended origin of this list
        is to have it be those tuples that were found to have sufficiently large time-domain intersections
        as judged by overlapping_pulses_for_interaction_pattern().

        thres_freq: A threshold (default: +/-8000 cm^-1) outside which no resonance in
        the vibrational manifold is considered. To be given in units of cm^-1.

        tol_n_dev: Tolerance parameter (for Gaussian-envelope pulses) (default: 5 units): Each pulse's frequency-domain
        bandwidth is taken as arg(freq domain max) +/- tol_n_dev * the pulse's frequency-domain deviation parameter.

        Returns: screened_compound_pulses: A dictionary of those {pulse tuple:EmPulse instance} pairs describing the
        cand_pulse_tuples candidates that satisfy the requirement
        """
        from wilson_suite.wilson_utils.unit_convertor import per_cm_x_fs

        screened_compound_pulses = {}

        # Loop over candidate tuples
        for c in cand_pulse_tuples:

            # Create compounded pulse instance
            comp_pulse = compound_pulses_in_list([self.signed_pulses[i] for i in c])

            # Determine bandwidth and check if any of it falls inside the threshold range
            w_range = [comp_pulse.cf - tol_n_dev * (per_cm_x_fs/comp_pulse.dev),
                       comp_pulse.cf + tol_n_dev * (per_cm_x_fs/comp_pulse.dev)]

            # If upper limit of bandwidth is >= lower threshold and lower limit of bandwith is <= upper threshold
            # then this compound pulse is deemed in range
            if not(w_range[1] < -1*thres_freq) and not(w_range[0] > thres_freq):
                screened_compound_pulses[c] = copy.deepcopy(comp_pulse)

        return screened_compound_pulses

    def new_field_with_changes(self, scans: tuple[SpecScan], index: tuple[int]):
        """
        Take a list of changes and return a new ElectricField instance corresponding to these changes
        applied to self.

        scans: The scans under which changes are to be made

        index: A tuple of integers specifying the index of each scan's range to be applied

        returns: A new ElectricField instance with the requested changes
        """

        pulse_dict = {i.id: copy.deepcopy(i) for i in self.pulses}

        for i in range(len(scans)):
            scan_base_val = scans[i].range[index[i]]

            for j in scans[i].scan_objs:

                if not (j.category == 'pulse'):
                    raise ValueError('Only pulse attributes may be changed')

                if not j.id in pulse_dict:
                    raise ValueError('Pulse ID not found in field')

                if j.subcategory == 'cf':
                    pulse_dict[j.id].cf += j.coeff * scan_base_val

                elif j.subcategory == 'tc':
                    pulse_dict[j.id].tc += j.coeff * scan_base_val

                elif j.subcategory == 'dev':
                    pulse_dict[j.id].dev += j.coeff * scan_base_val

                else:
                    raise ValueError('Invalid scan subcategory')

        return ElectricField(tuple(copy.deepcopy(pulse_dict.values())))

def compounding_patterns_at_order(n) -> list[tuple]:
    """
    Generate all compounding patterns at a given order

    n: The order to be considered

    Returns: A list of tuples: Each tuple is a compounding pattern at this order
    """

    def compounding_patterns_recurse(n, curr, acc):
        """
        Tail-recursive routine to accumulate compounding patterns
        """

        # Termination condition
        if n == 0:
            if not tuple(curr) in acc:
                acc.append(tuple(curr))

        else:

            # Add 1 to all in curr and recurse
            for i in curr:

                new_curr = copy.deepcopy(curr)
                new_curr[i] += 1
                compounding_patterns_recurse(n-1, new_curr, acc)

            # Extend curr by 1 and recurse
            new_curr = copy.deepcopy(curr)
            new_curr.append(1)
            compounding_patterns_recurse(n - 1, new_curr, acc)

        return

    # Initialize results, seed holder
    comp_patterns = []
    seed = []

    compounding_patterns_recurse(n, seed, comp_patterns)

    return comp_patterns

def pulse_time_ordering_valid(pulses: list[EmPulse] | tuple[EmPulse], thres_waiting_time: float|int  = 1000.) -> bool:
    """
    Take an ordered sequence of (compounded) pulses and determine
    """

    # Singleton sequences are trivially valid
    if len(pulses) == 1:
        return True

    for i in range(len(pulses) - 1):
        if (pulses[i + 1].tc - pulses[i].tc) > thres_waiting_time:
            return False

    return True


@dataclass
class FieldAnalysis:
    """
    Class to carry out and hold results from analysis done on a field

    field: Electric field instance on which the analysis is to be made

    interaction_filter: A list of interaction pattern(s) (each a list of signed integers referring to pulse IDs with
    the sign specifying the freq/wavevector parity) to be considered

    tol_n_dev_t: Float: Tolerance as number of standard deviations of Gaussian pulse's time broadness for
    deciding (compound) pulse time overlap. Default: 5.

    tol_n_dev_w: Float: Tolerance as number of standard deviations of Gaussian pulse's time broadness for
    deciding (compound) pulse ability to create vibrational resonance. Default: 5.

    thres_freq: A threshold: (compound) pulses whose no frequency components fall within the range
     [-thres_freq, thres_freq] are deemed unable to produced vibrational resonance. Default: 8000.0 (units: cm^-1).

    thres_waiting_time: A threshold above which (compound) pulses whose time maxima intervals are larger are deemed too
    distant to produce an appreciable signal. Default: 1500.0 (units: fs).

    wavevector_filter: String: Speficies regime under which matching wavevectors to that/those of the interaction
    will be searched for/included
    """

    field: ElectricField
    interaction_filter: list[list[int]]
    tol_n_dev_t: float | int = 5.0
    tol_n_dev_w: float | int = 5.0
    thres_freq: float | int = 8000.0
    thres_waiting_time: float | int = 1500.0
    wavevector_filter: str = 'same_order'

    def __post_init__(self):

        # TODO: Make simplified handling for ideal frequency-domain (and later also possibly ideal time-domain) experiments
        # TODO: Possibly introduce a phase shift to partly account for "non-ideally-Placzek" effects?
        # TODO: If reintroducing magnitude conditions, have them follow from the field and manage them here (with
        #  determination offloaded to external routine)

        from itertools import permutations as permutations

        # Dividing larger pieces of code into helper methods, incorporate all the processing leading to the
        # information that's needed for an integrator:

        if not len(self.interaction_filter) > 0:
            raise ValueError('Interaction filter must have nonzero length')

        # Find wavevector-matching interaction patterns
        if self.wavevector_filter in ['same_order', 'up_to_order']:
            if not len(self.interaction_filter) == 1:
                raise ValueError('Wavevector filter incompatible with > 1 interactions in filter')

            self.valid_int_patterns = self.field.wavevectors_matching_ids(self.interaction_filter[0], filter = self.wavevector_filter)

        elif self.wavevector_filter == 'only_specified':

            valid_int_patterns = [tuple(sorted(list(i))) for i in self.interaction_filter]

            # Check if all wavevectors in interaction filter point in the same direction; if not, raise error

            # Get all wavevectors matching the direction of a max order interaction in the filter with the
            # 'up_to_order' flag; the returned set must contain all requested interaction patterns
            maxlen = 0
            for i in range(len(self.interaction_filter)):
                if len(self.interaction_filter[i]) > maxlen:
                    maxlen = len(self.interaction_filter[i])
                    arg_max = i

            test_int_patterns = self.field.wavevectors_matching_ids(self.interaction_filter[arg_max], filter = 'up_to_order')

            for i in valid_int_patterns:
                if not i in test_int_patterns:
                    raise ValueError('Not all requested interaction patterns correspond to the same wavevector')

            self.valid_int_patterns = valid_int_patterns

        # Additionally compile information about which orders of interaction are represented in the interaction patterns
        int_pattern_orders = []

        for i in self.valid_int_patterns:
            if not len(i) in int_pattern_orders:
                int_pattern_orders.append(i)

        self.int_pattern_orders = sorted(int_pattern_orders)

        # Initialize result holders
        time_ovl_tuples = []
        time_not_res_compound_pulses = {}
        res_and_time_ovl_tuples = []
        res_and_time_compound_pulses = {}

        # For all valid interaction patterns:
        for i in self.valid_int_patterns:

            # Accumulate all valid and relevant time-overlapping tuples (compounding and non-compounding)
            time_ovl_tuples = list(set(time_ovl_tuples).union(self.field.overlapping_pulses_for_interaction_pattern(i, self.tol_n_dev_t)))

        # Screen these according to resonance possibility (obtaining compound pulse instances)
        res_and_time_compound_pulses = self.field.all_resonance_screened_compound_pulses(time_ovl_tuples, self.thres_freq, self.tol_n_dev_w)

        # Take keys of prev results as overview of both resonance and time-overlap screened tuples
        res_and_time_ovl_tuples = copy.deepcopy(res_and_time_compound_pulses.keys())

        # Obtain remaining compound pulses (time-overlapping but not resonance-possible) (they apply for the last interaction)
        for i in time_ovl_tuples:
            if not(i in res_and_time_ovl_tuples):
                time_not_res_compound_pulses[i] = compound_pulses_in_list([self.field.signed_pulses[j] for j in i])

        self.time_ovl_tuples = time_ovl_tuples
        self.res_and_time_ovl_tuples = res_and_time_ovl_tuples
        self.res_and_time_comp_pulses = res_and_time_compound_pulses
        self.time_not_res_comp_pulses = time_not_res_compound_pulses

        time_ovl_comp_lvls = []
        for i in self.time_ovl_tuples:
            if not len(i) in time_ovl_comp_lvls:
                time_ovl_comp_lvls.append(len(i))

        # Generate valid compounding patterns at all relevant orders
        comp_patterns_at_order = {}

        for i in self.int_pattern_orders:

            # Generate candidate patterns (all patterns but will be screened)
            candidate_compounding_patterns_at_order = compounding_patterns_at_order(i)

            # Screen based on occurrence of valid orders in time_ovl_tuples
            # (if no pulse tuple of a compounding level required in a compounding pattern can exist,
            # then that compounding pattern can already here be disregarded)
            for j in candidate_compounding_patterns_at_order:

                screened_patterns = []
                pattern_supported_by_tuples = True

                for k in j:
                    if not k in time_ovl_comp_lvls:
                        pattern_supported_by_tuples = False

                if pattern_supported_by_tuples:
                    screened_patterns.append(j)

            comp_patterns_at_order[i] = copy.deepcopy(screened_patterns)

        self.comp_patterns_at_order = comp_patterns_at_order

        int_patterns_by_order = {}

        for i in valid_int_patterns:

            if len(i) in int_patterns_by_order:
                int_patterns_by_order[len(i)].append(i)

            else:
                int_patterns_by_order[len(i)] = [[i]]

        # The "master" loop: Evaluate all available interaction sequences at all valid compounding patterns
        int_sequences_by_order = {}

        # For each valid order of interaction pattern
        for ord in int_patterns_by_order:

            int_sequences_by_order[ord] = {}

            for c in comp_patterns_at_order[ord]:
                int_sequences_by_order[ord][c] = []

            # For each permutation of indices at this order
            for p in permutations(range(ord)):

                # For each interaction pattern at this order
                for i in int_patterns_by_order[ord]:

                    p_pattern = []

                    # Permute the pulses to that ordering
                    for j in range(len(i)):
                        p_pattern.append(i[p[j]])

                    # For each compounding pattern at this interaction pattern's order:
                    for c in comp_patterns_at_order[ord]:

                        # FIXME: There should be several optimization opportunities here if needed:
                        #  - Precalculate motifs of compounding patterns for time-overlap/resonance criteria
                        #  - Precalculate motifs of time-enforcement patterns
                        #  - Preassemble some compound pulse index motifs?
                        #  - Also pre-solve for some permutation subsets in an outer loop?

                        # First test if this permutation contains all valid (compound) pulses
                        comp_valid = True
                        c_ctr = 0

                        c_pulse_inds = []
                        c_pulses = []

                        # For each compounding pattern element
                        for c_elem in range(len(c)):

                            # Generate the candidate (compound) pulse tuple
                            c_elem_len = len(c_elem)
                            c_pulse_ind = tuple(sorted(p_pattern[c_ctr: c_ctr + c_elem_len]))

                            # If this is not the last interaction, the candidate pulse tuple must
                            # be both time-overlapping and potentially able to make vibrational resonance
                            if c_elem + 1 < len(c):

                                if not c_pulse_ind in self.res_and_time_ovl_tuples:
                                    comp_valid = False
                                    break

                                c_pulses.append(self.res_and_time_comp_pulses[c_pulse_ind])
                                c_pulse_inds.append(c_pulse_ind)

                            # If this is the last interaction, only the time-overlapping criterion applies
                            else:

                                if not c_pulse_ind in self.time_ovl_tuples:
                                    comp_valid = False
                                    break

                                c_pulses.append(self.time_not_res_comp_pulses[c_pulse_ind])
                                c_pulse_inds.append(c_pulse_ind)

                            c_ctr += c_elem_len

                        # Cycle if any (compound) pulse was invalid
                        if not comp_valid:
                            continue

                        # Time-ordering enforcement screening: If also passing this criterion, register this
                        # interaction sequence as valid
                        if pulse_time_ordering_valid(c_pulses):
                            int_sequences_by_order[ord][c].append(tuple(c_pulse_inds))


@dataclass
class VibExperiment:
    """
    Class to represent a vibrational wave-mixing experiment

    field: ElectricField instance: A "base" perturbing field (upon which scans may be imposed)

    detector: SpecDetector instance: The detector for this experiment

    scans: List of SpecScan instances: Tells which parameters will be scanned over (and how) in this experiment
    """

    field: ElectricField
    detector: SpecDetector
    scans: tuple[SpecScan] = ()

    def __post_init__(self):

        if not isinstance(self.field, ElectricField):
            raise TypeError('The field attribute must be an ElectricField instance')

        if not isinstance(self.detector, SpecDetector):
            raise TypeError('The detector attribute must be a SpecDetector instance')

        if self.scans is not None:
            if not isinstance(self.scans, tuple):
                raise TypeError('The scans attribute, if specified, must be a tuple of SpecScan instances')
            for i in self.scans:
                if not isinstance(i, SpecScan):
                    raise TypeError('The scans attribute, if specified, must be a tuple of SpecScan instances')

        self.dim = self.findDimensionality()

        # TODO: Inspection of requested scans against field
        #  - If the experiment is an ideal frequency-domain experiment, then limit which pulse scanning
        #  attributes are valid (scanning time centerpoint is then less meaningful)
        #  - If the experiment is an ideal time-domain experiment, then also limit scan attribute validity
        #  (scanning carrier frequency is then less meaningful)
        #  - Maybe take tc resp. cf (for freq.-ideal resp. time-ideal) as "sleeping" parameters but need specification
        #  if scanning time spread? Then may need some inverse stuff for one case (touch/depart from infinity)
        #  (could also handle this more practically but less elegantly with large number instead of infty throughout)

        # Make scan grid with field analyses under scanning
        # (NOTE: Currently, field attributes are the only attributes whose scanning is supported and so,
        # a grid of (scanned) fields manifests all the scans' changes to the experiment)
        grid_dims = tuple([len(i.range) for i in self.scans])

        # To hold instances of FieldAnalysis
        self.field_analyses = np.empty(grid_dims, dtype=object)

        if len(self.scans) > 0:
            for scan_elem in np.ndindex(self.field_analyses.shape):
                self.field_analyses[scan_elem] = FieldAnalysis(
                    self.field.new_field_with_changes(self.scans, scan_elem),
                    self.detector.interaction_filter)

        # With no scans, the analysis is only done with respect to the base field
        else:
            self.field_analyses[()] = FieldAnalysis(self.field, self.detector.interaction_filter)

        # TODO: Probably move this inside FieldAnalysis since this is interaction-pattern dependent

        # Register all polarization vectors (associated with the detector and pulses) for convenience
        # Here I establish a convention: Macroscopic ranks are with respect to pulse IDs but first rank refers to the
        # detected signal (so detected, pulse ID 1, pulse ID 2, ...)

        all_polarizations = [copy.deepcopy(self.detector.detection_polarization)]

        # Could probably be done more elegantly but works
        for i in range(len(self.field.pulses)):
            for j in self.field.pulses:
                if j.id == i + 1:
                    all_polarizations.append(copy.deepcopy(j.pol))

        # Determine the macroscopic orientational average polarization vector
        from wilson_suite.wilson_intensities.amplitudes.averaging import get_pol_laser
        self.polarization_avg_vector = get_pol_laser(all_polarizations)

    def findDimensionality(self) -> int:
        """
        Using detector and scans information, determine the dimensionality of the spectral data
        that carrying out this experiment would produce

        Returns an integer d telling this dimensionality
        """

        d = 0
        d += len(self.scans)

        # Time or frequency series adds a dimension
        if self.detector.detection_method == 'time':
            if self.detector.detection_range is not None:
                return d + 1

        elif self.detector.detection_method == 'freq':
            if self.detector.detection_range is not None:
                return d + 1

        # Integrated intensity does not add a dimension
        elif self.detector.detection_method == 'int':
            return d

        else:
            raise AssertionError('Cannot determine dimensionality: Unrecognized detection method')

        return d
