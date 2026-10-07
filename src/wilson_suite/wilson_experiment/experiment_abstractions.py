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

    # FIXME: Turn back into wavevector filter?
    interaction_filter: List of interaction patterns as lists: Filter signal to include only this/these
    (signed) interaction patterns. Note that any ordering here is not taken as a causal interaction order requirement
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
    ignore_collinear: bool = True

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

        if not(self.ignore_collinear):
            raise NotImplementedError('Non-ignorance of effects collinear to specified condition(s) is currently not implemented')

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
    """

    env: str
    
    tc: float = None
    cf: float = None
    dev: float = None

    maxstr: float = 0.0

    wv: tuple[float] = (0.0, 0.0, 1.0)
    pol: tuple[float] = (1.0, 0.0, 0.0)
    pol_comp:  tuple[tuple[float]] = None
    overall_phase: complex = 0.0

    id: int = None
    id_comp: tuple = None

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

    return EmPulse(env = 'gaussian', tc = tc, cf = cf, dev = dev, cf_uv = cf_uv, maxstr = maxstr, wv = wv,
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

    return EmPulse(env = 'gaussian', tc = tc, cf = cf, dev = 0.0, cf_uv = cf_uv, maxstr = maxstr, wv = wv,
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

    return EmPulse(env = 'gaussian', tc = tc, cf = cf, dev = infinity, cf_uv = cf_uv, maxstr = maxstr, wv = wv,
                   pol = pol, overall_phase = overall_phase, id = id)

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

        valid_filters = ['same_order', 'up_to_order']


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


        if filter == 'same_order':
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
        can produce resonance in the IR region as judged by whether the compound's frequency components
        (with a cutoff) has a bandwidth that to a non-zero extent intersects with frequency components
        falling beneath a threshold, returning all such valid candidates as a {pulse tuple:EmPulse instance} dictionary
        This routine currently only supports Gaussian-envelope pulses.

        cand_pulse_tuples: A list of pulse tuples specified by (signed) pulse IDs. The main intended origin of this list
        is to have it be those tuples that were found to have sufficiently large time-domain intersections
        as judged by overlapping_pulses_for_interaction_pattern().

        thres_freq: A threshold (default: 8000 cm^-1) below which it is deemed that resonance in
        the vibrational manifold can take place. To be given in units of cm^-1.

        tol_n_dev: Tolerance parameter (for Gaussian-envelope pulses) (default: 5 units): Each pulse's frequency-domain
        bandwidth is taken as arg(freq domain max) +/- tol_n_dev * the pulse's frequency-domain deviation parameter.

        Returns: screened_compound_pulses: A dictionary of those {pulse tuple:EmPulse instance} pairs describing the
        cand_pulse_tuples candidates that satisfy the requirement
        """
        from wilson_suite.wilson_utils.unit_convertor import per_cm_x_fs

        screened_compound_pulses = {}

        # Loop over candidate tuples
        for c in cand_pulse_tuples:

            n_pulse = self.signed_pulses[c[0]]
            if not n_pulse.env == 'gaussian':
                raise ValueError('Only Gaussian pulses are currently supported for compounding')

            # Take the first pulse reference and represent it as an EmPulse in "compounding" mode
            comp_pulse = EmPulse(env = n_pulse.env, tc = n_pulse.tc, cf = n_pulse.cf, dev = n_pulse.dev,
                                    maxstr = n_pulse.maxstr, wv = n_pulse.wv, pol = None, pol_comp = (n_pulse.pol,),
                                    overall_phase= n_pulse.overall_phase, id_comp=(n_pulse.id,))

            # Form compound pulse iteratively with any further pulse refs in tuple
            for p in range(1, len(c)):
                n_pulse = self.signed_pulses[c[p]]

                if not n_pulse.env == 'gaussian':
                    raise ValueError('Only Gaussian pulses are currently supported for compounding')

                # Combining attributes of comp_pulse and the new n_pulse
                # Here using the fact that the product of Gaussians is another Gaussian with specific expressions for
                # the new means and variances
                npdsq = (n_pulse.dev)**2
                cpdsq = (comp_pulse.dev) ** 2
                ncpdsq = npdsq + cpdsq

                new_tc = (comp_pulse.tc * npdsq + n_pulse.tc * cpdsq) / ncpdsq
                new_cf = comp_pulse.cf + n_pulse.cf
                new_dev = (npdsq * cpdsq / ncpdsq)**0.5

                new_maxstr = comp_pulse.maxstr*n_pulse.maxstr * exp(-1 * ( ( comp_pulse.tc - n_pulse.tc)**2 ) / (2 * ncpdsq) )

                new_wv = tuple([comp_pulse.wv[j] + n_pulse.wv[j] for j in range(3)])

                # Extending tuple
                lcppc = list(comp_pulse.pol_comp)
                lcppc.append(n_pulse.pol)
                new_pol_comp = tuple(lcppc)

                # New phase is sum
                new_overall_phase = comp_pulse.overall_phase + n_pulse.overall_phase

                # Extending tuple
                nic = list(comp_pulse.id_comp)
                nic.append(n_pulse.id)
                new_id_comp = tuple(nic)

                # Update compounded pulse
                comp_pulse = EmPulse(env='gaussian', tc = new_tc, cf = new_cf, dev = new_dev, maxstr = new_maxstr,
                                     wv = new_wv, pol=None, pol_comp = new_pol_comp, overall_phase = new_overall_phase,
                                     id_comp = new_id_comp)

            # Determine bandwidth and check if any of it falls inside the threshold range

            w_range = [comp_pulse.cf - tol_n_dev * (per_cm_x_fs/comp_pulse.dev),
                       comp_pulse.cf + tol_n_dev * (per_cm_x_fs/comp_pulse.dev)]

            print('compound', c, 'bandwidth', w_range, 'dev', comp_pulse.dev)

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



@dataclass
class VibExperiment:
    """
    Class to represent a vibrational wave-mixing experiment

    field: ElectricField instance: A "base" perturbing field (upon which scans may be imposed)

    detector: SpecDetector instance: The detector for this experiment

    scans: List of SpecScan instances: Tells which parameters will be scanned over (and how) in this experiment

    magn_conditions: Tuple of tuples: Magnitude conditions for use in identifying terms that will not become
    fully resononant in this experiment. Format: Outer tuple collects magnitude conditions. Each inner tuple is
    a magnitude condition and consists of signed pulse references (NOTE: Currently not using the SignedPulseTuple class)
    where the sum of the associated frequencies are understood to be significantly > 0, where "significantly > 0" means
    "never close to zero".
    Example: ( (-1, 2), (2, 3, -4) ) denotes two magnitude conditions:
        a) -w1 + w2 is always significantly > 0,
        b) w2 + w3 - w4 is always significantly > 0
    """

    field: ElectricField
    detector: SpecDetector
    scans: tuple[SpecScan] = ()
    magn_conditions: tuple[tuple] = None

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

        if self.magn_conditions is not None:
            if not isinstance(self.magn_conditions, tuple):
                raise TypeError('The magn_conditions attribute, if specified, must be a tuple of tuples')
            for i in self.magn_conditions:
                if not isinstance(i, tuple):
                    raise TypeError('The magn_conditions attribute, if specified, must be a tuple of tuples')

        self.dim = self.findDimensionality()

        # Make scan grid with fields under scanning
        # (NOTE: Currently, field attributes are the only attributes whose scanning is supported and so,
        # a grid of (scanned) fields manifests all the scans' changes to the experiment)
        grid_dims = tuple([len(i.range) for i in self.scans])
        self.field_scan_grid = np.empty(grid_dims, dtype=object)

        if len(self.scans) > 0:
            for scan_elem in np.ndindex(scan_grid.shape):
                self.field_scan_grid[scan_elem] = self.field.new_field_with_changes(self.scans, scan_elem)

        # With no scans, the scan grid is just the "scalar" base field
        else:
            self.field_scan_grid[()] = self.field

        # NOTE: Below code to be reworked or rmd



        from wilson_suite.wilson_experiment.indep_vars_and_axes import (PhaseMatchingCondition, SignedPulseTuple,
                                                                        find_indep_exp_variables, find_valid_axes,
                                                                        find_canonical_axes)

        # Establishes an assumption: The order is the same as the number of pulses in the field
        # It is furthermore (but not strictly from this) assumed that in the experiment, the system will interact
        # once with each pulse
        self.order = len(self.field.pulses)

        # Determine which phase-matching condition(s) will come under consideration in this experiment
        relevant_phasematch = []

        # If no specified phase-matching (wavevector) filter, all are (potentially) relevant
        if self.detector.interaction_filter is None:

            from itertools import product as iter_prod
            k = 0

            for i in iter_prod([1, -1], repeat=len(self.field.pulses)):

                new_phasematch = []

                for j in range(len(self.field.pulses)):
                    new_phasematch.append(self.field.pulses[j].id * i[j])

                relevant_phasematch.append(PhaseMatchingCondition(SignedPulseTuple(tuple(new_phasematch)), k))
                k += 1

        # Otherwise, registered only that/those specified for the detector
        else:

            k = 0

            for i in range(len(self.detector.interaction_filter)):

                new_phasematch = []

                for j in self.detector.interaction_filter[i]:
                    new_phasematch.append(j * self.detector.interaction_filter[i][j])

                relevant_phasematch.append(PhaseMatchingCondition(SignedPulseTuple(tuple(new_phasematch)), k))
                k += 1

        self.relevant_phasematch = relevant_phasematch

        # Do first:
        # - For (and indexed by) relevant phase-matching conditions (as determined above):
        #   -  Generate a new field with the appropriate pulses (and possibly resultant wavevector)
        #       - Will need to lift pos. carrier freq condition on pulse for this
        # - For all lower- or same-order phase-matching conditions:
        #   - At least determine resultant wavevectors and determine if they are parallel to any of those of the
        #     relevant phase-matching conditions
        #   - (FOR LATER) Generate the corresponding fields/scans
        #   - For now, only warn if other cascading (collinear) effects may intrude on the signal if ignore_collinear is not set to True
        # - Take pulse with scans and


        # Here do:
        #  - If the experiment is an ideal frequency-domain experiment, then limit which pulse scanning
        #  attributes are valid (scanning time centerpoint is then less meaningful)
        #  - If the experiment is an ideal time-domain experiment, then also limit scan attribute validity
        #  (scanning carrier frequency is then less meaningful)
        # - Maybe take tc resp. cf (for freq.-ideal resp. time-ideal) as "sleeping" parameters but need specification
        #  if scanning time spread? Then may need some inverse stuff for one case (touch/depart from infinity)
        #  (could also handle this more practically but less elegantly with large number instead of infty throughout)

        # - Construct fields_under_scan attribute
        # - Handle better the discretization of scans
        # - Handle separation/combination of response-side/integration-side scans
        #  - Also consider and maybe handle phase question between "differently-compounding" features here




        # Find valid choices of independent variables
        self.indep_vars = find_indep_exp_variables(self.field.pulses, self.epochs, self.relevant_phasematch)

        # Find valid choices of spectral axes given the choices of independent variables determined above
        self.valid_axis_combs = find_valid_axes(self.indep_vars)

        # If no canonical axes can be determined, set to None
        try:
            self.canonical_axes = find_canonical_axes(self.indep_vars)
        except ValueError:
            self.canonical_axes = None

        # Register all polarization vectors (associated with the detector and pulses) for convenience
        # Here I establish a convention: Macroscopic ranks are with respect to pulse IDs but first rank refers to the
        # detected signal (so detected, pulse ID 1, pulse ID 2, ...)
        all_polarizations = [copy.deepcopy(self.detector.detection_polarization)]

        # Could probably be done more elegantly but works
        for i in range(len(self.field.pulses)):
            for j in self.field.pulses:
                if j.id == i + 1:
                    all_polarizations.append(copy.deepcopy(j.pol))

        self.all_polarizations = all_polarizations

        # Determine the macroscopic orientational average polarization vector
        from wilson_suite.wilson_intensities.amplitudes.averaging import get_pol_laser
        self.polarization_avg_vector = get_pol_laser(self.all_polarizations)

    def tell_axis_options(self):

        for i in range(len(self.valid_axis_combs)):
            self.valid_axis_combs[i].present_spectral_axis_choices(from_exp_index = i)

    def is_axis_set_by_ref_valid(self, ref: dict):
        """
        TODO: Take a reference describing an axis set choice in terms of
        the dict {label 1: ((indep var 1_1), (indep var 1_2 ) ...), label 2: ((indep var 2_1), ...) , ...}
        and return True if it's a valid choice for this experiment (TODO: ...phase-matching combination, indep var choice?)
        or False if it's not

        Sketch: Similar to choose_axis_set_by_ref

        """
        pass

    def choose_axis_set_by_ref(self, ref: dict):
        """
        TODO: Take a reference describing an axis set choice in terms of
        the dict {label 1: ((indep var 1_1), (indep var 1_2 ) ...), label 2: ((indep var 2_1), ...) , ...}
        and return the corresponding SpectralAxisSet instance

        Sketch: Look through self.valid_axis_choices and find out if any of the registered valid choices
        match: If yes, return that SpectralAxisSet instance - otherwise raise an error
        """
        pass

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



        if self.detector.interaction_filter is None:
            raise AssertionError('Interaction sequence determination currently only implemented for wavevector filter detector')

        if len(self.detector.interaction_filter) > 1:
            raise AssertionError('Interaction sequence determination currently not supported for more than one phase-matching direction')

        int_sequences = []
        int_seed = []

        for i in self.detector.interaction_filter:

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

    if not(tol == 0.0):
        raise ValueError('Non-zero tolerance not yet supported in find_epochs')

    for i in field.pulses:
        if not i.tendsImpulsive():
            raise AssertionError('Can currently only determine epochs for fields with impulsive-tending pulses')
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

    sorted_epochs = []

    for i in epochs:
        sorted_epochs.append(sorted(i))

    return sorted_epochs

def uv_cancels(coll: tuple, cfs_uv: dict, tol: float=1e-10) -> bool:
    """
    Do the UV/VIS frequency components of this collection of pulses cancel?

    coll: tuple: (Signed) pulse ID references
    cfs_uv: dictionary {non-signed pulse ID: non-signed UV/VIS frequency component, ...}
    tol: Tolerance: If the magnitude of the requested combination of freq components is beneath tol,
    then accept as cancelling

    return: True if cancellation was determined, False otherwise

    # FIXME: This routine is a bit confusingly made but should
    """


    if tol < 0.0:
        raise ValueError('The tolerance must be a nonnegative number')

    acc = 0.0

    for i in coll:

        sgn = (i > 0) - (i < 0)

        acc += sgn * cfs_uv[sgn * i]

    sgnacc = (acc > 0) - (acc < 0)

    return ((sgnacc * acc) <= tol)