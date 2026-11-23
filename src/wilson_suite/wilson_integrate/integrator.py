import copy
from dataclasses import dataclass
from collections.abc import Sequence
from ..wilson_experiment.experiment_abstractions import SpecDetector
from math import floor, ceil, inf
import numpy as np

# FIXME: Consider organizing SumFreqIntegrator and DirectIntegrator under ABC paradigm - may not be that important
# with only two integrators but could be more relevant with combined forms. CONCLUSION: Consider this later if relevant
# then.

class SumFreqIntegrator:

    def __init__(self):

        pass

    def integrate(self, comp_pattern, sumfreq_points, rsp_points, pulse_ranges_points, rsp_on_grid, pulses_on_grid):

        def integrate_recurse(r, acc_pts, curr_rank, max_rank, emission_compounding, rsp_points, pulse_ranges_points,
                              rsp_on_grid, pulses_on_grid):
            """
            Integration recursion

            TODO: If available, might be possible to make use of factorization hints for the response function
            TODO: This is a naive implementation and may want optimization

            """

            curr_min = pulse_ranges_points[curr_rank][0]
            curr_max = pulse_ranges_points[curr_rank][1]

            if curr_rank == max_rank:

                if r >= curr_min and r <= curr_max:

                    if emission_compounding:

                        res = rsp_on_grid[tuple(acc_pts[:-1])]

                    else:

                        res = rsp_on_grid[tuple(acc_pts)]

                    for i in range(max_rank):

                        res *= pulses_on_grid[i][acc_pts[i]]

            else:

                res = 0.0

                a_ind = 0

                for p in range(pulse_ranges_points[curr_rank][0], curr_rank[1]):

                    new_acc_pts = copy.deepcopy(acc_pts)
                    new_acc_pts.append(a_ind)

                    res += integrate_recurse(r - p, new_acc_pts, curr_rank + 1, max_rank, emission_compounding,
                                             rsp_points, pulse_ranges_points, rsp_points, pulses_on_grid)

                    a_ind += 1

                return res

        integration_result = np.zeros((sumfreq_points[1] - sumfreq_points[0] + 1,))
        emission_compounding = comp_pattern[-1] > 1

        # TODO: This loop should be readily parallelizable
        for p in range(sumfreq_points[0], sumfreq_points[1] + 1):

            pts_seed = []

            integration_result[p - sumfreq_points[0]] = integrate_recurse(p, pts_seed, 0,
                                                                          len(pulse_ranges_points),
                                                                          emission_compounding,
                                                                          rsp_points, pulse_ranges_points,
                                                                          rsp_on_grid, pulses_on_grid)

        return integration_result

class DirectIntegrator:

    def __init__(self):

        pass

    def integrate(self):

        # Here parallelize as needed (easy to parallelize over time points)

        integration_result = []

        return integration_result



@dataclass
class SpectralIntegration:
    """Class to carry out (nested) integrations """

    detector: SpecDetector

    int_ranges_coll: dict
    # FIXME: Review these declarations: There is likely a better way to organize this (for instance, the pulse banks
    # and their integration ranges conceptually belong together)
    # Should contain all int. ranges over all scan points and ocmpi index (i.e. would be taken from each FieldAnalysis instance) and
    # indexed in the same way
    int_ranges_scan: Sequence
    # Should contain all pulse banks for each scan point (and indexed by same)
    pulse_banks_scan: Sequence
    rsp_results: dict
    resolution: int | float
    style: str

    # FIXME: Provisional design choice: If data is to be yielded in the time domain
    # with "direct" integration, an additional grid of time points should be provided
    # This is an incomplete solution because a) the detector may specify a desired range
    # (but currently not a resolution) for this, b) a reasonable range (but not necessarily
    # a resolution) may be determined by inspecting the experiment time scale
    # TODO: Work out a consistent regime for handling this in a good way
    suppl_t_grid: Sequence | None = None

    def __post_init__(self):

        if len(self.int_ranges_coll.keys()) == 0:
            raise ValueError('No collective integration ranges')

        valid_styles = ['sum_freq', 'direct', 'auto']

        if not self.style in valid_styles:
            raise ValueError('Invalid integration style')

        # First pass at strategy selection logic
        # FIXME: This logic may need revision subject to changes to improve resolution of the suppl_t_grid
        # design choice above
        if self.style == 'sum_freq':
            self.int_strategy = 'sum_freq'

        elif self.style == 'direct':

            if self.suppl_t_grid is None:
                raise ValueError('A supplementary time-domain grid is needed for direct integration')

            else:
                self.int_strategy = 'direct'

        # The 'auto' integration choice for now chooses 'sum_freq' if freq. detection or time/integrated time
        # detection with no supplementary time-domain grid, or 'direct' if time/integrated time
        # detection with a supplementary time-domain grid given
        elif self.style == 'auto':

            if self.detector.detection_method == 'freq':
                self.int_strategy = 'sum_freq'

            elif self.detector.detection_method in ['time', 'int']:

                # FIXME: Choosing 'sum_freq' in this situation may perform slower than necessary
                # because the time-domain bins are fixed by the frequency-domain bins (and may be different from
                # than what is sufficient in the specific situation). Not sure whether to warn about this - will first
                # see if it's actually a problem in practice first
                if self.suppl_t_grid is None:
                    self.int_strategy = 'sum_freq'
                else:
                    self.int_strategy = 'direct'


    def __integrate_all_tasks(self):

        from ..wilson_utils.common_labels import cap_alpha_labels

        if not self.int_strategy == 'sum_freq':
            raise NotImplementedError('Only sum-frequency integration currently implemented')

        if self.int_strategy == 'sum_freq':

            integrator = SumFreqIntegrator()

            # Go through collective pulse ranges and determine the achievable sum-frequency
            # range over the experiment

            # Any collective pulse range will bound this range
            achievable_sumfreq_range = [inf, -inf]

            for ocmpi in self.int_ranges_coll:

                curr_sumfreq = [sum([i[0] for i in self.int_ranges_coll[ocmpi]]),
                                sum([i[1] for i in self.int_ranges_coll[ocmpi]])
                                ]

                if curr_sumfreq[0] < achievable_sumfreq_range[0]:
                    achievable_sumfreq_range[0] = curr_sumfreq[0]

                if curr_sumfreq[1] > achievable_sumfreq_range[1]:
                    achievable_sumfreq_range[1] = curr_sumfreq[1]

            # If the detector is 'freq' and has a specified range, use it (snapping to overall grid points)
            if self.detector.detection_method == 'freq' and self.detector.detection_interval is not None:

                    s_det_int = sorted(self.detector.detection_interval)

                    detector_range = [self.resolution * floor(s_det_int[0]  / self.resolution),
                                      self.resolution * ceil(s_det_int[1]  / self.resolution)]

                    # Also walk through achievable ranges and raise error if no sum-freq component is inside the range
                    # FIXME: Disjoint "ocmpi" intervals are not handled completely here: A detector can sit in the middle
                    #  of an "ocmpi dark spot".
                    if detector_range[0] > achievable_sumfreq_range[1] or detector_range[1] < achievable_sumfreq_range[0]:
                        raise ValueError("No experiment signal is within the detector's frequency range")

                    # FIXME: Consider registering this as attribute and splitting off above functionality into own routine
                    # However, may need to rethink how to align this with "direct" integration which would not use this
                    # Maybe therefore consider having separate integrator classes for "sumfreq" and "direct" with same
                    # parent class (i.e. ABC)
                    sumfreq_range = detector_range

            # Otherwise, have the detection range be the achievable range
            else:
                    sumfreq_range = achievable_sumfreq_range

            # Sumfreq starts/ends at grid points
            # FIXME: Consider this method more closely, is it reliable?
            sumfreq_points = [round(sumfreq_range[0]/self.resolution), round(sumfreq_range[1]/self.resolution)]

            rsp_points = {}

            # Rsp fn data starts/ends at grid points
            # The data is already snapped to the grid points
            # FIXME: Resolve axis and array rank ordering (now assuming same)
            #  Consider this task as part of working out possible grid/dimension arg pass changes to/from evaluator?
            for ocmpi in self.rsp_results:

                rsp_points[ocmpi] = []

                for d in range(len(self.rsp_results[ocmpi].amplitude.shape)):

                    curr_ax_pts = [self.rsp_results[ocmpi].axis_points[cap_alpha_labels[d]][0],
                                   self.rsp_results[ocmpi].axis_points[cap_alpha_labels[d]][1]]

                    rsp_points[ocmpi].append([round(curr_ax_pts[0]/self.resolution),
                                              round(curr_ax_pts[1]/self.resolution)])

            # Main integration calls

            int_results = {}

            # TODO: It would be easy to parallelize over the scan elements (but will of course not
            # then have good parallelism by this if only one or a few scan points). Consider two-tier
            # parallelization or other solution?

            # For each scan point
            for scan_elem in self.int_ranges_scan:

                int_results[scan_elem] = {}

                # For each ocmpi index at this scan point
                for ocmpi in self.int_ranges_scan[scan_elem]:

                    # - Express all ranges (sumfreq, rsp fn, (compound) pulses) as increments of grid and
                    #   calculate pulses on a range that snaps to the grid
                    curr_int_ranges_snap = [[self.resolution * floor(p[0]  / self.resolution),
                                             self.resolution * ceil(p[1]  / self.resolution)]
                                             for p in self.int_ranges_scan[scan_elem][ocmpi]]

                    # Pulse ranges start/end at grid points, pulses evaluated on grid range
                    pulse_ranges_points = []
                    pulses_on_grid = []

                    for p in range(len(curr_int_ranges_snap)):
                        pulse_ranges_points.append([round(curr_int_ranges_snap[p][0] / self.resolution),
                                                    round(curr_int_ranges_snap[p][1] / self.resolution) ] )

                        pulses_on_grid.append(
                                self.pulse_banks_scan[scan_elem][p].evalFreqSpace(
                                    np.linspace(curr_int_ranges_snap[p][0], curr_int_ranges_snap[p][1],
                                        pulse_ranges_points[p][1] - pulse_ranges_points[p][0] + 1)
                                )
                            )

                    # Integrate and yield as array
                    int_results[scan_elem][ocmpi] = integrator.integrate(
                        ocmpi[1],
                        sumfreq_points,
                        rsp_points[ocmpi],
                        pulse_ranges_points,
                        self.rsp_results[ocmpi].amplitude,
                        pulses_on_grid
                    )

                int_results_sum = {}

                # FIXME: Decide which of the detailed data will also be yielded on return

                for ocmpi in int_results[scan_elem]:
                    int_results_sum[scan_elem] += int_results[scan_elem][ocmpi]

            # TODO: Also make summed amplitudes over all scan points for one ocmpi?

            # FIXME: Make intensities here or do that later?

            if self.detector.detection_method == 'freq':

                return int_results_sum

            if self.detector.detection_method in ['time', 'int']:

                int_results_t = {}

                # FIXME: If also yielding detailed data, set up to yield time data over ocmpi?

                for scan_elem in int_results_sum:

                    # TODO: FT to time series
                    int_results_t[scan_elem] = None

                if self.detector.detection_method == 'time':

                    return int_results_t

                # FIXME: May need to make intensities here anyway
                elif self.detector.detection_method == 'int':

                    int_results_t_int = {}

                    for scan_elem in int_results_t:

                        # TODO: Integrate over time to scalar
                        int_results_t_int[scan_elem] = None

                    return int_results_t_int

