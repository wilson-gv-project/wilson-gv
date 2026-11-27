import copy
from dataclasses import dataclass
from collections.abc import Sequence
from ..wilson_experiment.experiment_abstractions import SpecDetector
from math import floor, ceil, inf
import numpy as np
import time
import scipy.integrate as scint
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# FIXME: Consider organizing SumFreqIntegrator and DirectIntegrator under ABC paradigm - may not be that important
# with only two integrators but could be more relevant with combined forms. CONCLUSION: Consider this later if relevant
# then.

class SumFreqIntegrator:

    def __init__(self):

        pass

    def integrate(self, comp_pattern, sumfreq_points,
                  rsp_points, pulse_ranges_points, rsp_on_grid, rsp_skip_ranges, pulses_on_grid,
                  parallel = False):

        def integrate_recurse(r, acc_pts, rsp_pts, curr_rank, max_rank, emission_compounding, rsp_points, pulse_ranges_points,
                              rsp_on_grid, rsp_skip_ranges, pulses_on_grid, curr_prod):
            """
            Integration recursion

            TODO: If available, might be possible to make use of factorization hints for the response function
            TODO: This is a naive implementation and may want optimization

            """

            curr_min = pulse_ranges_points[curr_rank][0]
            curr_max = pulse_ranges_points[curr_rank][1]

            if curr_rank == max_rank:

                if r >= curr_min and r <= curr_max:

                    last_pt = r - curr_min

                    if emission_compounding:

                        return rsp_on_grid[tuple(rsp_pts)] * pulses_on_grid[max_rank][last_pt] * curr_prod

                    else:

                        rsp_address = copy.deepcopy(rsp_pts)
                        rsp_address.append(last_pt + pulse_ranges_points[curr_rank][0] - rsp_points[curr_rank][0])

                        return rsp_on_grid[tuple(rsp_address)] * pulses_on_grid[max_rank][last_pt] * curr_prod

                else:
                    return 0.0

            else:

                res = 0.0
                a_ind = 0

                for p in range(curr_min, curr_max):

                    new_rsp_pt = a_ind + pulse_ranges_points[curr_rank][0] - rsp_points[curr_rank][0]

                    if not new_rsp_pt in rsp_skip_ranges[curr_rank]:

                        new_acc_pts = copy.deepcopy(acc_pts)
                        new_acc_pts.append(a_ind)

                        new_rsp_pts = copy.deepcopy(rsp_pts)
                        new_rsp_pts.append(new_rsp_pt)

                        res += integrate_recurse(r - p, new_acc_pts, new_rsp_pts, curr_rank + 1, max_rank, emission_compounding,
                                                 rsp_points, pulse_ranges_points, rsp_on_grid, rsp_skip_ranges, pulses_on_grid,
                                                 curr_prod * pulses_on_grid[curr_rank][a_ind])

                    a_ind += 1

                return res

        integration_result = np.zeros((sumfreq_points[1] - sumfreq_points[0] + 1,), dtype=complex)
        emission_compounding = comp_pattern[-1] > 1

        if not parallel:

            for p in range(sumfreq_points[0], sumfreq_points[1] + 1):

                pts_seed = []
                rsp_pt_seed = []

                integration_result[p - sumfreq_points[0]] = integrate_recurse(p, pts_seed, rsp_pt_seed, 0,
                                                                              len(pulse_ranges_points) - 1,
                                                                              emission_compounding,
                                                                              rsp_points,
                                                                              pulse_ranges_points,
                                                                              rsp_on_grid, rsp_skip_ranges,
                                                                              pulses_on_grid, 1.0)

            return integration_result

        else:

            def integrate_part(pts, offset,
                               emission_compounding, rsp_points, pulse_ranges_points,
                               rsp_on_grid, rsp_skip_ranges, pulses_on_grid, mp_queue):

                integration_result = np.zeros(len(pts), dtype=complex)

                for p in range(len(pts)):

                    pts_seed = []
                    rsp_pt_seed = []

                    integration_result[p] = integrate_recurse(pts[p] + offset, pts_seed, rsp_pt_seed, 0,
                                                              len(pulse_ranges_points) - 1,
                                                              emission_compounding,
                                                              rsp_points,
                                                              pulse_ranges_points,
                                                              rsp_on_grid, rsp_skip_ranges,
                                                              pulses_on_grid, 1.0
                                                              )


                mp_queue.put((pts, integration_result))

                return

            import multiprocessing as mp

            prev_start = 0
            num_units = sumfreq_points[1] - sumfreq_points[0] + 1
            # For now just a test choice
            pts_per_job = 50

            num_cores = mp.cpu_count()

            em_comp_core = []
            rsp_p_core = []
            pulse_r_core = []
            rsp_g_core = []
            rsp_sr_core = []
            pulse_g_core = []

            for c in range(num_cores):
                em_comp_core.append(copy.deepcopy(emission_compounding))
                rsp_p_core.append(copy.deepcopy(rsp_points))
                pulse_r_core.append(copy.deepcopy(pulse_ranges_points))
                rsp_g_core.append(copy.deepcopy(rsp_on_grid))
                rsp_sr_core.append(copy.deepcopy(rsp_skip_ranges))
                pulse_g_core.append(copy.deepcopy(pulses_on_grid))

            while prev_start < num_units:

                mp_queue = mp.Queue()

                processes = [mp.Process(
                    target = integrate_part, args =
                    (list(range(
                        prev_start + c * pts_per_job,
                        min(prev_start + (c + 1) * pts_per_job, num_units))),
                         sumfreq_points[0], em_comp_core[c], rsp_p_core[c], pulse_r_core[c],
                         rsp_g_core[c], rsp_sr_core[c], pulse_g_core[c], mp_queue)
                ) for c in range(num_cores)]

                for proc in processes:
                    proc.start()

                for proc in processes:
                    proc.join()

                for proc in processes:
                    proc.close()

                results = [mp_queue.get() for proc in processes]

                for r in results:
                    for elem in range(len(r[0])):

                        integration_result[r[0][elem]] = r[1][elem]

                prev_start += pts_per_job * num_cores

            return integration_result


class DirectIntegrator:

    def __init__(self):

        pass

    def integrate(self):

        # Here parallelize as needed (easy to parallelize over time points)

        integration_result = []

        return integration_result

def pad_for_ft(w_data, grid_range):

    max_abs_sample_point = max(abs(grid_range[0]), abs(grid_range[1]))

    padded_array = np.zeros(2 * max_abs_sample_point + 1, dtype=complex)
    padded_array[grid_range[0] + max_abs_sample_point:grid_range[1] + max_abs_sample_point + 1] = w_data

    return padded_array


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
    rsp_skip_ranges: dict
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

        self.int_strategy = self.__get_int_strategy()

        self.int_results = self.__integrate_all_tasks()


    def __get_int_strategy(self):

        # First pass at strategy selection logic
        # FIXME: This logic may need revision subject to changes to improve resolution of the suppl_t_grid
        # design choice above
        if self.style == 'sum_freq':
            int_strategy = 'sum_freq'

        elif self.style == 'direct':

            if self.suppl_t_grid is None:
                raise ValueError('A supplementary time-domain grid is needed for direct integration')

            else:
                int_strategy = 'direct'

        # The 'auto' integration choice for now chooses 'sum_freq' if freq. detection or time/integrated time
        # detection with no supplementary time-domain grid, or 'direct' if time/integrated time
        # detection with a supplementary time-domain grid given
        elif self.style == 'auto':

            if self.detector.detection_method == 'freq':
                int_strategy = 'sum_freq'

            elif self.detector.detection_method in ['time', 'int']:

                # FIXME: Choosing 'sum_freq' in this situation may perform slower than necessary
                # because the time-domain bins are fixed by the frequency-domain bins (and may be different from
                # than what is sufficient in the specific situation). Not sure whether to warn about this - will first
                # see if it's actually a problem in practice first
                if self.suppl_t_grid is None:
                    int_strategy = 'sum_freq'
                else:
                    int_strategy = 'direct'

        return int_strategy

    def __integrate_all_tasks(self):

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

            # Main integration calls

            int_results = {}

            int_results_sum = {}

            pltd_rsp = False

            # For each scan point
            for scan_elem in self.int_ranges_scan:

                ctime = time.time()

                print('Scan element:', scan_elem)

                int_results_sum[scan_elem] = np.zeros((sumfreq_points[1] - sumfreq_points[0] + 1,), dtype=complex)

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
                                self.pulse_banks_scan[scan_elem][ocmpi[4][p]].evalFreqSpace(
                                    np.linspace(curr_int_ranges_snap[p][0], curr_int_ranges_snap[p][1],
                                        pulse_ranges_points[p][1] - pulse_ranges_points[p][0] + 1)
                                )
                            )

                    rsp_points = []

                    for ax in self.rsp_results[ocmpi].axis_points:
                        rsp_points.append([round(self.rsp_results[ocmpi].axis_points[ax][0] / self.resolution),
                                           round(self.rsp_results[ocmpi].axis_points[ax][-1] / self.resolution)])


                    # Integrate and yield as array
                    int_results[scan_elem][ocmpi] = integrator.integrate(
                        ocmpi[1],
                        sumfreq_points,
                        rsp_points,
                        pulse_ranges_points,
                        self.rsp_results[ocmpi].amplitude,
                        self.rsp_skip_ranges[ocmpi],
                        pulses_on_grid,
                        parallel=True
                    )


                    if not pltd_rsp:

                        rsp_intn = np.square(np.absolute(self.rsp_results[ocmpi].amplitude))

                        z_max = np.amax(rsp_intn)
                        spec_fig = plt.figure(figsize=(6, 6))
                        ax_def = spec_fig.add_subplot(1, 1, 1)
                        # Logarithmic plot


                        palette = ax_def.pcolor(np.arange(rsp_points[1][0]*self.resolution, (rsp_points[1][1] + 1)*self.resolution, self.resolution),
                                                -1.0*np.arange(rsp_points[0][0] * self.resolution, (rsp_points[0][1] + 1) * self.resolution, self.resolution),
                                                rsp_intn,
                                                norm=colors.LogNorm(vmin=z_max / 1000, vmax=z_max),
                                                cmap='hot_r', shading='nearest')
                        ax_def.set_aspect('equal')
                        colorbar = spec_fig.colorbar(palette, ax=ax_def, extend='max')
                        spec_fig.savefig('rsp_plot.png', dpi=180)
                        plt.close(spec_fig)

                        pltd_rsp = True



                    #plt.plot(np.arange(pulse_ranges_points[0][0] * self.resolution, (pulse_ranges_points[0][1] + 1) * self.resolution,
                    #                   self.resolution),
                    #         pulses_on_grid[0])
                    #plt.savefig('p1_' + str(scan_elem) + '.png')
                    #plt.close()
                    #plt.plot(np.arange(pulse_ranges_points[1][0] * self.resolution, (pulse_ranges_points[1][1] + 1) * self.resolution,
                    #                   self.resolution),
                    #         pulses_on_grid[1])
                    #plt.savefig('p2_' + str(scan_elem) + '.png')
                    #plt.close()

                    #plt.plot(np.arange(pulse_ranges_points[2][0] * self.resolution, (pulse_ranges_points[2][1] + 1) * self.resolution,
                    #                   self.resolution),
                    #         pulses_on_grid[2])
                    #plt.savefig('p3_' + str(scan_elem) + '.png')
                    #plt.close()



                print('Processed in:', time.time() - ctime)

                # FIXME: Decide which of the detailed data will also be yielded on return

                for ocmpi in int_results[scan_elem]:
                    int_results_sum[scan_elem] += int_results[scan_elem][ocmpi]


            if self.detector.detection_method == 'freq':

                intensity_results_freq = {}

                for scan_elem in int_results_sum:
                    intensity_results_freq[scan_elem] = np.square(np.absolute(int_results_sum[scan_elem]))

                return intensity_results_freq

            intensity_results_freq = {}


            #for scan_elem in int_results_sum:
            #    intensity_results_freq[scan_elem] = np.square(np.absolute(int_results_sum[scan_elem]))
            #
            #    plt.plot(np.arange(sumfreq_points[0]*self.resolution, (sumfreq_points[1] + 1)*self.resolution, self.resolution ),
            #        intensity_results_freq[scan_elem])
            #    plt.savefig('plt_' + str(scan_elem) + '.png')
            #    plt.close()


            if self.detector.detection_method in ['time', 'int']:

                int_results_t = {}
                intensity_results_time = {}

                # FIXME: If also yielding detailed data, set up to yield time data over ocmpi?

                for scan_elem in int_results_sum:

                    # TODO: FT to time series
                    int_results_t[scan_elem] = np.array(np.fft.fft(np.fft.fftshift(pad_for_ft(int_results_sum[scan_elem], sumfreq_points))))

                for scan_elem in int_results_sum:
                    intensity_results_time[scan_elem] = np.square(np.absolute(int_results_t[scan_elem]))
                #
                #    plt.plot(range(len(intensity_results_time[scan_elem])), intensity_results_time[scan_elem])
                #    plt.savefig('plt_t' + str(scan_elem) + '.png')
                #    plt.close()

                # print('int res time', intensity_results_time)

                if self.detector.detection_method == 'time':

                    return int_results_t

                elif self.detector.detection_method == 'int':

                    int_results_time_integrated = {}

                    for scan_elem in int_results_t:

                        # TODO: Use appropriate dt
                        int_results_time_integrated[scan_elem] = scint.trapezoid(intensity_results_time[scan_elem])

                    return int_results_time_integrated

