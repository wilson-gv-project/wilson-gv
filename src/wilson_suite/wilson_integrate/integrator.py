from dataclasses import dataclass
from collections.abc import Sequence
from ..wilson_experiment.experiment_abstractions import SpecDetector, FieldAnalysis


@dataclass
class SpectralIntegration:
    """Class to carry out (nested) integrations """

    detector: SpecDetector
    # FIXME: Review this declaration
    # Should contain all pulse banks over all scan points (i.e. would be taken from each FieldAnalysis instance)
    pulse_banks: Sequence
    # TODO: May be enough with rsp_results on its own and not int_ranges
    int_ranges: dict
    rsp_results: dict
    resolution: int | float
    style: str
    grid_origin: float | int

    # FIXME: Provisional design choice: If data is to be yielded in the time domain
    # with "direct" integration, an additional grid of time points should be provided
    # This is an incomplete solution because a) the detector may specify a desired range
    # (but currently not a resolution) for this, b) a reasonable range (but not necessarily
    # a resolution) may be determined by inspecting the experiment time scale
    # TODO: Work out a consistent regime for handling this in a good way
    suppl_t_grid: Sequence | None = None

    def __post_init__(self):

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


    def __integrate_all_tasks(self, field):

        if not self.int_strategy == 'sum_freq':
            raise NotImplementedError('Only sum-frequency integration currently implemented')

        if self.int_strategy == 'sum_freq':


            if self.detector.detection_method == 'freq':

                # If the detector is 'freq' and has a specified range, use it (snapping to overall grid points)
                # Otherwise, walk through all integration ranges to find lower and upper overall bounds
                pass


            elif self.detector.detection_method in ['time', 'int']:

                # Here, walk through integration ranges and find lower/upper overall bounds
                # The time series will follow from Fourier transformation (and zero padding)

                pass

            # At this point a sum-frequency range will have been selected. The integration can now be done as:

            # - For each scan point:
            #   - For each ocmpi:
            #       - Determine individual frequency ranges for the various compound pulses (FIXME: Or do this outside?)
            #       - Walk through all sum-frequency components:
            #           - Make a nested summation for each integration (choosing freq. components that sum to that
            #             sum-freq component) and accumulate
            # - Afterwards, yield data as is for frequency detection or FT and integrate for time/int detection
            #   - By default, yield amplitudes for all scan points and ocmpi
            #       - For large scans, optionally suppress one or both levels
            # - Also add up and yield amplitudes for time/freq series
            # - Also yield intensities for time/freq series and integrated intensity

        pass