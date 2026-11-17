from dataclasses import dataclass
from functools import partial
from collections.abc import Sequence
from .simulation_core import SimulationCore
from ..wilson_system.system_data import MolSystemData
from math import floor, ceil
import numpy as np


@dataclass
class WilsonSimulation:
    """
    Class to organize and carry out a spectral calculation

    core: A SimulationCore instance holding information about the experiment and the terms derived to simulate it

    system_data: A MolSystemData instance holding the molecular information needed to evaluate the terms derived to
    simulate the experiment

    TODO: These are draft choices for the parameters - I think they will become a default way of specifying the setup
     but it may be that some situations call for more detail

    states_choice: String: Regime for vibrational states in resonance conditions and perturbed wavefn difference terms:
    Should be either "harmonic" or "anharmonic"

    dynrange: Float or integer: Dynamic range for response amplitude evaluation

    integration_resolution: Integer: A choice describing the resolution (units: cm^-1) to be used for a
    time/freq series. The range would follow from the detector's detector_interval attribute, or if that is not
    specified (or integrated detection) there would need to be an auto choice made (and registered) here.

    integration_style: String: A choice describing which style of integration will be used. Keeping in mind that other
    development may generalize this, the current span of options that the integrator might recognize are
    "sum_freq" (default), which integrates over sum-frequency components w1, w1 + w2, ..., and "direct", which
    integrates over (non-sum) frequency components w1, w2, ... directly. With the "sum_freq" style, the final w_sigma
    integration with e^i w_sigma t can be omitted to directly yield a w_sigma data series, useful for spectrometer
    detection. It is likely that to begin with, "sum-freq" will be the only available choice for the integrator. I let
    the default choice be "auto", which leaves it to the integrator to make the choice.

    integration_grid_origin: Float or integer (default 0.0): Grids in frequency space will be aligned so that a grid
    point coincides exactly with this origin.

    n_dev_w_int: Float or integer: Positive/negative cutoff (around maximum) for frequency window of (compound) pulses:
    Integration ranges will be limited by the windows dictated by this parameter.

    Creates these derived attributes:

    int_ranges: Dictionary: Keys are tuples of (order, compounding pattern, magnitude condition,
    macroscopic polarization and interaction sequence), values are frequency ranges for each variable
    that cover all integrations over all scan points
    """

    core: SimulationCore
    system_data: MolSystemData

    states_choice: str
    dynrange: float | int

    # FIXME: Consider wrapping these in an IntegrationSettings class or as an an instance of an Integrator
    # class that yields the integration results in some form as a method

    integration_resolution: int | float
    integration_style: str = 'auto'
    integration_grid_origin: float | int  = 0.0
    n_dev_w_int: float | int = 5.0


    def __post_init__(self):

        if not self.states_choice in ['harmonic', 'anharmonic']:
            raise ValueError('states_choice must be either harmonic or anharmonic')

        # Grid origin mod resolution
        self.grid_origin_mod_res = self.integration_grid_origin % self.integration_resolution

        # Get integration ranges for each integration task tuple (indexed by order, compounding pattern,
        # magnitude condition, macroscopic polarization and interaction sequence)
        self.int_ranges = self.__get_integration_ranges()

        # Call evaluator and get/organize return class instances
        self.rsp_results = self.__get_response_function_elements()

        # Evaluate response result and characterize/categorize zero intervals
        # FIXME: For now include everything and consider masking later

        # Integrator run; get main return data and diagnostics
        self.int_results = self.__get_integrated_results()

    def __get_integration_ranges(self):
        """
        Integrator "dry-run": Get integration ranges for each integration task
        """

        from wilson_suite.wilson_utils.unit_convertor import per_cm_x_fs

        int_ranges = {}

        # Scan elements
        for scan_elem in np.ndindex(self.core.experiment.field_analyses.shape):
            # NOTE: There should be optimization opportunities here if needed: Currently, this code
            # constructs integration ranges scan point by scan point which will usually be quite suboptimal;
            # however, analyzing the actual scans offers a chance to recognize invariance or recognize some
            # ranges more directly

            pulse_bank = self.core.experiment.field_analyses[scan_elem].all_valid_pulses

            int_sequences = self.core.experiment.field_analyses[scan_elem].int_sequences_by_order_comp_pol_magn

            # Order, compounding patters, magnitude condition, (macroscopic) polarization tuple
            for ocmp in int_sequences:

                # Translate "m" of ocmp to common "M" index
                found_in_bank = False
                m_common = 0
                curr_magn = self.core.experiment.field_analyses[scan_elem].magnitude_condition_bank[ocmp[2]]

                for i in self.core.coll_magn_bank:
                    if self.core.coll_magn_bank[i] == curr_magn:
                        found_in_bank = True
                        break
                    m_common += 1

                if not found_in_bank:
                    raise ValueError('Magnitude conditions not found in collective bank')

                # Translate "p" of ocmp to common "P" index
                found_in_bank = False
                p_common = 0
                curr_pol = self.core.experiment.field_analyses[scan_elem].pol_avg_vector_bank[ocmp[3]]

                for i in self.core.coll_pol_bank:
                    if self.core.coll_pol_bank[i] == curr_pol:
                        found_in_bank = True
                        break
                    p_common += 1

                if not found_in_bank:
                    raise ValueError('Polarization setup not found in collective bank')

                # Construct current ranges

                # Interaction sequences at this "ocmp tuple"
                for int_seq in int_sequences[ocmp]:

                    # ocMPi is an index which also includes the interaction sequence
                    ocmpi = (ocmp[0], ocmp[1], m_common, p_common, int_seq)

                    # The ranges will also be specific to interaction sequences and will here be kept separately
                    # NOTE: I think they will in general need to be handled separately during the integration,
                    # and any situation where they can usefully be combined (if there actually are any) can be
                    # considered later as optimization

                    # Make the freq ranges for each (compound) pulse in this sequence
                    # Taking carrier freq. -/+ n_dev * freq. domain deviation, then snap to next lower resp. larger
                    # grid point
                    curr_freq_ranges_raw = [[pulse_bank[p].cf - self.n_dev_w_int * (per_cm_x_fs / pulse_bank[p].dev),
                                         pulse_bank[p].cf + self.n_dev_w_int * (per_cm_x_fs / pulse_bank[p].dev)]
                                        for p in int_seq]

                    curr_freq_ranges = []

                    for p in curr_freq_ranges_raw:

                        range_low = self.grid_origin_mod_res + self.integration_resolution * floor(
                            (p[0] - self.grid_origin_mod_res) / self.integration_resolution
                        )

                        range_high = self.grid_origin_mod_res + self.integration_resolution * ceil(
                            (p[1] - self.grid_origin_mod_res) / self.integration_resolution
                        )

                        curr_freq_ranges.append([range_low, range_high])

                    # Check if an entry exists for this ocMPi; if not then initalize it with the current ranges
                    if not (ocmpi) in int_ranges:

                        int_ranges[ocmpi] = curr_freq_ranges

                    # Otherwise make union of current and existing ranges
                    # Note: If the current and existing ranges are disjoint, the intervening interval will be included
                    # as well. This is done on the assumption that most scans will altogether create a simply connected
                    # domain. If it doesn't, this will most likely only result in some unnecessary evaluation
                    else:

                        for p in range(len(int_seq)):
                            int_ranges[ocmpi][p] = [
                                min(int_ranges[ocmpi][p][0], curr_freq_ranges[p][0]),
                                max(int_ranges[ocmpi][p][1], curr_freq_ranges[p][1])
                            ]

        return int_ranges


    def __get_response_function_elements(self):
        """
        Evaluate all response function elements that will be used in the integration
        """

        from ..wilson_utils.common_labels import cap_alpha_labels
        from ..wilson_derive.term_var_translate import translate_one_term_to_axis_variables
        from ..wilson_intensities.refac_rsp_eval.pipeline import evaluate_response_terms, spectrum_on_grid
        from ..wilson_intensities.refac_rsp_eval.features import amplitude_half_widths

        # FIXME: Using a trivial lineshape parameter for testing
        lineshape_param = 5.0

        rsp_results = {}

        # Translate term to axis variables
        for ocmpi in self.int_ranges:

            terms_axis_vars = []

            axis_translation = {}
            window_box = {}

            points_per_axis = {}

            for ax in range(len(ocmpi[4])):
                axis_translation[ tuple([i + 1 for i in range(len(ax) + 1)]) ] = {cap_alpha_labels[ax]: 1.0}

                # Window boxes snap to overall grid by the range construction
                window_box[cap_alpha_labels[ax]] = (self.int_ranges[ocmpi][ax][0], self.int_ranges[ocmpi][ax][1])

                points_per_axis[cap_alpha_labels[ax]] = int(round(
                    (self.int_ranges[ocmpi][ax][1] - self.int_ranges[ocmpi][ax][0])/self.integration_resolution ) ) + 1

            for t in self.core.terms_by_order_comp_magn[(ocmpi[0], ocmpi[1], ocmpi[2])]:
                terms_axis_vars.append(translate_one_term_to_axis_variables(t, axis_translation))

            term_eval_result = evaluate_response_terms(terms_axis_vars,
                                                        molsys_data=self.system_data,
                                                        states_choice=self.states_choice,
                                                        laser_pol=self.core.coll_pol_bank[ocmpi[3]],
                                                        lineshape_param=lineshape_param
                                                       )

            rsp_result = spectrum_on_grid(term_eval_result,
                                          window_box=window_box,
                                          n_points_per_axis=points_per_axis,
                                          box_rule=partial(
                                              amplitude_half_widths, dynrange=self.dynrange
                                            )
                                          )

            rsp_results[ocmpi] = rsp_result

        return rsp_results

    def __get_integrated_results(self):



        int_results = []


        return int_results