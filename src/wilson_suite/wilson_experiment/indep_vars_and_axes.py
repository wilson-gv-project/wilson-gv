from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Iterable
import copy
from wilson_suite.wilson_experiment.experiment_abstractions import EmPulse, get_carrier_freqs_uv, uv_cancels

@dataclass
class SignedPulseTuple:
    """
    Class to represent a collection of signed references to pulse IDs. Applicable as a representation of an independent
    variable, but not limited to this.

    pulse_refs: A tuple of signed integer references to pulse IDs
    Example: For the independent variable -w1 + w2, a pulse_refs representation is (-1, 2)
    Example: For the phase-matching condition -k1 + k2 + k3, a pulse_refs representation is (-1, 2, 3)
    """

    pulse_refs: tuple

    def __post_init__(self):

        if not isinstance(self.pulse_refs, tuple):
            raise TypeError('pulse_refs must be a tuple of integers')

        else:
            for i in self.pulse_refs:
                if not isinstance(i, int):
                    raise TypeError('pulse_refs must be a tuple of integers')


@dataclass
class PhaseMatchingCondition:
    """
    Class to represent a phase-matching condition

    pulses: A SignedPulseTuple instance defining the phase-matching conditions
    id: Optional integer argument defining an identifier for this phase-matching condition (default: None)
    """

    pulses: SignedPulseTuple
    id: int = None

    def __post_init__(self):

        if not isinstance(self.pulses, SignedPulseTuple):
            raise TypeError('pulses must be a SignedPulseTuple instance')

        if not isinstance(self.id, int):
            if not self.id == None:
                raise TypeError('If not None, self.id must be nonnegative integer')
        else:
            if self.id < 0:
                raise TypeError('self.id must be nonnegative integer')

@dataclass
class IndependentVariableSet:
    """
    Class to represent a group of independent variables.

    var_set: A tuple of SignedPulseTuple instances
    """

    var_set: tuple[SignedPulseTuple]

    def __post_init__(self):

        if not isinstance(self.var_set, tuple):
            raise TypeError('var_set must be a tuple of SignedPulseTuple instances')
        else:
            for i in self.var_set:
                if not isinstance(i, SignedPulseTuple):
                    raise TypeError('var_set must be a tuple of SignedPulseTuple instances')

        enc_ids = []

        for i in self.var_set:
            for j in i.pulse_refs:
                if j >= 0:
                    if not j in enc_ids:
                        enc_ids.append(j)
                    else:
                        raise ValueError('Pulse IDs must not repeat in variable set')
                else:
                    if not -1 * j in enc_ids:
                        enc_ids.append(-1 * j)
                    else:
                        raise ValueError('Pulse IDs must not repeat in variable set')

@dataclass
class IndependentVariableChoices:
    """
    Class to represent a valid set of independent variables choices for a phase-matching condition

    phasematch_cond: A PhaseMatchingCondition instance defining the phase-matching condition for which this variable
    set is specified

    var_groups: A tuple of IndependentVariableSet instances describing different choices of independent variable sets

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

    def __post_init__(self):

        if not isinstance(self.phasematch_cond, PhaseMatchingCondition):
            raise TypeError('phasematch_cond must be a PhaseMatchingCondition instance')
        if not isinstance(self.var_groups, tuple):
            raise TypeError('var_groups must be a tuple of IndependentVariableSet instances')
        else:
            for i in self.var_groups:
                if not isinstance(i, IndependentVariableSet):
                    raise TypeError('var_groups must be a tuple of IndependentVariableSet instances')

@dataclass
class SpectralAxis:
    """
    Class to represent a choice of spectral axis

    label: A string describing the name of this axis
    var_set: An IndependentVariableSet instance describing the independent variables making up this axis
    """

    label: str
    var_set: IndependentVariableSet

    def __post_init__(self):

        if not isinstance(self.label, str):
            raise TypeError('label must be a string')
        if not isinstance(self.var_set, IndependentVariableSet):
            raise TypeError('var_set must be an IndependentVariableSet instance')

@dataclass
class SpectralAxisSet:
    """
    Class to represent a full choice of spectral axes

    axes: A tuple of SpectralAxis instances. All axes must have different names
    """

    axes: tuple[SpectralAxis]

    def __post_init__(self):

        if not isinstance(self.axes, tuple):
            raise TypeError('axes must be a tuple of SpectralAxis instances')
        else:
            for i in self.axes:
                if not isinstance(i, SpectralAxis):
                    raise TypeError('axes must be a tuple of SpectralAxisSet instances')

        enc_names = []

        for i in self.axes:
            if not i.label in enc_names:
                enc_names.append(i.label)
            else:
                raise ValueError('All axes in set must have unique labels')



@dataclass
class SpectralAxisChoices:
    """
    Class to represent a valid set of choices of axes for a choice of independent variables and a phase-matching condition

    phasematch_cond: A PhaseMatchingCondition instance describing the phase-matching condition for which this data was determined

    ind_vars: An IndependentVariableSet instance describing a choice ("group") of independent variables in terms of (signed) pulse IDs
    ((variable 1 (all signed) pulse ID 1, ID 2, ...), (variable 2 (all signed) pulse ID 1, ID 2, ...)). See explanation
    of var_set in class IndependentVariableSet for more details.

    valid_axis_combs: A tuple of SpectralAxisSet instances, each describing a valid choice of axes for this choice of
    independent variables.

    Example 1: For the independent variables -w1 and w2, suppose that two valid choices of axes were identified:
        - Axis 'A': -w1 and axis 'B': w2
        - Axis 'A': -w1 and axis 'B': -w1 + w2

    Then valid_axis_combs would be of the form (I, II) where
        - I is SpectralAxisSet instance with 'axes' attribute of the form (i, ii), where
            - i is a SpectralAxis instance with label attribute 'A' and var_group attribute with the var_set attribute
            on the form (X,), where
                - X is a SignedPulseTuple instance with the pulse_refs attribute (-1,)
            - ii is a SpectralAxis instance with label attribute 'B' and var_group attribute with the var_set attribute
            on the form (X,), where
                - X is a SignedPulseTuple instance with the pulse_refs attribute (2,)
        - II is SpectralAxisSet instance with 'axes' attribute of the form (i, ii), where
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
    - I is SpectralAxisSet instance with 'axes' attribute of the form (i, ii), where
        - i is a SpectralAxis instance with label attribute 'A' and var_group attribute with the var_set attribute
        on the form (X,), where
            - X is a SignedPulseTuple instance with the pulse_refs attribute (-1, 2)
        - ii is a SpectralAxis instance with label attribute 'B' and var_group attribute with the var_set attribute
        on the form (X,), where
            - X is a SignedPulseTuple instance with the pulse_refs attribute (3,)
    - II is SpectralAxisSet instance with 'axes' attribute of the form (i, ii), where
        - i is a SpectralAxis instance with label attribute 'A' and var_group attribute with the var_set attribute
        on the form (X,), where
            - X is a SignedPulseTuple instance with the pulse_refs attribute (-1, 2)
        - ii is a SpectralAxis instance with label attribute 'B' and var_group attribute with the var_set attribute
        on the form (X, Y), where
            - X is a SignedPulseTuple instance with the pulse_refs attribute (-1, 2)
            - X is a SignedPulseTuple instance with the pulse_refs attribute (3,)

    """

    phasematch_cond: PhaseMatchingCondition
    ind_vars: IndependentVariableSet
    valid_axis_combs: tuple[SpectralAxisSet]

    def __post_init__(self):

        if not isinstance(self.phasematch_cond, PhaseMatchingCondition):
            raise TypeError('phasematch_cond must be PhaseMatchingCondition instance')
        if not isinstance(self.ind_vars, IndependentVariableSet):
            raise TypeError('ind_vars must be IndependentVariableSet instance')
        if not isinstance(self.valid_axis_combs, tuple):
            raise TypeError('phasematch_cond must be a tuple of SpectralAxisSet instances')
        else:
            for i in self.valid_axis_combs:
                if not isinstance(i, SpectralAxisSet):
                    raise TypeError('phasematch_cond must be a tuple of SpectralAxisSet instances')


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
    Make branching choices of different valid UV/VIS partitioning choices (tail-recursive). IR-range pulses are
    added without branching since they each constitute an independent variable.

    combs: list: Accumulated combinations (tail-recursion result)
    orig_vars: list: Original variables (with branches as multi-entry lists)
    curr_comb: list: Branching choice in-recursion accumulator
    curr_epoch: int: Current epoch counter
    """

    # Termination criterion
    if curr_epoch == len(orig_vars):
        combs.append(copy.deepcopy(curr_comb))

    # Otherwise recurse over epochs
    else:

        # Take incoming combination and copy it for amendment and recursion
        new_comb_ir = copy.deepcopy(curr_comb)

        # First add all IR range pulses in this epoch (No
        uv_start = 0
        for i in orig_vars[curr_epoch]:

            if not isinstance(i, list):
                uv_start += 1
                new_comb_ir.append(i)

        if not (uv_start == (len(orig_vars[curr_epoch]))):

            for i in orig_vars[curr_epoch][uv_start]:
                new_comb_ir_uv = copy.deepcopy(new_comb_ir)
                new_comb_ir_uv.extend(i)

                find_branching_indep_var_combs(combs, orig_vars, new_comb_ir_uv, curr_epoch + 1)

        else:

            find_branching_indep_var_combs(combs, orig_vars, new_comb_ir, curr_epoch + 1)


def find_indep_vars_for_one_phasematch(pulses: list[EmPulse], epochs: list, pm_dir: SignedPulseTuple) -> tuple[
    IndependentVariableSet]:
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

    # Translate SignedPulseTuple data to form used in this fn {pulse ID: sign, ...}
    pm_dir_dict = {}
    for i in pm_dir.pulse_refs:
        if i < 0:
            pm_dir_dict[i * - 1] = -1
        else:
            pm_dir_dict[i] = 1

    # Loop over epochs
    for i in range(len(epochs)):

        raw_ind_vars_p_epoch = []
        cfuv_this_pm = {}
        uv_this = []
        ir_this = []

        # Pulses in epoch
        for k in epochs[i]:

            # Dress UV/VIS-range pulses with phase-matching sign
            if not (cfuv[k] == 0.0):
                cfuv_this_pm[k] = cfuv[k] * pm_dir_dict[k]
                uv_this.append(k * pm_dir_dict[k])

            else:
                ir_this.append(k * pm_dir_dict[k])

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
        if not (uv_subs_res == []):
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

    # Translate to IndependentVariableSet instance

    transl_sets = []

    # Each i should result in an IndependentVariableSet
    for i in ind_vars_p:

        my_new_ind_vars = []
        for j in i:
            my_new_ind_vars.append(SignedPulseTuple(j))

        transl_sets.append(IndependentVariableSet(tuple(my_new_ind_vars)))

    return tuple(transl_sets)


def find_indep_exp_variables(pulses: list[EmPulse], epochs: list, phasematch_dirs: list[PhaseMatchingCondition]) -> \
list[IndependentVariableChoices]:
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

    all_ind_var_cfgs_p = []

    for p in phasematch_dirs:

        all_ind_var_cfgs_p.append(
            IndependentVariableChoices(p, find_indep_vars_for_one_phasematch(pulses, epochs, p.pulses)))

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
        prev_var_superset = set(
            list(chain.from_iterable(combinations(ind_vars[:pos], r) for r in range(len(ind_vars[:pos]) + 1)))[1:])

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


def find_valid_axes_cfgs_for_one_phasematch(ind_vars: IndependentVariableChoices) -> dict[tuple, list[SpectralAxisSet]]:
    """
    Find valid axes choices for one phase-matching direction.

    ind_var: List of lists of tuples: independent variables for one phase-matching condition.
                    See return structure of find_indep_exp_variables.

    Returns: dict: valid_axes: For each set of independent variable choices (keys), return a list of
    valid axis choices. Each such list entry is a SpectralAxisSet instance
    """

    from wilson_suite.wilson_utils.common_labels import cap_alpha_labels
    from itertools import permutations

    valid_axes = {}
    seed_ax_list = []

    # Translate to internal format

    ind_vars_internal = []

    for i in ind_vars.var_groups:

        new_group = []

        for j in i.var_set:
            new_group.append(j.pulse_refs)

        ind_vars_internal.append(copy.deepcopy(new_group))

    # Loop over independent variable collection choices
    for i in ind_vars_internal:

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
                new_dress_v_a[cap_alpha_labels[k]] = j[k]

            dressed_valid_axes.append(copy.deepcopy(new_dress_v_a))

        # Enter in dictionary by sorted independent variable tuples
        valid_axes[tuple(sorted(i))] = copy.deepcopy(dressed_valid_axes)

    # Translate to SpectralAxisChoices
    transl_valid_axes = {}

    for i in valid_axes:

        new_combs = []

        # Each j to yield a SpectralAxisSet
        for j in range(len(valid_axes[i])):

            new_set = []

            for k in valid_axes[i][j]:

                this_axis_set = []

                for m in valid_axes[i][j][k]:
                    this_axis_set.append(SignedPulseTuple(m))

                new_set.append(SpectralAxis(k, IndependentVariableSet(tuple(this_axis_set))))

            new_combs.append(SpectralAxisSet(tuple(new_set)))

        transl_valid_axes[i] = tuple(new_combs)

    return transl_valid_axes


def find_canonical_axes_for_one_phasematch(ind_vars: IndependentVariableChoices) -> SpectralAxisSet:
    """
    Find valid axes choices for one phase-matching direction.

    ind_var: List of lists of tuples: independent variables for one phase-matching condition.
                    See return structure of find_indep_exp_variables.

    Returns: dict: valid_axes: For each set of independent variable choices (keys), return a list of
    valid axis choices. Each such list entry is a SpectralAxisSet instance
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

    # Translate to internal format

    ind_vars_internal = []

    for i in ind_vars.var_groups:

        new_group = []

        for j in i.var_set:
            new_group.append(j.pulse_refs)

        ind_vars_internal.append(copy.deepcopy(new_group))

    # Find entry/-ies with the most independent variables
    for i in ind_vars_internal:

        if len(i) == max_len_ind:
            max_len_entries.append(copy.deepcopy(i))

        elif len(i) > max_len_ind:
            max_len_ind = len(i)
            max_len_entries = [copy.deepcopy(i)]

    max_len_entries = sorted(max_len_entries)

    # Since max len entries is sorted, I can make a canonical choice with the first entry
    for i in range(len(max_len_entries[0])):

        # Dress canonical independent variables with axis labels
        canonical_axes[cap_alpha_labels[i]] = [max_len_entries[0][i]]

    # Translate to SpectralAxisSet
    transl_canonical_axes = {}
    new_set = []

    for k in canonical_axes:

        this_axis_set = []

        for m in canonical_axes[k]:
            this_axis_set.append(SignedPulseTuple(m))

        new_set.append(SpectralAxis(k, IndependentVariableSet(tuple(this_axis_set))))

    return SpectralAxisSet(tuple(new_set))


# TODO: Have option to let user fix one or more axes and recurse starting from that instead
# TODO: Add support for more than one independent variable choice set (usually means more than one phase-matching
# condition)
def find_canonical_axes(all_ind_var_cfgs_p: list[IndependentVariableChoices]) -> SpectralAxisSet:
    """
    Find canonical axes for a collection of phase-matching directions.

    all_ind_var_cfgs_p: List of IndependentVariableChoices. See that class' definition and find_indep_exp_variables

    Returns: A SpectralAxisSet instance describing the canonical axes
    """


    if len(all_ind_var_cfgs_p) > 1:
        raise ValueError('Only one independent variable choice set currently supported')

    for i in all_ind_var_cfgs_p:

        canonical_axes = find_canonical_axes_for_one_phasematch(i)

        # TODO for > 1 ind var choice set support:
        # Make the canonical axes for the further sets
        # Check if the results are related to the first canonical axes by a simple transformation
        # If no, raise ValueError (no canonical axes could be found)
        # If yes, pass to next set
        # If all pass relation test, return the canonical axes

    return canonical_axes


# TODO: Have option to let user fix one or more axes and recurse starting from that instead
def find_valid_axes(all_ind_var_cfgs_p: list[IndependentVariableChoices]) -> list[SpectralAxisChoices]:
    """
    Find valid axes for a collection of phase-matching directions.

    all_ind_var_cfgs_p: List of IndependentVariableChoices. See that class' definition and find_indep_exp_variables

    Returns: list[SpectralAxisChoices]: final_valid_axes: For each set of independent variable choices, a description of
    valid axis choices. Each such list entry is a SpectralAxisChoices instance
    """

    # Find valid axes for each phase-matching direction
    valid_axes = []

    for i in all_ind_var_cfgs_p:

        new_axes = find_valid_axes_cfgs_for_one_phasematch(i)

        for j in new_axes:

            new_iv_set = IndependentVariableSet(tuple([SignedPulseTuple(k) for k in j]))
            valid_axes.append(SpectralAxisChoices(i.phasematch_cond, new_iv_set, tuple(new_axes[j])))

    return valid_axes
