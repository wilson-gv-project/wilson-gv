"""
PROPERTIES in VibPerturbedTerm ---- #TODO still

---
Claude Code summary

# Unique averaged tensors: identification and use

## The problem

An "averaged expression" is a product of molecular properties with an orientational
average over Cartesian components, e.g.

    polgrad['a'] * dipgrad['b'] * dipgrad['c']

To get a number out of it you must say which normal mode goes into each property slot.
Feed it modes (2, 0, 5) and you get one number; (1, 1, 4) gives another. So the
expression is a FUNCTION of its mode indices.

calculate_avrg_tensor() evaluates that function at every combination of modes and
stores the results in an array with one axis per distinct mode index. Three slots means
a 3-dimensional array; with 30 modes that is 27000 orientational averages. Expensive.

Every term in the response expansion has such an expression. Computing one array per
term would repeat the same work many times over.

## The observation

Two expressions that differ only in the NAMES of their indices are the same function:

    polgrad['a'] * dipgrad['b'] * dipgrad['c']
    polgrad['c'] * dipgrad['a'] * dipgrad['b']

Both say "polgrad gets some mode, each dipgrad gets some mode". The letters are labels,
not content. One array serves both.

For the standard EVV term set: 14 terms -> 7 distinct expressions -> 3 arrays.

## Identification (make_unique_avrg_tensors_mapping)

Step 1 - extract the averaged part of each term
    PropsCollection(term.props).get_averaged_props().sort()
    keeps the properties carrying EM operators, ordered by first operator index.

Step 2 - group by numerator motif        group_PropsColls_by_numerator()
    identify_avrg_motif() copies the expression and sets every index to None, giving a
    key that captures WHICH properties with WHICH operators and derivative orders,
    ignoring index names. Expressions in different motifs are different quantities and
    can never share an array.

Step 3 - key each member by its repetition pattern
                                    nm_indices_repetition_reduce_deriv_symmetry()
    Within a motif, members differ only in which label sits in which slot. Two members
    can share an array iff their labels REPEAT in the same way. The key is built as:

        expr labels           ['b', 'a', 'a', 'b']
        repetition codes      ( 1 ,  2 ,  2 ,  1 )     0 = occurs once
        split per property    [[1], [2], [2, 1]]       template [1, 1, 2]
        sort each group       [[1], [2], [1, 2]]       <-- see note below
        key                   (1, 2, 1, 2)

    NOTE - the sort within each property is the permutational symmetry of a geometric
    derivative in its own mode indices (d2u/dQa dQb == d2u/dQb dQa). It is what lets two
    expressions that differ only in how a derivative's indices line up share one array.
    It assumes the derivative arrays (diphess, polhess, cff, qff) are symmetric in their
    mode indices. Nothing in the code enforces this.

Step 4 - one base expression per key
    The key is decoded to canonical letters and used to refill the motif:

        key (1,2,1,2) -> ['a','b','a','b']
                      -> polgrad['a'] * dipgrad['b'] * diphess['a','b']

    If any member of a motif has ALL-DISTINCT indices, its array has one axis per slot,
    so every other member can be read off it. That member's key becomes the base for the
    whole motif and the pattern groups merge into it.

Result: a mapping {expression -> base expression}. The distinct values are the arrays
that actually have to be computed.

## Use in calculations

Precalculation                          precalculate_unique_coeff_parts()
    One calculate_avrg_tensor() call per distinct base. The array's axes are the base's
    unique labels in ALPHABETICAL order, so for base
    polgrad['a'] * dipgrad['b'] * diphess['a','b']:

        axis 0 = 'a' = the polgrad slot
        axis 1 = 'b' = the dipgrad slot

Evaluation                              eval_avrg_per_indexdict()
    Reading a value out of the base's array takes two hops:

        hop 1   base's index label  ->  this expression's index label
        hop 2   this expression's label  ->  normal mode number   (index_dict)

    get_ind_tuple_from_base() does both and returns the index tuple.

    Hop 1 exists because the base is built from canonical letters, which are in general
    a PERMUTATION of the labels the expression uses:

        expr   polgrad['b'] * dipgrad['a'] * diphess['a','b']
        base   polgrad['a'] * dipgrad['b'] * diphess['a','b']    'a' and 'b' swapped

    With index_dict {'a': 0, 'b': 1}: the expression wants polgrad on mode 1 and dipgrad
    on mode 0, which is tensor[1, 0]. Skipping hop 1 reads tensor[0, 1] - the transposed
    element, silently wrong.

    Hop 1 cannot be done by lining the two expressions up as they stand:

        base:  a  b  a  b
        expr:  b  a  a  b     base 'a' -> 'b', then base 'a' -> 'a': contradiction

    The base's letters were laid down against the codes AFTER those codes were sorted
    within each property (step 3), so the expression's labels must go through the same
    sort first:

        expr sorted:  b  a  b  a
        base:         a  b  a  b     base 'a' -> 'b', base 'b' -> 'a', repeats agree

Downstream
    The coefficient is a product of this averaged value with the non-averaged properties,
    the vibrational energy denominators and the vibrational difference terms. Terms
    sharing a resonance motif and location are SUMMED into one SpectralFeature amplitude,
    which is then dropped if it is exactly zero. So an error in the averaged value can
    change a peak height, or make a peak appear or vanish.

## Rules the mapping obeys

    - The array's axes are the BASE's unique labels, alphabetically.

    - The base -> expression label map must be a well-defined FUNCTION (one value per
      base label). It need NOT be injective.

    - Several base labels -> the same expression label is FINE and expected. The
      expression uses fewer distinct modes than the base has axes, so it reads a
      diagonal of the base's array:

          base ['a','b','c'] serving expr ['a','c','c']
          gives {a: a, b: c, c: c} and index (i_a, i_c, i_c)
          - a rank 3 array serving a rank 2 expression.

    - One base label -> several expression labels is IMPOSSIBLE. The expression needs
      more distinct modes than the base has axes; that base cannot serve it.

    - A base whose letters are a permutation of the expression's is NOT an error - it is
      what the deduplication is for. Nothing is wrong until an index is built from it.

    - Equal slot counts between base and expression do NOT establish that the base can
      represent the expression: two motifs can have the same total slot count
      (polhess*dip*dip and polgrad*dip*diphess both have 4). Sharing a numerator motif is
      what makes a base valid, and it holds by construction.
"""
from typing import Callable
import copy

import numpy as np
from wilson_suite.wilson_intensities.amplitudes.term_parts import PropsCollection
from wilson_suite.wilson_intensities.amplitudes.utils import generate_index_choices_general
from wilson_suite.wilson_main.abstractions import MolPropsCollection

import logging
logger = logging.getLogger("wilson")


def group_PropsColls_by_numerator(list_props_collections: list['PropsCollection']) -> dict[PropsCollection, list[PropsCollection]]:
    """
    [x] DONE
    returns groups of avrg props motifs by numerator
    """
    groups_here: dict['PropsCollection', list] = {}
    for props_collection in list_props_collections:
        # props_collection.identify_avrg_motif() returns props stripped from nm indices
        groups_here.setdefault(props_collection.identify_avrg_motif(), []).append(props_collection)
    return groups_here


def make_gen_func_to_compute_avrg(*,
                              avrg_expression: 'PropsCollection',
                              pulse_polarization_vector: list) -> Callable[[dict, 'MolPropsCollection'], float]:
    """
    for an expression with properties data values,
    compute average with given polarization setup for a choice of normal mode indices
    """
    num_pulses = len(avrg_expression.get_cart_axes())  # should this be a set?
    from .averaging import getGeneralPolarizationAveragingExpression

    polarization_linear_comb = getGeneralPolarizationAveragingExpression(rank = num_pulses,
                                                                        laser_pol = pulse_polarization_vector)

    def compute_for_idx_choice(index_choices: dict, props_data: 'MolPropsCollection') -> float:
        """
        index_choices: dict, props_data: 'MolPropsCollection'
        """

        if not isinstance(props_data, MolPropsCollection):
            raise TypeError(
                f"props_data must be a MolPropsCollection, got {type(props_data).__name__}"
            )

        # Validate index_choices has all required keys
        required_inds = {i for prop in avrg_expression for i in prop.inds}
        missing = required_inds - index_choices.keys()
        if missing:
            raise KeyError(
                f"index_choices is missing required mode indices: {missing}"
            )

        from wilson_suite.wilson_utils.prop_trivname import prop_trivname

        total = 0.

        for cart_axes in polarization_linear_comb:

            # Comment (MR): Noting that I considered if there would be any issues with this in generalized routine,
            # couldn't think of any but want to discuss and double check for safety

            product = 1.

            for prop in avrg_expression:

                prop_tuple_key = prop_trivname(ord_el=len(prop.ops), ord_geo=prop.dord)

                nm_inds = tuple([index_choices[i] for i in prop.inds])
                cart_inds = tuple([cart_axes[i.o] for i in prop.ops])
                all_inds = (*nm_inds, *cart_inds)

                # retrieve data for preperty (prop_key) and idxs_key which is (tuple(mode inds), tuple(cart inds))
                product *= props_data.get(prop_tuple_key).vals[all_inds]

            if product != 0.:
                logger.debug(f"Avrg prop contribution for indices {index_choices} and cart axes {cart_axes} with coefficient {polarization_linear_comb[cart_axes]}: {product}")

            total += product * polarization_linear_comb[cart_axes]

        return total

    return compute_for_idx_choice


def calculate_avrg_tensor(avrg_expression: 'PropsCollection',
                          pulse_polarization_vector: list,
                          props_data: 'MolPropsCollection',
                          number_of_nmodes: int,
                          nm_inds_choices: list[int]):
    """
    Precalculating the full tensor for given avrg_expression

    nm_inds_choices - could be generated with for all normal modes with:
        nm_inds_choices: list[dict[str, int]] = generate_index_choices_general(indlabels_in_motif=mode_inds, labels=list(range(number_of_nmodes)))

    """
    # so indices are in alphabetical order in full_tensor below
    mode_inds = sorted(set(avrg_expression.get_mode_indices()))  # list, deterministic order
    ind_choices: list[dict[str, int]] = generate_index_choices_general(indlabels_in_motif=mode_inds, labels=nm_inds_choices)

    # Indicating generalized version for updating
    func_general = make_gen_func_to_compute_avrg(avrg_expression=avrg_expression, pulse_polarization_vector=pulse_polarization_vector)

    full_tensor = np.zeros((number_of_nmodes,)*len(mode_inds))

    for idx in ind_choices:
        # so indices are in alphabetical order
        full_tensor[tuple(idx[k] for k in mode_inds)] = func_general(idx, props_data)

    return full_tensor


def group_PropsColls_by_repetition_pattern(avrg_expressions: list[PropsCollection]):
    """
    For a list of averaged properties expressions already grouped by numerator motifs

    """
    max_nm_inds = 0
    all_encoded: dict[PropsCollection, list[PropsCollection]] = {}

    for prop_coll in avrg_expressions:
        # number of max of uniques nm indices is equal to number of boxes for derivatives 
        #           - all indices are different and all props are 1st order ders
        nm_indx = prop_coll.get_mode_indices()
        num_unique_nm_idx = len(set(nm_indx))

        # if max number of unique indices is found then 
        # that would be the model expression for the whole group with this numerator motif
        if num_unique_nm_idx == len(nm_indx):
            all_encoded[nm_indices_repetition_reduce_deriv_symmetry(prop_coll)] = []
            for pr_coll in avrg_expressions:
                all_encoded[nm_indices_repetition_reduce_deriv_symmetry(prop_coll)].append(pr_coll)
            return all_encoded
        
        max_nm_inds = max(num_unique_nm_idx, max_nm_inds)
        all_encoded.setdefault(nm_indices_repetition_reduce_deriv_symmetry(prop_coll), []).append(prop_coll)
    
    return all_encoded



def make_unique_avrg_tensors_mapping(avrg_expressions: list[PropsCollection]):
    """

    """
    numerator_groups = group_PropsColls_by_numerator(avrg_expressions)

    numer_upd = {k:group_PropsColls_by_repetition_pattern(v) for k,v in numerator_groups.items()}

    flat_dict = {}
    for num_group in numer_upd:
        for pattern in numer_upd[num_group]:
            new_inds = nm_indices_repetition_decoding(pattern)
            model_expr = reconstruct_unique_avrg_expression(numerator_group=num_group, nm_indices=new_inds)
            for expression in numer_upd[num_group][pattern]:
                flat_dict[expression] = model_expr
                
    return flat_dict

def nm_indices_repetition_encoding(nm_indices: list[str]):
    """
    [a, b, c, d, b] - (0, 2, 0, 0, 2)
    [a, a, b, c, d] - (1, 1, 0, 0, 0)
    [a, b, c, d, d] - (0, 0, 0, 4, 4)
    """
    counts_dict = {i:nm_indices.count(i) for i in nm_indices}
    repeated = {k:i+1 for i,k in enumerate(counts_dict.keys()) if counts_dict[k]>1}

    encoded = [0] * len(nm_indices)
    for i, ind in enumerate(nm_indices):
        encoded[i] = repeated.get(ind, 0)
    return tuple(encoded)

def nm_indices_repetition_decoding(encoded_idx: tuple[int]):
    """
    (0, 2, 0, 0, 2) - [a, b, c, d, b]
    (1, 1, 0, 0, 0) - [a, a, b, c, d]
    (0, 0, 0, 4, 4) - [a, b, c, d, d]
    """
    lat_letters_for_zeros = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    lat_letters_for_ones = copy.deepcopy(lat_letters_for_zeros)
    
    result = []

    for coded in encoded_idx:
        curr_letter = 0
        if coded == 0:
            result.append(lat_letters_for_zeros.pop(curr_letter))

        else:
            result.append(lat_letters_for_ones[coded-1])
            if lat_letters_for_ones[coded-1] in lat_letters_for_zeros:
                lat_letters_for_zeros.remove(lat_letters_for_ones[coded-1])
    return result

def nm_indices_repetition_reduce_deriv_symmetry(props: PropsCollection) -> tuple[int]:
    """
    returns a sorted encoding, unlike nm_indices_repetition_encoding
    """
    nm_group_template = props.get_mode_indices_group_template()
    nm_indices_encoded = nm_indices_repetition_encoding(props.get_mode_indices())
    grouped_coded = group_nm_indices(nm_indices_encoded, nm_group_template)

    for g in grouped_coded:
        g.sort()
    return tuple([el for group in grouped_coded for el in group])

def group_nm_indices(nm_indices, grouping_template) -> list[list]:
    """
    nm_indices - coded or not list of nm indices
    
    ['d', 'd', 'a', 'c', 'b'], [2, 1, 1, 1] --> [['d', 'd'], ['a'], ['c'], ['b']]
    """
    result = []
    curr = 0
    
    for gr in grouping_template:
        result.append(list(nm_indices[curr: curr+gr]))
        curr += gr
    return result

def reconstruct_unique_avrg_expression(numerator_group: 'PropsCollection',
                                       nm_indices: list[str]) -> 'PropsCollection':
    """
    dipNone[0] * dipNone[1] * dipNone[2] * dipNone[3] * dipNone[4]
    hypNone[0, 2, 4] * dipNone[1] * dipNone[3] * dipNone[5]

    """
    upd_props = []
    index_tracker = 0
    
    for prop in numerator_group:
        prop = copy.deepcopy(prop)
        prop.inds = nm_indices[index_tracker: index_tracker + prop.dord]
        index_tracker += prop.dord
        upd_props.append(prop)
    return PropsCollection(props=upd_props)

def identify_unique_avrg_tensors(avrg_expressions: list[PropsCollection]) -> list[PropsCollection]:
    """
        
    """
    return set(make_unique_avrg_tensors_mapping(avrg_expressions).values())

'''
#####################################################################################
# SUPERSEDED - kept only for side-by-side comparison, and SHADOWED by the definition
# of the same name below (Python keeps the last one, so this body never runs).
# Delete once compared.
#####################################################################################
def get_ind_tuple_from_base(expr: PropsCollection, base_expr: PropsCollection, index_dict: dict):
    """Map expr to indices according to base expression's unique symbols.

    Both branches return an index tuple into the base's tensor, whose axes are the base's unique
    labels in alphabetical order. They split on whether the base has repeated indices:

        len(base_unique) == len(expr_inds)   the base's indices are all distinct
        len(base_unique) <  len(expr_inds)   the base has repeats, so fewer axes than index slots

    (Within one numerator motif every expression has the same number of index slots, so the
    comparison is really just asking "is the base all-distinct?".)

    -- the '==' branch is correct --

    When the base is all-distinct its labels are already 'a', 'b', 'c', ... in slot order, so
    tensor axis k corresponds to slot k, and indexing by expr's own labels taken slot by slot is
    exactly right:

        expr   polgrad['b'] * dipgrad['a'] * dipgrad['c']      base   ['a', 'b', 'c']
        ->  (index_dict['b'], index_dict['a'], index_dict['c'])       correct

    Expressions with repeated labels are handled by the same branch, and land on a diagonal of
    the base's tensor:

        expr   polgrad['b'] * dipgrad['a'] * dipgrad['b']
        ->  (index_dict['b'], index_dict['a'], index_dict['b'])       correct

    -- the '<' branch is WRONG, and is why this version was replaced --

    First, to be clear about what is NOT the problem: a base whose letters are a permutation of
    the expression's is perfectly fine, and is exactly what the deduplication is for. The base is
    only a representative expression. The '==' branch above meets relabelled bases too and still
    gets the right answer. Nothing is wrong until an index is built from the base.

    The defect is in how this branch builds that index. It walks the BASE's labels and looks them
    up in index_dict, which is keyed by THIS EXPRESSION's labels - silently assuming the two use
    the same letter for the same slot. Where they happen to, the answer is right. Where they do
    not, it is wrong and nothing complains:

        expr   polgrad['b'] * dipgrad['a'] * diphess['a','b']
        base   polgrad['a'] * dipgrad['b'] * diphess['a','b']      'a' and 'b' swapped

        base_unique is ['a', 'b'], so this returns (index_dict['a'], index_dict['b'])
        but the base's first axis is the polgrad slot, which in expr holds 'b'
        -> the wanted value is at (index_dict['b'], index_dict['a']): the TRANSPOSED element

    No error is raised; a wrong number is returned. With index_dict {'a': 0, 'b': 1} it reads
    tensor[0, 1] where the value lives at tensor[1, 0].

    The replacement below fixes this by rebuilding the base-label -> expr-label correspondence
    instead of assuming it, which also removes the need for the two branches.
    """
    base_unique = sorted(list(set(base_expr.get_mode_indices())))
    expr_inds = expr.get_mode_indices()

    if len(base_unique) < len(expr_inds):
        # walk through only base_unique ind labels
        # BUG: base_unique are the BASE's letters, but index_dict is keyed by EXPR's letters
        return tuple(index_dict[sym] for sym in base_unique)
    elif len(base_unique) == len(expr_inds):
        # walk through all expr_inds labels, there are repeated labels
        # correct: expr's own labels, slot by slot, against an all-distinct base
        return tuple(index_dict[sym] for sym in expr_inds)
    else:
        # this should not be possible in the worflow
        raise ValueError('This base_expr cannot be a base expression for this expr')
'''

def get_ind_tuple_from_base(expr: PropsCollection, base_expr: PropsCollection, index_dict: dict):
    """
    Work out WHICH ELEMENT of the base expression's precalculated tensor holds the value of
    'expr' for this choice of normal mode indices.

    -- why there is a base at all --

    An averaged expression is a function of "which normal mode goes into which property slot".
    calculate_avrg_tensor evaluates that function at every combination of modes and stores the
    results in an array with one axis per distinct mode index. Expressions that differ only in
    the NAMES of their indices are the same function, so one array can serve all of them:
    make_unique_avrg_tensors_mapping picks one of them (the "base") and maps the rest onto it.

    Reusing the array means reading the right element out of it, which takes two steps:

        hop 1:  base's index label   ->  this expression's index label
        hop 2:  this expression's index label  ->  normal mode number     (this is index_dict)

    -- why hop 1 is needed --

    The base is built from canonical letters ('a', 'b', ...), which are in general a PERMUTATION
    of the labels this expression uses:

        expr   polgrad['b'] * dipgrad['a'] * diphess['a','b']
        base   polgrad['a'] * dipgrad['b'] * diphess['a','b']        'a' and 'b' are swapped

    so with index_dict {'a': 0, 'b': 1} the wanted value sits at tensor[1, 0] and NOT at
    tensor[0, 1]. Skipping hop 1 returns the transposed element.

    -- why the two cannot simply be lined up as they stand --

        base:  a  b  a  b
        expr:  b  a  a  b       base 'a' -> 'b', then base 'a' -> 'a': contradiction

    The base's letters were laid down against the repetition codes AFTER those codes had been
    sorted within each property (see nm_indices_repetition_reduce_deriv_symmetry; that sort is
    legitimate because a geometric derivative is symmetric in its own mode indices). So this
    expression's labels must be put through the same sort before they will line up:

        expr labels           ['b', 'a', 'a', 'b']
        repetition codes      ( 1 ,  2 ,  2 ,  1 )
        split per property    [['b'], ['a'], ['a', 'b']]    codes [[1], [2], [2, 1]]
        sort each by code     [['b'], ['a'], ['b', 'a']]    ->  ['b', 'a', 'b', 'a']

        base:  a  b  a  b
        expr:  b  a  b  a       base 'a' -> 'b', base 'b' -> 'a', and the repeats agree

    NOTE: if nm_indices_repetition_reduce_deriv_symmetry ever changes how it orders indices,
    the sort below has to change with it.

    Returns the index tuple into the base's tensor, whose axes are the base's unique labels in
    alphabetical order (the order calculate_avrg_tensor built them in).
    """
    # ---- hop 1: rebuild which of the base's labels stands for which of this expression's ----

    # how many mode indices each property owns, e.g. [1, 1, 2] for polgrad * dipgrad * diphess
    template = expr.get_mode_indices_group_template()

    # chop the codes and the labels into the same per-property groups so they stay aligned
    grouped_codes = group_nm_indices(nm_indices_repetition_encoding(expr.get_mode_indices()), template)
    grouped_labels = group_nm_indices(expr.get_mode_indices(), template)

    # sort each property's slots by repetition code and move its labels the same way, which
    # reproduces the ordering the base's letters were assigned in
    expr_inds_sorted = []
    for codes, labels in zip(grouped_codes, grouped_labels):
        order = sorted(range(len(codes)), key=lambda i: codes[i])
        expr_inds_sorted.extend(labels[i] for i in order)

    base_inds = base_expr.get_mode_indices()

    # Shape guard only - it protects the zip below, which would otherwise truncate silently.
    # Equal slot counts do NOT establish that base_expr can represent expr: two different
    # numerator motifs can have the same total number of slots (polhess * dip * dip and
    # polgrad * dip * diphess both have 4). What actually makes a base valid is sharing the
    # expression's numerator motif, and that holds by construction - the base is built from that
    # motif in reconstruct_unique_avrg_expression - so it is not re-tested here, where this
    # function runs inside the per-index evaluation loop.
    if len(base_inds) != len(expr_inds_sorted):
        raise ValueError(
            f'base_expr has {len(base_inds)} mode index slots but expr has '
            f'{len(expr_inds_sorted)}: they cannot belong to the same numerator motif')

    # The two now line up slot by slot. This map only has to be a well-defined FUNCTION - each
    # base label with a single value - it does NOT have to be injective:
    #
    #   several base labels -> the same expr label     FINE, and expected. The expression uses
    #       fewer distinct modes than the base has axes, so it reads a diagonal of the base's
    #       tensor. Base ['a','b','c'] serving expr ['a','c','c'] gives {a: a, b: c, c: c},
    #       hence the index (index_dict['a'], index_dict['c'], index_dict['c']) - a rank 3
    #       tensor serving a rank 2 expression.
    #
    #   one base label -> several expr labels          IMPOSSIBLE. The expression needs more
    #       distinct modes than the base has axes, so this base cannot serve it at all.
    base_to_expr = {}
    for base_sym, expr_sym in zip(base_inds, expr_inds_sorted):
        if base_to_expr.setdefault(base_sym, expr_sym) != expr_sym:
            raise ValueError(
                f'This base_expr cannot be a base expression for this expr: base index '
                f'{base_sym!r} maps to both {base_to_expr[base_sym]!r} and {expr_sym!r}')

    # ---- hop 2: expression labels -> mode numbers, in the tensor's axis order ----
    return tuple(index_dict[base_to_expr[sym]] for sym in sorted(set(base_inds)))

