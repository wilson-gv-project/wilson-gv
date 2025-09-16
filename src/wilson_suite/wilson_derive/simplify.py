import copy

from .abstractions import VibPerturbedTerm

def terms_simplify(terms_in: list[VibPerturbedTerm], nm_inds: list):
    """
    Simplify terms: Look for (up to coefficient) algebraically identical terms and add them together
    according to their coefficients

    terms_in: list of VibPerturbedTerm instances: The terms to be simplified
    nm_inds: List of characters: Canonical normal mode index symbols
    """

    # Term registry
    t_reg = {}

    for i in terms_in:

        this_h = i.h(also_sort=True, nm_inds=nm_inds)

        if this_h in t_reg:
            t_reg[this_h].coeff += i.coeff

            if t_reg[this_h].coeff == 0:
                del t_reg[this_h]

        else:
            t_reg[this_h] = copy.deepcopy(i)

    return t_reg