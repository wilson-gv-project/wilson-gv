import copy

def terms_simplify(terms_in, nm_inds):

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