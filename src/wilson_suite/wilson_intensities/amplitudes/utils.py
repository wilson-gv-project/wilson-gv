

def generate_index_choices_general(indlabels_in_motif, labels):
    """
    indlabels_in_motif - collection of symbolic label indices
    labels - collection of numerical or string values
    """
    import itertools
    return [dict(zip(indlabels_in_motif, combo)) for combo in itertools.product(labels, repeat=len(indlabels_in_motif))]