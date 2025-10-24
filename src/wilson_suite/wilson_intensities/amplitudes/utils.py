from wilson_suite.wilson_analysis.render.render import get_axes_in_resmotif


def initialize_resonance_dict(motif):
    """
    Initialize a dictionary with axes in the motif as keys; values are None
    """
    axes_in_motif = sorted(get_axes_in_resmotif(motif))
    return {ax: None for ax in axes_in_motif}


def generate_index_choices_general(indlabels_in_motif, labels):
    """
    indlabels_in_motif - collection of symbolic label indices
    labels - collection of numerical or string values
    """
    import itertools
    return [dict(zip(indlabels_in_motif, combo)) for combo in itertools.product(labels, repeat=len(indlabels_in_motif))]