from spectrum2D import Spectrum2D, numcombperm

def get_spectrum2D(system, experiment, terms, props, spec_eval_setup, vib_ana_setup, with_diagnostics=False):

    if not(with_diagnostics):
        return []

    else:
        return [], []

def spectrum2D_with_diagnostics(system, experiment, terms, props, spec_eval_setup, vib_ana_setup):

    return get_spectrum2D(system, experiment, terms, props, spec_eval_setup, vib_ana_setup, with_diagnostics=True)


def spectrum2D(system, experiment, terms, props, spec_eval_setup, vib_ana_setup):

    return get_spectrum2D(system, experiment, terms, props, spec_eval_setup, vib_ana_setup, with_diagnostics=True)



