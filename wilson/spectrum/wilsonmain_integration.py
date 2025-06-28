from .spectrum2D import Spectrum2D, numcombperm

def parse_wmain2wpart2(term):

    return


def get_spectrum2D(system, experiment, terms, props, spec_eval_setup, vib_ana, with_diagnostics=False):

    # Check if the requested spectrum can be evaluated by this function
    if not(experiment.dim == 2):
        raise AssertionError('This evaluator is currently only for 2D spectrum experiments')

    if not(len(spec_eval_setup.axes.axes) == 2):
        raise AssertionError('This evaluator is currently only for 2D spectrum evaluations')

    # TODO: Check for supported orders of anharmonicity

    # Make spectrum2D instance
    spectrumObj = Spectrum2D(axes = spec_eval_setup.axes)

    # Take "pan-spectrum" settings (global damping if using, diag extra, etc.), extract from spec_eval_setup and make
    # as input to precalc/calc routines
    spec_settings = {}

    if with_diagnostics:
        diagnostics = {}
    else:
        diagnostics = None

    # Set up and precalculate
    spectrumObj.launch_sequence_wmain(terms, props, vib_ana, spec_settings, diagnostics=diagnostics)

    # Run for amplitudes (or wilson main convention fn equivalent) and square for intensity
    # TODO: Do first without mask, later bring in mask functionality
    # Amplitudes here considered as diagnostics
    intensities = spectrumObj.intensity_generic(diagnostics=diagnostics)
    nan_mask = np.isnan(intensities)
    intensities[nan_mask] = 0 + 0j

    if not(with_diagnostics):
        return intensities

    else:
        return intensities, diagnostics

def spectrum2D_with_diagnostics(system, experiment, terms, props, spec_eval_setup, vib_ana_setup):

    return get_spectrum2D(system, experiment, terms, props, spec_eval_setup, vib_ana_setup, with_diagnostics=True)


def spectrum2D(system, experiment, terms, props, spec_eval_setup, vib_ana_setup):

    return get_spectrum2D(system, experiment, terms, props, spec_eval_setup, vib_ana_setup, with_diagnostics=False)



