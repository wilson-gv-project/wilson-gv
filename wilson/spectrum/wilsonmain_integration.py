from .spectrum2D import Spectrum2D, numcombperm

def parse_wmain2wpart2(term):

    return


def get_spectrum2D(system, experiment, terms, props, spec_eval_setup, vib_ana_setup, with_diagnostics=False):

    # Check if the requested spectrum can be evaluated by this function

    if not(experiment.dim == 2):
        raise AssertionError('This evaluator is made solely for 2D spectrum experiments')

    if not(len(spec_eval_setup.axes.a) == 2):
        raise AssertionError('This evaluator is made solely for 2D spectrum evaluations')

    # CONTINUE HERE: Use and rework existing functionality as necessary

    # - make spectrum2D instance
    # - Do first without mask, later bring in mask functionality

    # - Set up conditions
    # - Run launch sequence (or wilson main convention fn equivalent)
    # - Run for amplitudes (or wilson main convention fn equivalent)
    # - Square for intensity
    # - Write diagnostics handling

    # Get from wmain convention classes
    # dynamic_range_n = conditions.dynamic_range_n
    # omega1, omega2 = conditions.omega1, conditions.omega2

    # Here also get from wmain convention but now props
    # parser.load()
    # parsed_data = parser.parse(linear_molecule=False)

    # spectrumObj = Spectrum2D(omega1, omega2)
    # Write wmain convention version
    # dict0 = spectrumObj.launch_sequence1_wmain(parsed_data, conditions,
    #                                     print_level=0)

    mask = None

    if compute_intensity:
        sec_hypol_dataALL_ref = spectrumObj.intensity_generic(selectionCond=mask)
        nan_mask = np.isnan(sec_hypol_dataALL_ref)
        sec_hypol_dataALL_ref[nan_mask] = 0 + 0j


    if not(with_diagnostics):
        return []

    else:
        return [], []

def spectrum2D_with_diagnostics(system, experiment, terms, props, spec_eval_setup, vib_ana_setup):

    return get_spectrum2D(system, experiment, terms, props, spec_eval_setup, vib_ana_setup, with_diagnostics=True)


def spectrum2D(system, experiment, terms, props, spec_eval_setup, vib_ana_setup):

    return get_spectrum2D(system, experiment, terms, props, spec_eval_setup, vib_ana_setup, with_diagnostics=True)



