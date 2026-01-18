import numpy as np

def test_excluded_mode_has_zeros():
    print()
    from ...anharmonic_treatment.vpt2 import get_X
    list2exclude = [1]

    harmonic_energies = {0: 13., 1: 25., 2: 32.}
    original_len_ene = len(harmonic_energies)

    harmonic_energies = {k: v for k, v in harmonic_energies.items() if k not in list2exclude}

    cubic_forcefield = np.linspace(1, 27, 27).reshape(3,3,3)
    quartic_forcefield = np.linspace(11, 91, 81).reshape(3,3,3,3)
    rot_const = []


    X, X_cubic, X_quartic, _ = get_X(
        harmonic_energies,
        cubic_forcefield,
        quartic_forcefield,
        rot_const, None, False, [], original_len_ene
    )

    assert X[1, 1] == 0.0
    assert X[0, 1] == 0.0
    assert X[1, 0] == 0.0
    assert X[1, 2] == 0.0
    assert X[2, 1] == 0.0
    assert X_cubic[1, 1] == 0.0
    assert X_cubic[0, 1] == 0.0
    assert X_cubic[2, 1] == 0.0
    assert X_quartic[1, 1] == 0.0
    assert X_quartic[0, 1] == 0.0
    assert X_quartic[2, 1] == 0.0

    print(X)
    print(X_cubic)
    print(X_quartic)

def test_anharm_corr_energies_results():
    pass

def test_anharm_corr_energies_emptyinput():
    pass

def test_anharm_corr_energies_excludedmodes():
    from ...anharmonic_treatment.vpt2 import anharm_corr_energies
    harmonic_energies = {0: 13., 1: 25., 2: 32.}
    cubic_forcefield = np.linspace(1, 28, 27).reshape(3,3,3)
    quartic_forcefield = np.linspace(11, 92, 81).reshape(3,3,3,3)
    rot_const = []

    anharm_corr_energies(harmonic_energies=harmonic_energies,
                         cubic_forcefield=cubic_forcefield,
                         quartic_forcefield=quartic_forcefield,
                         rotational_constant=rot_const,
                         coriolis_constant=None,
                         anharmonic_type='GVPT2',
                         list2exclude=[])

    anharm_corr_energies(harmonic_energies=harmonic_energies,
                         cubic_forcefield=cubic_forcefield,
                         quartic_forcefield=quartic_forcefield,
                         rotational_constant=rot_const,
                         coriolis_constant=None,
                         anharmonic_type='GVPT2',
                         list2exclude=[1])

def test_anharm_corr_energies_FR():
    pass

def test_identify_fermi():
    pass

def test_identify_fermi_noFR():
    pass

def test_identify_fermi_c4():
    pass

def test_identify_fermi_c4_noFR():
    pass

