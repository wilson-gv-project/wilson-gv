import numpy as np


def anharm_corr_energiesVPT2(harmonic_energies, cubic_forcefield, quartic_forcefield,
                                   rotational_constant, coriolis_constant, anharmonic_type):
    """
    Takes in cm-1 unit for all the arguments:
        harmonic_energies, cubic_forcefield, quartic_forcefield, rotational_constant, coriolis_constant(unit?)
        (nmodes,);   (nmodes, nmodes, nmodes);   (nmodes, nmodes, nmodes, nmodes);   [x,y,z];   (nmodes, nmodes)

    anharmonic_type options:
            'Anharmonic: VPT2'                                                              - don't do_res, don't do_var
            'Anharmonic: DVPT2' = 'Anharmonic: Freq DVPT2, Int VPT2'
                    = 'Anharmonic: DVPT2, w/ 1-1 checks'                                    - do_res, don't do_var
            'Anharmonic: Freq GVPT2, Int DVPT2' = 'Anharmonic: Freq GVPT2, Int DVPT2, w/ 1-1 checks'
                    = 'Anharmonic: Freq GVPT2, Int DVPT2, w/ 1-1 checks and forced removal' - do_res, do_var

    returns:
        fundamental, overtones, combotones, over3q, combo3q
    """
    if anharmonic_type == 'Anharmonic: Freq GVPT2, Int DVPT2':
        do_variational_correction = True
        do_resonance_checks = True
    elif anharmonic_type == 'Anharmonic: VPT2':
        do_resonance_checks = False
        do_variational_correction = False
    elif anharmonic_type == 'Anharmonic: DVPT2':
        do_resonance_checks = True
        do_variational_correction = False
    elif anharmonic_type == 'Anharmonic: Freq DVPT2, Int VPT2':
        do_resonance_checks = True
        do_variational_correction = False
    elif anharmonic_type == 'Anharmonic: DVPT2, w/ 1-1 checks':
        do_resonance_checks = True
        do_variational_correction = False
    elif anharmonic_type == 'Anharmonic: Freq GVPT2, Int DVPT2, w/ 1-1 checks':
        do_resonance_checks = True
        do_variational_correction = True
    elif anharmonic_type == 'Anharmonic: Freq GVPT2, Int DVPT2, w/ 1-1 checks and forced removal':
        do_resonance_checks = True
        do_variational_correction = True
    else:
        print('\n')
        print('Something strange has happened in anharm_corrected_vibrational_energies')
        print('Anharmonic is called, but which type isn/t specified')
        exit()

    fundamental = np.zeros((len(harmonic_energies)))
    overtones = np.zeros((len(harmonic_energies)))
    combotones = np.zeros((len(harmonic_energies), len(harmonic_energies)))
    over3q = np.zeros((len(harmonic_energies)))
    combo3q = np.zeros((len(harmonic_energies), len(harmonic_energies), len(harmonic_energies)))

    X, fermi_resonance, X_cubic, X_quartic = get_XVPT2(harmonic_energies, cubic_forcefield, quartic_forcefield,
                                                       rotational_constant, coriolis_constant, do_resonance_checks)

    if fermi_resonance: # if not an empty list
        print('Fermi resonances_args' , fermi_resonance)
        for a in range(len(fermi_resonance)):
            i = fermi_resonance[a][0]
            j = fermi_resonance[a][1]
            k = fermi_resonance[a][2]

            # print(i, j, k, harmonic_energies[i], harmonic_energies[j], harmonic_energies[k],
            #       'type: ', fermi_resonance[a][3],
            #       np.multiply(cubic_forcefield[i][j][k], 0.01 / (plancs_constant * speed_of_light)))

    funds_corrections = np.zeros((len(harmonic_energies)))
    for i in range(len(harmonic_energies)):
        fundamental[i] += harmonic_energies[i] + 2 * X[i][i]

        fscr = 0
        for j in range(len(harmonic_energies)):
            if j != i:
                fscr += 0.5 * X[i][j]

        fundamental[i] += fscr
        funds_corrections[i] += 2 * X[i][i] + fscr

    overtones_corrections = np.zeros((len(harmonic_energies)))
    combotones_corrections = np.zeros((len(harmonic_energies), len(harmonic_energies)))
    over3q_corrections = np.zeros((len(harmonic_energies)))
    combo3q_corrections = np.zeros((len(harmonic_energies), len(harmonic_energies), len(harmonic_energies)))

    for i in range(len(harmonic_energies)):
        overtones[i] += 2 * fundamental[i] + 2 * X[i][i]
        overtones_corrections[i] += 2 * X[i][i]

        over3q[i] += 3 * fundamental[i] + 6 * X[i][i]
        over3q_corrections[i] += 6 * X[i][i]

        for j in range(len(harmonic_energies)):
            if i == j:
                for k in range(len(harmonic_energies)):
                    if k != i:
                        # continue
                        combo3q[i][i][k] += 2 * fundamental[i] + 2 * X[i][i] + fundamental[k] + 2 * X[i][k]
                        combo3q_corrections[i][i][k] += 2 * X[i][i] + 2 * X[i][k]

            else:
                combotones[i][j] += fundamental[i] + fundamental[j] + X[i][j]
                combotones_corrections[i][j] += X[i][j]

                for k in range(len(harmonic_energies)):
                    if k == i or k == j:
                        continue
                    combo3q[i][j][k] += fundamental[i] + fundamental[j] + fundamental[k] + X[i][j] + X[i][k] + X[j][k]
                    combo3q_corrections[i][j][k] += X[i][j] + X[i][k] + X[j][k]

    if do_variational_correction:
        # selectedFR = range((len(fermi_resonance)))
        selectedFR = [2]
        adjusted_fundamental, adjusted_overtones, adjusted_combotones = \
            adjust_for_fermi_resonance(fundamental, overtones, combotones, over3q, combo3q, cubic_forcefield,
                                       [fermi_resonance[i] for i in selectedFR])
        # anharmonic_energies = anharmonicProperty(harmonic_energies, adjusted_fundamental,
        #                                          adjusted_overtones, adjusted_combotones)
        return adjusted_fundamental, adjusted_overtones, adjusted_combotones, over3q, combo3q

    else:
        return fundamental, overtones, combotones, over3q, combo3q


def get_XVPT2(harmonic_energies, cubic_forcefield, quartic_forcefield,
          rotational_constant, coriolis_constant, do_resonance_checks):

    fermi_resonance = []
    X = np.zeros((len(harmonic_energies), len(harmonic_energies)))
    X_cubic = np.zeros((len(harmonic_energies), len(harmonic_energies)))
    X_quartic = np.zeros((len(harmonic_energies), len(harmonic_energies)))
    X_coriolis = np.zeros((len(harmonic_energies), len(harmonic_energies)))

    for i in range(len(harmonic_energies)):
        vi = harmonic_energies[i]
        X[i][i] = quartic_forcefield[i][i][i][i]/16.0
        X_quartic[i][i] = quartic_forcefield[i][i][i][i]/16.0

        rhs = 0

        for k in range(len(harmonic_energies)):
            vk = harmonic_energies[k]
            kiik = cubic_forcefield[i][i][k]

            tmp1 = 4.0/vk
            tmp2 = 1/(2.0*vi + vk)
            if not is_fermi_resonance(2 * vi - vk, kiik, True) or not do_resonance_checks:
                tmp3 = 1/(2.0*vi - vk)
            else:
                fermi_resonance = add_fermi_resonance(fermi_resonance, [k, i, i, True])
                tmp3 = 0.0

            rhs += (kiik**2/32.0)*(tmp1 + tmp2 - tmp3)

        X[i][i] = X[i][i] - rhs

        for j in range(i):
            vj = harmonic_energies[j]
            X[i][j] = quartic_forcefield[i][i][j][j]/4.0
            X_quartic[i][j] = quartic_forcefield[i][i][j][j]/4.0

            A = 0
            for k in range(len(harmonic_energies)):
                A += cubic_forcefield[i][i][k]*cubic_forcefield[j][j][k]/(4.0*harmonic_energies[k])

            B = 0
            for k in range(len(harmonic_energies)):
                vk = harmonic_energies[k]
                kijk = cubic_forcefield[i][j][k]

                tmp1 = 1/(vi + vj + vk)
                if (not is_fermi_resonance(-vi + vj + vk, kijk, k == j)) or (not do_resonance_checks):
                    # perturb if no fermi resonance or dont do resonance checks
                    tmp2 = 1/(-vi + vj + vk)
                else:
                    # deperturbing otherwise - when resonance and do checks
                    fermi_resonance = add_fermi_resonance(fermi_resonance, [i, j, k, k == j])
                    tmp2 = 0.0

                if (not is_fermi_resonance(vi - vj + vk, kijk, k == i)) or (not do_resonance_checks):
                    # perturb if no fermi resonance or dont do resonance checks
                    tmp3 = 1/(vi -vj + vk)
                else:
                    # deperturbing otherwise - when resonance and do checks
                    fermi_resonance = add_fermi_resonance(fermi_resonance, [j, k, i, k == i])
                    tmp3 = 0.0

                if not (is_fermi_resonance(vi + vj - vk, kijk, False)) or (not do_resonance_checks):
                    # perturb if no fermi resonance or dont do resonance checks
                    tmp4 = 1/(vi + vj - vk)
                else:
                    # deperturbing otherwise - when resonance and do checks
                    fermi_resonance = add_fermi_resonance(fermi_resonance, [k, i, j, False])
                    tmp4 = 0.0

                B += kijk**2/8.0*(tmp1 + tmp2 + tmp3 - tmp4)

            C = 0
            if not type(coriolis_constant) == str:
                for k in range(len(rotational_constant)):
                    C += rotational_constant[k]*coriolis_constant[k][i][j]**2*\
                        (harmonic_energies[i]/harmonic_energies[j] +
                         harmonic_energies[j]/harmonic_energies[i])

            X[i][j] = X[i][j] - A - B + C
            X[j][i] = np.copy(X[i][j])

            X_coriolis[i][j] = C
            X_coriolis[j][i] = np.copy(X_coriolis[i][j])
            X_cubic[i][j] = B
            X_cubic[j][i] = np.copy(X_cubic[i][j])

    fermi_resonance = sorted(fermi_resonance)

    return X, fermi_resonance, X_cubic, X_quartic

fermi_threshold  = 200.0
martin_threshold = 1.0

def is_fermi_resonance(delta, cubic_force_ijk, i_is_j):
    fermi = False

    if abs(delta) <= fermi_threshold: # in FR should be less than 200 cm-1
        if i_is_j:
            martin_parameter = cubic_force_ijk**4/(256.0*delta**3)
            if abs(martin_parameter) >= martin_threshold: # in FR should be greater than 1 cm-1
                fermi = True
                # print(abs(delta), fermi_threshold)
                # print(abs(martin_parameter), martin_threshold)
            else:
                fermi = False
        else:
            martin_parameter = cubic_force_ijk**4/(64.0*delta**3)
            if abs(martin_parameter) >= martin_threshold:
                fermi = True
                # print(abs(delta), fermi_threshold)
                # print(abs(martin_parameter), martin_threshold)
            else:
                fermi = False

    return fermi


def add_fermi_resonance(total_list, new_element):

    i = new_element[0]
    j = new_element[1]
    k = new_element[2]
    l = new_element[3]

    j, k = sorted([j, k])
    new_element[0] = i
    new_element[1] = j
    new_element[2] = k
    new_element[3] = l

    if not new_element in total_list:
        total_list.append(new_element)

    return total_list


def adjust_for_fermi_resonance(fundamental, overtones, combotones, over3q, combo3q, cubic_forcefield, fermi_resonance):

    num_modes = len(fundamental)
    red_num_combotones = (num_modes**2 - num_modes)//2

    num_frequencies = num_modes*2 + red_num_combotones

    adjusted_frequencies = np.zeros((num_frequencies))
    V = np.zeros((num_frequencies, num_frequencies))

    k = 2*num_modes
    for i in range(num_modes):

        V[i][i] = fundamental[i]
        V[i + num_modes][i + num_modes] = overtones[i]

        for j in range(i):
            V[k][k] = combotones[i][j]

            k = k + 1

    for a in range(len(fermi_resonance)):
        i = fermi_resonance[a][0]
        j = fermi_resonance[a][1]
        k = fermi_resonance[a][2]
        fermi_type = fermi_resonance[a][3]

        if fermi_type:
            V[i][num_modes + j] = cubic_forcefield[j][k][i]/4.0
            V[num_modes + j][i] = V[i][num_modes + j]
        else:
            V[i][x_matrix_position(j, k, num_modes)] = cubic_forcefield[j][k][i]/np.sqrt(8.0)
            V[x_matrix_position(j, k, num_modes)][i] = V[i][x_matrix_position(j, k, num_modes)]

    eigenvalue, eigenvector = np.linalg.eig(V)

    k = 0
    for i in range(num_frequencies):
        orig_character = 0.0

        for j in range(num_frequencies):
            if abs(eigenvector[i][j]) > orig_character:
                orig_character = abs(eigenvector[i][j])
                adjusted_frequencies[k] = eigenvalue[j]

                i_index = i
                j_index = j

        # A little uncertain of this one...
        for j in range(num_frequencies):
            eigenvector[j][j_index] = 0.0

        k = k + 1

    adjusted_fundamental = np.zeros((num_modes))
    adjusted_overtones = np.zeros((num_modes))
    adjusted_combotones = np.zeros((num_modes, num_modes))

    k = 2*num_modes
    for i in range(num_modes):
        adjusted_fundamental[i] = adjusted_frequencies[i]
        adjusted_overtones[i] = adjusted_frequencies[i + num_modes]

        for j in range(i):
            adjusted_combotones[i][j] = adjusted_frequencies[k]
            adjusted_combotones[j][i] = adjusted_combotones[i][j]

            k = k + 1

    return adjusted_fundamental, adjusted_overtones, adjusted_combotones


def x_matrix_position(a, b, n):
    pos = a
    for i in range(b):
        pos += i

    return pos + 2*n