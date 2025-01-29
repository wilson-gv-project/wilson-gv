import numpy as np

def anharm_corr_energiesVPT2(harmonic_energies, cubic_forcefield, quartic_forcefield,
                                   rotational_constant, coriolis_constant, anharmonic_type):
    """
    Takes in cm-1 unit for all the arguments:
        harmonic_energies, cubic_forcefield, quartic_forcefield, rotational_constant, coriolis_constant(unit?)
        (nmodes,);   (nmodes, nmodes, nmodes);   (nmodes, nmodes, nmodes, nmodes);   [x,y,z];   (nmodes, nmodes)

    anharmonic_type options:
            'VPT2'                        - don't do_res, don't do_var
            'DVPT2'                       - do_res, don't do_var
            'GVPT2'                       - do_res, do_var
    returns:
        fundamental, overtones, combotones, over3q, combo3q
    """
    if anharmonic_type == 'GVPT2':
        do_variational_correction = True
        do_resonance_checks = True
    elif anharmonic_type == 'VPT2':
        do_resonance_checks = False
        do_variational_correction = False
    elif anharmonic_type == 'DVPT2':
        do_resonance_checks = True
        do_variational_correction = False
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

    fermi_resonance = identify_fermi(harmonic_energies, cubic_forcefield, do_resonance_checks)

    X, X_cubic, X_quartic, X_coriolis = get_XVPT2(harmonic_energies, cubic_forcefield, quartic_forcefield,
                                                  rotational_constant, coriolis_constant, do_resonance_checks,
                                                  fermi_resonance)

    # np.set_printoptions(suppress=True, precision=5)
    # print('X\n', X, '\n')
    # print('X_cubic\n', X_cubic, '\n')
    # print('X_quartic\n', X_quartic, '\n')
    # print('X_coriolis\n', X_coriolis, '\n')

    if fermi_resonance: # if not an empty list
        print(f'Fermi identified - {len(fermi_resonance)}:' , fermi_resonance)
        # for a in range(len(fermi_resonance)):
        #     i = fermi_resonance[a][0]
        #     j = fermi_resonance[a][1]
        #     k = fermi_resonance[a][2]

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
        selectedFR = range((len(fermi_resonance)))
        adjusted_fundamental, adjusted_overtones, adjusted_combotones = \
            adjust_for_fermi_resonance(fundamental, overtones, combotones, over3q, combo3q, cubic_forcefield,
                                       [fermi_resonance[i] for i in selectedFR])
        return adjusted_fundamental, adjusted_overtones, adjusted_combotones, over3q, combo3q

    else:
        return fundamental, overtones, combotones, over3q, combo3q


def identify_fermi(harmonic_energies, cubic_forcefield, do_resonance_checks):

    fermi_resonance = []
    for i in range(len(harmonic_energies)):
        vi = harmonic_energies[i]

        for k in range(len(harmonic_energies)):
            vk = harmonic_energies[k]
            kiik = cubic_forcefield[i][i][k]

            isfermi = is_fermi_resonance(2 * vi - vk, kiik, True)
            if isfermi and do_resonance_checks:
                fermi_resonance = add_fermi_resonance(fermi_resonance, [k, i, i, True])

        for j in range(i):
            vj = harmonic_energies[j]

            for k in range(len(harmonic_energies)):
                vk = harmonic_energies[k]
                kijk = cubic_forcefield[i][j][k]

                if is_fermi_resonance(-vi + vj + vk, kijk, k == j) and do_resonance_checks:
                    fermi_resonance = add_fermi_resonance(fermi_resonance, [i, *sorted([j, k]), k == j])
                if is_fermi_resonance(vi - vj + vk, kijk, k == i) and do_resonance_checks:
                    fermi_resonance = add_fermi_resonance(fermi_resonance, [j, *sorted([k, i]), k == i])
                if is_fermi_resonance(vi + vj - vk, kijk, i == j) and do_resonance_checks:
                    fermi_resonance = add_fermi_resonance(fermi_resonance, [k, *sorted([i, j]), i == j])

    return fermi_resonance

# def get_XVPT2(harmonic_energies, cubic_forcefield, quartic_forcefield,
#           rotational_constant, coriolis_constant, do_resonance_checks):
#
#     fermi_resonance = []
#     X = np.zeros((len(harmonic_energies), len(harmonic_energies)))
#     X_cubic = np.zeros((len(harmonic_energies), len(harmonic_energies)))
#     X_quartic = np.zeros((len(harmonic_energies), len(harmonic_energies)))
#     X_coriolis = np.zeros((len(harmonic_energies), len(harmonic_energies)))
#
#     for i in range(len(harmonic_energies)):
#         vi = harmonic_energies[i]
#         X[i][i] = quartic_forcefield[i][i][i][i]/16.0
#         X_quartic[i][i] = quartic_forcefield[i][i][i][i]/16.0
#
#         rhs = 0
#
#         for k in range(len(harmonic_energies)):
#             vk = harmonic_energies[k]
#             kiik = cubic_forcefield[i][i][k]
#
#             tmp1 = 4.0/vk
#             tmp2 = 1/(2.0*vi + vk)
#             isfermi = is_fermi_resonance(2 * vi - vk, kiik, True)
#             print(i, i, k, isfermi)
#             if not isfermi or not do_resonance_checks:
#                 tmp3 = 1/(2.0*vi - vk)
#             else:
#                 fermi_resonance = add_fermi_resonance(fermi_resonance, [k, i, i, True])
#                 tmp3 = 0.0
#                 if i == 8 and k == 5:
#                     tmp3 = 1 / (2.0 * vi - vk)
#
#             print('kiik, tmp1, tmp2, tmp3, +=', (i, k), kiik, tmp1, tmp2, tmp3, (kiik**2/32.0)*(tmp1 + tmp2 - tmp3))
#
#             rhs += (kiik**2/32.0)*(tmp1 + tmp2 - tmp3)
#             # if i == 8:
#             #     print('>>>>>>>>> cubic+= for', i, k)
#             #     print('kiik, tmp1, tmp2, tmp3, +=', (i, k), kiik, tmp1, tmp2, tmp3, (kiik**2/32.0)*(tmp1 + tmp2 - tmp3))
#
#         # X[i][i] = X[i][i] - rhs
#         X[i][i] += - rhs
#         X_cubic[i][i] = - rhs
#
#         for j in range(i):
#             vj = harmonic_energies[j]
#             X[i][j] = quartic_forcefield[i][i][j][j]/4.0
#             X_quartic[i][j] = quartic_forcefield[i][i][j][j]/4.0
#
#             A = 0
#             for k in range(len(harmonic_energies)):
#                 A += cubic_forcefield[i][i][k]*cubic_forcefield[j][j][k]/(4.0*harmonic_energies[k])
#                 if i == 8 and j == 5:
#                     print('\n>>>>>>>>> A+= for', i, j, k)
#                     print('Fiik, Fjjk, 4*vk, vk', cubic_forcefield[i][i][k],cubic_forcefield[j][j][k],(4.0*harmonic_energies[k]), harmonic_energies[k])
#                     print('A+=', cubic_forcefield[i][i][k]*cubic_forcefield[j][j][k]/(4.0*harmonic_energies[k]))
#
#             B = 0
#             for k in range(len(harmonic_energies)):
#                 vk = harmonic_energies[k]
#                 kijk = cubic_forcefield[i][j][k]
#
#                 tmp1 = 1/(vi + vj + vk)
#                 if (not is_fermi_resonance(-vi + vj + vk, kijk, k == j)) or (not do_resonance_checks):
#                     # perturb if no fermi resonance or dont do resonance checks
#                     tmp2 = 1/(-vi + vj + vk)
#                 else:
#                     # deperturbing otherwise - when resonance and do checks
#                     fermi_resonance = add_fermi_resonance(fermi_resonance, [i, j, k, k == j])
#                     tmp2 = 0.0
#
#                 if (not is_fermi_resonance(vi - vj + vk, kijk, k == i)) or (not do_resonance_checks):
#                     # perturb if no fermi resonance or dont do resonance checks
#                     tmp3 = 1/(vi -vj + vk)
#
#                 else:
#                     if i == 8 and j == 5 and k ==8:
#                         tmp3 = 1 / (vi - vj + vk)
#                     else:
#                         # deperturbing otherwise - when resonance and do checks
#                         fermi_resonance = add_fermi_resonance(fermi_resonance, [j, k, i, k == i])
#                         tmp3 = 0.0
#
#                 if (not is_fermi_resonance(vi + vj - vk, kijk, i == j)) or (not do_resonance_checks):
#                     # perturb if no fermi resonance or dont do resonance checks
#                     tmp4 = 1/(vi + vj - vk)
#                 else:
#                     # deperturbing otherwise - when resonance and do checks
#                     fermi_resonance = add_fermi_resonance(fermi_resonance, [k, i, j, i == j])
#                     tmp4 = 0.0
#                 if i == 8 and j == 5:
#                     print('kijk, tmp1, tmp2, tmp3, -tmp4, B +=', (i,j,k), kijk, tmp1, tmp2, tmp3, -tmp4, kijk**2/8.0*(tmp1 + tmp2 + tmp3 - tmp4))
#                 B += kijk**2/8.0*(tmp1 + tmp2 + tmp3 - tmp4)
#
#             if i==8 and j==5:
#                 print('>>>>>>>>> if i==8 and j==5', i, j)
#                 print('-A, -B', -A, -B, '\n')
#
#             C = 0
#
#             for k in range(len(rotational_constant)):
#
#                 C += rotational_constant[k]*coriolis_constant[k][i][j]**2*\
#                     (harmonic_energies[i]/harmonic_energies[j] +
#                      harmonic_energies[j]/harmonic_energies[i])
#
#             X[i][j] = X[i][j] - A - B + C
#             X[j][i] = np.copy(X[i][j])
#
#             X_coriolis[i][j] = C
#             X_coriolis[j][i] = np.copy(X_coriolis[i][j])
#             X_cubic[i][j] = - A - B
#             X_cubic[j][i] = np.copy(X_cubic[i][j])
#
#     fermi_resonance = sorted(fermi_resonance)
#
#     return X, fermi_resonance, X_cubic, X_quartic, X_coriolis

def get_XVPT2(harmonic_energies, cubic_forcefield, quartic_forcefield,
          rotational_constant, coriolis_constant, do_resonance_checks, fermi_resonance):

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

            if [k, i, i, True] in fermi_resonance and do_resonance_checks:
                tmp3 = 0.0
            else:
                tmp3 = 1 / (2.0 * vi - vk)

            rhs += (kiik**2/32.0)*(tmp1 + tmp2 - tmp3)

        X[i][i] += - rhs
        X_cubic[i][i] = - rhs

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

                if [i, *sorted([j, k]), k == j] in fermi_resonance and do_resonance_checks:
                    tmp2 = 0.0
                else:
                    tmp2 = 1/(-vi + vj + vk)

                if [j, *sorted([k, i]), k == i] in fermi_resonance and do_resonance_checks:
                    tmp3 = 0.0
                else:
                    tmp3 = 1 / (vi - vj + vk)

                if [k, *sorted([i, j]), i == j] in fermi_resonance and do_resonance_checks:
                    tmp4 = 0.0
                else:
                    tmp4 = 1/(vi + vj - vk)

                B += kijk**2/8.0*(tmp1 + tmp2 + tmp3 - tmp4)

            # if i==8 and j==5:
            #     print('>>>>>>>>> if i==8 and j==5', i, j)
            #     print('-A, -B', -A, -B, '\n')

            C = 0

            for k in range(len(rotational_constant)):

                C += rotational_constant[k]*coriolis_constant[k][i][j]**2*\
                    (harmonic_energies[i]/harmonic_energies[j] +
                     harmonic_energies[j]/harmonic_energies[i])

            X[i][j] = X[i][j] - A - B + C
            X[j][i] = np.copy(X[i][j])

            X_coriolis[i][j] = C
            X_coriolis[j][i] = np.copy(X_coriolis[i][j])
            X_cubic[i][j] = - A - B
            X_cubic[j][i] = np.copy(X_cubic[i][j])

    return X, X_cubic, X_quartic, X_coriolis

fermi_threshold  = 200.0
martin_threshold = 1.0

def is_fermi_resonance(delta, cubic_force_ijk, i_is_j):
    fermi = False

    if abs(delta) <= fermi_threshold: # in FR should be less than 200 cm-1
        if i_is_j:
            martin_parameter = cubic_force_ijk**4/(256.0*delta**3)
            print('martin_parameter', abs(martin_parameter), abs(martin_parameter) >= martin_threshold, martin_threshold)
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
    # i = new_element[0]
    # j = new_element[1]
    # k = new_element[2]
    # l = new_element[3]
    #
    # j, k = sorted([j, k])
    # new_element[0] = i
    # new_element[1] = j
    # new_element[2] = k
    # new_element[3] = l
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