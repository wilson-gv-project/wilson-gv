import numpy as np
import copy
from CQCParse.debug import debugfunc


def anharm_corr_energies(harmonic_energies, cubic_forcefield, quartic_forcefield,
                         rotational_constant, coriolis_constant, anharmonic_type,
                         list2exclude):
    """
    Takes in cm-1 unit for all the arguments:
    UPD! harmonic_energies is a dictionary - parserObj.fundamentals_harmonic_int

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
    original_len_ene = len(harmonic_energies)
    harmonic_energies = {k: v for k, v in harmonic_energies.items() if k not in list2exclude}

    fundamental = np.zeros((original_len_ene))
    overtones = np.zeros((original_len_ene))
    combotones = np.zeros((original_len_ene, original_len_ene))
    over3q = np.zeros((original_len_ene))
    combo3q = np.zeros((original_len_ene, original_len_ene, original_len_ene))

    fermi_resonance = identify_fermi(harmonic_energies, cubic_forcefield, do_resonance_checks)
    # fermi_resonance = [fermi_resonance[0]]
    X, X_cubic, X_quartic, X_coriolis = get_X(harmonic_energies, cubic_forcefield, quartic_forcefield,
                                              rotational_constant, coriolis_constant, do_resonance_checks,
                                              fermi_resonance, original_len_ene)


    if fermi_resonance: # if not an empty list
        debugfunc(f'Fermi resonances identified - {len(fermi_resonance)}: {fermi_resonance}',
                  tag='vpt2.anharm_corr_energies')

    funds_corrections = np.zeros((original_len_ene))
    # for i in range(len(harmonic_energies)):
    for i in harmonic_energies:
        fundamental[i] += harmonic_energies[i] + 2 * X[i][i]

        fscr = 0
        # for j in range(len(harmonic_energies)):
        for j in harmonic_energies:
            if j != i:
                fscr += 0.5 * X[i][j]

        fundamental[i] += fscr
        funds_corrections[i] += 2 * X[i][i] + fscr

    overtones_corrections = np.zeros((original_len_ene))
    combotones_corrections = np.zeros((original_len_ene, original_len_ene))
    over3q_corrections = np.zeros((original_len_ene))
    combo3q_corrections = np.zeros((original_len_ene, original_len_ene, original_len_ene))

    # for i in range(len(harmonic_energies)):
    for i in harmonic_energies:
        overtones[i] += 2 * fundamental[i] + 2 * X[i][i]
        overtones_corrections[i] += 2 * X[i][i]

        over3q[i] += 3 * fundamental[i] + 6 * X[i][i]
        over3q_corrections[i] += 6 * X[i][i]

        # for j in range(len(harmonic_energies)):
        for j in harmonic_energies:
            if i == j:
                # for k in range(len(harmonic_energies)):
                for k in harmonic_energies:
                    if k != i:
                        combo3q[i][i][k] += 2 * fundamental[i] + 2 * X[i][i] + fundamental[k] + 2 * X[i][k]
                        combo3q_corrections[i][i][k] += 2 * X[i][i] + 2 * X[i][k]

            else:
                combotones[i][j] += fundamental[i] + fundamental[j] + X[i][j]
                combotones_corrections[i][j] += X[i][j]

                # for k in range(len(harmonic_energies)):
                for k in harmonic_energies:
                    if k == i or k == j:
                        continue
                    combo3q[i][j][k] += fundamental[i] + fundamental[j] + fundamental[k] + X[i][j] + X[i][k] + X[j][k]
                    combo3q_corrections[i][j][k] += X[i][j] + X[i][k] + X[j][k]

    if do_variational_correction:
        selectedFR = range((len(fermi_resonance)))
        adjusted_fundamental, adjusted_overtones, adjusted_combotones = \
            adjust_for_fermi_resonance(fundamental, overtones, combotones, over3q, combo3q, cubic_forcefield,
                                       [fermi_resonance[i] for i in selectedFR])
        return (adjusted_fundamental, adjusted_overtones, adjusted_combotones, over3q, combo3q), fermi_resonance

    else:
        return (fundamental, overtones, combotones, over3q, combo3q), fermi_resonance


def identify_fermi(harmonic_energies, cubic_forcefield, do_resonance_checks):
    """
    UPD! harmonic_energies is a dictionary - parserObj.fundamentals_harmonic_int

    """
    fermi_resonance = []
    # for i in range(len(harmonic_energies)):
    for i in harmonic_energies:
        vi = harmonic_energies[i]

        # for k in range(len(harmonic_energies)):
        for k in harmonic_energies:
            vk = harmonic_energies[k]
            kiik = cubic_forcefield[i][i][k]

            isfermi = is_fermi_resonance(2 * vi - vk, kiik, True)
            if isfermi and do_resonance_checks:
                fermi_resonance = add_fermi_resonance(fermi_resonance, [k, i, i, True])

        # attention
        for j in range(i):
            if j in harmonic_energies:
                vj = harmonic_energies[j]

                # for k in range(len(harmonic_energies)):
                for k in harmonic_energies:
                    vk = harmonic_energies[k]
                    kijk = cubic_forcefield[i][j][k]

                    if is_fermi_resonance(-vi + vj + vk, kijk, k == j) and do_resonance_checks:
                        fermi_resonance = add_fermi_resonance(fermi_resonance, [i, *sorted([j, k]), k == j])
                    if is_fermi_resonance(vi - vj + vk, kijk, k == i) and do_resonance_checks:
                        fermi_resonance = add_fermi_resonance(fermi_resonance, [j, *sorted([k, i]), k == i])
                    if is_fermi_resonance(vi + vj - vk, kijk, i == j) and do_resonance_checks:
                        fermi_resonance = add_fermi_resonance(fermi_resonance, [k, *sorted([i, j]), i == j])

    return fermi_resonance


def identify_fermi_c4(harmonic_energies, cubic_forcefield, do_resonance_checks):

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

def get_X(harmonic_energies, cubic_forcefield, quartic_forcefield,
          rotational_constant, coriolis_constant, do_resonance_checks, fermi_resonance,
          original_len_ene):
    """
    UPD! harmonic_energies is a dictionary - parserObj.fundamentals_harmonic_int

    """
    X = np.zeros((original_len_ene, original_len_ene))
    X_cubic = np.zeros((original_len_ene, original_len_ene))
    X_quartic = np.zeros((original_len_ene, original_len_ene))
    X_coriolis = np.zeros((original_len_ene, original_len_ene))

    # for i in range(len(harmonic_energies)):
    for i in harmonic_energies:
        vi = harmonic_energies[i]
        X[i][i] = quartic_forcefield[i][i][i][i]/16.0
        X_quartic[i][i] = quartic_forcefield[i][i][i][i]/16.0

        rhs = 0

        # for k in range(len(harmonic_energies)):
        for k in harmonic_energies:
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

        # attention
        for j in range(i):
            if j in harmonic_energies:

                vj = harmonic_energies[j]
                X[i][j] = quartic_forcefield[i][i][j][j]/4.0
                X_quartic[i][j] = quartic_forcefield[i][i][j][j]/4.0

                A = 0
                # for k in range(len(harmonic_energies)):
                for k in harmonic_energies:
                    A += cubic_forcefield[i][i][k]*cubic_forcefield[j][j][k]/(4.0*harmonic_energies[k])

                B = 0
                # for k in range(len(harmonic_energies)):
                for k in harmonic_energies:
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

                C = 0

                for k in range(len(rotational_constant)):
                    # print(type(rotational_constant[k]), 'vpt2.py line 260')
                    # print(rotational_constant[k], float(rotational_constant[k]))
                    # C += rotational_constant[k]*coriolis_constant[k][i][j]**2*\
                    #     (harmonic_energies[i]/harmonic_energies[j] +
                    #      harmonic_energies[j]/harmonic_energies[i])
                    C += float(rotational_constant[k])*coriolis_constant[k][i][j]**2*\
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
            # print('martin_parameter', abs(martin_parameter), abs(martin_parameter) >= martin_threshold, martin_threshold)
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


def is_fermi_resonance_c4(delta, cubic_force_ijk, i_is_j):
    fermi = False
    fermi_threshold = 50.0
    martin_threshold = 1.0

    if abs(delta) <= fermi_threshold: # in FR should be less than 200 cm-1
        if i_is_j:
            martin_parameter = cubic_force_ijk**4/(256.0*delta**3)
            # print('martin_parameter', abs(martin_parameter), abs(martin_parameter) >= martin_threshold, martin_threshold)
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
    if new_element not in total_list:
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


def get_vpt2_corrected_levels(parsed_data, vpt2settings, list2exclude=None, print_level=0):
    """
    Returns VPT2 corrected energy levels of all states as a dictionary : {str(int): float}
    """

    if vpt2settings is None:
        vpt2settings = {'anharmonic_type': 'VPT2'}

    if list2exclude is None:
        list2exclude = []

    # if parserObj.DD11 or parserObj.DD13 or parserObj.DD22:
    #     print("Warning: found Darling-Dennison resonances_args in data:")
    #     print(f"DD 1-1: {parserObj.DD11}")
    #     print(f"DD 2-2: {parserObj.DD22}")
    #     print(f"DD 1-3: {parserObj.DD13}")

    one = {k: v for k,v in parsed_data.vib_states.anharmonic_states.items() if len(k) == 1}
    two = {k: v for k,v in parsed_data.vib_states.anharmonic_states.items() if len(k) == 2}

    if print_level == 1:
        print('\nOriginal anharm corrected:')
        print(dict(sorted(one.items())))
        print(dict(sorted(two.items())), '\n')

    cff_cm_1 = parsed_data.derivatives.cubic_cm_1
    qff_cm_1 = parsed_data.derivatives.quartic_cm_1
    rot_c = parsed_data.anharm_correction_data.rotational_constants
    cor_c = parsed_data.anharm_correction_data.coriolis_constants
    # list, not associated to normal mode indices

    # corrected_levels : funds, over2q, combo2q, over3q, combo3q
    # corrected_levels = anharm_corr_energies(upd_harmonic_energies,
    corrected_levels, fermi_resonance = anharm_corr_energies(parsed_data.vib_states.fundamentals_harmonic_int,
                                                             cff_cm_1, qff_cm_1, rot_c, cor_c,
                                                             vpt2settings['anharmonic_type'], list2exclude)
    # print(corrected_levels)
    # exit()
    all_states_corr = {}
    # for k, v in parsed_data.vib_states.anharmonic_states.items():
    #     t1 = tuple([int(i) for i in k])
    #     if len(t1)==1:
    #         all_states_corr[k] = corrected_levels[0][t1]
    #     elif len(t1)==2 and t1[0]==t1[1]:
    #         all_states_corr[k] = corrected_levels[1][t1[0]]
    #     elif len(t1)==2 and t1[0]!=t1[1]:
    #         all_states_corr[k] = corrected_levels[2][t1]
    #     elif len(t1)==3 and t1[0]==t1[1]==t1[2]:
    #         all_states_corr[k] = corrected_levels[3][t1[0]]
    #     else:
    #         all_states_corr[k] = corrected_levels[4][t1]


    for i in range(len(parsed_data.vib_states.fundamentals_harmonic_int)):
        all_states_corr[(str(i),)] = corrected_levels[0][i]

        for j in range(i + 1):
            if i == j:
                all_states_corr[tuple([str(i), str(i)])] = corrected_levels[1][i]
            else:
                all_states_corr[tuple([str(el) for el in sorted([i, j])])] = corrected_levels[2][i, j]

            for k in range(len(parsed_data.vib_states.fundamentals_harmonic_int)):
                # if i==0 and j==0 and k==0:
                #     print('jhello')
                if i == j == k:
                    all_states_corr[tuple([str(i), str(i), str(i)])] = corrected_levels[3][i]
                else:
                    key = tuple([str(el) for el in sorted([i, j, k])])
                    if key not in all_states_corr:
                        if corrected_levels[4][i, j, k] != 0.:
                            all_states_corr[key] = corrected_levels[4][
                                i, j, k]

    all_states = copy.deepcopy(all_states_corr)
    one = {i: all_states[i] for i in all_states if len(i) == 1}
    two = {i: all_states[i] for i in all_states if len(i) == 2}

    if print_level == 1:
        print('\nGVPT2 anharm corrected:')
        print(dict(sorted(one.items())))
        print(dict(sorted(two.items())), '\n')

    return all_states, fermi_resonance