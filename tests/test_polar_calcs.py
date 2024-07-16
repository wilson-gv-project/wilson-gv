
import hyobj as hyo

def get_zmat_from_ZMAT(cfourZMATfile: str):
    """

    :param cfourZMATfile:
    :return:
    """
    with open(cfourZMATfile, "r") as f:
        cfourZMAT = f.read()

    blocks = cfourZMAT.split('\n\n')
    structure = blocks[0].strip().split('\n')[1:]
    values = blocks[1].strip().split('\n')

    value_dict = {}
    for value in values:
        key, val = value.split('=')
        key = key.strip()
        val = float(val.strip())
        value_dict[key] = val

    def replace_variables(line, value_dict):
        tokens = line.split()
        for i, token in enumerate(tokens):
            if token in value_dict:
                tokens[i] = str(value_dict[token])
        return ' '.join(tokens)

    transformed_structure = [replace_variables(line, value_dict) for line in structure]
    result = '\n'.join(transformed_structure)

    return result

def get_zmatLikeList(cfourZMATfile: str):
    with open(cfourZMATfile, "r") as f:
        cfourZMAT = f.read()
    blocks = cfourZMAT.split('\n\n')
    cartesianFormat = ('COORD=CARTESIAN' in cfourZMAT
                       and 'COORD=INTERNAL' not in cfourZMAT and len(blocks) == 2)

    if cartesianFormat:
        xyz_lines = blocks[0].strip().split('\n')[1:]
        # print(xyz_lines)
        xyz_lst = []
        for line in xyz_lines:
            line_list = line.strip().split()
            line_upd = [float(i) if line_list.index(i) != 0 else i for i in line_list]
            xyz_lst.append(line_upd)
        return xyz_lst
    else:
        structure = blocks[0].strip().split('\n')
        values = blocks[1].strip().split('\n')

        value_dict = {}
        for value in values:
            key, val = value.split('=')
            key = key.strip()
            val = float(val.strip())
            value_dict[key] = val

        # print(value_dict)
        def replace_variables(line, value_dict):
            tokens = line.split()
            result = []
            for token in tokens:
                if token.strip('*') in value_dict:
                    result.append(value_dict[token.strip('*')])
                else:
                    try:
                        result.append(float(token))
                    except ValueError:
                        result.append(token)
            return result

        transformed_structure = [replace_variables(line, value_dict) for line in structure]
        return transformed_structure[1:]

def get_molecules_PolarDir(polar_directory: str):
    import os
    subdirs = [d for d in os.listdir(polar_directory) if os.path.isdir(d)]

    collected_polar_displGeos = {}
    for subdir in subdirs:
        zmatName = '/'.join([polar_directory, subdir, 'ZMAT'])
        collected_polar_displGeos[subdir] = hyo.Molecule(get_zmatLikeList(zmatName), units='bohr')

    return collected_polar_displGeos

def get_molecules_Quadrature(cfourQuadratureFile: str, cfourMoldenFile: str) -> dict[str:hyo.Molecule]:
    from calculations import parseCFOUR_forWilson
    equilibrium_geometry_Molden, atomsStrs_Molden, normal_modes_Molden = parseCFOUR_forWilson.pMOLDEN(cfourMoldenFile)
    equilibrium_geometry, freqs, normal_modes = parseCFOUR_forWilson.pQUADRATURE(cfourQuadratureFile)

    single_mode_displaced_arrays = {}
    for displ in normal_modes:
        single_mode_displaced_arrays[f'{displ}n'] = equilibrium_geometry - 0.01 * normal_modes[displ]
        single_mode_displaced_arrays[f'{displ}p'] = equilibrium_geometry + 0.01 * normal_modes[displ]

    single_mode_displaced_xyzstr = {}
    for displ in normal_modes:
        displArr_n = equilibrium_geometry - 0.01 * normal_modes[displ]
        displArr_p = equilibrium_geometry + 0.01 * normal_modes[displ]

        single_mode_displaced_xyzstr[f'{displ}n'] = []
        for na1, atom1 in enumerate(displArr_n):
            oneline1 = [atomsStrs_Molden[na1][0]]
            oneline1.extend(atom1)
            single_mode_displaced_xyzstr[f'{displ}n'].append(oneline1)

        single_mode_displaced_xyzstr[f'{displ}p'] = []
        for na2, atom2 in enumerate(displArr_p):
            oneline2 = [atomsStrs_Molden[na2][0]]
            oneline2.extend(atom2)
            single_mode_displaced_xyzstr[f'{displ}p'].append(oneline2)

    displ_molecules = {}
    for i in single_mode_displaced_xyzstr:
        displ_molecules[i] = hyo.Molecule(single_mode_displaced_xyzstr[i])

    return displ_molecules

import sys
orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

def test_polar_displacement_geometries_FOAC():
    polar_directory = '/cluster/projects/nn14654k/vle014/refinedc4/formicac/CCSDTcc-pVQZ/polar/'
    cfourQuadratureFile = '/cluster/projects/nn14654k/vle014/refinedc4/formicac/CCSDTcc-pVQZ/polar/QUADRATURE_f'
    cfourMoldenFile = '/cluster/projects/nn14654k/vle014/refinedc4/formicac/CCSDTcc-pVQZ/polar/MOLDEN_f'

    fromPolDir = get_molecules_PolarDir(polar_directory)
    fromQuadrature = get_molecules_Quadrature(cfourQuadratureFile, cfourMoldenFile)

    results = {}
    print('\nFOAC')
    for directory in fromPolDir:
        results[directory] = fromPolDir[directory] == fromQuadrature[directory]
        print(f'\n{directory} fromPolDir\n', fromPolDir[directory].coordinates)
        print(f'\n{directory} fromQuadrature\n', fromQuadrature[directory].coordinates)
    assert all(results.values())

def test_polar_displacement_geometries_FORM():
    polar_directory = '/cluster/projects/nn14654k/vle014/refinedc4/coh2aldehyde/CCSDTcc_pVQZ/polar/'
    cfourQuadratureFile = '/cluster/projects/nn14654k/vle014/refinedc4/coh2aldehyde/CCSDTcc_pVQZ/polar/QUADRATURE_f'
    cfourMoldenFile = '/cluster/projects/nn14654k/vle014/refinedc4/coh2aldehyde/CCSDTcc_pVQZ/polar/MOLDEN_f'

    fromPolDir = get_molecules_PolarDir(polar_directory)
    fromQuadrature = get_molecules_Quadrature(cfourQuadratureFile, cfourMoldenFile)

    results = {}
    # print('\nFORM', file='./testout')
    for directory in fromPolDir:
        results[directory] = fromPolDir[directory] == fromQuadrature[directory]
        # print(f'\n{directory} fromPolDir\n', fromPolDir[directory].coordinates, file='./testout')
        # print(f'\n{directory} fromQuadrature\n', fromQuadrature[directory].coordinates, file='./testout')
    assert all(results.values())

def test_polar_displacement_geometries_METH():
    polar_directory = '/cluster/projects/nn14654k/vle014/refinedc4/methanol/ch3oh_ccsdt_QZ_s_30c_1t_opt_newgeo_new_tight/polar/'
    cfourQuadratureFile = '/cluster/projects/nn14654k/vle014/refinedc4/methanol/ch3oh_ccsdt_QZ_s_30c_1t_opt_newgeo_new_tight/polar/QUADRATURE_f'
    cfourMoldenFile = '/cluster/projects/nn14654k/vle014/refinedc4/methanol/ch3oh_ccsdt_QZ_s_30c_1t_opt_newgeo_new_tight/polar/MOLDEN_f'

    fromPolDir = get_molecules_PolarDir(polar_directory)
    fromQuadrature = get_molecules_Quadrature(cfourQuadratureFile, cfourMoldenFile)

    results = {}
    # print('\nMETH', file='./testout')
    for directory in fromPolDir:
        results[directory] = fromPolDir[directory] == fromQuadrature[directory]
        # print(f'\n{directory} fromPolDir\n', fromPolDir[directory].coordinates, file='./testout')
        # print(f'\n{directory} fromQuadrature\n', fromQuadrature[directory].coordinates, file='./testout')
    assert all(results.values())

sys.stdout = orig_stdout
f.close()