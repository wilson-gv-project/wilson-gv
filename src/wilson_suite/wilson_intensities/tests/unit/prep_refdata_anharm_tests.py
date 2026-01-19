import wilson_suite as ws
from wilson_suite.wilson_utils.wilson_data_obtainer import wilson_data_obtainer
from wilson_suite.wilson_utils.serialization import pickle_this_to, unpickle_smth_from
from os import listdir
from os.path import isfile, join

base_dir = "/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/dftGaussian/FORM/"

var_dirs = ['HF_STO_3G/', 'HF_STO_3G_DVPT2/', 'HF_STO_3G_DVPT2_Resonances/', 
            'HF_STO_3G_GVPT2/', 'HF_STO_3G_GVPT2_NoCoriolis/', 'HF_STO_3G_GVPT2_NoCoriolis_really/', 
            'HF_STO_3G_GVPT2_Resonances/', 
            'HF_STO_3G_GVPT2_lineqs_conv1/', 
            # 'HF_STO_3G_GVPT2_lineqs_conv2/', 
            # 'HF_STO_3G_GVPT2_lineqs_conv3/', 
            # 'HF_STO_3G_GVPT2_lineqs_conv4/',
            'HF_STO_3G_VPT2/']

def collect_pickles():
    for directory in var_dirs:
        # Source - https://stackoverflow.com/a
        # Posted by pycruft, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-01-19, License - CC BY-SA 4.0

        onlyfiles = [f for f in listdir(base_dir+directory) if isfile(join(base_dir+directory, f))]
        print('\ndirectory:', directory)
        print('files:', onlyfiles)

        vib_ana = ws.main.abstractions.VibAnaSetup(system='', regime='compare', vibana_own_analysis='none')

        calc_setup = ws.main.abstractions.DataOriginInfo(source_type='gaussian',
                                                            lvl_theory='HF', 
                                                            basis_set='STO-3G', 
                                                            base_file_loc=base_dir+directory+'g16_inputFull_3q.out')

        vib_ana, props = ws.main.main_functions.get_data_for_vibanalysers(vib_ana=vib_ana, 
                                                            calc_setup=calc_setup, 
                                                            obtainer=wilson_data_obtainer)
        pickle_this_to((vib_ana, props), filenamepkl=f'{directory[:-1]}.pkl', save_to='/home/vlev/pre_anharm_data_pkls/')

def analyze_from_pkl():
    pickles_dir = '/home/vlev/pre_anharm_data_pkls/'
    onlyfiles = [f for f in listdir(pickles_dir) if isfile(join(pickles_dir, f)) and 'txt' not in f]

    for f in onlyfiles:
        print(f'\n## Processing file: {f}')
        vib_ana, props = unpickle_smth_from(filenamepkl=f, load_from=pickles_dir)
        states_g16 = vib_ana.states

        vib_ana.states = vib_ana._harm_states
        # ---- do analysis
        states, diagn = ws.intensities.anharmonic_treatment.anharm_analyzer_data(props=props,
                                                                                nc_sqrt_eigval=vib_ana.nc_sqrt_eigval,
                                                                                regime="VPT2", #vib_ana.regime
                                                                                exclude_modes=None)        
        # ---- check results
        # st_dict = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states}
        st_dict_g16 = {','.join(list(s.harm_quanta_coeffs.keys())[0]): s.energy for s in states_g16}
        nc_sqrt_eigval_corrected = {int(list(s.harm_quanta_coeffs.keys())[0][0]): s.energy for s in states if len(list(s.harm_quanta_coeffs.keys())[0])==1}
        assert vib_ana.nc_sqrt_eigval != nc_sqrt_eigval_corrected

        # print(st_dict)
        # for k,v in st_dict.items():
        #     print(k.ljust(10), v)
        print(diagn)
        
        print('\n1 quantum levels')
        print('g16 nc_sqrt_eigval -- wilson corrm -- g16 corr')
        for k,v in vib_ana.nc_sqrt_eigval.items():
            print(k, '--', v, '--', nc_sqrt_eigval_corrected[k], '--', st_dict_g16[str(k)])
            # assert nc_sqrt_eigval_corrected[k] == st_dict_g16[str(k)]


if __name__ == '__main__':
    # collect_pickles()
    analyze_from_pkl()
