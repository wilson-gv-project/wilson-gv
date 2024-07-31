#!/usr/bin/env python
import os
import glob
import re

def check_anharm_save(file_path):
    # Define the regular expression pattern
    # Explanation:
    # - \s+ matches one or more whitespace characters
    # - \d+ matches one or more digits
    # - -?\d+\.\d+ matches an optional negative sign, followed by one or more digits, a decimal point, and one or more digits
    pattern = r"Z\s+Third\s+\d+\s+\d+\s+\d+\s+-?\d+\.\d+\s+-+\s+@CHECKOUT-I, Total execution time \(CPU/WALL\):"

    # Read the file and search for the pattern
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            if re.search(pattern, content, re.MULTILINE):
                return True
            else:
                return False
    except FileNotFoundError:
        print("File not found.")
        return False

def get_highest_numbered_file(directory, prefix, suffix):
    highest_number = -1
    highest_file = None

    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(suffix):
            try:
                number_part = filename[len(prefix):-len(suffix)]
                file_number = int(number_part)
                if file_number > highest_number:
                    highest_number = file_number
                    highest_file = filename
            except ValueError:
                continue

    return highest_file

def seeAllDirs():
    walking = next(os.walk('.'))[1]
    conclusions = {'anharm': [], 'polar': []}
    
    if 'anharm' in walking:
        necessaryFiles = ['dipolex', 'dipoley', 'dipolez', 'cubic', 'out']
        a_subdirs = sorted(next(os.walk('./anharm'), (None, None, []))[1])
        #print('\n', a_subdirs)
        if 'save' in a_subdirs:
            conclusions['anharm'].append("\nanharm/save directory exists")
            files_save = dict(zip(necessaryFiles, [os.path.isfile('anharm/save/'+f) for f in necessaryFiles]))
            if all([os.path.isfile('anharm/save/'+f) for f in necessaryFiles]):
                if check_anharm_save('anharm/save/out'):
                    conclusions['anharm'].append("ANHARM calculation is complete!\n")
            else:
                conclusions['anharm'].append(f"ANHARM files are missing: {files_save}\n")
                dictanh = {}
                for adir in a_subdirs:
                    if adir != 'save':
                        af = get_highest_numbered_file('./anharm/'+adir, 'outfile', '.out')
                        #print(af, adir)
                        with open('./anharm/'+adir+'/'+af, 'r') as outfile:
                            outfile_content = outfile.readlines()
                            dictanh[adir] = "The final electronic energy is" in ''.join(outfile_content)
                fanh = {k:v for k,v in dictanh.items() if v==False}
                conclusions['anharm'].append('>>>   anharm, non-complete ones: '+str(fanh))
                print('\ndictanh:\n', dictanh)
        for i in conclusions['anharm']: print(i)

    if 'polar' in walking:
        p_subdirs = sorted(next(os.walk('./polar'), (None, None, []))[1])
        #print('\n', p_subdirs)
        dictpol = {}
        for pdir in p_subdirs:
            pf = get_highest_numbered_file('./polar/'+pdir, 'outfile', '.out')
            #print(pf)
            with open('./polar/'+pdir+'/'+pf, 'r') as outfile:
                outfile_content = outfile.readlines()
                dictpol[pdir] = "The final electronic energy is" in ''.join(outfile_content)
        fpol = {k:v for k,v in dictpol.items() if v==False}
        conclusions['polar'].append('\n>>>   polar, non-complete ones: '+str(fpol))
        print('\ndictpol:\n', dictpol)
        for i in conclusions['polar']: print(i)
    

excludethisdirs = ['coh2_h2o', 'hp_coh2aldehyde', 'ch3coh']
andthis = ['coh2aldehyde', 'propyne', 'ethylamine', 'formicac', 'methanol']
seeAllDirs()

