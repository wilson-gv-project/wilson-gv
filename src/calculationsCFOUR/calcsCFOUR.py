##############################################################################
##                                                                          ##
##                             Samurai Tools                                ##
##                                                                          ##
##############################################################################
#
# 1. Optimize structure --> ZMATnew
# 2. With ZMATnew geometry, run ANH_ALGORITHM=PARALLEL, VIBRATION=ANALYTIC,
#                               FD_PROJECT=ON --> zmat0* files
#
#          xcfour > "$output_filename"
#   2a. Add lines to generated submit.sh
#          mkdir save
#          cp JOBARC ./save/
#          cp JAINDX ./save/
#          ../../../../../scriptsHPC/cfourscripts/vpt2_parallel/mkzmatdirs
# 3. Run all the new zmat0* in their directories
#
#           xcfour > "$output_filename"
#    3a. Add lines to generated submit.sh
#           cp DCT dct0
#           xja2fja >> out1
#           cp FJOBARC ../save/fja.004
# 4. Run post-processing script for fja.0* files
#
#         # Copy fja.x to FJOBARC
#         cp "$file" FJOBARC
#         # Execute xja2fja
#         xja2fja
#         # Execute xcubic and append output to out file
#         xcubic >> out
# 5. Make pickles from output files and save them in the data directory
# 6. Tadaaa
#


def makeOptZmat(settings: dict, molecule: str):
    levelT = settings['level of theory']
    GEO_CONV = settings['geoconv']
    CC_CONV = settings['ccconv']
    CC_MAXCYC = settings['cccycles']
    SCF_CONV = settings['scfconv']
    SCF_MAXCYC = settings['scfcycles']
    LINEQ_CONV = settings['lineqconv']
    LINEQ_MAXCY = settings['lineqcycles']

    startGeo = molecule
#     startGeo = """O
# C 1 B1*
# H 2 B2* 1 A1*
# H 2 B2* 1 A1* 3 D1
#
# B1   =        1.215105045020483
# B2   =        1.118923051621344
# A1   =      122.394541535245793
# D1   =      180.000000000000000
# """
    titleZmat = 'geo optimization'

    template = f"""formaldehyde {titleZmat}
{startGeo}
*CFOUR(CALC={levelT[0]}
BASIS={levelT[1]}
ABCDTYPE=AOBASIS
CC_PROG=ECC
GEO_CONV={GEO_CONV}
CC_CONV={CC_CONV}
CC_MAXCYC={CC_MAXCYC}
SCF_CONV={SCF_CONV}
SCF_MAXCYC={SCF_MAXCYC}
LINEQ_CONV={LINEQ_CONV}
LINEQ_MAXCY={LINEQ_MAXCY}
MEMORY_SIZE=5
MEM_UNIT=GB)

"""

    # print(template)
    # Writing to file
    with open("ZMAT", "w") as file1:
        # Writing data to a file
        file1.writelines(template)


def fromZmatNew2Zmat(zmatnew: str, settings: dict):
    jobtype = settings['jobtype']

    import os
    dir = f'./{jobtype}'
    os.makedirs(dir, exist_ok=True)

    import shutil
    shutil.copy(zmatnew, f'./{jobtype}/ZMAT')

    with open(f'./{jobtype}/ZMAT', "r+") as file1:
        # Reading form a file
        content = file1.readlines()

    coords = []
    params = []
    count = 0
    c = True
    for line in content:
        if line.strip() != '' and c:
            coords.append(line.strip())
        else:
            c = False
            if count == 0:
                params.append(line.strip())
                count += 1
                continue
            else:
                if line.strip() != '':
                    params.append(line.strip())
                else:
                    break

    coords = [s.replace("*", "") for s in coords]
    coords[0] += f'| here: {jobtype}'

    levelT = settings['level of theory']
    GEO_CONV = settings['geoconv']
    CC_CONV = settings['ccconv']
    CC_MAXCYC = settings['cccycles']
    SCF_CONV = settings['scfconv']
    SCF_MAXCYC = settings['scfcycles']
    LINEQ_CONV = settings['lineqconv']
    LINEQ_MAXCY = settings['lineqcycles']
    specificCalc = settings['job']
    GEO_MAXCYC = settings['geocycles']

    calc = f"""

*CFOUR(CALC={levelT[0]}
BASIS={levelT[1]}
ABCDTYPE=AOBASIS
CC_PROG=ECC
GEO_CONV={GEO_CONV}
GEO_MAXCYC={GEO_MAXCYC}
CC_CONV={CC_CONV}
CC_MAXCYC={CC_MAXCYC}
SCF_CONV={SCF_CONV}
SCF_MAXCYC={SCF_MAXCYC}
LINEQ_CONV={LINEQ_CONV}
LINEQ_MAXCY={LINEQ_MAXCY}
{specificCalc}
MEMORY_SIZE=5
MEM_UNIT=GB)

"""

    znew = '\n'.join(coords) + '\n' + '\n'.join(params) + calc
    # print(znew)

    # Writing to file
    with open(f'./{jobtype}/ZMAT', "w") as file1:
        # Writing data to a file
        file1.writelines(znew)


def generateSubmit(bash_script_path: str, config: dict, outname: str = None):
    """
    Running bash script that generates submit.sh

    :param outname:
    :param bash_script_path:
    :param config:
    :return:
    """
    machine = config['machine']
    minutes = config['minutes']
    hours = config['hours']

    import subprocess
    # Run a bash script (make sure the script has execute permissions)
    # bash_script_path = f'/cluster/projects/nn14654k/vle014/scriptsHPC/submit_utils/scrmaster_{machine}'

    cpus = 16 if machine == 'fram' else 20

    # Run a bash script in a specific location (cdw), here - in curdir
    script_args = [f'-h{hours}', f'-m{minutes}', '-n1', f'-c{cpus}', '-t2']
    subprocess.run(['bash', bash_script_path] + script_args, check=True)  # cwd=new_directory


def generateSubmitPy(config: dict, outname: str):
    """
    Generating submit.sh or other outname script with python

    :param outname:
    :param config:
    :return:
    """
    machine = config['machine']
    minutes = config['minutes']
    hours = config['hours']
    nodes = config['nodes']
    tasks = 2  # config['tasks']
    cfourpath = config['c4path']

    import os
    cwd = os.getcwd().split('/')

    if config['dir3']:
        dir_name = '/'.join([cwd[-3], cwd[-2], cwd[-1]])
    else:
        dir_name = '/'.join([cwd[-2], cwd[-1]])
    print('Job Name will be', dir_name)

    cpus = 16 if machine == 'fram' else 20

    text = rf"""#!/bin/bash
# ==================================================================================
# SLURM job scheduler for the OpenMP-parallel CFOUR program and {machine} supercomputer
# ==================================================================================
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tasks}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mail-type=ALL
#SBATCH --job-name={dir_name}
#SBATCH --account=nn14654k
#SBATCH --partition normal
#SBATCH --time={hours}:{minutes}:00

#load modules
module purge
module load gompi/2023a
module load imkl/2023.1.0
module load OpenMPI/4.1.5-GCC-12.3.0
module list

# Function to log job details
log_job_details() {{
    job_id=$1
    submit_time=$2
    current_time=$(date '+%Y-%m-%d %H:%M:%S')

    job_name=$3
    user=$4
    queue=$5
    current_directory=$6

    cfour_grep=$(grep "CFOUR(" ZMAT)
    first_line=$(head -n 1 ZMAT)

    job_info="\e[0;32m$job_id\e[0m,$job_name,$user,$queue,\e[0;31m$current_directory\e[0m,$submit_time,$cfour_grep,$first_line"

    echo "$job_info" >> ../../job_log.csv
}}

# Capture job details
job_id=$SLURM_JOB_ID
job_name=$SLURM_JOB_NAME
user=$USER
queue=$SLURM_QUEUE
current_directory=$SLURM_SUBMIT_DIR
submit_time=$(date '+%Y-%m-%d %H:%M:%S')

#find . -name 'slurm-*' -delete
#found_file=$(find . -type f -name "slurm-*" -printf "%f\\n")
#echo $found_file
#mv $found_file "./old_${{found_file}}"

directory_name=$(basename "$current_directory")

# Function call to log job details
log_job_details "$job_id" "$submit_time" "$job_name" "$user" "$queue" "$directory_name"


#set some local OMP-specific defaults
ulimit -s unlimited
export OMP_NUM_THREADS={cpus}
export OMP_STACKSIZE=4500m

#setup the scratch directory
#cd $SLURM_SUBMIT_DIR
#echo "The job submitting directory is:" ${{SLURM_SUBMIT_DIR}}    >>  log
#echo "The job submitting directory is:" $(pwd)    >>  log

#setup ReSpect installation directory
export CFOUR={cfourpath}

# add to path if not there
PATH=${{PATH}}:{cfourpath}

#execute CFOUR
#xcfour >> outfile

outfile="outfile"
extension=".out"  # Define the extension
counter=0

# Check if the output file exists
while [[ -e "$outfile$counter$extension" ]]; do
    (( counter++ ))
done

# Use the first available filename
output_filename="$outfile$counter$extension"

# Run your command with the updated output filename
xcfour > "$output_filename"

exit 0
"""

    with open(outname, "w") as file1:
        # Writing data to a file
        file1.writelines(text)

    # make executable
    import subprocess
    subprocess.check_call(['chmod', '+x', outname])


def extendSubmitEquilParAnh(submitfile: str, configHPC: dict):
    """
    Extend submit script for ANHARM=VPT2 ANH_ALGORITHM=PARALLEL, VIBRATION=ANALYTIC job

    New submit.sh will:
        1 - make 'save' directory for anharmonic parallel post-processing
        2 - save JOBARC and JAINDX files in 'save' directory
        3 - for each zmat0* file:
            - will make a directory '0*'
            - copy corresponding zmat0* file as ZMAT and GENBAS file from cfour src dir to new dir
            - generate a submit script for new dir with (cpus=16, tasks=2, nodes=1) and time of the original submit.sh
            - extend new submit with : cp DCT dct0; xja2fja >> out1; cp FJOBARC ../save/fja.$(basename "$(pwd)")
            - sbatch submit.sh

    :param submitfile:
    :return: after running this function, a big submit script will be created and after all submitted jobs are finished,
            'save' dir is ready for the final step of the calculation
    """

    # Reading from file
    with open(submitfile, "r+") as file1:
        # Reading form a file
        content = file1.readlines()

    for ln in content:
        if '#SBATCH --time=' in ln:
            '#SBATCH --time=00:10:00'
            time = ln.split('=')[1].split(':')
            hm = [time[0], time[1]]
            timeoriginal = tuple(hm)

    configHPCstr = f"""configHPC = {{'machine': '{configHPC['machine']}', 'minutes': {configHPC['minutes']}, 
             'hours': '{configHPC['hours']}', 'nodes': {configHPC['nodes']}, 'dir3': {configHPC['dir3']},
             'c4path': '/cluster/projects/nn14654k/vle014/cfour_serial/bin'}}"""

    index = [l.strip() for l in content].index('xcfour > "$output_filename"')
    elmnt = """
mkdir save
cp JOBARC ./save/
cp JAINDX ./save/
# ../../../../../scriptsHPC/cfourscripts/vpt2_parallel/mkzmatdirs

# Find all zmat0* files in the current directory
files=$(find . -maxdepth 1 -type f -name 'zmat0*')
echo $files

# Iterate through the found files
for file in $files; do
    # Extract the directory name from the file
    dir_name="${file:6}"
    # Create the directory if it doesn't exist
    mkdir -p "$dir_name"

    # Copy the file into the directory with name ZMAT
    cp "$file" "$dir_name/ZMAT"
    cp "/cluster/projects/nn14654k/vle014/cfour_serial/basis/GENBAS"  "$dir_name/" 

    echo "Created directory '$dir_name' and copied '$file' into '$dir_name/ZMAT'"
"""
    elemnt2 = f"""    # Define arguments for script2.sh
    hours={timeoriginal[0]}
    minutes={timeoriginal[1]}
    cpus=16
    tasks=2
    nodes=1

    cd "$dir_name"
    
    #source /cluster/projects/nn14654k/vle014/scriptsHPC/submit_utils/scrmaster_fram -h "$hours" -m "$minutes" -n "$nodes" -t "$tasks" -c "$cpus"
    python << END
from scriptsHPC.utils import calcsCFOUR
{configHPCstr}
calcsCFOUR.generateSubmitPy(configHPC, "submit.sh")
END
    
    file_path="./ZMAT"
    search_string="FD_PROJECT=ON"  # Replace with the string you want to search for
    new_content="FD_PROJECT=OFF"    # Replace with the new content you want to set

    line_to_insert='cp FJOBARC ../save/fja.$(basename "$(pwd)")'
    file_path11="./submit.sh"
    # Use sed to insert the line into the file after a specific pattern (here, we assume a placeholder pattern)
    sed -i '/xcfour > "$output_filename"/a '"$line_to_insert"'' "$file_path11"

    line_to_insert="xja2fja >> out1"  # Replace with the line you want to insert
    file_path11="./submit.sh"
    # Use sed to insert the line into the file after a specific pattern (here, we assume a placeholder pattern)
    sed -i '/xcfour > "$output_filename"/a '"$line_to_insert"'' "$file_path11"

    line_to_insert="cp DCT dct0"  # Replace with the line you want to insert
    file_path11="./submit.sh"
    # Use sed to insert the line into the file after a specific pattern (here, we assume a placeholder pattern)
    sed -i '/xcfour > "$output_filename"/a '"$line_to_insert"'' "$file_path11"

    sbatch submit.sh
    cd "../"
done

"""
    content.insert(index + 1, elmnt + elemnt2)

    with open(submitfile, "w") as file2:
        # Writing data to a file
        file2.writelines(content)


def sumbitSbatch(sbubmitname: str):
    """
    Should be called in the directory with submit.sh file in it
    :return:
    """
    import subprocess
    import re

    # Define the command to be run
    command = ["sbatch", sbubmitname]

    # Run the command and capture the output
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # sbatch typically returns a string like "Submitted batch job 12345"
        # We'll use a regular expression to extract the job ID
        job_id_search = re.search(r'Submitted batch job (\d+)', result.stdout)

        if job_id_search:
            job_id = job_id_search.group(1)
            print(f"Job submitted successfully. Job ID is {job_id}")
            return job_id
        else:
            print("Could not find Job ID in sbatch output.")
            print("sbatch output:", result.stdout)
            print("sbatch error output:", result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while submitting the job: {e}")
        print("sbatch output:", e.stdout)
        print("sbatch error output:", e.stderr)


def checkJobStatusID(jobid: str):
    # check running jobs from squeue
    import subprocess
    command_output = subprocess.run(['squeue', '-u', 'vle014', '-o', '%.18i %.9P %.48j %.8u %.8T %.10M %.12l %.6D %R'],
                                    capture_output=True, text=True)
    output_lines = command_output.stdout.splitlines()

    # Extracting the 'JOBID' column to a list for comparison
    runningIDs = [line.split()[0] for line in output_lines[1:] ]

    if jobid in runningIDs:
        return 'RUNNING'
    else:
        slurmout = f'slurm-{jobid}.out'
        with open(slurmout, 'r') as f:
            text = f.read()
        from scriptsHPC.report_utils import utils
        requested_time, elapsed_time, rw_unit, ew_unit, billing_hours = utils.extract_info_from_text(text)
        return {'requested_time': requested_time, 'rw_unit': rw_unit,
                'elapsed_time': elapsed_time, 'ew_unit': ew_unit,
                'billing_hours': billing_hours}

def checkJobStatusNAME(basedirname: str):
    # check running jobs from squeue
    import subprocess
    command_output = subprocess.run(['squeue', '-u', 'vle014', '-o', '%.18i %.9P %.48j %.8u %.8T %.10M %.12l %.6D %R'],
                                    capture_output=True, text=True)
    output_lines = command_output.stdout.splitlines()

    # Extracting the 'NAME' column to a list for comparison
    running_jobs = [line.split()[2] for line in output_lines[1:] ]  # Assuming the 'NAME' column is at index 3
#    print(running_jobs, basedirname)
#    print([basedirname in i.split('/') for i in running_jobs])

    if any([basedirname in i for i in running_jobs]):
        return 'RUNNING'
    else:
        return "DONE"


def checkStatus(typecheck: str, inputname: str):
    """
    Usage:
    resultJob = calcsCFOUR.checkJobStatus(jobid)

    while resultJob == 'RUNNNING':
        import time
        time.sleep(20)
        resultJob = calcsCFOUR.checkJobStatus(jobid)
    :param typecheck: id or name
    :param inputname: if id then it's job id; if name then it's basedir name
    :return:
    """
    import time

    if typecheck == 'id':
        resultJob = checkJobStatusID(inputname)
        print(f'\nFirst checkJobStatus: {resultJob}; while - {resultJob == "RUNNING"}')

        while resultJob == 'RUNNING':
            time.sleep(30)
            resultJob = checkJobStatusID(inputname)
            print(f'checkJobStatusID: {resultJob}; while - {resultJob == "RUNNING"}')

        print('\nAnharmonic parallel job (main) has finished now')
        print('Other jobs were submitted through bash scripts\n')

    elif typecheck == 'name':
        basedirDIR = inputname.split('/')[-1]
        # print('basedirDIR', basedirDIR, 'basedir', basedir, basedir.split('/'))
        resultJobOther = checkJobStatusNAME(basedirDIR)

        while resultJobOther == 'RUNNING':
            time.sleep(30)
            resultJobOther = checkJobStatusNAME(basedirDIR)
            print(f'checkJobStatusNAME: {resultJobOther}; while - {resultJobOther == "RUNNING"}')

        print(resultJobOther)
        print('\nOther jobs are completed')

def process_fja(config: dict):
    machine = config['machine']
    minutes = config['minutes']
    hours = config['hours']
    nodes = config['nodes']
    tasks = 2  # config['tasks']
    cfourpath = config['c4path']

    import os
    os.chdir('./save')
    cwd = os.getcwd().split('/')

    dir_name = '/'.join([cwd[-2], cwd[-1]])
    print(dir_name)

    cpus = 16 if machine == 'fram' else 20

    text = rf"""#!/bin/bash
# ==================================================================================
# SLURM job scheduler for the OpenMP-parallel CFOUR program and {machine} supercomputer
# ==================================================================================
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tasks}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mail-type=ALL
#SBATCH --job-name={dir_name}
#SBATCH --account=nn14654k
#SBATCH --partition normal
#SBATCH --time={hours}:{minutes}:00

#load modules
module purge
module load gompi/2023a
module load imkl/2023.1.0
module load OpenMPI/4.1.5-GCC-12.3.0
module list

#set some local OMP-specific defaults
ulimit -s unlimited
export OMP_NUM_THREADS={cpus}
export OMP_STACKSIZE=4500m

#setup ReSpect installation directory
export CFOUR={cfourpath}

# add to path if not there
PATH=${{PATH}}:{cfourpath}

#files=$(ls fja.* | grep -E 'fja\.[0-9]{3}$')
files=$(find . -type f -name 'fja.*' | sed 's/.*fja\.\([0-9]*\).*/\\1 &/' | sort -n | cut -d' ' -f2-)
echo $files

#cp jaindx.save JAINDX
#cp jobarc.save JOBARC

# Assuming fja.x files are in the current directory
for file in $files; do
    if [ -f "$file" ]; then
        # Copy fja.x to FJOBARC
        cp "$file" FJOBARC

        # Execute xja2fja
        xja2fja

        # Execute xcubic and append output to out file
        xcubic >> out

        echo "Processed $file" >> loglog
    fi
done
"""

    with open("submit.sh", "w") as file1:
        # Writing data to a file
        file1.writelines(text)

    # make executable
    import subprocess
    subprocess.check_call(['chmod', '+x', "submit.sh"])
    sumbitSbatch("submit.sh")


def makeDisplacements(delta: float, config: dict):
    import os
    dircur = os.getcwd() + '/'
    print('dircur:', dircur)

    import shutil
    # molden file from anharmonic/hessian calculation, with normal modes
    shutil.copy('../anharm/MOLDEN', f'./MOLDEN_f')

    # import sys
    # # sys.path.append('/cluster/projects/nn14654k/vle014/scriptsHPC/utils')
    # sys.path.append('/home/vlew/scriptsHPC/utils')
    # import scriptsHPC.utils.parseCFOUR as pc4
    from scriptsHPC.utils import parseCFOUR

    # get normal modes
    equilibrium_geometry, atomsMolden, normal_modes = parseCFOUR.pMOLDEN('MOLDEN_f')
    mode_numbers = list(normal_modes.keys())
    print('mode_numbers', mode_numbers)
    # quit()

    # Reading from file - ZMAT equilibrium
    with open('ZMAT', "r+") as file1:
        # Reading form a file
        zmat_template = file1.readlines()
    indx = zmat_template.index("\n")
    zmat_template.insert(-3, 'COORD=CARTESIAN,UNITS=BOHR\n')

    zm = zmat_template[indx + 1:]
    if zmat_template[indx + 1][0] != '*':
        indx1 = zm.index("\n")
        zmat_template = zm[indx1 + 1:]
    else:
        zmat_template = zmat_template[indx + 1:]

    # Generate single displacements for each mode
    for mode_number0 in mode_numbers:
        if mode_number0 <= 6:
            continue
        # for displacement in [delta, -delta]:
        for displacement in [-delta, delta]:
            mode_coords_list = [normal_modes[mode_number0]]

            # create_zmat_file((atoms, equilibrium_geometry), zmat_template, (mode_number,), (displacement,),
            #                  mode_coords_list, config)
            mode_numbers1 = (mode_number0,)
            # Generate a string for the displacement description
            displacement_descriptions = []
            for mode_number, d in zip(mode_numbers1, (displacement,)):
                direction = 'POSITIVE' if d > 0 else 'NEGATIVE'
                displacement_descriptions.append(f"{direction} DISPLACEMENT of {d}*Q{mode_number}")
            displacement_str = ' and '.join(displacement_descriptions)
            d_str = ''.join(['p' if d > 0 else 'n' for d in (displacement,)])

            import os
            # Create a new directory
            new_directory = dircur + "_".join(map(str, mode_numbers1)) + d_str
            os.makedirs(new_directory, exist_ok=True)
            print(new_directory)
            # Generate the filename
            zmat_filename = dircur + f'zmat{"_".join(map(str, mode_numbers1))}{d_str}'
            print(zmat_filename)
            with open(zmat_filename, 'w') as file:
                file.write(f'GEOMETRY {"_".join(map(str, mode_numbers1))} {displacement_str}\n')
                atoms, eq_coords = atomsMolden, equilibrium_geometry

                for i in range(len(atoms)):
                    line = [atoms[i]]
                    # Calculate the total displacement for this atom
                    displace = [sum(d * mode_coord[i][j] for d, mode_coord in zip((displacement,), mode_coords_list))
                                for
                                j in range(3)]
                    line.extend(["{:.10f}".format(c + disp) for c, disp in zip(eq_coords[i], displace)])
                    file.write(' '.join(line) + '\n')

                file.write('\n')
                file.write(''.join(zmat_template))

            shutil.copy(zmat_filename, new_directory + '/ZMAT')
            os.chdir(new_directory)
            generateSubmitPy(config, 'submit.sh')
            #sumbitSbatch("submit.sh")
            os.chdir('../')

    # Generate double displacements for pairs of modes
    for i, mode_number2 in enumerate(mode_numbers):
        #print('i, mode_number2 <= 6', i, mode_number2, mode_number2 <= 6)
        #if mode_number2 <= 6:
        #    continue
        for j, mode_number3 in enumerate(mode_numbers):
            #print(f"i: {i}, j: {j}, mode_number2: {mode_number2}, mode_number3: {mode_number3}, j <= i")
            if mode_number3 <= mode_number2:
                continue
            # Create displacement combinations for two different modes
            displacement_combinations = [
                (delta, delta), (delta, -delta),
                (-delta, delta), (-delta, -delta)
            ]
            for displacements in displacement_combinations:
                mode_coords_list = [normal_modes[mode_number2], normal_modes[mode_number3]]

                # create_zmat_file((atoms, equilibrium_geometry), zmat_template, (mode_number1, mode_number2),
                #                  displacements, mode_coords_list, config)

                mode_numbers2 = (mode_number2, mode_number3)
                # Generate a string for the displacement description
                displacement_descriptions = []
                for mode_number, d in zip(mode_numbers2, displacements):
                    direction = 'POSITIVE' if d > 0 else 'NEGATIVE'
                    displacement_descriptions.append(f"{direction} DISPLACEMENT of {d}*Q{mode_number}")
                displacement_str = ' and '.join(displacement_descriptions)
                d_str = ''.join(['p' if d > 0 else 'n' for d in displacements])

                import os
                # Create a new directory
                new_directory = dircur + "_".join(map(str, mode_numbers2)) + d_str
                os.makedirs(new_directory, exist_ok=True)

                # Generate the filename
                zmat_filename = dircur + f'zmat{"_".join(map(str, mode_numbers2))}{d_str}'

                with open(zmat_filename, 'w') as file:
                    file.write(f'GEOMETRY {"_".join(map(str, mode_numbers2))} {displacement_str} - regular\n')
                    atoms, eq_coords = atomsMolden, equilibrium_geometry

                    for i in range(len(atoms)):
                        line = [atoms[i]]
                        # Calculate the total displacement for this atom
                        displace = [
                            sum(d * mode_coord[i][j] for d, mode_coord in zip(displacements, mode_coords_list)) for
                            j in range(3)]
                        line.extend(["{:.10f}".format(c + disp) for c, disp in zip(eq_coords[i], displace)])
                        file.write(' '.join(line) + '\n')

                    file.write('\n')
                    file.write(''.join(zmat_template))

                shutil.copy(zmat_filename, new_directory + '/ZMAT')
                os.chdir(new_directory)
                generateSubmitPy(config, 'submit.sh')
                #sumbitSbatch("submit.sh")
                os.chdir('../')

def makeDisplacements_Dimless(delta: float, config: dict):
    import os
    dircur = os.getcwd() + '/'
    print('dircur:', dircur)

    import shutil
    # molden file from anharmonic/hessian calculation, with normal modes
    shutil.copy('../anharm/QUADRATURE', f'./QUADRATURE_f')
    shutil.copy('../anharm/MOLDEN', f'./MOLDEN_f')

    # import sys
    # # sys.path.append('/cluster/projects/nn14654k/vle014/scriptsHPC/utils')
    # sys.path.append('/home/vlew/scriptsHPC/utils')
    # import scriptsHPC.utils.parseCFOUR as pc4
    from scriptsHPC.utils import parseCFOUR

    # get normal modes
    equilibrium_geometry00, atomsMolden, normal_modes00 = parseCFOUR.pMOLDEN('MOLDEN_f')
    equilibrium_geometry, freqs, normal_modes = parseCFOUR.pQUADRATURE('./QUADRATURE_f')

    mode_numbers = list(normal_modes.keys())
    #print('mode_numbers', mode_numbers)
    # quit()

    # Reading from file - ZMAT equilibrium
    with open('ZMAT', "r+") as file1:
        # Reading form a file
        zmat_template = file1.readlines()
    indx = zmat_template.index("\n")
    zmat_template.insert(-3, 'COORD=CARTESIAN,UNITS=BOHR\n')

    zm = zmat_template[indx + 1:]
    if zmat_template[indx + 1][0] != '*':
        indx1 = zm.index("\n")
        zmat_template = zm[indx1 + 1:]
    else:
        zmat_template = zmat_template[indx + 1:]

    # Generate single displacements for each mode
    for mode_number0 in mode_numbers:
        if mode_number0 <= 6:
            continue
        # for displacement in [delta, -delta]:
        for displacement in [-delta, delta]:
            mode_coords_list = [normal_modes[mode_number0]]

            # create_zmat_file((atoms, equilibrium_geometry), zmat_template, (mode_number,), (displacement,),
            #                  mode_coords_list, config)
            mode_numbers1 = (mode_number0,)
            # Generate a string for the displacement description
            displacement_descriptions = []
            for mode_number, d in zip(mode_numbers1, (displacement,)):
                direction = 'POSITIVE' if d > 0 else 'NEGATIVE'
                displacement_descriptions.append(f"{direction} DISPLACEMENT of {d}*Q{mode_number}")
            displacement_str = ' and '.join(displacement_descriptions)
            d_str = ''.join(['p' if d > 0 else 'n' for d in (displacement,)])

            import os
            # Create a new directory
            new_directory = dircur + "_".join(map(str, mode_numbers1)) + d_str
            os.makedirs(new_directory, exist_ok=True)
            print(new_directory)
            # Generate the filename
            zmat_filename = dircur + f'zmat{"_".join(map(str, mode_numbers1))}{d_str}'
            print(zmat_filename)
            with open(zmat_filename, 'w') as file:
                file.write(f'GEOMETRY {"_".join(map(str, mode_numbers1))} {displacement_str} - dimensionless\n')
                atoms, eq_coords = atomsMolden, equilibrium_geometry

                for i in range(len(atoms)):
                    line = [atoms[i]]
                    # Calculate the total displacement for this atom
                    displace = [sum(d * mode_coord[i][j] for d, mode_coord in zip((displacement,), mode_coords_list))
                                for
                                j in range(3)]
                    line.extend(["{:.10f}".format(c + disp) for c, disp in zip(eq_coords[i], displace)])
                    file.write(' '.join(line) + '\n')

                file.write('\n')
                file.write(''.join(zmat_template))

            shutil.copy(zmat_filename, new_directory + '/ZMAT')
            os.chdir(new_directory)
            generateSubmitPy(config, 'submit.sh')
            sumbitSbatch("submit.sh")
            os.chdir('../')

    # Generate double displacements for pairs of modes
    for i, mode_number2 in enumerate(mode_numbers):
        #print('i, mode_number2 <= 6', i, mode_number2, mode_number2 <= 6)
        #if mode_number2 <= 6:
        #    continue
        for j, mode_number3 in enumerate(mode_numbers):
            #print(f"i: {i}, j: {j}, mode_number2: {mode_number2}, mode_number3: {mode_number3}, j <= i")
            if mode_number3 <= mode_number2:
                continue
            # Create displacement combinations for two different modes
            displacement_combinations = [
                (delta, delta), (delta, -delta),
                (-delta, delta), (-delta, -delta)
            ]
            for displacements in displacement_combinations:
                mode_coords_list = [normal_modes[mode_number2], normal_modes[mode_number3]]

                # create_zmat_file((atoms, equilibrium_geometry), zmat_template, (mode_number1, mode_number2),
                #                  displacements, mode_coords_list, config)

                mode_numbers2 = (mode_number2, mode_number3)
                # Generate a string for the displacement description
                displacement_descriptions = []
                for mode_number, d in zip(mode_numbers2, displacements):
                    direction = 'POSITIVE' if d > 0 else 'NEGATIVE'
                    displacement_descriptions.append(f"{direction} DISPLACEMENT of {d}*Q{mode_number}")
                displacement_str = ' and '.join(displacement_descriptions)
                d_str = ''.join(['p' if d > 0 else 'n' for d in displacements])

                import os
                # Create a new directory
                new_directory = dircur + "_".join(map(str, mode_numbers2)) + d_str
                os.makedirs(new_directory, exist_ok=True)

                # Generate the filename
                zmat_filename = dircur + f'zmat{"_".join(map(str, mode_numbers2))}{d_str}'

                with open(zmat_filename, 'w') as file:
                    file.write(f'GEOMETRY {"_".join(map(str, mode_numbers2))} {displacement_str} - dimensionless\n')
                    atoms, eq_coords = atomsMolden, equilibrium_geometry

                    for i in range(len(atoms)):
                        line = [atoms[i]]
                        # Calculate the total displacement for this atom
                        displace = [
                            sum(d * mode_coord[i][j] for d, mode_coord in zip(displacements, mode_coords_list)) for
                            j in range(3)]
                        line.extend(["{:.10f}".format(c + disp) for c, disp in zip(eq_coords[i], displace)])
                        file.write(' '.join(line) + '\n')

                    file.write('\n')
                    file.write(''.join(zmat_template))

                shutil.copy(zmat_filename, new_directory + '/ZMAT')
                os.chdir(new_directory)
                generateSubmitPy(config, 'submit.sh')
                sumbitSbatch("submit.sh")
                os.chdir('../')

def process_fja_files():
    import subprocess
    import os
    import glob

    import sys
    import re
    # Function to run a shell command and print the output
    def run_command(command):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running: {' '.join(command)}", file=sys.stderr)
            print(e.stderr, file=sys.stderr)

    # Function to load modules (assuming you have a module function in your environment)
    def load_module(module_name):
        run_command(['module', 'load', module_name])

    # Function to unload all modules
    def purge_modules():
        run_command(['module', 'purge'])

    # Function to list all loaded modules
    def list_modules():
        run_command(['module', 'list'])

    # Load modules
    purge_modules()
    load_module('gompi/2023a')
    load_module('imkl/2023.1.0')
    load_module('OpenMPI/4.1.5-GCC-12.3.0')
    list_modules()

    # Set some local OMP-specific defaults
    # os.environ['OMP_NUM_THREADS'] = '16'
    # os.environ['OMP_STACKSIZE'] = '4500m'

    # Setup CFOUR installation directory
    cfour_dir = '/cluster/projects/nn14654k/vle014/cfour_serial/bin'
    os.environ['CFOUR'] = cfour_dir

    # Add CFOUR to path if not there
    if cfour_dir not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep + cfour_dir

    # Find fja.x files in the current directory
    files = [f for f in glob.glob('fja.*') if re.match(r'fja\.\d{3}$', f)]
    print(files)

    # Process each file
    for file in files:
        if os.path.isfile(file):
            # Copy fja.x to FJOBARC
            subprocess.run(['cp', file, 'FJOBARC'])

            # Execute xja2fja
            run_command(['xja2fja'])

            # Execute xcubic and append output to out file
            with open('out', 'a') as outfile:
                subprocess.run(['xcubic'], stdout=outfile)

            # Log the processed file
            with open('loglog', 'a') as logfile:
                logfile.write(f"Processed {file}\n")
