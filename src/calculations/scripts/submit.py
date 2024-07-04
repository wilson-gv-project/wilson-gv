#!/usr/bin/env python
# ==================================================================================
# SLURM job scheduler for the OpenMP-parallel CFOUR program and FRAM supercomputer
# ==================================================================================
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --job-name=cfourscripts
#SBATCH --account=nn14654k
#SBATCH --partition normal
#SBATCH --time=00:10:00

import os
import subprocess
import datetime
import glob

# SLURM job parameters
job_params = {
    'nodes': 1,
    'ntasks_per_node': 2,
    'cpus_per_task': 16,
    'mail_type': 'ALL',
    'job_name': 'cfourscripts',
    'account': 'nn14654k',
    'partition': 'normal',
    'time': '00:10:00'
}

# Set environment variables
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['OMP_STACKSIZE'] = '4500m'

# Load modules
subprocess.run(['module', 'purge'])
subprocess.run(['module', 'load', 'gompi/2023a'])
subprocess.run(['module', 'load', 'imkl/2023.1.0'])
subprocess.run(['module', 'load', 'OpenMPI/4.1.5-GCC-12.3.0'])
subprocess.run(['module', 'list'])

# Setup the scratch directory
slurm_submit_dir = os.environ.get('SLURM_SUBMIT_DIR', os.getcwd())
os.chdir(slurm_submit_dir)

# Setup ReSpect installation directory
cfour_bin = '/cluster/projects/nn14654k/vle014/cfour_serial/bin'
os.environ['CFOUR'] = cfour_bin
os.environ['PATH'] += f":{cfour_bin}"

# Function to log job details
def log_job_details(job_id, submit_time, job_name, user, queue, current_directory):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cfour_grep = subprocess.check_output(['grep', 'CFOUR(', 'ZMAT']).decode().strip()
    with open('ZMAT', 'r') as zmat_file:
        first_line = zmat_file.readline().strip()

    job_info = f"\033[0;32m{job_id}\033[0m,{job_name},{user},{queue},\033[0;31m{current_directory}\033[0m,{submit_time},{cfour_grep},{first_line}"

    with open('../../job_log.csv', 'a') as log_file:
        log_file.write(f"{job_info}\n")

# Capture job details
job_id = os.environ.get('SLURM_JOB_ID', 'Unknown')
job_name = job_params['job_name']
user = os.environ.get('USER', 'Unknown')
queue = os.environ.get('SLURM_QUEUE', 'Unknown')
current_directory = os.path.basename(slurm_submit_dir)
submit_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Log job details
log_job_details(job_id, submit_time, job_name, user, queue, current_directory)

# Execute CFOUR and handle output files
outfile = "outfile"
extension = ".out"
counter = 0

while os.path.exists(f"{outfile}{counter}{extension}"):
    counter += 1

output_filename = f"{outfile}{counter}{extension}"

# Run CFOUR command
# subprocess.run(['xcfour'], stdout=open(output_filename, 'w'))

# Create save directory and copy files
os.makedirs('save', exist_ok=True)
subprocess.run(['cp', 'JOBARC', './save/'])
subprocess.run(['cp', 'JAINDX', './save/'])

# Find all zmat0* files in the current directory
zmat_files = glob.glob('zmat0*')

for file in zmat_files:
    dir_name = file[6:]
    os.makedirs(dir_name, exist_ok=True)
    subprocess.run(['cp', file, f"{dir_name}/ZMAT"])
    subprocess.run(['cp', '/cluster/projects/nn14654k/vle014/cfour_serial/basis/GENBAS', dir_name])

    print(f"Created directory '{dir_name}' and copied '{file}' into '{dir_name}/ZMAT'")

    # Define arguments for script2.sh
    hours = '00'
    minutes = '10'
    cpus = '16'
    tasks = '2'
    nodes = '1'

    os.chdir(dir_name)
    # Call the script2.sh equivalent in Python
    # subprocess.run(['/cluster/projects/nn14654k/vle014/scriptsHPC/submit_utils/scrmaster_fram', '-h', hours, '-m', minutes, '-n', nodes, '-t', tasks, '-c', cpus])

    # Modify the submit.sh file
    with open('submit.sh', 'a') as submit_file:
