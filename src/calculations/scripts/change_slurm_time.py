#!/usr/bin/env python
import os
import sys
import glob
import subprocess
import re

def main(directory, hours, submit_job=False):
    if not os.path.isdir(directory):
        print(f"Error: The directory {directory} does not exist.")
        return

    os.chdir(directory)

    # List of files to keep
    keep_files = {"ZMAT", "GENBAS", "submit.sh", "submitpy.sh"}

    # Remove other files
    for item in os.listdir('.'):
        if item not in keep_files:
            if os.path.isdir(item):
                subprocess.run(['rm', '-rv', item], check=True)
            else:
                os.remove(item)
            print(f"Removed: {item}")

    submit_file = 'submit.sh' if os.path.exists('submit.sh') else 'submitpy.sh' if os.path.exists('submitpy.sh') else None
    if submit_file:
        with open(submit_file, 'r') as file:
            lines = file.readlines()

        with open(submit_file, 'w') as file:
            for line in lines:
                if '#SBATCH --time=' in line:
                    new_line = re.sub(r'(?<=#SBATCH --time=)\d+', str(hours), line)
                    file.write(new_line)
                else:
                    file.write(line)

        print(f"Updated {submit_file} with new time.")
    
    if submit_job:
        if submit_file:
            result = subprocess.run(['sbatch', submit_file], capture_output=True, text=True)
            print(f"Executed sbatch for {submit_file}:")
            print(result.stdout)
        else:
            print("No submit file found to execute.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <directory_path> <hours>")
    else:
        directory_path = sys.argv[1]
        hours = int(sys.argv[2])
        main(directory_path, hours, True)
