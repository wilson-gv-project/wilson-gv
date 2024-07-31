#!/usr/bin/env python
import os
import sys
import subprocess

def run_sbatch(directory):
    # Print the current directory before changing
    print(f"Current directory before changing: {os.getcwd()}")

    # Save the current directory
    original_dir = os.getcwd()

    # Change to the target directory
    try:
        os.chdir(directory)
    except Exception as e:
        print(f"Error changing to directory {directory}: {e}")
        return

    # Print the current directory after changing
    print(f"Current directory after changing: {os.getcwd()}")

    # Define the filenames to look for
    filenames = ["submit.sh", "submitpy.sh"]

    # Check each file and run sbatch if the file exists
    for filename in filenames:
        if os.path.isfile(filename):
            # Run the sbatch command using subprocess
            command = f"sbatch {filename}"
            try:
                subprocess.run(command, check=True, shell=True)
                print(f"Successfully ran: {command}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to run {command}: {e}")
            break  # Exit after running sbatch for the first found file
    else:
        # If no files are found
        print(f"No submit.sh or submitpy.sh found in {directory}")

    # Change back to the original directory
    os.chdir(original_dir)
    print(f"Returned to original directory: {os.getcwd()}")

if __name__ == "__main__":
    # Check if directories are provided as command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory1> <directory2> ...")
        sys.exit(1)

    # Iterate over each directory provided as an argument
    for dir_arg in sys.argv[1:]:
        run_sbatch(dir_arg)
