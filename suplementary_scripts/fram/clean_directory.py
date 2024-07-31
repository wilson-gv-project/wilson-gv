#!/usr/bin/env python
import os
import sys
import subprocess
import glob

def clean_directory(directory):
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

    # Define patterns to keep
    if 'polar' in directory:
        keep_patterns = ["ZMAT", "submit.sh", "submitpy.sh", "outfile0.out", "POLAR", "slurm-*"]
    elif 'anharm' in directory:
        keep_patterns = ["ZMAT", "submit.sh", "submitpy.sh", "outfile0.out", "slurm-*", "out", "cubic", 
                "dipolex", "dipoley", "dipolez", "quartic", "FJOBARC", "out1", "corioliszeta", "coriolis"]
    keep_files = []

    # Gather files matching keep patterns
    for pattern in keep_patterns:
        keep_files.extend(glob.glob(pattern))

    # List all files and directories
    all_files = os.listdir()

    # Determine files to remove (those not matching keep patterns)
    files_to_remove = [file for file in all_files if file not in keep_files]

    # Remove files
    for file in files_to_remove:
        try:
            if os.path.isfile(file) or os.path.islink(file):
                os.unlink(file)
                print(f"Removed file: {file}")
            elif os.path.isdir(file):
                os.rmdir(file)
                print(f"Removed directory: {file}")
        except Exception as e:
            print(f"Failed to remove {file}: {e}")

    # Change back to the original directory
    os.chdir(original_dir)
    print(f"Returned to original directory: {os.getcwd()}")

#if __name__ == "__main__":
#    # Check if directories are provided as command-line arguments
#    if len(sys.argv) < 2:
#        print("Usage: python script.py <directory1> <directory2> ...")
#        sys.exit(1)
#
#    # Iterate over each directory provided as an argument
#    for dir_arg in sys.argv[1:]:
#        clean_directory(dir_arg)

def clean_all_subdirectories(root_directory):
    for subdir in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, subdir)
        if os.path.isdir(subdir_path):
            print(f"Cleaning directory: {subdir_path}")
            clean_directory(subdir_path)

if __name__ == "__main__":
    # Check if a root directory is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python script.py <root_directory>")
        sys.exit(1)

    # Get the root directory from the arguments
    root_directory = sys.argv[1]
    clean_all_subdirectories(root_directory)
