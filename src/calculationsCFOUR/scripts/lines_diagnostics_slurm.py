#!/usr/bin/env python
import pandas as pd
import glob
import os

from colorama import init, Fore, Style

init(autoreset=True)

def get_color_based_on_path(file_path):
    if 'anharm' in file_path:
        return Fore.RED
    elif '/polar' in file_path:
        return Fore.BLUE
    return Fore.RESET

def print_colored_text(text, file_path):
    color = get_color_based_on_path(file_path)
    print(color + text)


def parse_slurm_output(slurm_output):
    results = {
        "Req Wallclock": None,
        "Elapsed Wallclock": None,
        "CPU Used": None,
        "CPU Unused": None,
        "Mem Alloc": None,
        "Mem Used": None,
        "Bill Hours": None,
        "Error": None
    }

    lines = iter(slurm_output.splitlines())
    for line in lines:
        if "consumed" in line and "billing hours" in line:
            parts = line.split()
            results["Bill Hours"] = float(parts[3])
        elif "Requested wallclock time:" in line:
            parts = line.split(':')
            time_parts = parts[1].strip().rsplit(' ', 1)  # Split from the right to get the last part as the unit
            time_value, time_unit = time_parts[0], time_parts[1]
            time_unit = time_unit.replace("hours", "h").replace("minutes", "m").replace("seconds", "s")
            results["Req Wallclock"] = f"{time_value} {time_unit}"
        elif "Elapsed wallclock time:" in line:
            parts = line.split(':')
            time_parts = parts[1].strip().rsplit(' ', 1)  # Split from the right to get the last part as the unit
            time_value, time_unit = time_parts[0], time_parts[1]
            time_unit = time_unit.replace("hours", "h").replace("minutes", "m").replace("seconds", "s")
            results["Elapsed Wallclock"] = f"{time_value} {time_unit}"
        elif "Used CPU time:" in line:
            parts = line.split(':')
            time_parts = parts[1].strip().rsplit(' ', 2)  # Split from the right to get the last part as the unit
            time_value, time_unit = time_parts[0], time_parts[2]
            time_unit = time_unit.replace("seconds", "s").replace("minutes", "m").replace("hours", "h").replace("days", "d")
            results["CPU Used"] = f"{time_value} {time_unit}"
        elif "Unused CPU time:" in line:
            parts = line.split(':')
            time_parts = parts[1].strip().rsplit(' ', 2)  # Split from the right to get the last part as the unit
            time_value, time_unit = time_parts[0], time_parts[2]
            time_unit = time_unit.replace("seconds", "s").replace("minutes", "m").replace("hours", "h").replace("days", "d")
            results["CPU Unused"] = f"{time_value} {time_unit}"
        elif "Memory statistics, in GiB:" in line:
            next(lines)  # Skip header line
            next(lines)
            alloc_line = next(lines)  # First input_data_info line
            parts = alloc_line.split()
            results["Mem Alloc"] = float(parts[1])
            if len(parts) > 2:
                results["Mem Used"] = float(parts[2])
        elif "CANCELLED AT" in line:
            parts = line.split('DUE TO')
            results["Error"] = parts[1].strip(' *')

    return results


import re

def check_pattern_in_file(file_path):
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
            # Search for the pattern in the file content
            if re.search(pattern, content, re.MULTILINE):
                return True
            else:
                return False
    except FileNotFoundError:
        print("File not found.")
        return False

def collect_data_from_files():
    # Search for all slurm-*.out files in subdirectories
    slurm_file_paths = glob.glob('**/slurm-*.out', recursive=True)
    outfile_file_paths = glob.glob('**/outfile*.out', recursive=True)

    max_slurm_files = {}
    max_outfile_files = {}

    # Determine the file with the highest number in each directory for slurm files
    for file_path in slurm_file_paths:
        directory = os.path.dirname(file_path)
        file_number = int(os.path.basename(file_path).split('-')[1].split('.')[0])
        if directory not in max_slurm_files or file_number > max_slurm_files[directory][0]:
            max_slurm_files[directory] = (file_number, file_path)

    # Determine the file with the highest number in each directory for outfile files
    for file_path in outfile_file_paths:
        directory = os.path.dirname(file_path)
        file_number = int(os.path.basename(file_path).split('outfile')[1].split('.')[0])
        if directory not in max_outfile_files or file_number > max_outfile_files[directory][0]:
            max_outfile_files[directory] = (file_number, file_path)

    data = []

    # Process only the files with the highest number in their respective directories
    for directory, (file_number, file_path) in max_slurm_files.items():
        with open(file_path, 'r') as file:
            content = file.read()
            parsed_data = parse_slurm_output(content)
            parsed_data["File Path"] = file_path
            #print(directory)
            # Check corresponding outfile if it exists
            if directory in max_outfile_files: # and max_outfile_files[directory][0] == file_number:
                outfile_path = max_outfile_files[directory][1]
                with open(outfile_path, 'r') as outfile:
                    outfile_content = outfile.readlines()
                    parsed_data["Lines"] = len(outfile_content)
                    parsed_data["Status"] = "The final electronic energy is" in ''.join(outfile_content)
                    
                    #outfile_content = outfile.read()
                    #if "The final electronic energy is" in outfile_content:
                    #    parsed_data["Status"] = True
                    #else:
                    #    parsed_data["Status"] = False
            elif directory[-4:] == 'save':
                outfile_path = directory+'/out'
                if os.path.isfile(outfile_path):
                    #parsed_data["Status"] = check_pattern_in_file(outfile_path)
                    with open(outfile_path, 'r') as outfile:
                        outfile_content = outfile.readlines()
                        parsed_data["Lines"] = len(outfile_content)
                        parsed_data["Status"] = check_pattern_in_file(outfile_path)

            data.append(parsed_data)

    # Create a DataFrame
    df = pd.DataFrame(data)
    return df


def color_file_path(val):
    """ Color file paths based on keywords. """
    if 'anharm' in val:
        return '\x1b[31m' + val + '\x1b[0m'  # Red for '/anharm'
    elif '/polar' in val:
        return '\x1b[34m' + val + '\x1b[0m'  # Blue for '/polar'
    return val  # No color

def color_file_path(val):
    """ Color file paths based on keywords. """
    if 'anharm' in val:
        return '\x1b[38;5;208m' + val + '\x1b[0m'  # Orange for '/anharm'
    elif '/polar' in val:
        return '\x1b[34m' + val + '\x1b[0m'  # blue for '/polar'
    return val  # No color


def remove_ansi_codes(text):
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)

def print_dataframe_with_alignment(df):
    # Calculate the maximum width for each column considering ANSI codes
    column_widths = {}
    for column in df.columns:
        max_width = max(len(remove_ansi_codes(str(x))) for x in df[column])
        column_widths[column] = max_width

    # Print column headers
    header = " | ".join(f"{col.ljust(column_widths[col])}" for col in df.columns)
    print(header)
    print("-" * len(header))

    # Print rows with proper alignment, considering ANSI codes
    for _, row in df.iterrows():
        formatted_row = " | ".join(f"{str(row[col]).ljust(column_widths[col] + len(str(row[col])) - len(remove_ansi_codes(str(row[col]))))}" for col in df.columns)
        print(formatted_row)

# Collect input_data_info and display the DataFrame
df = collect_data_from_files()

df.style.hide()
df = df.sort_values(by=['Status', 'Elapsed Wallclock'], na_position='last', ascending=False)
#print(df)

with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):  # more options can be specified also
    #print(df)
    # Select the ones you want
    df1 = df[['Req Wallclock', 'Elapsed Wallclock','File Path', 'Status', 'Lines', 'Error']]
    df2 = df1[~df['File Path'].str.contains('coh2aldehyde/')]
    df2.loc[:, 'File Path'] = df2['File Path'].apply(color_file_path)
    #print(df1.to_string(index=False))
    print_dataframe_with_alignment(df2)
    print('Number of rows:', len(df1.index))
