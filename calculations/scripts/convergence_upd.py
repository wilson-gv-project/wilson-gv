#!/usr/bin/env python
import argparse
import re


def update_zmat_file(file_path, parameter, value):
    # Define the pattern to find the parameter and its value
    pattern = re.compile(rf'^({parameter}=)(\d+)', re.MULTILINE)
    
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Replace the old value with the new value using a lambda to correctly handle the backreference
    new_content, count = re.subn(pattern, lambda m: f"{m.group(1)}{value}", content)
    
    if count == 0:
        print(f"Parameter '{parameter}' not found.")
        return
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(new_content)
    
    print(f"Updated '{parameter}' to {value} in {file_path}.")


def main():
    parser = argparse.ArgumentParser(description="Update parameters in a ZMAT file.")
    parser.add_argument("file_path", type=str, help="Path to the ZMAT file")
    parser.add_argument("parameter", type=str, help="Parameter to update (e.g., GEO_CONV, CC_CONV)")
    parser.add_argument("value", type=int, help="New integer value for the parameter")

    args = parser.parse_args()

    update_zmat_file(args.file_path, args.parameter, args.value)


if __name__ == "__main__":
    main()
