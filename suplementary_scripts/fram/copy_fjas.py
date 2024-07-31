#!/usr/bin/env python
import os
import shutil

source_dir = "."
destination_dir = "./save"

#os.makedirs(destination_dir, exist_ok=True)
if not os.path.exists(destination_dir):
    print(f"Destination directory {destination_dir} does not exist.")
    sys.exit(1)

subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(d) and d.isdigit()]
print(subdirs)

for subdir in sorted(subdirs, key=int):
    source_file = os.path.join(source_dir, subdir, "FJOBARC")
    destination_file = os.path.join(destination_dir, f"fja.{subdir.zfill(3)}")
    
    print(source_file, destination_file)
    if os.path.exists(source_file):
        shutil.copy2(source_file, destination_file)
        print(f"Copied {source_file} to {destination_file}")
    else:
        print(f"File {source_file} does not exist")

