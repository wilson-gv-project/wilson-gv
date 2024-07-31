#!/bin/bash

# Find directories containing .pkl files and store them in an array
mapfile -t dirs < <(find . -type f -name "*.pkl" -exec dirname {} \; | sort -u)

# Print the root and then each directory path as a tree
echo "├── ."
for dir in "${dirs[@]}"; do
    IFS='/' read -ra ADDR <<< "$dir"
    prefix="│   "
    for ((j=1; j<${#ADDR[@]}; j++)); do  # Skip the first element since it's just "."
        if [[ $j -eq ${#ADDR[@]}-1 ]]; then
            echo "$prefix└── ${ADDR[j]}"
        else
            echo "$prefix├── ${ADDR[j]}"
            prefix+="│   "
        fi
    done
done
