#!/bin/bash

# Find directories containing .pkl files and store them in an array
mapfile -t dirs < <(find . -type f -name "*.pkl" -exec dirname {} \; | sort -u)

# Function to print directory in a tree-like format
print_tree() {
    local parent=$1
    local prefix=$2
    local last_prefix=$3
    local children=()

    # Gather all directories that are direct children of the current parent
    for dir in "${dirs[@]}"; do
        if [[ $dir != $parent && $dir =~ ^$parent/[^/]+$ ]]; then
            children+=("$dir")
        fi
    done

    # Print each child and recursively print its children
    local count=${#children[@]}
    for ((i=0; i<count; i++)); do
        local child=${children[i]}
        local is_last=$((i == count-1))

        if [[ $is_last -eq 1 ]]; then
            echo -e "${prefix}└── ${child#$parent/}"
            print_tree "$child" "$last_prefix    " "$last_prefix    "
        else
            echo -e "${prefix}├── ${child#$parent/}"
            print_tree "$child" "$prefix│   " "$last_prefix    "
        fi
    done
}

# Print the root and then each directory path as a tree
echo "├── ."
for dir in "${dirs[@]}"; do
    IFS='/' read -ra ADDR <<< "$dir"
    local path=""
    local prefix="│   "
    for i in "${ADDR[@]:1}"; do  # Skip the first element since it's just "."
        path+="/$i"
        # Check if this path is the last segment
        if [[ "$path" == "$dir" ]]; then
            echo "$prefix└── $i"
        else
            echo "$prefix├── $i"
            prefix+="    "
        fi
    done
done
