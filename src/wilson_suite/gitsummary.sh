#!/usr/bin/env bash
# chatgpt, tested

# Colors
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[1;34m"
CYAN="\033[0;36m"
RED="\033[0;31m"
BOLD="\033[1m"
RESET="\033[0m"
MAGENTA="\033[0;35m"

# Directory containing Git repositories - here it should be wilson-suite
PARENT_DIR="$(pwd)"

# Expand path and go to the parent directory
PARENT_DIR=$(eval echo "$PARENT_DIR")
cd "$PARENT_DIR" || exit

# info for parent dir

# Loop through each subdirectory
for dir in */; do
  if [ -d "$dir/.git" ]; then
    echo -e "${BOLD}${BLUE}===== Repository: $dir =====${RESET}"
    cd "$dir" || continue

    # Current branch
    branch=$(git rev-parse --abbrev-ref HEAD)

    # Status
    status=$(git status --short)

    # Last commit
    last_commit=$(git log -1 --pretty=format:"%h ${MAGENTA}%an${RESET} %ad: %s" --date=local)

    echo -e "Branch: ${YELLOW}$branch${RESET}"
    echo -e "Last commit: $last_commit"

    if [ -z "$status" ]; then
      echo -e "Status: ${GREEN}clean${RESET}"
      echo -e "${CYAN}Fetching updates from all remotes...${RESET}"
      git fetch --all
      status_summary=$(git status -sb)
      if echo "$status_summary" | grep -q "\[.*behind"; then
        echo -e "${YELLOW}📥 Remote has new commits. Consider pulling.${RESET}"
      else
        echo -e "${GREEN}Already up to date with remote.${RESET}"
      fi
    else
      echo -e "Status: ${RED}changes pending${RESET}"
      echo -e "$status"
    fi

    echo ""  # Empty line for readability
    cd ..
  fi
done

