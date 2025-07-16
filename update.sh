#!/usr/bin/env bash
# This script updates local git repositories using a repo_config.txt file.
# repo_config.txt is an example here. Made using chatgpt

set -e

echo ""
echo "📦 Updating Wilson Suite repositories from repo_config.txt..."

BASE_DIR=$(pwd)
CONFIG_FILE="repo_config.txt"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "❌ Configuration file '$CONFIG_FILE' not found!"
  exit 1
fi

while IFS= read -r line || [[ -n "$line" ]]; do
  # Skip empty lines and comments
  [[ -z "$line" || "$line" == \#* ]] && continue

  # Split line
  url_and_branch="${line% *}"
  dest="${line#* }"

  # Extract repo and branch@commit
  repo="${url_and_branch%:*}"
  branch_and_commit="${url_and_branch##*:}"

  # Check if there's a @commit
  if [[ "$branch_and_commit" == *@* ]]; then
    branch="${branch_and_commit%@*}"
    commit="${branch_and_commit#*@}"
  else
    branch="$branch_and_commit"
    commit=""
  fi

  echo ""
  echo "👉 Repository: $repo"
  echo "📂 Local dir: $dest"
  echo "🌿 Branch: $branch"
  [[ -n "$commit" ]] && echo "🔒 Commit: $commit"

  if [ ! -d "$dest" ]; then
    echo "📥 Cloning..."
    git clone --branch "$branch" "$repo" "$dest"
    cd "$dest"
  else
    echo "🔍 $dest already exists. Checking out correct branch..."
    cd "$dest"
    git fetch origin
    if git rev-parse --verify "$branch" >/dev/null 2>&1; then
      git checkout "$branch"
      git pull origin "$branch"
    else
      echo "❗ Branch $branch not found locally, trying origin..."
      git checkout -b "$branch" "origin/$branch"
    fi
  fi

  # Optional commit checkout
  if [[ -n "$commit" ]]; then
    git checkout "$commit"
  fi

  cd "$BASE_DIR"
done < "$CONFIG_FILE"

echo ""
echo "✅ All repositories are up to date!"
