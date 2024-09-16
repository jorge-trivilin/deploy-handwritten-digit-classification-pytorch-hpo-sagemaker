#!/bin/bash

# Gets the current branch
current_branch=$(git rev-parse --abbrev-ref HEAD)

git commit --allow-empty -m "Trigger GitHub Actions workflow -terraform"

# Pushes to the current branch
git push origin "$current_branch"
