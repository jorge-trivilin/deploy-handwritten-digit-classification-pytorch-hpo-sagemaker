#!/bin/bash

# Obtém o nome da branch atual
current_branch=$(git rev-parse --abbrev-ref HEAD)

git commit --allow-empty -m "Trigger GitHub Actions workflow -terraform"

# Faz o push para a branch atual
git push origin "$current_branch"
