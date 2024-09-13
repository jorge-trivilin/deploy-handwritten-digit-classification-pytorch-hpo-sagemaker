#!/bin/bash

# Obtém o nome da branch atual
current_branch=$(git rev-parse --abbrev-ref HEAD)

git pull origin "$current_branch"

git commit --allow-empty "Trigger Github Actions workflow -training"

# Faz o push para a branch atual
git push origin "$current_branch"
