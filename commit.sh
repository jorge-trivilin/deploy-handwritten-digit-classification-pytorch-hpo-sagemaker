#!/bin/bash

# Obtém o nome da branch atual
current_branch=$(git rev-parse --abbrev-ref HEAD)

git add .

git commit -m "Pipeline update"

git push origin "$current_branch"