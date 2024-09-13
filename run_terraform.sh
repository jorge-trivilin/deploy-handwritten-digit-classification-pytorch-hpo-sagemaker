#!/bin/bash

# Adiciona todas as mudanças
git add .

# Faz o commit com a mensagem especificada
git commit -m "-terraform"

# Obtém o nome da branch atual
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Faz o push para a branch atual
git push origin "$current_branch"
