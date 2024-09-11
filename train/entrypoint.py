import sys
import os
import subprocess

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Erro: Nome do script não fornecido.")
        sys.exit(1)

    script_name = sys.argv[1]
    script_path = os.path.join("/opt/ml/code", script_name)

    if not os.path.exists(script_path):
        print(f"Erro: Script não encontrado: {script_path}")
        sys.exit(1)

    # Executar o script com os argumentos restantes
    subprocess.run(["python3", script_path] + sys.argv[2:])
