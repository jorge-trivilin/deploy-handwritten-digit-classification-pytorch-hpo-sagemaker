import sys
import os
import subprocess
import torch  # type: ignore

def check_gpu_availability():
    """
    Verifica se há uma GPU disponível para uso.
    """
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("Nenhuma GPU disponível. Executando na CPU.")


def run_script(script_name, additional_args):
    """
    Executa um script Python localizado em /opt/ml/code com argumentos adicionais.
    """
    script_path = os.path.join("/opt/ml/code", script_name)

    if not os.path.exists(script_path):
        print(f"Erro: Script não encontrado: {script_path}")
        sys.exit(1)

    print(f"Executando o script: {script_name}")
    result = subprocess.run(["python3", script_path] + additional_args, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Erro ao executar o script {script_name}. Código de saída: {result.returncode}")
        print(f"Saída do erro: {result.stderr}")
        sys.exit(result.returncode)

    print(f"Saída do script {script_name}: {result.stdout}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Erro: Nome do script não fornecido.")
        sys.exit(1)

    script_name = sys.argv[1]  # Captura o nome do script passado como argumento
    additional_args = sys.argv[2:]  # Captura os argumentos adicionais

    # Verifica se há uma GPU disponível
    check_gpu_availability()

    # Executa o script passado como argumento
    run_script(script_name, additional_args)
