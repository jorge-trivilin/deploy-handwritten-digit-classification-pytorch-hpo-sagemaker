import sys
import os
import subprocess
import torch # type: ignore


def verify_directories():
    """
    Verifica se os diretórios SageMaker esperados estão presentes.
    """
    input_dir = "/opt/ml/input"
    output_dir = "/opt/ml/output"
    model_dir = "/opt/ml/model"

    if not os.path.exists(input_dir):
        print(f"Erro: Diretório de entrada não encontrado: {input_dir}")
        sys.exit(1)

    if not os.path.exists(output_dir):
        print(f"Erro: Diretório de saída não encontrado: {output_dir}")
        sys.exit(1)

    if not os.path.exists(model_dir):
        print(f"Erro: Diretório de modelo não encontrado: {model_dir}")
        sys.exit(1)


def check_gpu_availability():
    """
    Verifica se há uma GPU disponível para uso.
    """
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("Nenhuma GPU disponível. Executando na CPU.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Erro: Nome do script não fornecido.")
        sys.exit(1)

    script_name = sys.argv[1]
    script_path = os.path.join("/opt/ml/code", script_name)

    if not os.path.exists(script_path):
        print(f"Erro: Script não encontrado: {script_path}")
        sys.exit(1)

    # Verifica se os diretórios esperados pelo SageMaker estão presentes
    verify_directories()

    # Verifica se há uma GPU disponível
    check_gpu_availability()

    # Executa o script com os argumentos restantes
    subprocess.run(["python3", script_path] + sys.argv[2:])
