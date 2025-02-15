# entrypoint.py

"""
This script is used to check GPU availability and execute Python scripts located in a specific directory. While it is not used directly in the project, it serves as a useful orchestrator if integrated into a pipeline, such as Airflow.

Main Functionality:
- **check_gpu_availability**: Checks if a GPU is available and prints the detected GPU's name or a message indicating that execution will occur on the CPU.
- **run_script**: Executes a Python script located in the `/opt/ml/code` directory with additional provided arguments.

Usage:
- This script is executed from the command line. It expects the name of the script to be run as the first argument, followed by any additional arguments for the script.

Detailed Functions:
- **check_gpu_availability()**:
  - **Description**: Checks for available GPUs and prints information about the GPU or a warning if only the CPU is available.
  
- **run_script(script_name, additional_args)**:
  - **Inputs**: 
    - `script_name` (name of the Python script to be executed).
    - `additional_args` (list of additional arguments to pass to the Python script).
  - **Description**: Locates and executes the specified Python script with the provided arguments. If the script is not found or if an error occurs during execution, it prints the error details and exits.
"""


import sys
import os
import subprocess
import torch  # type: ignore


def check_gpu_availability():
    """
    Checks if a GPU is available for use.
    """
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available. Running on CPU.")


def run_script(script_name, additional_args):
    """
    Executes a Python script located in /opt/ml/code with additional arguments.
    """
    script_path = os.path.join("/opt/ml/code", script_name)
    if not os.path.exists(script_path):
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)

    print(f"Executing script: {script_name}")
    result = subprocess.run(
        ["python3", script_path] + additional_args, capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Error executing script {script_name}. Exit code: {result.returncode}")
        print(f"Error output: {result.stderr}")
        sys.exit(result.returncode)

    print(f"Output of script {script_name}: {result.stdout}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Script name not provided.")
        sys.exit(1)

    script_name = sys.argv[1]  # Captures the script name passed as an argument
    additional_args = sys.argv[2:]  # Captures additional arguments

    # Checks if a GPU is available
    check_gpu_availability()

    # Executes the script passed as an argument
    run_script(script_name, additional_args)
