import argparse
import os
import subprocess
import sys
import platform
import shutil
import venv
import re # Added for robust nvcc version parsing

# --- Configuration Variables ---
MIN_CUDA_VERSION_SUPPORTED_BY_SCRIPT = 121
MAX_CUDA_VERSION_SUPPORTED_BY_SCRIPT = 128
REQUIRED_TORCH_VERSION = "2.6.0"   # FramePack's minimum torch version
REQUIRED_CUDA_FOR_FRAME_PACK = 126  # FramePack's specific CUDA version requirement
MIN_PYTHON_VERSION_RECOMMENDED = (3, 10)  # Recommend Python 3.10 or newer
VENV_DIR_NAME = ".venv_goan"  # Standard virtual environment directory name
REQUIREMENTS_FILE_NAME = "requirements.txt"  # Expected name for project requirements

# --- Helper Functions for Cross-Platform Output ---
def print_info(message):
    print(f"\033[96mINFO: {message}\033[0m") # Cyan

def print_success(message):
    print(f"\033[92mSUCCESS: {message}\033[0m") # Green

def print_warning(message):
    print(f"\033[93mWARNING: {message}\033[0m") # Yellow

def print_error(message):
    print(f"\033[91mERROR: {message}\033[0m") # Red
    sys.exit(1)

def run_command(command, cwd=None, capture_output=False, check_output=True, env=None):
    """
    Runs a shell command and handles potential errors.
    `command` should be a list of strings (executable and its arguments).
    """
    try:
        if capture_output:
            # shell=False (default) is generally safer when command is a list
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=check_output, env=env)
            return result.stdout.strip()
        else:
            # shell=False (default) is generally safer when command is a list
            subprocess.run(command, cwd=cwd, check=check_output, env=env)
            return None
    except subprocess.CalledProcessError as e:
        # Check if stderr or stdout contains content to display
        error_output = e.stderr if e.stderr else e.output
        print_error(f"Command failed: {' '.join(command)}\nError: {error_output}")
    except FileNotFoundError:
        print_error(f"Command not found. Ensure necessary tools are in PATH: {command[0]}")

def get_venv_python_executable(venv_path):
    """
    Returns the path to the python executable within the virtual environment.
    """
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:
        return os.path.join(venv_path, "bin", "python")

def get_venv_pip_executable(venv_path):
    """
    Returns the path to the pip executable within the virtual environment.
    """
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        return os.path.join(venv_path, "bin", "pip")

def main():
    parser = argparse.ArgumentParser(
        description="Automates the installation of goan's Python dependencies, focusing on PyTorch with a specific CUDA version, within a virtual environment."
    )
    parser.add_argument(
        "cuda_version",
        type=int,
        help=f"The target CUDA version (e.g., 121, 126). Must be between {MIN_CUDA_VERSION_SUPPORTED_BY_SCRIPT} and {MAX_CUDA_VERSION_SUPPORTED_BY_SCRIPT} (inclusive). "
             f"Note: FramePack specifically requires torch>={REQUIRED_TORCH_VERSION}+cu{REQUIRED_CUDA_FOR_FRAME_PACK}. "
             f"Learn more about the NVIDIA CUDA toolkit and download a version from: https://developer.nvidia.com/cuda-toolkit-archive"
    )
    args = parser.parse_args()
    target_cuda_version = args.cuda_version

    if not (MIN_CUDA_VERSION_SUPPORTED_BY_SCRIPT <= target_cuda_version <= MAX_CUDA_VERSION_SUPPORTED_BY_SCRIPT):
        print_error(f"Provided CUDA version '{target_cuda_version}' is outside the script's supported range [{MIN_CUDA_VERSION_SUPPORTED_BY_SCRIPT}-{MAX_CUDA_VERSION_SUPPORTED_BY_SCRIPT}].")

    if target_cuda_version < REQUIRED_CUDA_FOR_FRAME_PACK:
        print_warning(f"FramePack requires torch>={REQUIRED_TORCH_VERSION}+cu{REQUIRED_CUDA_FOR_FRAME_PACK}.")
        print_warning(f"You specified CUDA version {target_cuda_version}, which might not meet FramePack's specific CUDA dependency.")
        print_warning(f"Consider using CUDA version {REQUIRED_CUDA_FOR_FRAME_PACK} or higher if available.")
        response = input("Do you wish to continue with this CUDA version? (y/N): ").strip().lower()
        if response != 'y':
            print_error("Installation aborted by user.")

    # Determine the base directory for the virtual environment.
    # When `install.sh` executes `python3 install/install.py`, the current working directory
    # is the project root (e.g., `/home/shane/github/shanevcantwell/goan_test_2`).
    # Therefore, the venv should be created in the current working directory.
    base_dir = os.getcwd() # Get the current working directory from where the script was launched.
    venv_path = os.path.join(base_dir, VENV_DIR_NAME)
    
    # The requirements file is expected to be in the same directory as the install.py script.
    # So, use the script's directory for the requirements file path.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_file_path = os.path.join(script_dir, REQUIREMENTS_FILE_NAME)

    print_info("--- Starting goan PyTorch Installation ---")

    # --- Pre-installation Checks ---
    print_info("--- Pre-installation Checks ---")

    # Check Python version running the script
    current_python_version = sys.version_info
    if current_python_version < MIN_PYTHON_VERSION_RECOMMENDED:
        print_warning(f"Detected Python version: {current_python_version.major}.{current_python_version.minor}.{current_python_version.micro}. "
                      f"Recommended version is {MIN_PYTHON_VERSION_RECOMMENDED[0]}.{MIN_PYTHON_VERSION_RECOMMENDED[1]} or newer for best compatibility.")
    print_info(f"Python executing script: {sys.executable} (Version: {current_python_version.major}.{current_python_version.minor}.{current_python_version.micro})")


    # Check for nvcc (CUDA Toolkit)
    print_info("Checking for CUDA Toolkit (nvcc)...")
    try:
        # Pass nvcc as a list for robustness. shell=False by default with lists.
        nvcc_output = run_command(["nvcc", "--version"], capture_output=True, check_output=False)
        if nvcc_output:
            nvcc_version_line = next((line for line in nvcc_output.splitlines() if "release" in line), None)
            if nvcc_version_line:
                # Extract version like "V12.2.140" or "release 12.2" and get major.minor (e.g., 12.2)
                # Use regex to find 'release X.Y' or 'VX.Y.Z'
                match = re.search(r'release (\d+\.\d+)', nvcc_version_line)
                if match:
                    nvcc_version_str = match.group(1)
                else:
                    match = re.search(r'V(\d+\.\d+)', nvcc_version_line)
                    if match:
                        nvcc_version_str = match.group(1)
                    else:
                        nvcc_version_str = "Unknown" # Fallback if neither pattern matches

                print_info(f"CUDA Toolkit (nvcc) found. Version: V{nvcc_version_str}")
            else:
                print_warning("nvcc found, but could not parse version output.")
        else:
            print_warning("CUDA Toolkit (nvcc) not found in PATH.")
            print_warning("PyTorch with CUDA will install, but it may not run on your GPU if the CUDA Toolkit and compatible NVIDIA drivers are not installed.")
    except Exception as e:
        print_warning(f"An error occurred while checking nvcc: {e}")
        print_warning("CUDA Toolkit (nvcc) not found in PATH.")
        print_warning("PyTorch with CUDA will install, but it may not run on your GPU if the CUDA Toolkit and compatible NVIDIA drivers are not installed.")


    # Check for requirements.txt
    if not os.path.exists(requirements_file_path):
        print_error(f"'{requirements_file_path}' not found. Please ensure your project's requirements file is present.")
    print_info(f"'{REQUIREMENTS_FILE_NAME}' found.")

    # --- Virtual Environment Setup ---
    print_info("--- Virtual Environment Setup ---")

    if os.path.exists(venv_path):
        print_info(f"Existing virtual environment found at '{venv_path}'. Removing and recreating...")
        try:
            shutil.rmtree(venv_path)
        except OSError as e:
            print_error(f"Error removing existing virtual environment: {e}. Please ensure no files are in use.")

    print_info(f"Creating virtual environment at '{venv_path}'...")
    try:
        venv.create(venv_path, with_pip=True, clear=True)
        print_success("Virtual environment created.")
    except Exception as e:
        print_error(f"Failed to create virtual environment: {e}")

    venv_python = get_venv_python_executable(venv_path)
    venv_pip = get_venv_pip_executable(venv_path)

    # --- Update pip within the virtual environment ---
    print_info("--- Updating pip in virtual environment ---")
    # Use venv_python to run the pip module for updating itself.
    # This ensures that pip itself is up-to-date before installing other packages.
    update_pip_command = [venv_python, "-m", "pip", "install", "-U", "pip"]
    print_info(f"Executing pip update command: {' '.join(update_pip_command)}")
    run_command(update_pip_command)
    print_success("pip updated successfully in the virtual environment.")

    # --- PyTorch Installation ---
    print_info("--- Installing PyTorch and Dependencies ---")

    # Corrected pip install command structure:
    # `venv_pip` is the executable, followed by 'install', then its arguments.
    # --break-system-packages is re-added to address potential "externally-managed-environment"
    # errors on Linux distributions like Ubuntu, even within a virtual environment,
    # due to system-level pip protections.
    pytorch_install_command = [
        venv_pip, "install", "--force-reinstall", "--break-system-packages",
        "torch", "torchvision", "torchaudio", "xformers",
        "--index-url", f"https://download.pytorch.org/whl/cu{target_cuda_version}"
    ]

    print_info(f"Executing PyTorch installation command: {' '.join(pytorch_install_command)}")
    run_command(pytorch_install_command)
    print_success("PyTorch and related packages installed successfully.")

    # --- Install other dependencies from requirements.txt ---
    print_info(f"--- Installing other dependencies from '{REQUIREMENTS_FILE_NAME}' ---")

    # Filter requirements.txt to exclude PyTorch components
    filtered_requirements = []
    with open(requirements_file_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            # Filter out lines that start with torch, torchvision, torchaudio, xformers
            # Also ensure the line is not empty after stripping
            if not stripped_line.startswith(tuple(["torch", "torchvision", "torchaudio", "xformers"])) and stripped_line:
                filtered_requirements.append(stripped_line)

    if filtered_requirements:
        # Create a temporary filtered requirements file
        temp_req_path = os.path.join(script_dir, "temp_filtered_requirements.txt") # Use script_dir for temp file
        with open(temp_req_path, 'w') as f:
            f.write("\n".join(filtered_requirements))

        # Corrected pip install command structure for other dependencies:
        # --break-system-packages is re-added here as well for consistency and robustness.
        other_deps_install_command = [venv_pip, "install", "--break-system-packages", "-r", temp_req_path]
        print_info(f"Installing additional requirements from '{REQUIREMENTS_FILE_NAME}'...")
        run_command(other_deps_install_command)
        print_success("Additional dependencies installed successfully.")
        os.remove(temp_req_path) # Clean up temporary file
    else:
        print_info(f"No additional non-PyTorch specific dependencies found in '{REQUIREMENTS_FILE_NAME}' to install.")

    # --- Post-installation Checks ---
    print_info("--- Post-installation Checks ---")

    print_info("Verifying PyTorch installation...")
    python_verify_script = """
import torch
import sys
import os

try:
    print(f'PyTorch version: {torch.__version__}')
    cuda_available = torch.cuda.is_available()
    print(f'CUDA available: {cuda_available}')
    if cuda_available:
        print(f'CUDA device count: {torch.cuda.device_count()}')
        if torch.cuda.device_count() > 0:
            print(f'CUDA device name (0): {torch.cuda.get_device_name(0)}')
            print(f'CUDA device capability (0): {torch.cuda.get_device_capability(0)}')
        else:
            print('No CUDA-capable GPU found, or PyTorch CUDA not configured correctly, despite CUDA being available.')
    else:
        print('No CUDA-capable GPU found or PyTorch CUDA not configured correctly.')
    sys.exit(0) # Explicitly exit success after printing
except Exception as e:
    print(f'ERROR: PyTorch verification failed: {e}', file=sys.stderr)
    sys.exit(1) # Explicitly exit with error
"""
    # Write the script to a temporary file for robust execution
    temp_verify_script_path = os.path.join(script_dir, "temp_verify_torch.py") # Use script_dir for temp file
    with open(temp_verify_script_path, "w") as f:
        f.write(python_verify_script)

    try:
        # Execute the temporary script using the venv's python
        # capture_output=False so it prints directly to console
        # check_output=True so it raises error if the verification script fails
        run_command([venv_python, temp_verify_script_path], capture_output=False, check_output=True)
        print_success("PyTorch verification complete and successful.")
    except Exception as e:
        # The run_command already handles printing errors if check_output is True,
        # but this outer try-except catches any unexpected issues during execution of run_command itself.
        print_error(f"An error occurred during PyTorch verification: {e}")
    finally:
        # Ensure the temporary file is cleaned up, even if command fails
        if os.path.exists(temp_verify_script_path):
            os.remove(temp_verify_script_path)


    print("\n" + "="*50)
    print_success("Installation complete!")
    print(f"To activate the virtual environment later:")
    if platform.system() == "Windows":
        print(f"  Open PowerShell/CMD and run: .\\{VENV_DIR_NAME}\\Scripts\\activate")
    else:
        print(f"  Open your shell and run: source ./{VENV_DIR_NAME}/bin/activate")
    print("To deactivate, simply run: deactivate")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()