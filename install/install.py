import argparse
import os
import subprocess
import sys
import platform
import shutil
import venv

# --- Configuration Variables ---
MIN_CUDA_VERSION_SUPPORTED_BY_SCRIPT = 121
MAX_CUDA_VERSION_SUPPORTED_BY_SCRIPT = 128
REQUIRED_TORCH_VERSION = "2.6.0"  # FramePack's minimum torch version
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
    """
    try:
        if capture_output:
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=check_output, shell=True, env=env)
            return result.stdout.strip()
        else:
            subprocess.run(command, cwd=cwd, check=check_output, shell=True, env=env)
            return None
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(command) if isinstance(command, list) else command}\nError: {e.stderr if e.stderr else e.output}")
    except FileNotFoundError:
        print_error(f"Command not found. Ensure necessary tools are in PATH: {' '.join(command) if isinstance(command, list) else command}")

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
             f"Learn more about the NVIDIA CUDA toolkit and download a version from: "
             f"https://developer.nvidia.com/cuda-toolkit-archive"
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

    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_path = os.path.join(current_dir, VENV_DIR_NAME)
    requirements_file_path = os.path.join(current_dir, REQUIREMENTS_FILE_NAME)

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
        nvcc_output = run_command("nvcc --version", capture_output=True, check_output=False)
        if nvcc_output:
            nvcc_version_line = next((line for line in nvcc_output.splitlines() if "release" in line), None)
            if nvcc_version_line:
                nvcc_version = nvcc_version_line.split(' ')[-1].strip().split('.')
                # Get major.minor, e.g., 12.2
                nvcc_version_str = f"{nvcc_version[0]}.{nvcc_version[1]}"
                print_info(f"CUDA Toolkit (nvcc) found. Version: {nvcc_version_str}")
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

    # --- PyTorch Installation ---
    print_info("--- Installing PyTorch and Dependencies ---")

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
            if not stripped_line.startswith(tuple(["torch", "torchvision", "torchaudio", "xformers"])) and stripped_line:
                filtered_requirements.append(stripped_line)

    if filtered_requirements:
        # Create a temporary filtered requirements file
        temp_req_path = os.path.join(current_dir, "temp_filtered_requirements.txt")
        with open(temp_req_path, 'w') as f:
            f.write("\n".join(filtered_requirements))

        other_deps_install_command = [venv_pip, "install", "-r", temp_req_path]
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
print(f'PyTorch version: {torch.__version__}')
cuda_available = torch.cuda.is_available()
print(f'CUDA available: {cuda_available}')
if cuda_available:
    try:
        print(f'CUDA device count: {torch.cuda.device_count()}')
        if torch.cuda.device_count() > 0:
            print(f'CUDA device name (0): {torch.cuda.get_device_name(0)}')
            print(f'CUDA device capability (0): {torch.cuda.get_device_capability(0)}')
    except Exception as e:
        print(f'Error getting CUDA device info: {e}')
        sys.exit(1) # Indicate error if device info fails
else:
    print('No CUDA-capable GPU found or PyTorch CUDA not configured correctly.')
    # Optionally, you might want to exit with an error here if CUDA is strictly required
    # sys.exit(1)
"""
    run_command([venv_python, "-c", python_verify_script])
    print_success("PyTorch verification complete.")

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