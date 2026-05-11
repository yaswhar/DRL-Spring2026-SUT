"""
Creates a Python .venv and installs all dependencies for the DQN project.

Usage:
    python setup_venv.py
"""

import os
import subprocess
import sys
import platform

def main():
    venv_dir = ".venv"

    # Create venv if it doesn't exist
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

    # Determine the pip executable inside the venv
    if platform.system() == "Windows":
        pip = os.path.join(venv_dir, "Scripts", "pip.exe")
        python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        pip = os.path.join(venv_dir, "bin", "pip")
        python = os.path.join(venv_dir, "bin", "python")

    # Upgrade pip
    subprocess.check_call([pip, "install", "--upgrade", "pip"])

    # Install gymnasium with box2d (needed for LunarLander)
    subprocess.check_call([pip, "install", "gymnasium[box2d]"])

    # Install remaining core dependencies from requirements.txt
    subprocess.check_call([pip, "install", "-r", "requirements.txt"])

    # --- PyTorch with CUDA ---
    # Ask user for CUDA version; default to 12.1 if not specified.
    cuda_ver = input("Which CUDA version do you have? (11.8, 12.1, 12.4, or leave blank for 12.1): ").strip()
    cuda_map = {
        "11.8": "https://download.pytorch.org/whl/cu118",
        "12.1": "https://download.pytorch.org/whl/cu121",
        "12.4": "https://download.pytorch.org/whl/cu124",
    }
    index_url = cuda_map.get(cuda_ver, "https://download.pytorch.org/whl/cu121")
    print(f"Installing PyTorch with CUDA index: {index_url}")
    subprocess.check_call([pip, "install", "torch", "torchvision", "torchaudio", "--index-url", index_url])

    # Verify CUDA availability with a quick test
    print("\nVerifying PyTorch CUDA...")
    subprocess.check_call([python, "-c", "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"])

    # Additional system note: mediapy needs ffmpeg
    print("\n--- IMPORTANT ---")
    print("mediapy requires ffmpeg for video writing/reading.")
    print("Please install ffmpeg and ensure it is in your PATH:")
    print("  Windows: download from https://ffmpeg.org/download.html and add to PATH")
    print("  Linux:   sudo apt install ffmpeg")
    print("  macOS:   brew install ffmpeg")

    print(f"\nSetup complete. Activate the environment with:\n  {venv_dir}/Scripts/activate  (Windows)\n  source {venv_dir}/bin/activate (Linux/macOS)")

if __name__ == "__main__":
    main()