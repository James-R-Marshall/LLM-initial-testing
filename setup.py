import os
import sys
import subprocess
import venv
from pathlib import Path

#!/usr/bin/env python3
"""
setup.py - create a virtual environment and install dependencies commonly needed
for running a LoRA finetuning / inference script that uses Llama-like models.

Usage:
    python setup.py          # creates ./venv and installs CPU-friendly packages
    python setup.py --cuda   # attempt to install CUDA-enabled PyTorch & bitsandbytes
    python setup.py --cuda cu118  # specify CUDA toolkit tag (cu117, cu118, cu121, ...)
"""


VENV_DIR = Path(".venv")

# Core Python packages typically required for LLaMA + LoRA workflows
BASE_PACKAGES = [
        "transformers>=4.30",
        "accelerate>=0.20",
        "peft>=0.4.0",
        "safetensors",
        "einops",
        "sentencepiece",
        "tokenizers",
        "protobuf",
        "datasets",
        "tqdm",
        "huggingface-hub",
        "unsloth",
        "flask",
]

# Optional packages that often are used (bitsandbytes requires CUDA)
BINARY_PACKAGES = [
        "bitsandbytes",  # only useful with CUDA-enabled installs
]

TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]

def run(cmd, **kwargs):
        print("+", " ".join(map(str, cmd)))
        subprocess.check_call(cmd, **kwargs)

def get_pip_exe(venv_dir: Path) -> str:
        if sys.platform == "win32":
                return str(venv_dir / "Scripts" / "pip.exe")
        return str(venv_dir / "bin" / "pip")

def create_venv(venv_dir: Path):
        if venv_dir.exists():
                print(f"Using existing virtualenv at '{venv_dir}'.")
        else:
                print(f"Creating virtualenv at '{venv_dir}'...")
                venv.EnvBuilder(with_pip=True).create(str(venv_dir))
        return get_pip_exe(venv_dir)

def install_packages(pip_exe: str, packages, index_url=None, extra_args=None):
        cmd = [pip_exe, "install", "--upgrade"]
        if extra_args:
                cmd += extra_args
        if index_url:
                # For PyTorch CUDA wheels we usually need --index-url
                cmd += ["--index-url", index_url]
        cmd += list(packages)
        run(cmd)

def write_requirements(path: Path, packages):
        path.write_text("\n".join(packages))
        print(f"Wrote requirements to {path}")

def main(argv):
        use_cuda = False
        cuda_tag = "cu118"  # default CUDA tag; change if needed (cu117, cu121, ...)
        if len(argv) >= 2 and argv[1] in ("-h", "--help"):
                print(__doc__)
                sys.exit(0)
        if len(argv) >= 2 and argv[1] == "--cuda":
                use_cuda = True
                if len(argv) >= 3:
                        cuda_tag = argv[2]

        pip_exe = create_venv(VENV_DIR)

        # Upgrade pip/setuptools/wheel first
        run([pip_exe, "install", "--upgrade", "pip", "setuptools", "wheel"])

        # Install torch packages.
        # If user requested CUDA, use the PyTorch index for the requested cuda_tag.
        # Example index: https://download.pytorch.org/whl/cu118
        try:
                if use_cuda:
                        index = f"https://download.pytorch.org/whl/{cuda_tag}"
                        print(f"Installing CUDA PyTorch from {index} (cuda tag {cuda_tag})")
                        install_packages(pip_exe, TORCH_PACKAGES, index_url=index)
                        # bitsandbytes is useful only with CUDA; try installing it but tolerate failure
                        try:
                                install_packages(pip_exe, BINARY_PACKAGES)
                        except subprocess.CalledProcessError:
                                print("Warning: bitsandbytes failed to install. If you need it, install manually.")
                else:
                        print("Installing CPU / generic PyTorch from PyPI (if you want CUDA, run with --cuda)")
                        install_packages(pip_exe, TORCH_PACKAGES)
        except subprocess.CalledProcessError as e:
                print("Error installing torch packages:", e)
                sys.exit(1)

        # Install the rest of the python packages
        try:
                install_packages(pip_exe, BASE_PACKAGES)
        except subprocess.CalledProcessError as e:
                print("Error installing base packages:", e)
                sys.exit(1)

        # Write a requirements.txt for reproducibility
        reqs = TORCH_PACKAGES + BASE_PACKAGES + (BINARY_PACKAGES if use_cuda else [])
        write_requirements(Path("requirements.txt"), reqs)

        print()
        print("Setup complete.")
        print(f"To activate the virtualenv:")
        if sys.platform == "win32":
                print(r"  .\venv\Scripts\activate")
        else:
                print("  source venv/bin/activate")
        print("Then run your llama-LorA.py script inside that environment.")

if __name__ == "__main__":
        main(sys.argv)