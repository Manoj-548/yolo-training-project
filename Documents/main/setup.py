#!/usr/bin/env python3
"""
Setup script for OCR Project
This script helps set up the development environment for both VS Code and PyCharm
"""

import os
import sys
import subprocess
import venv
from pathlib import Path


def create_virtual_environment():
    """Create a Python virtual environment if it doesn't exist"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        venv.create(venv_path, with_pip=True)
        print("Virtual environment created successfully!")
    else:
        print("Virtual environment already exists.")


def install_dependencies():
    """Install project dependencies"""
    print("Installing dependencies...")
    if os.name == 'nt':  # Windows
        pip_path = Path("venv/Scripts/pip.exe")
        python_path = Path("venv/Scripts/python.exe")
    else:  # Unix-like
        pip_path = Path("venv/bin/pip")
        python_path = Path("venv/bin/python")

    if pip_path.exists():
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully!")
    else:
        print("Error: pip not found in virtual environment")


def setup_vs_code():
    """Setup VS Code specific configurations"""
    print("Setting up VS Code configurations...")
    vscode_dir = Path(".vscode")
    if not vscode_dir.exists():
        print("VS Code configurations already exist.")
        return

    print("VS Code is ready to use!")
    print("To open in VS Code: code .")


def setup_pycharm():
    """Setup PyCharm specific configurations"""
    print("Setting up PyCharm configurations...")
    idea_dir = Path(".idea")
    if not idea_dir.exists():
        print("PyCharm configurations already exist.")
        return

    print("PyCharm is ready to use!")
    print("Open this directory in PyCharm to get started.")


def create_directories():
    """Create necessary directories for the project"""
    directories = ["output", "logs", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")


def main():
    """Main setup function"""
    print("=== OCR Project Setup ===")
    print("This script will set up your development environment.")
    print()

    # Create virtual environment
    create_virtual_environment()

    # Install dependencies
    install_dependencies()

    # Create necessary directories
    create_directories()

    # Setup IDE configurations
    setup_vs_code()
    setup_pycharm()

    print()
    print("=== Setup Complete! ===")
    print()
    print("Next steps:")
    print("1. Download your dataset from Roboflow and place it in './roboflow_data'")
    print("2. For VS Code: Run 'code .' to open the project")
    print("3. For PyCharm: Open this directory as a project")
    print("4. Use the run configurations to train or run inference")
    print()
    print("Training command: python ocr.py --mode train --data_path ./roboflow_data")
    print("Inference command: python ocr.py --mode infer --image_path your_image.jpg")


if __name__ == "__main__":
    main()