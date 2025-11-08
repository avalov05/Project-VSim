"""
Setup Script for VSim
"""

#!/usr/bin/env python3
"""Setup script for VSim installation"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("VSim Setup Script")
    print("="*60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
        print("Error: Python 3.9+ required")
        sys.exit(1)
    
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Create directories
    print("\nCreating directories...")
    directories = [
        'data/raw', 'data/processed',
        'models/trained', 'models/checkpoints',
        'results', 'logs', 'docs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if not run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip"
    ):
        print("Warning: Failed to upgrade pip, continuing...")
    
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing requirements"
    ):
        print("Error: Failed to install requirements")
        sys.exit(1)
    
    # Create sample genome if it doesn't exist
    sample_genome = Path('data/raw/sample_genome.fasta')
    if not sample_genome.exists():
        print("\nCreating sample genome file...")
        # Sample genome will be created by the write tool
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Place your viral genome FASTA file in data/raw/")
    print("2. Run analysis:")
    print("   python src/main.py data/raw/your_genome.fasta")
    print("\nOr start the web interface:")
    print("   python src/web/app.py")
    print("   Then visit http://localhost:8080")
    print("\nRun tests:")
    print("   pytest tests/")

if __name__ == "__main__":
    main()

