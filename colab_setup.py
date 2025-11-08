#!/usr/bin/env python3
"""
VSim - Google Colab Setup Script
Run this in Colab to setup and run VSim easily
"""

print("=" * 60)
print("VSim - Google Colab Setup")
print("=" * 60)

# Step 1: Install dependencies
print("\n[1/6] Installing dependencies...")
import subprocess
import sys

packages = [
    'biopython', 'numpy', 'pandas', 'scipy', 'scikit-learn',
    'pyyaml', 'requests', 'flask', 'matplotlib', 'plotly'
]

for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

print("✓ Dependencies installed")

# Step 2: Setup paths
print("\n[2/6] Setting up paths...")
import os
from pathlib import Path

if 'google.colab' in str(get_ipython()):
    print("Running in Google Colab")
    if not Path('/content/VSim').exists():
        print("⚠ VSim folder not found!")
        print("Please upload the Project-VSim folder:")
        print("  1. Zip the Project-VSim folder")
        print("  2. Upload: files.upload()")
        print("  3. Unzip: !unzip Project-VSim.zip")
        print("  4. Rename: !mv Project-VSim VSim")
    else:
        os.chdir('/content/VSim')
        sys.path.insert(0, '/content/VSim')
        print(f"✓ Working directory: {os.getcwd()}")
else:
    print("Running locally")
    os.chdir('/Users/antonvalov/Documents/Project-VSim')
    sys.path.insert(0, '/Users/antonvalov/Documents/Project-VSim')
    print(f"✓ Working directory: {os.getcwd()}")

# Step 3: Check GPU
print("\n[3/6] Checking GPU...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ No GPU - enable in Runtime → Change runtime type")
except:
    print("⚠ PyTorch not installed - GPU check skipped")

# Step 4: Upload genome
print("\n[4/6] Upload genome file...")
print("Run this in a separate cell:")
print("  from google.colab import files")
print("  uploaded = files.upload()")

# Step 5: Run analysis
print("\n[5/6] Ready to run analysis!")
print("Run this in a separate cell:")
print("  from src.main import main")
print("  import sys")
print("  sys.argv = ['main.py', 'data/raw/YOUR_FILE.fasta', '--output', 'results']")
print("  main()")

# Step 6: View results
print("\n[6/6] View results:")
print("  from IPython.display import IFrame")
print("  IFrame('results/comprehensive_report.html', width='100%', height=800)")

print("\n" + "=" * 60)
print("Setup complete! Follow the steps above to run analysis.")
print("=" * 60)



