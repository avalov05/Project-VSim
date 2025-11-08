# VSim - One-Cell Colab Setup
# Copy and paste this entire cell into Google Colab

# === SETUP ===
import os
import sys
from pathlib import Path

# Install dependencies
print("Installing dependencies...")
!pip install -q biopython numpy pandas scipy scikit-learn pyyaml requests flask matplotlib

# Setup paths
if 'google.colab' in str(get_ipython()):
    print("Running in Google Colab")
    # Mount Drive (optional)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except:
        pass
    
    # Check/upload VSim
    if not Path('/content/VSim').exists():
        print("\n⚠ VSim folder not found!")
        print("Please upload the Project-VSim folder:")
        print("1. Zip Project-VSim folder")
        print("2. Run: uploaded = files.upload()")
        print("3. Run: !unzip Project-VSim.zip && mv Project-VSim VSim")
    else:
        os.chdir('/content/VSim')
        sys.path.insert(0, '/content/VSim')
else:
    os.chdir('/Users/antonvalov/Documents/Project-VSim')
    sys.path.insert(0, '/Users/antonvalov/Documents/Project-VSim')

print(f"✓ Working directory: {os.getcwd()}")

# Check GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ No GPU - enable in Runtime → Change runtime type")
except:
    print("⚠ PyTorch not installed")

# === UPLOAD GENOME ===
print("\n" + "="*60)
print("STEP 1: Upload your genome FASTA file")
print("="*60)
from google.colab import files
uploaded = files.upload()

# Move to data directory
os.makedirs('data/raw', exist_ok=True)
for filename in uploaded.keys():
    import shutil
    shutil.move(filename, f'data/raw/{filename}')
    print(f"✓ Uploaded: data/raw/{filename}")

# === RUN ANALYSIS ===
print("\n" + "="*60)
print("STEP 2: Running analysis...")
print("="*60)

genomes = list(Path('data/raw').glob('*.fasta'))
if genomes:
    genome_file = str(genomes[0])
    print(f"Genome: {genome_file}")
    
    # Run analysis
    sys.path.insert(0, '.')
    from src.main import main
    import sys
    
    sys.argv = ['main.py', genome_file, '--output', 'results', '--verbose']
    main()
    
    print("\n✓ Analysis complete!")
    
    # Display report
    print("\n" + "="*60)
    print("STEP 3: View results")
    print("="*60)
    from IPython.display import IFrame
    report_path = Path('results/comprehensive_report.html')
    if report_path.exists():
        display(IFrame(str(report_path), width='100%', height=800))
    else:
        print("Report not found")
else:
    print("⚠ No genome file uploaded")



