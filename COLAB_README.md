# VSim - Google Colab Setup Guide

## Quick Start

### Method 1: Use the Notebook (Recommended)

1. **Go to Google Colab**: https://colab.research.google.com/
2. **Upload `VSim_Colab.ipynb`** notebook
3. **Enable GPU**: Runtime → Change runtime type → **GPU** (A100 if available)
4. **Run all cells** in order
5. **Upload your genome** FASTA file when prompted
6. **View results** in the notebook

### Method 2: Upload Project Folder

1. **Zip the Project-VSim folder** on your computer
2. **Open Colab**: https://colab.research.google.com/
3. **Upload the zip file**:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
4. **Unzip and setup**:
   ```python
   !unzip Project-VSim.zip
   !mv Project-VSim VSim
   %cd VSim
   ```
5. **Install dependencies**:
   ```python
   !pip install -q biopython numpy pandas scipy scikit-learn pyyaml requests flask matplotlib
   ```
6. **Run analysis**:
   ```python
   !python src/main.py data/raw/YOUR_GENOME.fasta --output results/
   ```

## GPU Setup

1. **Enable GPU**: Runtime → Change runtime type → GPU
2. **Select A100** if available (fastest)
3. **T4** works but slower

## Expected Performance

- **CPU**: ~60-90 minutes for SARS-CoV-2
- **GPU (T4)**: ~30-45 minutes
- **GPU (A100)**: ~15-30 minutes

## Files Created

- `VSim_Colab.ipynb` - Complete notebook with all steps
- `COLAB_SETUP.md` - This guide
- `COLAB_QUICKSTART.md` - Quick reference

## Troubleshooting

### GPU Not Available
- Check Runtime → Change runtime type → GPU
- Verify GPU is enabled

### Import Errors
- Run: `!pip install -r requirements.txt`
- Check VSim folder is uploaded correctly

### Out of Memory
- Reduce protein copy numbers in `config.yaml`
- Use smaller test genome first

## Notes

- **Results are saved** in `/content/VSim/results/`
- **Download before closing** Colab (results are temporary)
- **Mount Google Drive** to save permanently



