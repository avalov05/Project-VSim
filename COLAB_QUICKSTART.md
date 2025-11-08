# VSim - Colab Setup Instructions

## Quick Start Guide

### 1. Upload to Google Colab

**Option A: Upload Notebook**
1. Go to https://colab.research.google.com/
2. File → Upload notebook
3. Upload `VSim_Colab.ipynb`

**Option B: Upload Project Folder**
1. Zip the entire `Project-VSim` folder
2. Upload the zip to Colab
3. Unzip: `!unzip Project-VSim.zip`
4. Rename: `!mv Project-VSim VSim`

### 2. Enable GPU

1. Runtime → Change runtime type
2. Hardware accelerator → **GPU** (A100 if available)
3. Click Save

### 3. Run the Notebook

1. Run all cells sequentially (Runtime → Run all)
2. When prompted, upload your genome FASTA file
3. Wait for analysis to complete
4. View results in the notebook
5. Download results as ZIP

## File Structure in Colab

```
/content/VSim/
├── data/
│   └── raw/              # Upload genomes here
├── results/              # Analysis output
│   ├── comprehensive_report.html
│   ├── results.json
│   └── structures/       # PDB files
├── src/                  # VSim source code
└── VSim_Colab.ipynb      # This notebook
```

## GPU Acceleration

The notebook will automatically:
- Detect GPU availability
- Use GPU for faster computation
- Accelerate protein folding
- Speed up molecular dynamics

**Expected Speedup:**
- CPU: ~60-90 minutes for SARS-CoV-2
- GPU (A100): ~15-30 minutes for SARS-CoV-2

## Troubleshooting

### GPU Not Available
- Check Runtime → Change runtime type → GPU
- Verify GPU is enabled in Colab settings

### Import Errors
- Run Step 1 (Install Dependencies) again
- Check that VSim folder is uploaded correctly

### Out of Memory
- Reduce protein copy numbers in config.yaml
- Use smaller test genome first

### File Not Found
- Make sure to upload genome file in Step 4
- Check file is in `data/raw/` directory

## Tips

1. **Save Progress**: Mount Google Drive to save results permanently
2. **Large Genomes**: May take 30-60 minutes even with GPU
3. **Download Results**: Always download before closing Colab
4. **Multiple Runs**: Results are saved in `results/` folder

## Alternative: Use Colab Code Cell

If you prefer, you can also run this directly in a Colab code cell:

```python
# Install and setup
!pip install -q biopython numpy pandas scipy scikit-learn pyyaml requests flask matplotlib
!git clone https://github.com/yourusername/VSim.git || echo "Already cloned"
%cd VSim

# Upload genome
from google.colab import files
uploaded = files.upload()

# Run analysis
!python src/main.py data/raw/{uploaded_filename} --output results/
```

## Support

For issues or questions:
- Check logs in notebook output
- Verify all dependencies are installed
- Ensure GPU is enabled
- Check file paths are correct



