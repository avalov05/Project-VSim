# Google Colab Setup - Complete Guide

## 🚀 Quick Start

### Method 1: Use the Notebook (Easiest)

1. **Open Google Colab**: https://colab.research.google.com/
2. **Upload `VSim_Colab.ipynb`**
3. **Enable GPU**: Runtime → Change runtime type → **GPU (A100)**
4. **Run all cells** (Runtime → Run all)
5. **Upload genome** when prompted
6. **View results** inline

### Method 2: Upload Project Folder

1. **Zip** the `Project-VSim` folder
2. **In Colab**, upload the zip:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
3. **Unzip and setup**:
   ```python
   !unzip Project-VSim.zip
   !mv Project-VSim VSim
   %cd VSim
   !pip install -q biopython numpy pandas scipy scikit-learn pyyaml requests flask matplotlib
   ```
4. **Run analysis**:
   ```python
   !python src/main.py data/raw/YOUR_GENOME.fasta --output results/
   ```

### Method 3: One-Cell Script

Copy `COLAB_ONECELL.py` contents into a Colab cell and run.

## 📦 Files Created

- **`VSim_Colab.ipynb`** - Complete notebook with all steps
- **`COLAB_ONECELL.py`** - Single-cell script
- **`COLAB_SETUP.md`** - Detailed setup guide
- **`COLAB_README.md`** - Quick reference

## ⚡ GPU Setup

1. **Runtime → Change runtime type**
2. **Hardware accelerator → GPU**
3. **GPU type → A100** (fastest) or T4
4. **Save**

## ⏱️ Expected Performance

| Hardware | Time (SARS-CoV-2) |
|----------|-------------------|
| CPU      | 60-90 minutes     |
| GPU (T4) | 30-45 minutes     |
| GPU (A100)| 15-30 minutes    |

## 📁 Directory Structure

After setup:
```
/content/VSim/
├── data/raw/          # Upload genomes here
├── results/          # Analysis output
│   ├── comprehensive_report.html
│   ├── results.json
│   └── structures/   # PDB files
└── src/              # VSim source code
```

## 🔧 Troubleshooting

**GPU Not Available:**
- Runtime → Change runtime type → GPU
- Verify GPU is enabled

**Import Errors:**
- Run: `!pip install -r requirements.txt`
- Check VSim folder is uploaded

**Out of Memory:**
- Reduce copy numbers in `config.yaml`
- Use smaller test genome

## 💾 Saving Results

**Download ZIP:**
```python
from google.colab import files
!zip -r results.zip results/
files.download('results.zip')
```

**Save to Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r results /content/drive/MyDrive/VSim_results/
```

## ✅ Ready to Use!

Everything is set up for Google Colab with GPU acceleration!



