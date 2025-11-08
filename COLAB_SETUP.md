# VSim - Google Colab Installation Guide

## Quick Setup (3 Steps)

### Step 1: Upload to Colab

**Option A: Upload Notebook**
1. Go to https://colab.research.google.com/
2. File → Upload notebook
3. Upload `VSim_Colab.ipynb`

**Option B: Upload Project Folder**
1. Zip your `Project-VSim` folder
2. In Colab, run:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
3. Unzip:
   ```python
   !unzip Project-VSim.zip
   !mv Project-VSim VSim
   %cd VSim
   ```

### Step 2: Enable GPU

1. **Runtime → Change runtime type**
2. **Hardware accelerator → GPU**
3. **GPU type → A100** (if available) or T4
4. Click **Save**

### Step 3: Run

**Using the Notebook:**
- Run all cells sequentially
- Upload genome when prompted
- View results inline

**Using One-Cell Script:**
- Copy contents of `COLAB_ONECELL.py`
- Paste into Colab cell
- Run the cell

## What You Get

After running, you'll have:
```
/content/VSim/
├── data/raw/              # Your genomes
├── results/              # Analysis output
│   ├── comprehensive_report.html  # Interactive report
│   ├── results.json      # JSON data
│   └── structures/       # PDB files
└── VSim_Colab.ipynb      # Notebook
```

## Performance

- **CPU**: 60-90 minutes
- **GPU (T4)**: 30-45 minutes  
- **GPU (A100)**: 15-30 minutes

## Files Included

- `VSim_Colab.ipynb` - Complete notebook
- `COLAB_ONECELL.py` - One-cell script
- `COLAB_SETUP.md` - Detailed guide
- `COLAB_README.md` - Quick reference

## Tips

1. **Save Results**: Download ZIP before closing Colab
2. **GPU Runtime**: A100 is fastest (if available)
3. **Large Genomes**: May take 30-60 minutes even with GPU
4. **Drive Mount**: Mount Google Drive to save permanently

## Troubleshooting

- **GPU not available**: Runtime → Change runtime type → GPU
- **Import errors**: Run dependency installation again
- **Out of memory**: Reduce copy numbers in config.yaml
