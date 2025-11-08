# How to View Your 3D Models

## Quick Start (Easiest Method)

**Just run this command:**
```bash
python3 view_3d_models.py
```

This will:
1. List all available 3D models
2. Open the HTML report in your browser automatically
3. Show you where to find the interactive 3D viewer

## Method 1: HTML Report (Recommended) ⭐

The HTML report contains an **interactive 3D viewer** that works right in your browser.

### To open it:
```bash
# macOS
open results/comprehensive_report.html

# Linux
xdg-open results/comprehensive_report.html

# Windows
start results/comprehensive_report.html

# Or use the viewer script
python3 view_3d_models.py
```

### In the HTML report:
1. Scroll down to the **"Virus 3D Model"** section
2. You'll see an interactive 3D viewer (600px tall)
3. **Drag** with your mouse to rotate the model
4. **Scroll** to zoom in/out
5. The model is color-coded by spectrum (rainbow coloring)

### What you'll see:
- 🦠 Complete virus particle assembled from your genome
- 🧬 Individual protein structures (if available)
- 📊 Analysis results and metrics

## Method 2: Online Viewers

Upload your PDB files to these free online viewers:

### Mol* Viewer (Best Option)
1. Go to: https://molstar.org/viewer/
2. Drag and drop your PDB file
3. Interactive 3D visualization with professional features

### 3Dmol.js Viewer
1. Go to: https://3dmol.csb.pitt.edu/
2. Upload your PDB file or paste content
3. Simple and fast visualization

### Your PDB Files:
- **Virus particles**: `results/structures/virus_*.pdb`
- **Individual proteins**: `results/structures/proteins/protein_*.pdb`

## Method 3: Desktop Software

### PyMOL (Professional)
```bash
# Install PyMOL first, then:
pymol results/structures/virus_particle_detailed_*.pdb
```

### ChimeraX (Free, Professional)
```bash
# Install UCSF ChimeraX, then:
chimera results/structures/virus_particle_detailed_*.pdb
```

### VMD (Visual Molecular Dynamics)
```bash
# Install VMD, then:
vmd results/structures/virus_particle_detailed_*.pdb
```

## Method 4: Python/Jupyter Notebooks

### py3Dmol (Interactive in Jupyter)
```python
import py3Dmol

# View a virus particle
with open('results/structures/virus_particle_detailed_04048aec.pdb', 'r') as f:
    pdb_data = f.read()

view = py3Dmol.view(width=800, height=600)
view.addModel(pdb_data, 'pdb')
view.setStyle({'cartoon': {'color': 'spectrum'}})
view.zoomTo()
view.show()
```

### NGLview (Jupyter only)
```python
import nglview
view = nglview.show_file('results/structures/virus_particle_detailed_04048aec.pdb')
view
```

## Your Current Results

Based on your last run, you have:

✅ **6 virus particle models** ready to view
- `virus_particle_detailed_04048aec.pdb` (239 KB) - Recommended!
- `virus_particle_detailed_c4fcb7b4.pdb` (4.2 MB) - More detailed
- `virus_assembled_from_genome.pdb` (374 KB)
- `virus_coronavirus_realistic.pdb` (63 MB) - Very detailed!

✅ **79 individual protein structures** in `results/structures/proteins/`

## Troubleshooting

### The HTML viewer isn't showing the model?
- Make sure JavaScript is enabled in your browser
- Try refreshing the page
- Check browser console for errors (F12)

### Want to view a specific PDB file?
```bash
# List all available files
ls -lh results/structures/*.pdb
ls -lh results/structures/proteins/*.pdb

# View a specific file with online viewer
# Just drag and drop to https://molstar.org/viewer/
```

### Need help?
Run the viewer script for interactive help:
```bash
python3 view_3d_models.py
```

## Summary

**🎯 Easiest way:** Run `python3 view_3d_models.py` - it opens everything automatically!

**🌐 Best visualization:** Use Mol* Viewer (https://molstar.org/viewer/) - drag and drop your PDB file

**📊 Full analysis:** Open `results/comprehensive_report.html` in your browser

**🔬 Professional tools:** Use PyMOL or ChimeraX for advanced analysis

