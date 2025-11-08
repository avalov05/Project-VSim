# VSim - Usage Guide

## Quick Start

### 1. Basic Analysis

```bash
# Analyze a genome file
python3 src/main.py data/raw/sample_genome.fasta

# Or use SARS-CoV-2 (if downloaded)
python3 src/main.py data/raw/sars_cov2_complete.fasta
```

### 2. View Results

After analysis completes, open the HTML report:

```bash
# macOS
open results/comprehensive_report.html

# Linux
xdg-open results/comprehensive_report.html

# Windows
start results/comprehensive_report.html
```

### 3. Interact with 3D Model

In the HTML report, scroll to "Virus 3D Model" section:
- **Drag** to rotate the virus
- **Scroll** to zoom in/out
- **Change style** using the dropdown (Cartoon, Stick, Sphere, Surface)

## Command Line Options

```bash
# Basic usage
python3 src/main.py genome.fasta

# Specify output directory
python3 src/main.py genome.fasta --output my_results/

# Use custom config file
python3 src/main.py genome.fasta --config my_config.yaml

# Verbose output
python3 src/main.py genome.fasta --verbose
```

## Web Interface

Start the web server:

```bash
python3 src/web/app.py
```

Then open your browser to:
```
http://localhost:8080
```

Upload a genome file and view results interactively!

## What Happens During Analysis

1. **Genome Analysis** (5-10 seconds)
   - Parses genome sequence
   - Detects ORFs (Open Reading Frames)
   - Predicts proteins

2. **Protein Folding** (30 seconds - 5 minutes per protein)
   - Each protein is folded using molecular dynamics simulation
   - 500 iterations per protein
   - Energy minimization

3. **Capsid Assembly** (1-10 minutes)
   - Proteins are assembled into realistic virus structure
   - Physics-based simulation
   - Energy minimization

4. **Analysis** (5-10 seconds)
   - Environmental dynamics
   - Cell interactions
   - Cancer cell analysis
   - ML predictions

5. **Report Generation** (5 seconds)
   - HTML report with 3D visualization
   - JSON results file
   - PDB structure files

## Output Files

### Main Report
- `results/comprehensive_report.html` - Interactive HTML report with 3D visualization

### Structure Files
- `results/structures/virus_assembled_from_genome.pdb` - Complete virus particle
- `results/structures/protein_*.pdb` - Individual protein structures

### Data Files
- `results/results.json` - Complete analysis results in JSON format

## Example Workflow

```bash
# 1. Analyze genome
python3 src/main.py data/raw/sample_genome.fasta

# Output:
# ================================================================================
# Analysis complete!
# Report: results/comprehensive_report.html
# ================================================================================

# 2. Open the report
open results/comprehensive_report.html

# 3. Explore the report:
#    - Scroll to "Virus 3D Model" section
#    - Interact with the 3D visualization
#    - View synthesis feasibility
#    - Check environmental predictions
#    - Review cell interaction analysis

# 4. Download PDB files for advanced analysis
#    - Use PyMOL, Chimera, or other molecular viewers
```

## Understanding the Results

### Synthesis Feasibility
- **Score**: 0-100% likelihood the virus can exist
- **Confidence**: How confident the prediction is
- **Factors**: ORF quality, codon usage, regulatory elements

### Virus Structure
- **Type**: Classification (coronavirus, small_rna_virus, etc.)
- **Shape**: Capsid shape (icosahedral, helical, etc.)
- **Diameter**: Estimated virus size in nanometers
- **3D Model**: Complete virus particle with all proteins

### Environmental Analysis
- **Survival time**: How long virus survives without host
- **Stability**: Temperature and pH stability
- **Transmission**: Potential transmission routes

### Cell Interactions
- **Receptor binding**: Predicted cell receptors
- **Entry mechanism**: How virus enters cells
- **Tropism**: Which cell types are targeted

## Customization

### Edit Configuration

Edit `config.yaml` to customize:

```yaml
structure_prediction:
  method: esmfold  # or alphafold
  use_api: true
  
protein_folding:
  max_iterations: 500  # Reduce for faster results
  
capsid_assembly:
  energy_minimization: true
```

### Speed Up Analysis

For faster results (less accurate):
1. Reduce protein folding iterations in `src/structure/protein_folding.py`
   - Change `max_iterations` from 500 to 100-200
2. Skip energy minimization
   - Set `energy_minimization: false` in config

## Troubleshooting

### Analysis Takes Too Long
- **Large genomes**: Protein folding scales with protein count
- **Solution**: Reduce `max_iterations` or skip folding for known structures

### Can't See 3D Model
- **Check browser**: Enable JavaScript
- **Console errors**: Press F12 to check browser console
- **Try different browser**: Chrome/Firefox recommended
- **Download PDB**: Use PyMOL or Chimera to view directly

### Memory Issues
- **Large genomes**: May require 8GB+ RAM
- **Solution**: Process smaller genomes or reduce protein count

### Import Errors
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 verify_installation.py
```

## Advanced Usage

### Batch Processing

```bash
# Process multiple genomes
for genome in data/raw/*.fasta; do
    python3 src/main.py "$genome" --output "results/$(basename $genome .fasta)/"
done
```

### Programmatic Usage

```python
from src.genome.analyzer import GenomeAnalyzer
from src.structure.predictor import StructurePredictor

# Analyze genome
analyzer = GenomeAnalyzer()
genome_results = analyzer.analyze_genome('genome.fasta')

# Predict structures
predictor = StructurePredictor()
structure_results = predictor.predict_structures(genome_results)

# Access results
virus_model = structure_results['virus_particle']
print(f"Virus diameter: {virus_model['estimated_diameter_nm']} nm")
```

## Tips

1. **Start with sample genome**: Test with `data/raw/sample_genome.fasta` first
2. **Check report**: Always review the HTML report for visualization
3. **Download PDB**: Use molecular viewers for detailed analysis
4. **Customize config**: Adjust parameters for your needs
5. **Use web interface**: Easier for interactive exploration

## Need Help?

- Check `README.md` for overview
- Review `docs/` directory for detailed documentation
- Check `config.yaml` for configuration options
- Review code comments for implementation details

