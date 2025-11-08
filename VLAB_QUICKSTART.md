# VLab Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Analyze a Viral Genome

```bash
python3 src/vlab_main.py data/raw/sample_genome.fasta
```

This will:
- Annotate the genome
- Predict viability
- Predict protein structures
- Assemble 3D virion model
- Predict hosts and tropism
- Simulate infection dynamics
- Generate comprehensive reports

### 2. View Results

```bash
# Open HTML report
open results/report.html

# View 3D model
open results/viewer_3d_detailed.html
# Or use the quick viewer:
python3 quick_view_3d.py
```

### 3. Training Viability Model

#### Step 1: Prepare Data

```bash
# Create directory structure
mkdir -p data/training/train/viable
mkdir -p data/training/train/non_viable
mkdir -p data/training/val/viable
mkdir -p data/training/val/non_viable

# Add your genomes:
# - Viable genomes in data/training/train/viable/
# - Non-viable genomes in data/training/train/non_viable/
```

#### Step 2: Generate Synthetic Data (Optional)

```python
from src.vlab.data.collector import DataCollector
from pathlib import Path

collector = DataCollector()
collector.generate_synthetic_non_viable(
    count=1000,
    output_dir=Path("data/training/train/non_viable")
)
```

#### Step 3: Train Model

```bash
python3 src/vlab/training/viability_trainer.py \
    --data_dir data/training \
    --epochs 50 \
    --batch_size 32
```

#### Step 4: Use Trained Model

Update `config.yaml`:

```yaml
viability_model_path: models/viability_model_best.pth
```

## Configuration

Create `config.yaml`:

```yaml
# Output settings
output_dir: results
save_intermediates: false

# Computational resources
use_gpu: true
gpu_id: 0
num_workers: 4
batch_size: 8

# AlphaFold (optional)
use_alphafold: false
alphafold_path: null

# Viability prediction
viability_model_path: null  # Set after training
viability_threshold: 0.5
confidence_threshold: 0.8

# Assembly
assembly_method: hybrid  # "md", "ml", or "hybrid"

# Infection simulation
simulate_infection: true
simulation_time: 24.0  # hours

# Output
generate_3d_model: true
generate_report: true
visualization_quality: high
```

## Example Workflow

### Complete Analysis

```bash
# 1. Analyze genome
python3 src/vlab_main.py data/raw/sars_cov2.fasta --output results/sars_cov2

# 2. View results
open results/sars_cov2/report.html

# 3. Check 3D model
python3 quick_view_3d.py
```

### Programmatic Usage

```python
from src.vlab import VLabPipeline, VLabConfig
from pathlib import Path

# Load config
config = VLabConfig.from_file(Path("config.yaml"))

# Create pipeline
pipeline = VLabPipeline(config)

# Analyze genome
results = pipeline.analyze(Path("data/raw/genome.fasta"))

# Access results
print(f"Viability: {results.viability['score']:.3f}")
print(f"Human Risk: {results.host_prediction['human_infection_risk']:.3f}")
print(f"3D Model: {results.assembly['pdb_file']}")
```

## Output Files

```
results/
  report.json              # Complete JSON report
  report.html              # HTML report with visualization
  structures/
    *.pdb                  # Individual protein structures
    virus_assembled.pdb    # Complete virion 3D model
  checkpoints/             # Intermediate results (if enabled)
```

## Key Features

### 1. Viability Prediction

Determines if a genome will synthesize a functional virus:

- **ML Model**: Deep learning model (requires training)
- **Rule-Based**: Fallback system (works out of the box)
- **Confidence Scoring**: Indicates prediction reliability
- **Detailed Reasoning**: Explains why genome is viable/non-viable

### 2. 3D Structure

Generates complete 3D virion models:

- **Geometry Prediction**: Shape (spherical, filamentous, icosahedral)
- **Assembly Simulation**: Physics-based or ML-guided
- **Visualization**: Interactive 3D viewer
- **PDB Export**: Standard format for external viewers

### 3. Host Prediction

Predicts host species and cell tropism:

- **Host Species**: Likely host organisms
- **Receptor Binding**: Cell surface receptors
- **Tropism**: Cell types and tissues
- **Human Risk**: Infection risk assessment

### 4. Infection Simulation

Simulates viral infection dynamics:

- **Life Cycle**: Entry, replication, assembly, egress
- **Replication Dynamics**: Time-course simulation
- **Burst Size**: Virions per cell
- **Replication Time**: Cycle duration

## Troubleshooting

### GPU Not Available

```yaml
# In config.yaml
use_gpu: false
```

### Out of Memory

```yaml
# Reduce batch size
batch_size: 4

# Or disable intermediates
save_intermediates: false
```

### Structure Prediction Fails

```yaml
# Use simulated structures
use_alphafold: false
```

### Model Training Issues

```bash
# Reduce batch size
python3 src/vlab/training/viability_trainer.py --batch_size 16

# Or use CPU
# Set use_gpu: false in config
```

## Next Steps

1. **Train Models**: Collect data and train viability/host models
2. **Install AlphaFold**: For high-accuracy structure prediction
3. **Collect Data**: Build training datasets from public databases
4. **Customize**: Modify models and parameters for your needs

## Support

- Documentation: See `VLAB_README.md`
- Architecture: See `ARCHITECTURE.md`
- Issues: GitHub Issues

