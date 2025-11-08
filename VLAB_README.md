# Virtual In Silico Virus Laboratory (VLab)

A comprehensive computational virology platform that analyzes any viral genome and predicts:
- **Viability** - Will the genome synthesize a functional virus?
- **3D Structure** - Complete virion assembly and visualization
- **Host Interactions** - Host species prediction and cell tropism
- **Infection Dynamics** - Replication and infection simulation
- **Safety Metrics** - Comprehensive risk assessment

## Features

### 🎯 Core Capabilities

1. **Genome Annotation**
   - ORF prediction in all reading frames
   - Gene function prediction
   - Regulatory element detection
   - Codon usage analysis

2. **Viability Prediction**
   - Deep learning model for viability assessment
   - Rule-based fallback system
   - Confidence scoring
   - Detailed reasoning

3. **Structure Prediction**
   - AlphaFold integration (when available)
   - Protein folding pipeline
   - Complex prediction (multimers, capsids)
   - 3D model generation

4. **Virion Assembly**
   - Physics-based assembly simulation
   - Geometry prediction (icosahedral, helical, spherical, filamentous)
   - Complete 3D virion models
   - Envelope and spike modeling

5. **Host Prediction**
   - Host species prediction
   - Receptor binding prediction
   - Cell and tissue tropism
   - Human infectivity risk assessment

6. **Infection Simulation**
   - Viral life cycle modeling
   - Replication dynamics
   - Burst size prediction
   - Time-course simulation

7. **Comprehensive Reporting**
   - JSON and HTML reports
   - Interactive 3D visualization
   - Safety recommendations
   - Detailed metrics

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for structure prediction)
- 16GB+ RAM
- 50GB+ disk space

### Setup

```bash
# Clone repository
git clone <repository-url>
cd Project-VSim

# Install dependencies
pip install -r requirements.txt

# (Optional) Install AlphaFold for structure prediction
# Follow AlphaFold installation instructions
```

## Quick Start

### Basic Usage

```bash
# Analyze a viral genome
python3 src/vlab_main.py data/raw/genome.fasta

# With custom output directory
python3 src/vlab_main.py genome.fasta --output my_results/

# Verbose output
python3 src/vlab_main.py genome.fasta --verbose
```

### Configuration

Create a `config.yaml` file:

```yaml
output_dir: results
use_gpu: true
gpu_id: 0
use_alphafold: false  # Set to true if AlphaFold is installed
viability_threshold: 0.5
confidence_threshold: 0.8
simulate_infection: true
generate_3d_model: true
```

## Training Viability Model

### Data Preparation

Organize training data:

```
data/
  training/
    train/
      viable/
        *.fasta
      non_viable/
        *.fasta
    val/
      viable/
        *.fasta
      non_viable/
        *.fasta
```

### Training

```bash
# Train viability model
python3 src/vlab/training/viability_trainer.py \
    --data_dir data/training \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4
```

### Data Collection

```python
from src.vlab.data.collector import DataCollector

collector = DataCollector()
collector.collect_viable_genomes(Path("data/training/train/viable"))
collector.generate_synthetic_non_viable(1000, Path("data/training/train/non_viable"))
```

## Architecture

### Pipeline Flow

1. **Genome Annotation** → Annotates genome, predicts genes
2. **Viability Prediction** → Determines if genome is viable
3. **Structure Prediction** → Folds proteins (AlphaFold or simulated)
4. **Assembly Simulation** → Assembles complete virion
5. **Host Prediction** → Predicts hosts and tropism
6. **Infection Simulation** → Simulates infection dynamics
7. **Metrics & Reporting** → Generates comprehensive reports

### Key Modules

- `vlab/core/` - Core framework and pipeline
- `vlab/annotation/` - Genome annotation
- `vlab/viability/` - Viability prediction
- `vlab/structure/` - Structure prediction
- `vlab/assembly/` - Virion assembly
- `vlab/host/` - Host prediction
- `vlab/infection/` - Infection simulation
- `vlab/metrics/` - Metrics and reporting
- `vlab/training/` - Training pipelines
- `vlab/data/` - Data collection

## Output

### Results Structure

```
results/
  report.json              # Complete analysis results
  report.html              # HTML report
  structures/
    *.pdb                  # Protein structures
    virus_assembled.pdb    # Complete virion model
  checkpoints/             # Intermediate results (if enabled)
```

### Report Contents

- **Viability Score** - Probability genome is viable (0-1)
- **Safety Assessment** - Human infection risk and recommendations
- **Structural Properties** - Shape, diameter, symmetry
- **Host Prediction** - Predicted hosts with probabilities
- **Infection Dynamics** - Burst size, replication time
- **3D Model** - Interactive visualization

## Accuracy & Performance

### Target Accuracy

- **Viability Prediction**: >95% accuracy (with trained model)
- **Structure Prediction**: High confidence with AlphaFold
- **Host Prediction**: >80% accuracy
- **Assembly**: Realistic 3D models

### Performance

- **Runtime**: <24 hours on single A100 GPU
- **Memory**: 16GB+ RAM recommended
- **Storage**: 50GB+ for models and results

## Advanced Usage

### Custom Models

```python
from src.vlab import VLabPipeline, VLabConfig

config = VLabConfig()
config.viability_model_path = Path("models/custom_viability.pth")
config.host_model_path = Path("models/custom_host.pth")

pipeline = VLabPipeline(config)
results = pipeline.analyze(Path("genome.fasta"))
```

### Programmatic Access

```python
from src.vlab import VLabPipeline, VLabConfig

config = VLabConfig.from_file(Path("config.yaml"))
pipeline = VLabPipeline(config)

results = pipeline.analyze(Path("genome.fasta"))

print(f"Viability: {results.viability['score']:.3f}")
print(f"Human Risk: {results.host_prediction['human_infection_risk']:.3f}")
print(f"3D Model: {results.assembly['pdb_file']}")
```

## Limitations

1. **Structure Prediction**: Requires AlphaFold for high-accuracy structures
2. **Assembly**: Simplified models for complex viruses
3. **Host Prediction**: Limited by training data
4. **Infection Simulation**: Simplified models

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## License

[Your License Here]

## Citation

If you use VLab in your research, please cite:

```
Virtual In Silico Virus Laboratory (VLab)
Comprehensive Computational Virology Platform
```

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: [docs-url]
- Email: [support-email]

## Acknowledgments

- AlphaFold team for structure prediction
- NCBI for viral genome data
- BioPython for sequence analysis
- PyTorch for deep learning

