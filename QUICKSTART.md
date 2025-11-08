# VSim Quick Start Guide

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd Project-VSim
   ```

2. **Run the setup script:**
   ```bash
   python setup.py
   ```

   Or manually install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import Bio; print('BioPython installed successfully')"
   ```

## Usage

### Command Line Interface

Analyze a viral genome:
```bash
python src/main.py data/raw/sample_genome.fasta --output results/
```

Options:
- `--output, -o`: Output directory (default: results/)
- `--config, -c`: Configuration file (default: config.yaml)
- `--verbose, -v`: Verbose output

### Web Interface

Start the web server:
```bash
python src/web/app.py
```

Then open your browser to:
```
http://localhost:8080
```

Upload a FASTA file through the web interface and view results in real-time.

### API Usage

```python
import requests

# Upload and analyze
files = {'file': open('genome.fasta', 'rb')}
response = requests.post('http://localhost:8080/api/analyze', files=files)
results = response.json()

# Get results
results = requests.get('http://localhost:8080/api/results').json()

# Get report
report = requests.get('http://localhost:8080/api/report')
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Output

After analysis, you'll find:

1. **HTML Report**: `results/comprehensive_report.html`
   - Visual summary with key metrics
   - Detailed analysis sections
   - Interactive charts (if implemented)

2. **JSON Results**: `results/results.json`
   - Machine-readable format
   - Complete analysis data
   - Suitable for further processing

## Input Format

VSim accepts FASTA format files:

```
>sequence_id optional description
ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
...
```

## Configuration

Edit `config.yaml` to customize:

- Analysis parameters (ORF length, confidence thresholds)
- Structure prediction method
- Environmental conditions
- Cancer cell types
- ML model settings

## Troubleshooting

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.9+ required)

### File Not Found
- Ensure genome file exists and path is correct
- Check file permissions

### Analysis Errors
- Verify genome file is valid FASTA format
- Check file contains valid nucleotide sequences (A, T, G, C, N)

## Support

For detailed documentation, see:
- `docs/PHASE_1_GENOME_ANALYSIS.md`
- `docs/PHASE_2_STRUCTURE_PREDICTION.md`
- `docs/PHASE_3_ENVIRONMENTAL.md`
- `docs/PHASE_4_CELL_INTERACTIONS.md`
- `docs/PHASE_5_CANCER_ANALYSIS.md`
- `docs/PHASE_6_ML_PREDICTION.md`
- `docs/PHASE_7_INTEGRATION.md`

