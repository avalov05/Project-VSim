# Data Collection System - Complete Summary

## What Was Created

A comprehensive data collection system that:

1. **Harvests ALL viable viral genomes** from NCBI (up to 100,000+)
2. **Generates non-viable genomes** using multiple methods
3. **Organizes data** for training (train/val splits)
4. **Integrates seamlessly** with VLab training pipeline

## Key Features

### 1. Comprehensive Viable Genome Collection

- **Sources**: NCBI nucleotide database
- **Search Strategy**: Multi-tier approach
  - Tier 1: RefSeq complete genomes (highest quality)
  - Tier 2: Complete genomes by viral family
  - Tier 3: Broad complete genomes
  - Tier 4: Representative genomes
  - Tier 5: Reference genomes
- **Quality Filters**: Length, GC content, valid sequences
- **Pagination**: Handles NCBI's 10,000 ID limit
- **Checkpointing**: Resumes interrupted downloads

### 2. Non-Viable Genome Generation

**Synthetic Methods**:
- **Random**: Completely random sequences
- **Fragmented**: Missing essential parts
- **No Start**: No start codons (can't initiate)
- **No Stop**: No stop codons (can't terminate)
- **Invalid Codons**: Contains N's and invalid bases
- **Too Short**: Too short to be functional
- **Missing Genes**: Fragment-like sequences

**Mutation Methods** (from viable genomes):
- **Delete Start**: Remove all start codons
- **Insert Stop**: Insert premature stop codons
- **Frame Shift**: Insert/delete bases to shift reading frame
- **Corrupt**: Randomly corrupt sequence

### 3. Data Organization

```
data/training/
  train/
    viable/              # Viable genomes for training (90%)
    non_viable/          # Non-viable genomes for training (90%)
  val/
    viable/              # Viable genomes for validation (10%)
    non_viable/          # Non-viable genomes for validation (10%)
```

## Usage

### Quick Start

```bash
# Basic collection
python3 collect_training_data.py --email your.email@example.com

# Large scale
python3 collect_training_data.py \
    --email your.email@example.com \
    --api_key YOUR_API_KEY \
    --max_viable 100000 \
    --num_synthetic 20000 \
    --num_mutated 10000
```

### Programmatic Usage

```python
from src.vlab.data.collector import DataCollector
import asyncio

collector = DataCollector(email="your.email@example.com", api_key="YOUR_KEY")

results = asyncio.run(collector.collect_all_data(
    max_viable=50000,
    num_synthetic_non_viable=10000,
    num_mutated_non_viable=5000
))
```

## Architecture

### Components

1. **ViralGenomeHarvester** (`viral_harvester.py`)
   - Harvests viable genomes from NCBI
   - Generates synthetic non-viable genomes
   - Generates mutated non-viable genomes
   - Handles checkpointing and resumption

2. **DataCollector** (`collector.py`)
   - Main interface for data collection
   - Coordinates harvesting and generation
   - Provides statistics and monitoring

3. **Main Script** (`collect_training_data.py`)
   - Command-line interface
   - Handles arguments and logging
   - Provides progress monitoring

## Data Collection Process

### Step 1: Harvest Viable Genomes

1. Search NCBI with comprehensive terms
2. Use WebEnv for pagination (bypasses 10k limit)
3. Download sequences asynchronously
4. Filter for quality
5. Save to training directories
6. Split train/val (90/10)

### Step 2: Generate Synthetic Non-Viable

1. Generate sequences using various methods
2. Ensure they're non-viable (no start codons, etc.)
3. Save to training directories
4. Split train/val (90/10)

### Step 3: Generate Mutated Non-Viable

1. Select random viable genomes
2. Apply mutations (delete starts, insert stops, etc.)
3. Ensure mutations make them non-viable
4. Save to training directories
5. Split train/val (90/10)

## Performance

### Expected Times

- **1,000 genomes**: ~10 minutes
- **10,000 genomes**: ~1 hour
- **50,000 genomes**: ~4-8 hours
- **100,000 genomes**: ~12-24 hours

### Resource Usage

- **CPU**: Moderate (multi-threaded)
- **Memory**: ~2-4 GB
- **Disk**: ~10-50 GB (depending on collection size)
- **Network**: High bandwidth recommended

## Quality Assurance

### Quality Filters

- **Length**: 1,000 - 500,000 bp
- **GC Content**: 10% - 80%
- **Valid Sequences**: No invalid characters
- **Complete Genomes**: Preferred over fragments

### Validation

- Checkpointing ensures data integrity
- Resumption handles interruptions
- Statistics track collection progress
- Logs provide detailed information

## Integration with VLab

### Training Pipeline

After data collection, train the model:

```bash
python3 src/vlab/training/viability_trainer.py --data_dir data/training
```

### Model Usage

Update `config.yaml`:

```yaml
viability_model_path: models/viability_model_best.pth
```

Then use in VLab:

```bash
python3 src/vlab_main.py data/raw/genome.fasta
```

## Statistics

### Data Distribution

- **Train/VAL Split**: 90/10
- **Viable/Non-Viable**: Balanced (1:1 recommended)
- **Methods**: Multiple methods for non-viable generation

### Expected Numbers

For a complete training set:
- **Viable**: 50,000 - 100,000
- **Synthetic Non-Viable**: 10,000 - 20,000
- **Mutated Non-Viable**: 5,000 - 10,000
- **Total**: 65,000 - 130,000 genomes

## Troubleshooting

### Common Issues

1. **Rate Limiting**: Use API key, reduce concurrency
2. **Memory Issues**: Reduce batch sizes, limit collection
3. **Network Issues**: Check connection, use API key
4. **Interruptions**: Checkpointing handles resumption

### Solutions

- **API Key**: Significantly improves performance
- **Checkpointing**: Automatic resumption
- **Logging**: Detailed progress tracking
- **Statistics**: Monitor collection progress

## Next Steps

1. **Collect Data**: Run `collect_training_data.py`
2. **Train Model**: Run training pipeline
3. **Validate Model**: Check accuracy metrics
4. **Use Model**: Integrate with VLab

## Documentation

- **Quick Start**: `QUICK_START_DATA_COLLECTION.md`
- **Detailed Guide**: `DATA_COLLECTION_GUIDE.md`
- **This Summary**: `DATA_COLLECTION_SUMMARY.md`

## Support

For issues:
- Check logs: `logs/data_collection.log`
- Review checkpoint: `data/training/harvest_checkpoint.json`
- GitHub Issues: [repository-url]/issues

## Conclusion

The data collection system provides:
- **Comprehensive**: Collects all viable viral genomes
- **Diverse**: Multiple methods for non-viable generation
- **Reliable**: Checkpointing and error handling
- **Scalable**: Handles large-scale collection
- **Integrated**: Seamless integration with VLab

This enables training of high-accuracy viability prediction models (>95% target accuracy).

