# Data Collection Guide for VLab

## Overview

This guide explains how to collect training data for the VLab viability prediction model. The system collects:
1. **Viable viral genomes** - Real, functional viral genomes from NCBI
2. **Non-viable genomes** - Synthetic and mutated genomes that won't synthesize functional viruses

## Quick Start

### Basic Collection

```bash
python3 collect_training_data.py --email your.email@example.com
```

This will:
- Harvest up to 50,000 viable viral genomes from NCBI
- Generate 10,000 synthetic non-viable genomes
- Generate 5,000 mutated non-viable genomes
- Organize data into train/val splits (90/10)

### Large-Scale Collection

```bash
python3 collect_training_data.py \
    --email your.email@example.com \
    --api_key YOUR_NCBI_API_KEY \
    --max_viable 100000 \
    --num_synthetic 20000 \
    --num_mutated 10000
```

### Quick Test (Small Dataset)

```bash
python3 collect_training_data.py \
    --email your.email@example.com \
    --max_viable 1000 \
    --num_synthetic 500 \
    --num_mutated 250
```

## Data Sources

### Viable Genomes

Collected from NCBI using comprehensive search terms:
- RefSeq complete genomes (highest quality)
- Complete genomes by viral family
- Representative genomes
- Reference genomes

### Non-Viable Genomes

Generated using multiple methods:

1. **Random** - Completely random sequences
2. **Fragmented** - Missing essential parts
3. **No Start** - No start codons (can't initiate translation)
4. **No Stop** - No stop codons (can't terminate translation)
5. **Invalid Codons** - Contains invalid bases (N's)
6. **Too Short** - Too short to be functional
7. **Missing Genes** - Fragment-like sequences
8. **Mutated** - Mutated viable genomes (deleted starts, inserted stops, frame shifts)

## Data Organization

After collection, data is organized as:

```
data/training/
  train/
    viable/
      *.fasta          # Viable genomes for training
    non_viable/
      *.fasta          # Non-viable genomes for training
  val/
    viable/
      *.fasta          # Viable genomes for validation
    non_viable/
      *.fasta          # Non-viable genomes for validation
```

## NCBI Setup

### Email (Required)

NCBI requires a valid email address. Use the `--email` flag:

```bash
python3 collect_training_data.py --email your.email@example.com
```

### API Key (Optional but Recommended)

An NCBI API key speeds up downloads and increases rate limits:

1. Get API key: https://www.ncbi.nlm.nih.gov/account/settings/
2. Use with `--api_key`:

```bash
python3 collect_training_data.py --email your.email@example.com --api_key YOUR_API_KEY
```

## Collection Process

### Step 1: Harvest Viable Genomes

The system searches NCBI for:
- Complete viral genomes
- Reference genomes
- Representative genomes
- Genomes from major viral families

**Time**: ~2-10 hours depending on number of genomes

### Step 2: Generate Synthetic Non-Viable

Creates synthetic non-viable genomes using various methods:
- Random sequences
- Fragmented genomes
- Sequences missing essential elements

**Time**: ~1-5 minutes (fast)

### Step 3: Generate Mutated Non-Viable

Mutates viable genomes to make them non-viable:
- Delete start codons
- Insert stop codons
- Frame shifts
- Sequence corruption

**Time**: ~5-30 minutes

## Monitoring Progress

### Checkpointing

The system automatically saves progress:
- Checkpoint file: `data/training/harvest_checkpoint.json`
- Resumes from last checkpoint if interrupted
- No data loss on interruption

### Logs

Progress is logged to:
- Console output
- `logs/data_collection.log`

### Statistics

Check data statistics:

```python
from src.vlab.data.collector import DataCollector

collector = DataCollector()
stats = collector.get_data_statistics()
print(stats)
```

## Troubleshooting

### Rate Limiting

If you hit NCBI rate limits:
- Add `--api_key` for higher limits
- Reduce `--max_concurrent` in the code
- Wait and resume (checkpointing handles this)

### Memory Issues

If running out of memory:
- Reduce batch sizes in the code
- Process in smaller chunks
- Use `--max_viable` to limit collection

### Network Issues

If downloads fail:
- Check internet connection
- Retry (checkpointing will skip completed)
- Use `--api_key` for more stable connections

## Data Quality

### Quality Filters

Collected genomes are filtered for:
- Valid length (1,000 - 500,000 bp)
- Valid GC content (10% - 80%)
- Valid nucleotide sequences
- Complete genomes (preferred)

### Statistics

After collection, check data quality:

```bash
# Count genomes
find data/training -name "*.fasta" | wc -l

# Check sizes
du -sh data/training/*
```

## Next Steps

After data collection:

1. **Train Model**:
   ```bash
   python3 src/vlab/training/viability_trainer.py --data_dir data/training
   ```

2. **Validate Data**:
   - Check train/val splits
   - Verify viable/non-viable balance
   - Review sample sequences

3. **Use Model**:
   - Update `config.yaml` with model path
   - Run VLab analysis

## Advanced Usage

### Custom Collection

```python
from src.vlab.data.collector import DataCollector
import asyncio

collector = DataCollector(email="your.email@example.com", api_key="YOUR_KEY")

# Collect only viable genomes
viable = await collector.harvester.harvest_all_viable_genomes(max_sequences=10000)

# Generate custom non-viable
collector.harvester.generate_non_viable_genomes(
    count=5000,
    methods=['random', 'fragmented', 'no_start']
)
```

### Incremental Collection

The system supports incremental collection:
- Checkpoint file tracks completed IDs
- Resume interrupted collections
- Skip already downloaded genomes

## Performance

### Expected Times

- **50,000 viable genomes**: ~4-8 hours
- **10,000 synthetic non-viable**: ~5 minutes
- **5,000 mutated non-viable**: ~15 minutes

### Resource Usage

- **CPU**: Moderate (multi-threaded)
- **Memory**: ~2-4 GB
- **Disk**: ~10-50 GB (depending on collection size)
- **Network**: High bandwidth recommended

## Best Practices

1. **Start Small**: Test with `--max_viable 1000` first
2. **Use API Key**: Significantly faster
3. **Monitor Progress**: Check logs regularly
4. **Backup Data**: Save collected data
5. **Validate**: Check data quality before training

## Support

For issues:
- Check logs: `logs/data_collection.log`
- Review checkpoint: `data/training/harvest_checkpoint.json`
- GitHub Issues: [repository-url]/issues

## References

- NCBI Entrez API: https://www.ncbi.nlm.nih.gov/books/NBK25497/
- NCBI API Keys: https://www.ncbi.nlm.nih.gov/account/settings/
- BioPython: https://biopython.org/

