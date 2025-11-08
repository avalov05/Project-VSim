# Quick Start: Data Collection for VLab

## Overview

Collect training data for the VLab viability prediction model in 3 steps:

1. **Collect viable genomes** - Real viral genomes from NCBI
2. **Generate non-viable genomes** - Synthetic and mutated genomes
3. **Train model** - Train the viability predictor

## Step 1: Collect Data

### Basic (Recommended for First Run)

```bash
python3 collect_training_data.py --email your.email@example.com
```

This collects:
- 50,000 viable genomes
- 10,000 synthetic non-viable
- 5,000 mutated non-viable

### Fast Test (5 minutes)

```bash
python3 collect_training_data.py \
    --email your.email@example.com \
    --max_viable 1000 \
    --num_synthetic 500 \
    --num_mutated 250
```

### Large Scale (24+ hours)

```bash
python3 collect_training_data.py \
    --email your.email@example.com \
    --api_key YOUR_NCBI_API_KEY \
    --max_viable 100000 \
    --num_synthetic 20000 \
    --num_mutated 10000
```

## Step 2: Train Model

```bash
python3 src/vlab/training/viability_trainer.py \
    --data_dir data/training \
    --epochs 50 \
    --batch_size 32
```

## Step 3: Use Trained Model

Update `config.yaml`:

```yaml
viability_model_path: models/viability_model_best.pth
```

Then run VLab:

```bash
python3 src/vlab_main.py data/raw/genome.fasta
```

## Data Location

```
data/training/
  train/
    viable/          # Viable genomes for training
    non_viable/      # Non-viable genomes for training
  val/
    viable/          # Viable genomes for validation
    non_viable/      # Non-viable genomes for validation
```

## Expected Times

- **1,000 genomes**: ~10 minutes
- **10,000 genomes**: ~1 hour
- **50,000 genomes**: ~4-8 hours
- **100,000 genomes**: ~12-24 hours

## Tips

1. **Use API Key**: Significantly faster (get from https://www.ncbi.nlm.nih.gov/account/settings/)
2. **Start Small**: Test with 1,000 genomes first
3. **Checkpointing**: System saves progress automatically
4. **Resume**: If interrupted, just rerun - it will resume

## Troubleshooting

### Rate Limiting
- Add `--api_key` for higher limits
- Wait and resume (checkpointing handles this)

### Memory Issues
- Reduce `--max_viable`
- Process in smaller batches

### Network Issues
- Check connection
- Use `--api_key` for stability
- Retry (checkpointing skips completed)

## Next Steps

1. Collect data (this guide)
2. Train model (see training guide)
3. Use model in VLab (see VLAB_README.md)

