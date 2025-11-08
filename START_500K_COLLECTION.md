# Start 500,000 Genome Collection - Quick Reference

## ✅ Configuration Complete

Your system is now configured to collect **500,000 genomes** with:
- ✅ Your API key: `d7e5c7978697a8c4284af0fc71ce1a2b9808`
- ✅ Your email: `anton.valov05@gmail.com`
- ✅ Balanced dataset: 250,000 viable + 250,000 non-viable
- ✅ Auto-calculation: System will balance automatically

## 🚀 Start Collection

### Option 1: Simple (Recommended)

```bash
python3 collect_training_data.py --total_target 500000
```

### Option 2: Using Script

```bash
./start_large_collection.sh
```

### Option 3: Background Process

```bash
nohup python3 collect_training_data.py --total_target 500000 > collection.log 2>&1 &
```

## 📊 What Will Be Collected

- **250,000 Viable Genomes**: Real viral genomes from NCBI
  - Successfully synthesized or naturally existing
  - Complete genomes, reference genomes, RefSeq genomes
  
- **250,000 Non-Viable Genomes**: Generated examples
  - **175,000 Synthetic**: Random, fragmented, missing essential elements
  - **75,000 Mutated**: Mutated from viable genomes (deleted starts, inserted stops, frame shifts)

## ⏱️ Expected Timeline

- **Viable genomes (250k)**: ~20-40 hours
- **Non-viable genomes (250k)**: ~2-3 hours
- **Total**: ~24-48 hours

## 💾 Disk Space Required

- **Total**: ~500 GB - 1 TB
- **Viable**: ~250-500 GB
- **Non-viable**: ~250-500 GB

## 📈 Monitor Progress

### Check Progress

```bash
# Count collected genomes
find data/training -name "*.fasta" | wc -l

# Watch log file
tail -f logs/data_collection.log

# Check disk usage
du -sh data/training/
```

### Python Check

```python
from src.vlab.data.collector import DataCollector

collector = DataCollector()
stats = collector.get_data_statistics()
print(f"Collected: {stats['total']:,} genomes")
print(f"  Viable: {stats['train_viable'] + stats['val_viable']:,}")
print(f"  Non-viable: {stats['train_non_viable'] + stats['val_non_viable']:,}")
```

## 🔄 Checkpointing

- **Automatic**: Saves every 1,000 sequences
- **Resume**: Just rerun the command if interrupted
- **No data loss**: Checkpoint file tracks progress

## ✅ After Collection

1. **Verify**: Check that you have ~500,000 genomes
2. **Train**: Run training pipeline
3. **Use**: Update config.yaml with model path

## 📝 Notes

- Collection will take 24-48 hours - be patient
- System handles interruptions automatically
- API key is configured for faster downloads
- Balanced 50/50 split for optimal training

## 🎯 Ready to Start?

```bash
python3 collect_training_data.py --total_target 500000
```

Good luck! 🚀

