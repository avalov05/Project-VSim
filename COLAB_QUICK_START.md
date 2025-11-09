# Quick Start: Run VSim Training in Colab (5 Minutes)

## 🚀 EASIEST METHOD: One-Cell Setup

**Just want to paste and run?** See [`COLAB_ONE_CELL.md`](COLAB_ONE_CELL.md) for a single code cell that does everything!

---

## Method 1: Step-by-Step Setup

### Step 1: Open Colab & Enable GPU

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **"New notebook"**
3. **Runtime → Change runtime type → GPU → Save**

## Step 2: Clone & Setup (Run These Cells)

```python
# Cell 1: Clone your repo
!git clone https://github.com/YOUR_USERNAME/Project-VSim.git

# Cell 2: Navigate to project
import os
import sys
os.chdir('/content/Project-VSim')
sys.path.insert(0, '/content/Project-VSim')
print(f"✓ Working in: {os.getcwd()}")

# Cell 3: Install dependencies
!pip install -q biopython numpy pandas scipy scikit-learn torch torchvision torchaudio pyyaml requests aiohttp tqdm backoff matplotlib

# Cell 4: Check GPU
import torch
print(f"✓ GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ Device: {torch.cuda.get_device_name(0)}")
```

## Step 3: Configure (Edit These Values)

```python
# Cell 5: Configuration
EMAIL = "your-email@example.com"  # ← Change this
API_KEY = "your-api-key"  # ← Change this (or leave empty)
TOTAL_TARGET = 10000  # ← Start with 10000 for testing (500000 for full)

# Model settings (already optimized for quality)
INPUT_DIM = 1024
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 16
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 20
GRADIENT_CLIP_VAL = 1.0
USE_FOCAL_LOSS = True
USE_MIXED_PRECISION = True
```

## Step 4: Run Training

**Option A: Use the Notebook File**
```python
# If you have VSim_Train_Model_Colab.ipynb in your repo:
# Just open it and run all cells (Runtime → Run all)
```

**Option B: Run Cells Manually**

```python
# Cell 6: Data Collection
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sys.path.insert(0, str(Path.cwd()))

from src.vlab.data.collector import DataCollector

collector = DataCollector(email=EMAIL, api_key=API_KEY)
results = asyncio.run(collector.collect_all_data(
    max_viable=None,
    num_synthetic_non_viable=None,
    num_mutated_non_viable=None,
    total_target=TOTAL_TARGET
))

stats = collector.get_data_statistics()
print(f"✓ Collected {stats['total']:,} genomes")

# Cell 7: Training
import torch
from src.vlab.training.viability_trainer import ViabilityTrainer, collect_training_data
from src.vlab.core.config import VLabConfig

config = VLabConfig()
config.use_gpu = torch.cuda.is_available()
config.models_dir = Path('models')
config.models_dir.mkdir(parents=True, exist_ok=True)

train_annotations, train_labels, val_annotations, val_labels = collect_training_data(Path('data/training'))

trainer = ViabilityTrainer(
    config,
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    use_focal_loss=USE_FOCAL_LOSS,
    use_mixed_precision=USE_MIXED_PRECISION
)

trainer.train(
    train_annotations, train_labels,
    val_annotations, val_labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_epochs=WARMUP_EPOCHS,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    gradient_clip_val=GRADIENT_CLIP_VAL
)

trainer.save_model(config.models_dir / "viability_model_final.pth")
print("✓ Training complete!")

# Cell 8: Download Model
from google.colab import files
import zipfile

models_dir = Path('models')
output_zip = '/tmp/vsim_model.zip'

with zipfile.ZipFile(output_zip, 'w') as zipf:
    for model_file in models_dir.glob('*.pth'):
        zipf.write(model_file, model_file.name)

files.download(output_zip)
print("✓ Model downloaded!")
```

## That's It!

**Timeline:**
- Setup: 5 minutes
- Data Collection: 1-3 hours (test) or 24-48 hours (full)
- Training: 1-2 hours (test) or 10-20 hours (full)
- **Total Test Run: 2-5 hours**
- **Total Full Run: 34-68 hours**

## Pro Tips

1. **Start Small**: Use `TOTAL_TARGET = 10000` first to test
2. **Save to Drive**: Mount Google Drive to persist data between sessions
3. **Monitor Progress**: Watch the output for epoch-by-epoch metrics
4. **Download Immediately**: Models are lost when Colab session ends

## Troubleshooting

- **"Module not found"**: Run `os.chdir('/content/Project-VSim')` again
- **"Out of memory"**: Reduce `BATCH_SIZE` to 32 or 16
- **"Session crashed"**: Colab sessions can timeout after 12 hours
- **"Git clone failed"**: Check your repository URL is correct

## Need Help?

See `COLAB_SETUP_GUIDE.md` for detailed instructions.

