# Step-by-Step Guide: Running VSim Training in Google Colab

This guide walks you through running the training notebook in Google Colab after uploading your project to Git.

## Prerequisites

1. Your project is uploaded to GitHub (or another Git repository)
2. You have a Google account
3. Access to Google Colab (free tier works, but GPU is recommended)

## Step 1: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account if needed
3. Click **"New notebook"** or **"File → New notebook"**

## Step 2: Enable GPU (Recommended)

1. Click **"Runtime"** in the top menu
2. Select **"Change runtime type"**
3. Set **"Hardware accelerator"** to **"GPU"** (T4 or better if available)
4. Click **"Save"**

**Note**: GPU is highly recommended for training. Without GPU, training will be very slow (10-100x slower).

## Step 3: Clone Your Repository

In the first cell of your notebook, run:

```python
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/Project-VSim.git
```

Replace `YOUR_USERNAME` with your GitHub username.

**Alternative**: If your repo is private, you'll need to authenticate:
```python
# For private repositories
from google.colab import drive
drive.mount('/content/drive')

# Or use personal access token
!git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/Project-VSim.git
```

## Step 4: Navigate to Project Directory

```python
import os
import sys
from pathlib import Path

# Change to project directory
os.chdir('/content/Project-VSim')
sys.path.insert(0, '/content/Project-VSim')

print(f"✓ Working directory: {os.getcwd()}")
```

## Step 5: Upload the Notebook (If Not in Repo)

**Option A**: If the notebook is in your Git repo, skip this step.

**Option B**: If you need to upload the notebook:
1. Click **"File" → "Upload notebook"**
2. Select `VSim_Train_Model_Colab.ipynb`
3. Upload it

**Option C**: Create a new notebook and copy cells from the repository:
1. Open the notebook file from your cloned repo
2. Copy all cells to your Colab notebook

## Step 6: Install Dependencies

Run this in a new cell:

```python
# Install all required packages
!pip install -q biopython numpy pandas scipy scikit-learn torch torchvision torchaudio pyyaml requests aiohttp tqdm backoff matplotlib
```

Wait for installation to complete (usually takes 1-2 minutes).

## Step 7: Verify Setup

```python
# Check GPU availability
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device('cpu')
    print("⚠ GPU not available, using CPU")
    print("⚠ For faster training, enable GPU: Runtime → Change runtime type → GPU")

# Set environment variables
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

## Step 8: Configure Settings

Run the configuration cell (this is Step 4 in the notebook):

```python
# ===== MAXIMUM QUALITY CONFIGURATION =====
# Optimized for maximum model quality (training time not a concern)

# Data Collection Settings
EMAIL = "your-email@example.com"  # Your email for NCBI (required)
API_KEY = "your-api-key-here"  # NCBI API key (optional, speeds up downloads)

# Training Data Settings - MAXIMUM QUALITY
TOTAL_TARGET = 500000  # Full dataset (will take ~24-48 hours for data collection)
# For testing, use smaller numbers:
# TOTAL_TARGET = 10000  # Medium dataset (will take ~1-3 hours)

# Model Architecture Settings - ENHANCED
INPUT_DIM = 1024
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 16

# Training Settings - MAXIMUM QUALITY
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 20
GRADIENT_CLIP_VAL = 1.0

# Advanced Features
USE_FOCAL_LOSS = True
USE_MIXED_PRECISION = True

print("="*70)
print("MAXIMUM QUALITY CONFIGURATION")
print("="*70)
print(f"Email: {EMAIL}")
print(f"API Key: {'Provided' if API_KEY else 'Not provided'}")
print(f"Total Target Genomes: {TOTAL_TARGET:,}")
print("="*70)
```

**Important**: 
- Replace `your-email@example.com` with your actual email
- Replace `your-api-key-here` with your NCBI API key (or leave as empty string if you don't have one)
- For testing, reduce `TOTAL_TARGET` to 10000 or 50000

## Step 9: Run Data Collection

Run the data collection cell (Step 5):

```python
import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from src.vlab.data.collector import DataCollector

print("="*70)
print("STARTING DATA COLLECTION")
print("="*70)
print(f"This will collect {TOTAL_TARGET:,} viral genomes from NCBI")
print("This may take a while depending on your target size...")
print("="*70)
print()

# Create collector
collector = DataCollector(email=EMAIL, api_key=API_KEY)

# Collect data
try:
    results = asyncio.run(collector.collect_all_data(
        max_viable=None,
        num_synthetic_non_viable=None,
        num_mutated_non_viable=None,
        total_target=TOTAL_TARGET
    ))
    
    # Get statistics
    stats = collector.get_data_statistics()
    
    # Print summary
    print("\n" + "="*70)
    print("DATA COLLECTION COMPLETE!")
    print("="*70)
    print(f"Training Data:")
    print(f"  Viable: {stats['train_viable']:,}")
    print(f"  Non-viable: {stats['train_non_viable']:,}")
    print(f"  Total: {stats['total_train']:,}")
    print(f"\nValidation Data:")
    print(f"  Viable: {stats['val_viable']:,}")
    print(f"  Non-viable: {stats['val_non_viable']:,}")
    print(f"  Total: {stats['total_val']:,}")
    print(f"\nGrand Total: {stats['total']:,} genomes")
    print(f"\nData location: data/training/")
    print("="*70)
    
except KeyboardInterrupt:
    logger.error("\nData collection interrupted by user")
    raise
except Exception as e:
    logger.error(f"Data collection failed: {e}", exc_info=True)
    raise
```

**Note**: 
- This will take a long time (hours to days depending on `TOTAL_TARGET`)
- Colab sessions timeout after 12 hours of inactivity, but active sessions can run longer
- Data is saved to `/content/Project-VSim/data/training/`

## Step 10: Run Model Training

After data collection completes, run the training cell (Step 6):

```python
import torch
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from src.vlab.training.viability_trainer import ViabilityTrainer, collect_training_data
from src.vlab.core.config import VLabConfig

print("="*70)
print("STARTING MODEL TRAINING")
print("="*70)
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Device: {device}")
print("="*70)
print()

# Load configuration
config = VLabConfig()
config.use_gpu = torch.cuda.is_available()
config.gpu_id = 0
config.models_dir = Path('models')
config.models_dir.mkdir(parents=True, exist_ok=True)

# Collect training data
data_dir = Path('data/training')
print(f"Loading data from: {data_dir}")

try:
    train_annotations, train_labels, val_annotations, val_labels = collect_training_data(data_dir)
    
    if not train_annotations:
        raise ValueError("No training data found! Please run data collection first.")
    
    print(f"\nLoaded {len(train_annotations)} training samples")
    if val_annotations:
        print(f"Loaded {len(val_annotations)} validation samples")
    
    # Create trainer with enhanced architecture
    print(f"\nCreating enhanced model with:")
    print(f"  Input dim: {INPUT_DIM}, Hidden dim: {HIDDEN_DIM}")
    print(f"  Layers: {NUM_LAYERS}, Heads: {NUM_HEADS}")
    print(f"  Focal loss: {USE_FOCAL_LOSS}, Mixed precision: {USE_MIXED_PRECISION}")
    
    trainer = ViabilityTrainer(
        config,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        use_focal_loss=USE_FOCAL_LOSS,
        use_mixed_precision=USE_MIXED_PRECISION
    )
    
    # Train model with maximum quality settings
    print("\nStarting training with maximum quality configuration...")
    print("This will take a while but will produce the best possible model.")
    print()
    
    trainer.train(
        train_annotations, train_labels,
        val_annotations if val_annotations else None,
        val_labels if val_labels else None,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        gradient_clip_val=GRADIENT_CLIP_VAL
    )
    
    # Save final model
    final_model_path = config.models_dir / "viability_model_final.pth"
    trainer.save_model(final_model_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Model saved to: {final_model_path}")
    
    # Check for best model
    best_model_path = config.models_dir / "viability_model_best.pth"
    if best_model_path.exists():
        print(f"Best model saved to: {best_model_path}")
    
    print("="*70)
    
except Exception as e:
    logger.error(f"Training failed: {e}", exc_info=True)
    raise
```

**Note**: 
- Training will take several hours (10-20 hours for 200 epochs)
- You'll see progress updates every epoch
- Best model is saved automatically based on validation AUC

## Step 11: Verify Model

Run the verification cell (Step 7):

```python
from pathlib import Path
import torch

# Check if model files exist
models_dir = Path('models')

print("="*70)
print("MODEL VERIFICATION")
print("="*70)

final_model = models_dir / "viability_model_final.pth"
best_model = models_dir / "viability_model_best.pth"

if final_model.exists():
    size_mb = final_model.stat().st_size / (1024 * 1024)
    print(f"✓ Final model: {final_model} ({size_mb:.2f} MB)")
    
    # Try loading the model
    try:
        checkpoint = torch.load(final_model, map_location='cpu')
        print(f"✓ Model loaded successfully")
        print(f"  Model class: {checkpoint.get('model_class', 'Unknown')}")
        if 'model_params' in checkpoint:
            params = checkpoint['model_params']
            print(f"  Architecture: dim={params.get('input_dim')}, layers={params.get('num_layers')}")
    except Exception as e:
        print(f"⚠ Error loading model: {e}")
else:
    print(f"⚠ Final model not found: {final_model}")

if best_model.exists():
    size_mb = best_model.stat().st_size / (1024 * 1024)
    print(f"✓ Best model: {best_model} ({size_mb:.2f} MB)")
else:
    print(f"ℹ Best model not found (this is okay if training didn't use validation)")

print("="*70)
```

## Step 12: Download Model

Run the download cell (Step 8):

```python
from google.colab import files
from pathlib import Path
import zipfile
import os

# Create a zip file with the trained models
models_dir = Path('models')
output_zip = '/tmp/vsim_trained_model.zip'

if models_dir.exists():
    print("Creating model archive...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all model files
        for model_file in models_dir.glob('*.pth'):
            zipf.write(model_file, model_file.name)
            print(f"  Added: {model_file.name}")
    
    # Download
    print(f"\nDownloading model archive...")
    files.download(output_zip)
    print("\n✓ Model downloaded successfully!")
    print("\nYou can now use this model in your VSim application.")
else:
    print("⚠ Models directory not found. Please run training first.")
```

This will automatically download the model files to your computer.

## Alternative: Use the Notebook File Directly

If you uploaded the `VSim_Train_Model_Colab.ipynb` file to your repo, you can:

1. **Open it directly in Colab**:
   - Go to your GitHub repo
   - Click on `VSim_Train_Model_Colab.ipynb`
   - Click "Open in Colab" button (if available)
   - Or copy the URL and open it in Colab

2. **Upload to Colab**:
   - In Colab, click "File → Upload notebook"
   - Select `VSim_Train_Model_Colab.ipynb`
   - Run all cells in order

## Important Notes

### Session Management
- **Colab sessions timeout** after 12 hours of inactivity
- **Active sessions** can run longer (up to 24 hours on free tier)
- **Save your work** regularly (download models, save checkpoints)

### Data Persistence
- **Data is NOT saved** between Colab sessions (unless you save to Google Drive)
- **Models are NOT saved** automatically (download them!)
- **Use Google Drive** if you need persistence:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  # Save to /content/drive/MyDrive/VSim/models/
  ```

### GPU Limits
- **Free tier**: Limited GPU hours per day
- **Colab Pro**: More GPU hours, better GPUs
- **Monitor usage**: Runtime → Manage sessions

### Troubleshooting

**Issue**: "Module not found"
- **Solution**: Make sure you're in the correct directory (`/content/Project-VSim`)
- Run: `!pwd` to check current directory

**Issue**: "Out of memory"
- **Solution**: Reduce `BATCH_SIZE` (try 32 or 16)
- Reduce `TOTAL_TARGET` for smaller dataset

**Issue**: "Session crashed"
- **Solution**: Colab sessions can crash if they run too long
- Save checkpoints regularly
- Consider using smaller dataset for testing

**Issue**: "Git clone failed"
- **Solution**: Check your repository URL
- For private repos, use personal access token
- Or upload files manually via Colab file upload

## Quick Start Checklist

- [ ] Open Google Colab
- [ ] Enable GPU (Runtime → Change runtime type → GPU)
- [ ] Clone repository: `!git clone https://github.com/YOUR_USERNAME/Project-VSim.git`
- [ ] Navigate to project: `os.chdir('/content/Project-VSim')`
- [ ] Install dependencies: `!pip install -q ...`
- [ ] Configure settings (email, API key, dataset size)
- [ ] Run data collection (Step 5)
- [ ] Run model training (Step 6)
- [ ] Verify model (Step 7)
- [ ] Download model (Step 8)

## Estimated Times

| Step | Time (Full Dataset) | Time (Test Dataset) |
|------|-------------------|-------------------|
| Setup | 5 minutes | 5 minutes |
| Data Collection | 24-48 hours | 1-3 hours |
| Model Training | 10-20 hours | 1-2 hours |
| **Total** | **34-68 hours** | **2-5 hours** |

For testing, use `TOTAL_TARGET = 10000` to reduce time to 2-5 hours total.

## Next Steps After Training

1. **Download the model** (Step 8)
2. **Use in your application**:
   ```python
   from src.vlab.viability.predictor import ViabilityPredictorWrapper
   from src.vlab.core.config import VLabConfig
   
   config = VLabConfig()
   config.viability_model_path = Path('viability_model_best.pth')
   predictor = ViabilityPredictorWrapper(config)
   ```
3. **Test on new genomes**
4. **Monitor performance**

Good luck with your training! 🚀

