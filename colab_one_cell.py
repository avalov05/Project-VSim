"""
VSim Training - Single Cell for Google Colab
Just paste this entire code into one Colab cell and run it!
"""

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================
GITHUB_REPO = "avalov05/Project-VSim"  # ← Change this to your repo
EMAIL = "anton.valov05@gmail.com"  # ← Your email for NCBI
API_KEY = "d7e5c7978697a8c4284af0fc71ce1a2b9808"  # ← Your NCBI API key (or "" if none)
TOTAL_TARGET = 500000  # ← Start with 10000 for testing, 500000 for full training

# Model settings (optimized for maximum quality)
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

# ============================================================================
# SETUP - NO NEED TO EDIT BELOW
# ============================================================================
import os
import sys
import subprocess
from pathlib import Path

print("="*70)
print("VSIM TRAINING - SINGLE CELL SETUP")
print("="*70)

# Step 1: Install dependencies
print("\n[1/6] Installing dependencies...")
os.system("pip install -q biopython numpy pandas scipy scikit-learn torch torchvision torchaudio pyyaml requests aiohttp tqdm backoff matplotlib nest-asyncio")

# Step 2: Clone repository (if not already cloned)
repo_name = GITHUB_REPO.split('/')[-1]
project_dir = Path(f'/content/{repo_name}')
if not project_dir.exists():
    print(f"\n[2/6] Cloning repository: {GITHUB_REPO}...")
    repo_url = f"https://github.com/{GITHUB_REPO}.git"
    try:
        subprocess.run(['git', 'clone', repo_url, str(project_dir)], check=True, capture_output=True)
        print(f"✓ Repository cloned to {project_dir}")
    except subprocess.CalledProcessError as e:
        print(f"⚠ Git clone failed: {e}")
        print("  Make sure the repository URL is correct and accessible")
        raise
else:
    print(f"\n[2/6] Repository already exists at {project_dir}")

# Step 3: Navigate to project directory
print(f"\n[3/6] Setting up environment...")
os.chdir(project_dir)
sys.path.insert(0, str(project_dir))
print(f"✓ Working directory: {os.getcwd()}")

# Step 4: Check GPU
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ GPU not available, using CPU (training will be slow)")

# Step 5: Import project modules
print(f"\n[4/6] Importing modules...")
import asyncio
import nest_asyncio
import logging
from src.vlab.data.collector import DataCollector
from src.vlab.training.viability_trainer import ViabilityTrainer, collect_training_data
from src.vlab.core.config import VLabConfig

# Enable nested event loops (required for Colab/Jupyter)
nest_asyncio.apply()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

print("✓ Modules imported successfully")

# ============================================================================
# DATA COLLECTION
# ============================================================================
print("\n" + "="*70)
print("[5/6] STARTING DATA COLLECTION")
print("="*70)
print(f"Target: {TOTAL_TARGET:,} viral genomes")
print(f"This may take a while... (1-3 hours for 10K, 24-48 hours for 500K)")
print("="*70)

collector = DataCollector(email=EMAIL, api_key=API_KEY if API_KEY else None)

try:
    # Use get_event_loop for Colab compatibility
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(collector.collect_all_data(
        max_viable=None,
        num_synthetic_non_viable=None,
        num_mutated_non_viable=None,
        total_target=TOTAL_TARGET,
        skip_ncbi_if_fails=True  # Use synthetic data if NCBI fails (common in Colab)
    ))
    
    stats = collector.get_data_statistics()
    
    print("\n" + "="*70)
    print("DATA COLLECTION COMPLETE!")
    print("="*70)
    print(f"Training Data: {stats['total_train']:,} genomes")
    print(f"  Viable: {stats['train_viable']:,}")
    print(f"  Non-viable: {stats['train_non_viable']:,}")
    print(f"\nValidation Data: {stats['total_val']:,} genomes")
    print(f"  Viable: {stats['val_viable']:,}")
    print(f"  Non-viable: {stats['val_non_viable']:,}")
    print(f"\nGrand Total: {stats['total']:,} genomes")
    print("="*70)
    
except KeyboardInterrupt:
    logger.error("\nData collection interrupted by user")
    raise
except Exception as e:
    logger.error(f"Data collection failed: {e}", exc_info=True)
    raise

# ============================================================================
# MODEL TRAINING
# ============================================================================
print("\n" + "="*70)
print("[6/6] STARTING MODEL TRAINING")
print("="*70)
print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
print(f"Model: dim={INPUT_DIM}, layers={NUM_LAYERS}, heads={NUM_HEADS}")
print(f"Advanced: Focal Loss={USE_FOCAL_LOSS}, Mixed Precision={USE_MIXED_PRECISION}")
print("="*70)

# Load configuration
config = VLabConfig()
config.use_gpu = torch.cuda.is_available()
config.gpu_id = 0
config.models_dir = Path('models')
config.models_dir.mkdir(parents=True, exist_ok=True)

# Load training data
data_dir = Path('data/training')
print(f"\nLoading data from: {data_dir}")

try:
    train_annotations, train_labels, val_annotations, val_labels = collect_training_data(data_dir)
    
    if not train_annotations:
        raise ValueError("No training data found! Data collection may have failed.")
    
    print(f"✓ Loaded {len(train_annotations)} training samples")
    if val_annotations:
        print(f"✓ Loaded {len(val_annotations)} validation samples")
    
    # Create trainer
    print(f"\nCreating enhanced model...")
    trainer = ViabilityTrainer(
        config,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        use_focal_loss=USE_FOCAL_LOSS,
        use_mixed_precision=USE_MIXED_PRECISION
    )
    
    # Train model
    print(f"\nStarting training...")
    print("This will take a while... (1-2 hours for test, 10-20 hours for full)")
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
    print(f"✓ Final model saved: {final_model_path}")
    
    best_model_path = config.models_dir / "viability_model_best.pth"
    if best_model_path.exists():
        print(f"✓ Best model saved: {best_model_path}")
    
    print("="*70)
    
except Exception as e:
    logger.error(f"Training failed: {e}", exc_info=True)
    raise

# ============================================================================
# DOWNLOAD MODEL (Optional)
# ============================================================================
print("\n" + "="*70)
print("DOWNLOAD MODEL")
print("="*70)
print("To download your model, run this in a new cell:")
print("""
from google.colab import files
import zipfile
from pathlib import Path

models_dir = Path('models')
output_zip = '/tmp/vsim_model.zip'

with zipfile.ZipFile(output_zip, 'w') as zipf:
    for model_file in models_dir.glob('*.pth'):
        zipf.write(model_file, model_file.name)
        print(f"  Added: {model_file.name}")

files.download(output_zip)
print("✓ Model downloaded!")
""")
print("="*70)
print("\n✓ ALL DONE! Your model is ready at: models/viability_model_final.pth")

