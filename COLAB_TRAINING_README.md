# VSim Model Training on Google Colab

This guide explains how to use the `VSim_Train_Model_Colab.ipynb` notebook to collect training data and train your viability prediction model on Google Colab.

## Quick Start

1. **Open the notebook** in Google Colab
2. **Enable GPU** (recommended for faster training):
   - Runtime → Change runtime type → GPU (T4 or better)
3. **Run all cells** in order (Runtime → Run all)
4. **Wait for completion** - this will take several hours depending on your dataset size
5. **Download your trained model** from the last cell

## What the Notebook Does

The notebook automatically:
1. Installs all required dependencies
2. Sets up the VSim environment
3. Collects viral genome data from NCBI
4. Trains the viability prediction model
5. Saves the trained model
6. Allows you to download the model

## Configuration

In **Step 4**, you can configure:

### Data Collection Settings
- `EMAIL`: Your email for NCBI (required)
- `API_KEY`: NCBI API key (optional, speeds up downloads)
- `TOTAL_TARGET`: Number of genomes to collect
  - `1000` - Quick test (~10-30 minutes)
  - `10000` - Medium dataset (~1-3 hours) ⭐ Recommended
  - `50000` - Large dataset (~5-10 hours)
  - `500000` - Full dataset (~24-48 hours)

### Training Settings
- `EPOCHS`: Number of training epochs (default: 20)
- `BATCH_SIZE`: Batch size (default: 32, increase if you have more GPU memory)
- `LEARNING_RATE`: Learning rate (default: 1e-4)

## Uploading Project Files

You have two options:

### Option 1: Upload ZIP File (Recommended)
1. Zip your `Project-VSim` folder
2. In Step 2, click "Choose Files" and select the ZIP file
3. The notebook will automatically extract it

### Option 2: Clone from GitHub
1. In Step 2, set `USE_GITHUB = True`
2. Set `GITHUB_REPO = "your-username/Project-VSim"` (replace with your repo)
3. The notebook will clone from GitHub

### Option 3: Use Existing Files
If you've already uploaded files in a previous run, the notebook will detect them and skip the upload step.

## Output

After training completes, you'll have:
- `models/viability_model_final.pth` - Final trained model
- `models/viability_model_best.pth` - Best model (if validation was used)
- Training data in `data/training/`

## Downloading the Model

Use **Step 8** to download your trained model as a ZIP file. The model can then be used in your VSim application.

## Using the Trained Model

```python
from src.vlab.viability.predictor import ViabilityPredictor
import torch

# Load the model
model = ViabilityPredictor()
checkpoint = torch.load('viability_model_final.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for predictions
# ... (your prediction code here)
```

## Troubleshooting

### GPU Not Available
- Make sure GPU is enabled: Runtime → Change runtime type → GPU
- The notebook will work on CPU, but training will be much slower

### Out of Memory
- Reduce `BATCH_SIZE` in Step 4
- Reduce `TOTAL_TARGET` to collect less data
- Use a smaller dataset for testing

### Data Collection Fails
- Check your internet connection
- Verify your email and API key are correct
- NCBI may rate-limit requests - the notebook handles this automatically

### Training Fails
- Make sure data collection completed successfully
- Check that you have enough training samples (at least 100 recommended)
- Verify GPU is available and has enough memory

## Tips

1. **Start small**: Use `TOTAL_TARGET = 1000` for your first run to test everything works
2. **Use GPU**: Training is 10-100x faster on GPU
3. **Save progress**: Colab sessions timeout after inactivity - the notebook saves checkpoints automatically
4. **Monitor progress**: Watch the output to see data collection and training progress
5. **Download immediately**: Download your model as soon as training completes

## Estimated Times

| Dataset Size | Data Collection | Training (GPU) | Training (CPU) |
|-------------|----------------|----------------|----------------|
| 1,000        | ~10-30 min     | ~5-10 min      | ~30-60 min     |
| 10,000       | ~1-3 hours     | ~20-40 min     | ~2-4 hours     |
| 50,000       | ~5-10 hours    | ~1-2 hours     | ~10-20 hours   |
| 500,000      | ~24-48 hours   | ~5-10 hours    | ~50-100 hours  |

*Times are estimates and may vary based on network speed, GPU type, and other factors.*

## Support

If you encounter issues:
1. Check the error messages in the notebook output
2. Verify all cells ran successfully in order
3. Make sure you have enough disk space (Colab provides ~80GB)
4. Check that all dependencies installed correctly

## Next Steps

After training:
1. Download your model
2. Integrate it into your VSim application
3. Test it on new viral genomes
4. Fine-tune hyperparameters if needed

Happy training! 🚀

