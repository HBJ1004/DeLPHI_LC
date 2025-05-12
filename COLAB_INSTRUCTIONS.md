# Running the Asteroid Light Curve Analysis Pipeline in Google Colab

This guide provides comprehensive instructions for running the asteroid light curve analysis pipeline in Google Colab.

## Setup Instructions

### 1. Upload Project to Google Drive

First, upload the entire project to your Google Drive in one of these recommended locations:
- `/content/drive/MyDrive/Colab Notebooks/lc_pipeline/`
- `/content/drive/MyDrive/asteroid_lightcurve_pipeline/`

**Required Files and Folders:**
- `lc_pipeline/` directory (the main package)
- `colab_run.py` (the Colab-specific run script)
- `requirements.txt` (for dependencies)
- `config.yaml` (configuration file)
- `DAMIT_csv/` (data directory, if you'll use the DAMIT dataset)

### 2. Create a New Colab Notebook

Open [Google Colab](https://colab.research.google.com/) and create a new notebook. You have two options:
- **Option 1:** Upload and use the provided `colab_run.ipynb` notebook
- **Option 2:** Create a new notebook and paste in the commands from this guide

### 3. Run the Pipeline in Colab

#### Option 1: Using colab_run.py (Recommended)

1. **Mount Google Drive and install dependencies:**
   ```python
   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Install dependencies
   !pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
   !pip install numpy pandas matplotlib tqdm scikit-learn astropy requests pyyaml psutil
   ```

2. **Navigate to your project directory:**
   ```python
   # Adjust this path to match your Google Drive structure
   %cd /content/drive/MyDrive/Colab Notebooks/lc_pipeline
   # Or to wherever you uploaded your project
   # %cd /content/drive/MyDrive/asteroid_lightcurve_pipeline
   
   # Check the files are present
   !ls -la
   ```

3. **Run the colab_run.py script:**
   ```python
   # For full pipeline execution
   !python colab_run.py
   
   # For a smaller test run (faster)
   !python colab_run.py --test
   ```

#### Option 2: Manual Execution

1. **Mount Google Drive and navigate to your project:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/Colab Notebooks/lc_pipeline  # Adjust as needed
   ```

2. **Add the project directory to Python path:**
   ```python
   import sys
   import os
   
   # Adjust this path to match your Google Drive structure
   PROJECT_DIR = "/content/drive/MyDrive/Colab Notebooks/lc_pipeline"
   if PROJECT_DIR not in sys.path:
       sys.path.insert(0, PROJECT_DIR)
   ```

3. **Run the pipeline using main.py:**
   ```python
   !python main.py config.yaml --test
   ```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. CUDA Out of Memory Errors
- **Symptoms:** RuntimeError: CUDA out of memory
- **Solution:** Reduce batch sizes in `config.yaml` or in `colab_run.py`:
  ```python
  # Modify these values in colab_run.py
  PERIOD_BATCH_SIZE = 32  # Reduced from 64
  AXIS_BATCH_SIZE = 32    # Reduced from 64
  ```

#### 2. Module Not Found Errors
- **Symptoms:** ModuleNotFoundError: No module named 'lc_pipeline'
- **Solution:** Check your project structure and make sure the path is correctly added to sys.path:
  ```python
  import sys
  print(sys.path)  # Check if your project directory is included
  
  # If not found, try finding it automatically
  import glob
  potential_paths = glob.glob("/content/drive/MyDrive/**/lc_pipeline", recursive=True)
  print("Found possible paths:", potential_paths)
  
  # Add the parent directory of the first found lc_pipeline directory
  if potential_paths:
      parent_dir = os.path.dirname(potential_paths[0])
      sys.path.insert(0, parent_dir)
      print(f"Added {parent_dir} to sys.path")
  ```

#### 3. Tensor Device Mismatch
- **Symptoms:** Expected all tensors to be on the same device, but found at least two devices
- **Solution:** Use our compatibility module:
  ```python
  from lc_pipeline.models.model_compatibility import ensure_model_on_device
  
  # Move model to correct device
  model = ensure_model_on_device(model, device)
  ```

#### 4. Cache Directory Issues
- **Symptoms:** Failed to save AsteroidDataset to cache: No such file or directory
- **Solution:** Create cache directories manually:
  ```python
  os.makedirs("data_cache/asteroid_dataset", exist_ok=True)
  ```

## Data Management Tips

### Saving and Storing Results

Remember to save your results and models back to Google Drive periodically:

```python
# Save results back to Drive
!mkdir -p /content/drive/MyDrive/asteroid_results
!cp -r results/ /content/drive/MyDrive/asteroid_results/
!cp -r models/ /content/drive/MyDrive/asteroid_results/
!cp -r figures/ /content/drive/MyDrive/asteroid_results/
```

### Managing Large Datasets

If working with large datasets:
1. Consider using subsets for testing (use the `--max_files` parameter)
2. Use the cache system to avoid reloading data between runs
3. Clear memory periodically during long runs:
   ```python
   import gc
   import torch
   
   # Clear memory
   gc.collect()
   torch.cuda.empty_cache()
   ```

## Advanced Configuration

### Modifying Pipeline Parameters

You can adjust model parameters directly in Colab:

```python
# Import config module
from lc_pipeline.config import load_config, PeriodModelConfig

# Load config
config = load_config("config.yaml")

# Modify parameters
config.period_model.batch_size = 32
config.period_model.epochs = 5
config.data.max_files = 50

# Run with modified config
from lc_pipeline.main import run_pipeline
run_pipeline(config)
```

## Getting Help

If you encounter issues not covered in this guide:
1. Check the log files in the `logs/` directory
2. Examine the Colab console output for error messages
3. Consult the project documentation in `docs/` or `README.md` 