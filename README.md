# ADRD Detection Pipeline

Modular system for Alzheimer's Disease and Related Dementias detection using linguistic and paralinguistic features.

## Architecture

The project consists of 3 main components:
1. **Linguistic Model** - Text-based feature extraction and classification
2. **Paralinguistic Model** - Audio-based feature extraction and classification  
3. **Aggregate Final Model** - Combines outputs from both models for final prediction

## Project Structure

### Aggregate Final Model
- **`final_model/`**
  - `train.py` - Final ensemble model training script

### Linguistic Model

- **`linguistic_synthetic_data/`**
  - `CLAUDE.md` - Documentation for Claude-based data generation
  - `input/` - Input CSV files with language markers and synthetic scenes
  - `output/` - Generated synthetic data for 66 linguistic symptoms
  - `pytest.ini` - Test configuration
  - `README.md` - Module documentation
  - `run_tests.py` - Test runner script
  - `synthetic_data_runner.py` - Main data generation script
  - `tests/` - Unit and integration tests

- **`linguistic_transcription/`**
  - `output/` - Transcriptions with dysfluency annotations
  - `transcribe_audio_phi4.py` - Audio transcription using Phi-4 model

- **`linguistic_models/`**
  - `cache/` - Cached embeddings and metadata for training
  - `cache_debug/` - Debug version of cached embeddings
  - `cache_feature56/` - Cached embeddings for specific feature
  - `models/` - Trained binary classifiers for each linguistic symptom
  - `README.md` - Module documentation
  - `results/` - Training statistics and metrics
  - `train_binary_models.py` - Training script for linguistic classifiers

- **`linguistic_inference/`**
  - `embeddings_cache.pkl` - Cached embeddings for inference
  - `inference.py` - Linguistic feature inference script
  - `output/` - Inference results in CSV and PKL format
  - `README.md` - Module documentation

### Paralinguistic Model

- **`paralinguistic_data/`**: Download and prepare CommonVoice to train the acoustic feature encoder.
  - `data/` - CommonVoice 50+ dataset features and metadata
  - `debug_cache/` - Cached debug datasets
  - `download_and_filter.py` - Dataset download and filtering script
  - `feature_extraction.py` - Acoustic feature extraction script
  - `inspect_data_format.py` - Data format inspection utility
  - `prep_data.py` - Data preprocessing script

- **`paralinguistic_models/`** Train the acoustic feature encoder on CommonVoice.
  - `checkpoints/` - Model checkpoints and normalization stats
  - `checkpoints_debug/` - Debug model checkpoints
  - `data_loader.py` - Data loading utilities
  - `test_data_loading.py` - Data loader testing script
  - `train.py` - Model training script
  - `wandb/` - Weights & Biases experiment logs

- **`paralinguistic_classifier/`** Fine tune the acoustic feature encoder using ADRD data.
  - `best_classifier.pth` - Best trained classifier model
  - `feature_cache/` - Cached acoustic features and metadata
  - `pretrain_inference.py` - Inference with pretrained model
  - `test.py` - Model testing script
  - `train_classifier.py` - Classifier training script


### Shared Data
- **`data/`**
  - `acoustic_additional_metadata.csv` - Additional metadata for acoustic features
  - `acoustic_test_labels.csv` - Test set labels for acoustic classification
  - `metadata_dev.csv` - Development set metadata
  - `metadata.csv` - Main dataset metadata
  - `submission_format.csv` - Template for competition submission
  - `test_audios/` - Test audio files (MP3 format)
  - `test_features.csv` - Extracted features for test set
  - `train_audios/` - Training audio files (MP3 format)
  - `train_audios_sample/` - Sample subset of training audio files
  - `train_features.csv` - Extracted features for training set
  - `train_labels.csv` - Training set labels

### Root Files
- `LICENSE` - Project license
- `master_todo.md` - Project task tracking
- `pyproject.toml` - Python project configuration and dependencies
- `requirements.txt` - Python package requirements
- `scratch.ipynb` - Experimental notebook
- `uv.lock` - Dependency lock file