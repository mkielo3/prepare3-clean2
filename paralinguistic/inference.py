"""
Inference script for acoustic contrastive model

Usage:
    python inference.py audio.mp3                          # Use best model checkpoint
    python inference.py audio.mp3 --checkpoint model.pth   # Use specific checkpoint
    python inference.py audio.mp3 --normalize-stats stats.npz  # Use saved normalization stats

(unsloth_env) gildroid@noblestone:~/workspace2025/prepare3-clean2/paralinguistic_inference$ python inference.py /home/gildroid/workspace2025/prepare3-clean2/data/test_audios/atpg.mp3 --checkpoint /home/gildroid/workspace2025/prepare3-clean2/paralinguistic_models/checkpoints/best_model.pth	

@TODO: file currently only uses first 45 timeshots
@TODO: file needs normalization values

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import librosa
import opensmile
import argparse
import os
from pathlib import Path
import warnings

# Suppress audio processing warnings
warnings.filterwarnings("ignore", message="Segment too short, filling with NaN.")

# Import shared model architecture
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from shared.model import UpgradedAcousticEncoder

def extract_audio_features(audio_path, segment_duration=0.2):
	"""Extract eGeMAPS features from audio file, matching the training pipeline"""
	print(f"\n=== DEBUG: Extracting features ===")
	print(f"Audio path: {audio_path}")
	print(f"File exists: {os.path.exists(audio_path)}")
	print(f"File size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'} bytes")
	
	# Load and normalize audio (same as feature_extraction.py)
	print(f"Loading audio with librosa...")
	try:
		y, sr = librosa.load(audio_path, sr=None)
		print(f"Audio loaded successfully")
		print(f"Raw audio shape: {y.shape}")
		print(f"Sample rate: {sr}Hz")
	except Exception as e:
		print(f"Error loading audio: {e}")
		raise
	
	# y = librosa.util.normalize(y)
	total_duration = len(y) / sr
	
	print(f"Audio duration: {total_duration:.2f}s")
	print(f"Normalized audio range: [{y.min():.4f}, {y.max():.4f}]")
	
	# Initialize eGeMAPS feature extractor
	print(f"Initializing openSMILE eGeMAPS extractor...")
	try:
		smile = opensmile.Smile(
			feature_set=opensmile.FeatureSet.eGeMAPSv02,
			feature_level=opensmile.FeatureLevel.Functionals,
		)
		print(f"openSMILE initialized successfully")
	except Exception as e:
		print(f"Error initializing openSMILE: {e}")
		raise
	
	# Create segments (same as feature_extraction.py)
	segment_starts = np.arange(0, total_duration, segment_duration)
	segment_ends = np.minimum(segment_starts + segment_duration, total_duration)
	
	print(f"Created {len(segment_starts)} segments of {segment_duration}s each")
	print(f"First few segments: {list(zip(segment_starts[:3], segment_ends[:3]))}")
	
	all_features = []
	print(f"Processing segments...")
	for i, (row_index, (start, end)) in enumerate(zip(range(len(segment_starts)), zip(segment_starts, segment_ends))):
		if i % 50 == 0:  # Print progress every 50 segments
			print(f"  Processing segment {i+1}/{len(segment_starts)}")
			
		start_sample = int(start * sr)
		end_sample = int(end * sr)
		segment_audio = y[start_sample:end_sample]
		
		if len(segment_audio) == 0:
			print(f"Warning: Empty segment at {start:.3f}-{end:.3f}s")
			continue
		
		# Extract features for this segment
		try:
			features = smile.process_signal(segment_audio, sr)
			print(f"  Segment {i}: features shape {features.shape}") if i < 3 else None
		except Exception as e:
			print(f"Error processing segment {i}: {e}")
			continue
		
		# Store features as row
		feature_row = {'row_index': row_index}
		for col in features.columns:
			feature_row[col] = features.iloc[0][col]
		
		all_features.append(feature_row)
	
	feature_df = pd.DataFrame(all_features)
	print(f"Extracted {len(feature_df)} feature rows with {len(feature_df.columns)-1} features each")
	print(f"Feature columns: {list(feature_df.columns[:5])}... (showing first 5)")
	
	return feature_df

def prepare_model_input(feature_df, max_length=45, normalization_stats=None):
	"""Convert features to model input format, matching data_loader.py processing"""
	print(f"\n=== DEBUG: Preparing model input ===")
	print(f"Input feature_df shape: {feature_df.shape}")
	print(f"Max length: {max_length}")
	print(f"Normalization stats provided: {normalization_stats is not None}")
	
	# Get feature columns (exclude metadata)
	metadata_cols = ['row_index']
	feature_cols = [col for col in feature_df.columns if col not in metadata_cols]
	
	print(f"Total columns: {len(feature_df.columns)}")
	print(f"Metadata columns: {metadata_cols}")
	print(f"Feature columns: {len(feature_cols)}")
	print(f"Expected 88 features, got: {len(feature_cols)}")
	
	if len(feature_cols) != 88:
		print(f"WARNING: Expected 88 eGeMAPS features, got {len(feature_cols)}")
		print(f"Feature columns: {feature_cols}")
	
	# Extract feature matrix: [timesteps, features] -> [features, timesteps]
	feature_matrix = feature_df[feature_cols].values.T  # Shape: [88, n_timesteps]
	print(f"Feature matrix shape after transpose: {feature_matrix.shape}")
	print(f"Feature matrix dtype: {feature_matrix.dtype}")
	print(f"Feature matrix range: [{feature_matrix.min():.4f}, {feature_matrix.max():.4f}]")
	
	# Handle sequence length
	n_timesteps = feature_matrix.shape[1]
	print(f"Number of timesteps: {n_timesteps}")
	
	if n_timesteps > max_length:
		print(f"Truncating sequence from {n_timesteps} to {max_length} timesteps")
		feature_matrix = feature_matrix[:, :max_length]
		n_timesteps = max_length
	
	# Create mask (True for valid timesteps)
	mask = np.ones(max_length, dtype=bool)
	if n_timesteps < max_length:
		mask[n_timesteps:] = False
		print(f"Mask: {n_timesteps} True values, {max_length - n_timesteps} False values")
	else:
		print(f"Mask: all {max_length} values are True")
	
	# Pad sequence if needed
	if n_timesteps < max_length:
		padding = np.zeros((88, max_length - n_timesteps))
		feature_matrix = np.concatenate([feature_matrix, padding], axis=1)
		print(f"Added padding: {padding.shape}")
	
	print(f"Final feature matrix shape: {feature_matrix.shape}")
	
	# Apply normalization if provided
	if normalization_stats is not None:
		print("Applying normalization...")
		means = normalization_stats['means']
		stds = normalization_stats['stds']
		print(f"Normalization means shape: {means.shape}")
		print(f"Normalization stds shape: {stds.shape}")
		print(f"Means range: [{means.min():.4f}, {means.max():.4f}]")
		print(f"Stds range: [{stds.min():.4f}, {stds.max():.4f}]")
		
		means = means[:, np.newaxis]
		stds = stds[:, np.newaxis]
		feature_matrix_before = feature_matrix.copy()
		feature_matrix = (feature_matrix - means) / stds
		feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
		
		print(f"Feature matrix range before normalization: [{feature_matrix_before.min():.4f}, {feature_matrix_before.max():.4f}]")
		print(f"Feature matrix range after normalization: [{feature_matrix.min():.4f}, {feature_matrix.max():.4f}]")
	
	# Add batch dimension: [features, timesteps] -> [1, features, timesteps]
	feature_tensor = torch.FloatTensor(feature_matrix).unsqueeze(0)
	mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
	
	print(f"Final tensor shapes:")
	print(f"  Feature tensor: {feature_tensor.shape}")
	print(f"  Mask tensor: {mask_tensor.shape}")
	print(f"  Valid timesteps: {mask_tensor.sum().item()}")
	print(f"  Feature tensor range: [{feature_tensor.min():.4f}, {feature_tensor.max():.4f}]")
	
	return feature_tensor, mask_tensor

def load_normalization_stats(stats_path=None, training_data_path=None):
	"""Load normalization statistics from file or calculate from training data"""
	print(f"\n=== DEBUG: Loading normalization stats ===")
	print(f"Stats path: {stats_path}")
	print(f"Training data path: {training_data_path}")
	
	if stats_path and os.path.exists(stats_path):
		print(f"Loading normalization stats from: {stats_path}")
		try:
			stats = np.load(stats_path)
			print(f"Loaded stats keys: {list(stats.keys())}")
			means = stats['means']
			stds = stats['stds']
			print(f"Means shape: {means.shape}, range: [{means.min():.4f}, {means.max():.4f}]")
			print(f"Stds shape: {stds.shape}, range: [{stds.min():.4f}, {stds.max():.4f}]")
			return {'means': means, 'stds': stds}
		except Exception as e:
			print(f"Error loading stats file: {e}")
			print(f"Falling back to training data...")
	
	if training_data_path and os.path.exists(training_data_path):
		print(f"Calculating normalization stats from training data: {training_data_path}")
		try:
			df = pd.read_parquet(training_data_path)
			print(f"Training data shape: {df.shape}")
			print(f"Training data columns: {list(df.columns[:10])}... (first 10)")
		except Exception as e:
			print(f"Error loading training data: {e}")
			return None
		
		# Get feature columns
		metadata_cols = ['client_id', 'sentence', 'audio_path', 'segment_start_sec', 'segment_end_sec', 'row_index']
		feature_cols = [col for col in df.columns if col not in metadata_cols]
		
		print(f"Found {len(feature_cols)} feature columns")
		print(f"Expected 88, got {len(feature_cols)}")
		
		if len(feature_cols) != 88:
			print(f"WARNING: Unexpected number of features")
			print(f"First 10 feature columns: {feature_cols[:10]}")
		
		means = []
		stds = []
		print(f"Calculating stats for each feature...")
		for i, col in enumerate(feature_cols):
			if i % 20 == 0:
				print(f"  Processing feature {i+1}/{len(feature_cols)}: {col}")
			values = df[col].dropna()
			mean_val = values.mean()
			std_val = max(values.std(), 1e-6)  # Avoid division by zero
			means.append(mean_val)
			stds.append(std_val)
		
		stats = {'means': np.array(means), 'stds': np.array(stds)}
		
		print(f"Calculated stats:")
		print(f"  Means range: [{stats['means'].min():.4f}, {stats['means'].max():.4f}]")
		print(f"  Stds range: [{stats['stds'].min():.4f}, {stats['stds'].max():.4f}]")
		
		# Save stats for future use
		save_path = "inference_normalization_stats.npz"
		np.savez(save_path, means=stats['means'], stds=stats['stds'])
		print(f"Saved normalization stats to: {save_path}")
		
		return stats
	
	else:
		print("Warning: No normalization stats provided. Using raw features.")
		if training_data_path:
			print(f"Training data path does not exist: {training_data_path}")
		return None

def load_model(checkpoint_path, device='cpu'):
	"""Load trained model from checkpoint"""
	print(f"\n=== DEBUG: Loading model ===")
	print(f"Checkpoint path: {checkpoint_path}")
	print(f"File exists: {os.path.exists(checkpoint_path)}")
	print(f"File size: {os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else 'N/A'} bytes")
	print(f"Device: {device}")
	
	# Initialize model with same config as training
	print(f"Initializing model architecture...")
	model = UpgradedAcousticEncoder(
		n_channels=88,
		embedding_dim=512,
		nhead=16,
		num_encoder_layers=8,
		dim_feedforward=2048
	)
	
	total_params = sum(p.numel() for p in model.parameters())
	print(f"Model initialized with {total_params:,} parameters")
	
	# Load state dict
	print(f"Loading checkpoint...")
	try:
		if checkpoint_path.endswith('.pth'):
			# Direct state dict file
			print(f"Loading as direct state dict (.pth file)")
			state_dict = torch.load(checkpoint_path, map_location=device)
			print(f"State dict keys: {list(state_dict.keys())[:5]}... (first 5)")
			model.load_state_dict(state_dict)
		else:
			# Full checkpoint with optimizer etc.
			print(f"Loading as full checkpoint")
			checkpoint = torch.load(checkpoint_path, map_location=device)
			print(f"Checkpoint keys: {list(checkpoint.keys())}")
			model.load_state_dict(checkpoint['model_state_dict'])
		
		print(f"Checkpoint loaded successfully")
	except Exception as e:
		print(f"Error loading checkpoint: {e}")
		raise
	
	model.to(device)
	model.eval()
	
	print(f"Model loaded successfully on {device}")
	print(f"Model is in eval mode: {not model.training}")
	
	return model

def find_best_checkpoint(checkpoint_dir="checkpoints"):
	"""Find the best model checkpoint"""
	print(f"\n=== DEBUG: Looking for checkpoints ===")
	checkpoint_dir = Path(checkpoint_dir)
	print(f"Checkpoint directory: {checkpoint_dir}")
	print(f"Directory exists: {checkpoint_dir.exists()}")
	
	if checkpoint_dir.exists():
		print(f"Directory contents:")
		for item in checkpoint_dir.iterdir():
			print(f"  {item} ({'file' if item.is_file() else 'dir'})")
	
	# Look for best model first
	best_model_path = checkpoint_dir / "best_model.pth"
	print(f"Looking for: {best_model_path}")
	print(f"Exists: {best_model_path.exists()}")
	if best_model_path.exists():
		print(f"Found best_model.pth!")
		return str(best_model_path)
	
	# Look for best_grokking_encoder.pth
	grokking_path = checkpoint_dir / "best_grokking_encoder.pth"
	print(f"Looking for: {grokking_path}")
	print(f"Exists: {grokking_path.exists()}")
	if grokking_path.exists():
		print(f"Found best_grokking_encoder.pth!")
		return str(grokking_path)
	
	# Look for any .pth files
	pth_files = list(checkpoint_dir.glob("*.pth"))
	print(f"Found .pth files: {pth_files}")
	if pth_files:
		# Return the most recent one
		latest = max(pth_files, key=lambda p: p.stat().st_mtime)
		print(f"Using most recent: {latest}")
		return str(latest)
	
	# Also check parent directories
	print(f"\n=== Checking parent directories ===")
	current_dir = Path.cwd()
	print(f"Current directory: {current_dir}")
	for parent in [current_dir, current_dir.parent]:
		parent_checkpoints = parent / "checkpoints"
		print(f"Checking: {parent_checkpoints}")
		if parent_checkpoints.exists():
			print(f"  Exists! Contents:")
			for item in parent_checkpoints.iterdir():
				print(f"    {item}")
	
	raise FileNotFoundError(f"No model checkpoints found in {checkpoint_dir}")

def run_inference(audio_path, checkpoint_path=None, normalization_stats_path=None, 
				  training_data_path=None, device=None, checkpoint_dir="checkpoints"):
	"""Complete inference pipeline"""
	print(f"\n{'='*60}")
	print(f"STARTING INFERENCE PIPELINE")
	print(f"{'='*60}")
	print(f"Audio path: {audio_path}")
	print(f"Checkpoint path: {checkpoint_path}")
	print(f"Normalization stats path: {normalization_stats_path}")
	print(f"Training data path: {training_data_path}")
	print(f"Checkpoint directory: {checkpoint_dir}")
	
	# Setup device
	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")
	print(f"CUDA available: {torch.cuda.is_available()}")
	if torch.cuda.is_available():
		print(f"CUDA device: {torch.cuda.get_device_name()}")
	
	# Find checkpoint if not specified
	if checkpoint_path is None:
		print(f"No checkpoint specified, searching for best checkpoint...")
		checkpoint_path = find_best_checkpoint(checkpoint_dir)
	
	print(f"Using checkpoint: {checkpoint_path}")
	
	# Extract features
	feature_df = extract_audio_features(audio_path)
	
	# Load normalization stats
	norm_stats = load_normalization_stats(normalization_stats_path, training_data_path)
	
	# Prepare model input
	feature_tensor, mask_tensor = prepare_model_input(feature_df, normalization_stats=norm_stats)
	feature_tensor = feature_tensor.to(device)
	mask_tensor = mask_tensor.to(device)
	
	print(f"Tensors moved to device: {device}")
	
	# Load model
	model = load_model(checkpoint_path, device)
	
	# Run inference
	print(f"\n=== DEBUG: Running model inference ===")
	print(f"Input tensor shape: {feature_tensor.shape}")
	print(f"Input mask shape: {mask_tensor.shape}")
	print(f"Model device: {next(model.parameters()).device}")
	print(f"Input device: {feature_tensor.device}")
	
	with torch.no_grad():
		try:
			embedding = model(feature_tensor, mask_tensor)
			print(f"Model inference successful!")
			print(f"Output embedding shape: {embedding.shape}")
		except Exception as e:
			print(f"Error during model inference: {e}")
			raise
	
	# Convert to numpy
	embedding_np = embedding.cpu().numpy()
	
	print(f"\n=== INFERENCE COMPLETED ===")
	print(f"Generated embedding shape: {embedding_np.shape}")
	print(f"Embedding L2 norm: {np.linalg.norm(embedding_np):.4f}")
	print(f"Embedding range: [{embedding_np.min():.4f}, {embedding_np.max():.4f}]")
	print(f"Embedding mean: {embedding_np.mean():.4f}")
	print(f"Embedding std: {embedding_np.std():.4f}")
	
	return embedding_np, feature_df

def main():
	parser = argparse.ArgumentParser(description='Run inference on audio file')
	parser.add_argument('audio_path', help='Path to input audio file (.mp3, .wav, etc.)')
	parser.add_argument('--checkpoint', help='Path to model checkpoint (auto-detected if not provided)')
	parser.add_argument('--checkpoint-dir', default='checkpoints', help='Directory to search for checkpoints')
	parser.add_argument('--normalize-stats', help='Path to normalization stats (.npz file)')
	parser.add_argument('--training-data', default='data/commonvoice_50plus_features.parquet',
						help='Path to training features for calculating normalization stats')
	parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to use for inference')
	parser.add_argument('--output', help='Save embedding to file (.npy)')
	
	args = parser.parse_args()
	
	print(f"\n{'='*60}")
	print(f"INFERENCE SCRIPT STARTED")
	print(f"{'='*60}")
	print(f"Arguments:")
	for arg, value in vars(args).items():
		print(f"  {arg}: {value}")
	
	# Check input file exists
	print(f"\n=== Checking input file ===")
	print(f"Audio path: {args.audio_path}")
	if not os.path.exists(args.audio_path):
		print(f"Error: Audio file not found: {args.audio_path}")
		print(f"Current working directory: {os.getcwd()}")
		print(f"Directory contents:")
		try:
			parent_dir = os.path.dirname(args.audio_path) or "."
			for item in os.listdir(parent_dir):
				print(f"  {item}")
		except:
			print("  Could not list directory")
		return
	else:
		print(f"✓ Audio file exists")
		file_size = os.path.getsize(args.audio_path)
		print(f"  File size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
	
	# Check checkpoint directory
	print(f"\n=== Checking checkpoint directory ===")
	print(f"Checkpoint directory: {args.checkpoint_dir}")
	print(f"Directory exists: {os.path.exists(args.checkpoint_dir)}")
	if os.path.exists(args.checkpoint_dir):
		contents = os.listdir(args.checkpoint_dir)
		print(f"Directory contents: {contents}")
	
	try:
		# Run inference
		embedding, features = run_inference(
			audio_path=args.audio_path,
			checkpoint_path=args.checkpoint,
			normalization_stats_path=args.normalize_stats,
			training_data_path=args.training_data,
			device=args.device,
			checkpoint_dir=args.checkpoint_dir
		)
		
		print(f"\n{'='*60}")
		print(f"INFERENCE COMPLETED SUCCESSFULLY!")
		print(f"{'='*60}")
		print(f"Input: {args.audio_path}")
		print(f"Embedding shape: {embedding.shape}")
		print(f"Feature segments: {len(features)}")
		
		# Save embedding if requested
		if args.output:
			np.save(args.output, embedding)
			print(f"Embedding saved to: {args.output}")
		
		# Print first few values as sample
		print(f"Embedding sample (first 5 values): {embedding[0, :5]}")
		
	except Exception as e:
		print(f"\n{'='*60}")
		print(f"ERROR DURING INFERENCE")
		print(f"{'='*60}")
		print(f"Error: {e}")
		import traceback
		traceback.print_exc()
		
		print(f"\n=== DEBUGGING INFORMATION ===")
		print(f"Python version: {sys.version}")
		print(f"Current working directory: {os.getcwd()}")
		print(f"Available files in current directory:")
		for item in os.listdir("."):
			print(f"  {item}")

if __name__ == "__main__":
	import sys
	main()