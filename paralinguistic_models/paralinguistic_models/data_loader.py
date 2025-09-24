import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing as mp

def load_acoustic_features(features_path):
	"""Load features dataframe and return basic info"""
	print("Loading features...")
	df = pd.read_parquet(features_path)
	
	print(f"Loaded {len(df)} segments from {df['audio_path'].nunique()} audio files")
	print(f"Features shape: {df.shape}")
	print(f"Columns: {list(df.columns)}")
	
	# Check for required columns
	required_cols = ['audio_path', 'row_index']
	missing_cols = [col for col in required_cols if col not in df.columns]
	if missing_cols:
		raise ValueError(f"Missing required columns: {missing_cols}")
	
	# Check row_index format
	print(f"Row index range: {df['row_index'].min()} to {df['row_index'].max()}")
	
	return df

def get_feature_columns(df):
	"""Extract the 88 eGeMAPS feature columns, excluding metadata"""
	# Get all columns except metadata
	metadata_cols = ['client_id', 'sentence', 'audio_path', 'segment_start_sec', 'segment_end_sec', 'row_index']
	feature_cols = [col for col in df.columns if col not in metadata_cols]
	print(f"Found {len(feature_cols)} feature columns")
	return feature_cols

def process_file_group(args):
	"""Process a single file's data - designed for multiprocessing"""
	file_path, group_data, feature_cols, max_length = args
	
	# Sort by row_index for temporal ordering
	group_data = group_data.sort_values('row_index')
	
	# Initialize file tensor
	file_tensor = np.full((len(feature_cols), max_length), np.nan, dtype=np.float32)
	
	# Fill valid timesteps
	valid_timesteps = min(len(group_data), max_length)
	for i in range(valid_timesteps):
		file_tensor[:, i] = group_data.iloc[i][feature_cols].values
	
	return file_path, file_tensor

def reshape_to_3d_tensor(df_indexed, max_length=45):
	"""
	Fast multiprocessing version of reshape_to_3d_tensor
	
	Args:
		df_indexed: DataFrame with MultiIndex [audio_path, row_index] and feature columns
		max_length: Maximum number of timesteps to keep
	
	Returns:
		tensor_3d: numpy array [n_files, n_features, n_timesteps]
		file_list: list of audio file paths in same order as tensor
	"""
	print("Reshaping to 3D tensor...")
	
	# Reset index to work with groupby
	df_reset = df_indexed.reset_index()
	feature_cols = list(df_indexed.columns)
	
	print(f"Processing {df_reset['audio_path'].nunique()} files with {len(feature_cols)} features")
	print(f"Max timesteps per file: {max_length}")
	print(f"Using 30 CPU cores for parallel processing")
	
	# Group by audio_path
	grouped = df_reset.groupby('audio_path')
	files = list(grouped.groups.keys())
	
	# Estimate memory
	total_elements = len(files) * len(feature_cols) * max_length
	memory_gb = (total_elements * 4) / (1024**3)
	print(f"Estimated memory usage: {memory_gb:.1f} GB")
	
	# Prepare arguments for multiprocessing
	process_args = []
	for file_path, group in grouped:
		process_args.append((file_path, group, feature_cols, max_length))
	
	# Process in parallel with 30 cores
	print("Processing files in parallel...")
	with mp.Pool(30) as pool:
		results = list(tqdm(
			pool.imap(process_file_group, process_args),
			total=len(process_args),
			desc="Processing files"
		))
	
	# Sort results by file path to ensure consistent ordering
	results.sort(key=lambda x: x[0])
	
	# Combine into final tensor
	print("Combining results...")
	tensor_3d = np.stack([result[1] for result in results], axis=0)
	file_list = [result[0] for result in results]
	
	# Calculate statistics
	timesteps_per_file = [np.sum(~np.isnan(tensor_3d[i, 0, :])) for i in range(len(tensor_3d))]
	truncated_files = sum(1 for args in process_args if len(args[1]) > max_length)
	
	print(f"Final tensor shape: {tensor_3d.shape}")
	print(f"NaN ratio: {np.isnan(tensor_3d).mean():.3f}")
	print(f"Timesteps per file: min={min(timesteps_per_file)}, max={max(timesteps_per_file)}, avg={np.mean(timesteps_per_file):.1f}")
	
	if truncated_files > 0:
		print(f"Truncated {truncated_files}/{len(files)} files ({100*truncated_files/len(files):.1f}%) to {max_length} timesteps")
	
	return tensor_3d, file_list

def create_train_val_split(df, test_size=0.1, random_state=42):
	"""Split data by unique audio files to ensure no leakage"""
	unique_files = df['audio_path'].unique()
	train_files, val_files = train_test_split(unique_files, test_size=test_size, random_state=random_state)
	
	train_df = df[df['audio_path'].isin(train_files)]
	val_df = df[df['audio_path'].isin(val_files)]
	
	print(f"Train: {len(train_df)} segments from {len(train_files)} files")
	print(f"Val: {len(val_df)} segments from {len(val_files)} files")
	
	return train_df, val_df

def process_acoustic_data(features_path, max_length=45, test_size=0.1):
	"""
	Complete pipeline to load and process acoustic data
	
	Returns:
		train_data: [n_train_files, n_features, max_length]
		val_data: [n_val_files, n_features, max_length] 
		train_files: list of training file paths
		val_files: list of validation file paths
	"""
	# Step 1: Load data
	df = load_acoustic_features(features_path)
	
	# Step 2: Get feature columns
	feature_cols = get_feature_columns(df)
	
	# Step 3: Split by files (no leakage)
	train_df, val_df = create_train_val_split(df, test_size=test_size)
	
	# Step 4: Prepare for reshaping - keep only features and indexing columns
	train_features = train_df[['audio_path', 'row_index'] + feature_cols]
	val_features = val_df[['audio_path', 'row_index'] + feature_cols]
	
	# Step 5: Set index for reshaping
	train_indexed = train_features.set_index(['audio_path', 'row_index'])[feature_cols]
	val_indexed = val_features.set_index(['audio_path', 'row_index'])[feature_cols]
	
	# Step 6: Reshape to tensors
	train_data, train_files = reshape_to_3d_tensor(train_indexed, max_length)
	val_data, val_files = reshape_to_3d_tensor(val_indexed, max_length)
	
	return train_data, val_data, train_files, val_files

def create_masks(tensor_data):
	"""Create boolean masks indicating valid (non-NaN) timesteps for each sample"""
	# Mask is True where we have valid data across ALL features for that timestep
	masks = ~np.isnan(tensor_data).all(axis=1)  # [n_files, n_timesteps]
	print(f"Created masks with shape: {masks.shape}")
	print(f"Average valid timesteps per sample: {masks.sum(axis=1).mean():.1f}")
	return masks

# Test the pipeline
if __name__ == "__main__":
	# Test with your data - auto-detect normal file first, fallback to debug
	import os
	
	# Look for normal features file first, then debug
	normal_features_path = "data/commonvoice_50plus_features.parquet"
	debug_features_path = "data/commonvoice_50plus_features_debug.parquet"
	
	if os.path.exists(normal_features_path):
		features_path = normal_features_path
		print(f"Using normal features file: {features_path}")
	elif os.path.exists(debug_features_path):
		features_path = debug_features_path
		print(f"Using debug features file: {features_path}")
	else:
		print("No features file found. Expected one of:")
		print(f"  - {normal_features_path}")
		print(f"  - {debug_features_path}")
		exit(1)
	
	print("=== TESTING DATA LOADING PIPELINE ===")
	
	# Load and process data
	train_data, val_data, train_files, val_files = process_acoustic_data(
		features_path, 
		max_length=45, 
		test_size=0.1
	)
	
	# Create masks
	train_masks = create_masks(train_data)
	val_masks = create_masks(val_data)
	
	print("\n=== FINAL RESULTS ===")
	print(f"Train data shape: {train_data.shape}")  # [n_files, 88_features, 45_timesteps]
	print(f"Val data shape: {val_data.shape}")
	print(f"Train masks shape: {train_masks.shape}")  # [n_files, 45_timesteps]
	print(f"Val masks shape: {val_masks.shape}")
	
	# Sanity checks
	print(f"\nData ranges:")
	print(f"Train data range: [{np.nanmin(train_data):.2f}, {np.nanmax(train_data):.2f}]")
	print(f"Val data range: [{np.nanmin(val_data):.2f}, {np.nanmax(val_data):.2f}]")
	
	print(f"\nSample file paths:")
	print(f"First train file: {train_files[0] if train_files else 'None'}")
	print(f"First val file: {val_files[0] if val_files else 'None'}")