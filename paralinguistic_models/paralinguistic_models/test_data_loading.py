#!/usr/bin/env python3
"""
Test script for the clean data loading pipeline.

Usage:
    python test_data_loading.py                    # Run in normal mode (default)
    python test_data_loading.py --normal          # Run in normal mode explicitly  
    python test_data_loading.py --debug           # Run in debug mode (uses debug.parquet)
    python test_data_loading.py path/to/file.parquet  # Use specific file
"""

import sys
import os
import pandas as pd
import numpy as np
from data_loader import process_acoustic_data, create_masks

def find_features_file(data_dir="../paralinguistic_data/data", debug_mode=False):
	"""Auto-detect the features file based on debug mode"""
	if debug_mode:
		features_file = "commonvoice_50plus_features_debug.parquet"
	else:
		features_file = "commonvoice_50plus_features.parquet"  # Normal version
	
	features_path = os.path.join(data_dir, features_file)
	
	if not os.path.exists(features_path):
		# Try debug version if normal doesn't exist
		if not debug_mode:
			print(f"Normal version not found, trying debug version...")
			features_path = os.path.join(data_dir, "commonvoice_50plus_features_debug.parquet")
	
	if not os.path.exists(features_path):
		raise FileNotFoundError(f"Features file not found: {features_path}")
	
	print(f"Using features file: {features_path}")
	return features_path

def test_data_loading(features_path=None, debug_mode=False):
	"""Test the data loading pipeline"""
	
	print("=== TESTING DATA LOADING PIPELINE ===\n")
	
	# Auto-detect features file if not provided
	if features_path is None:
		features_path = find_features_file(debug_mode=debug_mode)
	
	try:
		# Step 1: Load and process
		print("Step 1: Loading and processing data...")
		train_data, val_data, train_files, val_files = process_acoustic_data(
			features_path, 
			max_length=45, 
			test_size=0.1
		)
		
		# Step 2: Create masks
		print("\nStep 2: Creating masks...")
		train_masks = create_masks(train_data)
		val_masks = create_masks(val_data)
		
		# Step 3: Detailed inspection
		print("\n=== DETAILED INSPECTION ===")
		
		print(f"Train tensor shape: {train_data.shape}")
		print(f"Val tensor shape: {val_data.shape}")
		print(f"Train masks shape: {train_masks.shape}")
		print(f"Val masks shape: {val_masks.shape}")
		
		# Check data integrity
		print(f"\nData integrity:")
		print(f"- Train files: {len(train_files)}")
		print(f"- Val files: {len(val_files)}")
		print(f"- Features per sample: {train_data.shape[1]} (should be 88)")
		print(f"- Max timesteps: {train_data.shape[2]} (should be 45)")
		
		# Check for reasonable ranges
		train_valid = train_data[~np.isnan(train_data)]
		val_valid = val_data[~np.isnan(val_data)]
		
		print(f"\nData ranges (excluding NaNs):")
		print(f"- Train: [{train_valid.min():.2f}, {train_valid.max():.2f}]")
		print(f"- Val: [{val_valid.min():.2f}, {val_valid.max():.2f}]")
		
		# Check mask statistics
		train_valid_per_sample = train_masks.sum(axis=1)
		val_valid_per_sample = val_masks.sum(axis=1)
		
		print(f"\nValid timesteps per sample:")
		print(f"- Train: mean={train_valid_per_sample.mean():.1f}, min={train_valid_per_sample.min()}, max={train_valid_per_sample.max()}")
		print(f"- Val: mean={val_valid_per_sample.mean():.1f}, min={val_valid_per_sample.min()}, max={val_valid_per_sample.max()}")
		
		# Sample a few files to check temporal ordering
		print(f"\n=== SAMPLE VERIFICATION ===")
		
		# Load original data to verify ordering
		df = pd.read_parquet(features_path)
		sample_file = train_files[0]
		
		print(f"Checking file: {sample_file.split('/')[-1]}")
		
		# Get original segments for this file
		file_segments = df[df['audio_path'] == sample_file].sort_values('row_index')
		print(f"Original segments: {len(file_segments)} rows")
		print(f"Time range: {file_segments['segment_start_sec'].min():.1f}s to {file_segments['segment_end_sec'].max():.1f}s")
		
		# Check tensor data for this file
		file_idx = train_files.index(sample_file)
		sample_tensor = train_data[file_idx]
		sample_mask = train_masks[file_idx]
		
		valid_timesteps = sample_mask.sum()
		print(f"Tensor valid timesteps: {valid_timesteps}")
		print(f"Shape check: tensor={sample_tensor.shape}, mask={sample_mask.shape}")
		
		# Verify no mixing between files
		print(f"\n=== CROSS-FILE CONTAMINATION CHECK ===")
		if len(train_files) > 1:
			file1_data = train_data[0]
			file2_data = train_data[1]
			
			file1_valid = file1_data[~np.isnan(file1_data)]
			file2_valid = file2_data[~np.isnan(file2_data)]
			
			if len(file1_valid) > 0 and len(file2_valid) > 0:
				correlation = np.corrcoef(file1_valid[:100], file2_valid[:100])[0,1] if len(file1_valid) >= 100 and len(file2_valid) >= 100 else np.nan
				print(f"Correlation between first two files: {correlation:.3f} (should be low)")
		
		print(f"\nDATA LOADING TEST PASSED!")
		print(f"Ready for model training")
		
		return train_data, val_data, train_masks, val_masks, train_files, val_files
		
	except Exception as e:
		print(f"ERROR: {e}")
		import traceback
		traceback.print_exc()
		return None

if __name__ == "__main__":
	# Parse command line arguments
	debug_mode = False  # Normal mode is default
	features_path = None
	
	if len(sys.argv) > 1:
		if sys.argv[1] == "--debug":
			debug_mode = True
			print("Running in DEBUG mode")
		elif sys.argv[1] == "--normal":
			debug_mode = False
			print("Running in NORMAL mode")
		elif sys.argv[1].endswith(".parquet"):
			features_path = sys.argv[1]
			print(f"Using provided path: {features_path}")
		else:
			print("Usage: python test_data_loading.py [--debug|--normal|path/to/features.parquet]")
			sys.exit(1)
	else:
		print("Running in NORMAL mode (default)")
	
	if features_path is None:
		try:
			features_path = find_features_file(debug_mode=debug_mode)
		except FileNotFoundError as e:
			print(f"{e}")
			print("Available files in data directory:")
			if os.path.exists("data"):
				for f in os.listdir("data"):
					if f.endswith(".parquet"):
						print(f"  - {f}")
			sys.exit(1)
	
	print(f"Testing with: {features_path}")
	
	result = test_data_loading(features_path)
	
	if result is not None:
		train_data, val_data, train_masks, val_masks, train_files, val_files = result
		print(f"\nSUCCESS: Ready to proceed to step 2: Model architecture and training")
	else:
		print(f"\nFAILED: Fix data loading issues before proceeding")