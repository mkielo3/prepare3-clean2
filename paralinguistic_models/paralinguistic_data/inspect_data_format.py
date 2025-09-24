"""
Inspect the exact format of processed Common Voice 50+ data

Usage:
	python inspect_data_format.py          # Inspect full dataset
	python inspect_data_format.py --debug  # Inspect debug dataset
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def inspect_dataframe(df, name):
	"""Print comprehensive format information for a dataframe"""
	print(f"\n=== {name} ===")
	print(f"Shape: {df.shape}")
	print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
	
	print(f"\nColumn dtypes:")
	for col, dtype in df.dtypes.items():
		null_count = df[col].isnull().sum()
		null_pct = (null_count / len(df)) * 100
		print(f"  {col}: {dtype} ({null_count} nulls, {null_pct:.1f}%)")
	
	print(f"\nFirst 3 rows:")
	with pd.option_context('display.max_columns', None, 'display.width', None):
		print(df.head(3).to_string(index=False))
	
	print(f"\nSample statistics for numeric columns:")
	numeric_cols = df.select_dtypes(include=[np.number]).columns
	if len(numeric_cols) > 0:
		stats = df[numeric_cols].describe()
		print(stats.round(3).to_string())
	
	print(f"\nUnique value counts for categorical columns:")
	categorical_cols = df.select_dtypes(include=['object']).columns
	for col in categorical_cols[:5]:  # Limit to first 5 to avoid spam
		unique_count = df[col].nunique()
		print(f"  {col}: {unique_count} unique values")
		if unique_count <= 10:
			print(f"    Values: {list(df[col].unique())}")
		else:
			print(f"    Sample: {list(df[col].unique()[:5])} ...")

def validate_row_index(df):
	"""Validate row_index column works correctly on sample of data"""
	print(f"\n=== ROW INDEX VALIDATION ===")
	
	# Check column exists and dtype
	if 'row_index' not in df.columns:
		print("ERROR: row_index column missing")
		return False
	
	if df['row_index'].dtype != 'int64':
		print(f"WARNING: row_index dtype is {df['row_index'].dtype}, expected int64")
	
	# Check for nulls and negatives (fast aggregate checks)
	null_count = df['row_index'].isnull().sum()
	if null_count > 0:
		print(f"ERROR: {null_count} null values in row_index")
		return False
	
	negative_count = (df['row_index'] < 0).sum()
	if negative_count > 0:
		print(f"ERROR: {negative_count} negative values in row_index")
		return False
	
	# Sample a few audio files for detailed validation
	unique_audio_files = df['audio_path'].unique()
	sample_size = min(10, len(unique_audio_files))
	sample_files = unique_audio_files[:sample_size]  # Take first N files instead of random
	
	print(f"Detailed validation on {sample_size} of {len(unique_audio_files)} audio files...")
	
	errors = []
	for audio_path in sample_files:
		group = df[df['audio_path'] == audio_path].sort_values('segment_start_sec')
		
		# Check row_index starts at 0
		if group['row_index'].iloc[0] != 0:
			errors.append(f"{audio_path}: row_index doesn't start at 0")
		
		# Check sequential increments
		expected_sequence = list(range(len(group)))
		actual_sequence = group['row_index'].tolist()
		if actual_sequence != expected_sequence:
			errors.append(f"{audio_path}: row_index not sequential")
		
		# Check alignment with segment timing
		start_times = group['segment_start_sec'].tolist()
		if start_times != sorted(start_times):
			errors.append(f"{audio_path}: segment_start_sec not ordered")
	
	# Report sample errors
	if errors:
		print(f"SAMPLE VALIDATION FAILED: {len(errors)} sample files with errors")
		for error in errors:
			print(f"  {error}")
		return False
	
	# Quick aggregate checks on full dataset
	total_audio_files = df['audio_path'].nunique()
	segments_per_file = df.groupby('audio_path').size()
	max_indices_per_file = df.groupby('audio_path')['row_index'].max()
	expected_max_indices = segments_per_file - 1
	
	# Check if max row_index matches segment count for all files
	mismatched_files = (max_indices_per_file != expected_max_indices).sum()
	if mismatched_files > 0:
		print(f"ERROR: {mismatched_files} of {total_audio_files} files have mismatched max row_index")
		return False
	
	# Display sample sequences
	print(f"\nSample sequences from {min(3, len(sample_files))} audio files:")
	for audio_path in sample_files[:3]:
		file_data = df[df['audio_path'] == audio_path].sort_values('segment_start_sec')
		print(f"\n{audio_path[-50:]}:")  # Show last 50 chars of path
		display_cols = ['row_index', 'segment_start_sec', 'segment_end_sec']
		sample_rows = file_data[display_cols].head(5)  # Show first 5 segments only
		print(sample_rows.to_string(index=False))
		if len(file_data) > 5:
			print(f"  ... ({len(file_data) - 5} more segments)")
	
	# Summary stats
	print(f"\n=== ROW INDEX SUMMARY ===")
	print(f"Total audio files: {total_audio_files}")
	print(f"Segments per file - Min: {segments_per_file.min()}, Max: {segments_per_file.max()}, Avg: {segments_per_file.mean():.1f}")
	print(f"Overall row_index range: {df['row_index'].min()} to {df['row_index'].max()}")
	print("ROW INDEX VALIDATION PASSED")
	return True

def main():
	parser = argparse.ArgumentParser(description='Inspect processed Common Voice data format')
	parser.add_argument('--debug', action='store_true', help='Inspect debug dataset')
	args = parser.parse_args()
	
	suffix = "_debug" if args.debug else ""
	data_dir = Path("./data")
	
	metadata_path = data_dir / f"commonvoice_50plus_metadata{suffix}.parquet"
	features_path = data_dir / f"commonvoice_50plus_features{suffix}.parquet"
	
	if not metadata_path.exists():
		print(f"Metadata file not found: {metadata_path}")
		return
	
	if not features_path.exists():
		print(f"Features file not found: {features_path}")
		return
	
	print("Loading data files...")
	metadata_df = pd.read_parquet(metadata_path)
	features_df = pd.read_parquet(features_path)
	
	inspect_dataframe(metadata_df, "METADATA")
	inspect_dataframe(features_df, "FEATURES")
	
	print(f"\n=== RELATIONSHIP ANALYSIS ===")
	unique_files_metadata = metadata_df['client_id'].nunique()
	unique_files_features = features_df['client_id'].nunique()
	unique_audio_paths = features_df['audio_path'].nunique()
	
	print(f"Unique speakers in metadata: {unique_files_metadata}")
	print(f"Unique speakers in features: {unique_files_features}")
	print(f"Unique audio files in features: {unique_audio_paths}")
	print(f"Average segments per audio file: {len(features_df) / unique_audio_paths:.1f}")
	
	feature_cols = [col for col in features_df.columns if col not in 
	               ['client_id', 'sentence', 'audio_path', 'segment_start_sec', 'segment_end_sec', 'row_index']]
	print(f"eGeMAPS feature columns: {len(feature_cols)}")
	print(f"Sample feature names: {feature_cols[:5]}")
	
	# Validate row_index ordering
	validate_row_index(features_df)

if __name__ == "__main__":
	main()