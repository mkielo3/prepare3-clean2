"""
Complete data preparation pipeline for Common Voice 50+ dataset

Runs download_and_filter.py then feature_extraction.py

Usage:
	python prep_data.py          # Full pipeline
	python prep_data.py --debug  # Debug pipeline with small samples

Downloading speed depends on connection.
Data processing speed is 90 minutes using 30 cores on a Ryzen 5950x CPU
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

def run_script(script_name, debug=False):
	"""Run a script with optional debug flag, return execution time"""
	cmd = [sys.executable, script_name]
	if debug:
		cmd.append('--debug')
	
	print(f"Running: {' '.join(cmd)}")
	start_time = time.time()
	
	try:
		result = subprocess.run(cmd, check=True, capture_output=False)
		elapsed = time.time() - start_time
		print(f"{script_name} completed in {elapsed:.1f}s")
		return elapsed
	except subprocess.CalledProcessError as e:
		elapsed = time.time() - start_time
		print(f"{script_name} failed after {elapsed:.1f}s (exit code: {e.returncode})")
		raise

def validate_final_output(debug=False):
	"""Verify both metadata and features files exist and are loadable"""
	print("\n=== Final Pipeline Validation ===")
	
	suffix = "_debug" if debug else ""
	data_dir = Path("./data")
	metadata_path = data_dir / f"commonvoice_50plus_metadata{suffix}.parquet"
	features_path = data_dir / f"commonvoice_50plus_features{suffix}.parquet"
	
	if not metadata_path.exists():
		print(f"Missing: {metadata_path}")
		return False
	if not features_path.exists():
		print(f"Missing: {features_path}")
		return False
	
	try:
		import pandas as pd
		metadata_df = pd.read_parquet(metadata_path)
		features_df = pd.read_parquet(features_path)
		
		if len(metadata_df) == 0:
			print(f"Empty metadata file")
			return False
		if len(features_df) == 0:
			print(f"Empty features file")
			return False
		
		print(f"Metadata: {len(metadata_df)} audio files")
		print(f"Features: {len(features_df)} segments from {features_df['audio_path'].nunique()} files")
		return True
		
	except Exception as e:
		print(f"Error loading files: {e}")
		return False

def main():
	parser = argparse.ArgumentParser(description='Run complete Common Voice data preparation pipeline')
	parser.add_argument('--debug', action='store_true', help='Run pipeline in debug mode with small samples')
	args = parser.parse_args()
	
	mode = "debug" if args.debug else "full"
	print(f"=== Common Voice 50+ Data Pipeline ({mode} mode) ===\n")
	
	total_start = time.time()
	
	try:
		print("Step 1: Download and filter Common Voice datasets for 50+ age group")
		if args.debug:
			print("Using existing filtered data, creating 2000-sample debug subset")
		else:
			print("Full download and filtering")
		
		step1_time = run_script('download_and_filter.py', debug=args.debug)
		
		print(f"\nStep 2: Extract eGeMAPS features from audio segments")
		print("Processing audio files in 0.2s segments with parallel processing")
		
		step2_time = run_script('feature_extraction.py', debug=args.debug)
		
		if not validate_final_output(debug=args.debug):
			print("\nPIPELINE VALIDATION FAILED")
			sys.exit(1)
		
		total_time = time.time() - total_start
		print(f"\nPIPELINE COMPLETED SUCCESSFULLY")
		print(f"Total time: {total_time:.1f}s (download: {step1_time:.1f}s, features: {step2_time:.1f}s)")
		
		suffix = "_debug" if args.debug else ""
		print(f"\nOutput files:")
		print(f"data/commonvoice_50plus_metadata{suffix}.parquet")
		print(f"data/commonvoice_50plus_features{suffix}.parquet")
		
		if args.debug:
			print(f"\nDebug datasets cached in: ./debug_cache/")
			
	except subprocess.CalledProcessError:
		print(f"\nPIPELINE FAILED - check error messages above")
		sys.exit(1)
	except KeyboardInterrupt:
		print(f"\nPIPELINE INTERRUPTED by user")
		sys.exit(1)

if __name__ == "__main__":
	main()