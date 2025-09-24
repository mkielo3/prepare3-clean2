"""
Extract eGeMAPS audio features from Common Voice datasets

Usage:
    python feature_extraction.py          # Process all filtered data
    python feature_extraction.py --debug  # Process debug sample (4000 files)
"""

import os
import pandas as pd
import numpy as np
import librosa
import opensmile
import warnings
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import argparse

# Configuration
CACHE_DIR = Path("/d1/adrd/common")
OUTPUT_DIR = Path("./data")
SEGMENT_DURATION = 0.2

warnings.filterwarnings("ignore", message="Segment too short, filling with NaN.")

_directory_cache = None

def build_directory_cache():
	"""Build cache of extraction directories once at startup"""
	global _directory_cache
	if _directory_cache is not None:
		return _directory_cache
		
	print("Building directory cache...")
	_directory_cache = {'en': [], 'es': []}
	
	extracted_base = CACHE_DIR / "downloads" / "extracted"
	if extracted_base.exists():
		for hash_dir in extracted_base.iterdir():
			if hash_dir.is_dir():
				for subdir in hash_dir.iterdir():
					if subdir.is_dir():
						if subdir.name.startswith('en_validated'):
							_directory_cache['en'].append(subdir)
						elif subdir.name.startswith('es_validated'):
							_directory_cache['es'].append(subdir)
	
	print(f"Found {len(_directory_cache['en'])} English, {len(_directory_cache['es'])} Spanish directories")
	return _directory_cache

def find_audio_file(filename):
	"""Find actual path for audio file using directory cache"""
	dirs = build_directory_cache()
	
	if filename.startswith('common_voice_en_'):
		search_dirs = dirs['en']
	elif filename.startswith('common_voice_es_'):
		search_dirs = dirs['es']
	else:
		raise FileNotFoundError(f"Unknown language for file: {filename}")
	
	for directory in search_dirs:
		filepath = directory / filename
		if filepath.exists():
			return str(filepath)
	
	raise FileNotFoundError(f"Audio file not found: {filename}")

def load_metadata(debug=False):
	"""Load metadata from parquet file created by download_and_filter.py"""
	suffix = "_debug" if debug else ""
	metadata_path = OUTPUT_DIR / f"commonvoice_50plus_metadata{suffix}.parquet"
	
	if not metadata_path.exists():
		raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
	
	return pd.read_parquet(metadata_path)

def extract_audio_features(row_data):
	"""Extract eGeMAPS features from audio segments"""
	audio_path = row_data['audio_path']
	client_id = row_data['client_id']
	sentence = row_data['sentence']
	
	if not os.path.exists(audio_path):
		audio_path = find_audio_file(os.path.basename(audio_path))
	
	y, sr = librosa.load(audio_path, sr=None)
	# y = librosa.util.normalize(y)
	total_duration = len(y) / sr
	
	smile = opensmile.Smile(
		feature_set=opensmile.FeatureSet.eGeMAPSv02,
		feature_level=opensmile.FeatureLevel.Functionals,
	)
	
	segment_starts = np.arange(0, total_duration, SEGMENT_DURATION)
	segment_ends = np.minimum(segment_starts + SEGMENT_DURATION, total_duration)
	
	all_features = []
	for row_index, (start, end) in enumerate(zip(segment_starts, segment_ends)):
		start_sample = int(start * sr)
		end_sample = int(end * sr)
		segment_audio = y[start_sample:end_sample]
		
		features = smile.process_signal(segment_audio, sr)
		feature_row = {
			'client_id': client_id,
			'sentence': sentence,
			'audio_path': audio_path,
			'segment_start_sec': start,
			'segment_end_sec': end,
			'row_index': row_index
		}
		
		for col in features.columns:
			feature_row[col] = features.iloc[0][col]
		
		all_features.append(feature_row)
	
	return pd.DataFrame(all_features)

def validate_output(debug=False):
	"""Basic validation of extracted features"""
	print("\n=== Validating Output ===")
	
	suffix = "_debug" if debug else ""
	features_path = OUTPUT_DIR / f"commonvoice_50plus_features{suffix}.parquet"
	
	if not features_path.exists():
		print("Features file missing")
		return False
	
	df = pd.read_parquet(features_path)
	errors = []
	
	required_cols = ['client_id', 'sentence', 'audio_path', 'segment_start_sec', 'segment_end_sec', 'row_index']
	missing_cols = [col for col in required_cols if col not in df.columns]
	if missing_cols:
		errors.append(f"Missing columns: {missing_cols}")
	
	feature_cols = [col for col in df.columns if col not in required_cols]
	if len(feature_cols) < 80:
		errors.append(f"Too few feature columns: {len(feature_cols)}")
	
	for col in required_cols:
		if col in df.columns and df[col].isnull().sum() > 0:
			errors.append(f"Null values in {col}")
	
	total_segments = len(df)
	unique_files = df['audio_path'].nunique()
	
	if debug and (total_segments < 50000 or total_segments > 200000):
		errors.append(f"Debug segment count: {total_segments}")
	elif not debug and total_segments < 500000:
		errors.append(f"Segment count too low: {total_segments}")
	
	if errors:
		print("VALIDATION FAILED:")
		for error in errors:
			print(f"  {error}")
		return False
	else:
		print(f"VALIDATION PASSED: {total_segments} segments from {unique_files} files ({len(feature_cols)} features)")
		return True

def main():
	parser = argparse.ArgumentParser(description='Extract features from filtered Common Voice datasets')
	parser.add_argument('--debug', action='store_true', help='Process debug data only')
	args = parser.parse_args()
	
	metadata_df = load_metadata(debug=args.debug)
	print(f"Loaded {len(metadata_df)} audio files")
	
	row_data_list = []
	for idx, row in metadata_df.iterrows():
		row_data_list.append({
			'client_id': row.get('client_id', f'unknown_{idx}'),
			'sentence': row['sentence'],
			'audio_path': row['path']
		})
	
	build_directory_cache()
	
	num_processes = max(1, os.cpu_count() - 2)
	print(f"Processing {len(row_data_list)} files with {num_processes} processes")
	
	if len(row_data_list) > 50:
		with Pool(num_processes) as pool:
			results = list(tqdm(pool.imap(extract_audio_features, row_data_list), total=len(row_data_list)))
	else:
		results = [extract_audio_features(row) for row in tqdm(row_data_list)]
	
	feature_df = pd.concat([r for r in results if not r.empty], ignore_index=True)
	
	suffix = "_debug" if args.debug else ""
	features_path = OUTPUT_DIR / f"commonvoice_50plus_features{suffix}.parquet"
	feature_df.to_parquet(features_path)
	
	print(f"Saved {len(feature_df)} segments to {features_path}")
	
	success = validate_output(debug=args.debug)
	if not success:
		exit(1)

if __name__ == "__main__":
	main()