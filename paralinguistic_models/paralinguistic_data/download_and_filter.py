"""
Download and filter Common Voice dataset for 50+ speakers

Usage:
    python download_and_filter.py          # Full download and filter
    python download_and_filter.py --debug  # Create 2000-sample debug dataset in local cache
    
Note: Debug mode requires existing filtered datasets from a previous full run.
Debug datasets are saved to ./debug_cache/ for easy sharing with researchers.
"""

import os
import pandas as pd
from pathlib import Path
from datasets import load_dataset, load_from_disk
import argparse

# Configuration
CACHE_DIR = Path("/d1/adrd/common")
OUTPUT_DIR = Path("./data")
LOCAL_DEBUG_CACHE = Path("./debug_cache")

def setup_environment():
	"""Setup cache directories and environment"""
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	LOCAL_DEBUG_CACHE.mkdir(parents=True, exist_ok=True)
	
	os.environ['HF_HUB_CACHE'] = str(CACHE_DIR)
	os.environ['HF_DATASETS_CACHE'] = str(CACHE_DIR)

def filter_common_voice(dataset):
	"""Apply quality and demographic filters"""
	age_categories_50_plus = ['fifties', 'sixties', 'seventies', 'eighties', 'nineties']
	
	dataset = dataset.filter(lambda x: x["age"] in age_categories_50_plus)
	
	dataset = dataset.filter(lambda x: 
		x['up_votes'] >= 2 and 
		x['down_votes'] == 0 and
		x['age'] != '' and x['age'] is not None and
		x['gender'] != '' and x['gender'] is not None and
		len(x['sentence'].strip()) > 0)
	
	return dataset

def extract_metadata(dataset, language):
	"""Extract metadata to DataFrame"""
	metadata = []
	
	for item in dataset:
		metadata.append({
			'path': item['audio']['path'],
			'sampling_rate': item['audio']['sampling_rate'],
			'sentence': item['sentence'],
			'age': item['age'],
			'gender': item['gender'],
			'accent': item['accent'],
			'locale': item['locale'],
			'segment': item['segment'],
			'variant': item['variant'],
			'language': language,
			'client_id': item.get('client_id', f'unknown_{len(metadata)}')
		})
	
	return pd.DataFrame(metadata)

def process_language(language_code, language_name, debug=False):
	"""Download, filter and save a single language"""
	print(f"Processing {language_name}...")
	
	if debug:
		full_dataset_path = CACHE_DIR / f"filtered_commonvoice_50plus_{language_code}_clean"
		if not full_dataset_path.exists():
			raise FileNotFoundError(f"Debug mode requires existing filtered dataset at {full_dataset_path}")
		
		dataset = load_from_disk(str(full_dataset_path))
		debug_size = min(2000, len(dataset))
		dataset = dataset.select(range(debug_size))
		print(f"Using {len(dataset)} samples")
		
		debug_path = LOCAL_DEBUG_CACHE / f"filtered_commonvoice_50plus_{language_code}_clean_debug"
		dataset.save_to_disk(str(debug_path))
		
	else:
		output_path = CACHE_DIR / f"filtered_commonvoice_50plus_{language_code}_clean"
		
		if output_path.exists():
			dataset = load_from_disk(str(output_path))
			print(f"Loaded existing: {len(dataset)} samples")
		else:
			dataset = load_dataset("mozilla-foundation/common_voice_17_0", language_code, 
								  split="train", trust_remote_code=True)
			
			original_size = len(dataset)
			dataset = filter_common_voice(dataset)
			print(f"Filtered: {original_size} -> {len(dataset)} samples")
			
			dataset.save_to_disk(str(output_path))
	
	return extract_metadata(dataset, language_name)

def validate_output(debug=False):
	"""Validate that filtering and metadata extraction worked correctly"""
	print("\n=== Validating Output ===")
	
	errors = []
	
	if debug:
		en_path = LOCAL_DEBUG_CACHE / f"filtered_commonvoice_50plus_en_clean_debug"
		es_path = LOCAL_DEBUG_CACHE / f"filtered_commonvoice_50plus_es_clean_debug" 
		suffix = "_debug"
	else:
		en_path = CACHE_DIR / f"filtered_commonvoice_50plus_en_clean"
		es_path = CACHE_DIR / f"filtered_commonvoice_50plus_es_clean"
		suffix = ""
	
	if not en_path.exists():
		errors.append(f"English filtered dataset missing: {en_path}")
	if not es_path.exists():
		errors.append(f"Spanish filtered dataset missing: {es_path}")
	
	metadata_path = OUTPUT_DIR / f"commonvoice_50plus_metadata{suffix}.parquet"
	if not metadata_path.exists():
		errors.append(f"Metadata file missing: {metadata_path}")
		return False
	
	df = pd.read_parquet(metadata_path)
	
	required_cols = ['path', 'sentence', 'age', 'gender', 'language', 'client_id']
	missing_cols = [col for col in required_cols if col not in df.columns]
	if missing_cols:
		errors.append(f"Missing columns: {missing_cols}")
	
	if 'age' in df.columns:
		valid_ages = {'fifties', 'sixties', 'seventies', 'eighties', 'nineties'}
		invalid_ages = set(df['age'].unique()) - valid_ages
		if invalid_ages:
			errors.append(f"Invalid age categories: {invalid_ages}")
	
	if 'language' in df.columns:
		expected_langs = {'English', 'Spanish'}
		actual_langs = set(df['language'].unique())
		if actual_langs != expected_langs:
			errors.append(f"Wrong languages: {actual_langs}")
	
	total_samples = len(df)
	if debug and (total_samples > 5000 or total_samples < 100):
		errors.append(f"Debug sample count: {total_samples} (expected ~4000)")
	elif not debug and total_samples < 1000:
		errors.append(f"Sample count too low: {total_samples}")
	
	for col in ['path', 'sentence', 'age', 'gender']:
		if col in df.columns and df[col].isnull().sum() > 0:
			errors.append(f"Null values in {col}")
	
	if errors:
		print("VALIDATION FAILED:")
		for error in errors:
			print(f"  {error}")
		return False
	else:
		print(f"VALIDATION PASSED: {len(df)} samples")
		if 'language' in df.columns and 'age' in df.columns:
			breakdown = df.groupby(['language', 'age']).size()
			for (lang, age), count in breakdown.items():
				print(f"  {lang} {age}: {count}")
		return True

def main():
	parser = argparse.ArgumentParser(description='Download and filter Common Voice 50+ dataset')
	parser.add_argument('--debug', action='store_true', help='Create 2000-sample debug dataset in local cache')
	args = parser.parse_args()
	
	setup_environment()
	
	en_metadata = process_language("en", "English", debug=args.debug)
	es_metadata = process_language("es", "Spanish", debug=args.debug)
	
	combined_metadata = pd.concat([en_metadata, es_metadata], ignore_index=True)
	
	suffix = "_debug" if args.debug else ""
	metadata_path = OUTPUT_DIR / f"commonvoice_50plus_metadata{suffix}.parquet"
	combined_metadata.to_parquet(metadata_path, index=False)
	
	print(f"Saved metadata: {len(combined_metadata)} examples to {metadata_path}")
	
	success = validate_output(debug=args.debug)
	if not success:
		exit(1)

if __name__ == "__main__":
	main()