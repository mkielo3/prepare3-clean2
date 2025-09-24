"""
Fine-tune classifier by extracting features from MP3 files with caching and multiprocessing.
Modified to save predictions for all datasets after training.
"""


# Global configuration
ENCODER_PATH = '../paralinguistic_models/paralinguistic_models/checkpoints/best_model.pth'
METADATA_PATH = 'data/metadata.csv'
ADDITIONAL_METADATA_PATH = 'data/acoustic_additional_metadata.csv'
DEFAULT_AUDIO_DIR = 'data/train_audios'
DEFAULT_TEST_AUDIO_DIR = 'data/test_audios'
DEFAULT_LABELS_PATH = 'data/train_labels.csv'
DEFAULT_CACHE_DIR = 'paralinguistic/feature_cache'
OUTPUT_MODEL_PATH = 'paralinguistic/best_classifier.pth'
PREDICTIONS_OUTPUT_PATH = 'paralinguistic/all_predictions.pkl'

# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
PATIENCE = 10
SEGMENTS_PER_SAMPLE = 3
INFERENCE_SAMPLES = 20
SEGMENT_DURATION = 0.2
SEGMENT_LENGTH = 45

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import random
import argparse
import os
import hashlib
from pathlib import Path
import librosa
import opensmile
import warnings
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle

warnings.filterwarnings("ignore", message="Segment too short, filling with NaN.")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'paralinguistic_models'))
from shared.model import UpgradedAcousticEncoder

def save_features_to_cache(cache_dir, train_features_df, val_features_df, test_features_df, 
						  train_uid_list, val_uid_list, test_uid_list):
	"""Save extracted features to cache files"""
	cache_path = Path(cache_dir)
	cache_path.mkdir(exist_ok=True)
	
	if len(train_features_df) > 0:
		train_features_df.to_parquet(cache_path / "train_features.parquet")
	if len(val_features_df) > 0:
		val_features_df.to_parquet(cache_path / "val_features.parquet")
	if len(test_features_df) > 0:
		test_features_df.to_parquet(cache_path / "test_features.parquet")
	
	cache_metadata = {
		'train_uid_list': train_uid_list,
		'val_uid_list': val_uid_list,
		'test_uid_list': test_uid_list
	}
	
	with open(cache_path / "metadata.pkl", 'wb') as f:
		pickle.dump(cache_metadata, f)

def load_features_from_cache(cache_dir):
	"""Load extracted features from cache files"""
	cache_path = Path(cache_dir)
	
	metadata_file = cache_path / "metadata.pkl"
	if not metadata_file.exists():
		return None
	
	with open(metadata_file, 'rb') as f:
		cache_metadata = pickle.load(f)
	
	train_features_df = pd.DataFrame()
	val_features_df = pd.DataFrame()
	test_features_df = pd.DataFrame()
	
	train_file = cache_path / "train_features.parquet"
	if train_file.exists():
		train_features_df = pd.read_parquet(train_file)
	
	val_file = cache_path / "val_features.parquet"
	if val_file.exists():
		val_features_df = pd.read_parquet(val_file)
	
	test_file = cache_path / "test_features.parquet"
	if test_file.exists():
		test_features_df = pd.read_parquet(test_file)
	
	return (train_features_df, val_features_df, test_features_df, 
			cache_metadata['train_uid_list'], cache_metadata['val_uid_list'], cache_metadata['test_uid_list'])

def extract_features_from_audio_file(audio_info):
	"""Extract eGeMAPS features from a single audio file - for multiprocessing"""
	uid, audio_path = audio_info
	segment_duration = SEGMENT_DURATION
	
	y, sr = librosa.load(audio_path, sr=None)
	total_duration = len(y) / sr
	
	smile = opensmile.Smile(
		feature_set=opensmile.FeatureSet.eGeMAPSv02,
		feature_level=opensmile.FeatureLevel.Functionals,
	)
	
	segment_starts = np.arange(0, total_duration, segment_duration)
	segment_ends = np.minimum(segment_starts + segment_duration, total_duration)
	
	all_features = []
	for row_index, (start, end) in enumerate(zip(segment_starts, segment_ends)):
		start_sample = int(start * sr)
		end_sample = int(end * sr)
		segment_audio = y[start_sample:end_sample]
		
		if len(segment_audio) == 0:
			continue
		
		features = smile.process_signal(segment_audio, sr)
		
		feature_row = {
			'uid': uid,
			'row_index': row_index,
			'segment_start_sec': start
		}
		for col in features.columns:
			feature_row[col] = features.iloc[0][col]
		all_features.append(feature_row)
	
	if len(all_features) == 0:
		return uid, pd.DataFrame()
	
	return uid, pd.DataFrame(all_features)

def prepare_for_reshape(df, feature_cols, split_name="unknown"):
	"""Convert to stacked format expected by reshape functions"""
	if len(df) == 0:
		empty_index = pd.MultiIndex.from_tuples([], names=['uid', 'segment_start_sec', 'feature'])
		return pd.Series([], dtype=float, index=empty_index)
	
	stacked_data = []
	for _, row in df.iterrows():
		uid = row['uid']
		segment_start = row['segment_start_sec']
		
		for feature_col in feature_cols:
			if feature_col in df.columns:
				stacked_data.append({
					'uid': uid,
					'segment_start_sec': segment_start,
					'feature': feature_col,
					'value': row[feature_col]
				})
	
	if len(stacked_data) == 0:
		empty_index = pd.MultiIndex.from_tuples([], names=['uid', 'segment_start_sec', 'feature'])
		return pd.Series([], dtype=float, index=empty_index)
	
	stacked_df = pd.DataFrame(stacked_data)
	return stacked_df.set_index(['uid', 'segment_start_sec', 'feature'])['value']

def process_uid_batch(uid_batch_data):
	"""Process a batch of UIDs - for multiprocessing"""
	uid_batch, features = uid_batch_data
	
	data_list = []
	file_list = []
	timestep_lengths = []
	
	for uid, uid_data in uid_batch:
		timestamps = sorted(uid_data['segment_start_sec'].unique())
		
		if len(timestamps) == 0:
			continue
		
		uid_matrix = np.full((len(features), len(timestamps)), np.nan)
		
		for _, row in uid_data.iterrows():
			feature_idx = features.index(row['feature'])
			time_idx = timestamps.index(row['segment_start_sec'])
			uid_matrix[feature_idx, time_idx] = row['value']
		
		data_list.append(uid_matrix)
		file_list.append(f"{uid}_segment")
		timestep_lengths.append(len(timestamps))
	
	return data_list, file_list, timestep_lengths

def reshape_to_variable_length_list_parallel(stacked_series, split_name="unknown", n_processes=None):
	"""Multiprocessing version for faster reshaping"""
	if len(stacked_series) == 0:
		return [], []
	
	if n_processes is None:
		n_processes = max(1, cpu_count() - 1)
	
	print(f"Reshaping {split_name} with {n_processes} processes ({len(stacked_series):,} rows)")
	
	df = stacked_series.reset_index()
	grouped = df.groupby('uid')
	uids = list(grouped.groups.keys())
	features = sorted(df['feature'].unique())
	
	batch_size = max(1, len(uids) // n_processes)
	uid_batches = []
	
	for i in range(0, len(uids), batch_size):
		batch_uids = uids[i:i + batch_size]
		batch_data = [(uid, grouped.get_group(uid)) for uid in batch_uids]
		uid_batches.append((batch_data, features))
	
	if len(uid_batches) > 1:
		with Pool(n_processes) as pool:
			results = list(tqdm(
				pool.imap(process_uid_batch, uid_batches), 
				total=len(uid_batches),
				desc=f"Processing {split_name} batches"
			))
	else:
		results = [process_uid_batch(batch) for batch in uid_batches]
	
	all_data_list = []
	all_file_list = []
	all_timestep_lengths = []
	
	for data_list, file_list, timestep_lengths in results:
		all_data_list.extend(data_list)
		all_file_list.extend(file_list)
		all_timestep_lengths.extend(timestep_lengths)
	
	if len(all_data_list) > 0:
		print(f"{split_name} reshape complete: {len(all_data_list)} samples, timesteps: min={min(all_timestep_lengths)}, max={max(all_timestep_lengths)}, mean={np.mean(all_timestep_lengths):.1f}")
	
	return all_data_list, all_file_list

CHANNEL_STATS = {
	0: (20.82, 15.07), 1: (11.49, 99.74), 2: (31.44, 124.67), 3: (2.37, 5.67), 4: (19.66, 14.46),
	5: (20.79, 15.21), 6: (22.02, 16.21), 7: (1.95, 13.86), 8: (0.05, 0.09), 9: (1.38, 13.10),
	10: (-93.27, 81.61), 11: (-0.57, 350.86), 12: (865.55, 567.03), 13: (0.11, 0.13), 14: (468.79, 341.06),
	15: (0.17, 0.17), 16: (-96.88, 75.14), 17: (-0.48, 75.02), 18: (675.73, 476.66), 19: (0.17, 0.18),
	20: (1251.23, 784.36), 21: (0.08, 0.07), 22: (-98.71, 73.35), 23: (-0.39, 146.04), 24: (632.08, 449.11),
	25: (0.19, 0.19), 26: (2025.64, 1237.87), 27: (0.05, 0.05), 28: (1.91, 4.07), 29: (0.39, 624.89),
	30: (0.05, 0.05), 31: (0.06, 0.05), 32: (0.00, 0.00), 33: (0.00, 0.00), 34: (6.00, 4.15),
	35: (-2.91, 9.31), 36: (-9.65, 11.89), 37: (-0.19, 147.97), 38: (-30.46, 20.84), 39: (8.13, 11.67),
	40: (16.44, 14.90), 41: (0.07, 190.18), 42: (0.03, 0.05), 43: (0.34, 0.37), 44: (14.41, 17.64),
	45: (2.34, 4205.81), 46: (2.85, 13.07), 47: (0.82, 1718.57), 48: (5.44, 3.92), 49: (1.24, 0.98),
	50: (4.04, 14.67), 51: (13.76, 15.59), 52: (0.91, 0.88), 53: (0.79, 0.75), 54: (1.20, 1.02),
	55: (1.70, 1.33), 56: (3.19, 6.23), 57: (0.40, 0.28), 58: (2.14, 4.69), 59: (18.48, 19.06),
	60: (0.01, 880.50), 61: (18.68, 16.31), 62: (0.78, 759.66), 63: (6.21, 14.19), 64: (0.25, 492.10),
	65: (7.32, 13.66), 66: (-1.02, 1783.17), 67: (11.96, 17.84), 68: (0.18, 355.16), 69: (13.14, 16.16),
	70: (0.63, 761.13), 71: (-0.97, 14.66), 72: (-0.03, 321.70), 73: (-0.20, 13.66), 74: (-2.88, 7410.15),
	75: (1.12, 1.08), 76: (0.29, 0.28), 77: (0.01, 0.05), 78: (0.00, 0.01), 79: (0.02, 0.05),
	80: (-0.20, 693.57), 81: (-0.01, 0.02), 82: (-1.06, 1259.15), 83: (0.39, 0.80), 84: (1.25, 1.29),
	85: (0.26, 0.26), 86: (1.16, 1.08), 87: (0.56, 0.34)
}

def normalize_variable_length_data(data_list):
	"""Normalize using channel statistics"""
	normalized_list = []
	
	for data_array in data_list:
		data_norm = data_array.copy()
		
		for channel in range(min(88, data_array.shape[0])):
			if channel in CHANNEL_STATS:
				mean, std = CHANNEL_STATS[channel]
				if std > 0:
					data_norm[channel, :] = (data_array[channel, :] - mean) / (std + 1e-8)
		
		data_norm = np.nan_to_num(data_norm, nan=0.0, posinf=0.0, neginf=0.0)
		normalized_list.append(data_norm)
	
	return normalized_list

def load_audio_files_and_extract_features(audio_dir, labels_path, metadata_path, additional_metadata_path, test_audio_dir=DEFAULT_TEST_AUDIO_DIR, cache_dir="./feature_cache", force_recompute=False):
	"""Load MP3 files and extract features with simplified caching"""

	cached_data = None
	if not force_recompute:
		print("Checking for cached features...")
		cached_data = load_features_from_cache(cache_dir)
		
		if cached_data is not None:
			print("USING CACHED DATA - test files may not be included!")
			# Load metadata and labels even when using cache
			train_labels_df = pd.read_csv(labels_path)
			
			if 'diagnosis_mci' in train_labels_df.columns and 'diagnosis_adrd' in train_labels_df.columns:
				train_labels_df['y'] = (train_labels_df['diagnosis_mci'] + train_labels_df['diagnosis_adrd']).astype(int)
			
			metadata_df = pd.read_csv(metadata_path).set_index('uid')
			additional_df = pd.read_csv(additional_metadata_path).set_index('uid')
			metadata = pd.concat([metadata_df, additional_df], axis=1)
			
			# Debug cached data
			train_features_df, val_features_df, test_features_df, train_uid_list, val_uid_list, test_uid_list = cached_data
			print(f"CACHED DATA: train={len(train_features_df)}, val={len(val_features_df)}, test={len(test_features_df)}")
			print(f"CACHED UIDS: train={len(train_uid_list)}, val={len(val_uid_list)}, test={len(test_uid_list)}")
			
			return cached_data, metadata, train_labels_df
	
	train_labels_df = pd.read_csv(labels_path)
	
	if 'diagnosis_mci' in train_labels_df.columns and 'diagnosis_adrd' in train_labels_df.columns:
		train_labels_df['y'] = (train_labels_df['diagnosis_mci'] + train_labels_df['diagnosis_adrd']).astype(int)
	
	metadata_df = pd.read_csv(metadata_path).set_index('uid')
	additional_df = pd.read_csv(additional_metadata_path).set_index('uid')
	metadata = pd.concat([metadata_df, additional_df], axis=1)
	
	train_ids = list(metadata[metadata['split'] == 'train'].index)
	test_ids = list(metadata[metadata['split'] == 'test'].index)
	
	print(f"Found {len(train_ids)} train IDs and {len(test_ids)} test IDs in metadata")
	
	# Find all available audio files (both train and test)
	train_audio_files = list(Path(audio_dir).glob("*.mp3"))
	test_audio_files = list(Path(test_audio_dir).glob("*.mp3"))
	
	print(f"Searching for train files in: {audio_dir}")
	print(f"Searching for test files in: {test_audio_dir}")
	print(f"Found {len(train_audio_files)} train .mp3 files, {len(test_audio_files)} test .mp3 files")
	
	available_uids = set()
	uid_to_audio = {}
	train_files_found = 0
	test_files_found = 0
	
	# Process train files
	for audio_file in train_audio_files:
		uid = audio_file.stem
		if uid in train_ids:
			available_uids.add(uid)
			uid_to_audio[uid] = str(audio_file)
			train_files_found += 1
	
	# Process test files
	for audio_file in test_audio_files:
		uid = audio_file.stem
		if uid in test_ids:
			available_uids.add(uid)
			uid_to_audio[uid] = str(audio_file)
			test_files_found += 1
	
	print(f"Matched files: {train_files_found} train, {test_files_found} test")
	print(f"Found {len(available_uids)} total available audio files")
	
	# Filter to only available files
	train_ids = [uid for uid in train_ids if uid in available_uids]
	test_ids = [uid for uid in test_ids if uid in available_uids]
	
	print(f"After filtering: {len(train_ids)} train files, {len(test_ids)} test files")
	
	adrd_train_ids, adrd_val_ids = train_test_split(train_ids, test_size=0.1, random_state=42)

	def extract_features_for_split_parallel(uid_list, split_name):
		if len(uid_list) == 0:
			return pd.DataFrame(), []
		
		audio_info_list = [(uid, uid_to_audio[uid]) for uid in uid_list]
		num_processes = max(1, cpu_count() - 2)
		
		if len(audio_info_list) > 10:
			with Pool(num_processes) as pool:
				results = list(tqdm(
					pool.imap(extract_features_from_audio_file, audio_info_list), 
					total=len(audio_info_list),
					desc=f"Processing {split_name}"
				))
		else:
			results = [extract_features_from_audio_file(info) for info in audio_info_list]
		
		all_segments = []
		successful_uids = []
		
		for uid, feature_df in results:
			if len(feature_df) > 0:
				all_segments.append(feature_df)
				successful_uids.append(uid)
		
		if len(all_segments) == 0:
			return pd.DataFrame(), []
		
		combined_df = pd.concat(all_segments, ignore_index=True)
		print(f"{split_name} features extracted: {len(combined_df)} segments from {len(successful_uids)} files")
		
		return combined_df, successful_uids
	
	train_features_df, train_uid_list = extract_features_for_split_parallel(adrd_train_ids, "train")
	val_features_df, val_uid_list = extract_features_for_split_parallel(adrd_val_ids, "validation")
	
	if len(test_ids) > 0:
		test_features_df, test_uid_list = extract_features_for_split_parallel(test_ids, "test")
	else:
		test_features_df = pd.DataFrame()
		test_uid_list = []
	
	print("Saving extracted features to cache...")
	save_features_to_cache(cache_dir, train_features_df, val_features_df, test_features_df,
						  train_uid_list, val_uid_list, test_uid_list)
	
	return (train_features_df, val_features_df, test_features_df, train_uid_list, val_uid_list, test_uid_list), metadata, train_labels_df

class VariableLengthSegmentDataset(Dataset):
	"""Dataset for variable-length sequences with segment sampling"""
	def __init__(self, data_list, labels, segment_length=SEGMENT_LENGTH, mode='train', segments_per_sample=SEGMENTS_PER_SAMPLE):
		self.data_list = data_list
		self.labels = labels
		self.segment_length = segment_length
		self.mode = mode
		self.segments_per_sample = segments_per_sample if mode == 'train' else 1
		
		lengths = [arr.shape[1] for arr in data_list]
		print(f"{mode} dataset: {len(data_list)} samples, timesteps: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
		
	def __len__(self):
		return len(self.data_list) * self.segments_per_sample
	
	def __getitem__(self, idx):
		sample_idx = idx // self.segments_per_sample
		
		full_sequence = self.data_list[sample_idx]
		label = self.labels[sample_idx]
		actual_timesteps = full_sequence.shape[1]
		
		valid_mask = ~np.isnan(full_sequence).all(axis=0)
		valid_indices = np.where(valid_mask)[0]
		
		if len(valid_indices) < self.segment_length:
			if actual_timesteps < self.segment_length:
				padding_needed = self.segment_length - actual_timesteps
				padding = np.zeros((full_sequence.shape[0], padding_needed))
				padded_sequence = np.concatenate([full_sequence, padding], axis=1)
				segment = np.nan_to_num(padded_sequence, nan=0.0)
				mask = np.concatenate([valid_mask, np.zeros(padding_needed, dtype=bool)])
			else:
				segment = np.nan_to_num(full_sequence[:, :self.segment_length], nan=0.0)
				mask = valid_mask[:self.segment_length]
		else:
			max_start = len(valid_indices) - self.segment_length
			start_idx = random.randint(0, max_start)
			selected_indices = valid_indices[start_idx:start_idx + self.segment_length]
			
			segment = np.nan_to_num(full_sequence[:, selected_indices], nan=0.0)
			mask = np.ones(self.segment_length, dtype=bool)
		
		return torch.FloatTensor(segment), torch.BoolTensor(mask), torch.LongTensor([label])

class ClassificationModel(nn.Module):
	"""Classification model with pretrained encoder"""
	def __init__(self, encoder_path, num_classes=2, freeze_encoder=True, partial_unfreeze=False):
		super().__init__()
		
		self.encoder = UpgradedAcousticEncoder(
			n_channels=88, 
			embedding_dim=512,
			nhead=16,
			num_encoder_layers=8,
			dim_feedforward=2048
		)
		
		self.encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
		
		if freeze_encoder:
			for param in self.encoder.parameters():
				param.requires_grad = False
		elif partial_unfreeze:
			for param in self.encoder.parameters():
				param.requires_grad = False
			for param in self.encoder.transformer_encoder.layers[-1].parameters():
				param.requires_grad = True
			for param in self.encoder.projection.parameters():
				param.requires_grad = True
		
		encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
		encoder_total = sum(p.numel() for p in self.encoder.parameters())
		print(f"Encoder parameters: {encoder_trainable:,} trainable / {encoder_total:,} total")
		
		self.classifier = nn.Sequential(
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(128, num_classes)
		)
		
	def forward(self, x, mask=None):
		if not any(p.requires_grad for p in self.encoder.parameters()):
			with torch.no_grad():
				embeddings = self.encoder(x, mask)
		else:
			embeddings = self.encoder(x, mask)
		
		logits = self.classifier(embeddings)
		return logits

def train_classifier(model, train_loader, val_loader, device, epochs=50, lr=1e-3, weight_decay=1e-3, patience=10):
	"""Train the classifier"""
	model.to(device)
	
	encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
	classifier_params = list(model.classifier.parameters())
	
	if encoder_params:
		optimizer = torch.optim.Adam([
			{'params': encoder_params, 'lr': lr * 0.1},
			{'params': classifier_params, 'lr': lr}
		], weight_decay=weight_decay)
	else:
		optimizer = torch.optim.Adam(classifier_params, lr=lr, weight_decay=weight_decay)
	
	criterion = nn.CrossEntropyLoss()
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
	
	best_val_auc = 0
	patience_counter = 0
	best_epoch = 0
	
	for epoch in range(epochs):
		# Training
		model.train()
		train_loss = 0
		train_preds, train_targets, train_probs = [], [], []
		
		for batch_x, batch_mask, batch_y in train_loader:
			batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.squeeze().to(device)
			
			optimizer.zero_grad()
			logits = model(batch_x, batch_mask)
			loss = criterion(logits, batch_y)
			loss.backward()
			optimizer.step()
			
			train_loss += loss.item()
			probs = F.softmax(logits, dim=1)
			train_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
			train_targets.extend(batch_y.cpu().numpy())
			train_probs.extend(probs[:, 1].detach().cpu().numpy())
		
		# Validation
		model.eval()
		val_loss = 0
		val_preds, val_targets, val_probs = [], [], []
		
		with torch.no_grad():
			for batch_x, batch_mask, batch_y in val_loader:
				batch_x, batch_mask, batch_y = batch_x.to(device), batch_mask.to(device), batch_y.squeeze().to(device)
				
				logits = model(batch_x, batch_mask)
				loss = criterion(logits, batch_y)
				
				val_loss += loss.item()
				probs = F.softmax(logits, dim=1)
				val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
				val_targets.extend(batch_y.cpu().numpy())
				val_probs.extend(probs[:, 1].cpu().numpy())
		
		# Calculate metrics
		train_acc = accuracy_score(train_targets, train_preds)
		train_f1 = f1_score(train_targets, train_preds)
		train_auc = roc_auc_score(train_targets, train_probs)
		
		val_acc = accuracy_score(val_targets, val_preds)
		val_f1 = f1_score(val_targets, val_preds)
		val_auc = roc_auc_score(val_targets, val_probs)
		
		scheduler.step(val_loss)
		
		print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
		print(f"                    Val   Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
		
		if val_auc > best_val_auc:
			best_val_auc = val_auc
			best_epoch = epoch + 1
			patience_counter = 0
			torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
			print(f"*** New best validation AUC: {val_auc:.4f} ***")
		else:
			patience_counter += 1
		
		if patience_counter >= patience:
			print(f"Early stopping triggered after {epoch+1} epochs!")
			print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
			break
	
	return model

def inference_with_averaging_variable_length(model, data_sample, device, n_samples=INFERENCE_SAMPLES, segment_length=SEGMENT_LENGTH):
	"""Inference with multiple segment sampling and averaging"""
	model.eval()
	model.to(device)
	
	actual_timesteps = data_sample.shape[1]
	valid_mask = ~np.isnan(data_sample).all(axis=0)
	valid_indices = np.where(valid_mask)[0]
	
	if len(valid_indices) < segment_length:
		if actual_timesteps < segment_length:
			padding = np.zeros((data_sample.shape[0], segment_length - actual_timesteps))
			padded_sample = np.concatenate([data_sample, padding], axis=1)
		else:
			padded_sample = data_sample[:, :segment_length]
		
		segment = np.nan_to_num(padded_sample, nan=0.0)
		mask = np.concatenate([valid_mask[:min(len(valid_indices), segment_length)], 
							  np.zeros(max(0, segment_length - len(valid_indices)), dtype=bool)])
		
		with torch.no_grad():
			segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(device)
			mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(device)
			logits = model(segment_tensor, mask_tensor)
			probs = F.softmax(logits, dim=1)
		
		return probs.cpu().numpy()[0]
	
	all_probs = []
	confidences = []
	
	with torch.no_grad():
		for _ in range(n_samples):
			max_start = len(valid_indices) - segment_length
			start_idx = random.randint(0, max_start)
			selected_indices = valid_indices[start_idx:start_idx + segment_length]
			
			segment = np.nan_to_num(data_sample[:, selected_indices], nan=0.0)
			mask = np.ones(segment_length, dtype=bool)
			
			segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(device)
			mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(device)
			
			logits = model(segment_tensor, mask_tensor)
			probs = F.softmax(logits, dim=1)
			prob_array = probs.cpu().numpy()[0]
			
			all_probs.append(prob_array)
			confidences.append(np.max(prob_array))
	
	all_probs = np.array(all_probs)
	confidences = np.array(confidences)
	weights = confidences / confidences.sum()
	averaged_probs = np.average(all_probs, axis=0, weights=weights)
	
	return averaged_probs

def evaluate_all_splits(model, train_data_norm, val_data_norm, test_data_norm, 
					   train_labels, val_labels, test_labels, 
					   train_files, val_files, test_files, device):
	"""Run evaluation on all data splits and save predictions in inference script format"""
	print("\nRunning final evaluation on all datasets...")
	print(f"Data counts: train={len(train_data_norm)}, val={len(val_data_norm)}, test={len(test_data_norm)}")
	
	model.eval()
	results = []
	
	# Process each split separately
	splits_to_process = [
		(train_data_norm + val_data_norm, 
		 np.concatenate([train_labels, val_labels]) if len(val_labels) > 0 else train_labels,
		 train_files + val_files, 'train'),
		(test_data_norm, test_labels, test_files, 'test')
	]
	
	for data_norm, labels, files, split_name in splits_to_process:
		if len(data_norm) == 0:
			print(f"Skipping {split_name} split - no data")
			continue
		
		print(f"\nProcessing {split_name} split with {len(data_norm)} segments")
		
		# Group segments by UID (audio file) to get one prediction per file
		uid_to_data = {}
		for i, file_id in enumerate(files):
			uid = file_id.split("_")[0]
			if uid not in uid_to_data:
				uid_to_data[uid] = {'indices': [], 'label': labels[i]}
			uid_to_data[uid]['indices'].append(i)
		
		print(f"Found {len(uid_to_data)} unique audio files in {split_name} split")
		
		file_predictions = []
		file_labels = []
		
		# Process each unique audio file
		for uid, uid_info in tqdm(uid_to_data.items(), desc=f"{split_name.capitalize()} files"):
			indices = uid_info['indices']
			true_label = uid_info['label']
			
			# Run inference on all segments from this audio file
			segment_probs = []
			for idx in indices:
				probs = inference_with_averaging_variable_length(
					model, data_norm[idx], device, n_samples=INFERENCE_SAMPLES
				)
				segment_probs.append(probs)
			
			# Average probabilities across all segments from same audio file
			avg_probs = np.mean(segment_probs, axis=0)
			pred = np.argmax(avg_probs)
			confidence = np.max(avg_probs)
			
			# Estimate duration from segments
			total_timesteps = sum(data_norm[idx].shape[1] for idx in indices)
			avg_duration_per_segment = total_timesteps * SEGMENT_DURATION / len(indices)
			
			result = {
				'uid': uid,
				'split': split_name,
				'prediction': int(pred),
				'confidence': float(confidence),
				'prob_class_0': float(avg_probs[0]),
				'prob_class_1': float(avg_probs[1]),
				'n_windows': len(indices),  # Number of segments averaged
				'duration': avg_duration_per_segment
			}
			results.append(result)
			
			file_predictions.append(pred)
			file_labels.append(true_label)
		
		# Calculate and report metrics for this split
		if len(file_predictions) > 0:
			acc = accuracy_score(file_labels, file_predictions)
			f1 = f1_score(file_labels, file_predictions)
			auc = roc_auc_score(file_labels, [r['prob_class_1'] for r in results if r['split'] == split_name])
			print(f"{split_name.capitalize()} results: {len(file_predictions)} files - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
			
			if split_name == 'test':
				print(f"\nTest Classification Report:")
				print(classification_report(file_labels, file_predictions, target_names=['Control', 'Case']))
	
	# Save results
	results_df = pd.DataFrame(results)
	results_df.to_pickle(PREDICTIONS_OUTPUT_PATH.replace('.pkl', '.p'))
	
	print(f"\nFinal results saved to: {PREDICTIONS_OUTPUT_PATH.replace('.pkl', '.p')}")
	print(f"Total files: {len(results_df)} (expected: 1646 train + 412 test = 2058)")
	print(f"Actual counts: train={len(results_df[results_df['split']=='train'])}, test={len(results_df[results_df['split']=='test'])}")
	
	return results_df

def main():
	parser = argparse.ArgumentParser(description='Train classifier with variable-length sequences + multiprocessing + caching')
	
	parser.add_argument('--audio-dir', default=DEFAULT_AUDIO_DIR, help='Path to train audio directory')
	parser.add_argument('--test-audio-dir', default=DEFAULT_TEST_AUDIO_DIR, help='Path to test audio directory')
	parser.add_argument('--labels-path', default=DEFAULT_LABELS_PATH, help='Path to labels CSV')
	parser.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR, help='Cache directory')
	parser.add_argument('--force-recompute', action='store_true', help='Force recompute features')
	parser.add_argument('--freeze-encoder', action='store_true', help='Freeze encoder completely')
	parser.add_argument('--partial-unfreeze', action='store_true', help='Unfreeze only last transformer layer')
	
	args = parser.parse_args()
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	print(f"Using device: {device}")
	print("Loading MP3 files and extracting features...")
	
	feature_data, metadata, train_labels_df = load_audio_files_and_extract_features(
		args.audio_dir, args.labels_path, METADATA_PATH, ADDITIONAL_METADATA_PATH, 
		test_audio_dir=args.test_audio_dir,
		cache_dir=args.cache_dir, force_recompute=args.force_recompute
	)
	
	(train_features_df, val_features_df, test_features_df, train_uid_list, val_uid_list, test_uid_list) = feature_data
	
	metadata_cols = ['uid', 'segment_start_sec', 'row_index']
	feature_cols = [col for col in train_features_df.columns if col not in metadata_cols]
	
	print("Preparing data for training...")
	train_stacked = prepare_for_reshape(train_features_df, feature_cols, "train")
	val_stacked = prepare_for_reshape(val_features_df, feature_cols, "validation")
	test_stacked = prepare_for_reshape(test_features_df, feature_cols, "test")
	
	train_data_list, train_files = reshape_to_variable_length_list_parallel(train_stacked, "train")
	val_data_list, val_files = reshape_to_variable_length_list_parallel(val_stacked, "validation")
	
	if len(test_stacked) > 0:
		test_data_list, test_files = reshape_to_variable_length_list_parallel(test_stacked, "test")
	else:
		test_data_list = []
		test_files = []
	
	def map_to_y(file_id):
		uid = file_id.split("_")[0]
		if uid in train_labels_df['uid'].values:
			return int(train_labels_df[train_labels_df['uid'] == uid]['y'].iloc[0])
		return 0
	
	train_labels = np.array([map_to_y(x) for x in train_files])
	val_labels = np.array([map_to_y(x) for x in val_files])
	test_labels = np.array([map_to_y(x) for x in test_files]) if len(test_files) > 0 else np.array([])
	
	print("Normalizing data...")
	train_data_norm = normalize_variable_length_data(train_data_list)
	val_data_norm = normalize_variable_length_data(val_data_list)
	test_data_norm = normalize_variable_length_data(test_data_list) if len(test_data_list) > 0 else []
	
	train_dataset = VariableLengthSegmentDataset(train_data_norm, train_labels, segment_length=SEGMENT_LENGTH, 
												mode='train', segments_per_sample=SEGMENTS_PER_SAMPLE)
	val_dataset = VariableLengthSegmentDataset(val_data_norm, val_labels, segment_length=SEGMENT_LENGTH, 
											  mode='val', segments_per_sample=1)
	
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
	
	model = ClassificationModel(
		ENCODER_PATH, 
		num_classes=2, 
		freeze_encoder=args.freeze_encoder, 
		partial_unfreeze=args.partial_unfreeze
	)
	
	print("Training classifier...")
	trained_model = train_classifier(
		model, train_loader, val_loader, device, 
		epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, patience=PATIENCE
	)
	
	# Load best model and run final evaluation
	print("Loading best model for final evaluation...")
	trained_model.load_state_dict(torch.load(OUTPUT_MODEL_PATH))
	trained_model.eval()
	
	# Evaluate all datasets and save predictions
	all_predictions = evaluate_all_splits(
		trained_model, train_data_norm, val_data_norm, test_data_norm,
		train_labels, val_labels, test_labels,
		train_files, val_files, test_files, device
	)
	
	print("\nTraining and evaluation complete!")

if __name__ == "__main__":
	main()