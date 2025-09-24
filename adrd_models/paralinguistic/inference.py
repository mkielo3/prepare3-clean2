"""
Inference script for acoustic contrastive model with caching support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import librosa
import opensmile
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message="Segment too short, filling with NaN.")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'paralinguistic_models'))
from shared.model import UpgradedAcousticEncoder

# Configuration
TRAIN_AUDIO_DIR = 'data/train_audios'
TEST_AUDIO_DIR = 'data/test_audios'
ENCODER_PATH = '../paralinguistic_models/paralinguistic_models/checkpoints/best_model.pth'
CLASSIFIER_PATH = 'paralinguistic/best_classifier.pth'
OUTPUT_DIR = 'paralinguistic'

SEGMENT_DURATION = 0.2
SEGMENT_LENGTH = 45
WINDOW_STRIDE = 20

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

class ClassificationModel(nn.Module):
	def __init__(self, encoder_path, num_classes=2):
		super().__init__()
		
		self.encoder = UpgradedAcousticEncoder(
			n_channels=88, 
			embedding_dim=512,
			nhead=16,
			num_encoder_layers=8,
			dim_feedforward=2048
		)
		
		self.encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
		
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
		embeddings = self.encoder(x, mask)
		logits = self.classifier(embeddings)
		return logits

def extract_features_from_audio_file(audio_path, segment_duration=SEGMENT_DURATION):
	y, sr = librosa.load(audio_path, sr=None)
	total_duration = len(y) / sr
	
	smile = opensmile.Smile(
		feature_set=opensmile.FeatureSet.eGeMAPSv02,
		feature_level=opensmile.FeatureLevel.Functionals,
	)
	
	segment_starts = np.arange(0, total_duration, segment_duration)
	segment_ends = np.minimum(segment_starts + segment_duration, total_duration)
	
	all_features = []
	uid = Path(audio_path).stem
	
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
	
	features_df = pd.DataFrame(all_features)
	
	metadata = {
		'audio_path': audio_path,
		'duration': total_duration,
		'sample_rate': sr,
		'segment_duration': segment_duration,
		'uid': uid
	}
	
	return features_df, metadata

def prepare_for_reshape(df, feature_cols):
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

def reshape_to_variable_length_array(stacked_series):
	if len(stacked_series) == 0:
		return None
	
	df = stacked_series.reset_index()
	features = sorted(df['feature'].unique())
	
	uid_data = df.groupby('uid').first().reset_index()
	uid = uid_data.iloc[0]['uid']
	uid_df = df[df['uid'] == uid]
	timestamps = sorted(uid_df['segment_start_sec'].unique())
	
	uid_matrix = np.full((len(features), len(timestamps)), np.nan)
	
	for _, row in uid_df.iterrows():
		feature_idx = features.index(row['feature'])
		time_idx = timestamps.index(row['segment_start_sec'])
		uid_matrix[feature_idx, time_idx] = row['value']
	
	return uid_matrix

def normalize_variable_length_data(data_array):
	normalized_array = data_array.copy()
	
	for channel in range(min(88, data_array.shape[0])):
		if channel in CHANNEL_STATS:
			mean, std = CHANNEL_STATS[channel]
			if std > 0:
				normalized_array[channel, :] = (data_array[channel, :] - mean) / (std + 1e-8)
	
	normalized_array = np.nan_to_num(normalized_array, nan=0.0, posinf=0.0, neginf=0.0)
	return normalized_array

def prepare_single_segment(data_segment, max_length=SEGMENT_LENGTH):
	n_features, n_timesteps = data_segment.shape
	
	mask = np.ones(max_length, dtype=bool)
	
	if n_timesteps < max_length:
		mask[n_timesteps:] = False
		padding = np.zeros((n_features, max_length - n_timesteps))
		data_segment = np.concatenate([data_segment, padding], axis=1)
	elif n_timesteps > max_length:
		data_segment = data_segment[:, :max_length]
	
	feature_tensor = torch.FloatTensor(data_segment).unsqueeze(0)
	mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
	
	return feature_tensor, mask_tensor

def generate_sliding_windows(data_array, window_length=SEGMENT_LENGTH, stride=20):
	n_features, n_timesteps = data_array.shape
	valid_mask = ~np.isnan(data_array).all(axis=0)
	valid_indices = np.where(valid_mask)[0]
	
	if len(valid_indices) < window_length:
		return [data_array]
	
	windows = []
	for start_idx in range(0, len(valid_indices) - window_length + 1, stride):
		selected_indices = valid_indices[start_idx:start_idx + window_length]
		window = data_array[:, selected_indices]
		windows.append(window)
	
	return windows

def load_model(device='cpu'):
	model = ClassificationModel(
		encoder_path=ENCODER_PATH,
		num_classes=2,
	)
	
	state_dict = torch.load(CLASSIFIER_PATH, map_location=device)
	model.load_state_dict(state_dict)
	
	model.to(device)
	model.eval()
	
	return model

def run_inference(audio_path, device=None, window_length=SEGMENT_LENGTH, stride=WINDOW_STRIDE):
	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	features_df, metadata = extract_features_from_audio_file(audio_path)
	
	metadata_cols = ['uid', 'segment_start_sec', 'row_index']
	feature_cols = [col for col in features_df.columns if col not in metadata_cols]
	
	stacked_features = prepare_for_reshape(features_df, feature_cols)
	data_array = reshape_to_variable_length_array(stacked_features)
	
	normalized_array = normalize_variable_length_data(data_array)
	
	model = load_model(device)
	
	windows = generate_sliding_windows(normalized_array, window_length, stride)
	
	all_probabilities = []
	with torch.no_grad():
		for window in windows:
			feature_tensor, mask_tensor = prepare_single_segment(window, window_length)
			feature_tensor = feature_tensor.to(device)
			mask_tensor = mask_tensor.to(device)
			
			logits = model(feature_tensor, mask_tensor)
			probs = F.softmax(logits, dim=1)
			all_probabilities.append(probs.cpu().numpy())
	
	averaged_probabilities = np.mean(all_probabilities, axis=0)
	prediction = np.argmax(averaged_probabilities)
	confidence = np.max(averaged_probabilities)
	
	return averaged_probabilities, prediction, confidence, features_df, metadata

def get_audio_files(audio_dir):
	audio_dir = Path(audio_dir)
	if not audio_dir.exists():
		return []
	
	audio_extensions = ['.mp3', '.wav', '.flac', '.m4a']
	audio_files = []
	
	for ext in audio_extensions:
		audio_files.extend(audio_dir.glob(f"*{ext}"))
	
	return sorted(audio_files)

def main():
	output_dir = Path(OUTPUT_DIR)
	output_dir.mkdir(parents=True, exist_ok=True)
	
	train_files = get_audio_files(TRAIN_AUDIO_DIR)
	test_files = get_audio_files(TEST_AUDIO_DIR)
	
	all_files = [(f, 'train') for f in train_files] + [(f, 'test') for f in test_files]
	
	print(f"Found {len(train_files)} train files, {len(test_files)} test files")
	
	results = []
	
	for audio_path, split in all_files:
		print(f"\nProcessing {split}/{audio_path.name}")
		
		probs, pred, conf, features, metadata = run_inference(
			audio_path=str(audio_path)
		)
		
		result = {
			'uid': audio_path.stem,
			'split': split,
			'prediction': int(pred),
			'confidence': float(conf),
			'prob_class_0': float(probs[0][0]),
			'prob_class_1': float(probs[0][1]),
			'n_windows': len(features) // SEGMENT_LENGTH,
			'duration': metadata['duration']
		}
		
		results.append(result)
		
		print(f"Prediction: {pred}, Confidence: {conf:.3f}, Windows: {result['n_windows']}")
	
	results_df = pd.DataFrame(results)
	results_df.to_pickle(output_dir / "predictions.p")
	print(f"\nSaved predictions to: {output_dir / 'predictions.p'}")

if __name__ == "__main__":
	main()