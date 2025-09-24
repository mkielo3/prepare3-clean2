"""
Clean training script for acoustic contrastive learning WITH EXTENSIVE DEBUGGING.

This version calculates normalization statistics from the training data and saves
them to a file for later use during inference.

Usage:
	python train.py                    # Normal training mode
	python train.py --debug            # Debug mode (smaller data, fewer epochs)
	python train.py --no-wandb         # Disable wandb logging
	python train.py --debug --no-wandb # Debug mode without wandb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse
import warnings

try:
	import wandb
	WANDB_AVAILABLE = True
except ImportError:
	print("Warning: wandb not available. Install with: pip install wandb")
	WANDB_AVAILABLE = False

# This script now relies on your data_loader.py file.
# Ensure it is in the same directory and provides the following functions:
# - process_acoustic_data(path, max_length) -> (train_data, val_data, train_files, val_files)
# - create_masks(data) -> masks
from data_loader import process_acoustic_data, create_masks


def check_tensor_health(tensor, name, abort_on_nan=False):
	"""Check tensor for NaN/Inf values and print statistics"""
	is_healthy = True
	if not isinstance(tensor, torch.Tensor):
		print(f" {name}: Not a tensor, skipping health check.")
		return True

	nan_count = torch.isnan(tensor).sum().item()
	inf_count = torch.isinf(tensor).sum().item()
	
	if nan_count > 0 or inf_count > 0:
		is_healthy = False
		print(f"\n WARNING: {name} contains {nan_count} NaN and {inf_count} Inf values!")
		print(f"    Shape: {tensor.shape}")
		print(f"    Total elements: {tensor.numel()}")
		
		if abort_on_nan:
			raise ValueError(f"Found NaN/Inf in {name}. Aborting.")
	
	# Print stats
	if tensor.numel() > 0:
		finite_values = tensor[torch.isfinite(tensor)]
		if len(finite_values) > 0:
			stats_str = (f"min={finite_values.min().item():.4f}, max={finite_values.max().item():.4f}, "
						 f"mean={finite_values.mean().item():.4f}, std={finite_values.std().item():.4f}")
			if is_healthy:
				print(f" {name}: {stats_str}")
			else:
				print(f"    Finite stats: {stats_str}")
		else:
			print(f" {name}: No finite values!")
	
	return is_healthy

# Import shared model architecture  
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from shared.model import UpgradedAcousticEncoder

class ContrastiveLoss(nn.Module):
	def __init__(self, temperature=0.1):
		super().__init__()
		self.temperature = temperature
		
	def forward(self, z1, z2, debug=False):
		if debug:
			print(f"\n--- Contrastive Loss Debug ---")
			check_tensor_health(z1, "z1 embeddings")
			check_tensor_health(z2, "z2 embeddings")
		
		batch_size = z1.size(0)
		z = torch.cat([z1, z2], dim=0)
		
		sim_matrix = torch.mm(z, z.t()) / self.temperature
		if debug: check_tensor_health(sim_matrix, "Similarity matrix", abort_on_nan=True)
		
		labels = torch.cat([torch.arange(batch_size, 2*batch_size), torch.arange(batch_size)]).to(z.device)
		mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
		sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
		
		loss = F.cross_entropy(sim_matrix, labels)
		if debug: check_tensor_health(loss, "Final loss", abort_on_nan=True)
		
		return loss

class AcousticDataset(Dataset):
	def __init__(self, data, masks, file_paths, min_length=10, max_length=45, augment=True):
		
		print(f"\n=== Initializing Dataset ===")
		print(f"Original dataset size: {len(data)} samples")

		# --- Filtering logic from original script ---
		filtered_data = []
		filtered_masks = []
		filtered_files = []

		for i in range(len(data)):
			valid_length = masks[i].sum()
			
			if valid_length >= min_length:
				filtered_data.append(data[i])
				filtered_masks.append(masks[i])
				filtered_files.append(file_paths[i])
		
		self.data = np.array(filtered_data)
		self.masks = np.array(filtered_masks)
		self.file_paths = filtered_files
		self.augment = augment
		
		print(f"Filtered dataset size: {len(self.data)} sequences (min_length={min_length})")
		
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		sample = self.data[idx].copy()
		mask = self.masks[idx].copy()
		
		sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
		
		if self.augment:
			sample1, mask1 = self._augment(sample, mask)
			sample2, mask2 = self._augment(sample, mask)
			
			if np.isnan(sample1).any() or np.isnan(sample2).any():
				print(f"WARNING: NaN detected after augmentation for index {idx}. Replacing.")
				sample1 = np.nan_to_num(sample1, nan=0.0)
				sample2 = np.nan_to_num(sample2, nan=0.0)

			return (torch.FloatTensor(sample1), torch.FloatTensor(sample2), 
					torch.BoolTensor(mask1), torch.BoolTensor(mask2),
					idx)
		else:
			return torch.FloatTensor(sample), torch.BoolTensor(mask), idx
	
	def _augment(self, seq, mask):
		seq_aug = seq.copy()
		mask_aug = mask.copy()
		valid_indices = np.where(mask_aug)[0]
		
		if len(valid_indices) == 0:
			return seq_aug, mask_aug

		noise_std = 0.08
		noise = np.random.normal(0, noise_std, (88, len(valid_indices)))
		seq_aug[:, valid_indices] += noise
		
		if len(valid_indices) > 6:
			n_segments = np.random.randint(1, 4)
			for _ in range(n_segments):
				segment_len = np.random.randint(2, 6)
				if len(valid_indices) >= segment_len:
					start_idx = np.random.choice(len(valid_indices) - segment_len + 1)
					mask_indices = valid_indices[start_idx:start_idx + segment_len]
					seq_aug[:, mask_indices] = 0
					mask_aug[mask_indices] = False
		
		rand_roll = np.random.random()
		if rand_roll < 0.4: n_channels_mask = np.random.randint(1, 15)
		elif rand_roll < 0.6: n_channels_mask = np.random.randint(15, 30)
		elif rand_roll < 0.75: n_channels_mask = np.random.randint(30, 50)
		else: n_channels_mask = 0
		
		if n_channels_mask > 0:
			channel_indices = np.random.choice(88, n_channels_mask, replace=False)
			seq_aug[channel_indices] = 0
		
		return seq_aug, mask_aug

def calculate_and_apply_normalization(train_data, val_data):
	"""
	Calculates normalization statistics from the training data, applies them
	to both training and validation sets, and returns the stats for saving.
	"""
	print("\n=== Calculating and Applying Normalization Stats from Training Data ===")
	
	means = []
	stds = []
	
	for channel in range(train_data.shape[1]): # 88 channels
		channel_values = train_data[:, channel, :]
		valid_values = channel_values[~np.isnan(channel_values)]
		
		if len(valid_values) > 0:
			mean = valid_values.mean()
			std = valid_values.std()
			means.append(mean)
			# Use a small epsilon for std dev if it's zero or too small
			stds.append(std if std > 1e-6 else 1e-6)
		else:
			# Fallback for empty channels
			means.append(0)
			stds.append(1e-6)
			
	# Convert to 1D numpy arrays for saving
	means_1d = np.array(means)
	stds_1d = np.array(stds)
	
	# Reshape for broadcasting during normalization
	means_reshaped = means_1d[:, np.newaxis]
	stds_reshaped = stds_1d[:, np.newaxis]
	
	print("Applying normalization to training data...")
	train_norm = (train_data - means_reshaped) / stds_reshaped
	train_norm = np.nan_to_num(train_norm, nan=0.0, posinf=0.0, neginf=0.0)

	print("Applying normalization to validation data...")
	val_norm = (val_data - means_reshaped) / stds_reshaped
	val_norm = np.nan_to_num(val_norm, nan=0.0, posinf=0.0, neginf=0.0)
	
	print("Normalization complete.")
	
	# Create a dictionary to return the original (1D) stats
	stats_to_save = {'means': means_1d, 'stds': stds_1d}
	
	return train_norm, val_norm, stats_to_save


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, train_files, clip_grad=True):
	model.train()
	total_loss = 0
	
	pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
	for batch_idx, batch in enumerate(pbar):
		seq1, seq2, mask1, mask2, indices = batch
		seq1, seq2, mask1, mask2 = seq1.to(device), seq2.to(device), mask1.to(device), mask2.to(device)
		
		debug_this_batch = (epoch == 0 and batch_idx == 0)
		
		optimizer.zero_grad(set_to_none=True)
		
		z1 = model(seq1, mask1, debug=debug_this_batch)
		z2 = model(seq2, mask2, debug=debug_this_batch)
		
		loss = criterion(z1, z2, debug=debug_this_batch)
		
		if torch.isnan(loss) or torch.isinf(loss):
			print(f"\n\n{'='*20} 🚨 FATAL: NaN/Inf LOSS DETECTED 🚨 {'='*20}")
			print(f"Epoch: {epoch}, Batch: {batch_idx}")
			problem_indices = indices.cpu().numpy()
			print(f"Problematic original data indices: {problem_indices}")
			try:
				problem_files = [train_files[i] for i in problem_indices]
				print(f"Corresponding file paths: {problem_files}")
			except IndexError:
				problem_files = ["Could not retrieve file paths for indices."]

			print("\n--- Inspecting raw batch inputs that led to this error ---")
			check_tensor_health(seq1, "seq1 (problematic input)")
			check_tensor_health(seq2, "seq2 (problematic input)")
			print("\n--- Inspecting model outputs that led to this error ---")
			check_tensor_health(z1, "z1 (model output)")
			check_tensor_health(z2, "z2 (model output)")
			
			error_message = (
				f"\n\nNaN/Inf loss detected. Aborting training.\n"
				f"Files: {problem_files}"
			)
			raise ValueError(error_message)
		
		loss.backward()
		
		if clip_grad:
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		
		optimizer.step()
		
		total_loss += loss.item()
		pbar.set_postfix(loss=loss.item())

		if WANDB_AVAILABLE and wandb.run is not None and batch_idx % 100 == 0:
			wandb.log({
				"batch_loss": loss.item(),
				"batch_idx": batch_idx,
				"running_avg_loss": total_loss / (batch_idx + 1)
			})
	
	avg_loss = total_loss / len(dataloader)
	return avg_loss

def validate(model, dataloader, criterion, device):
	model.eval()
	total_loss = 0
	valid_batches = 0
	
	with torch.no_grad():
		for batch in dataloader:
			seq1, seq2, mask1, mask2, _ = batch
			seq1, seq2, mask1, mask2 = seq1.to(device), seq2.to(device), mask1.to(device), mask2.to(device)
			
			z1 = model(seq1, mask1)
			z2 = model(seq2, mask2)
			loss = criterion(z1, z2)
			
			if not (torch.isnan(loss) or torch.isinf(loss)):
				total_loss += loss.item()
				valid_batches += 1
	
	return total_loss / max(1, valid_batches)

def get_config(debug_mode=False):
	if debug_mode:
		config = {
			'features_path': '../paralinguistic_data/data/commonvoice_50plus_features_debug.parquet',
			'batch_size': 4, 'learning_rate': 1e-5, 'weight_decay': 1e-3,
			'epochs': 3, 'patience': 5, 'scheduler_patience': 5,
			'embedding_dim': 512, 'nhead': 16, 'num_encoder_layers': 8,
			'dim_feedforward': 2048, 'max_length': 45,
			'output_dir': 'checkpoints_debug', 'project_name': 'acoustic-contrastive-debug',
			'run_name': 'debug-run', 'save_every': 1
		}
	else:
		normal_path = '../paralinguistic_data/data/commonvoice_50plus_features.parquet'
		config = {
			'features_path': normal_path, 'batch_size': 32, 'learning_rate': 5e-5,
			'weight_decay': 1e-2, 'epochs': 1000, 'patience': 150,
			'scheduler_patience': 50, 'embedding_dim': 512, 'nhead': 16,
			'num_encoder_layers': 8, 'dim_feedforward': 2048, 'max_length': 45,
			'output_dir': 'checkpoints', 'project_name': 'acoustic-grokking',
			'run_name': None, 'save_every': 25
		}
	return config

def main(debug_mode=False, use_wandb=True):
	warnings.filterwarnings("ignore", message=".*does not have many workers.*")
	config = get_config(debug_mode)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}, Debug mode: {debug_mode}")
	
	if use_wandb and WANDB_AVAILABLE:
		wandb.init(project=config['project_name'], name=config['run_name'], config=config)
	
	os.makedirs(config['output_dir'], exist_ok=True)
	
	print("LOADING AND PROCESSING DATA")
	train_data, val_data, train_files, val_files = process_acoustic_data(
		config['features_path'], max_length=config['max_length']
	)
	
	if debug_mode:
		train_data, val_data = train_data[:20], val_data[:10]
		train_files, val_files = train_files[:20], val_files[:10]
	
	# Calculate normalization stats and apply them
	train_data, val_data, norm_stats = calculate_and_apply_normalization(train_data, val_data)
	
	# Save the calculated stats to a file for later use in inference
	stats_path = os.path.join(config['output_dir'], 'normalization_stats.npz')
	np.savez(stats_path, means=norm_stats['means'], stds=norm_stats['stds'])
	print(f"Normalization stats saved to {stats_path}")
	
	train_masks = create_masks(train_data)
	val_masks = create_masks(val_data)
	
	train_dataset = AcousticDataset(train_data, train_masks, train_files, augment=True)
	val_dataset = AcousticDataset(val_data, val_masks, val_files, augment=False)
	
	num_workers = 4 if not debug_mode else 0
	train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=num_workers)
	val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers)
	
	model = UpgradedAcousticEncoder(
		embedding_dim=config['embedding_dim'], nhead=config['nhead'],
		num_encoder_layers=config['num_encoder_layers'], dim_feedforward=config['dim_feedforward']
	).to(device)
	
	criterion = ContrastiveLoss(temperature=0.1)
	optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config['scheduler_patience'], factor=0.7)
	
	best_val_loss = float('inf')
	patience_counter = 0
	
	for epoch in range(config['epochs']):
		print(f"\n{'='*60}\nEPOCH {epoch + 1}/{config['epochs']}\n{'='*60}")
		
		train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, train_dataset.file_paths)
		val_loss = validate(model, val_loader, criterion, device)
		scheduler.step(val_loss)
		
		current_lr = optimizer.param_groups[0]['lr']
		print(f"\nEpoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.1e}")
		
		if use_wandb and WANDB_AVAILABLE:
			wandb.log({
				"epoch": epoch + 1,
				"train_loss": train_loss,
				"val_loss": val_loss,
				"learning_rate": current_lr,
				"train_val_gap": train_loss - val_loss if val_loss > train_loss else 0,
				"patience_counter": patience_counter
			})
		
		if val_loss < best_val_loss and not np.isnan(val_loss):
			best_val_loss = val_loss
			patience_counter = 0
			torch.save(model.state_dict(), os.path.join(config['output_dir'], 'best_model.pth'))
			print(f"New best model saved! Val Loss: {val_loss:.4f}")
			if use_wandb and WANDB_AVAILABLE:
				wandb.log({"best_val_loss": best_val_loss})
		else:
			patience_counter += 1
		
		if patience_counter >= config['patience']:
			print(f"Early stopping at epoch {epoch+1}")
			break
			
	if use_wandb and WANDB_AVAILABLE:
		wandb.finish()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train acoustic contrastive encoder')
	parser.add_argument('--debug', action='store_true', help='Run in debug mode')
	parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
	args = parser.parse_args()
	main(debug_mode=args.debug, use_wandb=not args.no_wandb)
