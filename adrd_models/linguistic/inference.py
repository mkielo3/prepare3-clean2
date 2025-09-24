import torch
import torch.nn as nn
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path
from glob import glob
import pandas as pd
from tqdm import tqdm
import json

# Global paths
TRANSCRIPTS_PATH = 'linguistic/transcription/transcripts.p'
TRAINING_STATS_PATH = "../linguistic_models/models/results/training_stats.json"
MODELS_PATH = "../linguistic_models/models/models/*.pt"
OUTPUT_PATH = 'linguistic/predictions.p'

class DysfluencyClassifier(nn.Module):
	def __init__(self, embedding_dim=1024, hidden_dim=512):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(embedding_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(hidden_dim, hidden_dim // 2),
			nn.LayerNorm(hidden_dim // 2),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(hidden_dim // 2, 1)
		)
	
	def forward(self, x):
		return self.mlp(x).squeeze()

def get_epoch_49_val_loss(pathology_stat):
	"""Extract validation loss from epoch 49"""
	training_history = pathology_stat.get('training_history', [])
	epoch_49 = next((ep for ep in training_history if ep.get('epoch') == 49), None)
	return epoch_49.get('val_loss') if epoch_49 else None

def filter_models_by_performance(model_files, pathology_stats, max_val_loss=0.1):
	"""Filter models based on epoch 49 validation loss threshold"""
	filtered_models = []
	for model_path in model_files:
		checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
		pathology_name = checkpoint.get('pathology_name', Path(model_path).stem)
		epoch_49_val_loss = get_epoch_49_val_loss(pathology_stats.get(pathology_name, {}))
		if epoch_49_val_loss and epoch_49_val_loss < max_val_loss:
			filtered_models.append(model_path)
	return filtered_models

def load_training_stats():
	"""Load training statistics and create pathology filter"""
	with open(TRAINING_STATS_PATH, 'r') as f:
		stats = json.load(f)
	return {stat['pathology_name']: stat for stat in stats}

def run_batch_inference():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	# Load transcripts
	with open(TRANSCRIPTS_PATH, 'rb') as f:
		transcripts = pickle.load(f)
	
	# Pre-compute embeddings for all transcripts
	embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
	transcript_embeddings = {}
	
	print("Computing embeddings for all transcripts...")
	for filename, transcript_data in tqdm(transcripts.items()):
		text = transcript_data['text']
		embedding = embedding_model.encode([text.strip()], normalize_embeddings=True)[0]
		transcript_embeddings[filename] = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
	
	# Load training stats and filter models
	pathology_stats = load_training_stats()
	all_model_files = glob(MODELS_PATH)
	filtered_model_files = filter_models_by_performance(all_model_files, pathology_stats, max_val_loss=0.2)
	
	print(f"Total models: {len(all_model_files)}")
	print(f"Models meeting criteria (epoch 49 val_loss < 0.01): {len(filtered_model_files)}")
	
	results = []
	
	# Process only filtered models with pre-computed embeddings
	for model_path in tqdm(filtered_model_files, desc="Processing filtered models"):
		checkpoint = torch.load(model_path, map_location=device)
		model = DysfluencyClassifier().to(device)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.eval()
		
		pathology_name = checkpoint.get('pathology_name', Path(model_path).stem)
		
		# Run inference on all transcripts using pre-computed embeddings
		for filename, transcript_data in transcripts.items():
			embedding_tensor = transcript_embeddings[filename]
			
			with torch.no_grad():
				logit = model(embedding_tensor)
				probability = torch.sigmoid(logit).item()
			
			text = transcript_data['text']
			results.append({
				'filename': filename,
				'pathology': pathology_name,
				'probability': probability,
				'text': text[:100] + '...' if len(text) > 100 else text
			})
	
	# Save results
	df = pd.DataFrame(results)
	df.to_pickle(OUTPUT_PATH)

if __name__ == "__main__":
	run_batch_inference()