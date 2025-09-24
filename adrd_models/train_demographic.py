#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# PATHS
LABELS_FILE = 'data/train_labels.csv'
TEST_LABELS_FILE = 'data/acoustic_test_labels.csv'
METADATA_FILE = 'data/metadata.csv'
ADDITIONAL_METADATA_FILE = 'data/acoustic_additional_metadata.csv'

OUTPUT_DIR = 'results_demographic'
MODEL_FILE = 'adrd_model_demographic.pkl'
TEST_PREDICTIONS_FILE = 'test_predictions_demographic.pkl'
TRAIN_PREDICTIONS_FILE = 'train_predictions_demographic.pkl'

# Model parameters
DEMOGRAPHIC_FEATURES = ['age', 'is_male']
N_CHAINS = 4
N_DRAWS = 2000
N_TUNE = 3000
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
	"""Load labels and metadata"""
	# Labels
	labels_df = pd.read_csv(LABELS_FILE)
	labels_df['y'] = (labels_df['diagnosis_mci'] + labels_df['diagnosis_adrd']).astype(int)

	test_labels_df = pd.read_csv(TEST_LABELS_FILE)
	test_labels_df['y'] = (test_labels_df['diagnosis_mci'] + test_labels_df['diagnosis_adrd']).astype(int)

	# Metadata
	metadata_df = pd.read_csv(METADATA_FILE).set_index('uid')
	additional_df = pd.read_csv(ADDITIONAL_METADATA_FILE).set_index('uid')
	metadata = pd.concat([metadata_df, additional_df], axis=1)

	return labels_df, test_labels_df, metadata


def prepare_data(labels_df, test_labels_df, metadata):
	"""Prepare data matrices with scaling and corpus pooling, using only demographic info."""
	print("\n=== DATA PREPARATION (Demographic Only) ===")

	all_labels_df = pd.concat([labels_df, test_labels_df], ignore_index=True)
	metadata['is_male'] = (metadata['gender'] == 'male').astype(int)

	available_uids = set(all_labels_df['uid']) & set(metadata.index)
	print(f"Total UIDs with labels and metadata: {len(available_uids)}")

	train_uids_from_meta = set(metadata[metadata['split'] == 'train'].index)
	test_uids_from_file = set(test_labels_df['uid'])

	train_uids = list(train_uids_from_meta.intersection(available_uids))
	test_uids = list(test_uids_from_file.intersection(available_uids))
	print(f"Final splits - Train UIDs: {len(train_uids)}, Test UIDs: {len(test_uids)}")

	all_used_uids = train_uids + test_uids
	labels_filtered = all_labels_df[all_labels_df['uid'].isin(all_used_uids)].set_index('uid')
	metadata_filtered = metadata.loc[all_used_uids]

	print("\n--- Pooling small corpuses ---")
	train_corpus_counts = metadata_filtered.loc[train_uids]['corpus'].value_counts()
	small_corpuses = train_corpus_counts[train_corpus_counts < 20].index
	large_corpuses = train_corpus_counts[train_corpus_counts >= 20].index
	print(f"Found {len(small_corpuses)} small corpuses to pool.")

	corpus_map = {corpus: i for i, corpus in enumerate(large_corpuses)}
	pooled_id = len(large_corpuses)
	for corpus in small_corpuses:
		corpus_map[corpus] = pooled_id

	final_corpus_map_for_diags = {v: k for k, v in corpus_map.items()}
	final_corpus_map_for_diags[pooled_id] = "pooled_corpus"

	def create_matrices(uid_list, split_name):
		print(f"\n--- Creating {split_name} matrices ---")
		if not uid_list:
			return (np.empty((0, 2)), np.empty(0), np.empty(0, dtype=np.int32))

		corpus_data = metadata_filtered.loc[uid_list]['corpus']
		corpus_assignments = corpus_data.map(lambda x: corpus_map.get(x, pooled_id)).values.astype(np.int32)
		demographics = metadata_filtered.loc[uid_list][DEMOGRAPHIC_FEATURES].astype(np.float64).values
		labels = labels_filtered.loc[uid_list]['y'].values.astype(np.float64)

		return (demographics, labels, corpus_assignments)

	X_demo_train, y_train, corpus_idx_train = create_matrices(train_uids, "TRAIN")
	X_demo_test, y_test, corpus_idx_test = create_matrices(test_uids, "TEST")

	print("\n--- Scaling demographic features ---")
	demo_scaler = StandardScaler()
	X_demo_train_scaled = demo_scaler.fit_transform(X_demo_train)
	X_demo_test_scaled = demo_scaler.transform(X_demo_test)

	train_data_tuple = (X_demo_train_scaled, y_train, corpus_idx_train)
	test_data_tuple = (X_demo_test_scaled, y_test, corpus_idx_test)

	return (train_data_tuple, final_corpus_map_for_diags), (test_data_tuple, {}), demo_scaler


def build_model(train_data):
	"""Build hierarchical Bayesian model using only demographic features."""
	X_demo, y_train, corpus_idx = train_data
	n_corpuses = len(np.unique(corpus_idx))

	with pm.Model() as model:
		# Data containers
		X_demo_data = pm.Data("X_demo_data", X_demo, mutable=True)
		corpus_idx_data = pm.Data("corpus_idx_data", corpus_idx, mutable=True)
		y_data = pm.Data("y_data", y_train.astype(float), mutable=True)

		# --- Non-Centered Hierarchical Priors for Regularization ---

		# 1. Corpus baseline effects (Random Intercepts)
		corpus_effect_sd = pm.HalfNormal("corpus_effect_sd", sigma=0.5)
		corpus_effects_raw = pm.Normal("corpus_effects_raw", mu=0, sigma=1, shape=n_corpuses)
		corpus_effects = pm.Deterministic("corpus_effects", corpus_effects_raw * corpus_effect_sd)

		# 2. Hierarchical Demographics (Random Slopes)
		demo_betas_mu = pm.Normal("demo_betas_mu", mu=0, sigma=1, shape=X_demo.shape[1])
		demo_betas_sd = pm.HalfNormal("demo_betas_sd", sigma=1.0, shape=X_demo.shape[1])
		demo_betas_raw = pm.Normal("demo_betas_raw", mu=0, sigma=1, shape=(n_corpuses, X_demo.shape[1]))
		demo_betas = pm.Deterministic("demo_betas", demo_betas_mu + demo_betas_raw * demo_betas_sd)

		# --- Combine components ---
		corpus_contribution = corpus_effects[corpus_idx_data]
		demographic_effect = pm.math.sum(X_demo_data * demo_betas[corpus_idx_data], axis=1)

		mu = corpus_contribution + demographic_effect
		p = pm.math.invlogit(mu)
		pm.Bernoulli("y_obs", p=p, observed=y_data)

	return model


def fit_model(model):
	"""Fit model and return trace"""
	with model:
		trace = pm.sample(
			draws=N_DRAWS,
			tune=N_TUNE,
			chains=N_CHAINS,
			random_seed=RANDOM_SEED,
			target_accept=0.98
		)
	return trace


def print_model_diagnostics(trace):
	"""Prints coefficient summaries."""
	print("\n" + "=" * 60)
	print(" " * 18 + "MODEL DIAGNOSTICS (Demographic)")
	print("=" * 60)

	print("\n--- Linear Component Effects (Posterior Summaries) ---")
	var_names = ["corpus_effects", "demo_betas_mu"]
	summary = az.summary(trace, var_names=var_names)
	print(summary)
	print("=" * 60 + "\n")


def save_train_predictions(model, trace, train_data):
	"""Generates and saves predictions for the training set."""
	print("Generating predictions for the training set...")
	_, y_train, _ = train_data

	with model:
		ppc = pm.sample_posterior_predictive(
			trace,
			var_names=["y_obs"],
			random_seed=RANDOM_SEED
		)

	y_pred_proba = ppc.posterior_predictive["y_obs"].mean(dim=("chain", "draw")).values
	y_pred_binary = (y_pred_proba > 0.5).astype(int)

	predictions = pd.DataFrame({
		'y_true': y_train,
		'y_pred_proba': y_pred_proba,
		'y_pred_binary': y_pred_binary
	})

	with open(os.path.join(OUTPUT_DIR, TRAIN_PREDICTIONS_FILE), 'wb') as f:
		pickle.dump(predictions, f)
	print(f"Train predictions saved to {os.path.join(OUTPUT_DIR, TRAIN_PREDICTIONS_FILE)}")


def evaluate_model(model, trace, test_data):
	"""Evaluate on test set"""
	X_demo_test, y_test, corpus_idx_test = test_data
	dummy_y = np.empty(X_demo_test.shape[0], dtype=np.float64)

	with model:
		pm.set_data({
			"X_demo_data": X_demo_test,
			"corpus_idx_data": corpus_idx_test,
			"y_data": dummy_y
		})
		ppc = pm.sample_posterior_predictive(
			trace,
			var_names=["y_obs"],
			random_seed=RANDOM_SEED
		)

	y_pred_proba = ppc.posterior_predictive["y_obs"].mean(dim=("chain", "draw")).values
	y_pred_binary = (y_pred_proba > 0.5).astype(int)

	accuracy = accuracy_score(y_test, y_pred_binary)
	auc = roc_auc_score(y_test, y_pred_proba)

	predictions = pd.DataFrame({'y_true': y_test, 'y_pred_proba': y_pred_proba, 'y_pred_binary': y_pred_binary})
	with open(os.path.join(OUTPUT_DIR, TEST_PREDICTIONS_FILE), 'wb') as f:
		pickle.dump(predictions, f)

	return accuracy, auc


def main():
	"""Main pipeline"""
	print("Loading data...")
	labels_df, test_labels_df, metadata = load_data()

	(train_data, train_corpus_map), (test_data, _), demo_scaler = prepare_data(labels_df, test_labels_df, metadata)

	print("Building model...")
	model = build_model(train_data)

	print("Fitting model...")
	trace = fit_model(model)

	print_model_diagnostics(trace)

	save_train_predictions(model, trace, train_data)

	_, y_test, _ = test_data
	print(f"\nTest data size: {len(y_test)}")

	if len(y_test) > 0:
		print("Evaluating on test set...")
		accuracy, auc = evaluate_model(model, trace, test_data)
	else:
		print("No test data - skipping evaluation")
		accuracy, auc = 0.0, 0.0

	model_data = {
		'trace': trace,
		'accuracy': accuracy,
		'auc': auc,
		'train_corpus_map': train_corpus_map,
		'demo_scaler': demo_scaler
	}

	with open(os.path.join(OUTPUT_DIR, MODEL_FILE), 'wb') as f:
		pickle.dump(model_data, f)

	print(f"\nAccuracy: {accuracy:.3f}, AUC: {auc:.3f}")
	print(f"Model saved to {os.path.join(OUTPUT_DIR, MODEL_FILE)}")
	print(f"Train predictions saved to {os.path.join(OUTPUT_DIR, TRAIN_PREDICTIONS_FILE)}")
	if len(y_test) > 0:
		print(f"Test predictions saved to {os.path.join(OUTPUT_DIR, TEST_PREDICTIONS_FILE)}")

	return model_data


if __name__ == "__main__":
	results = main()
