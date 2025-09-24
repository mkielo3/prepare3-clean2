#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
LABELS_FILE = 'data/train_labels.csv'
TEST_LABELS_FILE = 'data/acoustic_test_labels.csv'
METADATA_FILE = 'data/metadata.csv'
ADDITIONAL_METADATA_FILE = 'data/acoustic_additional_metadata.csv'
LINGUISTIC_FEATURES_PATH = 'linguistic/predictions.p'
PARALINGUISTIC_FEATURES_PATH = 'paralinguistic/predictions.p'

OUTPUT_DIR = 'results'
MODEL_FILE = 'adrd_model.pkl'
TEST_PREDICTIONS_FILE = 'test_predictions.pkl'
TRAIN_PREDICTIONS_FILE = 'train_predictions.pkl'

# Model parameters
DEMOGRAPHIC_FEATURES = ['age', 'is_male']
BART_TREES = 100
N_CHAINS = 4
N_DRAWS = 4000
N_TUNE = 4000
RANDOM_SEED = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
	"""Load all data sources"""
	# Labels
	labels_df = pd.read_csv(LABELS_FILE)
	labels_df['y'] = (labels_df['diagnosis_mci'] + labels_df['diagnosis_adrd']).astype(int)

	test_labels_df = pd.read_csv(TEST_LABELS_FILE)
	test_labels_df['y'] = (test_labels_df['diagnosis_mci'] + test_labels_df['diagnosis_adrd']).astype(int)

	# Metadata
	metadata_df = pd.read_csv(METADATA_FILE).set_index('uid')
	additional_df = pd.read_csv(ADDITIONAL_METADATA_FILE).set_index('uid')
	metadata = pd.concat([metadata_df, additional_df], axis=1)

	# Features - load pickle files
	print(f"Loading linguistic features from {LINGUISTIC_FEATURES_PATH}")
	with open(LINGUISTIC_FEATURES_PATH, 'rb') as f:
		linguistic_data = pickle.load(f)

	linguistic_df = pd.DataFrame(linguistic_data).rename(columns={'filename': 'uid'}).set_index('uid')

	print(f"Loading paralinguistic features from {PARALINGUISTIC_FEATURES_PATH}")
	with open(PARALINGUISTIC_FEATURES_PATH, 'rb') as f:
		paralinguistic_data = pickle.load(f)
	print (pd.DataFrame(paralinguistic_data).head())
	paralinguistic_df = pd.DataFrame(paralinguistic_data).rename(columns={'filename': 'uid'}).set_index('uid')

	return labels_df, test_labels_df, metadata, linguistic_df, paralinguistic_df


def prepare_data(labels_df, test_labels_df, metadata, linguistic_df, paralinguistic_df):
	"""Prepare 4-component data matrices with scaling and corpus pooling."""
	print("\n=== DATA PREPARATION ===")

	all_labels_df = pd.concat([labels_df, test_labels_df], ignore_index=True)
	metadata['is_male'] = (metadata['gender'] == 'male').astype(int)

	available_uids = (set(all_labels_df['uid']) & set(metadata.index) &
	                  set(linguistic_df.index) & set(paralinguistic_df.index))
	print(f"Total UIDs with ALL data: {len(available_uids)}")

	train_uids_from_meta = set(metadata[metadata['split'] == 'train'].index)
	test_uids_from_file = set(test_labels_df['uid'])

	train_uids = list(train_uids_from_meta.intersection(available_uids))
	test_uids = list(test_uids_from_file.intersection(available_uids))
	print(f"Final splits - Train UIDs: {len(train_uids)}, Test UIDs: {len(test_uids)}")

	all_used_uids = train_uids + test_uids
	labels_filtered = all_labels_df[all_labels_df['uid'].isin(all_used_uids)].set_index('uid')
	metadata_filtered = metadata.loc[all_used_uids]
	linguistic_filtered = linguistic_df.loc[all_used_uids]
	paralinguistic_filtered = paralinguistic_df.loc[all_used_uids]

	# --- Corpus Pooling ---
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
			return (np.empty((0, 54)), np.empty((0, 2)), np.empty((0, 1)), np.empty(0), np.empty(0, dtype=np.int32))

		corpus_data = metadata_filtered.loc[uid_list]['corpus']
		corpus_assignments = corpus_data.map(lambda x: corpus_map.get(x, pooled_id)).values.astype(np.int32)

		demographics = metadata_filtered.loc[uid_list][DEMOGRAPHIC_FEATURES].astype(np.float64).values

		linguistic_subset = linguistic_filtered.loc[uid_list].reset_index()
		linguistic_subset['pred_idx'] = linguistic_subset.groupby('uid').cumcount()
		linguistic_pivoted = linguistic_subset.pivot(index='uid', columns='pred_idx', values='probability')
		linguistic_features = linguistic_pivoted.reindex(uid_list).fillna(0).values.astype(np.float64)

		paralinguistic_col = "prob_class_1" #"confidence" #"prob_class_1" #0 #'prob_class_1' #'confidence' #'prob_class_1' if 'prob_class_1' in paralinguistic_filtered.columns else 'prediction'
		paralinguistic_preds = paralinguistic_filtered.loc[uid_list][[paralinguistic_col]].astype(np.float64).values

		labels = labels_filtered.loc[uid_list]['y'].values.astype(np.float64)

		return (linguistic_features, demographics, paralinguistic_preds, labels, corpus_assignments)

	train_matrices = create_matrices(train_uids, "TRAIN")
	test_matrices = create_matrices(test_uids, "TEST")

	# --- Feature Scaling ---
	print("\n--- Scaling demographic and paralinguistic features ---")
	_, X_demo_train, X_para_train, _, _ = train_matrices
	_, X_demo_test, X_para_test, _, _ = test_matrices

	demo_scaler = StandardScaler()
	X_demo_train_scaled = demo_scaler.fit_transform(X_demo_train)
	X_demo_test_scaled = demo_scaler.transform(X_demo_test)

	para_scaler = StandardScaler()
	X_para_train_scaled = para_scaler.fit_transform(X_para_train)
	X_para_test_scaled = para_scaler.transform(X_para_test)

	train_data_tuple = (train_matrices[0], X_demo_train_scaled, X_para_train_scaled, train_matrices[3], train_matrices[4])
	test_data_tuple = (test_matrices[0], X_demo_test_scaled, X_para_test_scaled, test_matrices[3], test_matrices[4])

	return (train_data_tuple, final_corpus_map_for_diags), (test_data_tuple, {}), demo_scaler, para_scaler


def build_model(train_data):
	"""Build 4-component hierarchical Bayesian model with non-centered parameterization."""
	X_linguistic, X_demo, X_paralinguistic, y_train, corpus_idx = train_data
	n_corpuses = len(np.unique(corpus_idx))

	with pm.Model() as model:
		# Data containers
		X_linguistic_data = pm.Data("X_linguistic_data", X_linguistic, mutable=True)
		X_demo_data = pm.Data("X_demo_data", X_demo, mutable=True)
		X_paralinguistic_data = pm.Data("X_paralinguistic_data", X_paralinguistic, mutable=True)
		corpus_idx_data = pm.Data("corpus_idx_data", corpus_idx, mutable=True)
		y_data = pm.Data("y_data", y_train.astype(float), mutable=True)

		# --- Non-Centered Hierarchical Priors for Regularization ---

		# 1. Corpus baseline effects (Random Intercepts)
		corpus_effect_sd = pm.HalfNormal("corpus_effect_sd", sigma=0.5)
		corpus_effects_raw = pm.Normal('corpus_effects_raw', mu=0, sigma=1, shape=n_corpuses)
		corpus_effects = pm.Deterministic('corpus_effects', corpus_effects_raw * corpus_effect_sd)

		# 2. Hierarchical Demographics (Random Slopes)
		demo_betas_mu = pm.Normal("demo_betas_mu", mu=0, sigma=1, shape=X_demo.shape[1])
		demo_betas_sd = pm.HalfNormal("demo_betas_sd", sigma=1.0, shape=X_demo.shape[1])
		demo_betas_raw = pm.Normal("demo_betas_raw", mu=0, sigma=1, shape=(n_corpuses, X_demo.shape[1]))
		demo_betas = pm.Deterministic("demo_betas", demo_betas_mu + demo_betas_raw * demo_betas_sd)

		# 3. Hierarchical Paralinguistics (Random Slopes)
		para_betas_mu = pm.Normal("para_betas_mu", mu=0, sigma=1, shape=X_paralinguistic.shape[1])
		para_betas_sd = pm.HalfNormal("para_betas_sd", sigma=1.0, shape=X_paralinguistic.shape[1])
		para_betas_raw = pm.Normal("para_betas_raw", mu=0, sigma=1, shape=(n_corpuses, X_paralinguistic.shape[1]))
		para_betas = pm.Deterministic("para_betas", para_betas_mu + para_betas_raw * para_betas_sd)

		# 4. BART for Linguistic Features
		bart_linguistic = pmb.BART("bart_linguistic", X=X_linguistic_data, Y=y_data, m=BART_TREES)

		# --- Combine components ---
		corpus_contribution = corpus_effects[corpus_idx_data]
		demographic_effect = pm.math.sum(X_demo_data * demo_betas[corpus_idx_data], axis=1)
		paralinguistic_effect = pm.math.sum(X_paralinguistic_data * para_betas[corpus_idx_data], axis=1)

		mu = corpus_contribution + demographic_effect + paralinguistic_effect + bart_linguistic
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
			target_accept=0.98  # stricter convergence
		)
	return trace


def print_model_diagnostics(trace, corpus_map, train_data):
	"""Prints feature importances, coefficient summaries, and contribution analysis."""
	print("\n" + "=" * 60)
	print(" " * 22 + "MODEL DIAGNOSTICS")
	print("=" * 60)

	print("\n--- BART: Top 10 Linguistic Feature Importances ---")
	bart_vi = trace.sample_stats["variable_inclusion"].mean(("chain", "draw")).values
	feature_importances = pd.Series(bart_vi, name="Importance")
	feature_importances.index.name = "Feature Index"
	top_10_features = feature_importances.sort_values(ascending=False).head(10)
	print(top_10_features)
	print("(Note: Indices correspond to columns in the reshaped linguistic matrix)")

	print("\n--- Linear Component Effects (Posterior Summaries) ---")
	var_names = ["corpus_effects", "demo_betas_mu", "para_betas_mu"]
	summary = az.summary(trace, var_names=var_names)
	print(summary)

	print("\n--- Training Set Feature Contribution Analysis ---")
	# Manually calculate contributions from posterior samples
	X_linguistic, X_demo, X_paralinguistic, _, corpus_idx = train_data

	post = trace.posterior
	corpus_contrib = post["corpus_effects"].isel(corpus_effects_dim_0=corpus_idx).mean(dim=("chain", "draw")).values

	# Reshape for broadcasting
	demo_betas_samples = post["demo_betas"].isel(demo_betas_dim_0=corpus_idx)  # (chain, draw, sample, feature)
	demographic_contrib = (demo_betas_samples * X_demo).sum(dim="demo_betas_dim_1").mean(dim=("chain", "draw")).values

	para_betas_samples = post["para_betas"].isel(para_betas_dim_0=corpus_idx)
	paralinguistic_contrib = (para_betas_samples * X_paralinguistic).sum(dim="para_betas_dim_1").mean(
		dim=("chain", "draw")).values

	bart_contrib = post["bart_linguistic"].mean(dim=("chain", "draw")).values

	contributions_df = pd.DataFrame({
		'corpus_effect': corpus_contrib,
		'demographic_effect': demographic_contrib,
		'linguistic_effect (BART)': bart_contrib,
		'paralinguistic_effect': paralinguistic_contrib
	})

	print("\nMean absolute contribution (logit scale):")
	mean_abs_contributions = contributions_df.abs().mean().sort_values(ascending=False)
	print(mean_abs_contributions.round(3))

	print("\nDominant component frequency:")
	dominant_component = contributions_df.abs().idxmax(axis=1)
	dominance_counts = dominant_component.value_counts()
	dominance_pct = dominance_counts / len(contributions_df) * 100
	dominance_summary = pd.DataFrame({'count': dominance_counts, 'percentage': dominance_pct.round(1)})
	print(dominance_summary)

	print("=" * 60 + "\n")


def save_train_predictions(model, trace, train_data):
	"""Generates and saves predictions for the training set."""
	print("Generating predictions for the training set...")
	_, _, _, y_train, _ = train_data

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

	with open(f"{OUTPUT_DIR}/{TRAIN_PREDICTIONS_FILE}", 'wb') as f:
		pickle.dump(predictions, f)
	print(f"Train predictions saved to {OUTPUT_DIR}/{TRAIN_PREDICTIONS_FILE}")


def evaluate_model(model, trace, test_data):
	"""Evaluate on test set"""
	X_linguistic_test, X_demo_test, X_paralinguistic_test, y_test, corpus_idx_test = test_data

	dummy_y = np.empty(X_linguistic_test.shape[0], dtype=np.float64)

	with model:
		pm.set_data({
			"X_linguistic_data": X_linguistic_test,
			"X_demo_data": X_demo_test,
			"X_paralinguistic_data": X_paralinguistic_test,
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
	with open(f"{OUTPUT_DIR}/{TEST_PREDICTIONS_FILE}", 'wb') as f:
		pickle.dump(predictions, f)

	return accuracy, auc


def main():
	"""Main pipeline"""
	print("Loading data...")
	data = load_data()

	(train_data, train_corpus_map), (test_data, _), demo_scaler, para_scaler = prepare_data(*data)

	print("Building model...")
	model = build_model(train_data)

	print("Fitting model...")
	trace = fit_model(model)

	print_model_diagnostics(trace, train_corpus_map, train_data)

	save_train_predictions(model, trace, train_data)

	_, _, _, y_test, _ = test_data
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
		'demo_scaler': demo_scaler,
		'para_scaler': para_scaler,
		'n_features': train_data[0].shape[1]
	}

	with open(f"{OUTPUT_DIR}/{MODEL_FILE}", 'wb') as f:
		pickle.dump(model_data, f)

	print(f"\nAccuracy: {accuracy:.3f}, AUC: {auc:.3f}")
	print(f"Model saved to {OUTPUT_DIR}/{MODEL_FILE}")
	print(f"Train predictions saved to {OUTPUT_DIR}/{TRAIN_PREDICTIONS_FILE}")
	if len(y_test) > 0:
		print(f"Test predictions saved to {OUTPUT_DIR}/{TEST_PREDICTIONS_FILE}")

	return model_data


if __name__ == "__main__":
	results = main()
