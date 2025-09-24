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
from sklearn.metrics import accuracy_score, roc_auc_score
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

OUTPUT_DIR = 'results_linguistic'
MODEL_FILE = 'adrd_model_linguistic.pkl'
TEST_PREDICTIONS_FILE = 'test_predictions_linguistic.pkl'
TRAIN_PREDICTIONS_FILE = 'train_predictions_linguistic.pkl'

# Model parameters
BART_TREES = 100
N_CHAINS = 4
N_DRAWS = 2000
N_TUNE = 3000
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load labels, metadata, and linguistic features"""
    # Labels
    labels_df = pd.read_csv(LABELS_FILE)
    labels_df['y'] = (labels_df['diagnosis_mci'] + labels_df['diagnosis_adrd']).astype(int)

    test_labels_df = pd.read_csv(TEST_LABELS_FILE)
    test_labels_df['y'] = (test_labels_df['diagnosis_mci'] + test_labels_df['diagnosis_adrd']).astype(int)

    # Metadata
    metadata_df = pd.read_csv(METADATA_FILE).set_index('uid')
    additional_df = pd.read_csv(ADDITIONAL_METADATA_FILE).set_index('uid')
    metadata = pd.concat([metadata_df, additional_df], axis=1)

    # Linguistic Features
    print(f"Loading linguistic features from {LINGUISTIC_FEATURES_PATH}")
    with open(LINGUISTIC_FEATURES_PATH, 'rb') as f:
        linguistic_data = pickle.load(f)
    linguistic_df = pd.DataFrame(linguistic_data).rename(columns={'filename': 'uid'}).set_index('uid')

    return labels_df, test_labels_df, metadata, linguistic_df


def prepare_data(labels_df, test_labels_df, metadata, linguistic_df):
    """Prepare data matrices using only linguistic features."""
    print("\n=== DATA PREPARATION (Linguistic Only) ===")

    all_labels_df = pd.concat([labels_df, test_labels_df], ignore_index=True)

    available_uids = (set(all_labels_df['uid']) & set(metadata.index) &
                      set(linguistic_df.index))
    print(f"Total UIDs with labels, metadata, and linguistic data: {len(available_uids)}")

    train_uids_from_meta = set(metadata[metadata['split'] == 'train'].index)
    test_uids_from_file = set(test_labels_df['uid'])

    train_uids = list(train_uids_from_meta.intersection(available_uids))
    test_uids = list(test_uids_from_file.intersection(available_uids))
    print(f"Final splits - Train UIDs: {len(train_uids)}, Test UIDs: {len(test_uids)}")

    all_used_uids = train_uids + test_uids
    labels_filtered = all_labels_df[all_labels_df['uid'].isin(all_used_uids)].set_index('uid')
    metadata_filtered = metadata.loc[all_used_uids]
    linguistic_filtered = linguistic_df.loc[all_used_uids]

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
            return (np.empty((0, 54)), np.empty(0), np.empty(0, dtype=np.int32))

        corpus_data = metadata_filtered.loc[uid_list]['corpus']
        corpus_assignments = corpus_data.map(lambda x: corpus_map.get(x, pooled_id)).values.astype(np.int32)

        linguistic_subset = linguistic_filtered.loc[uid_list].reset_index()
        linguistic_subset['pred_idx'] = linguistic_subset.groupby('uid').cumcount()
        linguistic_pivoted = linguistic_subset.pivot(index='uid', columns='pred_idx', values='probability')
        linguistic_features = linguistic_pivoted.reindex(uid_list).fillna(0).values.astype(np.float64)

        labels = labels_filtered.loc[uid_list]['y'].values.astype(np.float64)

        return (linguistic_features, labels, corpus_assignments)

    X_ling_train, y_train, corpus_idx_train = create_matrices(train_uids, "TRAIN")
    X_ling_test, y_test, corpus_idx_test = create_matrices(test_uids, "TEST")

    train_data_tuple = (X_ling_train, y_train, corpus_idx_train)
    test_data_tuple = (X_ling_test, y_test, corpus_idx_test)

    return (train_data_tuple, final_corpus_map_for_diags), (test_data_tuple, {})


def build_model(train_data):
    """Build hierarchical Bayesian model with BART for linguistic features."""
    X_linguistic, y_train, corpus_idx = train_data
    n_corpuses = len(np.unique(corpus_idx))

    with pm.Model() as model:
        # Data containers
        X_linguistic_data = pm.Data("X_linguistic_data", X_linguistic, mutable=True)
        corpus_idx_data = pm.Data("corpus_idx_data", corpus_idx, mutable=True)
        y_data = pm.Data("y_data", y_train.astype(float), mutable=True)

        # Corpus baseline effects (Random Intercepts)
        corpus_effect_sd = pm.HalfNormal("corpus_effect_sd", sigma=0.5)
        corpus_effects_raw = pm.Normal("corpus_effects_raw", mu=0, sigma=1, shape=n_corpuses)
        corpus_effects = pm.Deterministic("corpus_effects", corpus_effects_raw * corpus_effect_sd)

        # BART for Linguistic Features
        bart_linguistic = pmb.BART("bart_linguistic", X=X_linguistic_data, Y=y_data, m=BART_TREES)

        # --- Combine components ---
        corpus_contribution = corpus_effects[corpus_idx_data]
        mu = corpus_contribution + bart_linguistic
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
    """Prints BART feature importances and coefficient summaries."""
    print("\n" + "=" * 60)
    print(" " * 18 + "MODEL DIAGNOSTICS (Linguistic)")
    print("=" * 60)

    print("\n--- BART: Top 10 Linguistic Feature Importances ---")
    bart_vi = trace.sample_stats["variable_inclusion"].mean(("chain", "draw")).values
    feature_importances = pd.Series(bart_vi, name="Importance")
    feature_importances.index.name = "Feature Index"
    top_10_features = feature_importances.sort_values(ascending=False).head(10)
    print(top_10_features)

    print("\n--- Corpus Effects (Posterior Summary) ---")
    summary = az.summary(trace, var_names=["corpus_effects"])
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
    X_linguistic_test, y_test, corpus_idx_test = test_data
    dummy_y = np.empty(X_linguistic_test.shape[0], dtype=np.float64)

    with model:
        pm.set_data({
            "X_linguistic_data": X_linguistic_test,
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
    labels_df, test_labels_df, metadata, linguistic_df = load_data()

    (train_data, train_corpus_map), (test_data, _) = prepare_data(labels_df, test_labels_df, metadata, linguistic_df)

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
        'n_features': train_data[0].shape[1]
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
