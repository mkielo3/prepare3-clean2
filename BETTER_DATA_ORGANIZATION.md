# Better Data Organization: Module-Specific Folders

## Proposed Structure

```
synthetic_data/                    # Module 1: Synthetic Data Generation
├── input/
│   └── symptoms.csv              # Input symptom definitions
├── output/
│   ├── all_synthetic_data.json   # Generated synthetic examples
│   ├── memory_loss_synthetic.json
│   └── generation_summary.json
└── README.md

linguistic_models/                 # Module 2: Binary Linguistic Models  
├── input/                        # → reads from synthetic_data/output/
├── models/
│   ├── memory_loss_model.joblib  # Trained classifiers
│   ├── word_finding_model.joblib
│   └── model_list.json
├── results/
│   └── training_stats.json      # Training performance metrics
└── README.md

linguistic_inference/              # Module 3: Linguistic Inference
├── input/                        # → reads from linguistic_models/models/
├── results/
│   ├── predictions_batch1.json   # Inference outputs
│   └── summary_report.json
└── README.md

egemaps_foundation/               # Module 4: eGemaps Foundation Training
├── input/
│   └── audio/                   # Healthy speech audio files
│       ├── mozilla_common_voice/
│       └── other_datasets/
├── models/
│   ├── egemaps_foundation_model.pth
│   ├── feature_scaler.joblib
│   └── model_info.json
├── results/
│   └── training_history.json
└── README.md

egemaps_inference/                # Module 5: eGemaps Inference
├── input/                        # → reads from egemaps_foundation/models/
├── results/
│   ├── embeddings_batch1.npy
│   └── predictions_batch1.json
└── README.md

aggregate_training/               # Module 6: Aggregate Model Training
├── input/                        # → reads from linguistic_inference/results/ + egemaps_inference/results/
├── models/
│   ├── final_model.joblib
│   ├── feature_pipeline.joblib
│   └── model_metadata.json
├── results/
│   └── training_stats.json
└── README.md

final_inference/                  # Module 7: Final Inference
├── input/                        # → reads from aggregate_training/models/
├── results/
│   ├── end_to_end_predictions.json
│   └── interpretable_results.json
└── README.md
```

## Benefits

### 1. Complete Self-Containment
Each module has everything it needs in one place:
- Input data
- Trained models  
- Results/outputs
- Documentation

### 2. Easy Module Development
Work on one module without touching others:
```bash
cd synthetic_data/
# Everything for Module 1 is here
```

### 3. Clear Dependencies  
Modules specify exactly where they read from:
```python
# Module 2 
linguistic_models.train(
    input_dir="../synthetic_data/output/",
    output_dir="./models/"
)

# Module 3
linguistic_inference.predict(
    models_dir="../linguistic_models/models/",  
    output_dir="./results/"
)
```

### 4. Independent Cleanup
Delete entire module folders without affecting others:
```bash
rm -rf linguistic_inference/results/  # Clear inference results
rm -rf synthetic_data/                # Start synthetic data from scratch
```

### 5. Parallel Development
Different people can work on different modules simultaneously without conflicts.

## Module Interface Pattern

Each module becomes a mini-project:
```python
# Standard interface for all modules
def run_module(input_dir, output_dir, config=None):
    # 1. Read from input_dir (or external source)
    # 2. Process data
    # 3. Save to output_dir
    # 4. Return summary
```

## File Naming Convention

```
{module_name}/
├── input/           # What this module reads
├── models/          # What this module trains (if applicable)
├── results/         # What this module outputs
└── README.md        # Module-specific documentation
```

## Cross-Module Dependencies

Modules can reference each other's outputs:
```python
# Module configuration
SYNTHETIC_DATA_OUTPUT = "../synthetic_data/output/"
LINGUISTIC_MODELS_DIR = "../linguistic_models/models/"
EGEMAPS_MODELS_DIR = "../egemaps_foundation/models/"
```

This approach makes each module a completely self-contained workspace!