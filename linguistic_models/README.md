# Module 1: Synthetic Data Generation

Automatically generates synthetic training data by using ChatGPT API to modify base stories to exhibit specific linguistic markers.

## Input
- `input/language_markers.csv` - Symptom definitions with columns: category, test_name, description, control, adrd
- `input/synthetic_scenes.csv` - Base story scenarios to be modified

## Output  
- `output/{test_id}-{symptom_name}-{severity_id}.csv` - Generated synthetic examples for each symptom/severity combination
- Each output file contains: custom_id, content columns

## Usage
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run the synthetic data generation
python synthetic_data_runner.py

# Run tests
python run_tests.py
```

## Dependencies
- openai>=1.0.0
- pandas>=2.0.0
- tqdm>=4.65.0
- pytest>=7.0.0 (for testing)
- pytest-mock>=3.10.0 (for testing)

## Testing
- `tests/unit/test_synthetic_data_runner.py` - Unit tests with mocked OpenAI API
- `tests/unit/test_integration.py` - Integration tests for complete workflow
- `run_tests.py` - Convenience script to run all tests

## Used By
- Module 2 (linguistic_models) reads from `output/` to train binary classifiers
