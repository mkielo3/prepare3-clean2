"""
Shared test fixtures for synthetic data generation tests.
"""

import pytest
import os
import tempfile
import shutil
import pandas as pd
from unittest.mock import Mock


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client with configurable responses."""
    client = Mock()
    
    # Default successful response
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "Generated synthetic text"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    client.chat.completions.create.return_value = mock_response
    
    return client


@pytest.fixture
def sample_symptoms_data():
    """Sample language markers data for testing."""
    return {
        'category': [
            'Lexical Signature',
            'Syntactic Features', 
            'Self-Correction'
        ],
        'test_name': [
            'Generic Word Overuse',
            'Sentence Length Distribution',
            'False Starts'
        ],
        'description': [
            'Use of generic terms like "thing", "stuff"',
            'Variation in sentence complexity and length',
            'Frequency of sentence restarts'
        ],
        'control': [
            'Uses specific, precise terminology',
            'Natural mix of short, medium, and long sentences',
            'Occasional false starts with smooth self-correction'
        ],
        'adrd': [
            'High frequency of generic terms; avoids specific nouns',
            'Uniformly short sentences or dramatic length inconsistency',
            'Frequent false starts: "The boy... the child is climbing"'
        ]
    }


@pytest.fixture  
def sample_scenes_data():
    """Sample synthetic scenes data for testing."""
    return {
        'Unnamed: 0': [
            'Kitchen Scene',
            'Breakfast Scene',
            'Garden Scene'
        ],
        '0': [
            'There is a man in the kitchen making toast.',
            'A family is sitting around the table having breakfast.',
            'An elderly woman is watering flowers in her garden.'
        ]
    }


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with input files."""
    temp_dir = tempfile.mkdtemp()
    
    # Create input directory
    input_dir = os.path.join(temp_dir, 'input')
    os.makedirs(input_dir)
    
    # Create sample language_markers.csv
    symptoms_data = {
        'category': ['Lexical Signature', 'Syntactic Features'],
        'test_name': ['Generic Word Overuse', 'Sentence Length Distribution'],
        'description': ['Use of generic terms', 'Variation in sentence length'],
        'control': ['Uses specific terminology', 'Natural sentence mix'],
        'adrd': ['High frequency of generic terms', 'Uniformly short sentences']
    }
    symptoms_df = pd.DataFrame(symptoms_data)
    symptoms_df.to_csv(os.path.join(input_dir, 'language_markers.csv'), index=False)
    
    # Create sample synthetic_scenes.csv
    scenes_data = {
        'Unnamed: 0': ['Scene 1', 'Scene 2'],
        '0': [
            'There is a man in the kitchen making toast.',
            'A family is having breakfast at the table.'
        ]
    }
    scenes_df = pd.DataFrame(scenes_data)
    scenes_df.to_csv(os.path.join(input_dir, 'synthetic_scenes.csv'), index=False)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def api_responses():
    """Predefined API responses for different test scenarios."""
    return {
        'healthy_control': "There is a man in the kitchen preparing breakfast items carefully.",
        'mild_symptomatic': "There is a person in the kitchen making food things.",
        'severe_symptomatic': "There is someone in the place doing stuff with things.",
        'api_error': "API_ERROR",
        'empty_response': ""
    }