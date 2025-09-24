"""
Integration tests for the complete synthetic data generation workflow.
"""

import os
import sys
import pandas as pd
import pytest
from unittest.mock import patch, Mock
import tempfile
import shutil

# Add the synthetic_data directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from synthetic_data_runner import main


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test the complete end-to-end workflow."""
    
    @pytest.fixture
    def complete_workspace(self):
        """Create a complete workspace with realistic data."""
        temp_dir = tempfile.mkdtemp()
        input_dir = os.path.join(temp_dir, 'input')
        os.makedirs(input_dir)
        
        # Create more realistic language_markers.csv with actual data
        symptoms_data = {
            'category': [
                'Lexical Signature',
                'Lexical Signature', 
                'Syntactic Features',
                'Self-Correction',
                'Content Organization'
            ],
            'test_name': [
                'Generic Word Overuse',
                'Semantic Approximations',
                'Sentence Length Distribution',
                'False Starts',
                'Redundant Descriptions'
            ],
            'description': [
                'Frequency of non-specific terms like "thing," "stuff," "person"',
                'Use of descriptive phrases instead of precise terms',
                'Variation in sentence complexity and length',
                'Frequency of sentence restarts and reformulations',
                'Saying the same information multiple ways'
            ],
            'control': [
                'Uses specific, precise terminology; rarely resorts to generic terms',
                'Uses exact terminology (e.g., "kitchen," "jar")',
                'Natural mix of short, medium, and long sentences',
                'Occasional false starts with smooth self-correction',
                'Efficient description; minimal redundancy'
            ],
            'adrd': [
                'High frequency of generic terms; consistently avoids specific nouns',
                'Uses circumlocutory descriptions (e.g., "cooking place," "food container")',
                'Uniformly short sentences or dramatic length inconsistency',
                'Frequent false starts: "The boy... the child is climbing"',
                'Multiple ways of saying same thing without adding detail'
            ]
        }
        symptoms_df = pd.DataFrame(symptoms_data)
        symptoms_df.to_csv(os.path.join(input_dir, 'language_markers.csv'), index=False)
        
        # Create realistic scenes data
        scenes_data = {
            'Unnamed: 0': [
                'Kitchen Chaos',
                'Breakfast Spill',
                'Garden Morning',
                'Living Room'
            ],
            '0': [
                'There is a man in the kitchen making toast, but he got distracted by his phone and the toast is burning.',
                'A family is eating breakfast when the youngest child accidentally knocks over a glass of orange juice.',
                'An elderly woman is carefully watering her roses in the early morning sunlight.',
                'Two children are watching television while their father reads a newspaper in his favorite chair.'
            ]
        }
        scenes_df = pd.DataFrame(scenes_data)
        scenes_df.to_csv(os.path.join(input_dir, 'synthetic_scenes.csv'), index=False)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('synthetic_data_runner.get_openai_client')
    @patch('synthetic_data_runner.generate_single_text')
    def test_complete_generation_workflow(self, mock_generate, mock_client, complete_workspace):
        """Test the complete generation workflow with realistic data."""
        original_cwd = os.getcwd()
        os.chdir(complete_workspace)
        
        try:
            # Mock the OpenAI client
            mock_client.return_value = Mock()
            
            # Create realistic mock responses that demonstrate the symptoms
            def generate_realistic_response(client, prompt):
                if 'Generic Word Overuse' in prompt and 'clear healthy control' in prompt:
                    return "There is a man in the kitchen making toast, but he got distracted by his phone and the bread is burning."
                elif 'Generic Word Overuse' in prompt and 'severe symptomatic' in prompt:
                    return "There is a person in the place making things, but he got distracted by his stuff and the things are burning."
                elif 'Semantic Approximations' in prompt and 'clear healthy control' in prompt:
                    return "There is a man in the kitchen making toast, but he got distracted by his phone and the toast is burning."
                elif 'Semantic Approximations' in prompt and 'severe symptomatic' in prompt:
                    return "There is a person in the cooking place making food items, but he got distracted by his communication device and the bread things are burning."
                else:
                    return "Generated response for testing"
            
            mock_generate.side_effect = generate_realistic_response
            
            # Run the main function
            main()
            
            # Verify output structure
            output_dir = os.path.join(complete_workspace, 'output')
            assert os.path.exists(output_dir)
            
            # Should have files for each symptom × severity combination
            # 5 symptoms × 2 severity levels = 10 files
            output_files = os.listdir(output_dir)
            assert len(output_files) == 10
            
            # Verify file naming pattern
            expected_patterns = [
                '0-Generic_Word_Overuse-100.csv',
                '0-Generic_Word_Overuse-2.csv',
                '1-Semantic_Approximations-100.csv',
                '1-Semantic_Approximations-2.csv',
                '2-Sentence_Length_Distribution-100.csv',
                '2-Sentence_Length_Distribution-2.csv',
                '3-False_Starts-100.csv',
                '3-False_Starts-2.csv',
                '4-Redundant_Descriptions-100.csv',
                '4-Redundant_Descriptions-2.csv'
            ]
            
            for expected_file in expected_patterns:
                assert expected_file in output_files, f"Missing expected file: {expected_file}"
            
            # Verify content of a sample file
            sample_file = os.path.join(output_dir, '0-Generic_Word_Overuse-100.csv')
            df = pd.read_csv(sample_file)
            
            assert len(df) == 4  # Should have 4 rows (one for each scene)
            assert all(col in df.columns for col in ['custom_id', 'content'])
            
            # Verify custom_id format
            assert df.iloc[0]['custom_id'] == '0-0-100'  # story_id-test_id-severity_id
            assert df.iloc[1]['custom_id'] == '1-0-100'
            
            # Verify content is different between healthy and symptomatic
            healthy_file = os.path.join(output_dir, '0-Generic_Word_Overuse-100.csv')
            symptomatic_file = os.path.join(output_dir, '0-Generic_Word_Overuse-2.csv')
            
            healthy_df = pd.read_csv(healthy_file)
            symptomatic_df = pd.read_csv(symptomatic_file)
            
            # Content should be different
            assert healthy_df.iloc[0]['content'] != symptomatic_df.iloc[0]['content']
            
            # Verify that mock was called the expected number of times
            # 5 symptoms × 2 severity levels × 4 scenes = 40 calls
            assert mock_generate.call_count == 40
            
        finally:
            os.chdir(original_cwd)
    
    def test_data_consistency(self, complete_workspace):
        """Test that input data is loaded consistently."""
        original_cwd = os.getcwd()
        os.chdir(complete_workspace)
        
        try:
            # Load the data as the main function would
            symptoms_df = pd.read_csv("input/language_markers.csv").reset_index().rename(columns={'index': 'test_id'})
            stories_df = pd.read_csv("input/synthetic_scenes.csv").rename(columns={'Unnamed: 0': 'title', '0': 'text'}).reset_index().rename(columns={'index': 'story_id'})
            
            # Verify data integrity
            assert len(symptoms_df) == 5
            assert len(stories_df) == 4
            
            # Verify required columns exist
            required_symptom_cols = ['test_id', 'category', 'test_name', 'description', 'control', 'adrd']
            assert all(col in symptoms_df.columns for col in required_symptom_cols)
            
            required_story_cols = ['story_id', 'title', 'text']
            assert all(col in stories_df.columns for col in required_story_cols)
            
            # Verify no missing data in critical columns
            assert not symptoms_df['test_name'].isnull().any()
            assert not symptoms_df['control'].isnull().any()
            assert not symptoms_df['adrd'].isnull().any()
            assert not stories_df['text'].isnull().any()
            
        finally:
            os.chdir(original_cwd)


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in realistic scenarios."""
    
    @patch('synthetic_data_runner.get_openai_client')
    @patch('synthetic_data_runner.generate_single_text')
    def test_partial_api_failures(self, mock_generate, mock_client, temp_workspace):
        """Test handling of partial API failures."""
        original_cwd = os.getcwd()
        os.chdir(temp_workspace)
        
        try:
            mock_client.return_value = Mock()
            
            # Simulate some API calls succeeding and others failing
            # Need 8 responses: 2 symptoms × 2 severity × 2 stories
            responses = [
                "Successful generation 1",
                "API_ERROR",
                "Successful generation 2", 
                "API_ERROR",
                "Successful generation 3",
                "API_ERROR",
                "Successful generation 4", 
                "API_ERROR"
            ]
            mock_generate.side_effect = responses
            
            main()
            
            # Verify that files are still created even with partial failures
            output_dir = os.path.join(temp_workspace, 'output')
            assert os.path.exists(output_dir)
            
            # Should have files for both symptoms × both severity levels
            output_files = os.listdir(output_dir)
            assert len(output_files) == 4
            
            # Verify that API_ERROR responses are included in the output
            sample_file = os.path.join(output_dir, '0-Generic_Word_Overuse-100.csv')
            df = pd.read_csv(sample_file)
            
            # Should contain both successful and error responses
            content_values = df['content'].tolist()
            assert "Successful generation 1" in content_values
            assert "API_ERROR" in content_values
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])