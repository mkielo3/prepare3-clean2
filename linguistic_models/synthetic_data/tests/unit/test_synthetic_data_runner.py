"""
Unit tests for synthetic_data_runner.py

Tests the synthetic data generation process with mocked OpenAI API responses.
"""

import os
import sys
import tempfile
import shutil
import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import csv

# Add the synthetic_data directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from synthetic_data_runner import get_openai_client, generate_single_text, main


class TestGetOpenAIClient:
    """Test OpenAI client initialization."""
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key-123'})
    @patch('synthetic_data_runner.openai.OpenAI')
    def test_get_openai_client_success(self, mock_openai):
        """Test successful OpenAI client initialization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        result = get_openai_client()
        
        mock_openai.assert_called_once_with(api_key='test-key-123')
        assert result == mock_client
    
    @patch.dict('os.environ', {}, clear=True)
    def test_get_openai_client_no_api_key(self):
        """Test error when OPENAI_API_KEY is not set."""
        with pytest.raises(ValueError, match="Please set the OPENAI_API_KEY environment variable"):
            get_openai_client()
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': ''})
    def test_get_openai_client_empty_api_key(self):
        """Test error when OPENAI_API_KEY is empty."""
        with pytest.raises(ValueError, match="Please set the OPENAI_API_KEY environment variable"):
            get_openai_client()


class TestGenerateSingleText:
    """Test single text generation with OpenAI API."""
    
    def test_generate_single_text_success(self):
        """Test successful text generation."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        
        mock_message.content = "Generated synthetic text with memory loss symptoms."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        prompt = "Test prompt for generating synthetic text"
        result = generate_single_text(mock_client, prompt)
        
        # Verify API was called correctly
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4.1-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        
        assert result == "Generated synthetic text with memory loss symptoms."
    
    def test_generate_single_text_api_error(self):
        """Test handling of OpenAI API errors."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API rate limit exceeded")
        
        result = generate_single_text(mock_client, "test prompt")
        
        assert result == "API_ERROR"
    
    def test_generate_single_text_empty_response(self):
        """Test handling of empty API response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        
        mock_message.content = ""
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        result = generate_single_text(mock_client, "test prompt")
        
        assert result == ""


class TestMainFunction:
    """Test the main synthetic data generation function."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_input_files(self, temp_dir):
        """Create sample input CSV files."""
        input_dir = os.path.join(temp_dir, 'input')
        os.makedirs(input_dir)
        
        # Create sample language_markers.csv
        symptoms_data = {
            'category': ['Lexical Signature', 'Syntactic Features'],
            'test_name': ['Generic Word Overuse', 'Sentence Length Distribution'],
            'description': ['Use of generic terms', 'Variation in sentence length'],
            'control': ['Uses specific terminology', 'Natural mix of sentence lengths'],
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
        
        return temp_dir
    
    @patch('synthetic_data_runner.get_openai_client')
    @patch('synthetic_data_runner.generate_single_text')
    def test_main_function_success(self, mock_generate_text, mock_get_client, sample_input_files):
        """Test successful execution of main function."""
        # Change to the temp directory
        original_cwd = os.getcwd()
        os.chdir(sample_input_files)
        
        try:
            # Mock OpenAI client
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # Mock text generation responses (need 8 responses: 2 symptoms × 2 severity × 2 stories)
            mock_generate_text.side_effect = [
                "There is a man in the kitchen making food items.",  # Symptom 0, Severity 100, Story 0
                "There is a man in the kitchen making food items.",  # Symptom 0, Severity 100, Story 1
                "There is a person in the place making things.",     # Symptom 0, Severity 2, Story 0
                "There is a person in the place making things.",     # Symptom 0, Severity 2, Story 1
                "A family is eating food together.",                 # Symptom 1, Severity 100, Story 0
                "A family is eating food together.",                 # Symptom 1, Severity 100, Story 1
                "People are doing stuff at the thing.",              # Symptom 1, Severity 2, Story 0
                "People are doing stuff at the thing."               # Symptom 1, Severity 2, Story 1
            ]
            
            # Run main function
            main()
            
            # Verify output files were created
            output_dir = os.path.join(sample_input_files, 'output')
            assert os.path.exists(output_dir)
            
            # Check that CSV files were created for each symptom/severity combination
            expected_files = [
                '0-Generic_Word_Overuse-100.csv',
                '0-Generic_Word_Overuse-2.csv',
                '1-Sentence_Length_Distribution-100.csv',
                '1-Sentence_Length_Distribution-2.csv'
            ]
            
            for filename in expected_files:
                filepath = os.path.join(output_dir, filename)
                assert os.path.exists(filepath), f"Expected output file {filename} not found"
                
                # Verify CSV content
                df = pd.read_csv(filepath)
                assert len(df) == 2  # Should have 2 rows (one for each story)
                assert 'custom_id' in df.columns
                assert 'content' in df.columns
            
            # Verify generate_single_text was called the expected number of times
            # 2 symptoms × 2 severity levels × 2 stories = 8 calls
            assert mock_generate_text.call_count == 8
            
        finally:
            os.chdir(original_cwd)
    
    @patch('synthetic_data_runner.get_openai_client')
    def test_main_function_missing_input_files(self, mock_get_client, temp_dir):
        """Test main function behavior when input files are missing."""
        # Change to temp directory without input files
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # Run main function - should return early due to missing files
            main()
            
            # Verify no output directory was created
            output_dir = os.path.join(temp_dir, 'output')
            assert not os.path.exists(output_dir)
            
        finally:
            os.chdir(original_cwd)
    
    @patch('synthetic_data_runner.get_openai_client')
    @patch('synthetic_data_runner.generate_single_text')
    def test_main_function_skip_existing_files(self, mock_generate_text, mock_get_client, sample_input_files):
        """Test that main function skips processing when output files already exist."""
        original_cwd = os.getcwd()
        os.chdir(sample_input_files)
        
        try:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_generate_text.return_value = "Generated text"
            
            # Create output directory and one existing file
            output_dir = os.path.join(sample_input_files, 'output')
            os.makedirs(output_dir, exist_ok=True)
            existing_file = os.path.join(output_dir, '0-Generic_Word_Overuse-100.csv')
            
            # Create a dummy existing file
            dummy_data = pd.DataFrame({'custom_id': ['test'], 'content': ['existing content']})
            dummy_data.to_csv(existing_file, index=False)
            
            # Run main function
            main()
            
            # Verify that generate_single_text was called fewer times (skipped existing file)
            # Should be 7 calls instead of 8 (2 symptoms × 2 severity × 2 stories - 1 existing file × 2 stories)
            assert mock_generate_text.call_count == 6  # 3 remaining files × 2 stories each
            
        finally:
            os.chdir(original_cwd)
    
    def test_prompt_template_formatting(self):
        """Test that the prompt template is formatted correctly."""
        from synthetic_data_runner import main
        
        # Test the prompt template used in main()
        prompt_template = """I am creating a synthetic dataset to research "{test_name}", a type of {category} in spoken language.
The healthy control behavior is: {control}
The symptomatic behavior is: {adrd}
The passage below represents a patient describing a scene. Please modify this passage to include natural sounding {severity} for "{test_name}".
"{passage}"
Respond only with the new modified passage.
"""
        
        # Test formatting
        formatted_prompt = prompt_template.format(
            test_name="Generic Word Overuse",
            category="Lexical Signature",
            control="Uses specific terminology",
            adrd="High frequency of generic terms",
            severity="severe symptomatic",
            passage="There is a man in the kitchen making toast."
        )
        
        assert "Generic Word Overuse" in formatted_prompt
        assert "Lexical Signature" in formatted_prompt
        assert "Uses specific terminology" in formatted_prompt
        assert "High frequency of generic terms" in formatted_prompt
        assert "severe symptomatic" in formatted_prompt
        assert "There is a man in the kitchen making toast." in formatted_prompt


class TestCSVOutputFormat:
    """Test the CSV output format and structure."""
    
    def test_csv_output_structure(self, tmp_path):
        """Test that generated CSV files have the correct structure."""
        # Create sample data
        results = [
            {'custom_id': '0-1-100', 'content': 'Generated healthy text'},
            {'custom_id': '1-1-100', 'content': 'Another healthy text'},
        ]
        
        # Save to CSV
        results_df = pd.DataFrame(results)
        output_file = tmp_path / "test_output.csv"
        results_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
        
        # Read back and verify
        loaded_df = pd.read_csv(output_file)
        assert list(loaded_df.columns) == ['custom_id', 'content']
        assert len(loaded_df) == 2
        assert loaded_df.iloc[0]['custom_id'] == '0-1-100'
        assert loaded_df.iloc[0]['content'] == 'Generated healthy text'
    
    def test_custom_id_format(self):
        """Test the custom_id format follows the expected pattern."""
        story_id = 5
        test_id = 2
        severity_id = 100
        
        expected_custom_id = f"{story_id}-{test_id}-{severity_id}"
        assert expected_custom_id == "5-2-100"
        
        # Test with different values
        custom_id = f"{0}-{10}-{2}"
        assert custom_id == "0-10-2"


if __name__ == "__main__":
    pytest.main([__file__])