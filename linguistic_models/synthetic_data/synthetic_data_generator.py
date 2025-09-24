import os
import pandas as pd
import openai
from tqdm import tqdm
import logging
import csv
import sys

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Debug mode configuration
DEBUG_MODE = '--debug' in sys.argv or os.getenv('DEBUG_MODE', '').lower() == 'true'
DEBUG_MAX_SYMPTOMS = 2
DEBUG_MAX_STORIES = 2

# --- Helper Functions ---
def get_openai_client():
	"""
	Initializes and returns the OpenAI client.
	Retrieves the API key from the OPENAI_API_KEY environment variable.
	"""
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		logging.error("OPENAI_API_KEY environment variable not set.")
		raise ValueError("Please set the OPENAI_API_KEY environment variable.")
	return openai.OpenAI(api_key=api_key)

def generate_single_text(client, prompt):
	"""
	Calls the OpenAI Chat Completions API to generate a single modified passage.
	"""
	try:
		response = client.chat.completions.create(
			model="gpt-4.1-2025-04-14",
			messages=[{"role": "user", "content": prompt}],
			temperature=0.7,
			max_tokens=1024
		)
		return response.choices[0].message.content
	except Exception as e:
		logging.error(f"OpenAI API call failed: {e}")
		return "API_ERROR"

def main():
	"""
	Main function to generate the synthetic dataset.
	"""
	if DEBUG_MODE:
		print("="*60)
		print("DEBUG MODE ENABLED")
		print(f"- Limited to first {DEBUG_MAX_SYMPTOMS} symptoms")
		print(f"- Limited to first {DEBUG_MAX_STORIES} stories")
		print(f"- Expected output: {DEBUG_MAX_SYMPTOMS * 2 * DEBUG_MAX_STORIES} API calls")
		print(f"- Expected files: {DEBUG_MAX_SYMPTOMS * 2} CSV files")
		print("="*60)
	
	# 1. Initialize client and define the prompt template from generate_variants.ipynb
	client = get_openai_client()

	prompt_template = """I am creating a synthetic dataset to research "{test_name}", a type of {category} in spoken language.
The healthy control behavior is: {control}
The symptomatic behavior is: {adrd}
The passage below represents a patient describing a scene. Please modify this passage to include natural sounding {severity} for "{test_name}".
"{passage}"
Respond only with the new modified passage.
"""

	# Define the specific cases to generate per the user's request
	SEVERITY_CASES = [
		(100, 'clear healthy control behavior'), # The "exemplar healthy" case
		(2, 'severe symptomatic')              # The "symptomatic" case (code 2)
	]

	# 2. Load input data
	try:
		symptoms_df = pd.read_csv("input/language_markers.csv").reset_index().rename(columns={'index': 'test_id'})
		stories_df = pd.read_csv("input/synthetic_scenes.csv").rename(columns={'Unnamed: 0': 'title', '0': 'text'}).reset_index().rename(columns={'index': 'story_id'})
		
		# Apply debug mode limits
		if DEBUG_MODE:
			symptoms_df = symptoms_df.head(DEBUG_MAX_SYMPTOMS)
			stories_df = stories_df.head(DEBUG_MAX_STORIES)
			logging.info(f"DEBUG MODE: Limited to {len(symptoms_df)} symptoms and {len(stories_df)} stories.")
		
		logging.info(f"Successfully loaded {len(symptoms_df)} symptoms and {len(stories_df)} stories.")
	except FileNotFoundError as e:
		logging.error(f"Input file not found: {e}. Make sure 'input/language_markers.csv' and 'input/synthetic_scenes.csv' are present.")
		return

	# Create output directory
	output_dir = "output"
	os.makedirs(output_dir, exist_ok=True)

	# 3. Iterate through each symptom and specified severity case
	for _, symptom_row in symptoms_df.iterrows():
		test_id = symptom_row['test_id']
		test_name_clean = symptom_row['test_name'].replace(" ", "_").replace("/", "_").lower()

		for severity_id, severity_text in SEVERITY_CASES:
			output_filename = os.path.join(output_dir, f"{test_id}-{test_name_clean}-{severity_id}.csv")

			if os.path.exists(output_filename):
				logging.info(f"Output file for '{test_name_clean}' (Severity {severity_id}) already exists. Skipping.")
				continue

			logging.info(f"Processing symptom: '{symptom_row['test_name']}' with severity: '{severity_text}'")
			
			results = []
			progress_bar = tqdm(stories_df.iterrows(), total=len(stories_df), desc=f"Symptom {test_id} Sev {severity_id}")

			for _, story_row in progress_bar:
				prompt = prompt_template.format(
					test_name=symptom_row['test_name'],
					category=symptom_row['category'],
					control=symptom_row['control'],
					adrd=symptom_row['adrd'],
					passage=story_row['text'],
					severity=severity_text
				)

				if DEBUG_MODE:
					logging.info(f"DEBUG: Processing story {story_row['story_id']}: '{story_row['text'][:50]}...'")
					logging.info(f"DEBUG: Prompt preview: {prompt[:100]}...")

				# 4. Generate a single text variant using the API
				generated_text = generate_single_text(client, prompt)
				
				if DEBUG_MODE:
					logging.info(f"DEBUG: Generated text: '{generated_text[:100]}...'")
				
				results.append({
					'custom_id': f"{story_row['story_id']}-{test_id}-{severity_id}",
					'content': generated_text
				})

			# 5. Save results to a dedicated CSV file for this symptom/severity combination
			results_df = pd.DataFrame(results)
			results_df.to_csv(output_filename, index=False, quoting=csv.QUOTE_ALL)
			logging.info(f"Saved {len(results)} results to {output_filename}")

if __name__ == "__main__":
	main()