import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoProcessor
import warnings
from tqdm import tqdm
from glob import glob
import pickle
from copy import deepcopy
import librosa
import soundfile as sf
import os

# Global paths
OUTPUT_FILE = 'linguistic/transcription/transcripts.p'
TEST_AUDIO_PATH = "data/test_audios/*.mp3"
TRAIN_AUDIO_PATH = "data/train_audios/*.mp3"

warnings.filterwarnings(action='ignore')

def process_audio(audio_file, model, processor, device, prompt):
	# Load audio at 16kHz mono for model compatibility
	audio, sr = librosa.load(audio_file, sr=16000, mono=True)
	
	inputs = processor(
		text=prompt,
		audios=[(audio, sr)],
		return_tensors="pt"
	).to(device)
	
	with torch.no_grad():
		outputs = model.generate(
			**inputs,
			max_new_tokens=200,
			do_sample=False,
			pad_token_id=processor.tokenizer.eos_token_id,
			num_logits_to_keep=1
		)
	
	# Decode only new tokens (skip input prompt)
	response = processor.batch_decode(
		outputs[:, inputs['input_ids'].shape[1]:],
		skip_special_tokens=True
	)[0]
	
	return {"text": response.strip()}

def main():
	
	if os.path.exists(OUTPUT_FILE):
		print (f"Output {OUTPUT_FILE} already exists.")
		return
	
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
	
	model_name = "microsoft/Phi-4-multimodal-instruct"
	processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
	
	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		torch_dtype=torch_dtype,
		trust_remote_code=True,
		low_cpu_mem_usage=True,
		attn_implementation="eager"
	).to(device)
	
	all_files = glob(TEST_AUDIO_PATH) + glob(TRAIN_AUDIO_PATH)
	
	# Phi-4 specific prompt format
	prompt = "<|user|><|audio_1|>Transcribe the audio clip into text. Include natural speech patterns like hesitations and repetitions when they occur.<|end|><|assistant|>"
	
	results = {}
	for audio_file in tqdm(all_files, desc="Processing"):
		fn = audio_file.split('/')[-1].replace('.mp3','')
		output_phi = process_audio(audio_file, model, processor, device, prompt)
		results[fn] = deepcopy(output_phi)
		
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	
	with open(OUTPUT_FILE, 'wb') as f:
		pickle.dump(results, f)

if __name__ == "__main__":
	main()