import torch
import torchaudio
import os
from yarngpt.audiotokenizer import AudioTokenizer
from transformers import AutoModelForCausalLM

# Define model paths
hf_path = "saheedniyi/YarnGPT"
wav_tokenizer_config_path = "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
wav_tokenizer_model_path = "wavtokenizer_large_speech_320_24k.ckpt"

# Load the audio tokenizer
audio_tokenizer = AudioTokenizer(hf_path, wav_tokenizer_model_path, wav_tokenizer_config_path)
model = AutoModelForCausalLM.from_pretrained(hf_path, torch_dtype="auto").to(audio_tokenizer.device)

def text_to_speech_yarngpt(text, speaker="idera"):
    """Converts text to speech using YarnGPT."""
    prompt = audio_tokenizer.create_prompt(text, speaker)
    input_ids = audio_tokenizer.tokenize_prompt(prompt)

    # Generate speech tokens
    output = model.generate(input_ids=input_ids, temperature=0.1, repetition_penalty=1.1, max_length=6000)
    
    # Convert output to audio
    codes = audio_tokenizer.get_codes(output)
    audio = audio_tokenizer.get_audio(codes)
    
    # Save the audio file
    os.makedirs("audio_files", exist_ok=True)
    audio_path = f"audio_files/{hash(text)}.wav"
    torchaudio.save(audio_path, audio, sample_rate=21000)
    
    return audio_path
