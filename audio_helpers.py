import requests
import tempfile
import os
from werkzeug.utils import secure_filename
from config import Config

import torch
import soundfile as sf
from vall_e_x import VallExModel

# Load the VALL-E X model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VallExModel().to(device)
model.load_state_dict(torch.load('path/to/vallex-checkpoint.pt'))
model.eval()

def _text_to_speech(text):
    print(Config.ELEVENLABS_API_KEY)
    print(Config.VOICE_ID)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{Config.VOICE_ID}"
    headers = {
        'Content-Type': 'application/json',
        'xi-api-key': Config.ELEVENLABS_API_KEY
    }
    data = {
        "model_id": "eleven_monolingual_v1",
        "text": text,
        "voice_settings": {
            "similarity_boost": 0.8,
            "stability": 0.5,
            "use_speaker_boost": True
        }
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to generate speech: {response.text}")

def _save_audio_file(audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', dir='audio_files') as tmpfile:
        tmpfile.write(audio_data)
        return tmpfile.name

def text_to_speech(text):
    # Generate speech using VALL-E X
    input_text = torch.tensor([text]).to(model.device)
    output = model.generate(input_text)
    return output

def save_audio_file(audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir='audio_files') as tmpfile:
        sf.write(tmpfile.name, audio_data.cpu().numpy(), model.sample_rate)
        return tmpfile.name
