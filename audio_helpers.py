import requests
import tempfile
import os
from werkzeug.utils import secure_filename
from config import Config

def text_to_speech(text):
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

def save_audio_file(audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', dir='audio_files') as tmpfile:
        tmpfile.write(audio_data)
        return tmpfile.name
