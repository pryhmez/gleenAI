import requests
import tempfile
import os
import io
import ffmpeg 
import logging
from werkzeug.utils import secure_filename
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
# Convert raw audio bytes to WAV format
def convert_audio_to_wav(audio_chunk):
    try:
        audio_input = io.BytesIO(audio_chunk)  # Wrap in BytesIO
        audio_output = io.BytesIO()

        process = (
            ffmpeg
            .input('pipe:0', format='s16le', acodec='pcm_s16le', ac=1, ar='16000')
            .output('pipe:1', format='wav')
            .run(input=audio_input.read(), capture_stdout=True, capture_stderr=True)
        )

        audio_output.write(process[0])
        audio_output.seek(0)  # Reset pointer to start

        return audio_output  # Return WAV buffer

    except Exception as e:
        logger.error(f"FFmpeg audio conversion error: {e}")
        return None
