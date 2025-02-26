import requests
import tempfile
import os
import io
import ffmpeg 
import logging
from werkzeug.utils import secure_filename
from config import Config
import wave
import librosa
import numpy as np
from kokoro import KPipeline
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Kokoro pipeline
pipeline = KPipeline(lang_code='a')  # 'a' => American English, adjust as needed

def text_to_speech(text, voice='af_heart', speed=1, TTS=1):
    if TTS == 1:
        """Generate speech from text using Kokoro TTS."""
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')

        # Generate speech audio and save the file
        audio_data = []
        for _, _, audio in generator:
            audio_data.extend(audio)

        return audio_data
    elif TTS == 2:
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
    
def convert_audio_to_pcm(audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
        tmp_mp3.write(audio_data)
        tmp_mp3.flush()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcm") as tmp_pcm:
            (
                ffmpeg
                .input(tmp_mp3.name)
                .output(tmp_pcm.name, format="s16le", acodec="pcm_s16le", ac=1, ar="8000")
                .run(quiet=True, overwrite_output=True)
            )
            
            with open(tmp_pcm.name, "rb") as f:
                pcm_audio = f.read()

        os.unlink(tmp_mp3.name)
        os.unlink(tmp_pcm.name)

        return pcm_audio

    
def resample_audio(pcm_audio, orig_sr=8000, target_sr=16000):
    """Resample PCM audio from 8000Hz to 16000Hz."""
    try:
        audio_array = np.frombuffer(pcm_audio, dtype=np.int16).astype(np.float32) / 32768.0
        resampled_audio = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)
        resampled_audio = (resampled_audio * 32768.0).astype(np.int16)  # Convert back to int16
        return resampled_audio.tobytes()
    except Exception as e:
        print(f"Error resampling audio: {e}")
        return None

    
def save_as_wav_inmem(audio_bytes, sample_rate=16000):
    """Save PCM audio bytes as an in-memory WAV file."""
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit PCM (2 bytes)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    wav_io.seek(0)
    return wav_io
# def transcribe_audio_file(audio_file_path):
#     """
#     Transcribes an audio file using the Faster Whisper model and logs the output.

#     Args:
#         audio_file_path (str): The path to the audio file to transcribe.
#     """
#     try:
#         # Open the audio file as a binary stream
#         with open(audio_file_path, 'rb') as f:
#             audio_data = f.read()
        
#         # Create a file-like object from the audio data
#         audio_buffer = io.BytesIO(audio_data)
        
#         # Transcribe the audio
#         segments, info = whisper_model.transcribe(audio_buffer)
        
#         # Combine the transcribed text from all segments
#         transcription = " ".join([segment.text for segment in segments])
        
#         # Log the output
#         print(f"Transcription for '{audio_file_path}':\n{transcription}")
#     except Exception as e:
#         print(f"Error transcribing '{audio_file_path}': {e}")

