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
import base64
import json
import numpy as np
from kokoro import KPipeline
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Kokoro pipeline
pipeline = KPipeline(lang_code='a')  # 'a' => American English, adjust as needed

import numpy as np

async def stream_audio_to_twilio(websocket, stream_sid, text):
    """Convert text to speech and stream it back to Twilio in real-time."""
    generator = pipeline(text, voice='af_heart', speed=1, split_pattern=r'\n+')

    for i, (gs, ps, audio_chunk) in enumerate(generator):
        print(f"Streaming chunk {i}: {gs}")

        # Convert NumPy audio chunk to μ-law format
        audio_bytes = convert_to_mulaw(audio_chunk)

        # Base64 encode
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

        # Create Twilio media message
        media_message = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": encoded_audio
            }
        }

        # Send to Twilio WebSocket
        await websocket.send(json.dumps(media_message))
        print(f"Sent chunk {i} to Twilio")

def text_to_speech(text, voice='af_heart', speed=1, TTS=1):
    if TTS == 1:
        """Generate speech from text using Kokoro TTS."""
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')
    
        #     # Collect all audio chunks
        # audio_chunks = []
        # for i, (gs, ps, audio) in enumerate(generator):
        #     print(f"Chunk {i}: {gs}")  # Log each chunk
        #     audio_chunks.append(audio)

        # # Ensure we have audio data
        # if not audio_chunks:
        #     raise ValueError("No audio generated from text-to-speech.")

        # # Merge all chunks into a single array
        # audio_data = np.concatenate(audio_chunks, axis=0)

        # return audio_data  # Return as NumPy array
    
          # Collect all audio chunks using bytearray
        audio_chunks = []
        for i, (gs, ps, audio) in enumerate(generator):
            print(f"Chunk {i}: {gs}")  # Log each chunk
            audio_chunks.append(audio)

        # Ensure we have audio data
        if not audio_chunks:
            raise ValueError("No audio generated from text-to-speech.")

        # Convert bytearray to NumPy array
        audio_data = np.concatenate(audio_chunks, axis=0)
        del audio_chunks  # Clear audio_chunks from memory

        return audio_data  # Return as NumPy array

    elif TTS == 2:
        """Generate speech from text using ElevenLabs API."""
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
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise an error if the request failed
            audio_content = response.content
            return audio_content
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to generate speech: {str(e)}")
        finally:
            response.close() 

def save_audio_file(audio_data):
    """Save audio as a playable file, handling both NumPy arrays (Kokoro) and raw bytes (ElevenLabs)."""
    os.makedirs("audio_files", exist_ok=True)  # Ensure directory exists

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir='audio_files') as tmpfile:
        if isinstance(audio_data, np.ndarray):
            # If it's a NumPy array (Kokoro), save as WAV with PCM_16 format
            sf.write(tmpfile.name, audio_data, samplerate=24000, subtype='PCM_16')
        elif isinstance(audio_data, bytes):
            tmpfile.write(audio_data)
        else:
            raise TypeError("Audio data must be either a NumPy array or raw bytes.")

        return tmpfile.name  # Return file path

    
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

        audio_input.close()  # Free up memory
        del audio_input

        return audio_output  

    except Exception as e:
        logger.error(f"FFmpeg audio conversion error: {e}")
        return None
    
def convert_audio_to_pcm(audio_data):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
            tmp_mp3.write(audio_data)
            tmp_mp3.flush()
            tmp_mp3_path = tmp_mp3.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcm") as tmp_pcm:
            tmp_pcm_path = tmp_pcm.name
            (
                ffmpeg
                .input(tmp_mp3_path)
                .output(tmp_pcm_path, format="s16le", acodec="pcm_s16le", ac=1, ar="8000")
                .run(quiet=True, overwrite_output=True)
            )
            
            with open(tmp_pcm_path, "rb") as f:
                pcm_audio = f.read()

        os.unlink(tmp_mp3_path)
        os.unlink(tmp_pcm_path)

        return pcm_audio

    except Exception as e:
        logger.error(f"FFmpeg audio conversion error: {e}")
        return None

def convert_to_mulaw(audio_chunk, samplerate=8000):
    """Convert NumPy audio chunk to μ-law format (8000 Hz)."""
    with io.BytesIO() as buffer:
        sf.write(buffer, audio_chunk, samplerate=samplerate, subtype="PCM_U8", format='RAW')
        return buffer.getvalue()

    
def resample_audio(pcm_audio, orig_sr=8000, target_sr=16000):
    """Resample PCM audio from 8000Hz to 16000Hz."""
    try:
        audio_array = np.frombuffer(pcm_audio, dtype=np.int16).astype(np.float32) / 32768.0
        resampled_audio = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)
        resampled_audio = (resampled_audio * 32768.0).astype(np.int16)  # Convert back to int16
        
        del audio_array  # Clear audio_array from memory
        return resampled_audio.tobytes()
    except Exception as e:
        logger.error(f"Error resampling audio: {e}")
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

