# stream_processor.py
import torch
import numpy as np
import time
import os
import io
import audioop
import wave
import librosa
from silero_vad import get_speech_timestamps, VADIterator
from faster_whisper import WhisperModel
from audio_helpers import resample_audio, save_as_wav_inmem


from audio_helpers import text_to_speech, save_audio_file, convert_audio_to_wav

# Initialize Faster Whisper
whisper_model = WhisperModel("distil-large-v2", device="cuda", compute_type="float16")

 

class StreamProcessor:
    def __init__(self, stream_sid, silence_duration=2, sample_rate=8000, save_interval=10):
        self.stream_sid = stream_sid
        self.audio_buffer = bytearray()
        self.silence_duration = silence_duration
        self.sample_rate = sample_rate
        self.speech_buffer = []
        self.all_audio_buffer = []
        self.last_speech_time = time.time()
        self.last_save_time = time.time()  # Timer for saving audio
        self.save_interval = save_interval  # Save interval in seconds
        self.audio_directory = 'audio'  # Directory to save audio files
        self.speech_detected = False
        self.recording_session_active = False
        self.end_speech_time = None
        self.pause_duration = 1
        self.transcription = None 

        # Ensure the audio directory exists
        os.makedirs(self.audio_directory, exist_ok=True)

        # Load Silero VAD
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False,
                                           opset_version=16)
        (self.get_speech_timestamps, self.save_audio, self.read_audio,
         self.VADIterator, self.collect_chunks) = utils

        self.vad_iterator = self.VADIterator(self.model, sampling_rate=sample_rate)
        self.window_size_samples = 256
        self.num_bytes_per_sample = 2   # int16 has 2 bytes per sample

    def get_transcription(self):
        transcription = self.transcription
        self.transcription = None
        return transcription

    def add_audio(self, audio_data):
        """
        Append received audio data to buffer and process VAD.
        """
        # print("Adding audio data")
        # print(f"Audio data length: {len(audio_data)} bytes")
        # Convert μ-law encoded audio data to linear PCM
        try:
            pcm_audio_data = audioop.ulaw2lin(audio_data, self.num_bytes_per_sample)
        except Exception as e:
            print(f"Error converting μ-law to PCM: {e}")
            return

        # self.audio_buffer += pcm_audio_data
        self.audio_buffer.extend(pcm_audio_data)
        # self.all_audio_buffer.append(pcm_audio_data)  # Store all audio data
        self.process_vad()

    def process_vad(self):
        """
        Process Voice Activity Detection (VAD) and determine when the user stops speaking.
        """
        # Calculate total samples in buffer
        total_samples = len(self.audio_buffer) // self.num_bytes_per_sample
        # print(f"Total samples in buffer: {total_samples}")

        # Process full chunks of window_size_samples
        while total_samples >= self.window_size_samples:
            # Extract a chunk of window_size_samples
            # chunk_size = self.window_size_samples * self.num_bytes_per_sample
            # chunk_bytes = self.audio_buffer[:chunk_size]
            # self.audio_buffer = self.audio_buffer[chunk_size:]
            # total_samples -= self.window_size_samples

            chunk_size = self.window_size_samples * self.num_bytes_per_sample
            # Get the next chunk
            chunk_bytes = self.audio_buffer[:chunk_size]
            # Temporarily remove chunk_bytes from the buffer
            del self.audio_buffer[:chunk_size]
            total_samples -= self.window_size_samples

            # Convert audio_data to float32 numpy array
            audio_array = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)

            # Pass audio chunk to VADIterator
            try:
                speech_dict = self.vad_iterator(audio_tensor)
                # print(f"VAD output: {speech_dict}")

                if speech_dict is not None:
                    if 'start' in speech_dict:
                        print("Detected start of speech")
                        self.speech_detected = True
                        self.recording_session_active = True
                        self.end_speech_time = None
                        self.speech_buffer.append(chunk_bytes)
                    elif 'end' in speech_dict:
                        print("Detected end of speech so waiting for pause grace")
                        self.speech_detected = False
                        self.end_speech_time = time.time()
                        self.speech_buffer.append(chunk_bytes)
                    elif self.recording_session_active:
                        print('still in active session waiting for timeout1')
                        self.speech_buffer.append(chunk_bytes)
                else:
                    if self.recording_session_active:
                        print('still in active session waiting for timout2')
                        self.speech_buffer.append(chunk_bytes)

            except ValueError as e:
                print(f"VAD processing error: {e}")
                # Prepend the chunk_bytes back to the audio_buffer so it can be combined with future data
                self.audio_buffer = bytearray(chunk_bytes) + self.audio_buffer
                break    # Exit the loop to accumulate more data

        if self.recording_session_active and self.end_speech_time and (time.time() - self.end_speech_time) > self.pause_duration:
            print("Pause duration elapsed, transcribing audio")
            # self.save_compiled_audio()
            self.transcription = self.transcribe_audio()
            if self.transcription:
                print(f"Transcription: {self.transcription}")
            self.speech_buffer = []
            self.end_speech_time = None
            self.speech_detected = False
            self.recording_session_active = False  # End recording session
            self.vad_iterator.reset_states()

    def save_compiled_audio(self):
        """
        Save the compiled speech buffer to a file for inspection.
        """
        print("================================================================================================================================Saving compiled audio...")

        if self.speech_buffer:
            print("Compiling all audio buffer...")

            compiled_audio = b"".join(self.speech_buffer)

            # Define the filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.audio_directory, f"compiled_audio_{timestamp}.wav")
            total_samples = len(compiled_audio) // self.num_bytes_per_sample
            expected_duration = total_samples / self.sample_rate
            print(f"Compiled audio data length: {len(compiled_audio)} bytes")
            print(f"Total samples: {total_samples}")
            print(f"Expected duration: {expected_duration:.2f} seconds")

            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.num_bytes_per_sample)
                wf.setframerate(self.sample_rate)
                wf.writeframes(compiled_audio)
            print(f"================================================================================================================================Saved compiled audio to {filename}")
        else:
            print("================================================================================================================================No audio buffer to save.")

    def transcribe_audio(self):
        """
        Transcribe buffered speech after detecting silence.
        """
        print('=================================================================================================starting transcription')
        if not self.speech_buffer:
            return None

        audio_chunk = b"".join(self.speech_buffer)
        self.speech_buffer = []  # Clear after using

        # Save audio_chunk to WAV format for transcription
        # Ensure the audio is at 16kHz for Whisper
        # We need to resample it from 8kHz to 16kHz
        try:
            #resample audio from 8KHz to 16KHz
            resampled_audio_bytes = resample_audio(audio_chunk, orig_sr=8000, target_sr=16000)

            # Save to BytesIO buffer as WAV
            wav_io = save_as_wav_inmem(resampled_audio_bytes, sample_rate=16000)

            # Transcribe using Whisper
            segments, _ = whisper_model.transcribe(wav_io)
            transcription = " ".join([segment.text for segment in segments])
            if transcription.strip():
                return transcription
        except Exception as e:
            print(f"Error during transcription: {e}")
        return None