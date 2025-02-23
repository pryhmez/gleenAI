
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


from audio_helpers import text_to_speech, save_audio_file, convert_audio_to_wav

# Initialize Faster Whisper
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")

 

class StreamProcessor:
    def __init__(self, stream_sid, silence_duration=4, sample_rate=8000, save_interval=10):
        self.stream_sid = stream_sid
        self.audio_buffer = bytes()
        self.silence_duration = silence_duration
        self.sample_rate = sample_rate
        self.speech_buffer = []
        self.all_audio_buffer = []
        self.last_speech_time = time.time()
        self.last_save_time = time.time()  # Timer for saving audio
        self.save_interval = save_interval  # Save interval in seconds
        self.audio_directory = 'audio'  # Directory to save audio files

        # Ensure the audio directory exists
        os.makedirs(self.audio_directory, exist_ok=True)

        # Load Silero VAD
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False)
        (self.get_speech_timestamps, self.save_audio, self.read_audio,
         self.VADIterator, self.collect_chunks) = utils

        self.vad_iterator = self.VADIterator(self.model)
        self.window_size_samples = 256
        self.num_bytes_per_sample = 2   # int16 has 2 bytes per sample

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

        self.audio_buffer += pcm_audio_data
        self.all_audio_buffer.append(pcm_audio_data)  # Store all audio data
        self.process_vad()

    def get_buffer_duration(self):
        """
        Calculate the duration of audio in the buffer.
        """
        total_samples = len(self.audio_buffer) // self.num_bytes_per_sample
        return total_samples / self.sample_rate

    def save_audio_file(self, audio_data, filename):
        """
        Save the raw audio data to a file for inspection.
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join('audio_files', f"chunk_audio_{timestamp}.wav")
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.num_bytes_per_sample)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
        print(f"Saved audio to {filename}")


    def save_compiled_audio(self):
        """
        Save the compiled speech buffer to a file for inspection.
        """
        print("================================================================================================================================Saving compiled audio...")

        if self.all_audio_buffer:
            print("Compiling all audio buffer...")

            compiled_audio = b"".join(self.all_audio_buffer)
            self.all_audio_buffer = []  # Clear after saving

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

    def process_vad(self):
        """
        Process Voice Activity Detection (VAD) and determine when the user stops speaking.
        """
        # Calculate total samples in buffer
        total_samples = len(self.audio_buffer) // self.num_bytes_per_sample
        print(f"Total samples in buffer: {total_samples}")

        # Process full chunks of window_size_samples
        while total_samples >= self.window_size_samples:
            # # Extract a chunk of window_size_samples
            # chunk_bytes = self.audio_buffer[:self.window_size_samples * self.num_bytes_per_sample]
            # self.audio_buffer = self.audio_buffer[self.window_size_samples * self.num_bytes_per_sample:]
            # total_samples -= self.window_size_samples

            # Extract a chunk of window_size_samples
            chunk_size = self.window_size_samples * self.num_bytes_per_sample
            chunk_bytes = self.audio_buffer[:chunk_size]
            self.audio_buffer = self.audio_buffer[chunk_size:]
            total_samples -= self.window_size_samples

            # Convert audio_data to float32 numpy array
            audio_array = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_array)

            self.save_audio_file(chunk_bytes, 'chunk_audio.wav')

            # Print details about the audio tensor
            # print(f"Processing chunk of size: {audio_tensor.size()}")
            # print(f"Audio tensor: {audio_tensor}")

            # Pass audio chunk to VADIterator
            # print("Passing chunk to VADIterator")
            try:
                speech_dict = self.vad_iterator(audio_tensor)
                print(f"VAD output: {speech_dict}")
                if speech_dict:
                    print("Detected speech in chunk")
                    self.last_speech_time = time.time()
                    self.speech_buffer.append(chunk_bytes)
                else:
                    if time.time() - self.last_speech_time > self.silence_duration:
                        print("Silence detected")
                        if self.speech_buffer:
                            print("================================================================================================================================Transcription started")
                            transcription = self.transcribe_audio()
                            if transcription:
                                print(f"Transcription: {transcription}")
                            self.speech_buffer = []
                            self.vad_iterator.reset_states()
            except ValueError as e:
                print(f"VAD processing error: {e}")
                break  # Exit the loop to accumulate more data

        # Check if 30 seconds have passed and save the compiled audio
        if time.time() - self.last_save_time > self.save_interval:
            self.save_compiled_audio()
            self.last_save_time = time.time()
        # else:
        #     print(f"Time since last save: {time.time() - self.last_save_time} seconds")

    def transcribe_audio(self):
        """
        Transcribe buffered speech after detecting silence.
        """
        if not self.speech_buffer:
            return None

        audio_chunk = b"".join(self.speech_buffer)
        self.speech_buffer = []  # Clear after using

        # Save audio_chunk to WAV format for transcription
        # Ensure the audio is at 16kHz for Whisper
        # We need to resample it from 8kHz to 16kHz
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            # Resample to 16000 Hz
            
            resampled_audio = librosa.resample(audio_array.astype(np.float32), orig_sr=8000, target_sr=16000)
            resampled_audio = (resampled_audio * 32768.0).astype(np.int16)
            # Convert back to bytes
            resampled_audio_bytes = resampled_audio.tobytes()
            # Save to BytesIO buffer as WAV
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(16000)
                wf.writeframes(resampled_audio_bytes)
            wav_io.seek(0)

            # Transcribe using Whisper
            segments, _ = whisper_model.transcribe(wav_io)
            transcription = " ".join([segment.text for segment in segments])
            if transcription.strip():
                return transcription
        except Exception as e:
            print(f"Error during transcription: {e}")
        return None




# class StreamProcessor:
#     def __init__(self, stream_sid):
#         self.stream_sid = stream_sid
#         self.audio_buffer = []
#         self.last_audio_time = time.time()
#         self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', source='github')
#         self.get_speech_timestamps, _, self.read_audio, _, _ = self.utils

#     def add_audio(self, audio_data):
#         self.audio_buffer.append(audio_data)
#         self.last_audio_time = time.time()

#     def is_silence(self):
#         if not self.audio_buffer:
#             return False

#         combined_audio = b''.join(self.audio_buffer)
#         audio_array = np.frombuffer(combined_audio, dtype=np.int16)
#         audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

#         speech_timestamps = self.get_speech_timestamps(audio_tensor, self.model, return_seconds=True)
#         return len(speech_timestamps) == 0

#     def process_buffer(self):
#         if self.audio_buffer:
#             audio_chunk = b"".join(self.audio_buffer)
#             self.audio_buffer = []  # Clear the buffer after processing
#             return audio_chunk
#         return None

#     def save_audio_to_file(self, audio_chunk, filename):
#         file_path = os.path.join(audio_files_directory, filename)
#         with open(file_path, 'wb') as f:
#             f.write(audio_chunk)
#         return file_path


# ========================
#  STREAM PROCESSOR CLASS
# ========================    
# class StreamProcessor:
#     def __init__(self, stream_sid):
#         self.stream_sid = stream_sid
#         self.audio_buffer = []
#         self.last_process_time = time.time()

#     def add_audio(self, audio_data):
#         self.audio_buffer.append(audio_data)

#     def should_process(self):
#         # Check if the buffer length is sufficient for processing
#         return len(self.audio_buffer) > 50 or time.time() - self.last_process_time > 3

#     def process_buffer(self):
#         if self.audio_buffer:
#             audio_chunk = b"".join(self.audio_buffer)
#             self.audio_buffer = []  
#             return audio_chunk
#         return None
