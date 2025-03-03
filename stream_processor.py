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
import threading
from concurrent.futures import ThreadPoolExecutor

# Initialize Faster Whisper
whisper_model = WhisperModel("large-v1", device="cuda", compute_type="float16")

 
class StreamProcessor:
    def __init__(self, stream_sid, silence_duration=2, sample_rate=8000, save_interval=10, stream_chunk_duration=2):
        self.stream_sid = stream_sid
        self.audio_buffer = bytearray()
        self.silence_duration = silence_duration
        self.sample_rate = sample_rate
        self.speech_buffer = bytearray()
        self.all_audio_buffer = []
        self.last_speech_time = time.time()
        self.last_save_time = time.time()  # Timer for saving audio
        self.save_interval = save_interval  # Save interval in seconds
        self.audio_directory = 'audio'  # Directory to save audio files
        self.speech_detected = False
        self.recording_session_active = False
        self.end_speech_time = None
        self.pause_duration = 0.5
        self.stream_chunk_duration = stream_chunk_duration

        self.transcription_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.last_partial_ts = time.time()
        self.transcription = None
        self.running_transcript = ""


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
                if speech_dict is not None:
                    if 'start' in speech_dict:
                        print("Detected start of speech")
                        self.speech_detected = True
                        self.recording_session_active = True
                        self.end_speech_time = None
                        self.speech_buffer.extend(chunk_bytes)
                    elif 'end' in speech_dict:
                        print("Detected end of speech so waiting for pause grace")
                        self.speech_detected = False
                        self.end_speech_time = time.time()
                        self.speech_buffer.extend(chunk_bytes)
                    elif self.recording_session_active:
                        # print('still in active session waiting for timeout1')
                        self.speech_buffer.extend(chunk_bytes)
                else:
                    if self.recording_session_active:
                        # print('still in active session waiting for timout2')
                        self.speech_buffer.extend(chunk_bytes)

            except ValueError as e:
                print(f"VAD processing error: {e}")
                # Prepend the chunk_bytes back to the audio_buffer so it can be combined with future data
                self.audio_buffer = bytearray(chunk_bytes) + self.audio_buffer
                break    # Exit the loop to accumulate more data

        if self.recording_session_active:
            total_samples_in_buffer = len(self.speech_buffer) // self.num_bytes_per_sample
            duration_seconds = total_samples_in_buffer / self.sample_rate
            
            # Check if we have reached at least 1 second of audio
            if duration_seconds >= self.stream_chunk_duration:
                # Compute the number of samples (and bytes) for one second
                one_second_samples = int(self.sample_rate * self.stream_chunk_duration)
                one_second_bytes = one_second_samples * self.num_bytes_per_sample
                
                # Extract one second worth of audio from the speech buffer
                audio_segment = bytes(self.speech_buffer[:one_second_bytes])
                
                # Submit the partial transcription task in the background
                self.executor.submit(self.run_partial_transcription, audio_segment)
                
                # Remove only that portion from the speech buffer, leaving any excess
                del self.speech_buffer[:one_second_bytes]

        if self.recording_session_active and self.end_speech_time and (time.time() - self.end_speech_time) > self.pause_duration:
            print("Pause duration elapsed, transcribing audio")
            # self.save_compiled_audio()
            self.transcription = self.transcribe_audio()
            if self.transcription:
                print(f"Transcription: {self.transcription}")
            self.speech_buffer = bytearray()
            self.end_speech_time = None
            self.speech_detected = False
            self.recording_session_active = False  # End recording session
            self.vad_iterator.reset_states()

    def run_partial_transcription(self, audio_data):
        if not self.speech_buffer:
            return
        try:
            # audio_chunk = bytes(self.speech_buffer)
            # self.speech_buffer = bytearray()            
            resampled_audio_bytes = resample_audio(audio_data, orig_sr=8000, target_sr=16000)
            wav_io = save_as_wav_inmem(resampled_audio_bytes, sample_rate=16000)
            segments, _ = whisper_model.transcribe(wav_io, beam_size=8)
            partial = " ".join([segment.text for segment in segments])
            if partial:
                with self.transcription_lock:
                    # Append the new partial result to what already exists.
                    # Optionally, you might want to trim overlaps or spaces.
                    if self.running_transcript:
                        self.running_transcript += " " + partial
                    else:
                        self.running_transcript = partial
                print("Partial transcription updated:", partial)
        except Exception as e:
            print(f"Error during partial transcription: {e}")
            
    def transcribe_audio(self):
        transcription_start_time = time.time()
        if not self.speech_buffer:
            return None

        # Save audio_chunk to WAV format for transcription7
        # We need to resample it from 8kHz to 16kHz
        try:
            # audio_chunk = b"".join(self.speech_buffer)
            audio_chunk = bytes(self.speech_buffer)
            self.speech_buffer = bytearray()  # Clear after using
            #resample audio from 8KHz to 16KHz
            resampled_audio_bytes = resample_audio(audio_chunk, orig_sr=8000, target_sr=16000)

            # Save to BytesIO buffer as WAV
            wav_io = save_as_wav_inmem(resampled_audio_bytes, sample_rate=16000)

            # Transcribe using Whisper
            segments, _ = whisper_model.transcribe(wav_io, beam_size=5)
            partial = " ".join([segment.text for segment in segments])
            self.running_transcript += " " + partial
            transcription = self.running_transcript
            print(f"transcription time was: {time.time() - transcription_start_time}s")
            if transcription.strip():
                return transcription
        except Exception as e:
            print(f"Error during transcription: {e}")
        return None
    

        # def save_compiled_audio(self):
    #     """
    #     Save the compiled speech buffer to a file for inspection.
    #     """
    #     print("================================================================================================================================Saving compiled audio...")

    #     if self.speech_buffer:
    #         print("Compiling all audio buffer...")

    #         compiled_audio = b"".join(self.speech_buffer)

    #         # Define the filename with timestamp
    #         timestamp = time.strftime("%Y%m%d-%H%M%S")
    #         filename = os.path.join(self.audio_directory, f"compiled_audio_{timestamp}.wav")
    #         total_samples = len(compiled_audio) // self.num_bytes_per_sample
    #         expected_duration = total_samples / self.sample_rate
    #         print(f"Compiled audio data length: {len(compiled_audio)} bytes")
    #         print(f"Total samples: {total_samples}")
    #         print(f"Expected duration: {expected_duration:.2f} seconds")

    #         with wave.open(filename, 'wb') as wf:
    #             wf.setnchannels(1)
    #             wf.setsampwidth(self.num_bytes_per_sample)
    #             wf.setframerate(self.sample_rate)
    #             wf.writeframes(compiled_audio)
    #         print(f"================================================================================================================================Saved compiled audio to {filename}")
    #     else:
    #         print("================================================================================================================================No audio buffer to save.")

# def record_chunk(p, stream, file_path, chunk_length=1):
#     frames= []
#     for _ in range(0, int(16000/1024*chunk_length)):
#         data = stream.read(1024)
#         frames.append(data)
#     wf = wave.open(file_path, 'wb')
#     wf.setnchannels(1)  # Mono audio
#     wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))  # 16-bit PCM (2 bytes)
#     wf.setframerate(16000)
#     wf.writeframes(b''.join(frames))
#     wf.close()
