
# stream_processor.py
import torch
import numpy as np
import time
from silero_vad import get_speech_timestamps, VADIterator
from faster_whisper import WhisperModel


from audio_helpers import text_to_speech, save_audio_file, convert_audio_to_wav

# Initialize Faster Whisper
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")

 

class StreamProcessor:
    def __init__(self, stream_sid, silence_duration=1.5, sample_rate=16000):
        self.stream_sid = stream_sid
        self.audio_buffer = bytes()
        self.silence_duration = silence_duration
        self.sample_rate = sample_rate
        self.speech_buffer = []
        self.last_speech_time = time.time()

        
        # Load Silero VAD
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False
                                           )
        (self.get_speech_timestamps, self.save_audio, self.read_audio,
         self.VADIterator, self.collect_chunks) = utils

        self.vad_iterator = self.VADIterator(self.model)
        self.window_size_samples = 512  # For 16000 Hz audio
        self.num_bytes_per_sample = 2   # int16 has 2 bytes per sample

    def add_audio(self, audio_data):
        """
        Append received audio data to buffer and process VAD.
        """
        self.audio_buffer += audio_data
        self.process_vad()

    def process_vad(self):
        """
        Process Voice Activity Detection (VAD) and determine when the user stops speaking.
        """
        # Calculate total samples in buffer
        total_samples = len(self.audio_buffer) // self.num_bytes_per_sample
        # Process full chunks of window_size_samples
        while total_samples >= self.window_size_samples:
            # Extract a chunk of window_size_samples
            chunk_bytes = self.audio_buffer[:self.window_size_samples * self.num_bytes_per_sample]
            self.audio_buffer = self.audio_buffer[self.window_size_samples * self.num_bytes_per_sample:]
            total_samples -= self.window_size_samples

            # Convert audio_data to float32 numpy array
            audio_array = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_array)

            # Pass audio chunk to VADIterator
            speech_dict = self.vad_iterator(audio_tensor)
            if speech_dict:
                print('speech')
                # Detected speech in this chunk
                self.last_speech_time = time.time()
                self.speech_buffer.append(chunk_bytes)
            else:
                print('silence')
                # Check if silence duration exceeded
                if time.time() - self.last_speech_time > self.silence_duration:
                    print('silence grace elapsed')
                    if self.speech_buffer:
                        transcription = self.transcribe_audio()
                        if transcription:
                            # Handle transcription (e.g., send response)
                            print(f"Transcription: {transcription}")
                        self.speech_buffer = []
                        self.vad_iterator.reset_states()

    def transcribe_audio(self):
        """
        Transcribe buffered speech after detecting silence.
        """
        if not self.speech_buffer:
            return None

        audio_chunk = b"".join(self.speech_buffer)
        self.speech_buffer = []  # Clear after using

        # Save audio_chunk to WAV format for transcription
        wav_audio = convert_audio_to_wav(audio_chunk)
        if wav_audio:
            segments, _ = whisper_model.transcribe(wav_audio, beam_size=5)
            transcription = " ".join([segment.text for segment in segments])
            if transcription.strip():
                return transcription
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
