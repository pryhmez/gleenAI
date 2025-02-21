
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
    def __init__(self, stream_sid, silence_duration=1.5, sample_rate=16000, min_chunk_duration=0.2):
        self.stream_sid = stream_sid
        self.audio_buffer = []
        self.silence_duration = silence_duration
        self.sample_rate = sample_rate
        self.min_chunk_duration = min_chunk_duration  # Minimum chunk duration in seconds
        self.speech_buffer = []
        self.last_speech_time = time.time()
        
        # Load Silero VAD
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        (self.get_speech_timestamps, _, _, VADIterator, _) = utils
        
        self.vad_iterator = VADIterator(self.model)
    
    def add_audio(self, audio_data):
        """
        Append received audio data to buffer and process VAD.
        """
        self.audio_buffer.append(audio_data)
        if self.get_buffer_duration() >= self.min_chunk_duration:
            self.process_vad()
    
    def get_buffer_duration(self):
        """
        Calculate the duration of audio in the buffer.
        """
        total_samples = len(b''.join(self.audio_buffer)) // 2  # 2 bytes per sample
        return total_samples / self.sample_rate
    
    def process_vad(self):
        """
        Process Voice Activity Detection (VAD) and determine when the user stops speaking.
        """
        combined_audio = b''.join(self.audio_buffer)
        self.audio_buffer = []

        # Convert audio_data to float32 numpy array
        audio_array = np.frombuffer(combined_audio, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_array)
        
        # Pass audio chunk to VADIterator
        speech_probs = self.vad_iterator(audio_tensor, return_seconds=True)
        
        if speech_probs:
            # Detected speech in this chunk
            self.last_speech_time = time.time()
            self.speech_buffer.append(combined_audio)
        else:
            # Check if silence duration exceeded
            if time.time() - self.last_speech_time > self.silence_duration:
                if self.speech_buffer:
                    transcription = self.transcribe_audio()
                    if transcription:
                        # Handle transcription (e.g., send response)
                        print(f"Transcription: {transcription}")
                    self.speech_buffer = []
                    self.vad_iterator.reset_states()
                    self.last_speech_time = time.time()
    
    def transcribe_audio(self):
        """
        Transcribe buffered speech after detecting silence.
        """
        if not self.speech_buffer:
            return None
        
        audio_chunk = b"".join(self.speech_buffer)
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
