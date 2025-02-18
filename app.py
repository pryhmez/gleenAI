from flask import Flask, request, jsonify
from twilio.twiml.voice_response import VoiceResponse, Start
from faster_whisper import WhisperModel
import numpy as np
import base64
import audioop
import redis
import json
import threading
from queue import Queue
import torch
from flask_socketio import SocketIO
from twilio.rest import Client

import os
from audio_helpers import text_to_speech, save_audio_file
from ai_helpers import process_initial_message, process_message, initiate_inbound_message
from config import Config
import threading
import time
from urllib.parse import quote_plus
import uuid
import logging

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)
socketio = SocketIO(app)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
twilio_client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)

try:
    val = redis_client.ping()
    logging.info("Connected to Redis")
except redis.ConnectionError as e:
    logging.error(f"Redis connection error: {e}")
    
# Initialize Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel("base", device=device, compute_type="float16" if device == "cuda" else "int8")

# Audio processing queues for different calls
call_queues = {}
active_calls = {}

class CallHandler:
    def __init__(self, call_sid):
        self.call_sid = call_sid
        self.audio_queue = Queue()
        self.is_processing = True
        self.current_response = None
        self.waiting_for_response = False

def process_audio_stream(call_sid):
    """Process audio stream for a specific call."""
    handler = active_calls[call_sid]
    audio_buffer = []
    
    while handler.is_processing:
        if handler.audio_queue.qsize() > 20:  # Process ~1.5 seconds of audio
            # Only process if we're not waiting for a response
            if not handler.waiting_for_response:
                audio_data = []
                for _ in range(20):
                    audio_data.extend(handler.audio_queue.get())
                
                audio_data = np.array(audio_data, dtype=np.float32)
                
                # Transcribe with Faster Whisper
                segments, _ = whisper_model.transcribe(audio_data, beam_size=5)
                
                # Get transcribed text
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        print(f"Transcribed: {text}")
                        
                        # Mark that we're waiting for a response
                        handler.waiting_for_response = True
                        
                        # Process through AI pipeline
                        message_history = json.loads(redis_client.get(call_sid) or '[]')
                        ai_response = process_message(message_history, text)
                        
                        # Convert response to speech
                        audio_data = text_to_speech(ai_response)
                        audio_file_path = save_audio_file(audio_data)
                        
                        # Create TwiML to play the response
                        response = VoiceResponse()
                        response.play(f"{Config.APP_PUBLIC_URL}/audio/{audio_file_path}")
                        
                        # Update the call with the new TwiML
                        twilio_client.calls(call_sid).update(twiml=str(response))
                        
                        # Update message history
                        message_history.append({"role": "user", "content": text})
                        message_history.append({"role": "assistant", "content": ai_response})
                        redis_client.set(call_sid, json.dumps(message_history))
                        
                        # Mark that we've handled the response
                        handler.waiting_for_response = False

@app.route('/voice', methods=['POST'])
def voice():
    """Handle incoming calls and start streaming."""
    response = VoiceResponse()
    call_sid = request.values.get('CallSid')
    
    # Initialize call handler
    active_calls[call_sid] = CallHandler(call_sid)
    
    # Start media stream
    start = Start()
    start.stream(url=f'wss://{request.host}/stream')
    response.append(start)
    
    # Start processing thread for this call
    thread = threading.Thread(target=process_audio_stream, args=(call_sid,))
    thread.daemon = True
    thread.start()
    
    # Add initial greeting
    response.say("Hello, how can I help you today?")
    
    return str(response)

@app.route('/stream', methods=['POST'])
def stream():
    """Handle incoming audio stream from Twilio."""
    if 'twilio-streaming-payload' in request.headers:
        call_sid = request.values.get('CallSid')
        
        if call_sid in active_calls:
            handler = active_calls[call_sid]
            
            payload = request.get_data()
            audio_data = base64.b64decode(payload)
            
            # Convert mulaw to PCM
            audio_data = audioop.ulaw2lin(audio_data, 2)
            
            # Resample to 16kHz and convert to float32
            audio_data = audioop.ratecv(audio_data, 2, 1, 8000, 16000, None)[0]
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to processing queue if not waiting for response
            if not handler.waiting_for_response:
                handler.audio_queue.put(audio_array)
    
    return '', 200

@app.route('/call-ended', methods=['POST'])
def call_ended():
    """Clean up resources when call ends."""
    call_sid = request.values.get('CallSid')
    if call_sid in active_calls:
        active_calls[call_sid].is_processing = False
        del active_calls[call_sid]
    return '', 200

if __name__ == '__main__':
    socketio.run(app, debug=True, ssl_context="adhoc")