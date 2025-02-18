from flask import Flask, request, jsonify, url_for, after_this_request, session, send_from_directory, abort
from flask_session import Session
from flask_socketio import SocketIO, emit
from twilio.twiml.voice_response import VoiceResponse, Start
from twilio.rest import Client
from werkzeug.utils import secure_filename
import os
import logging
import threading
import time
import redis
import json
import uuid
import base64
import numpy as np
from faster_whisper import WhisperModel
from io import BytesIO
from audio_helpers import text_to_speech, save_audio_file
from yarngpt_helper import text_to_speech_yarngpt
from ai_helpers import process_initial_message, process_message, initiate_inbound_message
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)

# Initialize Faster Whisper
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Flask and Session configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_REDIS'] = redis_client
Session(app)

client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)

class StreamProcessor:
    def __init__(self, call_sid):
        self.call_sid = call_sid
        self.buffer = []
        self.lock = threading.Lock()
        self.is_processing = False
        self.audio_queue = []
        self.last_process_time = time.time()
        
    def add_audio(self, audio_data):
        with self.lock:
            self.buffer.append(audio_data)
            
    def should_process(self):
        current_time = time.time()
        buffer_size = len(self.buffer)
        time_since_last = current_time - self.last_process_time
        
        # Process if we have enough data or enough time has passed
        return buffer_size >= 64 or (buffer_size > 0 and time_since_last > 1.0)

    def process_buffer(self):
        with self.lock:
            if not self.buffer:
                return None
                
            # Combine all audio data in buffer
            audio_data = b''.join(self.buffer)
            self.buffer = []
            
            # Convert mulaw audio to float32
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            return audio_np.astype(np.float32) / 32768.0

# Dictionary to store stream processors
stream_processors = {}

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected to WebSocket")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected from WebSocket")

@socketio.on('start')
def handle_start(data):
    """Handle start of a stream"""
    call_sid = data.get('streamSid', '')
    logger.info(f"Starting stream for call {call_sid}")
    stream_processors[call_sid] = StreamProcessor(call_sid)

@socketio.on('media')
def handle_media(data):
    """Handle incoming media packets"""
    try:
        # Extract stream SID and audio data
        stream_sid = data.get('streamSid')
        if not stream_sid or stream_sid not in stream_processors:
            logger.error(f"Invalid stream SID: {stream_sid}")
            return
            
        # Get the payload
        payload = data.get('payload')
        if not payload:
            return
            
        # Decode base64 audio data
        audio_data = base64.b64decode(payload)
        
        # Add to processor
        processor = stream_processors[stream_sid]
        processor.add_audio(audio_data)
        
        # Check if we should process
        if processor.should_process():
            audio_chunk = processor.process_buffer()
            if audio_chunk is not None:
                # Transcribe with faster-whisper
                segments, info = whisper_model.transcribe(audio_chunk, beam_size=5)
                
                # Get transcription text
                transcription = " ".join([segment.text for segment in segments])
                
                if transcription.strip():
                    # Process with AI
                    message_history_json = redis_client.get(processor.call_sid)
                    message_history = json.loads(message_history_json) if message_history_json else []
                    
                    ai_response_text = process_message(message_history, transcription)
                    response_text = clean_response(ai_response_text)
                    
                    # Generate audio response
                    # audio_data = text_to_speech(response_text)
                    # audio_file_path = save_audio_file(audio_data)
                    audio_file_path = text_to_speech_yarngpt(response_text)
                    audio_filename = os.path.basename(audio_file_path)
                    
                    # Create TwiML response
                    response = VoiceResponse()
                    response.play(url_for('serve_audio', filename=audio_filename, _external=True))
                    
                    if "<END_OF_CALL>" in ai_response_text:
                        response.hangup()
                    else:
                        # Resume streaming
                        start = Start()
                        start.stream(url=f"{Config.APP_PUBLIC_URL}/socket.io/")
                        response.append(start)
                    
                    # Update message history
                    message_history.append({"role": "user", "content": transcription})
                    message_history.append({"role": "assistant", "content": response_text})
                    redis_client.set(processor.call_sid, json.dumps(message_history))
                    
                    # Send TwiML response
                    client.calls(processor.call_sid).update(twiml=str(response))
                    
                processor.last_process_time = time.time()
                
    except Exception as e:
        logger.error(f"Error processing media: {str(e)}")

@socketio.on('stop')
def handle_stop(data):
    """Handle end of a stream"""
    stream_sid = data.get('streamSid')
    if stream_sid in stream_processors:
        del stream_processors[stream_sid]
    logger.info(f"Stream ended for {stream_sid}")

@app.route('/start-call', methods=['POST'])
def start_call():
    """Initiate outbound call with streaming"""
    unique_id = str(uuid.uuid4())
    data = request.json
    customer_name = data.get('customer_name', 'Valued Customer')
    customer_phonenumber = data.get('customer_phonenumber', '')
    customer_businessdetails = data.get('customer_businessdetails', '')
    
    # Process initial message
    ai_message = process_initial_message(customer_name, customer_businessdetails)
    initial_message = clean_response(ai_message)
    # audio_data = text_to_speech(initial_message)
    # audio_file_path = save_audio_file(audio_data)
    audio_file_path = text_to_speech_yarngpt(initial_message)
    audio_filename = os.path.basename(audio_file_path)
    
    # Create message history
    message_history = [
        {"role": "user", "content": f"Customer Name: {customer_name}. Business Details: {customer_businessdetails}"},
        {"role": "assistant", "content": initial_message}
    ]
    redis_client.set(unique_id, json.dumps(message_history))
    
    # Create TwiML response
    response = VoiceResponse()
    response.play(url_for('serve_audio', filename=audio_filename, _external=True))
    
    # Start streaming with WebSocket
    start = Start()
    start.stream(url=f"{Config.APP_PUBLIC_URL}/socket.io/")
    response.append(start)
    
    # Make the call
    call = client.calls.create(
        twiml=str(response),
        to=customer_phonenumber,
        from_=Config.TWILIO_FROM_NUMBER,
        method="GET",
        status_callback=Config.APP_PUBLIC_EVENT_URL,
        status_callback_method="POST"
    )
    
    return jsonify({'message': 'Call initiated', 'call_sid': call.sid})

# ... (rest of your routes remain the same)
def clean_response(unfiltered_response_text):
    return unfiltered_response_text.replace("<END_OF_TURN>", "").replace("<END_OF_CALL>", "")

@app.route('/ping', methods=['GET'])
def ping():
    return "pong", 200

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve audio files from directory."""
    directory = 'audio_files'
    
    @after_this_request
    def remove_file(response):
        full_path = os.path.join(directory, filename)
        threading.Timer(5.0, lambda: os.remove(full_path) if os.path.exists(full_path) else None).start()
        return response
    
    try:
        return send_from_directory(directory, filename)
    except FileNotFoundError:
        logger.error(f"Audio file not found: {filename}")
        abort(404)

@app.route('/event', methods=['POST'])
def event():
    """Handle status callback from Twilio calls."""
    call_status = request.values.get('CallStatus', '')
    if call_status in ['completed', 'busy', 'failed']:
        session.pop('message_history', None)
        logger.info(f"Call completed with status: {call_status}")
    return '', 204

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)