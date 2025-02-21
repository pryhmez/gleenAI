import io
import ffmpeg 
from flask import Flask, request, jsonify, url_for, session, after_this_request, send_from_directory, abort
# from flask_socketio import SocketIO
from flask_sock import Sock
from flask_session import Session
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Start, Connect
from faster_whisper import WhisperModel
from werkzeug.utils import secure_filename
from langchain_core.prompts import PromptTemplate

from config import Config
from ai_helpers import process_initial_message, process_message, initiate_inbound_message
from audio_helpers import text_to_speech, save_audio_file
from yarngpt_helper import text_to_speech_yarngpt

import base64
import os
import time
import redis
from urllib.parse import quote_plus
import uuid
import logging
import threading
import json
import websockets



redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
try:
    val = redis_client.ping()
    logging.info("Connected to Redis")
except redis.ConnectionError as e:
    logging.error(f"Redis connection error: {e}")

# Initialize Faster Whisper
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")

app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins="*")
sock = Sock(app)



app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config.from_object(Config)
# Session configuration
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False  # You can set True for permanent sessions
app.config['SESSION_USE_SIGNER'] = True  # Securely sign the session
# app.config['SESSION_REDIS'] = redis.from_url('redis://redis:6379')
app.config['SESSION_REDIS'] = redis_client
Session(app)
app.logger.setLevel(logging.DEBUG)

client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)



# Stores active stream processors and call sessions
stream_processors = {}
call_sessions = {}

def clean_response(unfiltered_response_text):
    # Remove specific substrings from the response text
    filtered_response_text = unfiltered_response_text.replace("<END_OF_TURN>", "").replace("<END_OF_CALL>", "")
    return filtered_response_text

def delayed_delete(filename, delay=5):
    def attempt_delete():
        time.sleep(delay)
        try:
            os.remove(filename)
            logger.info(f"Successfully deleted temporary audio file: {filename}")
        except Exception as error:
            logger.error(f"Error deleting temporary audio file: {filename} - {error}")

    thread = threading.Thread(target=attempt_delete)
    thread.start()

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


@app.route('/ping', methods=['GET'])
def ping():
    return "pong", 200

# ========================
#  INCOMING CALL HANDLER
# ========================
@app.route("/voice", methods=["POST"])
def voice():
    """Handles an incoming Twilio call and sets up streaming."""
    response = VoiceResponse()
    
    # Start streaming to WebSocket
    start = Start()
    start.stream(url=f"{Config.APP_PUBLIC_URL}/socket.io/")
    response.append(start)

    # Play a greeting message
    response.say("Hello, I'm your AI assistant. How can I help you?")

    return str(response)

# ========================
#  OUTGOING CALL HANDLER
# ========================
@app.route("/start-call", methods=["POST"])
def make_call():
    """Initiates an outbound call to a user."""
    unique_id = str(uuid.uuid4())
    session['conversation_stage_id'] = 1
    message_history = []
    data = request.json

    customer_name = data.get('customer_name', 'Valued Customer')
    customer_phonenumber = data.get('customer_phonenumber', '')
    customer_businessdetails = data.get('customer_businessdetails', 'No details provided.')

    # Process initial message and create audio
    ai_message = process_initial_message(customer_name, customer_businessdetails)
    initial_message = clean_response(ai_message)
    audio_file_path = text_to_speech_yarngpt(initial_message)
    audio_filename = os.path.basename(audio_file_path)

    # Store message history in Redis
    initial_transcript = "Customer Name:" + customer_name + ". Customer's business Details as filled up in the website:" + customer_businessdetails
    message_history.append({"role": "user", "content": initial_transcript})
    message_history.append({"role": "assistant", "content": initial_message})
    redis_client.set(unique_id, json.dumps(message_history))

    # Map call_sid to unique_id for later reference
    call_sessions = {}
    call_sessions['unique_id'] = unique_id

    # Create TwiML response
    response = VoiceResponse()
    
    # Then play the greeting
    response.play(url_for('serve_audio', filename=secure_filename(audio_filename), _external=True))

    # Redirect to start the media stream after playing
    # redirect_url = f"{Config.APP_PUBLIC_GATHER_URL}?CallSid={unique_id}"
    response.redirect(url=url_for('connect_media_stream', unique_id=unique_id, _external=True))


    call = client.calls.create(
        twiml=str(response),
        to=customer_phonenumber,
        from_=Config.TWILIO_FROM_NUMBER,
        method="GET",
        status_callback=Config.APP_PUBLIC_EVENT_URL,
        status_callback_method="POST",
        status_callback_event=["initiated", "ringing", "answered", "completed"],
    )

    # Track call session
    call_sessions[call.sid] = {"status": "initiated"}
    
    return jsonify({"status": "calling", "call_sid": call.sid})


# ========================
#  TWILIO EVENT HOOK
# ========================
@app.route("/event", methods=["POST"])
def twilio_events():
    """Handles Twilio call status events (ringing, in-progress, completed)."""
    call_sid = request.form.get("CallSid")
    call_status = request.form.get("CallStatus")

    if call_sid:
        call_sessions[call_sid] = {"status": call_status}

    print(f"Call {call_sid} status: {call_status}")

    # Handle cleanup on call completion
    if call_status in ["completed", "failed", "busy", "no-answer"]:
        stream_processors.pop(call_sid, None)
        call_sessions.pop(call_sid, None)

    return jsonify({"status": "received"})

# ========================
#  WEBSOCKET HANDLING
# ========================

@app.route('/connect-media-stream', methods=['GET', 'POST'])
def connect_media_stream():
    unique_id = request.args.get('unique_id')

    response = VoiceResponse()
    start = Start()
    start.stream(url=f"{Config.APP_SOCKET_URL}")
    response.append(start)
    response.say("You can start speaking now.")
    response.pause(length=60)
    return str(response)


@sock.route('/media-stream')
def handle_media(ws):
    print("Client connected")

    stream_to_unique_id = {}
    last_valid_message_time = time.time()

    while True:
        message = ws.receive()
        if message:
            data = json.loads(message)
            event = data.get('event')

            if event == 'start':
                stream_sid = data['start']['streamSid']
                call_sid = data['start']['callSid']
                stream_processors[stream_sid] = StreamProcessor(stream_sid)
                unique_id = call_sessions.get('unique_id')
                stream_to_unique_id[stream_sid] = unique_id
                print(f"Started streaming for call {stream_sid}")

            if event == 'media':
                stream_sid = data.get('streamSid')

                if not stream_sid or stream_sid not in stream_processors:
                    logger.warning(f"Invalid stream SID: {stream_sid}")
                    continue

                payload = data.get('media').get('payload')
                if not payload:
                    logger.warning("No payload received in 'media' event")
                    continue

                audio_data = base64.b64decode(payload)
                processor = stream_processors[stream_sid]
                processor.add_audio(audio_data)

                if processor.should_process():
                    audio_chunk = processor.process_buffer()
                    if audio_chunk is not None:
                        logger.debug(f"Audio Chunk Length: {len(audio_chunk)}")
                        logger.debug(f"Audio Chunk Type: {type(audio_chunk)}")

                        try:
                            # Convert audio to WAV format
                            wav_audio = convert_audio_to_wav(audio_chunk)

                            if wav_audio:
                                # Transcribe using Faster Whisper
                                segments, _ = whisper_model.transcribe(wav_audio, beam_size=5)
                                transcription = " ".join([segment.text for segment in segments])

                                if not transcription.strip():
                                    continue

                                logger.debug(f"Transcription: {transcription}")

                                # Retrieve unique_id for message history
                                unique_id = stream_to_unique_id.get(stream_sid)

                                # Process AI response and generate audio
                                message_history_json = redis_client.get(unique_id)
                                message_history = json.loads(message_history_json) if message_history_json else []
                                ai_response_text = process_message(message_history, transcription)
                                response_text = clean_response(ai_response_text)

                                logger.debug(f"AI Response: {response_text}")

                                audio_file_path = text_to_speech_yarngpt(response_text)
                                audio_filename = os.path.basename(audio_file_path)

                                message_history.append({"role": "user", "content": transcription})
                                message_history.append({"role": "assistant", "content": response_text})
                                redis_client.set(unique_id, json.dumps(message_history))

                                print(response_text)

                                processor.last_process_time = time.time()

                        except Exception as e:
                            logger.error(f"Error processing audio chunk: {e}")
                    last_valid_message_time = time.time()

        # Add a condition to break the loop and close the connection if no valid message is received within a certain period
        if time.time() - last_valid_message_time > 30:  # 1-minute timeout
            print("No valid message received in the last 60 seconds. Closing connection.")
            break

    ws.close()


# ========================
#  AUDIO FILE SERVING
# ========================
@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve audio files from directory."""
    directory = 'audio_files'

    print(directory + " playing audion now")

    @after_this_request
    def remove_file(response):
        full_path = os.path.join(directory, filename)
        delayed_delete(full_path)
        return response

    try:
        return send_from_directory(directory, filename)
    except FileNotFoundError:
        logger.error(f"Audio file not found: {filename}")
        abort(404)

# ========================
#  STREAM PROCESSOR CLASS
# ========================    
class StreamProcessor:
    def __init__(self, stream_sid):
        self.stream_sid = stream_sid
        self.audio_buffer = []
        self.last_process_time = time.time()

    def add_audio(self, audio_data):
        self.audio_buffer.append(audio_data)

    def should_process(self):
        # Check if the buffer length is sufficient for processing
        return len(self.audio_buffer) > 50 or time.time() - self.last_process_time > 3

    def process_buffer(self):
        if self.audio_buffer:
            audio_chunk = b"".join(self.audio_buffer)
            self.audio_buffer = []  
            return audio_chunk
        return None


# ========================
#  RUN APP
# ========================
if __name__ == "__main__":
  app.run(host='0.0.0.0', port=5000)

