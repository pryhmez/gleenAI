from flask import Flask, request, jsonify, url_for, session, after_this_request, send_from_directory, abort

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






# Initialize Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Whisper Model
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")

app = Flask(__name__)
sock = Sock(app)  # WebSocket integration

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)

# Stores active WebSocket connections and call sessions
active_connections = {}
call_sessions = {}

def clean_response(text):
    return text.replace("<END_OF_TURN>", "").replace("<END_OF_CALL>", "")

def delayed_delete(filename, delay=5):
    def attempt_delete():
        time.sleep(delay)
        try:
            os.remove(filename)
            logger.info(f"Deleted temporary audio file: {filename}")
        except Exception as error:
            logger.error(f"Error deleting {filename}: {error}")

    threading.Thread(target=attempt_delete).start()

@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

# ========================
#  INCOMING CALL HANDLER
# ========================
@app.route("/voice", methods=["POST"])
def voice():
    """Handles an incoming Twilio call and sets up streaming."""
    response = VoiceResponse()
    
    # Start streaming to our WebSocket
    start = Start()
    start.stream(url=f"{Config.APP_PUBLIC_URL}/ws")  # Twilio connects to this WebSocket
    response.append(start)

    # Play greeting
    response.say("Hello, I'm your AI assistant. How can I help you?")

    return str(response)

# ========================
#  OUTGOING CALL HANDLER
# ========================
@app.route("/start-call", methods=["POST"])
def make_call():
    """Initiates an outbound call."""
    unique_id = str(uuid.uuid4())
    session['conversation_stage_id'] = 1
    message_history = []
    data = request.json

    customer_name = data.get('customer_name', 'Valued Customer')
    customer_phonenumber = data.get('customer_phonenumber', '')
    customer_businessdetails = data.get('customer_businessdetails', 'No details provided.')

    # Generate AI response
    ai_message = process_initial_message(customer_name, customer_businessdetails)
    initial_message = clean_response(ai_message)
    audio_file_path = text_to_speech_yarngpt(initial_message)

    # Store conversation history in Redis
    initial_transcript = f"Customer Name: {customer_name}. Business Details: {customer_businessdetails}"
    message_history.append({"role": "user", "content": initial_transcript})
    message_history.append({"role": "assistant", "content": initial_message})
    redis_client.set(unique_id, json.dumps(message_history))

    # Twilio call setup
    response = VoiceResponse()
    response.play(url_for('serve_audio', filename=os.path.basename(audio_file_path), _external=True))

    connect = Connect()
    connect.stream(url=f"{Config.APP_SOCKET_URL}")  # WebSocket URL
    response.append(connect)

    call = client.calls.create(
        twiml=str(response),
        to=customer_phonenumber,
        from_=Config.TWILIO_FROM_NUMBER,
        method="GET",
        status_callback=Config.APP_PUBLIC_EVENT_URL,
        status_callback_method="POST",
        status_callback_event=["initiated", "ringing", "answered", "completed"],
    )

    call_sessions[call.sid] = {"status": "initiated"}
    
    return jsonify({"status": "calling", "call_sid": call.sid})

# ========================
#  TWILIO EVENT HOOK
# ========================
@app.route("/event", methods=["POST"])
def twilio_events():
    call_sid = request.form.get("CallSid")
    call_status = request.form.get("CallStatus")

    if call_sid:
        call_sessions[call_sid] = {"status": call_status}

    if call_status in ["completed", "failed", "busy", "no-answer"]:
        active_connections.pop(call_sid, None)
        call_sessions.pop(call_sid, None)

    return jsonify({"status": "received"})

# ========================
#  WEBSOCKET HANDLING
# ========================
@sock.route("/ws")
def websocket_handler(ws):
    """Handles WebSocket connections from Twilio."""
    call_sid = None
    try:
        while True:
            message = ws.receive()
            if not message:
                continue
            
            data = json.loads(message)
            event = data.get("event")
            
            if event == "start":
                call_sid = data.get("streamSid", "")
                if call_sid:
                    active_connections[call_sid] = ws
                    logger.info(f"Started WebSocket streaming for {call_sid}")

            elif event == "media":
                stream_sid = data.get("streamSid")
                payload = data.get("media", {}).get("payload")

                if not stream_sid or stream_sid not in active_connections or not payload:
                    continue
                
                # Decode and process audio
                audio_data = base64.b64decode(payload)
                segments, _ = whisper_model.transcribe(audio_data, beam_size=5)
                transcription = " ".join([segment.text for segment in segments])

                if not transcription.strip():
                    continue  # Ignore empty transcriptions
                
                # Fetch AI response
                message_history_json = redis_client.get(stream_sid)
                message_history = json.loads(message_history_json) if message_history_json else []
                ai_response_text = process_message(message_history, transcription)
                response_text = clean_response(ai_response_text)

                # Convert AI text to speech
                audio_file_path = text_to_speech_yarngpt(response_text)

                # Update message history
                message_history.append({"role": "user", "content": transcription})
                message_history.append({"role": "assistant", "content": response_text})
                redis_client.set(stream_sid, json.dumps(message_history))

                # Send audio URL over WebSocket
                ws.send(json.dumps({"event": "audio_response", "audio_url": f"{Config.APP_PUBLIC_URL}/audio/{os.path.basename(audio_file_path)}"}))

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if call_sid:
            active_connections.pop(call_sid, None)

# ========================
#  RUN APP
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
