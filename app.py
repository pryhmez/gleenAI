import io
import ffmpeg 
import torch
from flask import Flask, request, jsonify, url_for, session, after_this_request, send_from_directory, abort
# from flask_socketio import SocketIO
from flask_sock import Sock
from flask_session import Session
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Start, Connect, Stream
from faster_whisper import WhisperModel
from werkzeug.utils import secure_filename
from langchain_core.prompts import PromptTemplate

from config import Config
from ai_helpers import process_initial_message, process_message, initiate_inbound_message
from audio_helpers import text_to_speech, save_audio_file, convert_audio_to_wav, convert_audio_to_pcm
from yarngpt_helper import text_to_speech_yarngpt
from stream_processor import StreamProcessor

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
import numpy as np
import asyncio


redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Directory to save the final combined audio file
audio_files_directory = "audio_files"
os.makedirs(audio_files_directory, exist_ok=True)

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

    print("Initializing for inbound call...")
    unique_id = str(uuid.uuid4())
    
    # Start streaming to WebSocket
    start = Start()
    start.stream(url=f"{Config.APP_PUBLIC_URL}/socket.io/")
    response.append(start)

    # Play a greeting message
    response.say("Hello, I'm your AI assistant. How can I help you?")

    return str(response)

    # # Track call session
    # call_sessions[call.sid] = {
    #     "unique_id": unique_id,
    #     "status": "initiated"
    # }

    # session['conversation_stage_id'] = 1
    # message_history = []
    # agent_response= initiate_inbound_message()
    # audio_data = text_to_speech(agent_response)
    # audio_file_path = save_audio_file(audio_data)
    # audio_filename = os.path.basename(audio_file_path)
    # resp.play(url_for('serve_audio', filename=secure_filename(audio_filename), _external=True))
    # message_history.append({"role": "assistant", "content": agent_response})
    # redis_client.set(unique_id, json.dumps(message_history))
    # resp.redirect(url_for('gather_input', CallSid=unique_id))
    # return str(resp)

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
    # audio_file_path = text_to_speech_yarngpt(initial_message)
    audio_data = text_to_speech(initial_message)
    audio_file_path = save_audio_file(audio_data)
    audio_filename = os.path.basename(audio_file_path)

    # Store message history in Redis
    initial_transcript = "Customer Name:" + customer_name + ". Customer's business Details as filled up in the website:" + customer_businessdetails
    message_history.append({"role": "user", "content": initial_transcript})
    message_history.append({"role": "assistant", "content": initial_message})
    redis_client.set(unique_id, json.dumps(message_history))

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
    call_sessions[call.sid] = {
        "unique_id": unique_id,
        "status": "initiated"
    }
    
    return jsonify({"status": "calling", "call_sid": call.sid})


# ========================
#  TWILIO EVENT HOOK
# ========================
@app.route("/event", methods=["POST"])
def twilio_events():
    """Handles Twilio call status events (ringing, in-progress, completed)."""
    call_sid = request.form.get("CallSid")
    call_status = request.form.get("CallStatus")

    # if call_sid:
    #     call_sessions[call_sid] = {"status": call_status}

    if call_sid in call_sessions:
        call_sessions[call_sid]["status"] = call_status  # Update status

    print(f"Call {call_sid} status: {call_status}")

    # Handle cleanup on call completion
    if call_status in ["completed", "failed", "busy", "no-answer"]:
        #need to pop this correctly
        stream_processors.pop(call_sid, None)
        call_sessions.pop(call_sid, None)

    return jsonify({"status": "received"})

# ========================
#  WEBSOCKET HANDLING
# ========================


@app.route('/connect-media-stream', methods=['GET', 'POST'])
def connect_media_stream():
    unique_id = request.args.get('unique_id')

    logger.debug(f"connecting to socket{Config.APP_SOCKET_URL}")

    response = VoiceResponse()
    start = Start()
    stream = Stream(url=f"{Config.APP_SOCKET_URL}")
    stream.parameter(unique_id=unique_id)
    start.append(stream)
    response.append(start)
    response.pause(length=120)
    return str(response)


@sock.route('/media-stream')
def handle_media(ws):
    print("Client connected")

    stream_to_unique_id = {}

    while True:
        try:
            message = ws.receive()
            if message:
                data = json.loads(message)
                event = data.get('event')

                if event == 'start':
                    stream_sid = data['start']['streamSid']
                    call_sid = data['start']['callSid']
                    print(data)
                   
                    call_data = call_sessions.get(call_sid)  # Fetch call session
                    if not call_data:
                        print(f"Warning: No session found for CallSid {call_sid}")
                        return
                    
                    unique_id = call_data.get("unique_id")
                    print(f"Using existing unique_id: {unique_id}")

                    stream_processors[stream_sid] = StreamProcessor(stream_sid)
                    stream_to_unique_id[stream_sid] = unique_id
                    print(f"Started streaming for call {stream_sid}")

                elif event == 'media':
                    stream_sid = data.get('streamSid')

                    if not stream_sid or stream_sid not in stream_processors:
                        logger.warning(f"Invalid stream SID: {stream_sid}")
                        continue

                    payload = data.get('media').get('payload')
                    if not payload:
                        logger.warning("No payload received in 'media' event")
                        continue

                    try:
                        audio_data = base64.b64decode(payload)
                    except Exception as e:
                        logger.error(f"Error decoding base64 payload: {e}")
                        continue

                    processor = stream_processors[stream_sid]
                    processor.add_audio(audio_data)

                    # Check if transcription is available
                    transcription = processor.get_transcription()
                    if transcription:
                        # Retrieve unique_id for message history
                        unique_id = stream_to_unique_id.get(stream_sid)
                        print(unique_id)
                        # Process AI response and generate audio
                        message_history_json = redis_client.get(unique_id)
                        message_history = json.loads(message_history_json) if message_history_json else []
                        # print(message_history)
                        ai_response_text = process_message(message_history, transcription)
                        # print(f"ai_response_text: {ai_response_text}")
                        response_text = clean_response(ai_response_text)

                        logger.debug(f"AI Response: {response_text}")

                        torch.cuda.empty_cache()


                         # Generate speech from AI response
                        audio_data = text_to_speech(response_text)
                        audio_file_path = save_audio_file(audio_data)
                        audio_filename = os.path.basename(audio_file_path)

                        # pcm_audio = convert_audio_to_pcm(audio_data)

                        # Send back to the user via WebSocket
                        # send_audio_to_twilio(ws, pcm_audio)
                        # Send audio response back to the user if needed 
                        response = VoiceResponse()
                        response.play(url_for('serve_audio',  filename=secure_filename(audio_filename), _external=True))    
                        start = Start()
                        start.stream(url=f"{Config.APP_SOCKET_URL}")
                        response.append(start)

                        # Update Twilio call
                        client.calls(call_sid).update(twiml=str(response))             

                        message_history.append({"role": "user", "content": transcription})
                        message_history.append({"role": "assistant", "content": response_text})
                        redis_client.set(unique_id, json.dumps(message_history))

                        print(response_text)


                        

                else:
                    # Handle other events if necessary
                    pass

        except Exception as e:
            logger.error(f"Error in WebSocket handling: {e}")
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

def send_audio_to_twilio(ws, pcm_audio, chunk_size=320):
    """Sends PCM audio data in 20ms chunks to Twilio via WebSocket"""
    total_size = len(pcm_audio)
    num_chunks = total_size // chunk_size

    for i in range(num_chunks):
        chunk = pcm_audio[i * chunk_size:(i + 1) * chunk_size]
        ws.send(json.dumps({
            "event": "media",
            "media": {
                "payload": base64.b64encode(chunk).decode("utf-8")
            }
        }))
        asyncio.sleep(0.02)  # Non-blocking delay




# ========================
#  RUN APP
# ========================
if __name__ == "__main__":
  app.run(host='0.0.0.0', port=5000)

