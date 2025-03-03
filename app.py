import io
import ffmpeg
import torch
import base64
import os
import time
# import redis
from common.redis_client import redis_client
import uuid
import logging
import threading
import json
import numpy as np
import asyncio
import psutil
import websockets

from urllib.parse import quote_plus
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketState
from starlette.middleware.sessions import SessionMiddleware
from fastapi import Response

from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Start, Connect, Stream
from faster_whisper import WhisperModel
from langchain_core.prompts import PromptTemplate

from config import Config
from ai_helpers import process_initial_message, process_message, initiate_inbound_message
from audio_helpers import text_to_speech, save_audio_file, convert_audio_to_wav, convert_audio_to_pcm, stream_audio_to_twilio
from yarngpt_helper import text_to_speech_yarngpt
from stream_processor import StreamProcessor

# Initialize app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session Middleware
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection
# redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
# try:
#     redis_client.ping()
#     logging.info("Connected to Redis")
# except redis.ConnectionError as e:
#     logging.error(f"Redis connection error: {e}")

# Twilio Client
client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)

# Initialize Faster Whisper
whisper_model = WhisperModel("medium.en", device="cuda", compute_type="float16")

# Store active stream processors and call sessions
stream_processors = {}
call_sessions = {}
audio_play_start_time = time.time()

# Ensure audio directory exists
audio_files_directory = "audio_files"
os.makedirs(audio_files_directory, exist_ok=True)

def clean_response(response_text):
    return response_text.replace("<END_OF_TURN>", "").replace("<END_OF_CALL>", "")

def delayed_delete(filename, delay=5):
    def attempt_delete():
        time.sleep(delay)
        try:
            os.remove(filename)
            logger.info(f"Deleted temporary audio file: {filename}")
        except Exception as error:
            logger.error(f"Error deleting file {filename}: {error}")

    threading.Thread(target=attempt_delete).start()

@app.get("/ping")
def ping():
    return "pong"

@app.post("/voice-incoming")
async def voice(request: Request):
    form_data = await request.form()
    unique_id = str(uuid.uuid4())
    call_sid = form_data.get("CallSid")
    print(f"inbound call imminent....call sid: {call_sid}")
    message_history = []
    agent_response= initiate_inbound_message()
    initial_message = clean_response(agent_response)
    audio_data = text_to_speech(initial_message)
    audio_file_path = save_audio_file(audio_data)
    audio_filename = os.path.basename(audio_file_path)

    conversation_data = {
    "conversation_stage_id": 1,  # Setting initial stage
    "messages": [{"role": "assistant", "content": initial_message}]
    }
    redis_client.set(unique_id, json.dumps(conversation_data))
    call_sessions[call_sid] = {"unique_id": unique_id, "status": "answered"}

    response = VoiceResponse()
    response.play(f"{Config.APP_PUBLIC_URL}/audio/{audio_filename}")
    connect = Connect()
    stream = Stream(url=f"{Config.APP_SOCKET_URL}")
    stream.parameter(name="unique_id", value=unique_id)
    connect.append(stream)
    response.append(connect)
    response.pause(length=10)
    return Response(content=str(response), media_type="application/xml")

@app.post("/start-call")
async def make_call(request: Request):
    data = await request.json()
    unique_id = str(uuid.uuid4())
    customer_name = data.get("customer_name", "Valued Customer")
    customer_phonenumber = data.get("customer_phonenumber", "")
    customer_businessdetails = data.get("customer_businessdetails", "No details provided.")

    ai_message = process_initial_message(customer_name, customer_businessdetails)
    initial_message = clean_response(ai_message)
    audio_data = text_to_speech(initial_message)
    audio_file_path = save_audio_file(audio_data)
    audio_filename = os.path.basename(audio_file_path)

    conversation_data = {
    "conversation_stage_id": 1,  # Setting initial stage
    "messages": [{"role": "assistant", "content": initial_message}]
    }
    redis_client.set(unique_id, json.dumps(conversation_data))

    response = VoiceResponse()
    response.play(f"{Config.APP_PUBLIC_URL}/audio/{audio_filename}")
    redirect_url = f"{Config.APP_PUBLIC_URL}/connect-media-stream?unique_id={unique_id}"
    response.redirect(redirect_url)

    call = client.calls.create(
        twiml=str(response),
        to=customer_phonenumber,
        from_=Config.TWILIO_FROM_NUMBER,
        method="GET",
        status_callback=Config.APP_PUBLIC_EVENT_URL,
        status_callback_method="POST",
        status_callback_event=["initiated", "ringing", "answered", "completed"],
    )
    
    call_sessions[call.sid] = {"unique_id": unique_id, "status": "initiated"}
    return {"status": "calling", "call_sid": call.sid}

@app.post("/event")
async def twilio_events(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    call_status = form.get("CallStatus")
    
    if call_sid in call_sessions:
        call_sessions[call_sid]["status"] = call_status
    print(f"Call {call_sid} status: {call_status}")
    if call_status in ["completed", "failed", "busy", "no-answer"]:
        stream_processors.pop(call_sid, None)
        call_sessions.pop(call_sid, None)
    
    return {"status": "received"}

@app.api_route("/connect-media-stream", methods=["GET", "POST"])
async def connect_media_stream(request: Request):
    unique_id = request.query_params.get("unique_id")
    if not unique_id:
        return PlainTextResponse("Missing unique_id", status_code=400)

    logger.debug(f"Connecting to socket {Config.APP_SOCKET_URL}")

    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=f"{Config.APP_SOCKET_URL}")
    stream.parameter(name="unique_id", value=unique_id)
    connect.append(stream)
    response.append(connect)
    response.pause(length=10)
    return Response(content=str(response), media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    print("connecting to stream=================")
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            event = data.get("event")
            if event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"]["callSid"]
                unique_id = call_sessions.get(call_sid, {}).get("unique_id", str(uuid.uuid4()))
                stream_processors[stream_sid] = StreamProcessor(stream_sid)
            elif event == "media":
                stream_sid = data.get("streamSid")
                payload = base64.b64decode(data["media"]["payload"])
                processor = stream_processors.get(stream_sid)
                if processor:
                    processor.add_audio(payload)
                    transcription = processor.get_transcription()
                    if transcription:
                      conversation_data = {
                        "conversation_stage_id": 1,  # Default initial stage
                        "messages": []
                      }
                      stored_data = redis_client.get(unique_id)
                      if stored_data:
                          conversation_data = json.loads(stored_data)
                      message_history = conversation_data.get("messages", [])
                      response_text = clean_response(process_message(message_history, transcription, unique_id))
                      await stream_audio_to_twilio(websocket, stream_sid, response_text)
                      message_history.append({"role": "user", "content": transcription})
                      message_history.append({"role": "assistant", "content": response_text})
                      conversation_data["messages"] = message_history
                      redis_client.set(unique_id, json.dumps(conversation_data))
    except WebSocketDisconnect:
      stream_sid = websocket.scope.get("stream_sid")  # Store this earlier in `start` event
      if stream_sid:
          stream_processors.pop(stream_sid, None)
      await websocket.close()

@app.get("/audio/{filename}")
def serve_audio(filename: str, background_tasks: BackgroundTasks):
    file_path = os.path.join("audio_files", filename)

    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"message": "File not found"})
    background_tasks.add_task(delayed_delete, file_path)
    return FileResponse(file_path, media_type="audio/wav")
