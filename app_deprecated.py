from flask import Flask, request, jsonify, url_for, after_this_request, session, send_from_directory, abort
from flask_session import Session
from flask_socketio import SocketIO, emit
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from werkzeug.utils import secure_filename
from langchain_core.prompts import PromptTemplate
import os
from audio_helpers import text_to_speech, save_audio_file
from ai_helpers import process_initial_message, process_message, initiate_inbound_message
from config import Config
import logging
import threading
import time
import redis
import json
from urllib.parse import quote_plus
import uuid
import logging

import logging



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


app = Flask(__name__)
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


def clean_response(unfiltered_response_text):
    # Remove specific substrings from the response text
    filtered_response_text = unfiltered_response_text.replace("<END_OF_TURN>", "").replace("<END_OF_CALL>", "")
    return filtered_response_text

def delayed_delete(filename, delay=5):
    """ Delete the file after a specified delay in seconds. """
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
        
    
@app.route('/start-call', methods=['POST'])
def start_call():
    logger.info("Request recieved")
    """Endpoint to initiate a call."""
    unique_id = str(uuid.uuid4())
    # session['message_history'] = []
    session['conversation_stage_id'] = 1
    message_history = []
    data = request.json
    customer_name = data.get('customer_name', 'Valued Customer')
    customer_phonenumber = data.get('customer_phonenumber', '')
    customer_businessdetails = data.get('customer_businessdetails', 'No details provided.')
    # Call AI_Helpers with customer_name, customer_businessdetails to create the initial response and return the response
    ai_message=process_initial_message(customer_name,customer_businessdetails)
    initial_message=clean_response(ai_message)
    audio_data = text_to_speech(initial_message)
    audio_file_path = save_audio_file(audio_data)
    audio_filename = os.path.basename(audio_file_path)
    #create message history session variable and store the message history [WIP : Enhance this session management]
    initial_transcript = "Customer Name:" + customer_name + ". Customer's business Details as filled up in the website:" + customer_businessdetails
    # session['message_history'].append({"role": "user", "content": initial_transcript})
    # session['message_history'].append({"role": "assistant", "content": initial_message})
    message_history.append({"role": "user", "content": initial_transcript})
    message_history.append({"role": "assistant", "content": initial_message})
    redis_client.set(unique_id, json.dumps(message_history))
    # session.modified = True
    response = VoiceResponse()
    response.play(url_for('serve_audio', filename=secure_filename(audio_filename), _external=True))
    redirect_url = f"{Config.APP_PUBLIC_GATHER_URL}?CallSid={unique_id}"
    # print('=====================' + " " + url_for('serve_audio', filename=secure_filename(audio_filename), _external=True))
    response.redirect(redirect_url)
    call = client.calls.create(
        twiml=str(response),
        to=customer_phonenumber,
        from_=Config.TWILIO_FROM_NUMBER,
        method="GET",
        status_callback=Config.APP_PUBLIC_EVENT_URL,
        status_callback_method="POST"
    )
    return jsonify({'message': 'Call initiated', 'call_sid': call.sid})

@app.route('/gather', methods=['GET', 'POST'])
def gather_input():
    """Endpoint to gather customer's speech input."""
    call_sid = request.args.get('CallSid', 'default_sid')
    resp = VoiceResponse()
    gather = Gather(input='speech', action=url_for('process_speech', CallSid=call_sid), speechTimeout='auto', method="POST")
    resp.append(gather)
    resp.redirect(url_for('gather_input', CallSid=call_sid))  # Redirect to itself to keep gathering if needed
    return str(resp)

@app.route('/gather-inbound', methods=['GET', 'POST'])
def gather_input_inbound():
    """Gathers customer's speech input for both inbound and outbound calls."""
    resp = VoiceResponse()
    print("Initializing for inbound call...")
    unique_id = str(uuid.uuid4())
    # session['message_history'] = []
    session['conversation_stage_id'] = 1
    message_history = []
    agent_response= initiate_inbound_message()
    audio_data = text_to_speech(agent_response)
    audio_file_path = save_audio_file(audio_data)
    audio_filename = os.path.basename(audio_file_path)
    resp.play(url_for('serve_audio', filename=secure_filename(audio_filename), _external=True))
    message_history.append({"role": "assistant", "content": agent_response})
    redis_client.set(unique_id, json.dumps(message_history))
    resp.redirect(url_for('gather_input', CallSid=unique_id))
    return str(resp)


@app.route('/process-speech', methods=['POST'])
def process_speech():
    """Processes customer's speech input and generates a response."""
    speech_result = request.values.get('SpeechResult', '').strip()
    call_sid = request.args.get('CallSid', 'default_sid')
    print(call_sid)
    message_history_json = redis_client.get(call_sid)
    message_history = json.loads(message_history_json) if message_history_json else []

    # message_history = session.get('message_history', [])

    # Fetch AI Response based on tool calling wherever required.
    ai_response_text = process_message(message_history,speech_result)
    response_text = clean_response(ai_response_text)
    audio_data = text_to_speech(response_text)
    audio_file_path = save_audio_file(audio_data)
    audio_filename = os.path.basename(audio_file_path)

    resp = VoiceResponse()
    resp.play(url_for('serve_audio', filename=secure_filename(audio_filename), _external=True))
    if "<END_OF_CALL>" in ai_response_text:
        print("The conversation has ended.")
        resp.hangup()
    resp.redirect(url_for('gather_input', CallSid=call_sid))
    message_history.append({"role": "user", "content": speech_result})
    message_history.append({"role": "assistant", "content": response_text})
    redis_client.set(call_sid, json.dumps(message_history))
    # session['message_history'] = message_history
    # session.modified = True
    return str(resp)

@app.route('/event', methods=['POST'])
def event():
    """Handle status callback from Twilio calls."""
    call_status = request.values.get('CallStatus', '')
    print(call_status)
    if call_status in ['completed', 'busy', 'failed']:
        session.pop('message_history', None)  # Clean up session after the call
        logger.info(f"Call completed with status: {call_status}")
    return '', 204

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
