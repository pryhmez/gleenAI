@sock.route('/media-stream')
def handle_media(ws):
    print("Client connected")
    
    while True:
        message = ws.receive()
        if message:
            data = json.loads(message)
            event = data.get('event')

            if event == 'start':
                stream_sid = data['start']['streamSid']
                stream_processors[stream_sid] = StreamProcessor(stream_sid)
                # print(f"Started streaming for call {stream_sid}")

            if event == 'media':
                stream_sid = data.get('streamSid')
                # print(f"stream is media{stream_sid}")

                
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
                            # Convert to WAV format
                            wav_audio = convert_audio_to_wav(audio_chunk)

                            if wav_audio:
                                # Use Faster Whisper to transcribe
                                # print("yes wav can process")
                                segments, _ = whisper_model.transcribe(wav_audio, beam_size=5)
                                transcription = " ".join([segment.text for segment in segments])
                                print(transcription + "  --------------------------")
                                
                                if not transcription.strip():
                                    # print("not wave")
                                    continue
                                
                                logger.debug(f"Transcription: {transcription}")

                                # Process AI response and generate audio
                                message_history_json = redis_client.get(stream_sid)
                                message_history = json.loads(message_history_json) if message_history_json else []
                                ai_response_text = process_message(message_history, transcription)
                                response_text = clean_response(ai_response_text)

                                logger.debug(f"AI Response: {response_text}")

                                audio_file_path = text_to_speech_yarngpt(response_text)
                                audio_filename = os.path.basename(audio_file_path)

                                message_history.append({"role": "user", "content": transcription})
                                message_history.append({"role": "assistant", "content": response_text})
                                redis_client.set(stream_sid, json.dumps(message_history))

                                print(response_text)

                                processor.last_process_time = time.time()

                        except Exception as e:
                            logger.error(f"Error processing audio chunk: {e}")