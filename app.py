"""
app.py — AgriVoice Real-Time Streaming Server (Exotel AgentStream)

Architecture:
  - Exotel AgentStream (WebSocket) -> Decodes 8kHz PCM
  - Streams directly to Sarvam ASR (WebSocket)
  - On transcript -> router.py -> Sarvam TTS (HTTP)
  - Transcodes TTS audio via pydub -> 8kHz PCM
  - Streams chunks back to Exotel.
  - Handles barge-in via Exotel 'clear' events.

Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import io
import sys
import json
import base64
import asyncio
import traceback
import websockets
from datetime import datetime, timezone
from pydub import AudioSegment
from dotenv import load_dotenv

# Fix Windows console encoding for non-ASCII output
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import requests

# µ-law decoding support (Exotel may send mulaw-encoded audio)
try:
    import audioop
except ImportError:
    audioop = None
    print("[STARTUP] audioop not available — install 'audioop-lts' if Exotel sends mulaw audio")

# ── Business Logic & State ──
from router import process_farmer_query

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
EXOTEL_ACCOUNT_SID = os.getenv("EXOTEL_ACCOUNT_SID")

app = FastAPI(title="AgriVoice Super-Agent (AgentStream)", version="3.0")

# CORS — Allow Base44 frontend and any origin to call our API
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# SARVAM TTS & LANGUAGE HELPERS
# ============================================================

def detect_language_code(text: str) -> str:
    """Detects the language of the generated text."""
    SCRIPT_TO_LANG = {
        range(0x0900, 0x0980): "hi-IN",
        range(0x0980, 0x0A00): "bn-IN",
        range(0x0B80, 0x0C00): "ta-IN",
        range(0x0C00, 0x0C80): "te-IN",
        range(0x0C80, 0x0D00): "kn-IN",
        range(0x0D00, 0x0D80): "ml-IN",
        range(0x0A00, 0x0A80): "pa-IN",
    }
    for char in text:
        cp = ord(char)
        for script_range, code in SCRIPT_TO_LANG.items():
            if cp in script_range: return code
    return "en-IN"

def generate_tts_audio(text: str, language_code: str = "hi-IN") -> bytes | None:
    """Fetches high-quality WAV audio from Sarvam Bulbul v2."""
    print(f"[TTS] Generating audio for: {text[:60]}...")
    try:
        resp = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            json={
                "inputs": [text],
                "target_language_code": language_code,
                "speaker": "anushka",
                "model": "bulbul:v2",
                "enable_preprocessing": True,
            },
            headers={"Content-Type": "application/json", "api-subscription-key": SARVAM_API_KEY},
            timeout=30,
        )
        if resp.ok:
            audio_b64 = resp.json().get("audios", [""])[0]
            if audio_b64:
                return base64.b64decode(audio_b64)
        print(f"[TTS Error] {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[TTS Exception] {e}")
    return None

def convert_to_exotel_pcm(audio_bytes: bytes) -> bytes:
    """
    Converts incoming audio (WAV/MP3) to Exotel's strict PCM format:
    16-bit, 8000Hz, Mono, Little-Endian.
    Requires FFmpeg installed on the host.
    """
    print(f"[AUDIO] Transcoding {len(audio_bytes)} bytes to 8kHz PCM...")
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    
    # Resample to 8kHz, Mono, 16-bit
    audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    pcm_data = audio.raw_data
    print(f"[AUDIO] Transcoding complete: {len(pcm_data)} bytes of raw PCM.")
    return pcm_data


# ============================================================
# SARVAM ASR (REST API)
# ============================================================

def transcribe_audio_rest(pcm_bytes: bytes) -> str:
    """Sends recorded 8kHz PCM audio to Sarvam ASR REST API."""
    if not pcm_bytes:
        return ""
        
    import wave
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm_bytes)
    wav_io.seek(0)
    
    try:
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": ("audio.wav", wav_io, "audio/wav")},
            data={"model": "saaras:v3"},
            timeout=10
        )
        if resp.ok:
            return resp.json().get("transcript", "").strip()
        else:
            print(f"[ASR Error] {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[ASR Exception] {e}")
    return ""

# ============================================================
# GREETING & AUDIO HELPERS
# ============================================================

_greeting_pcm_cache = None

def _generate_greeting_sync():
    """Synchronously generate the welcome greeting PCM audio (called on startup)."""
    global _greeting_pcm_cache
    greeting = "வணக்கம்! நான் AgriVoice, உங்கள் விவசாய உதவியாளர். உங்கள் கேள்வியைக் கேளுங்கள்."
    print("[STARTUP] Generating greeting audio...", flush=True)
    wav_audio = generate_tts_audio(greeting, language_code="ta-IN")
    if wav_audio:
        _greeting_pcm_cache = convert_to_exotel_pcm(wav_audio)
        print(f"[STARTUP] [OK] Greeting audio ready ({len(_greeting_pcm_cache)} bytes)", flush=True)
    else:
        print("[STARTUP] [WARN] Failed to generate greeting audio (TTS error)", flush=True)


async def send_greeting(websocket: WebSocket, sid: str, seq_ref: list):
    """Stream the cached welcome greeting back to Exotel to keep the call alive."""
    try:
        if not _greeting_pcm_cache:
            print("[GREETING] [WARN] No cached greeting available, skipping.")
            return

        chunk_size = 3200  # 200ms at 8kHz 16-bit mono
        pcm = _greeting_pcm_cache
        print(f"[GREETING] Streaming welcome message ({len(pcm)} bytes)...")

        for i in range(0, len(pcm), chunk_size):
            chunk = pcm[i:i + chunk_size]
            b64_payload = base64.b64encode(chunk).decode("utf-8")
            chunk_idx = int(i / chunk_size) + 1
            timestamp_ms = int((i / chunk_size) * 200)
            
            await websocket.send_json({
                "event": "media",
                "stream_sid": sid,
                "media": {
                    "payload": b64_payload
                }
            })
            await asyncio.sleep(0.18)  # Slightly under real-time to buffer ahead

        print("[GREETING] [OK] Welcome message sent.")
    except Exception as e:
        # Ignore socket closed exceptions if Exotel hung up
        if "socket.send() raised exception" not in str(e):
            print(f"[GREETING ERROR] {e}")


def decode_exotel_audio(raw_bytes: bytes, encoding: str) -> bytes:
    """Decode Exotel audio to raw 16-bit PCM based on the stream's encoding format."""
    if "mulaw" in encoding or "mu-law" in encoding:
        if audioop:
            return audioop.ulaw2lin(raw_bytes, 2)
        else:
            print("[AUDIO] [WARN] mulaw audio received but audioop not available!")
            return raw_bytes
    # audio/x-l16 or unknown — assume raw 16-bit PCM already
    return raw_bytes


@app.on_event("startup")
async def preload_greeting():
    """Pre-generate the greeting audio in the background (non-blocking startup)."""
    async def _bg_generate():
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, _generate_greeting_sync)
        except Exception as e:
            print(f"[STARTUP] [WARN] Greeting generation failed: {e}", flush=True)
    asyncio.create_task(_bg_generate())


@app.get("/debug")
async def debug_status():
    """Quick diagnostic to verify greeting cache and server state."""
    return {
        "greeting_cached": _greeting_pcm_cache is not None,
        "greeting_bytes": len(_greeting_pcm_cache) if _greeting_pcm_cache else 0,
        "ffmpeg": "available" if os.popen("ffmpeg -version").read() else "missing",
    }


# ============================================================
# EXOTEL AGENTSTREAM WEBSOCKET
# ============================================================
import numpy as np

@app.websocket("/exotel-stream")
async def handle_exotel_stream(websocket: WebSocket):
    """
    Main WebSocket bridge between Exotel AgentStream and our bot logic.
    Uses Voice Activity Detection (VAD) to buffer audio and send to REST API.
    """
    await websocket.accept()
    print("\n" + "=" * 60)
    print("[EXOTEL STREAM] [OK] Connection established")
    print("=" * 60)

    # Session State
    stream_sid = None
    call_sid = None
    playback_task = None
    audio_encoding = "audio/x-l16"
    seq_num_ref = [1]
    farmer_lang_code = ["en-IN"]  # Track farmer's detected language
    awaiting_feedback = [False]    # True when waiting for spoken feedback

    # VAD State
    audio_buffer = []
    is_speaking = False
    silence_chunks = 0
    barge_in_counter = 0  # Tracks consecutive loud chunks during playback
    SILENCE_THRESHOLD_RMS = 150       # RMS threshold for normal speech detection
    BARGE_IN_THRESHOLD_RMS = 400      # Higher threshold during playback (avoids echo/noise)
    BARGE_IN_CHUNKS_REQUIRED = 5      # Need 5 consecutive loud chunks (~0.5s) to interrupt
    SILENCE_CHUNKS_LIMIT = 8          # ~0.8s of silence to end speech (reduced for faster response)

    async def _process_buffer(pcm_buffer: bytes):
        nonlocal playback_task
        print("[VAD] 🛑 Silence detected, processing speech...")
        transcript = await asyncio.to_thread(transcribe_audio_rest, pcm_buffer)
        
        if transcript and len(transcript.strip()) > 2:
            print(f"[SARVAM ASR] [OK] Final Transcript: {transcript}")
            
            # Check if we are waiting for feedback
            if awaiting_feedback[0]:
                print(f"[FEEDBACK] Farmer feedback received: '{transcript}'")
                awaiting_feedback[0] = False
                # Play a thank-you and end the call
                from router import translate_to_indian_language
                goodbye_text = "Thank you for your valuable feedback. Have a great day. Goodbye!"
                if farmer_lang_code[0] != "en-IN":
                    goodbye_text = translate_to_indian_language(goodbye_text, farmer_lang_code[0])
                goodbye_lang = detect_language_code(goodbye_text)
                try:
                    goodbye_wav = generate_tts_audio(goodbye_text, language_code=goodbye_lang)
                    if goodbye_wav:
                        goodbye_pcm = convert_to_exotel_pcm(goodbye_wav)
                        chunk_size = 3200
                        for i in range(0, len(goodbye_pcm), chunk_size):
                            chunk = goodbye_pcm[i:i + chunk_size]
                            b64_payload = base64.b64encode(chunk).decode("utf-8")
                            await websocket.send_json({
                                "event": "media",
                                "stream_sid": stream_sid,
                                "media": {"payload": b64_payload}
                            })
                            await asyncio.sleep(0.18)
                    print("[GOODBYE] Thank-you message sent. Ending call.")
                except Exception as e:
                    print(f"[GOODBYE ERROR] {e}")
                
                # Wait for the goodbye audio to actually finish playing on the phone line
                # 8kHz 16-bit mono = 16000 bytes per second
                audio_duration = len(goodbye_pcm) / 16000 if 'goodbye_pcm' in locals() else 3
                await asyncio.sleep(audio_duration + 0.5)  # Add 0.5s buffer
                
                # Close the WebSocket to end the call
                try:
                    await websocket.close()
                except Exception:
                    pass
                return
            
            # Normal flow: Barge-in and process query
            if playback_task and not playback_task.done():
                print("[BARGE-IN] Interrupting current playback...")
                playback_task.cancel()
                if stream_sid:
                    try:
                        await websocket.send_json({"event": "clear", "stream_sid": stream_sid})
                    except Exception:
                        pass
                    
            # Process query and play answer
            playback_task = asyncio.create_task(
                process_and_play(transcript, websocket, stream_sid, seq_num_ref, farmer_lang_code)
            )
        else:
            print("[SARVAM ASR] No speech recognized.")

    try:
        while True:
            data = await websocket.receive_json()
            event = data.get("event")

            if event == "connected":
                print("[EXOTEL STREAM] Received 'connected' event.")

            elif event == "start":
                start_data = data.get("start", {})
                stream_sid = start_data.get("stream_sid")
                call_sid = start_data.get("call_sid")
                media_format = start_data.get("media_format", {})
                audio_encoding = media_format.get("encoding", "audio/x-l16")
                print(f"[EXOTEL STREAM] 'start' | Stream: {stream_sid} | Call: {call_sid} | Encoding: {audio_encoding}")

                # Send greeting immediately to keep the call alive
                asyncio.create_task(send_greeting(websocket, stream_sid, seq_num_ref))

            elif event == "media":
                payload = data.get("media", {}).get("payload")
                if payload:
                    raw_bytes = base64.b64decode(payload)
                    pcm_bytes = decode_exotel_audio(raw_bytes, audio_encoding)
                    
                    # Voice Activity Detection (VAD) via RMS volume
                    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
                    if len(arr) > 0:
                        rms = float(np.sqrt(np.mean(np.square(arr.astype(np.float32)))))
                        
                        # Check if bot is currently playing audio back
                        bot_is_playing = playback_task and not playback_task.done()
                        
                        # Use a HIGHER threshold during playback to avoid echo/noise triggers
                        current_threshold = BARGE_IN_THRESHOLD_RMS if bot_is_playing else SILENCE_THRESHOLD_RMS
                        
                        if rms > current_threshold:
                            if bot_is_playing:
                                # During playback: require SUSTAINED loud speech before interrupting
                                barge_in_counter += 1
                                if barge_in_counter >= BARGE_IN_CHUNKS_REQUIRED:
                                    print("[BARGE-IN] Farmer is speaking! Interrupting playback...")
                                    playback_task.cancel()
                                    try:
                                        await websocket.send_json({"event": "clear", "stream_sid": stream_sid})
                                    except Exception:
                                        pass
                                    barge_in_counter = 0
                                    is_speaking = True
                                    silence_chunks = 0
                                    audio_buffer.append(pcm_bytes)
                            else:
                                # Not playing: normal VAD behavior
                                if not is_speaking:
                                    print("[VAD] 🎤 Speech detected, starting buffer...")
                                is_speaking = True
                                silence_chunks = 0
                                barge_in_counter = 0
                                audio_buffer.append(pcm_bytes)
                        else:
                            barge_in_counter = 0  # Reset barge-in counter on silence
                            if is_speaking:
                                audio_buffer.append(pcm_bytes)
                                silence_chunks += 1
                                if silence_chunks >= SILENCE_CHUNKS_LIMIT:
                                    final_audio = b"".join(audio_buffer)
                                    audio_buffer = []
                                    is_speaking = False
                                    silence_chunks = 0
                                    
                                    # Process if it's long enough (avoid short noise blips)
                                    if len(final_audio) > 8000: # at least ~0.5 second of audio
                                        asyncio.create_task(_process_buffer(final_audio))

            elif event == "dtmf":
                digit = data.get("dtmf", {}).get("digit")
                print(f"[EXOTEL STREAM] [DTMF] Digit pressed: {digit}")
                # Farmer pressed a key to end the call
                if playback_task and not playback_task.done():
                    playback_task.cancel()
                
                # Ask for feedback in the farmer's native language
                print("[FEEDBACK] Asking farmer for feedback...")
                from router import translate_to_indian_language
                feedback_prompt = "Thank you for using AgriVoice! Before you go, please tell us how was your experience? Your feedback helps us improve."
                if farmer_lang_code[0] != "en-IN":
                    feedback_prompt = translate_to_indian_language(feedback_prompt, farmer_lang_code[0])
                
                async def play_feedback():
                    try:
                        feedback_lang = detect_language_code(feedback_prompt)
                        feedback_wav = generate_tts_audio(feedback_prompt, language_code=feedback_lang)
                        if feedback_wav:
                            feedback_pcm = convert_to_exotel_pcm(feedback_wav)
                            chunk_size = 3200
                            for i in range(0, len(feedback_pcm), chunk_size):
                                chunk = feedback_pcm[i:i + chunk_size]
                                b64_payload = base64.b64encode(chunk).decode("utf-8")
                                await websocket.send_json({
                                    "event": "media",
                                    "stream_sid": stream_sid,
                                    "media": {"payload": b64_payload}
                                })
                                await asyncio.sleep(0.18)
                        print("[FEEDBACK] Feedback prompt sent. Waiting for farmer's response...")
                    except asyncio.CancelledError:
                        print("[FEEDBACK] Prompt cancelled by barge-in.")
                    except Exception as e:
                        print(f"[FEEDBACK ERROR] {e}")

                # Run the feedback prompt in the background so we can listen to the farmer immediately
                playback_task = asyncio.create_task(play_feedback())
                # Set the flag so the next VAD capture is treated as feedback
                awaiting_feedback[0] = True

            elif event == "stop":
                print("[EXOTEL STREAM] Received 'stop' event. Terminating stream.")
                break

    except WebSocketDisconnect:
        print("[EXOTEL STREAM] [ERR] WebSocket disconnected by Exotel.")
    except Exception as e:
        print(f"[STREAM ERROR] {e}")
        traceback.print_exc()
    finally:
        if playback_task and not playback_task.done():
            playback_task.cancel()
        print("[EXOTEL STREAM] Session ended.\n")


async def process_and_play(transcript: str, websocket: WebSocket, stream_sid: str, seq_ref: list, farmer_lang_ref: list = None):
    """
    Passes transcript to the router, generates TTS, transcodes to 8kHz PCM,
    and streams back to Exotel in chunks.
    """
    try:
        # 1. Routing
        print("[ROUTER] Analyzing farmer query...")
        answer, intent = process_farmer_query(transcript)
        print(f"[ROUTER] [OK] Answer: {answer}")

        # 2. Text to Speech
        lang_code = detect_language_code(answer)
        
        # Track the farmer's language for future use (feedback/goodbye)
        if farmer_lang_ref and lang_code != "en-IN":
            farmer_lang_ref[0] = lang_code
        
        wav_audio = generate_tts_audio(answer, language_code=lang_code)

        if not wav_audio:
            print("[ERROR] Failed to generate TTS audio.")
            return

        # 3. Transcode to 8kHz Raw PCM
        pcm_data = convert_to_exotel_pcm(wav_audio)

        # 4. Stream back to Exotel in chunks
        chunk_size = 3200
        
        print(f"[PLAYBACK] Streaming {len(pcm_data)} bytes of PCM back to Exotel...")
        
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i:i + chunk_size]
            b64_payload = base64.b64encode(chunk).decode("utf-8")
            
            await websocket.send_json({
                "event": "media",
                "stream_sid": stream_sid,
                "media": {
                    "payload": b64_payload
                }
            })
            
            await asyncio.sleep(0.18)

        print("[PLAYBACK] [OK] Streaming complete.")
        
        # 5. Push to Base44 Database
        asyncio.create_task(push_to_base44(stream_sid, transcript, answer, lang_code, intent))

        # After answering, play a suffix in the farmer's native language
        from router import translate_to_indian_language
        suffix_text = "You can ask me another question, or press any number on your phone to end the call."
        if farmer_lang_ref and farmer_lang_ref[0] != "en-IN":
            suffix_text = translate_to_indian_language(suffix_text, farmer_lang_ref[0])
        suffix_lang = detect_language_code(suffix_text)
        suffix_wav = generate_tts_audio(suffix_text, language_code=suffix_lang)
        if suffix_wav:
            suffix_pcm = convert_to_exotel_pcm(suffix_wav)
            for i in range(0, len(suffix_pcm), chunk_size):
                chunk = suffix_pcm[i:i + chunk_size]
                b64_payload = base64.b64encode(chunk).decode("utf-8")
                await websocket.send_json({
                    "event": "media",
                    "stream_sid": stream_sid,
                    "media": {"payload": b64_payload}
                })
                await asyncio.sleep(0.18)

    except asyncio.CancelledError:
        print("[PLAYBACK] [STOP] Playback task was cancelled (Barge-in occurred).")
    except Exception as e:
        print(f"[PLAYBACK ERROR] {e}")
        traceback.print_exc()

async def push_to_base44(phone: str, query: str, answer: str, lang: str, intent: str = "General"):
    """Pushes the completed call to the Base44 database webhook."""
    # We load these from the .env file so the API key stays secret
    base44_url = os.getenv("BASE44_API_URL")
    base44_key = os.getenv("BASE44_API_KEY")
    
    if not base44_url or not base44_key:
        print("[BASE44] Skipping DB insert — BASE44_API_URL or KEY not found in .env")
        return
        
    try:
        # Fast metadata extraction for analytics dashboard
        from router import call_llm
        import json
        
        prompt = (
            f"Analyze this farmer query: '{query}'\n"
            "Extract any mentioned crop, location, disease/pest, or government scheme.\n"
            "Respond ONLY with a valid JSON object using exactly these keys: 'crop', 'location', 'disease', 'scheme'.\n"
            "If a piece of information is not mentioned, use an empty string as the value. Do not add markdown or comments."
        )
        
        try:
            meta_str = await asyncio.to_thread(
                call_llm, 
                user_prompt=prompt, 
                system_prompt="You are a JSON data extractor. Output raw JSON only.", 
                max_tokens=100
            )
            # Clean up markdown if LLM adds it
            meta_str = meta_str.strip().strip("```json").strip("```").strip()
            meta = json.loads(meta_str)
        except Exception as e:
            print(f"[BASE44] Extractor failed/parsing error: {e}")
            meta = {"crop": "", "location": "", "disease": "", "scheme": ""}

        # Map router intents to the exact category strings expected by Base44 schema
        cat_map = {
            "DISEASE": "Crop Disease",
            "WEATHER": "Weather",
            "PRICE": "Market Price",
            "SCHEME": "Government Scheme",
            "GENERAL": "General"
        }
        mapped_category = cat_map.get(intent.upper(), "General")

        payload = {
            "phone_number": "+91 ****" + str(phone)[-4:] if phone else "+91 ****0000",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": 45,  # Estimated for now
            "language_detected": lang,
            "category": mapped_category,
            "farmer_query": query,
            "ai_response": answer,
            "status": "answered",
            "crop_mentioned": meta.get("crop", ""),
            "location_mentioned": meta.get("location", ""),
            "unanswered_reason": ""
        }
        
        import requests
        resp = await asyncio.to_thread(
            requests.post, 
            base44_url, 
            json=payload, 
            headers={"x-api-key": base44_key, "Content-Type": "application/json"}
        )
        if resp.ok:
            print(f"[BASE44] ✅ Call logged to Base44! Status: {resp.status_code}")
        else:
            print(f"[BASE44] ❌ Base44 push failed! Status: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"[BASE44] ❌ Error pushing to DB: {e}")


# ============================================================
# REST API ENDPOINTS (for Edesy / direct testing)
# ============================================================

from router import get_mandi_price, get_weather

@app.post("/api/crop-price")
async def api_crop_price(request):
    from starlette.requests import Request
    body = await request.json()
    crop = body.get("crop", "").strip()
    print(f"\n[API] Crop Price request: {crop}")
    if not crop:
        return {"result": "Please specify a crop name."}
    try:
        return {"result": get_mandi_price(crop)}
    except Exception as e:
        return {"result": f"Sorry, could not fetch price for {crop}."}

@app.post("/api/weather")
async def api_weather(request):
    from starlette.requests import Request
    body = await request.json()
    location = body.get("location", "").strip()
    print(f"\n[API] Weather request: {location}")
    if not location:
        return {"result": "Please specify a city."}
    try:
        return {"result": get_weather(location)}
    except Exception as e:
        return {"result": f"Sorry, could not fetch weather for {location}."}

@app.post("/api/disease-advice")
async def api_disease_advice(request):
    from starlette.requests import Request
    body = await request.json()
    query = body.get("query", "").strip()
    print(f"\n[API] Disease advice request: {query}")
    if not query:
        return {"result": "Please describe the problem."}
    try:
        return {"result": process_farmer_query(query)[0]}
    except Exception as e:
        return {"result": "Sorry, could not find advice."}

@app.post("/api/scheme-info")
async def api_scheme_info(request):
    from starlette.requests import Request
    body = await request.json()
    query = body.get("query", "").strip()
    print(f"\n[API] Scheme info request: {query}")
    if not query:
        return {"result": "Please specify a scheme."}
    try:
        return {"result": process_farmer_query(query)[0]}
    except Exception as e:
        return {"result": "Sorry, could not find that info."}

@app.post("/api/query")
async def api_general_query(request):
    from starlette.requests import Request
    body = await request.json()
    query = body.get("query", "").strip()
    print(f"\n[API] General query: {query}")
    if not query:
        return {"result": "Please ask a question."}
    try:
        return {"result": process_farmer_query(query)[0]}
    except Exception as e:
        return {"result": "Sorry, could not process your question."}


# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/")
async def health():
    return {
        "status": "AgriVoice Super-Agent is running!",
        "version": "3.0",
        "endpoints": {
            "ws://.../exotel-stream": "Exotel WebSocket Streaming",
            "/api/crop-price": "Get mandi/MSP price (POST)",
            "/api/weather": "Get weather (POST)",
            "/api/disease-advice": "Get disease advice (POST)",
            "/api/scheme-info": "Get scheme info (POST)",
            "/api/query": "General query (POST)",
            "/api/call-logs": "Get call logs (GET)",
            "/api/stats": "Get dashboard stats (GET)",
            "/api/log-call": "Log a completed call (POST)",
        },
        "powered_by": {
            "llm": "Sarvam sarvam-m4",
            "weather": "OpenWeatherMap",
            "price": "CEDA + data.gov.in + MSP Database",
            "rag": "ChromaDB + MiniLM-L6-v2",
        },
    }


# ============================================================
# CALL LOG STORAGE (in-memory for hackathon, swap to MongoDB later)
# ============================================================
from datetime import datetime, timezone
from collections import Counter

call_logs = []  # In-memory store — replace with MongoDB for production


@app.post("/api/log-call")
async def log_call(request):
    """Log a completed call. Called by the Exotel stream handler or manually."""
    body = await request.json()
    entry = {
        "id": len(call_logs) + 1,
        "timestamp": body.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "phone_number": body.get("phone_number", "+91 ****0000"),
        "duration_seconds": body.get("duration_seconds", 0),
        "language": body.get("language", "English"),
        "category": body.get("category", "General"),
        "farmer_query": body.get("farmer_query", ""),
        "ai_response": body.get("ai_response", ""),
        "status": body.get("status", "answered"),
        "crop_mentioned": body.get("crop_mentioned", ""),
        "location_mentioned": body.get("location_mentioned", ""),
    }
    call_logs.append(entry)
    print(f"[LOG] Call #{entry['id']} logged: {entry['category']} - {entry['farmer_query'][:50]}")
    return {"success": True, "id": entry["id"]}


@app.get("/api/call-logs")
async def get_call_logs():
    """Return all call logs (newest first) for the dashboard."""
    return {"logs": list(reversed(call_logs)), "total": len(call_logs)}


@app.get("/api/stats")
async def get_stats():
    """Return dashboard stats computed from call logs."""
    total = len(call_logs)
    if total == 0:
        return {
            "total_calls": 0,
            "answered": 0,
            "unanswered": 0,
            "avg_duration": 0,
            "category_distribution": {},
            "language_distribution": {},
            "top_crops": [],
            "top_locations": [],
        }

    answered = sum(1 for c in call_logs if c["status"] == "answered")
    unanswered = total - answered
    avg_dur = sum(c["duration_seconds"] for c in call_logs) / total

    cats = Counter(c["category"] for c in call_logs)
    langs = Counter(c["language"] for c in call_logs)
    crops = Counter(c["crop_mentioned"] for c in call_logs if c["crop_mentioned"])
    locs = Counter(c["location_mentioned"] for c in call_logs if c["location_mentioned"])

    return {
        "total_calls": total,
        "answered": answered,
        "unanswered": unanswered,
        "avg_duration": round(avg_dur, 1),
        "category_distribution": dict(cats),
        "language_distribution": dict(langs),
        "top_crops": crops.most_common(10),
        "top_locations": locs.most_common(10),
    }
