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

app = FastAPI(title="AgriVoice Super-Agent (AgentStream)", version="2.0")

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

    # VAD State
    audio_buffer = []
    is_speaking = False
    silence_chunks = 0
    SILENCE_THRESHOLD_RMS = 150  # Adjust if it's too sensitive or not sensitive enough
    SILENCE_CHUNKS_LIMIT = 15    # Number of silent chunks (approx 1.5 - 2s) to end speech

    async def _process_buffer(pcm_buffer: bytes):
        nonlocal playback_task
        print("[VAD] 🛑 Silence detected, processing speech...")
        transcript = await asyncio.to_thread(transcribe_audio_rest, pcm_buffer)
        
        if transcript and len(transcript.strip()) > 2:
            print(f"[SARVAM ASR] [OK] Final Transcript: {transcript}")
            
            # Barge-in: Interrupt TTS playback if we process new speech
            if playback_task and not playback_task.done():
                print("[BARGE-IN] Interrupting current playback...")
                playback_task.cancel()
                if stream_sid:
                    await websocket.send_json({"event": "clear", "stream_sid": stream_sid})
                    
            # Process query and play answer
            playback_task = asyncio.create_task(
                process_and_play(transcript, websocket, stream_sid, seq_num_ref)
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
                        
                        if rms > SILENCE_THRESHOLD_RMS:
                            if not is_speaking:
                                print("[VAD] 🎤 Speech detected, starting buffer...")
                                # Barge-in: if bot is playing, interrupt immediately when farmer speaks
                                if playback_task and not playback_task.done():
                                    print("[BARGE-IN] Farmer spoke! Interrupting current playback...")
                                    playback_task.cancel()
                                    await websocket.send_json({"event": "clear", "stream_sid": stream_sid})
                                    
                            is_speaking = True
                            silence_chunks = 0
                            audio_buffer.append(pcm_bytes)
                        else:
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
                print(f"[EXOTEL STREAM] [DTMF] DTMF Digit: {digit}")

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


async def process_and_play(transcript: str, websocket: WebSocket, stream_sid: str, seq_ref: list):
    """
    Passes transcript to the router, generates TTS, transcodes to 8kHz PCM,
    and streams back to Exotel in chunks.
    """
    try:
        # 1. Routing
        print("[ROUTER] Analyzing farmer query...")
        answer = process_farmer_query(transcript)
        print(f"[ROUTER] [OK] Answer: {answer}")

        # 2. Text to Speech
        lang_code = detect_language_code(answer)
        wav_audio = generate_tts_audio(answer, language_code=lang_code)

        if not wav_audio:
            print("[ERROR] Failed to generate TTS audio.")
            return

        # 3. Transcode to 8kHz Raw PCM
        pcm_data = convert_to_exotel_pcm(wav_audio)

        # 4. Stream back to Exotel in chunks
        # Exotel recommends chunks representing 100-200ms.
        # At 8000Hz, 16-bit Mono: 1 second = 16000 bytes.
        # Let's chunk by 3200 bytes (200ms).
        chunk_size = 3200
        
        print(f"[PLAYBACK] Streaming {len(pcm_data)} bytes of PCM back to Exotel...")
        
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i:i + chunk_size]
            b64_payload = base64.b64encode(chunk).decode("utf-8")
            chunk_idx = int(i / chunk_size) + 1
            timestamp_ms = int((i / chunk_size) * 200)
            
            await websocket.send_json({
                "event": "media",
                "stream_sid": stream_sid,
                "media": {
                    "payload": b64_payload
                }
            })
            
            # Sleep slightly faster than real-time to maintain an active buffer in Exotel
            await asyncio.sleep(0.18)

        print("[PLAYBACK] [OK] Streaming complete.")

    except asyncio.CancelledError:
        print("[PLAYBACK] [STOP] Playback task was cancelled (Barge-in occurred).")
    except Exception as e:
        print(f"[PLAYBACK ERROR] {e}")
        traceback.print_exc()


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
        return {"result": process_farmer_query(query)}
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
        return {"result": process_farmer_query(query)}
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
        return {"result": process_farmer_query(query)}
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
        },
        "powered_by": {
            "llm": "Sarvam sarvam-m4",
            "weather": "OpenWeatherMap",
            "price": "CEDA + data.gov.in + MSP Database",
            "rag": "ChromaDB + MiniLM-L6-v2",
        },
    }

