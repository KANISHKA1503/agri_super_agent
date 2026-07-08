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
import json
import base64
import asyncio
import traceback
import websockets
from pydub import AudioSegment
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import requests

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
# EXOTEL AGENTSTREAM WEBSOCKET
# ============================================================

@app.websocket("/exotel-stream")
async def handle_exotel_stream(websocket: WebSocket):
    """
    Main WebSocket bridge between Exotel AgentStream and our bot logic.
    """
    await websocket.accept()
    print("\n" + "=" * 60)
    print("[EXOTEL STREAM] 🟢 Connection established")
    print("=" * 60)

    # Session State
    stream_sid = None
    call_sid = None
    sarvam_ws = None
    playback_task = None  # To manage barge-in interruptions

    try:
        # 1. Connect to Sarvam ASR WebSocket
        print("[SARVAM STREAM] Connecting to Saaras ASR...")
        sarvam_uri = "wss://api.sarvam.ai/speech-to-text/ws"
        sarvam_ws = await websockets.connect(
            sarvam_uri, 
            additional_headers={"api-subscription-key": SARVAM_API_KEY}
        )
        print("[SARVAM STREAM] 🟢 Connected to ASR")
    

        # Send initialization config to Sarvam ASR
        config_payload = {
            "type": "config",
            "data": {
                "model": "saaras:v3",
                "language_code": "hi-IN",
                "sampling_rate": 8000,
                "encoding": "pcm_s16le"
            }
        }
        print("[SARVAM DEBUG] Config:", json.dumps(config_payload))
        await sarvam_ws.send(json.dumps(config_payload))
        print("[SARVAM STREAM] Sent configuration payload.")

        # --- DIAGNOSTICS START ---
        try:
            print("[SARVAM DIAGNOSTICS] Waiting for server acknowledgment...")
            ack_msg = await asyncio.wait_for(sarvam_ws.recv(), timeout=3.0)
            print(f"[SARVAM DIAGNOSTICS] Received exact response: {ack_msg}")
            
            # Simple check for acknowledgment or error
            if "error" in ack_msg.lower():
                print("[SARVAM DIAGNOSTICS] ❌ Server returned an error payload.")
            else:
                print("[SARVAM DIAGNOSTICS] ✅ Server acknowledged configuration.")
                
        except asyncio.TimeoutError:
            print("[SARVAM DIAGNOSTICS] ⚠️ Timeout waiting for configuration acknowledgment (proceeding anyway).")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[SARVAM DIAGNOSTICS] 🔴 Connection closed immediately! Code: {e.code}, Reason: {e.reason}")
        except Exception as e:
            print(f"[SARVAM DIAGNOSTICS] ❌ Unexpected error during handshake: {e}")
        # --- DIAGNOSTICS END ---

        # 2. Start Sarvam Receiver Task
        # Listens for transcripts coming back from Sarvam
        async def sarvam_receiver():
            nonlocal playback_task
            try:
                async for message in sarvam_ws:
                    # Sarvam sends JSON responses containing transcripts
                    data = json.loads(message)
                    transcript = data.get("transcript", "")
                    is_final = data.get("is_final", False)

                    if transcript:
                        print(f"[SARVAM ASR] Incoming: {transcript}")
                    
                    if transcript and is_final and len(transcript.strip()) > 2:
                        print(f"[SARVAM ASR] ✅ Final Transcript: {transcript}")
                        
                        # BARGE-IN: If TTS is currently playing, interrupt it!
                        if playback_task and not playback_task.done():
                            print("[BARGE-IN] Interrupting current playback...")
                            playback_task.cancel()
                            
                            # Send Exotel clear event to flush their playback buffer
                            if stream_sid:
                                await websocket.send_json({
                                    "event": "clear",
                                    "stream_sid": stream_sid
                                })
                                print("[EXOTEL STREAM] Sent 'clear' event.")
                        
                        # Process the new query and play the answer
                        playback_task = asyncio.create_task(
                            process_and_play(transcript, websocket, stream_sid)
                        )
            except websockets.exceptions.ConnectionClosed:
                print("[SARVAM STREAM] 🔴 Connection closed")
            except Exception as e:
                print(f"[SARVAM ERROR] {e}")

        sarvam_task = asyncio.create_task(sarvam_receiver())

        # 3. Listen to Exotel AgentStream
        while True:
            data = await websocket.receive_json()
            print(f"[EXOTEL RAW EVENT] {data}")
            event = data.get("event")

            if event == "connected":
                print("[EXOTEL STREAM] Received 'connected' event.")

            elif event == "start":
                stream_sid = data.get("start", {}).get("stream_sid")
                call_sid = data.get("start", {}).get("call_sid")
                print(f"[EXOTEL STREAM] 'start' | Stream: {stream_sid} | Call: {call_sid}")

            elif event == "media":
                # Exotel sends Base64-encoded 8kHz PCM chunks
                payload = data.get("media", {}).get("payload")
                if payload and sarvam_ws:
                    pcm_bytes = base64.b64decode(payload)
                    # Forward the binary PCM audio directly to Sarvam ASR
                    await sarvam_ws.send(pcm_bytes)

            elif event == "dtmf":
                digit = data.get("dtmf", {}).get("digit")
                print(f"[EXOTEL STREAM] 🔢 DTMF Digit: {digit}")

            elif event == "stop":
                print("[EXOTEL STREAM] Received 'stop' event. Terminating stream.")
                break

    except WebSocketDisconnect:
        print("[EXOTEL STREAM] 🔴 WebSocket disconnected by Exotel.")
    except Exception as e:
        print(f"[STREAM ERROR] {e}")
        traceback.print_exc()
    finally:
        # Cleanup connections
        if sarvam_ws:
            try:
                await sarvam_ws.close()
                print("[SARVAM STREAM] Closed ASR connection.")
            except Exception:
                pass
        if playback_task and not playback_task.done():
            playback_task.cancel()
        print("[EXOTEL STREAM] Session ended.\n")


async def process_and_play(transcript: str, websocket: WebSocket, stream_sid: str):
    """
    Passes transcript to the router, generates TTS, transcodes to 8kHz PCM,
    and streams back to Exotel in chunks.
    """
    try:
        # 1. Routing
        print("[ROUTER] Analyzing farmer query...")
        answer = process_farmer_query(transcript)
        print(f"[ROUTER] ✅ Answer: {answer}")

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
            
            await websocket.send_json({
                "event": "media",
                "stream_sid": stream_sid,
                "media": {
                    "payload": b64_payload
                }
            })
            
            # Sleep slightly faster than real-time to maintain an active buffer in Exotel
            await asyncio.sleep(0.18)

        print("[PLAYBACK] ✅ Streaming complete.")

    except asyncio.CancelledError:
        print("[PLAYBACK] 🛑 Playback task was cancelled (Barge-in occurred).")
    except Exception as e:
        print(f"[PLAYBACK ERROR] {e}")
        traceback.print_exc()


# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/")
async def health():
    return {
        "status": "🟢 AgriVoice AgentStream is running!",
        "version": "2.0",
        "endpoints": {
            "ws://.../exotel-stream": "Bidirectional Exotel Streaming Endpoint",
        }
    }
