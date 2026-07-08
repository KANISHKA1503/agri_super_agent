"""
local_voice_test.py — AgriVoice Local Microphone Demo

Speaks into your laptop mic → Sarvam ASR → Router → Sarvam TTS → Speaker playback.
This uses the EXACT SAME AI pipeline as the Exotel phone call, but locally.

Usage:
    python local_voice_test.py

Controls:
    - Press ENTER to start recording
    - Press ENTER again to stop recording
    - The bot will process and speak the answer
    - Type 'quit' to exit
"""

import os
import io
import sys
import base64
import numpy as np
import sounddevice as sd
import requests
from pydub import AudioSegment
from dotenv import load_dotenv

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8")

# ── Business Logic ──
from router import process_farmer_query

load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# ============================================================
# AUDIO SETTINGS
# ============================================================
SAMPLE_RATE = 16000   # 16kHz for better ASR quality
CHANNELS = 1          # Mono
DTYPE = "int16"       # 16-bit PCM


# ============================================================
# RECORDING
# ============================================================
def record_audio() -> bytes:
    """Records audio from microphone until user presses Enter."""
    print("\n  🎤 Recording... (press ENTER to stop)")
    
    frames = []
    is_recording = True
    
    def callback(indata, frame_count, time_info, status):
        if is_recording:
            frames.append(indata.copy())
    
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback,
        blocksize=1024
    )
    
    stream.start()
    input()  # Wait for Enter key
    is_recording = False
    stream.stop()
    stream.close()
    
    if not frames:
        return b""
    
    audio_data = np.concatenate(frames, axis=0)
    pcm_bytes = audio_data.tobytes()
    print(f"  ✅ Recorded {len(pcm_bytes)} bytes ({len(pcm_bytes) / (SAMPLE_RATE * 2):.1f} seconds)")
    return pcm_bytes


# ============================================================
# SARVAM ASR (Speech-to-Text via WebSocket)
# ============================================================
def transcribe_audio(pcm_bytes: bytes) -> str:
    """Sends recorded PCM audio to Sarvam ASR REST API and returns the transcript."""
    print("  🧠 Transcribing with Sarvam ASR...")
    
    import wave
    
    # Create WAV file in memory
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    wav_io.seek(0)
    
    # Call Sarvam REST API
    try:
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": ("audio.wav", wav_io, "audio/wav")},
            data={"model": "saaras:v3"},
            timeout=10
        )
        
        if resp.ok:
            data = resp.json()
            return data.get("transcript", "").strip()
        else:
            print(f"  ⚠️ ASR Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"  ⚠️ ASR Exception: {e}")
        
    return ""


# ============================================================
# SARVAM TTS (Text-to-Speech)
# ============================================================
def detect_language_code(text: str) -> str:
    """Detects the language of the text using Unicode script ranges."""
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
            if cp in script_range:
                return code
    return "en-IN"


def generate_and_play_tts(text: str):
    """Generates TTS audio and plays it through speakers."""
    print(f"  🔊 Generating voice response...")
    
    lang_code = detect_language_code(text)
    
    try:
        resp = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            json={
                "inputs": [text],
                "target_language_code": lang_code,
                "speaker": "anushka",
                "model": "bulbul:v2",
                "enable_preprocessing": True,
            },
            headers={
                "Content-Type": "application/json",
                "api-subscription-key": SARVAM_API_KEY,
            },
            timeout=30,
        )
        
        if not resp.ok:
            print(f"  ❌ TTS Error: {resp.status_code}")
            return
        
        audio_b64 = resp.json().get("audios", [""])[0]
        if not audio_b64:
            print("  ❌ No audio returned from TTS")
            return
        
        # Decode and play
        audio_bytes = base64.b64decode(audio_b64)
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # Convert to numpy array for sounddevice playback
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / (2**15)  # Normalize 16-bit to float
        
        print(f"  🔊 Playing response ({len(audio) / 1000:.1f}s)...")
        sd.play(samples, samplerate=audio.frame_rate)
        sd.wait()  # Wait until playback finishes
        print(f"  ✅ Playback complete.")
        
    except Exception as e:
        print(f"  ❌ TTS/Playback error: {e}")


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  🌾 AgriVoice — Local Microphone Test")
    print("=" * 60)
    print("  Speak into your mic → AI processes → Hear the answer")
    print("  Same pipeline as the Exotel phone call!")
    print()
    print("  Controls:")
    print("    • Press ENTER to START recording")
    print("    • Press ENTER again to STOP recording")
    print("    • Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        print("\n" + "-" * 60)
        cmd = input("  Press ENTER to speak (or type 'quit'): ").strip()
        if cmd.lower() == "quit":
            print("\n  👋 Goodbye!")
            break
        
        # 1. Record from microphone
        pcm_bytes = record_audio()
        if not pcm_bytes:
            print("  ⚠️ No audio recorded. Try again.")
            continue
        
        # 2. Transcribe with Sarvam ASR
        transcript = transcribe_audio(pcm_bytes)
        if not transcript:
            print("  ⚠️ Could not transcribe audio. Try speaking louder/clearer.")
            continue
        print(f"  📝 You said: \"{transcript}\"")
        
        # 3. Process through Router (same as phone call)
        print("  🧠 Processing query...")
        answer = process_farmer_query(transcript)
        print(f"  💬 Answer: \"{answer}\"")
        
        # 4. Generate TTS and play through speakers
        generate_and_play_tts(answer)


if __name__ == "__main__":
    main()
