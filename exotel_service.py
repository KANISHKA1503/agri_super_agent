"""
exotel_service.py — Exotel Integration Module

This module abstracts the interactions with the Exotel platform, such as downloading
recorded audio and temporarily storing state.
"""

import os
import requests
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_API_TOKEN")

# =======================================================================
# STATE MANAGEMENT
# =======================================================================
# To communicate the generated audio URL to the Exotel Dynamic Greeting applet,
# we need to hold state across two consecutive HTTP requests.
#
# Currently, this uses an in-memory dictionary.
# [IMPORTANT] For Production/Multi-Worker Deployments:
# In-memory dictionaries do not scale across multiple Uvicorn/Gunicorn workers.
# You should replace this with Redis (for distributed fast cache) or
# SQLite/PostgreSQL (to persist session states reliably).
#
# Structure: { "CallSid": "generated_filename.wav" }
audio_state: Dict[str, str] = {}

def set_audio_for_call(call_sid: str, filename: str) -> None:
    """Store the TTS filename associated with this Exotel Call SID."""
    audio_state[call_sid] = filename
    print(f"[STATE] Saved audio state for CallSid {call_sid} -> {filename}")

def get_audio_for_call(call_sid: str) -> Optional[str]:
    """Retrieve the TTS filename for this Exotel Call SID."""
    # Note: In a production Redis setup, you would typically GET the key and
    # perhaps DELETE it afterward to free up cache space.
    filename = audio_state.get(call_sid)
    if filename:
        print(f"[STATE] Retrieved audio state for CallSid {call_sid} -> {filename}")
    else:
        print(f"[STATE WARNING] No audio state found for CallSid {call_sid}")
    return filename

# =======================================================================
# EXOTEL RECORDING DOWNLOAD
# =======================================================================

def download_recording(recording_url: str) -> bytes:
    """
    Downloads the recorded audio from Exotel.
    Exotel recording URLs require Basic HTTP Authentication using
    the Exotel API Key and Token.
    """
    if not recording_url:
        raise ValueError("Recording URL is empty")
    
    # Exotel Basic Auth credentials
    auth = (EXOTEL_API_KEY, EXOTEL_API_TOKEN) if EXOTEL_API_KEY else None
    
    print(f"[EXOTEL] Downloading audio from {recording_url}")
    try:
        response = requests.get(recording_url, auth=auth, timeout=15)
        response.raise_for_status()
        audio_bytes = response.content
        print(f"[EXOTEL] Successfully downloaded {len(audio_bytes)} bytes of audio.")
        return audio_bytes
    except requests.exceptions.RequestException as e:
        print(f"[EXOTEL ERROR] Failed to download recording: {e}")
        raise
