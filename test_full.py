"""
AgriVoice Comprehensive Test Suite
Tests:
  1. Weather API (multiple cities)
  2. Price API (multiple crops)
  3. RAG (Disease + Scheme queries)
  4. Multilingual queries: Tamil, Telugu, Kannada, Hindi, Marathi, English
  5. Sarvam TTS multilingual voice test
"""

import sys
import os
import requests
import base64

sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from router import process_farmer_query

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

results = []

def run_test(label, query, expect_keywords=None):
    print(f"\n  Query : {query}")
    try:
        answer = process_farmer_query(query)
        print(f"  Answer: {answer}")
        if expect_keywords:
            found = any(kw.lower() in answer.lower() for kw in expect_keywords)
            status = PASS if found else WARN
            note = "" if found else f"(expected one of: {expect_keywords})"
        else:
            status = PASS if answer and len(answer) > 5 else FAIL
            note = ""
        results.append((status, label, answer[:80]))
        print(f"  {status} {note}")
    except Exception as e:
        print(f"  {FAIL} EXCEPTION: {e}")
        results.append((FAIL, label, str(e)))


# ============================================================
print("\n" + "=" * 65)
print("SECTION 1: WEATHER API — Multiple Cities")
print("=" * 65)

run_test("Weather Pune",    "Will it rain in Pune tomorrow?",       ["Pune", "cloud", "humidity", "umbrella"])
run_test("Weather Mumbai",  "What is the weather in Mumbai today?", ["Mumbai", "degree", "humidity"])
run_test("Weather Delhi",   "Mausam kaisa hai Delhi mein?",         ["Delhi", "degree"])  # Hindi

# ============================================================
print("\n" + "=" * 65)
print("SECTION 2: PRICE API — Multiple Crops")
print("=" * 65)

run_test("Price Tomato",  "What is the price of tomato today?",     ["tomato", "rupee", "quintal"])
run_test("Price Onion",   "Onion ka bhav kya hai aaj?",             ["onion", "rupee"])  # Hindi mix
run_test("Price Wheat",   "Gehu ka kya rate chal raha hai?",        ["wheat", "gehu", "rupee"])  # Hindi
run_test("Price Cotton",  "What is the cotton price in Nagpur?",    ["cotton", "rupee"])
run_test("Price Potato",  "Aloo ka mandi mein kya daam hai?",       ["potato", "aloo", "rupee"])  # Hindi

# ============================================================
print("\n" + "=" * 65)
print("SECTION 3: RAG — Disease Queries")
print("=" * 65)

run_test("Disease Pink Worms",  "My cotton has pink worms, what should I do?",        ["wound", "spray", "pesticide", "chlor"])
run_test("Disease Leaf Spot",   "My tomato plants have leaf spots. How to treat?",    ["spray", "fungus", "fungicide", "treatment"])
run_test("Disease Yellow Leaves","My rice crop leaves are turning yellow, help me!",  ["spray", "nitrogen", "fertilizer", "yellow"])

# ============================================================
print("\n" + "=" * 65)
print("SECTION 4: RAG — Scheme/General Queries")
print("=" * 65)

run_test("Scheme Bank Loan",     "How can I get a loan from the bank?",               ["bank", "contact", "nearest"])
run_test("Scheme PM Kisan",      "Tell me about PM Kisan Samman Nidhi scheme",        ["pm", "kisan", "scheme", "government", "bank"])
run_test("General Fertilizer",   "How much fertilizer should I use for wheat crop?",  ["fertilizer", "wheat", "spray", "kg"])

# ============================================================
print("\n" + "=" * 65)
print("SECTION 5: MULTILINGUAL QUERIES")
print("=" * 65)

# Tamil
run_test("Tamil — Price",   "தக்காளியின் விலை என்ன?",              ["rupee", "tomato", "2500"])
run_test("Tamil — Weather", "புனேவில் நாளை மழை பெய்யுமா?",        ["Pune", "cloud", "degree"])

# Telugu
run_test("Telugu — Disease","నా పత్తి పంటకు గులాబీ పురుగులు వచ్చాయి, ఏమి చేయాలి?", ["wound", "spray"])
run_test("Telugu — Price",  "ఉల్లిపాయ ధర ఎంత?",                   ["onion", "rupee"])

# Kannada
run_test("Kannada — Loan",  "ಬ್ಯಾಂಕ್ ಸಾಲ ಹೇಗೆ ಪಡೆಯಬಹುದು?",        ["bank", "contact"])
run_test("Kannada — Weather","ಪುಣೆಯಲ್ಲಿ ನಾಳೆ ಮಳೆ ಬರುತ್ತದೆಯೇ?",   ["Pune", "degree"])

# Hindi (Devanagari)
run_test("Hindi — Price",   "आज टमाटर का मंडी में क्या भाव है?",    ["tomato", "rupee"])
run_test("Hindi — Disease", "मेरी फसल में पत्तों पर धब्बे हैं।",   ["spray", "wound", "fungus"])

# Marathi
run_test("Marathi — Price", "कांद्याचा मंडीत काय भाव आहे?",         ["onion", "rupee"])

# ============================================================
print("\n" + "=" * 65)
print("SECTION 6: SARVAM TTS — Voice Output Test")
print("=" * 65)

tts_tests = [
    ("hi-IN", "anushka", "Namaste! Aaj Pune mein 26 degree temperature hai aur baarish ho sakti hai."),
    ("ta-IN", "anushka", "இன்று புனேயில் 26 டிகிரி வெப்பநிலை உள்ளது."),
    ("te-IN", "anushka", "ఈరోజు పూణేలో 26 డిగ్రీల ఉష్ణోగ్రత ఉంది."),
    ("kn-IN", "anushka", "ಇಂದು ಪೂಣೆಯಲ್ಲಿ 26 ಡಿಗ್ರಿ ತಾಪಮಾನ ಇದೆ."),
]

for lang_code, speaker, text in tts_tests:
    print(f"\n  [{lang_code}] {text[:50]}...")
    try:
        resp = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            json={
                "inputs": [text],
                "target_language_code": lang_code,
                "speaker": speaker,
                "model": "bulbul:v2",
                "enable_preprocessing": True,
            },
            headers={"Content-Type": "application/json", "api-subscription-key": SARVAM_API_KEY},
            timeout=20,
        )
        if resp.ok:
            audio_b64 = resp.json().get("audios", [""])[0]
            print(f"  {PASS} TTS audio generated: {len(audio_b64)} chars")
            results.append((PASS, f"TTS {lang_code}", f"{len(audio_b64)} chars"))
        else:
            print(f"  {FAIL} {resp.text[:200]}")
            results.append((FAIL, f"TTS {lang_code}", resp.text[:80]))
    except Exception as e:
        print(f"  {FAIL} {e}")
        results.append((FAIL, f"TTS {lang_code}", str(e)))

# ============================================================
print("\n\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

passed = sum(1 for s, _, _ in results if s == PASS)
warned = sum(1 for s, _, _ in results if s == WARN)
failed = sum(1 for s, _, _ in results if s == FAIL)
total  = len(results)

for status, label, answer in results:
    print(f"  {status} {label:<30} → {answer[:50]}")

print(f"\n  TOTAL: {passed}/{total} passed, {warned} warnings, {failed} failed")
if failed == 0:
    print("\n  🎉 ALL CORE TESTS PASSED — Ready for Day 3 (Telephony)!")
else:
    print(f"\n  {failed} test(s) need attention.")
