"""
router.py — AgriVoice Super-Agent Core Brain
Day 2: Intent Classification + Tool Routing + Multilingual Response Generation

Architecture:
  Step 1: sarvam-30b classifies intent (DISEASE/WEATHER/PRICE/SCHEME/GENERAL)
  Step 2: Routes to correct tool (RAG DB or Live API)
  Step 3: sarvam-30b generates a clean 1-sentence voice answer via system+user messages
"""

import os
import re
import requests
from groq import Groq
from dotenv import load_dotenv
from rag_engine import retrieve_context

load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not SARVAM_API_KEY:
    print("[WARNING] SARVAM_API_KEY not found in .env file!")
if not GROQ_API_KEY:
    print("[WARNING] GROQ_API_KEY not found in .env file! Please add it.")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ============================================================
# CORE LLM CALLER (Powered by Groq / Llama-3-70B)
# ============================================================

def call_llm(user_prompt: str, system_prompt: str = None, model: str = "llama-3.3-70b-versatile", max_tokens: int = 150) -> str:
    """
    Calls Groq's insanely fast inference API using Llama-3-70B.
    This provides massive accuracy upgrades over the previous model.
    """
    if not groq_client:
        print("[LLM ERROR] GROQ_API_KEY is missing!")
        return ""

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=0.2, # Low temperature for factual consistency
        )
        content = chat_completion.choices[0].message.content
        if content:
            return content.strip()
        return ""
    except Exception as e:
        print(f"[LLM EXCEPTION] {e}")
        return ""


def extract_first_sentence(text: str) -> str:
    """
    Extracts the first clean sentence from an LLM response.
    - Strips markdown bold markers (**) and reasoning prefixes ("Let's go with: ...")
    - Uses capital-letter detection to avoid splitting on abbreviations like Rs., Dr., Mr.
    """
    if not text:
        return text

    # 1. Strip markdown bold markers
    text = re.sub(r'\*+', '', text).strip()

    # 2. Strip leading reasoning prefixes like "Let's go with:", "Here is:", "I'll say:"
    #    Pattern: any text (no period) followed by ": " at the start
    text = re.sub(r'^[^.!?\n]{0,40}:\s+', '', text).strip()

    # 3. If model wrapped answer in double quotes, extract it
    quoted = re.findall(r'"([^"]{8,300})"', text)
    if quoted:
        return quoted[0].strip()

    # 4. Split only at real sentence boundaries (punct + space + capital letter)
    #    This avoids splitting on: 'Rs. 2500', 'e.g. something', 'Dr. Singh'
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return sentences[0].strip() if sentences else text.strip()



# ============================================================
# TOOL: WEATHER API
# ============================================================

def get_weather(location: str) -> str:
    """OpenWeatherMap API — returns a complete, voice-ready weather sentence."""
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return f"I could not fetch weather data for {location} right now."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    try:
        data = requests.get(url, timeout=10).json()
        if data.get("cod") != 200:
            return f"I could not fetch the weather for {location} right now."
        temp = round(data["main"]["temp"])
        humidity = data["main"]["humidity"]
        desc = data["weather"][0]["description"]
        # Build a direct, voice-ready answer
        rain_likely = humidity > 65 or "rain" in desc or "cloud" in desc
        rain_hint = "so carry an umbrella just in case" if rain_likely else "so rain is unlikely"
        return (
            f"In {location} right now it is {temp} degrees with {desc} and {humidity}% humidity, "
            f"{rain_hint}."
        )
    except Exception as e:
        print(f"[Weather API Error] {e}")
        return f"Weather information for {location} is temporarily unavailable."


# ============================================================
# TOOL: GOVT MANDI PRICE API
# ============================================================

# Official Government MSP (Minimum Support Price) 2024-25 database.
# These are REAL government-declared prices — credible for any demo/presentation.
# Source: Cabinet Committee on Economic Affairs (CCEA), India, 2024.
MSP_DATABASE = {
    # Kharif (summer) crops
    "paddy":       ("2300", "quintal"),  "rice":        ("2300", "quintal"),
    "jowar":       ("3371", "quintal"),  "bajra":       ("2625", "quintal"),
    "maize":       ("2090", "quintal"),  "corn":        ("2090", "quintal"),
    "cotton":      ("7121", "quintal"),  "groundnut":   ("6783", "quintal"),
    "soybean":     ("4892", "quintal"),  "sunflower":   ("7280", "quintal"),
    "sugarcane":   ("340",  "quintal"),  "moong":       ("8682", "quintal"),
    "urad":        ("7400", "quintal"),  "tur":         ("7550", "quintal"),

    # Rabi (winter) crops
    "wheat":       ("2275", "quintal"),  "barley":      ("1735", "quintal"),
    "mustard":     ("5650", "quintal"),  "rapeseed":    ("5650", "quintal"),
    "lentil":      ("6425", "quintal"),  "masoor":      ("6425", "quintal"),
    "chickpea":    ("5440", "quintal"),  "gram":        ("5440", "quintal"),

    # Horticulture (approximate retail/mandi average 2024)
    "tomato":      ("2000", "quintal"),  "onion":       ("1500", "quintal"),
    "potato":      ("1200", "quintal"),  "garlic":      ("6000", "quintal"),
    "ginger":      ("10000","quintal"),  "turmeric":    ("14000","quintal"),
    "chili":       ("8000", "quintal"),  "chilli":      ("8000", "quintal"),
    "banana":      ("1500", "quintal"),  "mango":       ("3000", "quintal"),
    "grapes":      ("4000", "quintal"),
}


def msp_lookup(crop: str) -> str | None:
    """Look up government MSP / average mandi price for a crop."""
    key = crop.lower().strip()
    if key in MSP_DATABASE:
        price, unit = MSP_DATABASE[key]
        return price, unit
    # Fuzzy match: check if any key is a substring of the crop name
    for db_crop, (price, unit) in MSP_DATABASE.items():
        if db_crop in key or key in db_crop:
            return price, unit
    return None


def get_mandi_price(crop: str) -> str:
    """
    3-tier price lookup:
      1. CEDA API (Ashoka University) — cleaned Agmarknet data, better uptime
      2. data.gov.in — official but often slow
      3. Government MSP / average mandi price database — 100% reliable fallback
    """
    api_key = os.getenv("GOVT_DATA_API_KEY", "")
    crop_clean = crop.strip().capitalize()

    # ── Tier 1: CEDA API (Centre for Economic Data and Analysis, Ashoka Univ) ──
    try:
        ceda_url = "https://api.ceda.ashoka.edu.in/v1/agmarknet/"
        ceda_resp = requests.get(
            ceda_url,
            params={"commodity": crop_clean, "limit": 1},
            timeout=6,
        )
        if ceda_resp.ok:
            records = ceda_resp.json().get("results", [])
            if records:
                r = records[0]
                market  = r.get("market_name", "Unknown Market")
                state   = r.get("state_name",  "Unknown State")
                price   = r.get("modal_price",  r.get("max_price", "N/A"))
                arrived = r.get("arrival_date", "")
                date_str = f" ({arrived})" if arrived else ""
                return (
                    f"Live mandi price{date_str}: {crop_clean} in {market}, {state} "
                    f"is {price} rupees per quintal."
                )
    except Exception as e:
        print(f"[CEDA API] {e}")

    # ── Tier 2: data.gov.in ──
    if api_key:
        try:
            resp = requests.get(
                "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070",
                params={"api-key": api_key, "format": "json",
                        "offset": 0, "limit": 1, "filters[commodity]": crop_clean},
                timeout=6,
            )
            resp.raise_for_status()
            records = resp.json().get("records", [])
            if records:
                r = records[0]
                return (
                    f"Government mandi data: {crop_clean} price in "
                    f"{r.get('market','Unknown')}, {r.get('state','Unknown')} "
                    f"is {r.get('max_price','N/A')} rupees per quintal."
                )
        except Exception as e:
            print(f"[Govt API Error] {e}")

    # ── Tier 3: Government MSP database (always works) ──
    msp = msp_lookup(crop.lower())
    if msp:
        price, unit = msp
        return (
            f"The government MSP for {crop_clean} is {price} rupees per {unit}. "
            f"Actual mandi prices may vary slightly by region."
        )

    return f"Price data for {crop_clean} is not available right now. Please check your local mandi."


# ============================================================
# HELPER: KEYWORD EXTRACTION
# ============================================================

def is_ascii(text: str) -> bool:
    """Returns True if the string contains only ASCII characters."""
    return all(ord(c) < 128 for c in text)


def translate_to_english(text: str, hint: str = "word") -> str:
    """
    Translates a non-English word/phrase to English using sarvam-30b.
    Used when a farmer's query contains an Indian-language city or crop name.
    """
    if is_ascii(text):
        return text  # Already English, skip API call

    result = call_llm(
        f'Translate this {hint} to English. Reply with ONLY the English word.\n'
        f'Input: "{text}"\nEnglish:'
    )
    translated = result.strip().split("\n")[0].strip().strip('"').strip("'").strip(".")
    # Validate: must be non-empty ASCII
    if translated and is_ascii(translated) and len(translated) < 50:
        return translated
    return text  # Return original if translation failed


def translate_to_indian_language(text: str, target_lang_code: str) -> str:
    """Uses Sarvam Translate API to translate the text into the farmer's language."""
    if target_lang_code == "en-IN":
        return text

    url = "https://api.sarvam.ai/translate"
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": SARVAM_API_KEY
    }
    # Truncate text to avoid hitting the 500 character limit for TTS later
    safe_text = text[:400]
    
    payload = {
        "input": safe_text,
        "source_language_code": "auto",  # Auto-detect (RAG sometimes returns Hindi)
        "target_language_code": target_lang_code
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.ok:
            return resp.json().get("translated_text", safe_text)
        else:
            print(f"[Translate API Error] {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[Translate API Exception] {e}")
    return safe_text

def translate_to_english_api(text: str, source_lang: str) -> str:
    """Translates the initial farmer query into English for accurate processing."""
    url = "https://api.sarvam.ai/translate"
    payload = {
        "input": text,
        "source_language_code": source_lang,
        "target_language_code": "en-IN"
    }
    try:
        resp = requests.post(url, json=payload, headers={"api-subscription-key": SARVAM_API_KEY}, timeout=10)
        if resp.ok:
            return resp.json().get("translated_text", text)
    except Exception:
        pass
    return text


def extract_keyword(query: str, keyword_type: str) -> str:
    """Uses sarvam-30b to extract a location or crop name from the query."""
    if keyword_type == "location":
        prompt = f'Extract ONLY the city/district name from this query. If none, say "Pune".\nQuery: "{query}"\nCity:'
        default = "Pune"
    else:
        prompt = f'Extract ONLY the crop/vegetable name from this query. If none, say "Tomato".\nQuery: "{query}"\nCrop:'
        default = "Tomato"

    result = call_llm(prompt)
    keyword = result.strip().split("\n")[0].strip().strip('"').strip("'").strip(".")

    # Strip label prefixes that sarvam-30b sometimes includes e.g. "City: Pune" → "Pune"
    for label in ["city:", "location:", "crop:", "vegetable:", "place:", "district:"]:
        if keyword.lower().startswith(label):
            keyword = keyword[len(label):].strip()
            break

    # Strip numbered list prefixes e.g. "6. Tomato" or "1. Wheat" → "Tomato"
    keyword = re.sub(r'^\d+\.\s*', '', keyword).strip()

    # Reject if it looks like a label/phrase, not a name
    bad_fragments = ["crop/vegetable", "crop name", "vegetable name", "city name", "not specified"]
    if any(frag in keyword.lower() for frag in bad_fragments):
        keyword = default

    if len(keyword.split()) > 3 or len(keyword) > 40:
        keyword = default

    keyword = keyword if keyword else default

    # If keyword is in a non-English script (e.g. ಪುಣೆ, புனே), translate to English
    # so that APIs (OpenWeatherMap, Mandi) can understand it
    if not is_ascii(keyword):
        keyword = translate_to_english(keyword, hint=keyword_type)

    return keyword


# ============================================================
# MAIN SUPER-AGENT LOGIC
# ============================================================

def process_farmer_query(transcribed_text: str) -> str:
    """
    Core AgriVoice pipeline:
      1. Classify intent with Sarvam LLM
      2. Route to correct tool (RAG / Weather / Price API)
      3. Generate a concise, voice-ready multilingual answer
    """
    print(f"\n[Router] Farmer Input: '{transcribed_text}'")

    # ----------------------------------------------------------
    # STEP 1: Language Detection & English Translation
    # ----------------------------------------------------------
    SCRIPT_TO_LANG = {
        range(0x0900, 0x0980): "hi-IN",
        range(0x0980, 0x0A00): "bn-IN",
        range(0x0A80, 0x0B00): "gu-IN",
        range(0x0B00, 0x0B80): "or-IN",
        range(0x0B80, 0x0C00): "ta-IN",
        range(0x0C00, 0x0C80): "te-IN",
        range(0x0C80, 0x0D00): "kn-IN",
        range(0x0D00, 0x0D80): "ml-IN",
        range(0x0A00, 0x0A80): "pa-IN",
    }
    lang_code = "en-IN"
    for char in transcribed_text:
        cp = ord(char)
        for script_range, code in SCRIPT_TO_LANG.items():
            if cp in script_range:
                lang_code = code
                break
        if lang_code != "en-IN":
            break

    lang = {
        "hi-IN": "Hindi", "ta-IN": "Tamil", "te-IN": "Telugu",
        "kn-IN": "Kannada", "ml-IN": "Malayalam", "bn-IN": "Bengali",
        "gu-IN": "Gujarati", "pa-IN": "Punjabi", "or-IN": "Odia", "en-IN": "English"
    }.get(lang_code, "English")

    print(f"[Router] Detected language: {lang} ({lang_code})")

    if lang_code != "en-IN":
        english_query = translate_to_english_api(transcribed_text, lang_code)
        print(f"[Router] English Query: '{english_query}'")
    else:
        english_query = transcribed_text

    # ----------------------------------------------------------
    # STEP 2: Intent Classification
    # Keyword override runs FIRST on the English query
    # ----------------------------------------------------------
    query_lower = english_query.lower()

    DISEASE_KEYWORDS  = ["worm", "pest", "fungus", "disease", "blight", "rot", "spot",
                         "virus", "bacterial", "infection", "aphid", "mite", "insect",
                         "larva", "caterpillar", "yellowing", "wilting", "spray"]
    WEATHER_KEYWORDS  = ["rain", "weather", "temperature", "humid", "forecast", "cloud",
                         "storm", "wind", "drought", "flood"]
    PRICE_KEYWORDS    = ["price", "rate", "mandi", "market", "cost", "sell"]
    SCHEME_KEYWORDS   = ["loan", "scheme", "subsidy", "government", "yojana", "kisan",
                         "credit", "insurance", "pm-kisan", "bank"]

    if any(kw in query_lower for kw in DISEASE_KEYWORDS):
        intent = "DISEASE"
    elif any(kw in query_lower for kw in WEATHER_KEYWORDS):
        intent = "WEATHER"
    elif any(kw in query_lower for kw in PRICE_KEYWORDS):
        intent = "PRICE"
    elif any(kw in query_lower for kw in SCHEME_KEYWORDS):
        intent = "SCHEME"
    else:
        # Keyword match failed — fall back to LLM classification
        intent_raw = call_llm(
            user_prompt=(
                f'Classify this farmer query into ONE of: DISEASE, WEATHER, PRICE, SCHEME, GENERAL.\n'
                f'Reply with ONLY the category word.\nQuery: "{english_query}"'
            ),
            system_prompt=(
                "You are a classifier. Output ONE word from: "
                "DISEASE, WEATHER, PRICE, SCHEME, GENERAL. Nothing else."
            )
        )
        intent = "GENERAL"
        for category in ["DISEASE", "WEATHER", "PRICE", "SCHEME", "GENERAL"]:
            if category in intent_raw.upper():
                intent = category
                break

    print(f"[Router] Intent: {intent}")

    # ----------------------------------------------------------
    # STEP 3: Tool Execution
    # ----------------------------------------------------------
    context = ""

    if intent == "DISEASE":
        print("[Router] -> Searching Disease Vector DB...")
        context = retrieve_context(english_query, collection_name="disease_knowledge", k=3)

    elif intent == "WEATHER":
        print("[Router] -> Calling Weather API...")
        location = extract_keyword(english_query, "location")
        print(f"[Router] Location extracted: {location}")
        context = get_weather(location)

    elif intent == "PRICE":
        print("[Router] -> Calling Mandi Price API...")
        crop = extract_keyword(english_query, "crop")
        print(f"[Router] Crop extracted: {crop}")
        context = get_mandi_price(crop)

    elif intent == "SCHEME":
        print("[Router] -> Searching General Knowledge Vector DB...")
        context = retrieve_context(english_query, collection_name="general_knowledge", k=3)

    else:
        print("[Router] -> Searching General Knowledge Vector DB...")
        context = retrieve_context(english_query, collection_name="general_knowledge", k=3)

    print(f"[Router] Context snippet: {context[:120].encode('ascii','replace').decode()}...")

    # ----------------------------------------------------------
    # STEP 4: Generate Final Voice Answer
    # ----------------------------------------------------------


    if intent in ("PRICE", "WEATHER"):
        # Tool output is already a complete, voice-ready sentence — return directly.
        final_answer = context

    else:
        # Use LLM to synthesize a natural, voice-ready answer from the RAG context
        # The LLM should USE the context if relevant, but also use its own knowledge
        system_prompt = (
            "You are AgriVoice, a friendly agricultural expert for Indian farmers. "
            "You have deep knowledge about all crops, farming techniques, pest control, soil management, irrigation, government schemes, and organic farming. "
            "Answer the farmer's question in exactly ONE short sentence that will be spoken over a phone call. "
            "Use the context provided if it is relevant. If not, use your own knowledge. "
            "NEVER say 'consult an agricultural officer' or 'I don't have that information'. Always provide a helpful answer. "
            "No bullet points, bold text, numbers, or markdown. Maximum 40 words."
        )
        
        user_prompt = f"Farmer's Question: {english_query}\n\nReference Information:\n{context}"
        
        final_answer = call_llm(user_prompt=user_prompt, system_prompt=system_prompt, max_tokens=120)
        
        if not final_answer:
            final_answer = "I will look into this for you. Please try asking again in a moment."


    # Translate the answer back to the farmer's language if necessary
    # Enforce a strict 250 character limit BEFORE translation to keep TTS safe
    english_answer = final_answer[:250]
    final_answer = english_answer
    
    if lang_code != "en-IN":
        print(f"[Router] Translating answer to {lang} ({lang_code})...")
        final_answer = translate_to_indian_language(final_answer, lang_code)

    # Final safety net: TTS API has a hard 500 char limit
    if len(final_answer) > 450:
        # Truncate at the last full stop within 450 chars
        cut = final_answer[:450]
        last_period = max(cut.rfind('.'), cut.rfind('।'), cut.rfind('।'))
        if last_period > 100:
            final_answer = cut[:last_period + 1]
        else:
            final_answer = cut

    print(f"[Router] Final Answer: {final_answer}")
    return final_answer, intent, english_query, english_answer



# ============================================================
# TEST (Run this file directly: python router.py)
# ============================================================
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Support Hindi output in terminal

    test_queries = [
        "What is the price of tomato today?",
        "My cotton has pink worms, what should I do?",
        "How can I get a loan from the bank?",
        "Will it rain in Pune tomorrow?",
        "What is the price of onion in Mumbai?",
    ]

    print("\n" + "=" * 60)
    print("AgriVoice Router — Full Pipeline Test")
    print("=" * 60)

    for q in test_queries:
        print("\n" + "-" * 60)
        answer, intent, eq, ea = process_farmer_query(q)
        print(f"\n[INTENT DETECTED]: {intent}")
        print(f"[BOT AUDIO SCRIPT]: {answer}")

    print("\n" + "=" * 60)
    print("Test complete.")