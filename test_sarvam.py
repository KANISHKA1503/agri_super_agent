"""
Deep-dive on sarvam-30b content — ASCII-safe print
"""
import os, requests, sys
from dotenv import load_dotenv

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
API_KEY = os.getenv("SARVAM_API_KEY")
headers = {"Content-Type": "application/json", "api-subscription-key": API_KEY}

prompts = [
    "Say hello in one sentence.",
    "You are an agricultural assistant. A farmer asked in Hindi: 'Meri cotton mein pink worms hain, kya karoon?' Answer in 1 Hindi sentence.",
    "Will it rain in Pune tomorrow? Answer in 1 sentence based on this context: Current weather in Pune is 31C, overcast clouds, humidity 50%.",
]

for i, prompt in enumerate(prompts, 1):
    print(f"\n[TEST {i}]")
    resp = requests.post(
        "https://api.sarvam.ai/v1/chat/completions",
        json={"model": "sarvam-30b", "messages": [{"role": "user", "content": prompt}]},
        headers=headers, timeout=30
    )
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    choices = data.get('choices', [])
    if choices:
        msg = choices[0].get('message', {})
        content = msg.get('content')
        reasoning = msg.get('reasoning_content', '')
        print(f"  content type: {type(content)}, value: {repr(content)}")
        if reasoning:
            print(f"  reasoning snippet (first 200): {reasoning[:200]}")
    else:
        print(f"  No choices in response")
