

import requests
import os
from dotenv import load_dotenv

load_dotenv()


CONSTRUCTION_KEYWORDS = [
    "cement", "brick", "wall", "beam", "column",
    "foundation", "floor plan", "dwg","pdf", "autocad",
    "room", "rooms", "dimension", "layout", "door",
    "window", "plumbing", "electrical", "structure",
    "staircase", "building", "construction", "material",
    "bedroom", "kitchen", "hall"
]

CONSTRUCTION_PATTERNS = [
    "kitne", "how many", "count", "list",
    "kaha", "where", "show", "find"
]


def keyword_match(query: str) -> bool:
    q = query.lower()

    if any(k in q for k in CONSTRUCTION_KEYWORDS):
        return True

    if any(p in q for p in CONSTRUCTION_PATTERNS) and any(
        k in q for k in ["room", "beam", "column", "wall"]
    ):
        return True

    return False


def llm_classify(query: str) -> str:
    prompt = f"""
You are an intent classifier.

Return ONLY one word:
construction OR general

Query: {query}
"""

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    data = response.json()
    result = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    return result.strip().lower()


def detect_intent(query: str) -> str:
    q = query.lower()

    if "room" in q or "rooms" in q:
        return "construction"

    if "beam" in q or "column" in q:
        return "construction"

    if keyword_match(query):
        return "construction"

    intent = llm_classify(query)

    if "construction" in intent:
        return "construction"

    return "general"