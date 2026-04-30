import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import requests

from intent_router import detect_intent

load_dotenv()

app = FastAPI()




app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# REQUEST MODEL

class QueryRequest(BaseModel):
    query: str
    project_id: str | None = None  



# RAG CALL 

def call_rag(query: str, project_id: str):
    try:
        payload = {
            "message": query,
            "project_id": project_id
        }

        print("RAG PAYLOAD:", payload)

        res = requests.post(
            os.getenv("RAG_API_URL"),
            json=payload,
            timeout=None
        )

        res.raise_for_status()
        data = res.json()

        print("RAG FULL RESPONSE:", data)        # ← full response dekho
        print("RAG ANSWER:", data.get("answer")) # ← answer field
        print("RAG KEYS:", data.keys())          # ← kaunse fields aa rahe hain

        return data.get("answer", "No answer from RAG")

    except Exception as e:
        print("RAG EXCEPTION:", str(e))
        return f"RAG Error: {str(e)}"


# DEEPSEEK LLM

def call_llm(query: str):
    SYSTEM_PROMPT = """

You are Archietech AI, a professional and intelligent assistant.

Your responsibilities:
- Help with general questions (weather, knowledge, casual queries)
- Help with construction and project-related queries
- Maintain a clean, short, and professional response style

GENERAL RULES:
- Always respond in English
- Keep answers short (1–3 sentences)
- Be clear and direct
- Do NOT over-explain

----------------------------------------
GREETING BEHAVIOR (STRICT):
----------------------------------------
If the user input is EXACTLY one of these:
hi, hello, hey, hy

→ Respond EXACTLY:
"Hi! I'm Archietech AI. How can I help you today?"

Do NOT trigger greeting for words like:
yes, ok, sure, thanks

----------------------------------------
GENERAL QUERIES:
----------------------------------------
- Answer normally (weather, knowledge, casual questions)
- Do NOT block general queries
- Be helpful and concise

Example:
User: what is weather today
Assistant: Provide a normal answer

----------------------------------------
PROJECT / CONSTRUCTION QUERIES:
----------------------------------------
If the query involves:
- rooms, layout,beams, walls, dimensions
- drawing, DWG, plan, building structure,pdf
- "kitne rooms", "wall thickness", etc.

→ Assume it is project-related

Rules:
- Answer ONLY based on project data (RAG context)
- If data is not available → say:
  "Not found in project data"

- NEVER guess or hallucinate project details

----------------------------------------
IRRELEVANT / INAPPROPRIATE QUERIES:
----------------------------------------
If query is:
- personal (dating, relationship, etc.)
- unrelated nonsense

→ Respond:
"I’m here to help with useful questions. Please ask something relevant."

----------------------------------------
IMPORTANT:
----------------------------------------
- NEVER make up project data
- NEVER assume room counts, dimensions, etc.
- If unsure → say "Not found in project data"
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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
        }
    )

    return response.json()["choices"][0]["message"]["content"]


# -----------------------------
# MAIN ROUTE
# -----------------------------
@app.post("/chat")
def chat(req: QueryRequest):
    query = req.query
    project_id = req.project_id   

    intent = detect_intent(query)

    #  RAG only if project_id present
    if intent == "construction" and project_id:
        rag_answer = call_rag(query, project_id)

        if "Error" in rag_answer or "Timeout" in rag_answer:
            return {
                "source": "LLM_FALLBACK",
                "answer": call_llm(query)
            }

        return {
            "source": "RAG",
            "answer": rag_answer
        }

    # If project-related but no project selected
    if intent == "construction" and not project_id:
        return {
            "source": "SYSTEM",
            "answer": "Please select a project first."
        }

    return {
        "source": "DEEPSEEK",
        "answer": call_llm(query)
    }