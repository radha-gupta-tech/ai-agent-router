import os
import requests

from fastapi import FastAPI, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from intent_router import detect_intent

load_dotenv()

app = FastAPI()


# CORS

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
    file_ids: list[str] | None = None
    top_k: int = 8
    include_sources: bool = True
    conversation_id: str | None = None


# RAG CALL

def call_rag(
    query: str,
    project_id: str,
    token: str | None = None,
    file_ids=None,
    top_k: int = 8,
    include_sources: bool = True,
    conversation_id: str | None = None,
):

    try:

        payload = {
            "message": query,
            "project_id": project_id,
            "file_ids": file_ids or [],
            "top_k": top_k,
            "include_sources": include_sources,
            "conversation_id": conversation_id or "default_conv"
        }

        print("RAG PAYLOAD:", payload)

        headers = {
            "Content-Type": "application/json"
        }

        # Only attach token if available
        if token:
            headers["Authorization"] = f"Bearer {token}"

        res = requests.post(
            f"{os.getenv('RAG_API_URL')}/api/v1/chat",
            headers=headers,
            json=payload,
            timeout=None
        )

        print("RAG STATUS:", res.status_code)
        print("RAG RESPONSE:", res.text)

        res.raise_for_status()

        data = res.json()

        return {
            "success": True,
            "answer": data.get("answer", "No answer found"),
            "sources": data.get("sources", [])
        }

    except Exception as e:

        print("RAG ERROR:", str(e))

        return {
            "success": False,
            "error": str(e)
        }

# DEEPSEEK LLM

def call_llm(query: str):

    SYSTEM_PROMPT = """
You are Archietech AI.

Rules:
- Always reply in English
- Keep answers short and professional
- Never hallucinate project data
- If data not available say:
  "Not found in project data"
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
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# MAIN CHAT ROUTE

@app.post("/chat")
def chat(
    req: QueryRequest,
    authorization: str = Header(None)
):

    query = req.query
    project_id = req.project_id

    # EXTRACT TOKEN FROM INCOMING REQUEST

    token = None

    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]

    print("TOKEN:", token)

    intent = detect_intent(query)

    print("QUERY:", query)
    print("INTENT:", intent)

    # CONSTRUCTION / PROJECT QUERY

    if intent == "construction":

        if not project_id:
            return {
                "source": "SYSTEM",
                "answer": "Please select a project first."
            }

        rag_response = call_rag(
            query=query,
            project_id=project_id,
            token=token,
            file_ids=req.file_ids,
            top_k=req.top_k,
            include_sources=req.include_sources,
            conversation_id=req.conversation_id
        )

        # FALLBACK TO LLM

        if not rag_response["success"]:

            return {
                "source": "LLM_FALLBACK",
                "answer": call_llm(query),
                "error": rag_response["error"]
            }

        return {
            "source": "RAG",
            "answer": rag_response["answer"],
            "sources": rag_response["sources"]
        }

    # GENERAL QUERY

    return {
        "source": "DEEPSEEK",
        "answer": call_llm(query)
    }


# HEALTH CHECK

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Archietech AI Router"
    }