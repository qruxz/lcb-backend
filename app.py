# app.py (text-only RAG with scraped_data + brand_data.json)

from fastapi import FastAPI, Request, Header
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from rag_system import RAGSystem
import logging
from datetime import datetime
import uuid
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Assistant Backend with RAG (Gemini)")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chat_logs.log"), logging.StreamHandler()],
)
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)


def log_message(user_id, message, request: Request, is_user=True, response=None, error=None):
    """Save chat logs"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "user_id": user_id,
        "message_type": "user" if is_user else "ai",
        "message": message,
        "response": response,
        "error": error,
        "ip_address": request.client.host if request.client else "Unknown",
        "user_agent": request.headers.get("user-agent", "Unknown"),
    }
    log_file = logs_dir / f"chat_logs_{datetime.now().strftime('%Y-%m-%d')}.json"
    try:
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to write to log file: {e}")

    if is_user:
        logging.info(f"User {user_id} ({log_entry['ip_address']}): {message}")
    else:
        logging.info(f"AI Response to {user_id}: {response[:100]}...")


def get_user_id(session_id: str = None):
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id


# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# ------------------ Init RAG ------------------
rag_system = None
if api_key:
    rag_system = RAGSystem(api_key)
    print("📑 Building vectorstore (brand_data.json + scraped_data/*.txt if available)...")
    rag_system.build_vectorstore(use_scraped=True)

# ----------------------------------------------------------------------

@app.post("/api/chat")
async def chat(request: Request, session_id: str = Header(default=None)):
    """Main chat endpoint"""
    try:
        data = await request.json()
        message = data.get("message", "")

        if not message:
            return {"error": "Message is required", "success": False}

        user_id = get_user_id(session_id)
        log_message(user_id, message, request, is_user=True)

        if not api_key:
            error_msg = "Gemini API key not configured."
            log_message(user_id, message, request, is_user=False, error=error_msg)
            return {"error": error_msg, "success": False}

        if not rag_system:
            error_msg = "RAG system not initialized"
            log_message(user_id, message, request, is_user=False, error=error_msg)
            return {"error": error_msg, "success": False}

        personal_info = rag_system.get_personal_info()
        profile_summary = rag_system.get_summary_document()

        # Refine Query
        query_refiner_prompt = f"""
Refine the user question into a precise search query.

User's Original Question: "{message}"
Refined Search Query:
"""
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            query_refiner_response = model.generate_content(query_refiner_prompt)
            refined_query = query_refiner_response.text.strip()
            print(f"🧠 Refined Search Query: {refined_query}")
        except Exception as e:
            print(f"⚠️ Query refinement failed: {e}")
            refined_query = message

        # Search RAG
        try:
            relevant_context = rag_system.search_relevant_context(refined_query, k=4)
            print("Retrieved relevant context.")
        except Exception as e:
            print(f"⚠️ RAG search failed: {e}")
            relevant_context = "Unable to retrieve relevant information."

        # Final Answer
        final_answer_prompt = f"""
You are a precise FAQ assistant for the brand {personal_info['name']}.

<USER_QUESTION>
{message}
</USER_QUESTION>

<DETAILED_CONTEXT>
{relevant_context}
</DETAILED_CONTEXT>

INSTRUCTIONS:
- Always answer respectfully, like a helpful assistant.
- Keep responses short (2–5 sentences).
- Format the answer using bullet points with relevant emojis for clarity.
- If the context has a clearly written answer, return it verbatim in this style.
- If no relevant answer exists, say politely: "Sorry, I don’t have that information right now."
"""
        final_model = genai.GenerativeModel("gemini-1.5-flash")
        final_response = final_model.generate_content(final_answer_prompt)
        ai_response = final_response.text.strip()

        log_message(user_id, message, request, is_user=False, response=ai_response)

        return {
            "response": ai_response,
            "success": True,
            "refined_query": refined_query,
            "session_id": user_id,
        }

    except Exception as e:
        error_msg = f"Failed to get AI response: {str(e)}"
        user_id = get_user_id(session_id)
        log_message(user_id, message if "message" in locals() else "Unknown", request, is_user=False, error=error_msg)
        print(f"Error: {str(e)}")
        return {"error": error_msg, "success": False}


@app.get("/api/health")
async def health_check():
    api_key_status = "configured" if api_key else "not configured"
    rag_status = "initialized" if rag_system else "not initialized"
    return {"status": "healthy", "api_key": api_key_status, "rag_system": rag_status}


@app.post("/api/rebuild-vectorstore")
async def rebuild_vectorstore():
    """Manually rebuild vector database from brand_data.json + scraped_data/*.txt"""
    try:
        if not api_key:
            return {"error": "Gemini API key not configured", "success": False}
        global rag_system
        rag_system = RAGSystem(api_key)
        rag_system.build_vectorstore(use_scraped=True)
        return {"message": "Vector database rebuilt", "success": True}
    except Exception as e:
        print(f"Error rebuilding vectorstore: {str(e)}")
        return {"error": "Failed to rebuild vector database", "success": False}


if __name__ == "__main__":
    print("🚀 Starting AI Assistant Backend with RAG (Gemini)...")
    print(f"📡 API Key Status: {'✅ Configured' if api_key else '❌ Not configured'}")
    print(f"🧠 RAG System: {'✅ Ready' if rag_system else '❌ Not ready'}")
    print("🌐 Server running at: http://localhost:5001")
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
