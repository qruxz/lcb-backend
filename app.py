from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from rag_system import RAGSystem
import logging
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat_logs.log'),
        logging.StreamHandler()
    ]
)

# Logs directory
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)

def log_message(user_id, message, is_user=True, response=None, error=None):
    timestamp = datetime.now().isoformat()
    log_entry = {
        'timestamp': timestamp,
        'user_id': user_id,
        'message_type': 'user' if is_user else 'ai',
        'message': message,
        'response': response,
        'error': error,
        'ip_address': request.remote_addr,
        'user_agent': request.headers.get('User-Agent', 'Unknown')
    }
    log_file = logs_dir / f'chat_logs_{datetime.now().strftime("%Y-%m-%d")}.json'
    try:
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to write to log file: {e}")

    if is_user:
        logging.info(f"User {user_id} ({request.remote_addr}): {message}")
    else:
        logging.info(f"AI Response to {user_id}: {response[:100]}...")

def get_user_id():
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id

# Configure Gemini
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    genai.configure(api_key=api_key)

# Initialize RAG
rag_system = None
if api_key:
    rag_system = RAGSystem(api_key)
    rag_system.build_vectorstore()

print("🔍 Environment check:")
print(f"   GEMINI_API_KEY from env: {'✅ Found' if api_key else '❌ Not found'}")
print()

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        user_id = get_user_id()
        log_message(user_id, message, is_user=True)

        if not api_key:
            error_msg = 'Gemini API key not configured.'
            log_message(user_id, message, is_user=False, error=error_msg)
            return jsonify({'error': error_msg, 'success': False}), 500

        if not rag_system:
            error_msg = 'RAG system not initialized'
            log_message(user_id, message, is_user=False, error=error_msg)
            return jsonify({'error': error_msg, 'success': False}), 500

        personal_info = rag_system.get_personal_info()
        profile_summary = rag_system.get_summary_document()

        # Refine Query
        query_refiner_prompt = f"""
You are a research assistant. Refine the user question into a precise search query for the brand knowledge base.

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
            print("Retrieved relevant context: ", relevant_context)
        except Exception as e:
            print(f"⚠️ RAG search failed: {e}")
            relevant_context = "Unable to retrieve relevant information."

        # Final Answer (short + direct)
        final_answer_prompt = f"""
You are a precise FAQ assistant for the brand {personal_info['name']}.

<USER_QUESTION>
{message}
</USER_QUESTION>

<DETAILED_CONTEXT>
{relevant_context}
</DETAILED_CONTEXT>

INSTRUCTIONS:
- Answer in **2–5 sentences maximum**.
- If the context has a clearly written "Ans.", return it verbatim.
- Do NOT add much extra explanations.
- If no relevant answer exists, say: "Sorry, I don’t have that information right now."
"""

        final_model = genai.GenerativeModel("gemini-1.5-flash")
        final_response = final_model.generate_content(final_answer_prompt)
        ai_response = final_response.text.strip()

        log_message(user_id, message, is_user=False, response=ai_response)

        return jsonify({
            'response': ai_response,
            'success': True,
            'refined_query': refined_query,
            'session_id': user_id
        })

    except Exception as e:
        error_msg = f'Failed to get AI response: {str(e)}'
        user_id = get_user_id()
        log_message(user_id, message if 'message' in locals() else 'Unknown', is_user=False, error=error_msg)
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Failed to get AI response', 'success': False}), 500

# Health check
@app.route('/api/health', methods=['GET'])
def health_check():
    api_key_status = "configured" if api_key else "not configured"
    rag_status = "initialized" if rag_system else "not initialized"
    return jsonify({'status': 'healthy','api_key': api_key_status,'rag_system': rag_status})

# Rebuild vectorstore
@app.route('/api/rebuild-vectorstore', methods=['POST'])
def rebuild_vectorstore():
    try:
        if not api_key:
            return jsonify({'error': 'Gemini API key not configured','success': False}), 500
        global rag_system
        rag_system = RAGSystem(api_key)
        rag_system.build_vectorstore()
        return jsonify({'message': 'Vector database rebuilt','success': True})
    except Exception as e:
        print(f"Error rebuilding vectorstore: {str(e)}")
        return jsonify({'error': 'Failed to rebuild vector database','success': False}), 500

if __name__ == '__main__':
    print("🚀 Starting AI Assistant Backend with RAG (Gemini)...")
    print(f"📡 API Key Status: {'✅ Configured' if api_key else '❌ Not configured'}")
    print(f"🧠 RAG System: {'✅ Ready' if rag_system else '❌ Not ready'}")

    port = int(os.environ.get("PORT", 5000))  # Render sets $PORT
    print(f"🌐 Server running at: http://0.0.0.0:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)
