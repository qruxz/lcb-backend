# AI Profile Assistant Backend with RAG

This is the backend API service for the AI Profile Assistant, integrated with a true RAG (Retrieval-Augmented Generation) system.

## Features

- Flask-based RESTful API
- OpenAI GPT-4 API integration
- **True RAG System**: Uses ChromaDB vector database
- Document chunking and vectorization
- Semantic similarity search
- CORS support for cross-origin requests
- Loads personal data from JSON file
- Health check endpoint
- Local deployment, no external hosting required

## RAG System Architecture

```
User Query → Vectorization → Similarity Search → Context Enhancement → AI Response Generation
    ↓
Personal Data → Document Chunking → Vector Storage → ChromaDB
```

### RAG Workflow:

1. **Document Processing**: Convert `personal_data.json` to structured documents
2. **Text Chunking**: Use recursive character splitter to divide documents into chunks
3. **Vectorization**: Use OpenAI embeddings to convert text to vectors
4. **Storage**: Store vectors in ChromaDB
5. **Retrieval**: Search for most relevant document chunks based on user query
6. **Generation**: Pass retrieved context as prompt to AI for response generation

## Local Development

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Environment Variables

Copy the environment variable template:

```bash
cp env.example .env
```

Edit the `.env` file and set your OpenAI API Key:

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. Run Server

```bash
python app.py
```

The server will start at `http://localhost:5001` and automatically build the vector database.

## API Endpoints

### POST /api/chat

Send a message to the AI assistant (uses RAG retrieval).

**Request Body:**
```json
{
  "message": "What is your background?"
}
```

**Response:**
```json
{
  "response": "I am Your Name, a Software Engineer...",
  "success": true,
  "context_used": "Relevant Information 1:\nName: Your Name\nTitle: Software Engineer..."
}
```

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "AI Assistant API is running",
  "api_key": "configured",
  "rag_system": "initialized"
}
```

### POST /api/rebuild-vectorstore

Rebuild the vector database (use after updating personal data).

**Response:**
```json
{
  "message": "Vector database rebuilt successfully",
  "success": true
}
```

## Personal Data Configuration

Edit the `personal_data.json` file to customize your personal information. The file structure includes:

- `basic`: Basic information (name, title, email, etc.)
- `skills`: Skills list
- `experience`: Work experience
- `projects`: Project experience
- `education`: Education background
- `certifications`: Certifications
- `interests`: Interests and hobbies
- `careerGoals`: Career objectives

**After updating personal data, call the `/api/rebuild-vectorstore` endpoint to rebuild the vector database.**

## RAG System Advantages

### Compared to Traditional Methods:

1. **Precise Retrieval**: Only retrieves information relevant to the question
2. **Reduced Hallucination**: Based on retrieved real information
3. **Scalability**: Supports large documents and complex queries
4. **Efficiency**: Avoids sending all information to AI
5. **Cost Optimization**: Reduces token usage

### Technical Features:

- **Document Chunking**: Intelligent splitting while maintaining semantic integrity
- **Vector Search**: Based on semantic similarity, not keyword matching
- **Context Enhancement**: Dynamically builds most relevant context
- **Metadata Management**: Adds type labels to each document chunk

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key (required)

## File Structure

```
backend/
├── app.py                 # Main Flask application
├── rag_system.py          # RAG system core module
├── personal_data.json     # Personal data
├── requirements.txt       # Python dependencies
├── env.example           # Environment variable template
├── .env                  # Environment variables (local)
├── chroma_db/            # Vector database storage (auto-generated)
└── README.md            # Documentation
```

## Notes

1. Ensure `OPENAI_API_KEY` is correctly set in the `.env` file
2. Personal data is stored in `personal_data.json` and can be edited anytime
3. Rebuild vector database after updating personal data
4. API supports CORS and can be called from any frontend application
5. Server runs on `http://localhost:5001` by default
6. Vector database is stored in `chroma_db/` directory (added to .gitignore) 