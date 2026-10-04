# =====================================================================
# 🚀 FASTAPI APP & ENDPOINTS
# =====================================================================

import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client.models import PointStruct

from config import COLLECTION_NAME, TRAINED_MODEL_ID
from database import initialize_database, qdrant
from services import (
    chunk_text,
    extract_text_from_file,
    get_embedding,
    query_trained_model,
    enhance_query   # 👈 Add this import
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield
    print("🛑 Shutting down server...")


app = FastAPI(
    title="Secure Software Delivery Assistant",
    description="DevSecOps AI triage bot supporting any file upload (PDF, JSON, CSV, TXT, SARIF) with a custom Neugen model.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        example="Which critical container vulnerabilities block our release?",
        description="The developer's question about security findings or delivery status"
    )
    top_k: int = Field(default=3, ge=1, le=5, description="Number of context chunks to retrieve")

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources_used: list[str]
    model_used: str

class FileUploadResponse(BaseModel):
    filename: str
    file_type: str
    chunks_indexed: int
    status: str


# --- Endpoints ---
@app.post("/upload-file", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Receives any document (PDF, JSON, CSV, TXT, SARIF), chunks it, and saves vectors to Qdrant."""
    try:
        file_bytes = await file.read()
        full_text = extract_text_from_file(file.filename, file_bytes)

        if not full_text:
            raise HTTPException(status_code=400, detail="Uploaded file contains no readable text.")

        # Cap at 30 chunks to prevent long processing loops
        chunks = chunk_text(full_text)[:30]

        points = []
        for chunk in chunks:
            vector = get_embedding(chunk)
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk, "source_file": file.filename}
            )
            points.append(point)

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        file_ext = file.filename.split(".")[-1].upper() if "." in file.filename else "TXT"

        return FileUploadResponse(
            filename=file.filename,
            file_type=file_ext,
            chunks_indexed=len(points),
            status="File successfully processed and vectorized into Qdrant."
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
def chat_with_assistant(payload: ChatRequest):
    """
    Enhances query, retrieves relevant document snippets from Qdrant,
    and queries your fine-tuned model.
    """
    # 1. Enhance the conversational query with DevSecOps keywords
    search_query = enhance_query(payload.question)

    # 2. Vectorize the enhanced search query
    query_vector = get_embedding(search_query)

    # 3. Search Qdrant for top matching chunks
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=payload.top_k
    )

    matched_texts = [p.payload["text"] for p in search_results.points]
    context_block = "\n---\n".join(matched_texts) if matched_texts else "No uploaded documents found."

    # 4. Generate answer using the original developer question + context
    generated_answer = query_trained_model(payload.question, context_block)

    return ChatResponse(
        question=payload.question,
        answer=generated_answer,
        sources_used=matched_texts,
        model_used=TRAINED_MODEL_ID
    )