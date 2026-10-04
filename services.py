# =====================================================================
# ⚙️ BUSINESS LOGIC & AI SERVICES
# =====================================================================

import io
import requests
from fastapi import HTTPException
from pypdf import PdfReader
from config import CHAT_URL, EMBEDDINGS_URL, HEADERS, TRAINED_MODEL_ID


def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    """Extracts text from any format: PDF, TXT, JSON, CSV, LOG, YAML, SARIF, etc."""
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    # 1. Parse PDF documents
    if ext == "pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            extracted_pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(extracted_pages).strip()
        except Exception as e:
            raise ValueError(f"Error reading PDF content: {str(e)}")

    # 2. Parse text-based files (JSON, CSV, SARIF, TXT, YAML, LOG)
    else:
        for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                return file_bytes.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Unable to decode text from '{filename}'.")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits long text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]


def get_embedding(text_to_embed: str) -> list[float]:
    """Sends text to Neugen Embeddings endpoint and returns 768-dim vector list."""
    payload = {
        "model": TRAINED_MODEL_ID,
        "input": text_to_embed
    }
    try:
        response = requests.post(EMBEDDINGS_URL, json=payload, headers=HEADERS, timeout=20)
        response.raise_for_status()
        data = response.json()

        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["embedding"]
        elif "embedding" in data:
            return data["embedding"]
        else:
            raise ValueError(f"Unrecognized response format: {data}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding API error: {str(e)}")


def query_trained_model(question: str, context_block: str) -> str:
    """Constructs prompt and queries the fine-tuned model directly."""
    prompt = f"""
You are an expert DevSecOps Security Assistant for automated software deployment pipelines.
Analyze the provided document context below alongside your security training to identify, 
understand, and prioritize risks (vulnerabilities, misconfigurations, secrets, third-party libraries).
Suggest exact remediation steps without unnecessarily slowing down delivery.

Document Context:
{context_block}

Developer Question:
{question}
"""

    chat_payload = {
        "model": TRAINED_MODEL_ID,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.2,
        "stream": False
    }

    try:
        response = requests.post(CHAT_URL, json=chat_payload, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Neugen Chat API error: {str(e)}")

def enhance_query(original_question: str) -> str:
    """
    Expands conversational user questions into domain-rich search queries
    containing DevSecOps terminology (CVE, severity, reachability, waivers, remediation).
    """
    prompt = f"""
Rewrite the following user question into an optimized semantic search query for DevSecOps reports,
vulnerability scans, secrets, and policy audits. Expand implicit terms (e.g., 'deploy' -> 'release gate blockers, critical severity, reachability, waivers').
Return ONLY the rewritten search query text without quotation marks or explanations.

User Question: {original_question}
Optimized Search Query:
"""
    chat_payload = {
        "model": TRAINED_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60,
        "temperature": 0.1,
        "stream": False
    }

    try:
        response = requests.post(CHAT_URL, json=chat_payload, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        enhanced = data["choices"][0]["message"]["content"].strip().strip('"')
        return enhanced if enhanced else original_question
    except Exception:
        # Fallback cleanly to the original question if the enhancer call fails
        return original_question