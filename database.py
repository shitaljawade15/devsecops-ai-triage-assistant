# =====================================================================
# 🗄️ QDRANT VECTOR DATABASE INITIALIZATION
# =====================================================================

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config import COLLECTION_NAME, VECTOR_DIMENSION

# Shared in-memory Qdrant instance
qdrant = QdrantClient(":memory:")

def initialize_database():
    """Creates the collection with 768-dimensional cosine distance."""
    print("🚀 Initializing Qdrant collection for Secure Software Delivery...")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE)
    )
    print("✅ Collection ready for document vectors.")