# =====================================================================
# 🔑 CONFIGURATION & CREDENTIALS
# =====================================================================

NEUGEN_API_KEY = "nugen-4b77084118fd74dd"

# Neugen Cloud endpoints
EMBEDDINGS_URL = "https://api.nugen.in/api/v3/inference/embeddings"
CHAT_URL = "https://api.nugen.in/api/v3/inference/chat/completions"

# Deployed domain-aligned model ID
TRAINED_MODEL_ID = "model_01m370f1zcf3wswn"

HEADERS = {
    "Authorization": f"Bearer {NEUGEN_API_KEY}",
    "Content-Type": "application/json"
}

# Vector Database Config
COLLECTION_NAME = "secure_delivery_knowledge_base"
VECTOR_DIMENSION = 768